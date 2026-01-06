from __future__ import annotations

import argparse
from pathlib import Path

from sub_translate.constants import DEFAULT_BATCH_SIZE, DEFAULT_THREAD_COUNT, LOGS_DIR
from sub_translate.dictionaries.languages import get_code
from sub_translate.enums import FileFormat
from sub_translate.models import TranslationOptions
from sub_translate.service import resolve_api, resolve_format, translate_subtitles
from sub_translate.translators.agent import set_system_prompt, set_user_prompt_template
from sub_translate.utils.io_utils import read_text
from sub_translate.utils.logging_utils import configure_rotating_logger
from sub_translate.utils.path_utils import split_lang_suffix
from sub_translate.utils.subtitle_cache import apply_output_cache, update_output_cache


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Консольный переводчик субтитров")
    parser.add_argument("-i", "--input", required=True, help="Путь к файлу субтитров")
    parser.add_argument("-o", "--output", help="Путь к выходному файлу")
    parser.add_argument("-f", "--format", choices=["ass", "srt", "vtt"], help="Формат субтитров")
    parser.add_argument("--from", dest="source_lang", default="auto", help="Исходный язык")
    parser.add_argument("--to", dest="target_lang", default="ru", help="Язык перевода")
    parser.add_argument("--api", default="google", help="Переводчик: google|agent|nllb|nllb-lite|seamless|madlad|fsm")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Размер пачки")
    parser.add_argument(
        "--threads",
        type=int,
        default=DEFAULT_THREAD_COUNT,
        help="Число параллельных запросов (для nllb/nllb-lite/seamless/madlad/fsm фиксируется 1)",
    )
    parser.add_argument("--smart-split", action="store_true", help="Умное объединение реплик")
    parser.add_argument("--tld", default="com", help="Домен Google Translate")
    parser.add_argument("--timeout", type=int, default=30, help="Таймаут запроса (сек)")
    parser.add_argument("--verbose", action="store_true", help="Подробный вывод")
    parser.add_argument(
        "--agent-system-prompt-file",
        help="Путь к файлу с системным промптом агента (UTF-8)",
    )
    parser.add_argument(
        "--agent-prompt-file",
        help="Путь к файлу с пользовательским промптом агента (UTF-8, с {source})",
    )
    return parser


def _load_prompt_file(path_value: str, label: str) -> str:
    path = Path(path_value).expanduser().resolve()
    if not path.exists():
        raise SystemExit(f"{label} не найден: {path}")
    return read_text(path)


def resolve_io_paths(
    input_path: Path,
    output_arg: str | None,
    source_lang: str,
    target_lang: str,
    api: str,
    file_format: FileFormat,
) -> tuple[str, Path]:
    base_stem, suffix_lang = split_lang_suffix(input_path.stem)
    resolved_source = suffix_lang or source_lang
    resolved_target = get_code(target_lang) or target_lang
    if output_arg:
        return resolved_source, Path(output_arg).expanduser().resolve()
    output_stem = f"{base_stem}.{api}.{resolved_target}"
    output_path = input_path.with_name(f"{output_stem}.{file_format.value}")
    return resolved_source, output_path


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        raise SystemExit(f"Файл не найден: {input_path}")

    api = resolve_api(args.api)
    file_format = resolve_format(input_path, args.format)
    source_lang, output_path = resolve_io_paths(
        input_path,
        args.output,
        args.source_lang,
        args.target_lang,
        api,
        file_format,
    )
    options = TranslationOptions(
        source_lang=source_lang,
        target_lang=args.target_lang,
        api=api,
        tld=args.tld,
    )

    if args.agent_system_prompt_file:
        system_prompt = _load_prompt_file(args.agent_system_prompt_file, "Системный промпт агента")
        set_system_prompt(system_prompt)
    if args.agent_prompt_file:
        user_prompt = _load_prompt_file(args.agent_prompt_file, "Пользовательский промпт агента")
        set_user_prompt_template(user_prompt, variant="file")

    logger = configure_rotating_logger(
        "sub_translate",
        LOGS_DIR / "sub_translate.log",
        verbose=args.verbose,
    )
    logger.info("Формат: %s", file_format.value)
    logger.info("Переводчик: %s", api)
    logger.info("Направление: %s -> %s", source_lang, args.target_lang)
    logger.info("Вход: %s", input_path)
    logger.info("Выход: %s", output_path)
    logger.info("Пачка: %s, потоки: %s", args.batch_size, args.threads)

    if apply_output_cache(
        input_path,
        output_path,
        api,
        args.target_lang,
        file_format,
        logger,
    ):
        logger.info("Готово")
        return 0

    translate_subtitles(
        input_path,
        output_path,
        file_format,
        options,
        api,
        thread_count=args.threads,
        batch_size=args.batch_size,
        smart_split=args.smart_split,
        timeout=args.timeout,
        logger=logger,
    )
    update_output_cache(
        input_path,
        output_path,
        api,
        args.target_lang,
        file_format,
        logger,
    )
    logger.info("Готово")
    return 0


__all__ = ["main", "build_parser"]
