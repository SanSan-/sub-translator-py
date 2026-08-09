from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from sub_translate import __version__
from sub_translate.constants import DEFAULT_BATCH_SIZE, DEFAULT_THREAD_COUNT, LOGS_DIR
from sub_translate.enums import FileFormat
from sub_translate.service import (
    ProcessingSettings,
    ProcessingStatus,
    process_subtitle,
    resolve_api,
    resolve_format,
    resolve_io_paths,
)
from sub_translate.translators.agent import (
    SYSTEM_PROMPT,
    get_agent_model_name,
    set_system_prompt,
    set_user_prompt_template,
)
from sub_translate.translators.agent_prompts import DEFAULT_PROMPT_VARIANT, get_prompt_template
from sub_translate.translators.base import TranslationError
from sub_translate.translators.google_web import normalize_google_tld
from sub_translate.translators.registry import (
    get_translator_metadata,
    list_translator_metadata,
    resolve_translator_id,
)
from sub_translate.utils.env_utils import load_env
from sub_translate.utils.io_utils import configure_utf8_stdio, read_text
from sub_translate.utils.logging_utils import configure_rotating_logger
from sub_translate.utils.subtitle_cache import prompt_signature

EXIT_SUCCESS = 0
EXIT_PROCESSING_ERROR = 1
EXIT_USAGE_ERROR = 2


def _localize_argparse_error(message: str) -> str:
    replacements = (
        ("the following arguments are required:", "не указаны обязательные аргументы:"),
        ("unrecognized arguments:", "неизвестные аргументы:"),
        (": invalid choice:", ": недопустимое значение:"),
        (": expected one argument", ": ожидалось одно значение"),
    )
    localized = message
    for source, target in replacements:
        localized = localized.replace(source, target)
    if localized.startswith("argument "):
        localized = f"аргумент {localized.removeprefix('argument ')}"
    return localized


class RussianArgumentParser(argparse.ArgumentParser):
    """Выводит пользовательскую ошибку argparse по-русски с кодом 2."""

    def error(self, message: str) -> None:
        self.print_usage(sys.stderr)
        self.exit(EXIT_USAGE_ERROR, f"{self.prog}: ошибка: {_localize_argparse_error(message)}\n")


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("ожидается положительное целое число") from None
    if parsed <= 0:
        raise argparse.ArgumentTypeError("ожидается положительное целое число")
    return parsed


def _parse_translator_id(value: str) -> str:
    if not value.strip():
        raise argparse.ArgumentTypeError("переводчик должен быть указан")
    canonical_id = resolve_translator_id(value)
    try:
        metadata = get_translator_metadata(canonical_id)
    except TranslationError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None
    if metadata.deprecated:
        raise argparse.ArgumentTypeError(f"Переводчик {canonical_id} больше не поддерживается.")
    return canonical_id


def _parse_google_tld(value: str) -> str:
    try:
        return normalize_google_tld(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


def build_parser() -> argparse.ArgumentParser:
    parser = RussianArgumentParser(description="Консольный переводчик субтитров")
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("-i", "--input", required=True, help="Путь к файлу субтитров")
    parser.add_argument("-o", "--output", help="Путь к выходному файлу")
    parser.add_argument("-f", "--format", choices=["ass", "srt", "vtt"], help="Формат субтитров")
    parser.add_argument("--from", dest="source_lang", default="auto", help="Исходный язык")
    parser.add_argument("--to", dest="target_lang", default="ru", help="Язык перевода")
    translator_profiles = ", ".join(
        f"{item.id} ({item.display_name})" for item in list_translator_metadata() if not item.deprecated
    )
    parser.add_argument(
        "--api",
        required=True,
        type=_parse_translator_id,
        help=f"Переводчик: {translator_profiles}",
    )
    parser.add_argument("--batch-size", type=_positive_int, default=DEFAULT_BATCH_SIZE, help="Размер пачки")
    parser.add_argument(
        "--threads",
        type=_positive_int,
        default=DEFAULT_THREAD_COUNT,
        help="Число параллельных запросов (для непотокобезопасных движков фиксируется 1)",
    )
    parser.add_argument(
        "--allow-cpu-fallback",
        action="store_true",
        help="Разрешить переход на CPU при ошибках загрузки локальных моделей",
    )
    parser.add_argument("--model-path", help="Каталог выбранной локальной модели")
    parser.add_argument("--model-revision", help="Закреплённая ревизия локальной модели")
    parser.add_argument(
        "--worker-python-path",
        help="Интерпретатор Python изолированной среды локальной модели",
    )
    parser.add_argument(
        "--auto-download-model",
        action="store_true",
        help="Докачать неполную локальную модель после промаха кеша",
    )
    parser.add_argument("--smart-split", action="store_true", help="Умное объединение реплик")
    parser.add_argument("--force", action="store_true", help="Игнорировать кеш и заменить выходной файл")
    parser.add_argument(
        "--tld",
        default="com",
        type=_parse_google_tld,
        help="Разрешённый доменный суффикс Google Translate",
    )
    parser.add_argument(
        "--timeout",
        type=_positive_int,
        help="Время ожидания запроса в секундах; без параметра используется значение профиля",
    )
    parser.add_argument("--verbose", action="store_true", help="Подробный вывод")
    parser.add_argument("--agent-model", help="Модель OpenAI для переводчика agent")
    parser.add_argument(
        "--agent-system-prompt-file",
        help="Путь к файлу с системной подсказкой агента (UTF-8)",
    )
    parser.add_argument(
        "--agent-prompt-file",
        help="Путь к файлу с пользовательской подсказкой агента (UTF-8, с {source})",
    )
    return parser


def _load_prompt_file(path_value: str, label: str) -> str:
    path = Path(path_value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} не найден: {path}")
    return read_text(path)


def _configure_agent_prompts(args: argparse.Namespace) -> dict[str, str]:
    system_prompt = SYSTEM_PROMPT
    user_prompt = get_prompt_template(DEFAULT_PROMPT_VARIANT)
    if args.agent_system_prompt_file:
        system_prompt = _load_prompt_file(args.agent_system_prompt_file, "Системная подсказка агента")
        set_system_prompt(system_prompt)
    if args.agent_prompt_file:
        user_prompt = _load_prompt_file(args.agent_prompt_file, "Пользовательская подсказка агента")
        set_user_prompt_template(user_prompt, variant="file")
    return {
        "system": prompt_signature(system_prompt),
        "user": prompt_signature(user_prompt),
    }


def _build_processing_settings(args: argparse.Namespace, api: str) -> ProcessingSettings:
    timeout = args.timeout or get_translator_metadata(api).default_timeout_seconds
    return ProcessingSettings(
        source_lang=args.source_lang,
        target_lang=args.target_lang,
        api=api,
        batch_size=args.batch_size,
        thread_count=args.threads,
        smart_split=args.smart_split,
        timeout=timeout,
        force=args.force,
        tld=args.tld,
        allow_cpu_fallback=args.allow_cpu_fallback,
        model_path=Path(args.model_path) if args.model_path else None,
        model_revision=args.model_revision,
        worker_python_path=(Path(args.worker_python_path) if args.worker_python_path else None),
        auto_download_model=args.auto_download_model,
        agent_model=(args.agent_model or get_agent_model_name()) if api == "agent" else None,
        prompt_signatures=_configure_agent_prompts(args) if api == "agent" else {},
    )


def _prepare_request(
    args: argparse.Namespace,
) -> tuple[Path, ProcessingSettings, FileFormat, str, Path]:
    input_path = Path(args.input).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Файл не найден: {input_path}")

    api = resolve_api(args.api)
    settings = _build_processing_settings(args, api)
    file_format = resolve_format(input_path, args.format)
    source_lang, output_path = resolve_io_paths(
        input_path,
        args.output,
        settings.source_lang,
        settings.target_lang,
        settings.api,
        file_format,
    )
    if output_path == input_path:
        raise ValueError("Исходный и выходной пути не должны совпадать.")
    return input_path, settings, file_format, source_lang, output_path


def _report_processing_error(error: Exception, logger: logging.Logger | None = None) -> int:
    if logger is not None:
        logger.error("Обработка завершилась ошибкой (%s).", type(error).__name__)
    message = str(error).strip() or "Не удалось обработать субтитры."
    print(f"Ошибка: {message}", file=sys.stderr)
    return EXIT_PROCESSING_ERROR


def _run_translation(
    args: argparse.Namespace,
    request: tuple[Path, ProcessingSettings, FileFormat, str, Path],
) -> int:
    input_path, settings, file_format, source_lang, output_path = request
    try:
        logger = configure_rotating_logger(
            "sub_translate",
            LOGS_DIR / "sub_translate.log",
            verbose=args.verbose,
        )
    except Exception as exc:
        return _report_processing_error(exc)
    logger.info("Формат: %s", file_format.value)
    logger.info("Переводчик: %s", settings.api)
    logger.info("Направление: %s -> %s", source_lang, settings.target_lang)
    logger.info("Вход: %s", input_path)
    logger.info("Выход: %s", output_path)
    logger.info("Пачка: %s, потоки: %s", settings.batch_size, settings.thread_count)

    try:
        result = process_subtitle(
            input_path,
            settings,
            logger=logger,
            output_path=output_path,
            format_arg=args.format,
        )
    except (OSError, TranslationError, ValueError) as exc:
        return _report_processing_error(exc, logger)
    except Exception as exc:
        logger.error("Непредвиденная ошибка обработки (%s).", type(exc).__name__)
        print("Ошибка: непредвиденная ошибка обработки.", file=sys.stderr)
        return EXIT_PROCESSING_ERROR
    if result.status is ProcessingStatus.CACHED:
        logger.info("Готово: результат восстановлен из кеша.")
    elif result.status is ProcessingStatus.OUTPUT_EXISTS:
        logger.info("Выходной файл уже существует; для замены укажите --force.")
    else:
        logger.info("Готово")
    return EXIT_SUCCESS


def main() -> int:
    """Возвращает 0 при успехе, 1 при ошибке обработки и 2 при неверных аргументах."""
    configure_utf8_stdio()
    parser = build_parser()
    args = parser.parse_args()
    try:
        load_env()
    except Exception as exc:
        return _report_processing_error(exc)
    try:
        request = _prepare_request(args)
    except (OSError, TranslationError, ValueError) as exc:
        parser.error(str(exc))
    return _run_translation(args, request)


__all__ = [
    "EXIT_PROCESSING_ERROR",
    "EXIT_SUCCESS",
    "EXIT_USAGE_ERROR",
    "build_parser",
    "main",
    "resolve_io_paths",
]
