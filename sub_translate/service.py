from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

from sub_translate.constants import DEFAULT_BATCH_SIZE, DEFAULT_THREAD_COUNT
from sub_translate.enums import FileFormat
from sub_translate.models import PrepareToTranslateItem, TranslatedItem, TranslationOptions
from sub_translate.translators.agent import AgentTranslator
from sub_translate.translators.base import TranslationError, Translator
from sub_translate.translators.fsm import FsmTranslator
from sub_translate.translators.google_web import GoogleWebTranslator
from sub_translate.translators.nllb import NllbTranslator
from sub_translate.utils.io_utils import read_text, split_lines, write_lines
from sub_translate.utils.line_utils import (
    analyse_lines,
    build_export_lines,
    build_prepare,
    build_translated_dialogs,
    parse_ass_dialogs,
    parse_srt_dialogs,
    parse_vtt_dialogs,
)

API_ALIASES = {
    "google": "google",
    "google-translate-api": "google",
    "agent": "agent",
    "openai": "agent",
    "nllb": "nllb",
    "fsm": "fsm",
    "fsmt": "fsm",
}


def resolve_api(api: str | None) -> str:
    if not api:
        return "google"
    key = api.strip().lower()
    return API_ALIASES.get(key, key)


def resolve_format(input_path: Path, format_arg: str | None) -> FileFormat:
    if format_arg:
        return FileFormat(format_arg.lower())
    ext = input_path.suffix.lower().lstrip(".")
    if ext:
        return FileFormat(ext)
    return FileFormat.ASS


def get_translator(api: str, timeout: int) -> Translator:
    if api == "google":
        return GoogleWebTranslator(timeout=timeout)
    if api == "agent":
        return AgentTranslator()
    if api == "nllb":
        return NllbTranslator()
    if api == "fsm":
        return FsmTranslator()
    raise TranslationError(f"Неизвестный переводчик: {api}")


def _translate_batches(
    prepare: List[PrepareToTranslateItem],
    translator: Translator,
    options: TranslationOptions,
    thread_count: int,
    batch_size: int,
    logger: logging.Logger,
) -> List[TranslatedItem]:
    total = len(prepare)
    if total == 0:
        return []
    translated: Dict[int, TranslatedItem] = {}
    batch_size = max(1, int(batch_size or DEFAULT_BATCH_SIZE))
    thread_count = max(1, int(thread_count or DEFAULT_THREAD_COUNT))
    if getattr(translator, "name", "") in {"agent", "nllb", "fsm"}:
        thread_count = 1

    processed = 0
    for start in range(0, total, batch_size * thread_count):
        chunk = prepare[start : start + batch_size * thread_count]
        batches = [chunk[i : i + batch_size] for i in range(0, len(chunk), batch_size)]
        if thread_count > 1 and len(batches) > 1:
            with ThreadPoolExecutor(max_workers=thread_count) as executor:
                future_map = {}
                for batch in batches:
                    texts = [item.to_translate for item in batch]
                    future_map[executor.submit(translator.translate_batch, texts, options)] = batch
                for future in as_completed(future_map):
                    batch = future_map[future]
                    translations = future.result()
                    if len(translations) != len(batch):
                        raise TranslationError("Ответ переводчика не совпадает с размером пачки.")
                    for item, text in zip(batch, translations):
                        translated[item.idx] = TranslatedItem(idx=item.idx, text=text, lines=item.lines)
                    processed += len(batch)
                    logger.info("Переведено %s/%s", processed, total)
        else:
            for batch in batches:
                translations = translator.translate_batch(
                    [item.to_translate for item in batch],
                    options,
                )
                if len(translations) != len(batch):
                    raise TranslationError("Ответ переводчика не совпадает с размером пачки.")
                for item, text in zip(batch, translations):
                    translated[item.idx] = TranslatedItem(idx=item.idx, text=text, lines=item.lines)
                processed += len(batch)
                logger.info("Переведено %s/%s", processed, total)
    return [translated[idx] for idx in sorted(translated.keys())]


def translate_subtitles(
    input_path: Path,
    output_path: Path,
    file_format: FileFormat,
    options: TranslationOptions,
    api: str,
    *,
    thread_count: int,
    batch_size: int,
    smart_split: bool,
    timeout: int,
    logger: logging.Logger,
) -> None:
    raw_text = read_text(input_path)
    origins = split_lines(raw_text)

    if file_format == FileFormat.ASS:
        dialogs = parse_ass_dialogs(origins)
    elif file_format == FileFormat.SRT:
        dialogs = parse_srt_dialogs(origins)
    elif file_format == FileFormat.VTT:
        dialogs = parse_vtt_dialogs(origins)
    else:
        raise TranslationError(f"Формат не поддерживается: {file_format}")

    prepare = build_prepare(dialogs, smart_split)
    analysis = analyse_lines(dialogs)

    translator = get_translator(api, timeout)
    translated_items = _translate_batches(
        prepare,
        translator,
        options,
        thread_count,
        batch_size,
        logger,
    )
    translated_dialogs = build_translated_dialogs(translated_items, analysis)
    export_lines = build_export_lines(origins, file_format.value, dialogs, translated_dialogs)
    write_lines(output_path, export_lines)


__all__ = ["translate_subtitles", "resolve_api", "resolve_format"]
