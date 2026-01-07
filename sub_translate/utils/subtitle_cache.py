from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from sub_translate.constants import SUBTITLE_CACHE_FILE
from sub_translate.dictionaries.languages import get_code
from sub_translate.enums import FileFormat
from sub_translate.utils.io_utils import read_text, split_lines, write_lines
from sub_translate.utils.path_utils import split_lang_suffix


def build_cache_key(input_path: Path, api: str, target_lang: str) -> str:
    base_stem, _ = split_lang_suffix(input_path.stem)
    resolved_target = get_code(target_lang) or target_lang
    return f"{base_stem}.{api}.{resolved_target}"


def _load_cache(logger: logging.Logger) -> dict[str, Any]:
    if not SUBTITLE_CACHE_FILE.exists():
        return {}
    try:
        data = json.loads(read_text(SUBTITLE_CACHE_FILE))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Не удалось прочитать кеш переводов: %s", exc)
        return {}
    if not isinstance(data, dict):
        logger.warning("Формат кеша переводов не распознан, начинаю с пустого состояния.")
        return {}
    return data


def _save_cache(cache: dict[str, Any], logger: logging.Logger) -> None:
    try:
        SUBTITLE_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        SUBTITLE_CACHE_FILE.write_text(
            json.dumps(cache, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError as exc:
        logger.warning("Не удалось сохранить кеш переводов: %s", exc)


def _extract_cached_lines(
    cache: dict[str, Any],
    cache_key: str,
    file_format: FileFormat,
) -> list[str] | None:
    entry = cache.get(cache_key)
    if isinstance(entry, dict):
        format_value = entry.get("format")
        if isinstance(format_value, str) and format_value and format_value != file_format.value:
            return None
        lines = entry.get("lines")
        if isinstance(lines, list) and all(isinstance(item, str) for item in lines):
            return lines
    if isinstance(entry, list) and all(isinstance(item, str) for item in entry):
        return entry
    return None


def load_cache_snapshot(logger: logging.Logger) -> dict[str, Any]:
    """Читает кеш переводов в память."""
    return _load_cache(logger)


def get_cached_lines(
    cache: dict[str, Any],
    cache_key: str,
    file_format: FileFormat,
) -> list[str] | None:
    """Возвращает строки перевода из кеша, если формат совпадает."""
    return _extract_cached_lines(cache, cache_key, file_format)


def _store_cached_lines(
    cache: dict[str, Any],
    cache_key: str,
    file_format: FileFormat,
    lines: list[str],
) -> None:
    cache[cache_key] = {
        "format": file_format.value,
        "lines": list(lines),
    }


def apply_output_cache(
    input_path: Path,
    output_path: Path,
    api: str,
    target_lang: str,
    file_format: FileFormat,
    logger: logging.Logger,
) -> bool:
    cache_key = build_cache_key(input_path, api, target_lang)
    cache = _load_cache(logger)
    cached_lines = _extract_cached_lines(cache, cache_key, file_format)
    if cached_lines is not None:
        if output_path.exists():
            logger.info("Кеш перевода найден (%s), файл уже существует, пропускаю.", cache_key)
            return True
        try:
            write_lines(output_path, cached_lines)
        except OSError as exc:
            logger.warning("Не удалось восстановить перевод из кеша (%s): %s", cache_key, exc)
            return False
        logger.info("Перевод восстановлен из кеша (%s).", cache_key)
        return True

    if output_path.exists():
        try:
            lines = split_lines(read_text(output_path))
        except OSError as exc:
            logger.warning("Не удалось прочитать файл результата для кеша (%s): %s", cache_key, exc)
            return True
        _store_cached_lines(cache, cache_key, file_format, lines)
        _save_cache(cache, logger)
        logger.info("Файл результата уже существует, сохранён в кеш (%s).", cache_key)
        return True
    return False


def update_output_cache(
    input_path: Path,
    output_path: Path,
    api: str,
    target_lang: str,
    file_format: FileFormat,
    logger: logging.Logger,
) -> None:
    if not output_path.exists():
        return
    cache_key = build_cache_key(input_path, api, target_lang)
    cache = _load_cache(logger)
    try:
        lines = split_lines(read_text(output_path))
    except OSError as exc:
        logger.warning("Не удалось прочитать файл результата для кеша (%s): %s", cache_key, exc)
        return
    cached_lines = _extract_cached_lines(cache, cache_key, file_format)
    if cached_lines == lines:
        return
    _store_cached_lines(cache, cache_key, file_format, lines)
    _save_cache(cache, logger)


__all__ = [
    "build_cache_key",
    "load_cache_snapshot",
    "get_cached_lines",
    "apply_output_cache",
    "update_output_cache",
]
