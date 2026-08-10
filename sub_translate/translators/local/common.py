"""Общие проверки адаптеров локальных переводчиков в изолированных процессах."""

from __future__ import annotations

from typing import Any

from sub_translate.dictionaries.languages import get_code
from sub_translate.translators.base import TranslationError

_LANGUAGE_ALIASES = {
    "английский": "en",
    "русский": "ru",
    "eng": "en",
    "rus": "ru",
}


def normalize_language(value: str | None, *, default: str) -> str:
    """Нормализует поддерживаемый язык локального переводчика."""
    if not value or not value.strip() or value.strip().casefold() == "auto":
        return default
    normalized = value.strip().casefold()
    normalized = _LANGUAGE_ALIASES.get(normalized, normalized)
    return get_code(normalized) or normalized


def validate_worker_translations(
    response: dict[str, Any],
    texts: list[str],
    worker_name: str,
) -> list[str]:
    """Проверяет мощность и тип ответа изолированного переводчика."""
    raw_translations = response.get("translations")
    if not isinstance(raw_translations, list):
        raise TranslationError(f"Процесс {worker_name} не вернул список переводов.")
    if len(raw_translations) != len(texts):
        raise TranslationError(
            f"Процесс {worker_name} вернул другое число переводов: "
            f"ожидалось {len(texts)}, получено {len(raw_translations)}."
        )
    if not all(isinstance(item, str) for item in raw_translations):
        raise TranslationError(f"Процесс {worker_name} вернул значение, которое не является строкой.")
    translations = [str(item) for item in raw_translations]
    if any(source and not translation.strip() for source, translation in zip(texts, translations, strict=True)):
        raise TranslationError(f"Процесс {worker_name} вернул пустой перевод непустой строки.")
    return translations
