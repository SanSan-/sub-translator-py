"""Переводчики проекта sub-translate."""

from sub_translate.translators.registry import (
    TranslatorMetadata,
    create_translator,
    get_translator_metadata,
    list_translator_metadata,
    resolve_registered_model,
    resolve_translator_id,
)

__all__ = [
    "TranslatorMetadata",
    "create_translator",
    "get_translator_metadata",
    "list_translator_metadata",
    "resolve_registered_model",
    "resolve_translator_id",
]
