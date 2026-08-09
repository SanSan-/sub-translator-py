"""Локальные переводчики проекта sub-translate."""

from sub_translate.translators.local.nllb import Nllb600MTranslator
from sub_translate.translators.local.seedx import SeedXTranslator
from sub_translate.translators.local.translategemma import (
    TranslateGemma12BTranslator,
    TranslateGemmaTranslator,
)

__all__ = [
    "Nllb600MTranslator",
    "SeedXTranslator",
    "TranslateGemma12BTranslator",
    "TranslateGemmaTranslator",
]
