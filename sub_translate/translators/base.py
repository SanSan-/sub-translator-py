from __future__ import annotations

from typing import List, Protocol

from sub_translate.models import TranslationOptions


class TranslationError(RuntimeError):
    """Ошибка перевода."""


class Translator(Protocol):
    name: str

    def translate_batch(self, texts: List[str], options: TranslationOptions) -> List[str]:
        ...

    def unload(self) -> None:
        ...


__all__ = ["TranslationError", "Translator"]