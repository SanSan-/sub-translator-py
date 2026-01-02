"""Шаблоны подсказок для переводчика agent."""

from __future__ import annotations

from textwrap import dedent
from typing import Dict

DEFAULT_PROMPT_VARIANT = "default"

PROMPT_VARIANTS: Dict[str, str] = {
    DEFAULT_PROMPT_VARIANT: dedent(
        """
        Переведи текст на русский. Сохрани количество строк и переносы строк.

        Верни только перевод, без тегов.

        Формат:
        <SOURCE>
        {source}
        </SOURCE>
        """
    ).strip(),
}


def get_prompt_template(variant: str) -> str:
    """Возвращает шаблон подсказки по имени варианта."""
    try:
        return PROMPT_VARIANTS[variant]
    except KeyError as exc:  # pragma: no cover - защита от неверных вариантов
        raise ValueError(f"Неизвестный вариант подсказки: {variant}") from exc


__all__ = [
    "DEFAULT_PROMPT_VARIANT",
    "PROMPT_VARIANTS",
    "get_prompt_template",
]
