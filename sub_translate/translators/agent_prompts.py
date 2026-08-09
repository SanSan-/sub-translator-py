"""Шаблоны подсказок для переводчика agent."""

from __future__ import annotations

from textwrap import dedent

DEFAULT_PROMPT_VARIANT = "default"
BATCH_PROMPT_VARIANT = "batch"

PROMPT_VARIANTS: dict[str, str] = {
    DEFAULT_PROMPT_VARIANT: dedent(
        """
        Переведи текст с языка {source_lang} на язык {target_lang}.
        Сохрани количество строк и переносы строк.

        Верни только перевод, без тегов.

        Формат:
        <SOURCE>
        {source}
        </SOURCE>
        """
    ).strip(),
    BATCH_PROMPT_VARIANT: dedent(
        """
        Переведи текст с языка {source_lang} на язык {target_lang}.
        Сохрани количество строк и переносы строк.

        Текст разбит на блоки. Между блоками стоит строка-разделитель:
        {separator}
        Разделитель не переводить, не удалять и не менять. Сохрани его в исходном виде.

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
    "BATCH_PROMPT_VARIANT",
    "DEFAULT_PROMPT_VARIANT",
    "PROMPT_VARIANTS",
    "get_prompt_template",
]
