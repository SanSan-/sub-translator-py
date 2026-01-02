"""Вспомогательные функции для постобработки и подготовки переводов."""

from __future__ import annotations

import re
from typing import Callable, List, Optional

SOURCE_WRAPPER_PATTERN = re.compile(r"^\s*<SOURCE>\s*(.*?)\s*</SOURCE>\s*$", re.DOTALL)


def strip_source_wrapper(text: str) -> str:
    """Удаляет обёртку <SOURCE>...</SOURCE>, если она присутствует."""
    if not isinstance(text, str):
        return ""
    match = SOURCE_WRAPPER_PATTERN.match(text)
    if match:
        return match.group(1).strip("\r\n")
    return text


def postprocess_translation(text: str) -> str:
    """Постобработка перевода: подчищаем Markdown и маркеры."""
    if not text:
        return text
    text = strip_source_wrapper(text)
    text = re.sub(
        r"(?<!\*)\*(?!\*)([^*\n]+?)(?<!\*)\*(?!\*)",
        lambda match: f" *{match.group(1).strip()}* ",
        text,
    )
    text = re.sub(r"(\d+)\^(\d+)\^", r"\1^\2", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    return text


def split_long_text(text: str, token_limit: int, token_counter: Callable[[str], int]) -> List[str]:
    """
    Дробит текст на части, чтобы каждая укладывалась в `token_limit`.

    Использует предложенный `token_counter` для оценки длины.
    """
    if token_counter(text) <= token_limit:
        return [text]

    sentences = re.split(r"(?<=[.!?])\s+", text)
    result: List[str] = []
    current = ""

    for sentence in sentences:
        if not sentence:
            continue
        tentative = f"{current} {sentence}".strip() if current else sentence
        if token_counter(tentative) <= token_limit:
            current = tentative
            continue

        if current:
            result.append(current)
        current = ""

        if token_counter(sentence) <= token_limit:
            current = sentence
            continue

        words = sentence.split()
        piece: List[str] = []
        for word in words:
            tentative_piece = " ".join(piece + [word])
            if token_counter(tentative_piece) > token_limit and piece:
                result.append(" ".join(piece))
                piece = [word]
            else:
                piece.append(word)
        if piece:
            result.append(" ".join(piece))

    if current:
        if token_counter(current) > token_limit:
            result.extend(split_long_text(current, token_limit, token_counter))
        else:
            result.append(current)

    return result


def translate_text(
    text: str,
    *,
    token_limit: int,
    token_counter: Callable[[str], int],
    chunk_translator: Callable[[str], str],
    postprocess: Optional[Callable[[str], str]] = postprocess_translation,
) -> str:
    """
    Переводит текст построчно, дробя длинные строки на части.

    :param text: исходная строка.
    :param token_limit: максимальное количество токенов на фрагмент.
    :param token_counter: функция, возвращающая количество токенов в строке.
    :param chunk_translator: функция, выполняющая перевод одного фрагмента.
    :param postprocess: функция постобработки результата (по умолчанию `postprocess_translation`).
    """
    lines = text.split("\n")
    translated_lines: List[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            translated_lines.append("")
            continue

        chunks = split_long_text(line, token_limit, token_counter)
        translated_chunks = [chunk_translator(chunk) for chunk in chunks]
        translated_line = " ".join(chunk.strip() for chunk in translated_chunks).strip()
        translated_lines.append(translated_line)

    combined = "\n".join(translated_lines)
    if postprocess is not None:
        combined = postprocess(combined)
    return combined


__all__ = ["postprocess_translation", "split_long_text", "translate_text"]
