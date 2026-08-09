"""Вспомогательные функции для постобработки и подготовки переводов."""

from __future__ import annotations

import re
from collections.abc import Callable

_SOURCE_OPEN_TAG = "<SOURCE>"
_SOURCE_CLOSE_TAG = "</SOURCE>"
_SINGLE_EMPHASIS_PATTERN = re.compile(r"(?<!\*)\*(?!\*)([^*\n]++)\*(?!\*)")


def _strip_trailing_line_spaces(text: str) -> str:
    return "\n".join(line.rstrip(" \t") for line in text.split("\n"))


def _decimal_run_end(text: str, start: int) -> int:
    end = start
    while end < len(text) and text[end].isdecimal():
        end += 1
    return end


def _read_exponent_fragment(text: str, start: int) -> tuple[str, int]:
    first_end = _decimal_run_end(text, start)
    if first_end >= len(text) or text[first_end] != "^":
        return text[start:first_end], first_end

    second_start = first_end + 1
    second_end = _decimal_run_end(text, second_start)
    if second_end == second_start:
        return text[start:second_start], second_start

    next_index = second_end
    if second_end < len(text) and text[second_end] == "^":
        next_index += 1
    return text[start:second_end], next_index


def _normalize_exponent_markers(text: str) -> str:
    """Удаляет второй маркер из записи вида ``2^3^`` за линейное время."""
    result: list[str] = []
    index = 0
    while index < len(text):
        if text[index].isdecimal():
            fragment, index = _read_exponent_fragment(text, index)
            result.append(fragment)
        else:
            result.append(text[index])
            index += 1
    return "".join(result)


def strip_source_wrapper(text: str) -> str:
    """Удаляет обёртку <SOURCE>...</SOURCE>, если она присутствует."""
    if not isinstance(text, str):
        return ""
    stripped = text.strip()
    if stripped.startswith(_SOURCE_OPEN_TAG) and stripped.endswith(_SOURCE_CLOSE_TAG):
        return stripped[len(_SOURCE_OPEN_TAG) : -len(_SOURCE_CLOSE_TAG)].strip()
    return text


def postprocess_translation(text: str) -> str:
    """Постобработка перевода: подчищаем Markdown и маркеры."""
    if not text:
        return text
    text = strip_source_wrapper(text)
    text = _SINGLE_EMPHASIS_PATTERN.sub(
        lambda match: f" *{match.group(1).strip()}* ",
        text,
    )
    text = _normalize_exponent_markers(text)
    return _strip_trailing_line_spaces(text)


def _split_sentence_by_words(
    sentence: str,
    token_limit: int,
    token_counter: Callable[[str], int],
) -> list[str]:
    result: list[str] = []
    piece: list[str] = []
    for word in sentence.split():
        tentative = " ".join([*piece, word])
        if piece and token_counter(tentative) > token_limit:
            result.append(" ".join(piece))
            piece = [word]
        else:
            piece.append(word)
    if piece:
        result.append(" ".join(piece))
    return result


def _start_next_sentence(
    result: list[str],
    current: str,
    sentence: str,
    token_limit: int,
    token_counter: Callable[[str], int],
) -> str:
    if current:
        result.append(current)
    if token_counter(sentence) <= token_limit:
        return sentence
    result.extend(_split_sentence_by_words(sentence, token_limit, token_counter))
    return ""


def split_long_text(text: str, token_limit: int, token_counter: Callable[[str], int]) -> list[str]:
    """
    Дробит текст на части, чтобы каждая укладывалась в `token_limit`.

    Использует предложенный `token_counter` для оценки длины.
    """
    if token_counter(text) <= token_limit:
        return [text]

    sentences = filter(None, re.split(r"(?<=[.!?])\s+", text))
    result: list[str] = []
    current = ""

    for sentence in sentences:
        tentative = f"{current} {sentence}".strip() if current else sentence
        if token_counter(tentative) <= token_limit:
            current = tentative
        else:
            current = _start_next_sentence(
                result,
                current,
                sentence,
                token_limit,
                token_counter,
            )

    if current:
        result.append(current)

    return result


def translate_text(
    text: str,
    *,
    token_limit: int,
    token_counter: Callable[[str], int],
    chunk_translator: Callable[[str], str],
    postprocess: Callable[[str], str] | None = postprocess_translation,
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
    translated_lines: list[str] = []

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
