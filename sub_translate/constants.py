from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

BASE_DIR = Path(__file__).resolve().parents[1]
RESOURCES_DIR = BASE_DIR / "resources"
CACHE_DIR = RESOURCES_DIR / "cache"
PERSIST_DIR = RESOURCES_DIR / "persist"
LOGS_DIR = BASE_DIR / "logs"
MODELS_DIR = BASE_DIR / "models"
TRANSLATOR_LOGS_DIR = LOGS_DIR / "translators"
USAGE_STATS_FILE = CACHE_DIR / "usage_stats.json"

NEW_LINE_SIGN = "\n"
HARD_NEW_LINE_SIGN = "\r\n"
BIG_NEW_LINE_SIGN = "\\N"
EQUAL_SIGN = "="
PLUS_SIGN = "+"
MINUS_SIGN = "-"
UNDERGROUND_SIGN = "_"
SLASH_SIGN = "/"
DOT_SIGN = "."
COMMA_SIGN = ","
SEMICOLON_SIGN = ";"
AMPERSAND_SIGN = "&"
ASTERISK_SIGN = "*"
QUESTION_SIGN = "?"
RIGHT_COMA_SIGN = ")"
LEFT_COMA_SIGN = "("
SPACE_SIGN = " "
FIRST_GROUP = "\\1"
SECOND_GROUP = "\\2"
EMPTY_STRING = ""
QUOTE_JOINER = ", "
SEMICOLON_JOINER = "; "

ZERO_SIGN = "0"
ZERO_INT_SIGN = 0

WEBVTT = "WEBVTT"
NOT_VTT_ERROR = "Это не субтитры WEBVTT."

DEFAULT_BATCH_SIZE = 7
DEFAULT_THREAD_COUNT = 1
DEFAULT_MAX_LEN = 7000


def correct_sort(values: Iterable[int]) -> List[int]:
    """Сортирует индексы так же, как в оригинале."""
    return sorted(values)


__all__ = [
    "BASE_DIR",
    "RESOURCES_DIR",
    "CACHE_DIR",
    "PERSIST_DIR",
    "LOGS_DIR",
    "MODELS_DIR",
    "TRANSLATOR_LOGS_DIR",
    "USAGE_STATS_FILE",
    "NEW_LINE_SIGN",
    "HARD_NEW_LINE_SIGN",
    "BIG_NEW_LINE_SIGN",
    "EQUAL_SIGN",
    "PLUS_SIGN",
    "MINUS_SIGN",
    "UNDERGROUND_SIGN",
    "SLASH_SIGN",
    "DOT_SIGN",
    "COMMA_SIGN",
    "SEMICOLON_SIGN",
    "AMPERSAND_SIGN",
    "ASTERISK_SIGN",
    "QUESTION_SIGN",
    "RIGHT_COMA_SIGN",
    "LEFT_COMA_SIGN",
    "SPACE_SIGN",
    "FIRST_GROUP",
    "SECOND_GROUP",
    "EMPTY_STRING",
    "QUOTE_JOINER",
    "SEMICOLON_JOINER",
    "ZERO_SIGN",
    "ZERO_INT_SIGN",
    "WEBVTT",
    "NOT_VTT_ERROR",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_THREAD_COUNT",
    "DEFAULT_MAX_LEN",
    "correct_sort",
]
