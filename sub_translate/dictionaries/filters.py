def only_end_symbols(key: str) -> bool:
    return key not in {"commaCount", "dashCount"}


def without_dashes(key: str) -> bool:
    return key != "dashCount"


def without_word_count(key: str) -> bool:
    return key != "wordCount"


def without_word_and_dashes(key: str) -> bool:
    return key not in {"wordCount", "dashCount"}


def without_non_end_symbols(key: str) -> bool:
    return key not in {"wordCount", "dashCount", "commaCount"}


__all__ = [
    "only_end_symbols",
    "without_dashes",
    "without_non_end_symbols",
    "without_word_and_dashes",
    "without_word_count",
]
