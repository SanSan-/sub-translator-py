import re

ITALIAN_MASK = re.compile(r"{\\i[01]}", re.IGNORECASE)
ASS_EFFECTS_MASK = re.compile(r"{\\[^}]+}", re.IGNORECASE)
ASS_COMMENTS_MASK = re.compile(r"{(?!\\)[^}]*}", re.IGNORECASE)
SRT_EFFECTS_MASK = re.compile(r"</?\w+>", re.IGNORECASE)
DRAW_MASK = re.compile(
    r"{(?:[^\\\r\n]|\\(?!p(?:bo|\d)))*\\p(?:bo|\d)[^\r\n]*",
    re.IGNORECASE,
)
DOUBLE_SPACES_MASK = re.compile(r"\s{2,}", re.IGNORECASE)
NEXT_LINE_MASK = re.compile(r"\\n", re.IGNORECASE)
NO_SPACE_NEXT_LINE_MASK = re.compile(r"\\N([\S]?)(?:$|\\N)", re.IGNORECASE)
DOT_MASK = re.compile(r"[.:]", re.IGNORECASE)
COMMA_MASK = re.compile(r",", re.IGNORECASE)
QUOTE_MASK = re.compile(r'["><]', re.IGNORECASE)
BRACKET_MASK = re.compile(r"[()]", re.IGNORECASE)
DASH_MASK = re.compile(r"-", re.IGNORECASE)
COLON_MASK = re.compile(r":", re.IGNORECASE)
SEMICOLON_MASK = re.compile(r";", re.IGNORECASE)
QUESTION_MARK_MASK = re.compile(r"\?", re.IGNORECASE)
EXCLAMATION_MARK_MASK = re.compile(r"!", re.IGNORECASE)
GOOD_END_SYMBOLS_MASK = re.compile(r'[.|?\-)"!~](</[\w]+>)?$', re.IGNORECASE)
_ASS_PREFIX_PATTERN = r"^(Dialogue|Comment): (\d++),"
_ASS_TIMING_PATTERN = r"(\d++:\d++:\d++\.\d++),(\d++:\d++:\d++\.\d++),"
_ASS_STYLE_PATTERN = r"([^,]*+),([^,]*+),"
_ASS_MARGIN_PATTERN = r"(\d++),(\d++),(\d++),"
_ASS_SUFFIX_PATTERN = r"([^,]*+),(.*+)$"
ASS_MASK = re.compile(
    _ASS_PREFIX_PATTERN + _ASS_TIMING_PATTERN + _ASS_STYLE_PATTERN + _ASS_MARGIN_PATTERN + _ASS_SUFFIX_PATTERN,
    re.IGNORECASE,
)
SRT_INDEX_MASK = re.compile(r"^(\d+)$", re.IGNORECASE)
SRT_TIME_MASK = re.compile(
    r"^(\d{0,2}:?\d{2}:\d{2}[,.]\d{2,3}) --> (\d{0,2}:?\d{2}:\d{2}[,.]\d{2,3})$",
    re.IGNORECASE,
)


class _VttHeaderPattern:
    """Совместимый с ``fullmatch`` линейный проверяющий объект для заголовка WebVTT."""

    @staticmethod
    def fullmatch(text: str) -> str | None:
        if text == _VTT_HEADER:
            return text
        if not text.startswith(_VTT_HEADER):
            return None
        metadata = text[len(_VTT_HEADER) :]
        if not metadata or metadata[0] not in " \t":
            return None
        if "\r" in metadata or "\n" in metadata or "-->" in metadata:
            return None
        return text


_VTT_HEADER = "WEBVTT"
VTT_HEADER_MASK = _VttHeaderPattern()
_VTT_TIMESTAMP_PATTERN = r"(?:\d{2,}:)?\d{2}:\d{2}\.\d{3}"
_VTT_CUE_SETTING_PATTERN = (
    r"(?:vertical:(?:rl|lr)|line:\S+|position:\S+|size:\S+|"
    r"align:(?:start|center|middle|end|left|right)|region:\S+)"
)
VTT_TIME_MASK = re.compile(
    rf"^({_VTT_TIMESTAMP_PATTERN}) --> ({_VTT_TIMESTAMP_PATTERN})"
    rf"((?:[ \t]+{_VTT_CUE_SETTING_PATTERN})*)$"
)
FILENAME_MASK = re.compile(
    r"filename[^;\n]*=\s*(UTF-\d['\"]*)?((['\"])[^;\n]*\3|[^;\n]+)?",
    re.IGNORECASE,
)

__all__ = [
    "ASS_COMMENTS_MASK",
    "ASS_EFFECTS_MASK",
    "ASS_MASK",
    "BRACKET_MASK",
    "COLON_MASK",
    "COMMA_MASK",
    "DASH_MASK",
    "DOT_MASK",
    "DOUBLE_SPACES_MASK",
    "DRAW_MASK",
    "EXCLAMATION_MARK_MASK",
    "FILENAME_MASK",
    "GOOD_END_SYMBOLS_MASK",
    "ITALIAN_MASK",
    "NEXT_LINE_MASK",
    "NO_SPACE_NEXT_LINE_MASK",
    "QUESTION_MARK_MASK",
    "QUOTE_MASK",
    "SEMICOLON_MASK",
    "SRT_EFFECTS_MASK",
    "SRT_INDEX_MASK",
    "SRT_TIME_MASK",
    "VTT_HEADER_MASK",
    "VTT_TIME_MASK",
]
