import re

ITALIAN_MASK = re.compile(r"{\\i[0|1]}", re.IGNORECASE)
ASS_EFFECTS_MASK = re.compile(r"({\\[^}]+})", re.IGNORECASE)
ASS_COMMENTS_MASK = re.compile(r"({(?!\\).*})", re.IGNORECASE)
SRT_EFFECTS_MASK = re.compile(r"(</?[\w]+>)", re.IGNORECASE)
DRAW_MASK = re.compile(r"({.*\\p(\d|bo).*}.+[{\\p0}]?)", re.IGNORECASE)
DOUBLE_SPACES_MASK = re.compile(r"\s{2,}", re.IGNORECASE)
NEXT_LINE_MASK = re.compile(r"\\n|\\N", re.IGNORECASE)
NO_SPACE_NEXT_LINE_MASK = re.compile(r"\\N([\S]?)(?:$|\\N)", re.IGNORECASE)
DOT_MASK = re.compile(r"[.:]", re.IGNORECASE)
COMMA_MASK = re.compile(r",", re.IGNORECASE)
QUOTE_MASK = re.compile(r"[\"><\"]", re.IGNORECASE)
BRACKET_MASK = re.compile(r"[()]", re.IGNORECASE)
DASH_MASK = re.compile(r"-", re.IGNORECASE)
COLON_MASK = re.compile(r":", re.IGNORECASE)
SEMICOLON_MASK = re.compile(r";", re.IGNORECASE)
QUESTION_MARK_MASK = re.compile(r"\?", re.IGNORECASE)
EXCLAMATION_MARK_MASK = re.compile(r"!", re.IGNORECASE)
GOOD_END_SYMBOLS_MASK = re.compile(r'[.|?\-)"!~](</[\w]+>)?$', re.IGNORECASE)
ASS_MASK = re.compile(
    r"^Dialogue: ([\d]+)[,]([\d]+[:][\d]+[:][\d]+[.][\d]+)[,]([\d]+[:][\d]+[:][\d]+[.][\d]+)[,]([^,]*)[,]([^,]*)[,]([\d]+)[,]([\d]+)[,]([\d]+)[,]([^,]*)[,](.*)$",
    re.IGNORECASE,
)
SRT_INDEX_MASK = re.compile(r"^([\d]+)$", re.IGNORECASE)
SRT_TIME_MASK = re.compile(
    r"^([\d]{0,2}:?[\d]{2}:[\d]{2}[,|.][\d]{2,3}) --> ([\d]{0,2}:?[\d]{2}:[\d]{2}[,|.][\d]{2,3})$",
    re.IGNORECASE,
)
FILENAME_MASK = re.compile(
    r"filename[^;\n]*=\s*(UTF-\d['\"]*)?((['\"]).*?[.]$\3|[^;\n]*)?",
    re.IGNORECASE,
)

__all__ = [
    "ITALIAN_MASK",
    "ASS_EFFECTS_MASK",
    "ASS_COMMENTS_MASK",
    "SRT_EFFECTS_MASK",
    "DRAW_MASK",
    "DOUBLE_SPACES_MASK",
    "NEXT_LINE_MASK",
    "NO_SPACE_NEXT_LINE_MASK",
    "DOT_MASK",
    "COMMA_MASK",
    "QUOTE_MASK",
    "BRACKET_MASK",
    "DASH_MASK",
    "COLON_MASK",
    "SEMICOLON_MASK",
    "QUESTION_MARK_MASK",
    "EXCLAMATION_MARK_MASK",
    "GOOD_END_SYMBOLS_MASK",
    "ASS_MASK",
    "SRT_INDEX_MASK",
    "SRT_TIME_MASK",
    "FILENAME_MASK",
]