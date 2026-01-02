from __future__ import annotations

from typing import Dict, List

from sub_translate.dictionaries.regex import ASS_MASK, SRT_INDEX_MASK, SRT_TIME_MASK
from sub_translate.models import AssSubtitlesItem


def ass_validator(text: str) -> bool:
    return bool(ASS_MASK.match(text))


def ass_separator(line: int, text: str) -> Dict[int, AssSubtitlesItem]:
    match = ASS_MASK.match(text)
    if not match or (match.lastindex or 0) < 10:
        return {}
    return {
        line: AssSubtitlesItem(
            layer=int(match.group(1) or 0),
            start_time=match.group(2),
            end_time=match.group(3),
            style=match.group(4),
            actor=match.group(5),
            margin_l=int(match.group(6) or 0),
            margin_r=int(match.group(7) or 0),
            margin_v=int(match.group(8) or 0),
            effect=match.group(9),
            text=match.group(10),
        )
    }


def srt_start_validator(text: str) -> bool:
    return bool(SRT_INDEX_MASK.match(text))


def srt_time_validator(text: str) -> bool:
    return bool(SRT_TIME_MASK.match(text))


def srt_time_extract(text: str) -> List[str]:
    match = SRT_TIME_MASK.match(text)
    if not match:
        return []
    return [match.group(1), match.group(2)]


def count_regexp_entry(text: str, pattern) -> int:
    return len(list(pattern.finditer(text)))


__all__ = [
    "ass_validator",
    "ass_separator",
    "srt_start_validator",
    "srt_time_validator",
    "srt_time_extract",
    "count_regexp_entry",
]