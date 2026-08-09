from __future__ import annotations

from sub_translate.dictionaries.regex import (
    ASS_MASK,
    SRT_INDEX_MASK,
    SRT_TIME_MASK,
    VTT_HEADER_MASK,
    VTT_TIME_MASK,
)
from sub_translate.models import AssSubtitlesItem


def ass_validator(text: str) -> bool:
    return bool(ASS_MASK.match(text))


def ass_separator(line: int, text: str) -> dict[int, AssSubtitlesItem]:
    match = ASS_MASK.match(text)
    if not match or (match.lastindex or 0) < 11:
        return {}
    event_type = match.group(1)
    return {
        line: AssSubtitlesItem(
            event_type=event_type,
            translatable=event_type.casefold() == "dialogue",
            layer=int(match.group(2) or 0),
            start_time=match.group(3),
            end_time=match.group(4),
            style=match.group(5),
            actor=match.group(6),
            margin_l=int(match.group(7) or 0),
            margin_r=int(match.group(8) or 0),
            margin_v=int(match.group(9) or 0),
            effect=match.group(10),
            text=match.group(11),
        )
    }


def srt_start_validator(text: str) -> bool:
    return bool(SRT_INDEX_MASK.match(text))


def srt_time_validator(text: str) -> bool:
    return bool(SRT_TIME_MASK.match(text))


def srt_time_extract(text: str) -> list[str]:
    match = SRT_TIME_MASK.match(text)
    if not match:
        return []
    return [match.group(1), match.group(2)]


def vtt_header_validator(text: str) -> bool:
    return bool(VTT_HEADER_MASK.fullmatch(text))


def vtt_time_validator(text: str) -> bool:
    return bool(VTT_TIME_MASK.fullmatch(text))


def vtt_time_extract(text: str) -> list[str]:
    match = VTT_TIME_MASK.fullmatch(text)
    if not match:
        return []
    return [match.group(1), f"{match.group(2)}{match.group(3)}"]


def count_regexp_entry(text: str, pattern) -> int:
    return len(list(pattern.finditer(text)))


__all__ = [
    "ass_separator",
    "ass_validator",
    "count_regexp_entry",
    "srt_start_validator",
    "srt_time_extract",
    "srt_time_validator",
    "vtt_header_validator",
    "vtt_time_extract",
    "vtt_time_validator",
]
