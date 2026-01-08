from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, TypedDict

from sub_translate.constants import (
    SMART_SPLIT_MAX_CHARS,
    SMART_SPLIT_MAX_DURATION_MS,
    SMART_SPLIT_MAX_GAP_MS,
    SMART_SPLIT_MAX_LINES,
    SMART_SPLIT_MAX_WORDS,
)

@dataclass(slots=True)
class SrtSubtitlesItem:
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    text: Optional[str] = None
    cue_id: Optional[str] = None


@dataclass(slots=True)
class AssSubtitlesItem(SrtSubtitlesItem):
    layer: Optional[int] = None
    style: Optional[str] = None
    actor: Optional[str] = None
    margin_l: Optional[int] = None
    margin_r: Optional[int] = None
    margin_v: Optional[int] = None
    effect: Optional[str] = None


@dataclass(slots=True)
class PrepareToTranslateItem:
    idx: int
    to_translate: str
    lines: List[int]


@dataclass(slots=True)
class TranslatedItem:
    idx: int
    text: str
    lines: List[int]


@dataclass(slots=True)
class TranslationOptions:
    source_lang: Optional[str] = None
    target_lang: Optional[str] = None
    api: Optional[str] = None
    tld: Optional[str] = None
    request_delay_ms: Optional[int] = None
    allow_cpu_fallback: bool = False
    agent_model: Optional[str] = None
    openai_api_key: Optional[str] = None
    except_paths: Optional[List[str]] = None
    detail: bool = False


@dataclass(slots=True)
class SmartSplitSettings:
    max_lines: int = SMART_SPLIT_MAX_LINES
    max_words: int = SMART_SPLIT_MAX_WORDS
    max_chars: int = SMART_SPLIT_MAX_CHARS
    max_gap_ms: int = SMART_SPLIT_MAX_GAP_MS
    max_duration_ms: int = SMART_SPLIT_MAX_DURATION_MS


class AnalysedLine(TypedDict, total=False):
    wordCount: int
    dotCount: int
    commaCount: int
    quoteCount: int
    bracketCount: int
    dashCount: int
    colonCount: int
    semicolonCount: int
    questionMarkCount: int
    exclamationMarkCount: int
    effects: Dict[int, str]


class AnalysedDialog(AnalysedLine, total=False):
    lines: List[AnalysedLine]


AnalysedItem = Dict[int, AnalysedDialog]
TranslatedDialogItem = Dict[int, str]


__all__ = [
    "SrtSubtitlesItem",
    "AssSubtitlesItem",
    "PrepareToTranslateItem",
    "TranslatedItem",
    "TranslationOptions",
    "AnalysedLine",
    "AnalysedDialog",
    "AnalysedItem",
    "TranslatedDialogItem",
    "SmartSplitSettings",
]
