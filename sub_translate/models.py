from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, TypedDict


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
    except_paths: Optional[List[str]] = None
    detail: bool = False


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
]
