from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

from sub_translate.constants import (
    SMART_SPLIT_MAX_CHARS,
    SMART_SPLIT_MAX_DURATION_MS,
    SMART_SPLIT_MAX_GAP_MS,
    SMART_SPLIT_MAX_LINES,
    SMART_SPLIT_MAX_WORDS,
)


@dataclass(slots=True)
class SrtSubtitlesItem:
    start_time: str | None = None
    end_time: str | None = None
    text: str | None = None
    cue_id: str | None = None


@dataclass(slots=True)
class AssSubtitlesItem(SrtSubtitlesItem):
    event_type: str = "Dialogue"
    translatable: bool = True
    layer: int | None = None
    style: str | None = None
    actor: str | None = None
    margin_l: int | None = None
    margin_r: int | None = None
    margin_v: int | None = None
    effect: str | None = None


@dataclass(slots=True)
class PrepareToTranslateItem:
    idx: int
    to_translate: str
    lines: list[int]


@dataclass(slots=True)
class TranslatedItem:
    idx: int
    text: str
    lines: list[int]


@dataclass(slots=True)
class TranslationOptions:
    source_lang: str | None = None
    target_lang: str | None = None
    api: str | None = None
    tld: str | None = None
    request_delay_ms: int | None = None
    allow_cpu_fallback: bool = False
    model_path: Path | None = None
    model_revision: str | None = None
    worker_python_path: Path | None = None
    auto_download_model: bool = False
    agent_model: str | None = None
    except_paths: list[str] | None = None
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
    effects: dict[int, str]


class AnalysedDialog(AnalysedLine, total=False):
    lines: list[AnalysedLine]


AnalysedItem = dict[int, AnalysedDialog]
TranslatedDialogItem = dict[int, str]


__all__ = [
    "AnalysedDialog",
    "AnalysedItem",
    "AnalysedLine",
    "AssSubtitlesItem",
    "PrepareToTranslateItem",
    "SmartSplitSettings",
    "SrtSubtitlesItem",
    "TranslatedDialogItem",
    "TranslatedItem",
    "TranslationOptions",
]
