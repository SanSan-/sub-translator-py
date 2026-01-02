from __future__ import annotations

from enum import Enum


class FileFormat(str, Enum):
    ASS = "ass"
    SRT = "srt"
    VTT = "vtt"


class TranslatorApiType(str, Enum):
    GOOGLE = "google"
    AGENT = "agent"
    YANDEX = "yandex"


__all__ = ["FileFormat", "TranslatorApiType"]