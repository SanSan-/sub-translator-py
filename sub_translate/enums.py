from __future__ import annotations

from enum import Enum, StrEnum


class FileFormat(StrEnum):
    ASS = "ass"
    SRT = "srt"
    VTT = "vtt"

    __str__ = Enum.__str__
    __format__ = Enum.__format__


__all__ = ["FileFormat"]
