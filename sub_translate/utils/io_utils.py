from __future__ import annotations

from pathlib import Path

from sub_translate.constants import HARD_NEW_LINE_SIGN, NEW_LINE_SIGN


def read_text(path: Path) -> str:
    data = path.read_text(encoding="utf-8")
    if data.startswith("\ufeff"):
        data = data.lstrip("\ufeff")
    return data


def split_lines(text: str) -> list[str]:
    if HARD_NEW_LINE_SIGN in text:
        return text.split(HARD_NEW_LINE_SIGN)
    return text.split(NEW_LINE_SIGN)


def write_lines(path: Path, lines: list[str]) -> None:
    content = HARD_NEW_LINE_SIGN.join(lines)
    with path.open("w", encoding="utf-8", newline="") as file:
        file.write(content)


__all__ = ["read_text", "split_lines", "write_lines"]
