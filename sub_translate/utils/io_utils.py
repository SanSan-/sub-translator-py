from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from filelock import FileLock

from sub_translate.constants import CACHE_DIR, HARD_NEW_LINE_SIGN, NEW_LINE_SIGN

_ATOMIC_REPLACE_ATTEMPTS = 6
_ATOMIC_REPLACE_DELAY_SECONDS = 0.02


def configure_utf8_stdio() -> None:
    """Настраивает доступные стандартные потоки на UTF-8 без изменения файлов."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")


def read_text(path: Path) -> str:
    data = path.read_text(encoding="utf-8")
    if data.startswith("\ufeff"):
        data = data.lstrip("\ufeff")
    return data


def split_lines(text: str) -> list[str]:
    if HARD_NEW_LINE_SIGN in text:
        return text.split(HARD_NEW_LINE_SIGN)
    return text.split(NEW_LINE_SIGN)


def _output_lock_path(path: Path) -> Path:
    normalized = str(path.expanduser().resolve()).casefold().encode("utf-8")
    digest = hashlib.sha256(normalized).hexdigest()
    return CACHE_DIR / "output-locks" / f"{digest}.lock"


def _replace_with_retry(source: Path, target: Path) -> None:
    """Повторяет атомарную замену при краткой блокировке файла в Windows."""
    for attempt in range(_ATOMIC_REPLACE_ATTEMPTS):
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if attempt + 1 >= _ATOMIC_REPLACE_ATTEMPTS:
                raise
            time.sleep(_ATOMIC_REPLACE_DELAY_SECONDS * (attempt + 1))


def atomic_write_text(path: Path, content: str, *, overwrite: bool = True) -> None:
    """Атомарно записывает UTF-8 без BOM под межпроцессной блокировкой."""
    resolved = path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    lock_path = _output_lock_path(resolved)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(lock_path, timeout=60):
        if not overwrite and resolved.exists():
            raise FileExistsError(f"Файл уже существует: {resolved}")
        temporary = resolved.with_name(f".{resolved.name}.{uuid4().hex}.tmp")
        try:
            with temporary.open("x", encoding="utf-8", newline="") as file:
                file.write(content)
                file.flush()
                os.fsync(file.fileno())
            if not overwrite and resolved.exists():
                raise FileExistsError(f"Файл уже существует: {resolved}")
            _replace_with_retry(temporary, resolved)
        finally:
            temporary.unlink(missing_ok=True)


def atomic_write_json(path: Path, payload: Any) -> None:
    content = json.dumps(payload, ensure_ascii=False, indent=2)
    atomic_write_text(path, content)


def write_lines(path: Path, lines: list[str], *, overwrite: bool = True) -> None:
    content = HARD_NEW_LINE_SIGN.join(lines)
    atomic_write_text(path, content, overwrite=overwrite)


__all__ = [
    "atomic_write_json",
    "atomic_write_text",
    "configure_utf8_stdio",
    "read_text",
    "split_lines",
    "write_lines",
]
