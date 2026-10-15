from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
import time
from collections.abc import Callable, Sequence
from functools import partial
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
    normalized = str(path).casefold().encode("utf-8")
    digest = hashlib.sha256(normalized).hexdigest()
    return CACHE_DIR / "output-locks" / f"{digest}.lock"


def resolve_output_path(path: Path) -> Path:
    """Разрешает родителя, сохраняя конечное имя без перехода по ссылке."""
    expanded = path.expanduser()
    return expanded.parent.resolve() / expanded.name


def validate_distinct_files(path: Path, *, protected_paths: Sequence[Path]) -> None:
    """Отклоняет совпадение путей и доступных файлов, не подменяя обработку ошибок ввода-вывода."""
    for protected in protected_paths:
        if path == protected:
            raise ValueError("Выходной и защищённый пути не должны совпадать.")
        try:
            same_file = path.samefile(protected)
        except OSError:
            continue
        if same_file:
            raise ValueError("Выходной и защищённый пути не должны совпадать.")


def validate_subtitle_output(path: Path, *, protected_paths: Sequence[Path] = ()) -> None:
    """Проверяет нормализованное назначение и файловую идентичность защищённых путей."""
    try:
        destination_stat = path.lstat()
    except FileNotFoundError:
        destination_stat = None
    if destination_stat is not None:
        attributes = getattr(destination_stat, "st_file_attributes", 0)
        if stat.S_ISLNK(destination_stat.st_mode) or attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT:
            raise ValueError("Выходной файл не должен быть ссылкой или точкой повторной обработки.")
        if not stat.S_ISREG(destination_stat.st_mode):
            raise ValueError("Выходной путь должен указывать на обычный файл.")
    validate_distinct_files(path, protected_paths=protected_paths)


def _replace_with_retry(source: Path, target: Path, check_target: Callable[[], None]) -> None:
    """Повторяет атомарную замену при краткой блокировке файла в Windows."""
    for attempt in range(_ATOMIC_REPLACE_ATTEMPTS):
        check_target()
        try:
            os.replace(source, target)
            return
        except PermissionError:
            attempt_number = attempt + 1
            if attempt_number >= _ATOMIC_REPLACE_ATTEMPTS:
                raise
            time.sleep(_ATOMIC_REPLACE_DELAY_SECONDS * attempt_number)


def _check_write_target(path: Path, overwrite: bool, protected_paths: Sequence[Path] | None) -> None:
    if protected_paths is not None:
        validate_subtitle_output(path, protected_paths=protected_paths)
    if not overwrite and path.exists():
        raise FileExistsError(f"Файл уже существует: {path}")


def _atomic_write_text(
    path: Path,
    content: str,
    *,
    overwrite: bool,
    protected_paths: Sequence[Path] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = _output_lock_path(path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    check_target = partial(_check_write_target, path, overwrite, protected_paths)
    with FileLock(lock_path, timeout=60):
        check_target()
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        try:
            with temporary.open("x", encoding="utf-8", newline="") as file:
                file.write(content)
                file.flush()
                os.fsync(file.fileno())
            _replace_with_retry(temporary, path, check_target)
        finally:
            temporary.unlink(missing_ok=True)


def atomic_write_text(path: Path, content: str, *, overwrite: bool = True) -> None:
    """Атомарно записывает UTF-8 без BOM под межпроцессной блокировкой."""
    _atomic_write_text(path.expanduser().resolve(), content, overwrite=overwrite)


def publish_subtitle_text(
    path: Path,
    content: str,
    *,
    overwrite: bool = True,
    protected_paths: Sequence[Path] = (),
) -> None:
    """Публикует субтитры без разрешения конечной ссылки, сохраняя защищённые файлы."""
    destination = resolve_output_path(path)
    protected = tuple(item.expanduser().resolve() for item in protected_paths)
    validate_subtitle_output(destination, protected_paths=protected)
    _atomic_write_text(destination, content, overwrite=overwrite, protected_paths=protected)


def atomic_write_json(path: Path, payload: Any) -> None:
    content = json.dumps(payload, ensure_ascii=False, indent=2)
    atomic_write_text(path, content)


def write_lines(
    path: Path,
    lines: list[str],
    *,
    overwrite: bool = True,
    protected_paths: Sequence[Path] = (),
) -> None:
    content = HARD_NEW_LINE_SIGN.join(lines)
    publish_subtitle_text(path, content, overwrite=overwrite, protected_paths=protected_paths)


__all__ = [
    "atomic_write_json",
    "atomic_write_text",
    "configure_utf8_stdio",
    "publish_subtitle_text",
    "read_text",
    "resolve_output_path",
    "split_lines",
    "validate_distinct_files",
    "validate_subtitle_output",
    "write_lines",
]
