from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

SUBTITLE_EXTENSIONS = frozenset({".ass", ".srt", ".vtt"})
MAX_PICK_PATHS = 10_000
_LOGGER = logging.getLogger("sub_translate_web")


class PickerError(RuntimeError):
    """Ошибка локального системного диалога или сбора субтитров."""


@dataclass(frozen=True, slots=True)
class PickSelection:
    """Нормализованный результат системного диалога."""

    mode: Literal["files", "folder"]
    paths: tuple[Path, ...]
    folder: Path | None = None


ProgressCallback = Callable[[dict[str, Any]], None]
PathPredicate = Callable[[Path], bool]


def pick_paths(
    kind: Literal["file", "folder"],
    *,
    recursive: bool = False,
    accept_path: PathPredicate | None = None,
    progress_callback: ProgressCallback | None = None,
) -> PickSelection:
    """Открывает системный диалог и возвращает поддерживаемые субтитры."""
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as exc:  # pragma: no cover - зависит от окружения Python
        raise PickerError("Не удалось загрузить системный диалог.") from exc

    root = None
    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        if kind == "folder":
            _report_progress(
                progress_callback,
                phase="dialog",
                message="Открыт системный диалог выбора каталога.",
            )
            selected = filedialog.askdirectory(title="Выберите каталог с субтитрами")
            if not selected:
                return PickSelection(mode="folder", paths=())
            folder = _normalize_selected_folder(selected)
            _report_progress(
                progress_callback,
                phase="collecting",
                message=f"Выбран каталог {folder}. Начат рекурсивный поиск субтитров.",
            )
            return PickSelection(
                mode="folder",
                paths=collect_subtitle_paths(
                    folder,
                    recursive=recursive,
                    accept_path=accept_path,
                    progress_callback=progress_callback,
                ),
                folder=folder,
            )
        if kind != "file":
            raise PickerError(f"Неизвестный режим выбора: {kind}")
        _report_progress(
            progress_callback,
            phase="dialog",
            message="Открыт системный диалог выбора субтитров.",
        )
        selected_paths = filedialog.askopenfilenames(
            title="Выберите субтитры",
            filetypes=[
                ("Субтитры", _subtitle_file_pattern()),
                ("Все файлы", "*.*"),
            ],
        )
        paths = tuple(_normalize_selected_path(value) for value in selected_paths)
        return PickSelection(
            mode="files",
            paths=filter_subtitle_paths(paths, accept_path=accept_path),
        )
    except PickerError:
        raise
    except Exception as exc:  # pragma: no cover - зависит от оконной системы
        raise PickerError("Не удалось завершить системный выбор субтитров.") from exc
    finally:
        if root is not None:
            root.destroy()


def collect_subtitle_paths(
    folder: Path,
    *,
    recursive: bool = True,
    accept_path: PathPredicate | None = None,
    max_paths: int = MAX_PICK_PATHS,
    progress_callback: ProgressCallback | None = None,
) -> tuple[Path, ...]:
    """Собирает субтитры каталога в стабильном порядке без обхода ссылок."""
    if max_paths <= 0:
        raise ValueError("Предел числа файлов должен быть положительным.")
    selected = folder.expanduser()
    try:
        if is_link_or_junction(selected):
            raise PickerError("Каталоги-ссылки не поддерживаются.")
        root = selected.resolve(strict=True)
    except PickerError:
        raise
    except OSError as exc:
        raise PickerError("Выбранный каталог не найден.") from exc
    if not root.is_dir():
        raise PickerError("Выбранный каталог не найден.")

    unique: dict[str, Path] = {}
    scanned = 0
    last_report = time.monotonic()
    directories = [root]
    while directories:
        current = directories.pop()
        entries = _read_directory_entries(
            current,
            root=root,
            discovered=len(unique),
            progress_callback=progress_callback,
        )
        if entries is None:
            continue
        nested: list[Path] = []
        for entry in entries:
            scanned += 1
            nested_path = _collect_directory_entry(
                entry,
                recursive=recursive,
                accept_path=accept_path,
                unique=unique,
                max_paths=max_paths,
                progress_callback=progress_callback,
            )
            if nested_path is not None:
                nested.append(nested_path)
            last_report = _report_scan_progress(
                scanned,
                last_report=last_report,
                discovered=len(unique),
                progress_callback=progress_callback,
            )
        directories.extend(reversed(nested))

    paths = tuple(sorted(unique.values(), key=_path_sort_key))
    _report_progress(
        progress_callback,
        phase="collecting",
        discovered=len(paths),
        total=len(paths),
        message=f"Сбор каталога завершён: найдено субтитров — {len(paths)}.",
    )
    return paths


def filter_subtitle_paths(
    paths: Iterable[str | Path],
    *,
    accept_path: PathPredicate | None = None,
    max_paths: int = MAX_PICK_PATHS,
) -> tuple[Path, ...]:
    """Фильтрует явный выбор без преждевременной проверки существования."""
    if max_paths <= 0:
        raise ValueError("Предел числа файлов должен быть положительным.")
    unique: dict[str, Path] = {}
    for raw_path in paths:
        path = Path(raw_path)
        if not _is_supported_subtitle(path) or (accept_path is not None and not accept_path(path)):
            continue
        _add_unique_candidate(unique, path, max_paths=max_paths)
    return tuple(sorted(unique.values(), key=_path_sort_key))


def _read_directory_entries(
    current: Path,
    *,
    root: Path,
    discovered: int,
    progress_callback: ProgressCallback | None,
) -> list[os.DirEntry[str]] | None:
    try:
        if is_link_or_junction(current):
            if current == root:
                raise PickerError("Каталоги-ссылки не поддерживаются.")
            _report_progress(
                progress_callback,
                phase="collecting",
                discovered=discovered,
                message=f"Пропущен вложенный каталог-ссылка: {current}.",
            )
            return None
        with os.scandir(current) as entries:
            return sorted(entries, key=lambda entry: (entry.name.casefold(), entry.name))
    except PickerError:
        raise
    except OSError as exc:
        if current == root:
            raise PickerError("Не удалось прочитать выбранный каталог.") from exc
        _LOGGER.exception(
            "Пропущен недоступный вложенный каталог %s (%s).",
            current,
            type(exc).__name__,
        )
        _report_progress(
            progress_callback,
            phase="collecting",
            discovered=discovered,
            message=f"Пропущен недоступный вложенный каталог: {current}.",
        )
        return None


def _collect_directory_entry(
    entry: os.DirEntry[str],
    *,
    recursive: bool,
    accept_path: PathPredicate | None,
    unique: dict[str, Path],
    max_paths: int,
    progress_callback: ProgressCallback | None,
) -> Path | None:
    candidate = Path(entry.path)
    try:
        if is_link_or_junction(entry.path):
            return None
        if recursive and entry.is_dir(follow_symlinks=False):
            return candidate
        if not entry.is_file(follow_symlinks=False):
            return None
        if not _is_supported_subtitle(candidate):
            return None
        if accept_path is not None and not accept_path(candidate):
            return None
        _add_unique_candidate(unique, candidate, max_paths=max_paths)
    except OSError as exc:
        _LOGGER.exception(
            "Не удалось проверить элемент каталога %s (%s).",
            candidate,
            type(exc).__name__,
        )
        _report_progress(
            progress_callback,
            phase="collecting",
            discovered=len(unique),
            message=f"Не удалось проверить элемент каталога {entry.name}.",
        )
        candidate_is_accepted = _is_supported_subtitle(candidate)
        if candidate_is_accepted and accept_path is not None:
            candidate_is_accepted = accept_path(candidate)
        if candidate_is_accepted:
            _add_unique_candidate(unique, candidate, max_paths=max_paths)
    return None


def _add_unique_candidate(
    unique: dict[str, Path],
    candidate: Path,
    *,
    max_paths: int,
) -> None:
    unique.setdefault(_path_identity(candidate), candidate)
    if len(unique) > max_paths:
        raise PickerError(f"За один запуск можно выбрать не более {max_paths} файлов.")


def _report_scan_progress(
    scanned: int,
    *,
    last_report: float,
    discovered: int,
    progress_callback: ProgressCallback | None,
) -> float:
    now = time.monotonic()
    if scanned % 128 != 0 and now - last_report < 0.25:
        return last_report
    _report_progress(
        progress_callback,
        phase="collecting",
        discovered=discovered,
    )
    return now


def is_link_or_junction(path: str | Path) -> bool:
    """Определяет символическую ссылку или Windows junction без обхода цели."""
    candidate = Path(path)
    if candidate.is_symlink():
        return True
    is_junction = getattr(os.path, "isjunction", None)
    return bool(is_junction and is_junction(candidate))


def _normalize_selected_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    try:
        return path.resolve(strict=False)
    except OSError:
        return Path(os.path.abspath(path))


def _normalize_selected_folder(value: str | Path) -> Path:
    return Path(os.path.abspath(Path(value).expanduser()))


def _subtitle_file_pattern() -> str:
    return " ".join(f"*{extension}" for extension in sorted(SUBTITLE_EXTENSIONS))


def _is_supported_subtitle(path: Path) -> bool:
    return path.suffix.casefold() in SUBTITLE_EXTENSIONS


def _path_identity(path: Path) -> str:
    return os.path.normcase(os.path.abspath(path))


def _path_sort_key(path: Path) -> tuple[str, str]:
    value = str(path)
    return value.casefold(), value


def _report_progress(
    callback: ProgressCallback | None,
    *,
    phase: str,
    discovered: int | None = None,
    total: int | None = None,
    message: str | None = None,
) -> None:
    if callback is None:
        return
    event: dict[str, Any] = {"phase": phase}
    if discovered is not None:
        event["discovered"] = discovered
    if total is not None:
        event["total"] = total
    if message:
        event["message"] = message
    try:
        callback(event)
    except Exception:
        return


__all__ = [
    "MAX_PICK_PATHS",
    "SUBTITLE_EXTENSIONS",
    "PickSelection",
    "PickerError",
    "collect_subtitle_paths",
    "filter_subtitle_paths",
    "pick_paths",
]
