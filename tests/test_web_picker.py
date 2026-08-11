from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from sub_translate.web import picker
from sub_translate.web.picker import (
    MAX_PICK_PATHS,
    PickerError,
    collect_subtitle_paths,
    filter_subtitle_paths,
)


def test_filter_subtitle_paths_supports_unicode_spaces_deduplication_and_limit(
    tmp_path: Path,
) -> None:
    first = tmp_path / "Серия 01.SRT"
    second = tmp_path / "подпапка" / "эпизод.vTt"
    unsupported = tmp_path / "заметки.txt"

    assert filter_subtitle_paths([second, first, unsupported, first]) == tuple(
        sorted((first, second), key=lambda path: (str(path).casefold(), str(path))),
    )
    with pytest.raises(PickerError, match="не более 1"):
        filter_subtitle_paths([first, second], max_paths=1)


def test_filter_subtitle_paths_accepts_10000_and_rejects_10001(tmp_path: Path) -> None:
    paths = [tmp_path / f"серия-{index:05}.srt" for index in range(MAX_PICK_PATHS)]

    assert len(filter_subtitle_paths(paths)) == 10_000
    with pytest.raises(PickerError, match="не более 10000"):
        filter_subtitle_paths([*paths, tmp_path / "лишний.srt"])


def test_collect_subtitle_paths_recurses_filters_and_orders_deterministically(
    tmp_path: Path,
) -> None:
    nested = tmp_path / "Уровень 1" / "уровень 2"
    nested.mkdir(parents=True)
    top = tmp_path / "Бета.SRT"
    child = nested / "альфа с пробелом.Ass"
    ignored = nested / "notes.txt"
    target = nested / "готово.ru.vtt"
    for path in (top, child, ignored, target):
        path.write_bytes(b"data")

    collected = collect_subtitle_paths(
        tmp_path,
        accept_path=lambda path: ".ru." not in path.name.casefold(),
    )

    assert collected == tuple(
        sorted((top, child), key=lambda path: (str(path).casefold(), str(path))),
    )


def test_collect_subtitle_paths_can_limit_scan_to_selected_directory(tmp_path: Path) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    top = tmp_path / "top.srt"
    child = nested / "child.vtt"
    top.write_bytes(b"top")
    child.write_bytes(b"child")

    assert collect_subtitle_paths(tmp_path, recursive=False) == (top,)


def test_collect_subtitle_paths_reports_progress_and_tolerates_callback_failure(
    tmp_path: Path,
) -> None:
    for index in range(260):
        (tmp_path / f"серия-{index:03}.srt").write_bytes(b"subtitle")
    calls = 0

    def broken_callback(_event: dict[str, object]) -> None:
        nonlocal calls
        calls += 1
        raise RuntimeError("интерфейс уже закрыт")

    collected = collect_subtitle_paths(tmp_path, progress_callback=broken_callback)

    assert len(collected) == 260
    assert calls >= 3


def test_collect_subtitle_paths_surfaces_unreadable_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        picker.os,
        "scandir",
        lambda _path: (_ for _ in ()).throw(PermissionError("нет доступа")),
    )

    with pytest.raises(PickerError, match="Не удалось прочитать выбранный каталог") as exc_info:
        collect_subtitle_paths(tmp_path)
    assert "нет доступа" not in str(exc_info.value)


def test_collect_subtitle_paths_skips_unreadable_nested_directory_and_continues(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    blocked = tmp_path / "недоступный"
    available = tmp_path / "доступный"
    blocked.mkdir()
    available.mkdir()
    subtitle = available / "лекция.srt"
    subtitle.write_bytes(b"subtitle")
    events: list[dict[str, object]] = []
    real_scandir = picker.os.scandir

    def selective_scandir(path: str | Path):
        if Path(path) == blocked:
            raise PermissionError("нет доступа")
        return real_scandir(path)

    monkeypatch.setattr(picker.os, "scandir", selective_scandir)

    collected = collect_subtitle_paths(tmp_path, progress_callback=events.append)

    assert collected == (subtitle,)
    assert any("Пропущен недоступный" in str(event.get("message")) for event in events)
    assert all("нет доступа" not in str(event) for event in events)


def test_collect_subtitle_paths_keeps_unreadable_supported_file_and_neighbor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    blocked = tmp_path / "01-blocked.srt"
    valid = tmp_path / "02-valid.vtt"

    class Entry:
        def __init__(self, path: Path, *, blocked_file: bool) -> None:
            self.path = str(path)
            self.name = path.name
            self._blocked_file = blocked_file

        @staticmethod
        def is_dir(*, follow_symlinks: bool) -> bool:
            assert follow_symlinks is False
            return False

        def is_file(self, *, follow_symlinks: bool) -> bool:
            assert follow_symlinks is False
            if self._blocked_file:
                raise PermissionError("запрет stat")
            return True

    class Entries:
        def __enter__(self):
            return [Entry(blocked, blocked_file=True), Entry(valid, blocked_file=False)]

        @staticmethod
        def __exit__(*args: object) -> None:
            del args

    monkeypatch.setattr(picker.os, "scandir", lambda _path: Entries())
    monkeypatch.setattr(picker, "is_link_or_junction", lambda _path: False)

    assert collect_subtitle_paths(tmp_path) == (blocked, valid)


def test_collect_subtitle_paths_does_not_follow_directory_symlink(tmp_path: Path) -> None:
    root = tmp_path / "выбрано"
    outside = tmp_path / "снаружи"
    root.mkdir()
    outside.mkdir()
    local = root / "локальный.srt"
    external = outside / "внешний.srt"
    local.write_bytes(b"local")
    external.write_bytes(b"outside")
    link = root / "ссылка"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"Символические ссылки недоступны: {exc}")

    assert collect_subtitle_paths(root) == (local,)
    with pytest.raises(PickerError, match="Каталоги-ссылки"):
        collect_subtitle_paths(link)


def test_collect_directory_entry_does_not_descend_into_windows_junction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class JunctionEntry:
        name = "junction"
        path = r"D:\выбрано\junction"

        @staticmethod
        def is_dir(*, follow_symlinks: bool) -> bool:
            raise AssertionError(f"junction не должен проверяться как каталог: {follow_symlinks}")

        @staticmethod
        def is_file(*, follow_symlinks: bool) -> bool:
            raise AssertionError(f"junction не должен проверяться как файл: {follow_symlinks}")

    monkeypatch.setattr(picker, "is_link_or_junction", lambda path: str(path) == JunctionEntry.path)

    assert (
        picker._collect_directory_entry(
            JunctionEntry(),
            recursive=True,
            accept_path=None,
            unique={},
            max_paths=10,
            progress_callback=None,
        )
        is None
    )


class FakeTkRoot:
    def __init__(self) -> None:
        self.destroyed = False
        self.topmost = False

    def withdraw(self) -> None:
        return None

    def attributes(self, name: str, value: bool) -> None:
        assert name == "-topmost"
        self.topmost = value

    def destroy(self) -> None:
        self.destroyed = True


def _install_fake_tk(
    monkeypatch: pytest.MonkeyPatch,
    *,
    folder: str = "",
    files: tuple[str, ...] = (),
) -> FakeTkRoot:
    root = FakeTkRoot()
    module = ModuleType("tkinter")
    module.Tk = lambda: root
    module.filedialog = SimpleNamespace(
        askdirectory=lambda **_kwargs: folder,
        askopenfilenames=lambda **_kwargs: files,
    )
    monkeypatch.setitem(sys.modules, "tkinter", module)
    return root


def test_pick_paths_returns_recursive_folder_and_destroys_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    subtitle = tmp_path / "эпизод.srt"
    subtitle.write_bytes(b"subtitle")
    root = _install_fake_tk(monkeypatch, folder=str(tmp_path))

    selection = picker.pick_paths("folder", recursive=True)

    assert selection.mode == "folder"
    assert selection.folder == tmp_path
    assert selection.paths == (subtitle,)
    assert root.topmost is True
    assert root.destroyed is True


def test_pick_paths_returns_files_or_cancelled_folder(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    subtitle = tmp_path / "episode.srt"
    _install_fake_tk(monkeypatch, files=(str(subtitle), str(tmp_path / "notes.txt")))

    assert picker.pick_paths("file").paths == (subtitle,)

    _install_fake_tk(monkeypatch, folder="")
    cancelled = picker.pick_paths("folder", recursive=True)
    assert cancelled.folder is None
    assert cancelled.paths == ()


def test_pick_paths_rejects_selected_windows_junction_without_scanning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected = r"D:\субтитры\junction"
    root = _install_fake_tk(monkeypatch, folder=selected)
    monkeypatch.setattr(picker, "is_link_or_junction", lambda path: str(path) == selected)
    monkeypatch.setattr(
        picker.os,
        "scandir",
        lambda _path: (_ for _ in ()).throw(AssertionError("обход junction запрещён")),
    )

    with pytest.raises(PickerError, match="Каталоги-ссылки"):
        picker.pick_paths("folder", recursive=True)
    assert root.destroyed is True


def test_pick_paths_reports_unavailable_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "tkinter", None)

    with pytest.raises(PickerError, match="Не удалось загрузить системный диалог"):
        picker.pick_paths("file")
