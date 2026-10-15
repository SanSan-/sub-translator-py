from __future__ import annotations

import os
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest

from sub_translate.utils import io_utils


@pytest.fixture(autouse=True)
def _isolate_output_locks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(io_utils, "CACHE_DIR", tmp_path / "locks")


def _create_symlink(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"Символические ссылки недоступны: {exc}")


def test_atomic_write_text_uses_utf8_without_bom(tmp_path: Path) -> None:
    output = tmp_path / "каталог" / "результат.srt"

    io_utils.atomic_write_text(output, "Привет\r\nмир")

    data = output.read_bytes()
    assert not data.startswith(b"\xef\xbb\xbf")
    assert data.decode("utf-8") == "Привет\r\nмир"


def test_atomic_write_text_preserves_existing_file_when_replace_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "result.srt"
    output.write_text("OLD", encoding="utf-8")

    def fail_replace(_source: Path, _target: Path) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(os, "replace", fail_replace)

    with pytest.raises(OSError, match="replace failed"):
        io_utils.atomic_write_text(output, "NEW")

    assert output.read_text(encoding="utf-8") == "OLD"
    assert list(tmp_path.glob(".*.tmp")) == []


def test_atomic_write_text_retries_temporary_windows_file_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "usage.json"
    output.write_text("OLD", encoding="utf-8")
    real_replace = os.replace
    attempts = 0
    delays: list[float] = []

    def replace_after_temporary_lock(source: Path, target: Path) -> None:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise PermissionError(13, "Файл временно занят")
        real_replace(source, target)

    monkeypatch.setattr(os, "replace", replace_after_temporary_lock)
    monkeypatch.setattr(io_utils.time, "sleep", delays.append)

    io_utils.atomic_write_text(output, "NEW")

    assert attempts == 3
    assert delays == pytest.approx([0.02, 0.04])
    assert output.read_text(encoding="utf-8") == "NEW"
    assert list(tmp_path.glob(".*.tmp")) == []


def test_atomic_write_text_refuses_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "result.srt"
    output.write_text("OWNER", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Файл уже существует"):
        io_utils.atomic_write_text(output, "NEW", overwrite=False)

    assert output.read_text(encoding="utf-8") == "OWNER"


def test_atomic_write_json_preserves_unicode(tmp_path: Path) -> None:
    output = tmp_path / "cache.json"

    io_utils.atomic_write_json(output, {"текст": "Привет"})

    assert output.read_text(encoding="utf-8") == '{\n  "текст": "Привет"\n}'


@pytest.mark.parametrize("overwrite", [False, True])
@pytest.mark.parametrize("target_name", ["source", "other", "missing"])
def test_publish_subtitle_text_rejects_existing_and_dangling_links(
    tmp_path: Path,
    overwrite: bool,
    target_name: str,
) -> None:
    source = tmp_path / "source.srt"
    other = tmp_path / "other.srt"
    missing = tmp_path / "missing.srt"
    output = tmp_path / "output.srt"
    source.write_text("SOURCE", encoding="utf-8")
    other.write_text("OTHER", encoding="utf-8")
    _create_symlink(output, {"source": source, "other": other, "missing": missing}[target_name])

    with pytest.raises(ValueError, match="не должен быть ссылкой"):
        io_utils.publish_subtitle_text(output, "NEW", overwrite=overwrite, protected_paths=(source,))

    assert source.read_text(encoding="utf-8") == "SOURCE"
    assert other.read_text(encoding="utf-8") == "OTHER"
    assert not missing.exists()
    assert output.is_symlink()
    assert list(tmp_path.glob(".*.tmp")) == []


@pytest.mark.parametrize("overwrite", [False, True])
def test_publish_subtitle_text_rejects_protected_hardlink(tmp_path: Path, overwrite: bool) -> None:
    source = tmp_path / "source.srt"
    output = tmp_path / "output.srt"
    source.write_text("SOURCE", encoding="utf-8")
    try:
        output.hardlink_to(source)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"Жёсткие ссылки недоступны: {exc}")

    with pytest.raises(ValueError, match="не должны совпадать"):
        io_utils.publish_subtitle_text(output, "NEW", overwrite=overwrite, protected_paths=(source,))

    assert source.read_bytes() == output.read_bytes() == b"SOURCE"


def test_publish_subtitle_text_rejects_windows_reparse_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "output.srt"
    output.write_text("OWNER", encoding="utf-8")
    real_lstat = Path.lstat

    def reparse_metadata(path: Path, **kwargs):
        if path == output:
            return SimpleNamespace(st_mode=stat.S_IFREG, st_file_attributes=stat.FILE_ATTRIBUTE_REPARSE_POINT)
        return real_lstat(path, **kwargs)

    monkeypatch.setattr(Path, "lstat", reparse_metadata)

    with pytest.raises(ValueError, match="точкой повторной обработки"):
        io_utils.publish_subtitle_text(output, "NEW")

    assert output.read_bytes() == b"OWNER"
    assert list(tmp_path.glob(".*.tmp")) == []


def test_publish_subtitle_text_rejects_directory(tmp_path: Path) -> None:
    output = tmp_path / "output.srt"
    output.mkdir()

    with pytest.raises(ValueError, match="обычный файл"):
        io_utils.publish_subtitle_text(output, "NEW")

    assert output.is_dir()


def test_publish_subtitle_text_rechecks_link_after_writing_temporary_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.srt"
    output = tmp_path / "output.srt"
    source.write_text("SOURCE", encoding="utf-8")
    _create_symlink(output, source)
    output.unlink()
    real_fsync = os.fsync

    def substitute_after_sync(descriptor: int) -> None:
        real_fsync(descriptor)
        output.symlink_to(source)

    monkeypatch.setattr(os, "fsync", substitute_after_sync)

    with pytest.raises(ValueError, match="не должен быть ссылкой"):
        io_utils.publish_subtitle_text(output, "NEW", protected_paths=(source,))

    assert source.read_bytes() == b"SOURCE"
    assert output.is_symlink()
    assert list(tmp_path.glob(".*.tmp")) == []


@pytest.mark.parametrize("target_name", ["source", "other"])
def test_publish_subtitle_text_does_not_follow_link_inserted_at_replace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target_name: str,
) -> None:
    source = tmp_path / "source.srt"
    other = tmp_path / "other.srt"
    output = tmp_path / "output.srt"
    source.write_text("SOURCE", encoding="utf-8")
    other.write_text("OTHER", encoding="utf-8")
    target = {"source": source, "other": other}[target_name]
    _create_symlink(output, target)
    output.unlink()
    real_replace = os.replace
    destinations = []

    def substitute_at_replace(temporary: Path, destination: Path) -> None:
        destinations.append(destination)
        output.symlink_to(target)
        real_replace(temporary, destination)

    monkeypatch.setattr(os, "replace", substitute_at_replace)

    try:
        io_utils.publish_subtitle_text(output, "NEW", protected_paths=(source,))
    except OSError, ValueError:
        assert output.is_symlink()
    else:
        assert not output.is_symlink()
        assert output.read_bytes() == b"NEW"

    assert destinations == [output]
    assert source.read_bytes() == b"SOURCE"
    assert other.read_bytes() == b"OTHER"
    assert list(tmp_path.glob(".*.tmp")) == []


def test_publish_subtitle_text_rechecks_link_between_permission_retries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.srt"
    output = tmp_path / "output.srt"
    source.write_text("SOURCE", encoding="utf-8")
    _create_symlink(output, source)
    output.unlink()
    attempts = []
    delays = []

    def locked_replace(_temporary: Path, destination: Path) -> None:
        attempts.append(destination)
        raise PermissionError(13, "Файл временно занят")

    def substitute_while_waiting(delay: float) -> None:
        delays.append(delay)
        output.symlink_to(source)

    monkeypatch.setattr(os, "replace", locked_replace)
    monkeypatch.setattr(io_utils.time, "sleep", substitute_while_waiting)

    with pytest.raises(ValueError, match="не должен быть ссылкой"):
        io_utils.publish_subtitle_text(output, "NEW", protected_paths=(source,))

    assert attempts == [output]
    assert delays == [0.02]
    assert source.read_bytes() == b"SOURCE"
    assert list(tmp_path.glob(".*.tmp")) == []


@pytest.mark.parametrize("writer", [io_utils.atomic_write_text, io_utils.publish_subtitle_text])
def test_atomic_writers_stop_after_six_permission_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    writer,
) -> None:
    output = tmp_path / "output.srt"
    output.write_text("OWNER", encoding="utf-8")
    attempts = []
    delays = []

    def locked_replace(_temporary: Path, destination: Path) -> None:
        attempts.append(destination)
        raise PermissionError(13, "Файл временно занят")

    monkeypatch.setattr(os, "replace", locked_replace)
    monkeypatch.setattr(io_utils.time, "sleep", delays.append)

    with pytest.raises(PermissionError):
        writer(output, "NEW")

    assert attempts == [output] * 6
    assert delays == pytest.approx([0.02, 0.04, 0.06, 0.08, 0.1])
    assert output.read_bytes() == b"OWNER"
    assert list(tmp_path.glob(".*.tmp")) == []


def test_atomic_write_json_keeps_existing_link_semantics(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    output = tmp_path / "cache.json"
    target.write_text("{}", encoding="utf-8")
    _create_symlink(output, target)

    io_utils.atomic_write_json(output, {"text": "Перевод"})

    assert output.is_symlink()
    assert target.read_text(encoding="utf-8") == '{\n  "text": "Перевод"\n}'


def test_output_lock_identity_does_not_resolve_final_link(tmp_path: Path) -> None:
    output = tmp_path / "output.srt"
    expected = io_utils._output_lock_path(output)
    _create_symlink(output, tmp_path / "source.srt")

    assert io_utils._output_lock_path(output) == expected
