from __future__ import annotations

import os
from pathlib import Path

import pytest

from sub_translate.utils import io_utils


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

    def replace_after_temporary_lock(source: Path, target: Path) -> None:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise PermissionError(13, "Файл временно занят")
        real_replace(source, target)

    monkeypatch.setattr(os, "replace", replace_after_temporary_lock)
    monkeypatch.setattr(io_utils.time, "sleep", lambda _seconds: None)

    io_utils.atomic_write_text(output, "NEW")

    assert attempts == 3
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
