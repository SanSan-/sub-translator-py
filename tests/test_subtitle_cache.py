import json
import logging
from pathlib import Path

from sub_translate.enums import FileFormat
from sub_translate.utils.io_utils import read_text, split_lines, write_lines
from sub_translate.utils import subtitle_cache


def _write_cache(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def test_apply_output_cache_restores_from_cache(tmp_path: Path, monkeypatch) -> None:
    cache_path = tmp_path / "cache.json"
    monkeypatch.setattr(subtitle_cache, "SUBTITLE_CACHE_FILE", cache_path)
    input_path = tmp_path / "sub.en.vtt"
    output_path = tmp_path / "sub.fsm.ru.vtt"
    cache_key = subtitle_cache.build_cache_key(input_path, "fsm", "ru")
    _write_cache(
        cache_path,
        {
            cache_key: {
                "format": "vtt",
                "lines": ["WEBVTT", "", "1", "00:00:00.000 --> 00:00:01.000", "Hello"],
            }
        },
    )

    logger = logging.getLogger("test")
    applied = subtitle_cache.apply_output_cache(
        input_path,
        output_path,
        "fsm",
        "ru",
        FileFormat.VTT,
        logger,
    )
    assert applied is True
    assert output_path.exists()
    lines = split_lines(read_text(output_path))
    assert lines[:2] == ["WEBVTT", ""]


def test_apply_output_cache_skips_when_cache_and_file_exist(tmp_path: Path, monkeypatch) -> None:
    cache_path = tmp_path / "cache.json"
    monkeypatch.setattr(subtitle_cache, "SUBTITLE_CACHE_FILE", cache_path)
    input_path = tmp_path / "sub.en.vtt"
    output_path = tmp_path / "sub.fsm.ru.vtt"
    write_lines(output_path, ["OLD"])
    cache_key = subtitle_cache.build_cache_key(input_path, "fsm", "ru")
    _write_cache(cache_path, {cache_key: {"format": "vtt", "lines": ["NEW"]}})

    logger = logging.getLogger("test")
    applied = subtitle_cache.apply_output_cache(
        input_path,
        output_path,
        "fsm",
        "ru",
        FileFormat.VTT,
        logger,
    )
    assert applied is True
    assert split_lines(read_text(output_path)) == ["OLD"]


def test_apply_output_cache_saves_when_file_exists_without_cache(tmp_path: Path, monkeypatch) -> None:
    cache_path = tmp_path / "cache.json"
    monkeypatch.setattr(subtitle_cache, "SUBTITLE_CACHE_FILE", cache_path)
    input_path = tmp_path / "sub.en.vtt"
    output_path = tmp_path / "sub.fsm.ru.vtt"
    write_lines(output_path, ["WEBVTT", "", "1"])

    logger = logging.getLogger("test")
    applied = subtitle_cache.apply_output_cache(
        input_path,
        output_path,
        "fsm",
        "ru",
        FileFormat.VTT,
        logger,
    )
    assert applied is True
    data = json.loads(cache_path.read_text(encoding="utf-8"))
    cache_key = subtitle_cache.build_cache_key(input_path, "fsm", "ru")
    assert data[cache_key]["lines"] == ["WEBVTT", "", "1"]
