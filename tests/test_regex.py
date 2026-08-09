import pytest

from sub_translate.dictionaries.regex import FILENAME_MASK
from sub_translate.utils.validation_utils import (
    ass_separator,
    srt_time_extract,
    srt_time_validator,
    vtt_header_validator,
)


def test_ass_dialog_pattern_preserves_all_fields_and_commas_in_text() -> None:
    line = "Dialogue: 1,0:00:01.25,0:00:02.50,Default,Actor,10,20,30,fx,Hello, world"

    dialog = ass_separator(7, line)[7]

    assert dialog.layer == 1
    assert dialog.start_time == "0:00:01.25"
    assert dialog.end_time == "0:00:02.50"
    assert dialog.style == "Default"
    assert dialog.actor == "Actor"
    assert (dialog.margin_l, dialog.margin_r, dialog.margin_v) == (10, 20, 30)
    assert dialog.effect == "fx"
    assert dialog.text == "Hello, world"


@pytest.mark.parametrize(
    ("time_line", "expected"),
    [
        ("00:01:02,345 --> 00:01:04,567", ["00:01:02,345", "00:01:04,567"]),
        ("01:02.34 --> 01:04.56", ["01:02.34", "01:04.56"]),
    ],
)
def test_srt_time_pattern_accepts_supported_timecodes(time_line: str, expected: list[str]) -> None:
    assert srt_time_validator(time_line)
    assert srt_time_extract(time_line) == expected


def test_srt_time_pattern_rejects_pipe_as_fraction_separator() -> None:
    assert not srt_time_validator("00:01:02|345 --> 00:01:04,567")


def test_vtt_header_validation_handles_long_metadata_without_backtracking() -> None:
    metadata = "описание" * 20_000

    assert vtt_header_validator(f"WEBVTT {metadata}")
    assert not vtt_header_validator(f"WEBVTT {metadata} --> реплика")


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        ('attachment; filename="subtitle.srt"', '"subtitle.srt"'),
        ("attachment; filename=subtitle.srt", "subtitle.srt"),
        ("attachment; filename='subtitle file.srt'", "'subtitle file.srt'"),
    ],
)
def test_filename_pattern_extracts_quoted_and_unquoted_values(header: str, expected: str) -> None:
    match = FILENAME_MASK.search(header)

    assert match is not None
    assert match.group(2) == expected
