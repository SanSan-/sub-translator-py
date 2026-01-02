from sub_translate.constants import BIG_NEW_LINE_SIGN, SPACE_SIGN, WEBVTT
from sub_translate.enums import FileFormat
from sub_translate.models import SrtSubtitlesItem
from sub_translate.utils.io_utils import write_lines
from sub_translate.utils.line_utils import build_export_lines, parse_vtt_dialogs


def test_parse_vtt_dialogs_keeps_cue_id() -> None:
    origins = [
        WEBVTT,
        "",
        "1",
        "00:00:00.000 --> 00:00:01.000",
        "Hello",
        "world",
        "",
    ]
    dialogs = parse_vtt_dialogs(origins)
    dialog = dialogs[1]
    assert dialog.cue_id == "1"
    assert dialog.start_time == "00:00:00.000"
    assert dialog.end_time == "00:00:01.000"
    assert dialog.text == f"Hello{SPACE_SIGN}{BIG_NEW_LINE_SIGN}world"


def test_build_export_lines_outputs_vtt_cue_id_before_time() -> None:
    dialogs = {
        1: SrtSubtitlesItem(
            start_time="00:00:00.000",
            end_time="00:00:01.000",
            text="Hello",
            cue_id="1",
        )
    }
    translated_dialogs = {1: f"Hello{BIG_NEW_LINE_SIGN}world"}
    lines = build_export_lines([], FileFormat.VTT.value, dialogs, translated_dialogs)
    assert lines[:4] == [WEBVTT, "", "1", "00:00:00.000 --> 00:00:01.000"]
    assert lines[4:6] == ["Hello", "world"]


def test_write_lines_does_not_double_crlf(tmp_path) -> None:
    path = tmp_path / "out.vtt"
    write_lines(path, ["A", "B"])
    data = path.read_bytes()
    assert b"\r\r\n" not in data
    assert data == b"A\r\nB"
