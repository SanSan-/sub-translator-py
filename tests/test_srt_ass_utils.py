import json
from pathlib import Path

from sub_translate.constants import BIG_NEW_LINE_SIGN
from sub_translate.enums import FileFormat
from sub_translate.utils.line_utils import build_export_lines, parse_ass_dialogs, parse_srt_dialogs


def _load_fixture(name: str) -> list[str]:
    path = Path(__file__).parent / "data" / name
    return json.loads(path.read_text(encoding="utf-8"))


def test_parse_srt_dialogs_free_movie_sample() -> None:
    dialogs = parse_srt_dialogs(_load_fixture("freeMovie.srt.json"))
    dialog = dialogs[3]
    assert dialog.start_time == "00:00:46,416"
    assert dialog.end_time == "00:00:49,666"
    assert dialog.text is not None
    assert BIG_NEW_LINE_SIGN in dialog.text


def test_build_export_lines_srt_order() -> None:
    origins = _load_fixture("freeMovie.srt.json")
    dialogs = parse_srt_dialogs(origins)
    dialog = dialogs[3]
    translated = {3: dialog.text or ""}
    lines = build_export_lines(origins, FileFormat.SRT.value, dialogs, translated)
    parts = (dialog.text or "").split(BIG_NEW_LINE_SIGN)
    assert lines[0] == "3"
    assert lines[1] == f"{dialog.start_time} --> {dialog.end_time}"
    assert lines[2 : 2 + len(parts)] == parts
    assert lines[2 + len(parts)] == ""


def test_build_export_lines_ass_keeps_dialogue_line() -> None:
    origins = _load_fixture("akebi11.ass.json")
    target_line = (
        "Dialogue: 0,0:00:06.47,0:00:09.73,main,Riri,0000,0000,0000,,"
        "Oh, they have that store in Tokyo, too?"
    )
    target_index = origins.index(target_line)
    dialogs = parse_ass_dialogs(origins)
    translated = {target_index: dialogs[target_index].text or ""}
    lines = build_export_lines(origins, FileFormat.ASS.value, dialogs, translated)
    parts = lines[target_index].split(",", 9)
    assert parts[0] == "Dialogue: 0"
    assert parts[1] == "0:00:06.47"
    assert parts[2] == "0:00:09.73"
    assert parts[3] == "main"
    assert parts[4] == "Riri"
    assert parts[5:8] == ["0", "0", "0"]
    assert parts[8] == ""
    assert parts[9] == "Oh, they have that store in Tokyo, too?"