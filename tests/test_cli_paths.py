from pathlib import Path

from sub_translate.cli import resolve_io_paths
from sub_translate.enums import FileFormat


def test_resolve_io_paths_suffix_overrides_source(tmp_path: Path) -> None:
    input_path = tmp_path / "sub.en.vtt"
    source_lang, output_path = resolve_io_paths(
        input_path,
        None,
        "auto",
        "ru",
        "fsm",
        FileFormat.VTT,
    )
    assert source_lang == "en"
    assert output_path == tmp_path / "sub.fsm.ru.vtt"


def test_resolve_io_paths_includes_api(tmp_path: Path) -> None:
    input_path = tmp_path / "sub.vtt"
    source_lang, output_path = resolve_io_paths(
        input_path,
        None,
        "en",
        "ru",
        "google",
        FileFormat.VTT,
    )
    assert source_lang == "en"
    assert output_path == tmp_path / "sub.google.ru.vtt"


def test_resolve_io_paths_respects_output_arg(tmp_path: Path) -> None:
    input_path = tmp_path / "sub.en.vtt"
    output_arg = tmp_path / "custom.vtt"
    source_lang, output_path = resolve_io_paths(
        input_path,
        str(output_arg),
        "auto",
        "ru",
        "fsm",
        FileFormat.VTT,
    )
    assert source_lang == "en"
    assert output_path == output_arg.resolve()
