import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

from sub_translate import __version__, cli
from sub_translate.cli import build_parser, resolve_io_paths
from sub_translate.enums import FileFormat
from sub_translate.service import ProcessingResult, ProcessingStatus
from sub_translate.translators.base import TranslationError


def test_resolve_io_paths_suffix_overrides_source(tmp_path: Path) -> None:
    input_path = tmp_path / "sub.en.vtt"
    source_lang, output_path = resolve_io_paths(
        input_path,
        None,
        "auto",
        "ru",
        "nllb-600m",
        FileFormat.VTT,
    )
    assert source_lang == "en"
    assert output_path == tmp_path / "sub.nllb-600m.ru.vtt"


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
        "nllb-600m",
        FileFormat.VTT,
    )
    assert source_lang == "en"
    assert output_path == output_arg.resolve()


def test_parser_lists_translategemma() -> None:
    help_text = build_parser().format_help()

    assert "translategemma" in help_text
    assert "translategemma-12b" in help_text
    assert "nllb-600m" in help_text
    assert "seedx" in help_text
    assert "madlad" not in help_text
    assert "seamless" not in help_text
    assert "fsm" not in help_text


def test_parser_reports_package_version(capsys) -> None:
    parser = build_parser()
    arguments = ["--version"]

    with pytest.raises(SystemExit) as exit_info:
        parser.parse_args(arguments)

    assert exit_info.value.code == 0
    assert capsys.readouterr().out.strip() == __version__


def test_parser_help_remains_available_without_processing_arguments(capsys) -> None:
    parser = build_parser()
    arguments = ["--help"]

    with pytest.raises(SystemExit) as exit_info:
        parser.parse_args(arguments)

    assert exit_info.value.code == 0
    assert "--api" in capsys.readouterr().out


def test_parser_requires_explicit_translator(capsys) -> None:
    parser = build_parser()
    arguments = ["--input", "episode.srt"]

    with pytest.raises(SystemExit) as exit_info:
        parser.parse_args(arguments)

    assert exit_info.value.code == cli.EXIT_USAGE_ERROR
    stderr = capsys.readouterr().err
    assert "не указаны обязательные аргументы: --api" in stderr


def test_parser_rejects_empty_translator(capsys) -> None:
    parser = build_parser()
    arguments = ["--input", "episode.srt", "--api", "   "]

    with pytest.raises(SystemExit) as exit_info:
        parser.parse_args(arguments)

    assert exit_info.value.code == cli.EXIT_USAGE_ERROR
    assert "переводчик должен быть указан" in capsys.readouterr().err


@pytest.mark.parametrize(
    "tld",
    ["com@attacker.example", "com/path", "com:443", "attacker.example"],
)
def test_parser_rejects_unsafe_google_tld_without_echo(tld: str, capsys) -> None:
    parser = build_parser()
    arguments = ["--input", "episode.srt", "--api", "google", "--tld", tld]

    with pytest.raises(SystemExit) as exit_info:
        parser.parse_args(arguments)

    assert exit_info.value.code == cli.EXIT_USAGE_ERROR
    stderr = capsys.readouterr().err
    assert "Домен Google Translate не поддерживается" in stderr
    assert tld not in stderr


def test_cli_import_does_not_load_local_model_stack() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            (
                "import sys; import sub_translate.cli; "
                "assert 'torch' not in sys.modules; "
                "assert 'transformers' not in sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
    )

    assert result.returncode == 0, result.stderr


def test_parser_resolves_registry_alias_and_rejects_unknown() -> None:
    parser = build_parser()

    parsed = parser.parse_args(["--input", "episode.srt", "--api", "translate-gemma-4b"])

    assert parsed.api == "translategemma"
    parsed_12b = parser.parse_args(["--input", "episode.srt", "--api", "translate-gemma-12b"])
    assert parsed_12b.api == "translategemma-12b"
    parsed_seedx = parser.parse_args(["--input", "episode.srt", "--api", "seed-x-ppo-7b-awq-int4"])
    assert parsed_seedx.api == "seedx"
    with pytest.raises(SystemExit):
        parser.parse_args(["--input", "episode.srt", "--api", "unknown"])


def test_cli_uses_profile_timeout_unless_user_overrides_it() -> None:
    parser = build_parser()
    slow = parser.parse_args(["--input", "episode.srt", "--api", "seedx"])
    explicit = parser.parse_args(
        ["--input", "episode.srt", "--api", "seedx", "--timeout", "45"]
    )

    assert cli._build_processing_settings(slow, slow.api).timeout == 3_600
    assert cli._build_processing_settings(explicit, explicit.api).timeout == 45


def test_parser_rejects_invalid_positive_number_with_code_2(capsys) -> None:
    parser = build_parser()
    arguments = ["--input", "episode.srt", "--api", "nllb-600m", "--threads", "0"]

    with pytest.raises(SystemExit) as exit_info:
        parser.parse_args(arguments)

    assert exit_info.value.code == cli.EXIT_USAGE_ERROR
    stderr = capsys.readouterr().err
    assert "ошибка:" in stderr
    assert "положительное целое число" in stderr
    assert "Traceback" not in stderr


def test_main_passes_one_processing_settings_object_to_service(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "субтитры с пробелом.en.srt"
    output = tmp_path / "готовый перевод.ru.srt"
    model_path = tmp_path / "модель"
    worker_python_path = tmp_path / "среда" / "python.exe"
    source.write_text("1\n", encoding="utf-8", newline="")
    captured: dict[str, object] = {}

    def process_stub(input_path, settings, **kwargs):
        captured.update(input_path=input_path, settings=settings, kwargs=kwargs)
        return ProcessingResult(input_path, output, ProcessingStatus.OUTPUT_EXISTS, FileFormat.SRT)

    monkeypatch.setattr(cli, "configure_utf8_stdio", Mock())
    monkeypatch.setattr(cli, "load_env", Mock())
    monkeypatch.setattr(cli, "configure_rotating_logger", Mock(return_value=Mock()))
    monkeypatch.setattr(cli, "process_subtitle", process_stub)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sub-translate",
            "--input",
            str(source),
            "--output",
            str(output),
            "--api",
            "google",
            "--model-path",
            str(model_path),
            "--model-revision",
            "revision-1",
            "--worker-python-path",
            str(worker_python_path),
            "--auto-download-model",
            "--force",
        ],
    )

    assert cli.main() == 0
    settings = captured["settings"]
    assert settings.api == "google"
    assert settings.force is True
    assert settings.model_path == model_path.resolve()
    assert settings.model_revision == "revision-1"
    assert settings.worker_python_path == worker_python_path.resolve()
    assert settings.auto_download_model is True
    assert captured["kwargs"]["output_path"] == output.resolve()


def test_main_reports_missing_input_as_invalid_argument(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr(cli, "configure_utf8_stdio", Mock())
    monkeypatch.setattr(cli, "load_env", Mock())
    monkeypatch.setattr(
        sys,
        "argv",
        ["sub-translate", "--input", str(tmp_path / "нет.srt"), "--api", "nllb-600m"],
    )

    with pytest.raises(SystemExit) as exit_info:
        cli.main()

    assert exit_info.value.code == cli.EXIT_USAGE_ERROR
    stderr = capsys.readouterr().err
    assert "ошибка: Файл не найден:" in stderr
    assert "Traceback" not in stderr


def test_main_returns_code_1_for_processing_error_without_traceback(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    source = tmp_path / "source.en.srt"
    source.write_text("1\n", encoding="utf-8", newline="")
    monkeypatch.setattr(cli, "configure_utf8_stdio", Mock())
    monkeypatch.setattr(cli, "load_env", Mock())
    monkeypatch.setattr(cli, "configure_rotating_logger", Mock(return_value=Mock()))
    monkeypatch.setattr(
        cli,
        "process_subtitle",
        Mock(side_effect=TranslationError("Локальная модель недоступна.")),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["sub-translate", "--input", str(source), "--api", "nllb-600m"],
    )

    assert cli.main() == cli.EXIT_PROCESSING_ERROR
    stderr = capsys.readouterr().err
    assert stderr.strip() == "Ошибка: Локальная модель недоступна."
    assert "Traceback" not in stderr
