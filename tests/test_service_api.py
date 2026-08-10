import logging
from pathlib import Path

import pytest

from sub_translate import service
from sub_translate.models import TranslationOptions
from sub_translate.service import (
    ProcessingSettings,
    ProcessingStatus,
    process_subtitle,
    process_subtitle_batch,
    resolve_api,
)
from sub_translate.translators.base import TranslationError

LOGGER = logging.getLogger("test.service")


class FakeTranslator:
    name = "google"

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def translate_batch(self, texts: list[str], _options: TranslationOptions) -> list[str]:
        self.calls.append(texts)
        return [f"Перевод {index}" for index, _text in enumerate(texts, start=1)]

    def unload(self) -> None:
        return None


def write_srt(path: Path, text: str = "Hello") -> None:
    path.write_text(
        f"1\n00:00:00,000 --> 00:00:01,000\n{text}\n",
        encoding="utf-8",
        newline="",
    )


def test_resolve_api_defaults_to_google() -> None:
    assert resolve_api(None) == "google"


@pytest.mark.parametrize("api", [None, "", "   "])
def test_processing_settings_require_explicit_translator(api: str | None) -> None:
    with pytest.raises(ValueError, match="Переводчик должен быть выбран явно"):
        ProcessingSettings(api=api)


def test_resolve_api_nllb_600m_alias() -> None:
    assert resolve_api("nllb-600m") == "nllb-600m"
    assert resolve_api("nllb-200-distilled-600m") == "nllb-600m"


def test_resolve_api_translategemma_variants() -> None:
    assert resolve_api("translategemma") == "translategemma"
    assert resolve_api("translate-gemma-4b") == "translategemma"
    assert resolve_api("translate-gemma-12b") == "translategemma-12b"


def test_resolve_api_seedx_alias() -> None:
    assert resolve_api("seed-x-ppo-7b") == "seedx"


@pytest.mark.parametrize(
    ("api", "expected_timeout"),
    [
        ("google", 30),
        ("translategemma", 3_600),
        ("translategemma-12b", 3_600),
        ("seedx", 3_600),
    ],
)
def test_processing_settings_use_profile_timeout_by_default(
    api: str,
    expected_timeout: int,
) -> None:
    assert ProcessingSettings(api=api).timeout == expected_timeout
    assert ProcessingSettings(api=api, timeout=45).timeout == 45


def test_processing_settings_pass_worker_python_to_translation_options(tmp_path: Path) -> None:
    worker_python = tmp_path / "среда" / "python.exe"
    settings = ProcessingSettings(api="google", worker_python_path=worker_python)

    options = service._translation_options(settings, "en")

    assert settings.worker_python_path == worker_python.resolve()
    assert options.worker_python_path == worker_python.resolve()


@pytest.mark.parametrize(
    ("response", "message"),
    [
        ("Перевод", "неподдерживаемом формате"),
        (["Один"], "не совпадает с размером пачки"),
        (["Один", 2], "не строку для элемента пачки 2"),
        (["Один", "  "], "пустой перевод для непустого элемента пачки 2"),
    ],
)
def test_common_batch_response_contract_rejects_invalid_result(
    response: object,
    message: str,
) -> None:
    with pytest.raises(TranslationError, match=message):
        service._validate_batch_response(["First", "Second"], response)


def test_common_batch_response_contract_allows_empty_translation_only_for_empty_text() -> None:
    response = ["", "Перевод"]

    assert service._validate_batch_response(["", "Text"], response) is response


def test_process_subtitle_restores_cache_before_creating_translator(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "урок с пробелом.en.srt"
    output = tmp_path / "результат.ru.srt"
    cache_path = tmp_path / "output-cache.json"
    write_srt(source)
    translator = FakeTranslator()
    monkeypatch.setattr(service, "get_translator", lambda *_args: translator)
    settings = ProcessingSettings(source_lang="auto", target_lang="ru", api="google")

    first = process_subtitle(
        source,
        settings,
        logger=LOGGER,
        output_path=output,
        cache_path=cache_path,
    )
    output.unlink()
    monkeypatch.setattr(
        service,
        "get_translator",
        lambda *_args: (_ for _ in ()).throw(AssertionError("переводчик не должен загружаться")),
    )
    second = process_subtitle(
        source,
        settings,
        logger=LOGGER,
        output_path=output,
        cache_path=cache_path,
    )

    assert first.status is ProcessingStatus.TRANSLATED
    assert second.status is ProcessingStatus.CACHED
    assert len(translator.calls) == 1
    assert output.read_bytes().startswith(b"1")
    assert not output.read_bytes().startswith(b"\xef\xbb\xbf")


def test_process_subtitle_does_not_overwrite_existing_output_without_force(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "source.en.srt"
    output = tmp_path / "source.google.ru.srt"
    write_srt(source)
    output.write_text("пользовательский результат", encoding="utf-8", newline="")
    monkeypatch.setattr(
        service,
        "get_translator",
        lambda *_args: (_ for _ in ()).throw(AssertionError("переводчик не должен загружаться")),
    )

    result = process_subtitle(
        source,
        ProcessingSettings(source_lang="en", target_lang="ru", api="google"),
        logger=LOGGER,
        output_path=output,
        cache_path=tmp_path / "cache.json",
    )

    assert result.status is ProcessingStatus.OUTPUT_EXISTS
    assert output.read_text(encoding="utf-8") == "пользовательский результат"


def test_process_subtitle_force_replaces_existing_output(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source.en.srt"
    output = tmp_path / "source.google.ru.srt"
    write_srt(source)
    output.write_text("старый результат", encoding="utf-8", newline="")
    translator = FakeTranslator()
    monkeypatch.setattr(service, "get_translator", lambda *_args: translator)

    result = process_subtitle(
        source,
        ProcessingSettings(source_lang="en", target_lang="ru", api="google", force=True),
        logger=LOGGER,
        output_path=output,
        cache_path=tmp_path / "cache.json",
    )

    assert result.status is ProcessingStatus.TRANSLATED
    assert "Перевод 1" in output.read_text(encoding="utf-8")
    assert len(translator.calls) == 1


def test_process_subtitle_batch_isolates_file_error(tmp_path: Path, monkeypatch) -> None:
    missing = tmp_path / "missing.srt"
    valid = tmp_path / "valid.en.srt"
    write_srt(valid)
    translator = FakeTranslator()
    monkeypatch.setattr(service, "get_translator", lambda *_args: translator)
    completed: list[ProcessingStatus] = []

    results = process_subtitle_batch(
        [missing, valid],
        ProcessingSettings(source_lang="en", target_lang="ru", api="google"),
        logger=LOGGER,
        cache_path=tmp_path / "cache.json",
        result_callback=lambda result, _index, _total: completed.append(result.status),
    )

    assert [result.status for result in results] == [
        ProcessingStatus.ERROR,
        ProcessingStatus.TRANSLATED,
    ]
    assert completed == [ProcessingStatus.ERROR, ProcessingStatus.TRANSLATED]
    assert (tmp_path / "valid.google.ru.srt").exists()
