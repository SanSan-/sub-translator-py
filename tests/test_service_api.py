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
from sub_translate.utils import io_utils

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


@pytest.mark.parametrize("force", [False, True])
@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize(
    "alias", ["source_symlink", "other_symlink", "missing_symlink", "source_hardlink", "cache_hardlink"]
)
def test_process_subtitle_rejects_output_alias_before_translation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    force: bool,
    explicit: bool,
    alias: str,
) -> None:
    source = tmp_path / "source.en.srt"
    other = tmp_path / "other.srt"
    missing = tmp_path / "missing.srt"
    output = tmp_path / "source.google.ru.srt"
    cache_path = tmp_path / "cache.json"
    write_srt(source)
    write_srt(other, "Другой файл")
    cache_path.write_text('{"keep": true}', encoding="utf-8")
    original = {path: path.read_bytes() for path in (source, other, cache_path)}
    targets = {"source": source, "other": other, "missing": missing, "cache": cache_path}
    target_name, link_type = alias.split("_")
    try:
        if link_type == "hardlink":
            output.hardlink_to(targets[target_name])
        else:
            output.symlink_to(targets[target_name])
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"Ссылки недоступны: {exc}")
    translator = FakeTranslator()
    monkeypatch.setattr(service, "get_translator", lambda *_args: translator)
    monkeypatch.setattr(io_utils, "CACHE_DIR", tmp_path / "locks")

    settings = ProcessingSettings(source_lang="en", target_lang="ru", api="google", force=force)
    with pytest.raises(ValueError):
        process_subtitle(
            source,
            settings,
            logger=LOGGER,
            output_path=output if explicit else None,
            cache_path=cache_path,
        )

    assert {path: path.read_bytes() for path in original} == original
    assert not missing.exists()
    assert translator.calls == []


def test_process_subtitle_rejects_link_created_during_translation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.en.srt"
    output = tmp_path / "source.google.ru.srt"
    cache_path = tmp_path / "cache.json"
    write_srt(source)
    original = source.read_bytes()
    try:
        output.symlink_to(source)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"Символические ссылки недоступны: {exc}")
    output.unlink()
    translator = FakeTranslator()
    translate_batch = translator.translate_batch

    def substitute_output(texts, options):
        output.symlink_to(source)
        return translate_batch(texts, options)

    monkeypatch.setattr(translator, "translate_batch", substitute_output)
    monkeypatch.setattr(service, "get_translator", lambda *_args: translator)
    monkeypatch.setattr(io_utils, "CACHE_DIR", tmp_path / "locks")

    settings = ProcessingSettings(api="google", force=True)
    with pytest.raises(ValueError, match="не должен быть ссылкой"):
        process_subtitle(
            source,
            settings,
            logger=LOGGER,
            cache_path=cache_path,
        )

    assert source.read_bytes() == original
    assert not cache_path.exists()
    assert len(translator.calls) == 1


@pytest.mark.parametrize("force", [False, True])
def test_process_subtitle_ignores_unreadable_cache_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    force: bool,
) -> None:
    source = tmp_path / "source.en.srt"
    cache_path = tmp_path / "cache.json"
    cache_path.mkdir()
    write_srt(source)
    original = source.read_bytes()
    translator = FakeTranslator()
    monkeypatch.setattr(service, "get_translator", lambda *_args: translator)
    monkeypatch.setattr(io_utils, "CACHE_DIR", tmp_path / "locks")

    result = process_subtitle(
        source,
        ProcessingSettings(api="google", force=force),
        logger=LOGGER,
        cache_path=cache_path,
    )

    assert result.status is ProcessingStatus.TRANSLATED
    assert result.output_path.read_text(encoding="utf-8").endswith("Перевод 1\n")
    assert source.read_bytes() == original
    assert cache_path.is_dir()
    assert list(cache_path.iterdir()) == []
