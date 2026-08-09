import json
import logging
from pathlib import Path

import pytest

from sub_translate import service
from sub_translate.enums import FileFormat
from sub_translate.service import ProcessingSettings, ProcessingStatus
from sub_translate.translators.registry import get_translator_metadata
from sub_translate.utils import huggingface as huggingface_utils
from sub_translate.utils.huggingface import MODEL_MARKER_FILENAME, model_content_fingerprint
from sub_translate.utils.subtitle_cache import (
    CacheRestoreStatus,
    build_cache_fingerprint,
    restore_output_cache,
    store_output_cache,
)

LOGGER = logging.getLogger("test.service-model-cache")


def _write_srt(path: Path, text: str) -> None:
    path.write_text(
        f"1\n00:00:00,000 --> 00:00:01,000\n{text}\n",
        encoding="utf-8",
        newline="",
    )


def _write_complete_nllb_model(path: Path) -> Path:
    metadata = get_translator_metadata("nllb-600m")
    path.mkdir(parents=True, exist_ok=True)
    for relative_path in metadata.required_files:
        target = path / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"{}")
    weight_path = path / metadata.required_file_groups[0][0]
    weight_path.write_bytes(b"initial-weights")
    return weight_path


def _nllb_settings(model_path: Path, *, auto_download: bool = False) -> ProcessingSettings:
    return ProcessingSettings(
        source_lang="en",
        target_lang="ru",
        api="nllb-600m",
        model_path=model_path,
        auto_download_model=auto_download,
    )


def _cache_identity(source: Path, settings: ProcessingSettings):
    return service.build_output_cache_identity(
        source,
        FileFormat.SRT,
        settings,
        source_lang="en",
    )


def test_translategemma_profiles_have_distinct_cache_identity(tmp_path: Path) -> None:
    source = tmp_path / "source.en.srt"
    _write_srt(source, "Hello")
    profile_4b = _cache_identity(
        source,
        ProcessingSettings(api="translategemma", model_path=tmp_path / "missing-4b"),
    )
    profile_12b = _cache_identity(
        source,
        ProcessingSettings(api="translategemma-12b", model_path=tmp_path / "missing-12b"),
    )

    assert profile_4b.translator_id == "translategemma"
    assert profile_4b.model_id == "google/translategemma-4b-it"
    assert profile_12b.translator_id == "translategemma-12b"
    assert profile_12b.model_id == "google/translategemma-12b-it"
    assert build_cache_fingerprint(profile_4b) != build_cache_fingerprint(profile_12b)


def test_manual_model_change_invalidates_output_cache(tmp_path: Path) -> None:
    source = tmp_path / "source.en.srt"
    output = tmp_path / "translated.srt"
    restored = tmp_path / "restored.srt"
    cache_path = tmp_path / "output-cache.json"
    model_path = tmp_path / "manual-model"
    _write_srt(source, "Hello")
    _write_srt(output, "Привет")
    weight_path = _write_complete_nllb_model(model_path)
    settings = _nllb_settings(model_path)

    initial_identity = _cache_identity(source, settings)
    assert initial_identity.model_content_fingerprint is not None
    assert store_output_cache(initial_identity, output, LOGGER, cache_path=cache_path)

    weight_path.write_bytes(b"changed-weights")
    changed_identity = _cache_identity(source, settings)

    assert not changed_identity.local_model_unavailable
    assert changed_identity.model_content_fingerprint is not None
    assert changed_identity.model_content_fingerprint != initial_identity.model_content_fingerprint
    assert build_cache_fingerprint(changed_identity) != build_cache_fingerprint(initial_identity)
    assert restore_output_cache(changed_identity, restored, LOGGER, cache_path=cache_path) is CacheRestoreStatus.MISS


@pytest.mark.parametrize("availability", ["missing", "incomplete"])
def test_manual_model_cache_restores_without_model_network_secret_or_adapter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    availability: str,
) -> None:
    source = tmp_path / "source.en.srt"
    output = tmp_path / "translated.srt"
    cache_path = tmp_path / "output-cache.json"
    model_path = tmp_path / "manual-model"
    _write_srt(source, "Hello")
    _write_srt(output, "Привет")
    weight_path = _write_complete_nllb_model(model_path)
    settings = _nllb_settings(model_path, auto_download=True)
    initial_identity = _cache_identity(source, settings)
    assert initial_identity.model_content_fingerprint is not None
    assert store_output_cache(initial_identity, output, LOGGER, cache_path=cache_path)
    output.unlink()
    if availability == "missing":
        model_path.rename(tmp_path / "manual-model-away")
    else:
        weight_path.unlink()

    def unexpected_call(*_args, **_kwargs):
        raise AssertionError("кеш-попадание не должно читать секрет, обращаться к сети или создавать адаптер")

    monkeypatch.setattr(huggingface_utils, "_download_model_snapshot", unexpected_call)
    monkeypatch.setattr(huggingface_utils, "_huggingface_token_from_environment", unexpected_call)
    monkeypatch.setattr(service, "get_translator", unexpected_call)
    monkeypatch.setattr(service, "translate_subtitles", unexpected_call)

    unavailable_identity = _cache_identity(source, settings)
    assert unavailable_identity.local_model_unavailable
    assert unavailable_identity.model_content_fingerprint is None
    result = service.process_subtitle(
        source,
        settings,
        logger=LOGGER,
        output_path=output,
        cache_path=cache_path,
    )

    assert result.status is ProcessingStatus.CACHED
    assert output.is_file()


def test_verified_revision_is_stable_and_cache_hit_has_no_network_or_secret_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.en.srt"
    output = tmp_path / "translated.srt"
    cache_path = tmp_path / "output-cache.json"
    model_path = tmp_path / "verified-model"
    _write_srt(source, "Hello")
    _write_srt(output, "Привет")
    weight_path = _write_complete_nllb_model(model_path)
    metadata = get_translator_metadata("nllb-600m")
    fingerprint = model_content_fingerprint(model_path)
    assert metadata.model_id is not None
    assert metadata.model_revision is not None
    assert fingerprint is not None
    (model_path / MODEL_MARKER_FILENAME).write_text(
        json.dumps(
            {
                "model_id": metadata.model_id,
                "revision": metadata.model_revision,
                "content_fingerprint": fingerprint,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        encoding="utf-8",
        newline="",
    )
    settings = _nllb_settings(model_path, auto_download=True)

    def unexpected_call(*_args, **_kwargs):
        raise AssertionError("при совпадении кеша сеть, секрет и переводчик не нужны")

    monkeypatch.setattr(huggingface_utils, "_download_model_snapshot", unexpected_call)
    monkeypatch.setattr(huggingface_utils, "_huggingface_token_from_environment", unexpected_call)
    initial_identity = _cache_identity(source, settings)
    repeated_identity = _cache_identity(source, settings)

    assert initial_identity.model_content_fingerprint is None
    assert build_cache_fingerprint(repeated_identity) == build_cache_fingerprint(initial_identity)
    assert store_output_cache(initial_identity, output, LOGGER, cache_path=cache_path)
    output.unlink()
    monkeypatch.setattr(service, "translate_subtitles", unexpected_call)

    result = service.process_subtitle(
        source,
        settings,
        logger=LOGGER,
        output_path=output,
        cache_path=cache_path,
    )

    assert result.status is ProcessingStatus.CACHED
    assert output.is_file()

    output.unlink()
    weight_path.write_bytes(b"changed-after-verification")
    changed_identity = _cache_identity(source, settings)

    assert changed_identity.local_model_unavailable is True
    assert changed_identity.model_content_fingerprint is None
    assert restore_output_cache(changed_identity, output, LOGGER, cache_path=cache_path) is CacheRestoreStatus.RESTORED
    assert output.is_file()


def test_process_does_not_cache_result_when_manual_model_appears_during_translation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.en.srt"
    output = tmp_path / "translated.srt"
    model_path = tmp_path / "downloaded-model"
    _write_srt(source, "Hello")
    settings = _nllb_settings(model_path, auto_download=True)
    stored_identities = []

    def fake_translate(
        _input_path: Path,
        output_path: Path,
        *_args,
        **_kwargs,
    ) -> None:
        _write_complete_nllb_model(model_path)
        _write_srt(output_path, "Привет")

    def capture_store(identity, *_args, **_kwargs) -> bool:
        stored_identities.append(identity)
        return True

    def unexpected_call(*_args, **_kwargs):
        raise AssertionError("проверка кеша не должна читать токен или запускать загрузку")

    monkeypatch.setattr(huggingface_utils, "_download_model_snapshot", unexpected_call)
    monkeypatch.setattr(huggingface_utils, "_huggingface_token_from_environment", unexpected_call)
    monkeypatch.setattr(service, "translate_subtitles", fake_translate)
    monkeypatch.setattr(service, "store_output_cache", capture_store)

    result = service.process_subtitle(
        source,
        settings,
        logger=LOGGER,
        output_path=output,
        cache_path=tmp_path / "output-cache.json",
    )

    assert result.status is ProcessingStatus.TRANSLATED
    assert stored_identities == []
    assert model_content_fingerprint(model_path) is not None
