import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from sub_translate.enums import FileFormat
from sub_translate.utils import io_utils, subtitle_cache
from sub_translate.utils.subtitle_cache import (
    CachePolicy,
    CacheRestoreStatus,
    OutputCacheIdentity,
    build_cache_fingerprint,
    prompt_signature,
    restore_output_cache,
    store_output_cache,
    validate_cached_lines,
)

LOGGER = logging.getLogger("test.subtitle-cache")

VALID_SRT = [
    "1",
    "00:00:00,000 --> 00:00:01,000",
    "Привет, мир!",
    "",
]
VALID_VTT = [
    "WEBVTT",
    "",
    "cue-1",
    "00:00:00.000 --> 00:00:01.000",
    "Привет, мир!",
    "",
]
VALID_ASS = [
    "[Script Info]",
    "Title: Проверка",
    "",
    "[Events]",
    "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    "Dialogue: 0,0:00:00.00,0:00:01.00,Default,,0,0,0,,Привет, мир!",
]


def _write_utf8(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\r\n".join(lines), encoding="utf-8", newline="")


def _identity(
    source: Path,
    *,
    file_format: FileFormat = FileFormat.SRT,
    settings: dict | None = None,
    model_id: str | None = "example/model",
    model_revision: str | None = "revision-1",
    model_content_fingerprint: str | None = None,
    prompts: dict[str, str] | None = None,
    algorithm_version: str = subtitle_cache.OUTPUT_CACHE_ALGORITHM_VERSION,
    source_language: str = "English",
    target_language: str = "Русский",
    translator_id: str = "Local-Translator",
    local_model_unavailable: bool = False,
) -> OutputCacheIdentity:
    return OutputCacheIdentity(
        source_path=source,
        file_format=file_format,
        source_language=source_language,
        target_language=target_language,
        translator_id=translator_id,
        model_id=model_id,
        model_revision=model_revision,
        model_content_fingerprint=model_content_fingerprint,
        settings=settings or {"batch_size": 8, "quantization": "int8"},
        prompt_signatures=prompts or {"system": prompt_signature("Переведи субтитры")},
        algorithm_version=algorithm_version,
        local_model_unavailable=local_model_unavailable,
    )


@pytest.fixture(autouse=True)
def _isolate_atomic_lock_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(io_utils, "CACHE_DIR", tmp_path / "служебные-блокировки")


def test_same_basename_with_different_content_has_distinct_fingerprint(tmp_path: Path) -> None:
    first = tmp_path / "один" / "episode.srt"
    second = tmp_path / "два" / "episode.srt"
    _write_utf8(first, VALID_SRT)
    _write_utf8(second, [*VALID_SRT[:2], "Другой текст", ""])

    assert build_cache_fingerprint(_identity(first)) != build_cache_fingerprint(_identity(second))


@pytest.mark.parametrize(
    ("override", "value"),
    [
        ("settings", {"batch_size": 9}),
        ("model_id", "example/other-model"),
        ("model_revision", "revision-2"),
        ("model_content_fingerprint", "a" * 64),
        ("prompts", {"system": prompt_signature("Другая подсказка")}),
        ("algorithm_version", "subtitle-output-v4"),
        ("source_language", "de"),
        ("target_language", "ja"),
        ("translator_id", "other-translator"),
        ("file_format", FileFormat.VTT),
    ],
)
def test_result_contract_changes_fingerprint(tmp_path: Path, override: str, value: object) -> None:
    source = tmp_path / "source.srt"
    _write_utf8(source, VALID_SRT)
    baseline = build_cache_fingerprint(_identity(source))

    assert build_cache_fingerprint(_identity(source, **{override: value})) != baseline


def test_canonical_translator_id_is_case_and_space_independent(tmp_path: Path) -> None:
    source = tmp_path / "source.srt"
    _write_utf8(source, VALID_SRT)

    assert build_cache_fingerprint(_identity(source, translator_id=" LOCAL-TRANSLATOR ")) == (
        build_cache_fingerprint(_identity(source, translator_id="local-translator"))
    )


def test_identity_rejects_source_changed_after_capture(tmp_path: Path) -> None:
    source = tmp_path / "source.srt"
    _write_utf8(source, VALID_SRT)
    identity = _identity(source)
    _write_utf8(source, [*VALID_SRT[:2], "Изменённый источник", ""])

    with pytest.raises(ValueError, match="Исходный файл изменился"):
        build_cache_fingerprint(identity)


def test_round_trip_restores_unicode_and_writes_cache_without_bom(tmp_path: Path) -> None:
    source = tmp_path / "исходник.srt"
    output = tmp_path / "перевод.srt"
    restored = tmp_path / "восстановлено.srt"
    cache_path = tmp_path / "кеш" / "готовые.json"
    _write_utf8(source, [*VALID_SRT[:2], "Hello, world!", ""])
    _write_utf8(output, VALID_SRT)
    identity = _identity(source)

    assert store_output_cache(identity, output, LOGGER, cache_path=cache_path)
    assert not cache_path.read_bytes().startswith(b"\xef\xbb\xbf")
    status = restore_output_cache(identity, restored, LOGGER, cache_path=cache_path)

    assert status is CacheRestoreStatus.RESTORED
    assert restored.read_bytes() == "\r\n".join(VALID_SRT).encode("utf-8")
    assert not restored.read_bytes().startswith(b"\xef\xbb\xbf")


def test_round_trip_preserves_lf_line_endings(tmp_path: Path) -> None:
    source = tmp_path / "source.srt"
    output = tmp_path / "translation.srt"
    restored = tmp_path / "restored.srt"
    cache_path = tmp_path / "cache.json"
    _write_utf8(source, VALID_SRT)
    output.write_text("\n".join(VALID_SRT), encoding="utf-8", newline="")
    identity = _identity(source)

    assert store_output_cache(identity, output, LOGGER, cache_path=cache_path)
    assert restore_output_cache(identity, restored, LOGGER, cache_path=cache_path) is CacheRestoreStatus.RESTORED
    assert restored.read_bytes() == "\n".join(VALID_SRT).encode("utf-8")


@pytest.mark.parametrize(
    ("file_format", "lines"),
    [
        (FileFormat.SRT, VALID_SRT),
        (FileFormat.VTT, VALID_VTT),
        (FileFormat.ASS, VALID_ASS),
    ],
)
def test_validator_accepts_complete_supported_documents(
    file_format: FileFormat,
    lines: list[str],
) -> None:
    assert validate_cached_lines(lines, file_format)


@pytest.mark.parametrize(
    ("file_format", "lines"),
    [
        (FileFormat.SRT, ["1", "без тайминга", "Текст"]),
        (FileFormat.SRT, ["1", "00:00:00,000 --> 00:00:01,000", "Строка\nв строке"]),
        (FileFormat.VTT, ["WEBVTT", "", "без тайминга"]),
        (FileFormat.ASS, ["[Events]", "Format: Layer, Start, End", "Dialogue: повреждено"]),
    ],
)
def test_validator_rejects_damaged_documents(file_format: FileFormat, lines: list[str]) -> None:
    assert not validate_cached_lines(lines, file_format)


@pytest.mark.parametrize(
    "invalid_payload",
    [
        b"{invalid json",
        b'{"old-key": {"format": "srt", "lines": []}}',
        b'{"schema_version": 1, "cache_type": "translated-subtitle-output", "entries": {}}',
        b'\xef\xbb\xbf{"schema_version": 2, "cache_type": "translated-subtitle-output", "entries": {}}',
    ],
)
def test_corrupt_legacy_or_incompatible_cache_is_not_overwritten(
    tmp_path: Path,
    invalid_payload: bytes,
) -> None:
    source = tmp_path / "source.srt"
    output = tmp_path / "output.srt"
    restored = tmp_path / "restored.srt"
    cache_path = tmp_path / "cache.json"
    _write_utf8(source, VALID_SRT)
    _write_utf8(output, VALID_SRT)
    cache_path.write_bytes(invalid_payload)
    identity = _identity(source)

    assert not store_output_cache(identity, output, LOGGER, cache_path=cache_path)
    assert restore_output_cache(identity, restored, LOGGER, cache_path=cache_path) is CacheRestoreStatus.CACHE_REJECTED
    assert cache_path.read_bytes() == invalid_payload
    assert not restored.exists()


def test_tampered_cached_lines_are_rejected_without_rewrite(tmp_path: Path) -> None:
    source = tmp_path / "source.srt"
    output = tmp_path / "output.srt"
    restored = tmp_path / "restored.srt"
    cache_path = tmp_path / "cache.json"
    _write_utf8(source, VALID_SRT)
    _write_utf8(output, VALID_SRT)
    identity = _identity(source)
    assert store_output_cache(identity, output, LOGGER, cache_path=cache_path)
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    entry = next(iter(payload["entries"].values()))
    entry["lines"][2] = "Подменённый текст"
    cache_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    tampered = cache_path.read_bytes()

    assert restore_output_cache(identity, restored, LOGGER, cache_path=cache_path) is CacheRestoreStatus.CACHE_REJECTED
    assert cache_path.read_bytes() == tampered


@pytest.mark.parametrize(
    ("field_path", "invalid_value"),
    [
        (("schema_version",), "2"),
        (("entries",), []),
        (("entry", "created_at"), True),
        (("entry", "line_separator"), 1),
        (("entry", "lines"), [1]),
        (("entry", "signature", "settings"), []),
        (("entry", "signature", "prompt_signatures"), {"system": "not-a-sha"}),
    ],
)
def test_json_schema_rejects_wrong_field_types_without_rewrite(
    tmp_path: Path,
    field_path: tuple[str, ...],
    invalid_value: object,
) -> None:
    source = tmp_path / "source.srt"
    output = tmp_path / "output.srt"
    restored = tmp_path / "restored.srt"
    cache_path = tmp_path / "cache.json"
    _write_utf8(source, VALID_SRT)
    _write_utf8(output, VALID_SRT)
    identity = _identity(source)
    assert store_output_cache(identity, output, LOGGER, cache_path=cache_path)
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    target = payload
    path = field_path
    if path[0] == "entry":
        target = next(iter(payload["entries"].values()))
        path = path[1:]
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = invalid_value
    cache_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8", newline="")
    damaged = cache_path.read_bytes()

    assert restore_output_cache(identity, restored, LOGGER, cache_path=cache_path) is CacheRestoreStatus.CACHE_REJECTED
    assert cache_path.read_bytes() == damaged


def test_existing_output_is_preserved_without_force_and_replaced_with_force(tmp_path: Path) -> None:
    source = tmp_path / "source.srt"
    cached_output = tmp_path / "cached.srt"
    destination = tmp_path / "destination.srt"
    cache_path = tmp_path / "cache.json"
    new_lines = [*VALID_SRT[:2], "Новый текст", ""]
    old_lines = [*VALID_SRT[:2], "Пользовательский текст", ""]
    _write_utf8(source, VALID_SRT)
    _write_utf8(cached_output, new_lines)
    _write_utf8(destination, old_lines)
    identity = _identity(source)
    assert store_output_cache(identity, cached_output, LOGGER, cache_path=cache_path)

    status = restore_output_cache(identity, destination, LOGGER, cache_path=cache_path)
    assert status is CacheRestoreStatus.OUTPUT_EXISTS
    assert destination.read_bytes() == "\r\n".join(old_lines).encode("utf-8")

    status = restore_output_cache(identity, destination, LOGGER, force=True, cache_path=cache_path)
    assert status is CacheRestoreStatus.RESTORED
    assert destination.read_bytes() == "\r\n".join(new_lines).encode("utf-8")


def test_cache_path_cannot_replace_source_or_output(tmp_path: Path) -> None:
    source = tmp_path / "source.srt"
    output = tmp_path / "output.srt"
    _write_utf8(source, VALID_SRT)
    _write_utf8(output, VALID_SRT)
    identity = _identity(source)

    with pytest.raises(ValueError, match="не должны совпадать"):
        store_output_cache(identity, output, LOGGER, cache_path=output)
    with pytest.raises(ValueError, match="не должны совпадать"):
        store_output_cache(identity, output, LOGGER, cache_path=source)
    with pytest.raises(ValueError, match="не должны совпадать"):
        restore_output_cache(identity, output, LOGGER, force=True, cache_path=output)


def test_output_with_bom_is_not_cached(tmp_path: Path) -> None:
    source = tmp_path / "source.srt"
    output = tmp_path / "output.srt"
    cache_path = tmp_path / "cache.json"
    _write_utf8(source, VALID_SRT)
    output.write_bytes(b"\xef\xbb\xbf" + "\r\n".join(VALID_SRT).encode("utf-8"))

    assert not store_output_cache(_identity(source), output, LOGGER, cache_path=cache_path)
    assert not cache_path.exists()


def test_missing_cache_is_a_clean_miss(tmp_path: Path) -> None:
    source = tmp_path / "source.srt"
    destination = tmp_path / "destination.srt"
    cache_path = tmp_path / "missing" / "cache.json"
    _write_utf8(source, VALID_SRT)

    assert (
        restore_output_cache(_identity(source), destination, LOGGER, cache_path=cache_path) is CacheRestoreStatus.MISS
    )
    assert not cache_path.exists()
    assert not destination.exists()


def test_concurrent_writers_do_not_lose_entries(tmp_path: Path) -> None:
    cache_path = tmp_path / "cache.json"
    policy = CachePolicy(ttl_seconds=3600, max_entries=64)
    items: list[tuple[OutputCacheIdentity, Path]] = []
    for index in range(24):
        source = tmp_path / "sources" / f"{index}.srt"
        output = tmp_path / "outputs" / f"{index}.srt"
        _write_utf8(source, [*VALID_SRT[:2], f"Source {index}", ""])
        _write_utf8(output, [*VALID_SRT[:2], f"Перевод {index}", ""])
        items.append((_identity(source), output))

    def _store(item: tuple[OutputCacheIdentity, Path]) -> bool:
        return store_output_cache(*item, LOGGER, cache_path=cache_path, policy=policy)

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(_store, items))

    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    assert all(results)
    assert len(payload["entries"]) == len(items)


def test_max_entries_prunes_only_old_service_records(tmp_path: Path, monkeypatch) -> None:
    cache_path = tmp_path / "cache.json"
    user_file = tmp_path / "user-result.srt"
    _write_utf8(user_file, VALID_SRT)
    clock = iter([100.0, 101.0, 102.0])
    monkeypatch.setattr(subtitle_cache.time, "time", lambda: next(clock))
    policy = CachePolicy(ttl_seconds=3600, max_entries=2)
    fingerprints: list[str] = []
    for index in range(3):
        source = tmp_path / "sources" / f"{index}.srt"
        output = tmp_path / "outputs" / f"{index}.srt"
        _write_utf8(source, [*VALID_SRT[:2], f"Source {index}", ""])
        _write_utf8(output, [*VALID_SRT[:2], f"Перевод {index}", ""])
        identity = _identity(source)
        fingerprints.append(build_cache_fingerprint(identity))
        assert store_output_cache(identity, output, LOGGER, cache_path=cache_path, policy=policy)

    entries = json.loads(cache_path.read_text(encoding="utf-8"))["entries"]
    assert len(entries) == 2
    assert fingerprints[0] not in entries
    assert user_file.exists()


def test_expired_entry_is_ignored_and_pruned(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source.srt"
    output = tmp_path / "output.srt"
    restored = tmp_path / "restored.srt"
    cache_path = tmp_path / "cache.json"
    _write_utf8(source, VALID_SRT)
    _write_utf8(output, VALID_SRT)
    identity = _identity(source)
    policy = CachePolicy(ttl_seconds=10, max_entries=10)
    monkeypatch.setattr(subtitle_cache.time, "time", lambda: 100.0)
    assert store_output_cache(identity, output, LOGGER, cache_path=cache_path, policy=policy)
    monkeypatch.setattr(subtitle_cache.time, "time", lambda: 111.0)

    assert (
        restore_output_cache(identity, restored, LOGGER, cache_path=cache_path, policy=policy)
        is CacheRestoreStatus.MISS
    )
    assert json.loads(cache_path.read_text(encoding="utf-8"))["entries"] == {}
    assert output.exists()


def test_secret_like_settings_are_rejected(tmp_path: Path) -> None:
    source = tmp_path / "source.srt"
    _write_utf8(source, VALID_SRT)

    with pytest.raises(ValueError, match="нельзя включать в кеш"):
        _identity(source, settings={"openai_api_key": "secret"})
    with pytest.raises(ValueError, match="нельзя включать в кеш"):
        _identity(source, settings={"provider": {"access_token": "secret"}})


def test_non_secret_token_count_setting_is_allowed_and_captured(tmp_path: Path) -> None:
    source = tmp_path / "source.srt"
    _write_utf8(source, VALID_SRT)
    identity = _identity(source, settings={"max_tokens": 512, "nested": {"temperature": 0}})
    fingerprint = build_cache_fingerprint(identity)
    identity.settings["nested"]["temperature"] = 1

    assert build_cache_fingerprint(identity) == fingerprint


@pytest.mark.parametrize(
    "kwargs",
    [
        {"ttl_seconds": 0},
        {"ttl_seconds": "10"},
        {"max_entries": 0},
        {"lock_timeout_seconds": float("inf")},
    ],
)
def test_cache_policy_rejects_invalid_bounds(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        CachePolicy(**kwargs)


def test_unavailable_model_flag_does_not_change_signature(tmp_path: Path) -> None:
    source = tmp_path / "source.srt"
    _write_utf8(source, VALID_SRT)

    available = _identity(source, model_content_fingerprint=None)
    unavailable = _identity(source, model_content_fingerprint=None, local_model_unavailable=True)

    assert build_cache_fingerprint(unavailable) == build_cache_fingerprint(available)


def test_unavailable_model_restores_only_unique_fingerprint_variant(tmp_path: Path) -> None:
    source = tmp_path / "source.srt"
    output = tmp_path / "output.srt"
    restored = tmp_path / "restored.srt"
    cache_path = tmp_path / "cache.json"
    _write_utf8(source, VALID_SRT)
    _write_utf8(output, VALID_SRT)
    stored = _identity(source, model_content_fingerprint="a" * 64)
    unavailable = _identity(source, model_content_fingerprint=None, local_model_unavailable=True)
    assert store_output_cache(stored, output, LOGGER, cache_path=cache_path)

    assert restore_output_cache(unavailable, restored, LOGGER, cache_path=cache_path) is CacheRestoreStatus.RESTORED
    assert restored.read_bytes() == output.read_bytes()


def test_unavailable_model_rejects_ambiguous_fingerprint_variants(tmp_path: Path) -> None:
    source = tmp_path / "source.srt"
    first_output = tmp_path / "first.srt"
    second_output = tmp_path / "second.srt"
    restored = tmp_path / "restored.srt"
    cache_path = tmp_path / "cache.json"
    _write_utf8(source, VALID_SRT)
    _write_utf8(first_output, VALID_SRT)
    _write_utf8(second_output, [*VALID_SRT[:2], "Другой перевод", ""])
    assert store_output_cache(
        _identity(source, model_content_fingerprint="a" * 64),
        first_output,
        LOGGER,
        cache_path=cache_path,
    )
    assert store_output_cache(
        _identity(source, model_content_fingerprint="b" * 64),
        second_output,
        LOGGER,
        cache_path=cache_path,
    )
    unavailable = _identity(source, model_content_fingerprint=None, local_model_unavailable=True)

    assert restore_output_cache(unavailable, restored, LOGGER, cache_path=cache_path) is CacheRestoreStatus.MISS
    assert not restored.exists()


@pytest.mark.parametrize(
    ("override", "value"),
    [
        ("settings", {"batch_size": 9}),
        ("model_revision", "revision-2"),
        ("prompts", {"system": prompt_signature("Другая подсказка")}),
    ],
)
def test_unavailable_model_fallback_requires_every_other_signature_field(
    tmp_path: Path,
    override: str,
    value: object,
) -> None:
    source = tmp_path / "source.srt"
    output = tmp_path / "output.srt"
    restored = tmp_path / "restored.srt"
    cache_path = tmp_path / "cache.json"
    _write_utf8(source, VALID_SRT)
    _write_utf8(output, VALID_SRT)
    assert store_output_cache(
        _identity(source, model_content_fingerprint="a" * 64),
        output,
        LOGGER,
        cache_path=cache_path,
    )
    unavailable = _identity(
        source,
        model_content_fingerprint=None,
        local_model_unavailable=True,
        **{override: value},
    )

    assert restore_output_cache(unavailable, restored, LOGGER, cache_path=cache_path) is CacheRestoreStatus.MISS
    assert not restored.exists()


def test_unavailable_model_identity_cannot_be_stored(tmp_path: Path) -> None:
    source = tmp_path / "source.srt"
    output = tmp_path / "output.srt"
    cache_path = tmp_path / "cache.json"
    _write_utf8(source, VALID_SRT)
    _write_utf8(output, VALID_SRT)

    assert not store_output_cache(
        _identity(source, model_content_fingerprint=None, local_model_unavailable=True),
        output,
        LOGGER,
        cache_path=cache_path,
    )
    assert not cache_path.exists()
