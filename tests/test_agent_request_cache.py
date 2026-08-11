from __future__ import annotations

import importlib
import json
import logging
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from sub_translate.models import TranslationOptions
from sub_translate.translators import agent
from sub_translate.utils import io_utils


def _cache_entry(translation: str, created_at: float, last_used_at: float) -> agent._RequestCacheEntry:
    return agent._RequestCacheEntry(
        translation=translation,
        created_at=created_at,
        last_used_at=last_used_at,
    )


@pytest.fixture
def isolated_request_cache(tmp_path, monkeypatch):
    cache_path = tmp_path / "agent-cache.json"
    policy = agent.RequestCachePolicy(
        ttl_seconds=100,
        max_entries=10,
        max_bytes=64 * 1024,
        lock_timeout_seconds=5,
    )
    monkeypatch.setattr(agent, "REQUEST_CACHE_FILE", cache_path)
    monkeypatch.setattr(agent, "REQUEST_CACHE_POLICY", policy)
    monkeypatch.setattr(agent, "_request_cache", {})
    monkeypatch.setattr(agent, "_request_cache_loaded", False)
    monkeypatch.setattr(agent, "_cache_dirty", False)
    monkeypatch.setattr(agent, "_client", None)
    monkeypatch.setattr(agent, "time", SimpleNamespace(time=lambda: 1_000.0))
    return cache_path, logging.getLogger(f"agent-cache-test-{tmp_path.name}")


def test_request_cache_rejects_expired_entry(isolated_request_cache) -> None:
    cache_path, logger = isolated_request_cache
    expired_key = "a" * 64
    fresh_key = "b" * 64
    cache_path.write_text(
        json.dumps(
            agent._request_cache_payload(
                {
                    expired_key: _cache_entry("Старый перевод", 800.0, 899.0),
                    fresh_key: _cache_entry("Свежий перевод", 900.0, 950.0),
                }
            ),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    agent._load_request_cache(logger)

    assert expired_key not in agent._request_cache
    assert agent._request_cache[fresh_key].translation == "Свежий перевод"
    persisted = json.loads(cache_path.read_text(encoding="utf-8"))
    assert set(persisted["entries"]) == {fresh_key}


def test_request_cache_prunes_oldest_entries_deterministically(
    isolated_request_cache,
    monkeypatch,
) -> None:
    cache_path, logger = isolated_request_cache
    monkeypatch.setattr(
        agent,
        "REQUEST_CACHE_POLICY",
        agent.RequestCachePolicy(ttl_seconds=1_000, max_entries=2, max_bytes=64 * 1024),
    )
    oldest_key = "a" * 64
    middle_key = "b" * 64
    newest_key = "c" * 64
    agent._request_cache = {
        oldest_key: _cache_entry("Первый", 100.0, 200.0),
        newest_key: _cache_entry("Третий", 300.0, 400.0),
        middle_key: _cache_entry("Второй", 200.0, 300.0),
    }
    agent._request_cache_loaded = True
    agent._cache_dirty = True

    agent._save_request_cache(logger)

    persisted = json.loads(cache_path.read_text(encoding="utf-8"))
    assert set(persisted["entries"]) == {middle_key, newest_key}


def test_request_cache_prunes_to_byte_limit(isolated_request_cache, monkeypatch) -> None:
    cache_path, logger = isolated_request_cache
    oldest_key = "a" * 64
    middle_key = "b" * 64
    newest_key = "c" * 64
    newest = _cache_entry("Новый перевод " * 20, 300.0, 400.0)
    one_entry_limit = agent._request_cache_payload_size({newest_key: newest})
    monkeypatch.setattr(
        agent,
        "REQUEST_CACHE_POLICY",
        agent.RequestCachePolicy(
            ttl_seconds=1_000,
            max_entries=10,
            max_bytes=one_entry_limit,
        ),
    )
    agent._request_cache = {
        oldest_key: _cache_entry("Старый перевод " * 20, 100.0, 200.0),
        middle_key: _cache_entry("Средний перевод " * 20, 200.0, 300.0),
        newest_key: newest,
    }
    agent._request_cache_loaded = True
    agent._cache_dirty = True

    agent._save_request_cache(logger)

    persisted = json.loads(cache_path.read_text(encoding="utf-8"))
    assert set(persisted["entries"]) == {newest_key}
    assert cache_path.stat().st_size <= one_entry_limit


def test_request_cache_round_trip_preserves_unicode_without_bom(isolated_request_cache) -> None:
    cache_path, logger = isolated_request_cache
    cache_key = "d" * 64
    translation = "Привет, мир! 👋 日本語"
    agent._request_cache_loaded = True

    agent._store_cached_translation(cache_key, translation)
    agent._save_request_cache(logger)

    raw = cache_path.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf")
    assert translation.encode("utf-8") in raw
    agent._request_cache = {}
    agent._request_cache_loaded = False
    agent._load_request_cache(logger)
    assert agent._request_cache[cache_key].translation == translation


def test_request_cache_save_merges_entry_written_by_another_process(isolated_request_cache) -> None:
    cache_path, logger = isolated_request_cache
    persisted_key = "2" * 64
    current_key = "3" * 64
    cache_path.write_text(
        json.dumps(
            agent._request_cache_payload({persisted_key: _cache_entry("Сохранённый перевод", 800.0, 950.0)}),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    agent._request_cache = {current_key: _cache_entry("Новый перевод", 900.0, 975.0)}
    agent._request_cache_loaded = True
    agent._cache_dirty = True

    agent._save_request_cache(logger)

    persisted = json.loads(cache_path.read_text(encoding="utf-8"))
    assert set(persisted["entries"]) == {persisted_key, current_key}


@pytest.mark.parametrize(
    "payload",
    [
        '{"entries":{"' + "e" * 64 + '":"секретный текст"}',
        json.dumps(
            {
                "schema_version": 2,
                "entries": {"f" * 64: "секретный текст"},
            },
            ensure_ascii=False,
        ),
        json.dumps(
            {
                "schema_version": agent.REQUEST_CACHE_VERSION,
                "cache_type": agent.REQUEST_CACHE_TYPE,
                "entries": {
                    "1" * 64: {
                        "translation": "секретный текст",
                        "created_at": "повреждено",
                        "last_used_at": 900.0,
                    }
                },
            },
            ensure_ascii=False,
        ),
    ],
)
def test_request_cache_rejects_corruption_without_logging_content(
    isolated_request_cache,
    caplog,
    payload: str,
) -> None:
    cache_path, logger = isolated_request_cache
    cache_path.write_text(payload, encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger=logger.name):
        agent._load_request_cache(logger)

    assert agent._request_cache == {}
    assert "секретный текст" not in caplog.text
    assert "e" * 64 not in caplog.text
    assert "f" * 64 not in caplog.text
    assert "1" * 64 not in caplog.text


def test_request_cache_hit_works_without_openai_client_or_api_key(
    isolated_request_cache,
    monkeypatch,
) -> None:
    cache_path, logger = isolated_request_cache
    source = "Private subtitle text"
    translation = "Закешированный перевод"
    options = TranslationOptions(source_lang="en", target_lang="ru")
    cache_key = agent._make_cache_key(source, options)
    cache_path.write_text(
        json.dumps(
            agent._request_cache_payload({cache_key: _cache_entry(translation, 900.0, 950.0)}),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with (
        patch.object(agent, "OpenAI") as openai_client,
        patch.object(agent, "_request_translation") as request_translation,
        patch.object(agent, "_get_logger", return_value=logger),
    ):
        result = agent.AgentTranslator().translate_batch([source], options)

    assert result == [translation]
    openai_client.assert_not_called()
    request_translation.assert_not_called()
    persisted = json.loads(cache_path.read_text(encoding="utf-8"))
    assert source not in cache_path.read_text(encoding="utf-8")
    assert persisted["entries"][cache_key]["created_at"] == 900.0
    assert persisted["entries"][cache_key]["last_used_at"] == 1_000.0


def test_openai_initialization_error_does_not_expose_api_key(
    isolated_request_cache,
    monkeypatch,
    caplog,
) -> None:
    _cache_path, logger = isolated_request_cache
    sentinel = "unit-test-openai-secret"
    monkeypatch.setenv("OPENAI_API_KEY", sentinel)
    monkeypatch.setattr(agent, "_request_cache_loaded", True)

    with (
        patch.object(agent, "OpenAI", side_effect=RuntimeError(sentinel)),
        patch.object(agent, "_get_logger", return_value=logger),
        caplog.at_level(logging.ERROR, logger=logger.name),
        pytest.raises(agent.TranslatorLoadError) as caught,
    ):
        agent.ensure_translator_ready()

    assert sentinel not in str(caught.value)
    assert sentinel not in caplog.text
    assert caught.value.__cause__ is None


def test_usage_persistence_error_does_not_discard_successful_response(monkeypatch, caplog) -> None:
    logger = logging.getLogger("agent-usage-persistence-test")
    usage = SimpleNamespace(input_tokens=12, output_tokens=8, total_tokens=20)
    response = SimpleNamespace(output_text="Готовый перевод", usage=usage)
    client = SimpleNamespace(responses=SimpleNamespace(create=lambda **_kwargs: response))

    def fail_record_usage(**_kwargs) -> None:
        raise PermissionError(13)

    monkeypatch.setattr(agent, "record_usage", fail_record_usage)
    monkeypatch.setattr(agent, "ensure_translator_ready", lambda: None)
    monkeypatch.setattr(agent, "_client", client)

    with caplog.at_level(logging.WARNING, logger=logger.name):
        result = agent._request_translation("Служебная подсказка", None, logger)

    assert result == "Готовый перевод"
    assert "Не удалось сохранить статистику использования агента (тип ошибки: PermissionError)." in caplog.text


def test_prompt_cache_metadata_contains_only_signatures(tmp_path, monkeypatch) -> None:
    cache_path = tmp_path / "prompt-cache.json"
    system_prompt = "секретная системная подсказка"
    user_prompt = "секретная пользовательская подсказка: {source}"
    monkeypatch.setattr(agent, "PROMPT_CACHE_META_FILE", cache_path)
    monkeypatch.setattr(agent, "SYSTEM_PROMPT", system_prompt)

    agent._persist_prompt_cache_meta("at-test", user_prompt, "custom")

    raw = cache_path.read_text(encoding="utf-8")
    payload = json.loads(raw)
    assert system_prompt not in raw
    assert user_prompt not in raw
    assert payload == {
        "key": "at-test",
        "model": agent.MODEL_NAME,
        "version": agent.PROMPT_CACHE_VERSION,
        "system_prompt_signature": agent.hashlib.sha256(system_prompt.encode("utf-8")).hexdigest(),
        "variant": "custom",
        "user_prompt_signature": agent.hashlib.sha256(user_prompt.encode("utf-8")).hexdigest(),
    }


def test_agent_module_initialization_does_not_write_prompt_cache(monkeypatch) -> None:
    def unexpected_write(*_args, **_kwargs):
        raise AssertionError("импорт не должен записывать кеш подсказок")

    monkeypatch.setattr(io_utils, "atomic_write_json", unexpected_write)

    importlib.reload(agent)
