import logging

import pytest
from openai import OpenAIError

from sub_translate.models import TranslationOptions
from sub_translate.translators import agent
from sub_translate.translators.base import TranslationError


def test_split_batch_output_returns_parts():
    separator = agent._build_batch_separator()
    texts = ["One", "Two\nTwo", "Three"]
    joined = agent._join_batch_texts(texts, separator)
    parts = agent._split_batch_output(joined, separator, len(texts))
    assert parts == texts


def test_split_batch_output_returns_none_on_mismatch():
    separator = agent._build_batch_separator()
    parts = agent._split_batch_output("One\nTwo", separator, 2)
    assert parts is None


def test_split_batch_output_allows_spaces_around_separator():
    separator = agent._build_batch_separator()
    output = f"First\n {separator} \nSecond"
    parts = agent._split_batch_output(output, separator, 2)
    assert parts == ["First", "Second"]


def test_split_batch_output_allows_inline_separator():
    separator = agent._build_batch_separator()
    output = f"First {separator} Second"
    parts = agent._split_batch_output(output, separator, 2)
    assert parts == ["First", "Second"]


def test_agent_cache_key_tracks_model_prompt_and_language_direction(monkeypatch):
    monkeypatch.setattr(agent, "MODEL_NAME", "model-a")
    monkeypatch.setattr(agent, "_prompt_cache_key", "prompt-a")
    en_ru = TranslationOptions(source_lang="en", target_lang="ru")
    ru_en = TranslationOptions(source_lang="ru", target_lang="en")

    first = agent._make_cache_key("Text", en_ru)
    assert first != agent._make_cache_key("Text", ru_en)

    monkeypatch.setattr(agent, "MODEL_NAME", "model-b")
    assert first != agent._make_cache_key("Text", en_ru)


def test_agent_prompt_uses_requested_language_direction():
    prompt = agent._format_prompt(
        "{source_lang}->{target_lang}:{source}",
        source="Текст",
        source_lang="ru",
        target_lang="en",
    )
    assert prompt == "ru->en:Текст"


def test_provider_error_does_not_expose_response_text(monkeypatch, caplog):
    secret_response = "provider response with source text"

    class FailingResponses:
        @staticmethod
        def create(**_kwargs):
            raise OpenAIError(secret_response)

    class FakeClient:
        responses = FailingResponses()

    monkeypatch.setattr(agent, "ensure_translator_ready", lambda: None)
    monkeypatch.setattr(agent, "_client", FakeClient())
    logger = logging.getLogger("agent-test")

    with caplog.at_level(logging.ERROR), pytest.raises(TranslationError) as caught:
        agent._request_translation("private prompt", None, logger)

    assert secret_response not in str(caught.value)
    assert secret_response not in caplog.text
    assert caught.value.__cause__ is None
