from __future__ import annotations

import json
import logging
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from sub_translate.models import TranslationOptions
from sub_translate.translators import agent
from sub_translate.web import app as web_app

STATIC_DIR = Path(web_app.__file__).resolve().parent / "static"


def test_ui_config_returns_only_safe_openai_status(monkeypatch) -> None:
    sentinel = "unit-test-openai-secret"
    monkeypatch.setenv("OPENAI_API_KEY", sentinel)

    config = web_app._build_ui_config()

    assert config["agent_settings"] == {
        "model": agent.get_agent_model_name(),
        "configured": True,
    }
    assert sentinel not in json.dumps(config, ensure_ascii=False)


def test_ui_config_reports_missing_openai_key(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    config = web_app._build_ui_config()

    assert config["agent_settings"]["configured"] is False


def test_ui_config_reports_huggingface_status_without_exposing_token(monkeypatch) -> None:
    sentinel = "unit-test-huggingface-secret"
    monkeypatch.setenv("HUGGINGFACE_TOKEN", sentinel)
    monkeypatch.delenv("HF_TOKEN", raising=False)

    config = web_app._build_ui_config()
    profile = next(item for item in config["translators"] if item["id"] == "translategemma")

    assert profile["model"]["requires_access_token"] is True
    assert profile["model"]["access_token_configured"] is True
    serialized = json.dumps(config, ensure_ascii=False)
    assert sentinel not in serialized
    assert "model_path" not in serialized
    assert "worker_python_path" not in serialized


def test_web_schema_rejects_openai_key_without_echoing_it() -> None:
    sentinel = "unit-test-openai-secret"
    response = TestClient(web_app.app, base_url="http://127.0.0.1").post(
        "/api/refresh",
        json={
            "paths": [],
            "settings": {"api": "google", "openai_api_key": sentinel},
        },
    )

    assert response.status_code == 422
    assert sentinel not in response.text


def test_browser_assets_have_no_openai_key_input_or_storage_field() -> None:
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    javascript = (STATIC_DIR / "app.js").read_text(encoding="utf-8")

    assert 'data-setting="openai_api_key"' not in html
    assert "openAiKeyArea" not in javascript
    assert "openai_api_key" not in javascript
    assert "allowedSettingKeys.has(key)" in javascript


def test_translation_options_do_not_accept_openai_key() -> None:
    assert "openai_api_key" not in TranslationOptions.__dataclass_fields__


def test_agent_translator_reads_key_from_environment(monkeypatch) -> None:
    sentinel = "unit-test-openai-secret"
    monkeypatch.setenv("OPENAI_API_KEY", sentinel)
    monkeypatch.setattr(agent, "_client", None)
    monkeypatch.setattr(agent, "_request_cache", {})
    logger = logging.getLogger("test_agent_translator_reads_key_from_environment")

    with (
        patch.object(agent, "OpenAI") as openai_client,
        patch.object(agent, "_get_logger", return_value=logger),
        patch.object(agent, "_load_request_cache"),
        patch.object(agent, "_load_pricing_table", return_value={}),
        patch.object(agent, "_request_translation", return_value="Привет"),
        patch.object(agent, "_save_request_cache"),
    ):
        result = agent.AgentTranslator().translate_batch(
            ["Hello"],
            TranslationOptions(source_lang="en", target_lang="ru"),
        )

    assert result == ["Привет"]
    openai_client.assert_called_once_with(api_key=sentinel)
