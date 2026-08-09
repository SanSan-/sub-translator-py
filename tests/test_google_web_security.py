from __future__ import annotations

import json
import urllib.parse
import urllib.request
from unittest.mock import Mock, patch

import pytest

from sub_translate.models import TranslationOptions
from sub_translate.service import ProcessingSettings
from sub_translate.translators.base import TranslationError
from sub_translate.translators.google_web import (
    GoogleWebTranslator,
    _de_map,
    _en_map,
    _parse_google_response,
    _request_google_translation,
    _SameGoogleHostRedirectHandler,
    _validate_google_endpoint,
    normalize_google_tld,
)


def _build_google_response(text: str, source_language: str) -> tuple[str, list[object]]:
    main_block = [text, "произношение", None, None, None, None]
    payload: list[object] = [
        [None, [None, None]],
        [[main_block], None, None, "auto"],
        source_language,
    ]
    nested_payload = json.dumps(payload, ensure_ascii=False)
    rpc_payload = json.dumps([[None, None, nested_payload]], ensure_ascii=False)
    return "xxxxxx" + str(len(rpc_payload)) + rpc_payload, payload


@pytest.mark.parametrize(
    "value",
    [
        "com@attacker.example",
        "com/path",
        "com:443",
        "attacker.example",
        "translate.google.com",
        "co.uk.attacker.example",
        "com?next=attacker.example",
        "com#fragment",
        "com.",
        "",
    ],
)
def test_google_tld_allowlist_rejects_unsafe_values_without_echo(value: str) -> None:
    with pytest.raises(ValueError) as error_info:
        normalize_google_tld(value)

    assert str(error_info.value) == "Домен Google Translate не поддерживается."
    if value:
        assert value not in str(error_info.value)


def test_google_tld_normalizer_accepts_only_known_suffixes() -> None:
    assert normalize_google_tld(None) == "com"
    assert normalize_google_tld(" COM ") == "com"
    assert normalize_google_tld("co.uk") == "co.uk"


def test_common_processing_settings_apply_google_tld_contract() -> None:
    assert ProcessingSettings(api="google", tld=" COM ").tld == "com"

    with pytest.raises(ValueError, match="Домен Google Translate не поддерживается"):
        ProcessingSettings(api="google", tld="com@attacker.example")


def test_google_translator_rejects_unsafe_tld_before_network() -> None:
    fetch_rpc_params = Mock()
    translator = GoogleWebTranslator()
    options = TranslationOptions(source_lang="en", target_lang="ru", tld="com/path")
    with (
        patch(
            "sub_translate.translators.google_web._fetch_rpc_params",
            fetch_rpc_params,
        ),
        pytest.raises(TranslationError, match="Домен Google Translate не поддерживается"),
    ):
        translator.translate_batch(["Hello"], options)

    fetch_rpc_params.assert_not_called()


@pytest.mark.parametrize(
    "url",
    [
        "http://translate.google.com/",
        "https://translate.google.com:443/",
        "https://user@translate.google.com/",
        "https://translate.google.com.attacker.example/",
        "https://attacker.example/",
    ],
)
def test_google_endpoint_validation_rejects_unsafe_hosts(url: str) -> None:
    with pytest.raises(TranslationError, match="небезопасное перенаправление"):
        _validate_google_endpoint(url)


def test_google_redirect_handler_rejects_foreign_host() -> None:
    handler = _SameGoogleHostRedirectHandler("translate.google.com")
    request = urllib.request.Request("https://translate.google.com/")

    with pytest.raises(TranslationError, match="небезопасное перенаправление"):
        handler.redirect_request(
            request,
            None,
            302,
            "Found",
            {},
            "https://attacker.example/collect",
        )


def test_google_mapping_preserves_duplicates_and_excluded_paths() -> None:
    source = {
        "title": "Привет",
        "items": ["Мир", "Привет"],
        "skip": "Не переводить",
        "url": "https://example.test/Привет",
    }

    mapping = _en_map(source, ["skip"])
    translatable = [item["v"] for item in mapping if not item.get("s")]
    restored = _de_map(source, mapping, "Hello\nWorld")

    assert translatable == ["Привет", "Мир"]
    assert restored == {
        "title": "Hello",
        "items": ["World", "Hello"],
        "skip": "Не переводить",
        "url": "https://example.test/Привет",
    }


def test_google_response_parser_preserves_protocol_metadata() -> None:
    response, payload = _build_google_response("Перевод", "en")

    result, complete = _parse_google_response(response)

    assert complete is True
    assert result["text"] == "Перевод"
    assert result["pronunciation"] == "произношение"
    assert result["from"]["language"] == {"didYouMean": False, "iso": "en"}
    assert result["raw"] == payload


def test_google_translator_reassembles_batch_without_network() -> None:
    response, _ = _build_google_response("Hello", "ru")
    options = TranslationOptions(source_lang="ru", target_lang="en", tld="com")
    with patch(
        "sub_translate.translators.google_web._request_google_translation",
        return_value=response,
    ) as request_translation:
        result = GoogleWebTranslator().translate_batch(["Привет"], options)

    assert result == ["Hello"]
    request_translation.assert_called_once_with("Привет", "ru", "en", "com", 30)


def test_google_rpc_request_uses_safe_random_values_without_network() -> None:
    with (
        patch(
            "sub_translate.translators.google_web._fetch_rpc_params",
            return_value={"f.sid": "session", "bl": "build"},
        ),
        patch("sub_translate.translators.google_web.secrets.choice", return_value="test-agent"),
        patch("sub_translate.translators.google_web.secrets.randbelow", return_value=17),
        patch("sub_translate.translators.google_web._http_post", return_value="response") as http_post,
    ):
        result = _request_google_translation("Hello", "en", "ru", "com", 10)

    url, _, headers, timeout = http_post.call_args.args
    query = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)
    assert result == "response"
    assert query["f.sid"] == ["session"]
    assert query["_reqid"] == ["1017"]
    assert headers["User-Agent"] == "test-agent"
    assert timeout == 10


def test_google_unload_has_explicit_resource_contract() -> None:
    assert GoogleWebTranslator().unload() is None
