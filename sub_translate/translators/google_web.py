from __future__ import annotations

import json
import re
import secrets
import threading
import time
import urllib.parse
import urllib.request
from typing import Any

from sub_translate.dictionaries.languages import get_code, is_supported
from sub_translate.models import TranslationOptions
from sub_translate.translators.base import TranslationError

BR = "\n"

DEFAULT_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
]

DEFAULT_GOOGLE_TLD = "com"
GOOGLE_TLD_ALLOWLIST = frozenset(
    {
        "ca",
        "co.in",
        "co.jp",
        "co.kr",
        "co.uk",
        "com",
        "com.au",
        "com.br",
        "com.mx",
        "de",
        "es",
        "fr",
        "it",
        "nl",
        "pl",
        "pt",
        "ru",
    }
)
_GOOGLE_HOST_PREFIX = "translate.google."
_UNSUPPORTED_GOOGLE_TLD_ERROR = "Домен Google Translate не поддерживается."
_UNSAFE_GOOGLE_ENDPOINT_ERROR = "Google Translate вернул небезопасное перенаправление."
_RPC_SESSION_KEY = "f.sid"

RESERVED_KEYWORDS = {
    "abstract",
    "await",
    "boolean",
    "break",
    "byte",
    "case",
    "catch",
    "char",
    "class",
    "const",
    "continue",
    "debugger",
    "default",
    "delete",
    "do",
    "double",
    "else",
    "enum",
    "export",
    "extends",
    "false",
    "final",
    "finally",
    "float",
    "for",
    "function",
    "goto",
    "if",
    "implements",
    "import",
    "in",
    "instanceof",
    "int",
    "interface",
    "let",
    "long",
    "native",
    "new",
    "null",
    "package",
    "private",
    "protected",
    "public",
    "return",
    "short",
    "static",
    "super",
    "switch",
    "synchronized",
    "this",
    "throw",
    "transient",
    "true",
    "try",
    "typeof",
    "var",
    "void",
    "volatile",
    "while",
    "with",
    "yield",
}

_delay_lock = threading.Lock()
_last_request_ts = 0.0

_ASCII_ONLY_RE = re.compile(
    r"^(?!([a-z]+|\d+|[\?=\.\*\[\]~!@#\$%\^&\(\)_+`/\-={}:';'<>,]+)$)"
    r"[a-z\d\?=\.\*\[\]~!@#\$%\^&\(\)_+`/\-={}:';'<>,]+$",
    re.IGNORECASE,
)


def normalize_google_tld(value: object) -> str:
    """Нормализует только явно разрешённые доменные суффиксы Google Translate."""
    if value is None:
        return DEFAULT_GOOGLE_TLD
    if not isinstance(value, str):
        raise ValueError(_UNSUPPORTED_GOOGLE_TLD_ERROR)
    normalized = value.strip().lower()
    if normalized not in GOOGLE_TLD_ALLOWLIST:
        raise ValueError(_UNSUPPORTED_GOOGLE_TLD_ERROR)
    return normalized


def _is_keyword(keyword: str) -> bool:
    return keyword in RESERVED_KEYWORDS


def _is_number(value: Any) -> bool:
    if isinstance(value, (int, float)):
        return True
    if not isinstance(value, str):
        return False
    cleaned = value.replace(",", "").replace(".", "")
    if cleaned.strip() == "":
        return False
    try:
        float(cleaned)
    except ValueError:
        return False
    return True


def _is_url(value: str) -> bool:
    try:
        parsed = urllib.parse.urlparse(value)
    except ValueError:
        return False
    return bool(parsed.scheme and parsed.netloc)


def _check_same(value: str, maps: list[dict[str, Any]]) -> int:
    for idx, item in enumerate(maps):
        if item.get("v") == value:
            return idx
    return -1


def _build_map_path(base: str, key: Any) -> str:
    if isinstance(key, int):
        return f"{base}[{key}]" if base else f"[{key}]"
    return f"{base}.{key}" if base else str(key)


def _compile_except_pattern(except_paths: list[str]) -> re.Pattern[str] | None:
    if not except_paths:
        return None
    escaped_paths = "|".join(map(re.escape, except_paths))
    return re.compile(rf"(^|\.)({escaped_paths})(\.|\[|$)", re.IGNORECASE)


def _should_map_value(value: Any, path: str, except_reg: re.Pattern[str] | None) -> bool:
    return (
        isinstance(value, str)
        and not _is_number(value)
        and not _is_url(value)
        and not _is_keyword(value)
        and not _ASCII_ONLY_RE.match(value)
        and (except_reg is None or not except_reg.search(path))
    )


def _append_map_value(map_list: list[dict[str, Any]], path: str, value: str) -> None:
    duplicate_index = _check_same(value, map_list)
    if duplicate_index >= 0:
        original = map_list[duplicate_index]
        map_list.insert(
            duplicate_index + 1,
            {
                "p": path,
                "v": value,
                "i": original["i"],
                "l": original["l"],
                "s": True,
            },
        )
        return
    last_map = map_list[-1] if map_list else None
    map_list.append(
        {
            "p": path,
            "v": value,
            "i": (last_map["i"] + last_map["l"]) if last_map else 0,
            "l": len(value.split(BR)),
            "s": False,
        }
    )


def _collect_map_values(
    obj: Any,
    path: str,
    map_list: list[dict[str, Any]],
    except_reg: re.Pattern[str] | None,
) -> None:
    if isinstance(obj, dict):
        entries = obj.items()
    elif isinstance(obj, list):
        entries = enumerate(obj)
    else:
        map_list.append({"p": "", "v": obj, "i": 0, "l": len(str(obj).split(BR))})
        return

    for key, value in entries:
        current_path = _build_map_path(path, key)
        if isinstance(value, (dict, list)):
            _collect_map_values(value, current_path, map_list, except_reg)
        elif _should_map_value(value, current_path, except_reg):
            _append_map_value(map_list, current_path, value)


def _en_map(
    obj: Any,
    except_paths: list[str] | None = None,
    path: str = "",
    map_list: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    result = [] if map_list is None else map_list
    except_reg = _compile_except_pattern(except_paths or [])
    _collect_map_values(obj, path, result, except_reg)
    return result


def _parse_path(path: str) -> list[str | int]:
    if not path:
        return []
    parts: list[str | int] = []
    i = 0
    while i < len(path):
        if path[i] == ".":
            i += 1
            continue
        if path[i] == "[":
            end = path.find("]", i)
            if end == -1:
                break
            idx = path[i + 1 : end]
            try:
                parts.append(int(idx))
            except ValueError:
                parts.append(idx)
            i = end + 1
            continue
        end = i
        while end < len(path) and path[end] not in ".[":
            end += 1
        parts.append(path[i:end])
        i = end
    return parts


def _ensure_list_index(target: list[Any], index: int) -> None:
    while len(target) <= index:
        target.append(None)


def _assign_path_value(target: Any, part: str | int, value: Any) -> bool:
    if isinstance(part, int):
        if not isinstance(target, list):
            return False
        _ensure_list_index(target, part)
        target[part] = value
        return True
    if not isinstance(target, dict):
        return False
    target[part] = value
    return True


def _descend_path(target: Any, part: str | int, next_part: str | int) -> Any | None:
    default_child: list[Any] | dict[str, Any] = [] if isinstance(next_part, int) else {}
    if isinstance(part, int):
        if not isinstance(target, list):
            return None
        _ensure_list_index(target, part)
        if target[part] is None:
            target[part] = default_child
        return target[part]
    if not isinstance(target, dict):
        return None
    if part not in target or target[part] is None:
        target[part] = default_child
    return target[part]


def _set_path(target: Any, path: str, value: Any) -> None:
    parts = _parse_path(path)
    if not parts:
        return
    current = target
    for index, part in enumerate(parts):
        if index == len(parts) - 1:
            _assign_path_value(current, part, value)
            return
        current = _descend_path(current, part, parts[index + 1])
        if current is None:
            return


def _de_map(src: Any, maps: list[dict[str, Any]], dest: str) -> Any:
    if isinstance(src, (dict, list)):
        if isinstance(src, dict):
            src_copy: Any = json.loads(json.dumps(src))
        else:
            src_copy = list(src)
        dest_lines = dest.split(BR)
        for map_item in maps:
            start = map_item["i"]
            length = map_item["l"]
            value = BR.join(dest_lines[start : start + length])
            _set_path(src_copy, map_item["p"], value)
        return src_copy
    return dest


def _extract(key: str, body: str) -> str:
    match = re.search(rf"'{re.escape(key)}':'.*?'", body)
    if match:
        return match.group(0).replace(f"'{key}':'", "")[:-1]
    return ""


def _validate_google_endpoint(url: str, expected_host: str | None = None) -> str:
    try:
        parsed = urllib.parse.urlsplit(url)
        port = parsed.port
    except ValueError:
        raise TranslationError(_UNSAFE_GOOGLE_ENDPOINT_ERROR) from None
    hostname = (parsed.hostname or "").lower()
    if (
        parsed.scheme != "https"
        or parsed.username is not None
        or parsed.password is not None
        or port is not None
        or parsed.fragment
        or not hostname.startswith(_GOOGLE_HOST_PREFIX)
    ):
        raise TranslationError(_UNSAFE_GOOGLE_ENDPOINT_ERROR)
    suffix = hostname.removeprefix(_GOOGLE_HOST_PREFIX)
    try:
        normalized_suffix = normalize_google_tld(suffix)
    except ValueError:
        raise TranslationError(_UNSAFE_GOOGLE_ENDPOINT_ERROR) from None
    allowed_host = f"{_GOOGLE_HOST_PREFIX}{normalized_suffix}"
    if hostname != allowed_host or (expected_host is not None and hostname != expected_host):
        raise TranslationError(_UNSAFE_GOOGLE_ENDPOINT_ERROR)
    return allowed_host


class _SameGoogleHostRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Разрешает перенаправление только внутри исходного узла Google Translate."""

    def __init__(self, expected_host: str) -> None:
        super().__init__()
        self._expected_host = expected_host

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        redirect_url = urllib.parse.urljoin(req.full_url, newurl)
        _validate_google_endpoint(redirect_url, self._expected_host)
        return super().redirect_request(req, fp, code, msg, headers, redirect_url)


def _open_google_request(request: urllib.request.Request, timeout: int):
    expected_host = _validate_google_endpoint(request.full_url)
    opener = urllib.request.build_opener(_SameGoogleHostRedirectHandler(expected_host))
    return opener.open(request, timeout=timeout)


def _http_get(url: str, headers: dict[str, str], timeout: int) -> str:
    request = urllib.request.Request(url, headers=headers, method="GET")
    with _open_google_request(request, timeout) as response:
        return response.read().decode("utf-8", errors="ignore")


def _http_post(url: str, body: str, headers: dict[str, str], timeout: int) -> str:
    data = body.encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with _open_google_request(request, timeout) as response:
        return response.read().decode("utf-8", errors="ignore")


def _fetch_rpc_params(tld: str, timeout: int, user_agent: str) -> dict[str, str]:
    url = f"https://translate.google.{tld}"
    headers = {"User-Agent": user_agent, "Accept-Encoding": "identity"}
    body = _http_get(url, headers, timeout)
    return {
        _RPC_SESSION_KEY: _extract("FdrFJe", body),
        "bl": _extract("cfb2h", body),
    }


def _validate_requested_language(language: str | None) -> None:
    if language and not is_supported(language):
        raise TranslationError(f"Язык '{language}' не поддерживается.")


def _resolve_language_pair(options: TranslationOptions) -> tuple[str, str]:
    _validate_requested_language(options.source_lang)
    _validate_requested_language(options.target_lang)
    source = get_code(options.source_lang or "auto") or "auto"
    target = get_code(options.target_lang or "en") or "en"
    return source, target


def _resolve_google_tld(options: TranslationOptions) -> str:
    try:
        return normalize_google_tld(options.tld)
    except ValueError as exc:
        raise TranslationError(str(exc)) from None


def _request_google_translation(
    payload_text: str,
    source: str,
    target: str,
    tld: str,
    timeout: int,
) -> str:
    user_agent = secrets.choice(DEFAULT_USER_AGENTS)
    params = _fetch_rpc_params(tld, timeout, user_agent)
    data = {
        "rpcids": "MkEWBc",
        _RPC_SESSION_KEY: params.get(_RPC_SESSION_KEY, ""),
        "bl": params.get("bl", ""),
        "hl": "en-US",
        "soc-app": 1,
        "soc-platform": 1,
        "soc-device": 1,
        "_reqid": 1000 + secrets.randbelow(9000),
        "rt": "c",
    }
    url = f"https://translate.google.{tld}/_/TranslateWebserverUi/data/batchexecute?" + urllib.parse.urlencode(data)
    req = json.dumps([[["MkEWBc", json.dumps([[payload_text, source, target, True], [None]]), None, "generic"]]])
    body = "f.req=" + urllib.parse.quote(req) + "&"
    headers = {
        "User-Agent": user_agent,
        "content-type": "application/x-www-form-urlencoded;charset=UTF-8",
        "Accept-Encoding": "identity",
    }
    return _http_post(url, body, headers, timeout)


def _empty_translation_result() -> dict[str, Any]:
    return {
        "text": "",
        "pronunciation": "",
        "from": {
            "language": {"didYouMean": False, "iso": ""},
            "text": {"autoCorrected": False, "value": "", "didYouMean": False},
        },
        "raw": "",
    }


def _decode_rpc_response(response: str) -> Any | None:
    json_payload = response[6:]
    length_match = re.match(r"\d+", json_payload)
    if not length_match:
        return None
    length_prefix = length_match.group(0)
    payload_length = int(length_prefix)
    json_payload = json_payload[len(length_prefix) : len(length_prefix) + payload_length]
    try:
        parsed = json.loads(json_payload)
        return json.loads(parsed[0][2])
    except ValueError, TypeError, IndexError:
        return None


def _extract_main_translation(parsed: Any) -> tuple[Any, Any] | None:
    try:
        main_block = parsed[1][0][0]
        segments = main_block[5]
        text = main_block[0] if segments is None else "".join(item[0] for item in segments if item)
        return text, main_block[1]
    except TypeError, IndexError:
        return None


def _extract_source_language(parsed: Any) -> dict[str, Any] | None:
    metadata = {"didYouMean": False, "iso": ""}
    try:
        suggestion = parsed[0] and parsed[0][1] and parsed[0][1][1]
        if suggestion:
            metadata["didYouMean"] = True
            metadata["iso"] = suggestion[0]
        elif parsed[1][3] == "auto":
            metadata["iso"] = parsed[2]
        else:
            metadata["iso"] = parsed[1][3]
    except IndexError, TypeError:
        return None
    return metadata


def _extract_source_text(parsed: Any) -> dict[str, Any] | None:
    metadata = {"autoCorrected": False, "value": "", "didYouMean": False}
    try:
        correction = parsed[0] and parsed[0][1] and parsed[0][1][0]
        if not correction:
            return metadata
        value = re.sub(r"<b>(<i>)?", "[", correction[0][1])
        metadata["value"] = re.sub(r"(</i>)?</b>", "]", value)
        if correction[2] == 1:
            metadata["autoCorrected"] = True
        else:
            metadata["didYouMean"] = True
    except IndexError, TypeError:
        return None
    return metadata


def _parse_google_response(response: str) -> tuple[dict[str, Any], bool]:
    result = _empty_translation_result()
    parsed = _decode_rpc_response(response)
    if parsed is None:
        return result, False
    result["raw"] = parsed

    main_translation = _extract_main_translation(parsed)
    if main_translation is None:
        return result, False
    result["text"], result["pronunciation"] = main_translation

    source_language = _extract_source_language(parsed)
    if source_language is None:
        return result, False
    result["from"]["language"] = source_language

    source_text = _extract_source_text(parsed)
    if source_text is None:
        return result, False
    result["from"]["text"] = source_text
    return result, True


def _translate(text: Any, options: TranslationOptions, timeout: int) -> Any:
    source, target = _resolve_language_pair(options)
    tld = _resolve_google_tld(options)
    str_map = _en_map(text, options.except_paths or [])
    filtered = [item for item in str_map if not item.get("s")]
    payload_text = BR.join(item["v"] for item in filtered)
    response = _request_google_translation(payload_text, source, target, tld, timeout)
    result, complete = _parse_google_response(response)
    if not complete:
        return result if options.detail else ""
    result["text"] = _de_map(text, str_map, result["text"])
    return result if options.detail else result["text"]


def _apply_request_delay(options: TranslationOptions) -> None:
    delay_ms = options.request_delay_ms
    if delay_ms is None:
        return
    try:
        delay_value = int(delay_ms)
    except TypeError, ValueError:
        return
    if delay_value <= 0:
        return
    delay_seconds = delay_value / 1000.0
    global _last_request_ts
    with _delay_lock:
        now = time.monotonic()
        wait_for = delay_seconds - (now - _last_request_ts)
        if wait_for > 0:
            time.sleep(wait_for)
        _last_request_ts = time.monotonic()


class GoogleWebTranslator:
    name = "google"

    def __init__(self, timeout: int = 30):
        self._timeout = timeout

    def translate_batch(self, texts: list[str], options: TranslationOptions) -> list[str]:
        if not texts:
            return []
        _apply_request_delay(options)
        payload = {str(idx): text for idx, text in enumerate(texts)}
        try:
            translated = _translate(payload, options, self._timeout)
        except TranslationError:
            raise
        except Exception as exc:
            raise TranslationError(f"Ошибка обращения к Google Translate: {exc}") from exc
        if isinstance(translated, dict):
            expected_keys = [str(idx) for idx in range(len(texts))]
            if any(key not in translated for key in expected_keys):
                raise TranslationError("Ответ Google Translate не содержит перевод для каждого элемента пачки.")
            return [translated[key] for key in expected_keys]
        if len(texts) == 1 and isinstance(translated, str):
            return [translated]
        raise TranslationError("Некорректный ответ от Google Translate.")

    def unload(self) -> None:
        return None


__all__ = ["GoogleWebTranslator", "normalize_google_tld"]
