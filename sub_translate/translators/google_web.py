from __future__ import annotations

import json
import random
import re
import threading
import time
import urllib.parse
import urllib.request
from typing import Any, Dict, List

from sub_translate.dictionaries.languages import get_code, is_supported
from sub_translate.models import TranslationOptions
from sub_translate.translators.base import TranslationError

BR = "\n"

DEFAULT_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
]

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


def _check_same(value: str, maps: List[Dict[str, Any]]) -> int:
    for idx, item in enumerate(maps):
        if item.get("v") == value:
            return idx
    return -1


def _en_map(obj: Any, except_paths: List[str] | None = None, path: str = "", map_list=None):
    if map_list is None:
        map_list = []
    except_paths = except_paths or []
    except_reg = (
        re.compile(r"(^|\.)(%s)(\.|\[|$)" % "|".join(map(re.escape, except_paths)), re.IGNORECASE)
        if except_paths
        else None
    )

    def _build_path(base: str, key: Any) -> str:
        if isinstance(key, int):
            return f"{base}[{key}]" if base else f"[{key}]"
        return f"{base}.{key}" if base else str(key)

    def _should_map_value(value: Any, cur_path: str) -> bool:
        return (
            isinstance(value, str)
            and not _is_number(value)
            and not _is_url(value)
            and not _is_keyword(value)
            and not _ASCII_ONLY_RE.match(value)
            and (not except_reg or not except_reg.search(cur_path))
        )

    def _append_value(cur_path: str, value: str) -> None:
        idx = _check_same(value, map_list)
        if idx > -1:
            map_list.insert(
                idx + 1,
                {
                    "p": cur_path,
                    "v": value,
                    "i": map_list[idx]["i"],
                    "l": map_list[idx]["l"],
                    "s": True,
                },
            )
            return
        last_map = map_list[-1] if map_list else None
        map_list.append(
            {
                "p": cur_path,
                "v": value,
                "i": (last_map["i"] + last_map["l"]) if last_map else 0,
                "l": len(value.split(BR)),
                "s": False,
            }
        )

    def _walk_value(cur_path: str, value: Any) -> None:
        if isinstance(value, (dict, list)):
            _en_map(value, except_paths, cur_path, map_list)
            return
        if _should_map_value(value, cur_path):
            _append_value(cur_path, value)

    if isinstance(obj, dict):
        for key, value in obj.items():
            _walk_value(_build_path(path, key), value)
    elif isinstance(obj, list):
        for idx, value in enumerate(obj):
            _walk_value(_build_path(path, idx), value)
    else:
        map_list.append({"p": "", "v": obj, "i": 0, "l": len(str(obj).split(BR))})
    return map_list


def _parse_path(path: str) -> List[str | int]:
    if not path:
        return []
    parts: List[str | int] = []
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


def _set_path(target: Any, path: str, value: Any) -> None:
    if not path:
        return
    parts = _parse_path(path)
    cur = target
    for idx, part in enumerate(parts):
        last = idx == len(parts) - 1
        if isinstance(part, int):
            if not isinstance(cur, list):
                return
            while len(cur) <= part:
                cur.append(None)
            if last:
                cur[part] = value
                return
            if cur[part] is None:
                next_part = parts[idx + 1]
                cur[part] = [] if isinstance(next_part, int) else {}
            cur = cur[part]
        else:
            if not isinstance(cur, dict):
                return
            if last:
                cur[part] = value
                return
            if part not in cur or cur[part] is None:
                next_part = parts[idx + 1]
                cur[part] = [] if isinstance(next_part, int) else {}
            cur = cur[part]


def _de_map(src: Any, maps: List[Dict[str, Any]], dest: str) -> Any:
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
    match = re.search(r"'%s':'.*?'" % re.escape(key), body)
    if match:
        return match.group(0).replace("'%s':'" % key, "")[:-1]
    return ""


def _http_get(url: str, headers: Dict[str, str], timeout: int) -> str:
    request = urllib.request.Request(url, headers=headers, method="GET")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="ignore")


def _http_post(url: str, body: str, headers: Dict[str, str], timeout: int) -> str:
    data = body.encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="ignore")


def _fetch_rpc_params(tld: str, timeout: int, user_agent: str) -> Dict[str, str]:
    url = f"https://translate.google.{tld}"
    headers = {"User-Agent": user_agent, "Accept-Encoding": "identity"}
    body = _http_get(url, headers, timeout)
    return {
        "f.sid": _extract("FdrFJe", body),
        "bl": _extract("cfb2h", body),
    }


def _translate(text: Any, options: TranslationOptions, timeout: int) -> Any:
    if options.source_lang and not is_supported(options.source_lang):
        raise TranslationError(f"Язык '{options.source_lang}' не поддерживается.")
    if options.target_lang and not is_supported(options.target_lang):
        raise TranslationError(f"Язык '{options.target_lang}' не поддерживается.")

    source = get_code(options.source_lang or "auto") or "auto"
    target = get_code(options.target_lang or "en") or "en"
    tld = options.tld or "com"
    except_paths = options.except_paths or []
    detail = options.detail

    user_agent = random.choice(DEFAULT_USER_AGENTS)
    params = _fetch_rpc_params(tld, timeout, user_agent)
    data = {
        "rpcids": "MkEWBc",
        "f.sid": params.get("f.sid", ""),
        "bl": params.get("bl", ""),
        "hl": "en-US",
        "soc-app": 1,
        "soc-platform": 1,
        "soc-device": 1,
        "_reqid": random.randint(1000, 9999),
        "rt": "c",
    }

    str_map = _en_map(text, except_paths)
    filtered = [item for item in str_map if not item.get("s")]
    payload_text = BR.join(item["v"] for item in filtered)

    url = (
        f"https://translate.google.{tld}/_/TranslateWebserverUi/data/batchexecute?"
        + urllib.parse.urlencode(data)
    )
    req = json.dumps([[["MkEWBc", json.dumps([[payload_text, source, target, True], [None]]), None, "generic"]]])
    body = "f.req=" + urllib.parse.quote(req) + "&"

    headers = {
        "User-Agent": user_agent,
        "content-type": "application/x-www-form-urlencoded;charset=UTF-8",
        "Accept-Encoding": "identity",
    }

    response = _http_post(url, body, headers, timeout)
    json_payload = response[6:]
    length_match = re.match(r"\d+", json_payload)
    result = {
        "text": "",
        "pronunciation": "",
        "from": {"language": {"didYouMean": False, "iso": ""}, "text": {"autoCorrected": False, "value": "", "didYouMean": False}},
        "raw": "",
    }
    if not length_match:
        return result if detail else ""
    length = int(length_match.group(0))
    json_payload = json_payload[len(length_match.group(0)) : len(length_match.group(0)) + length]
    try:
        parsed = json.loads(json_payload)
        parsed = json.loads(parsed[0][2])
        result["raw"] = parsed
    except (ValueError, TypeError, IndexError):
        return result if detail else ""

    try:
        main_block = parsed[1][0][0]
    except (TypeError, IndexError):
        return result if detail else ""

    if main_block[5] is None:
        result["text"] = main_block[0]
    else:
        result["text"] = "".join([item[0] for item in main_block[5] if item])

    result["pronunciation"] = main_block[1]

    try:
        if parsed[0] and parsed[0][1] and parsed[0][1][1]:
            result["from"]["language"]["didYouMean"] = True
            result["from"]["language"]["iso"] = parsed[0][1][1][0]
        elif parsed[1][3] == "auto":
            result["from"]["language"]["iso"] = parsed[2]
        else:
            result["from"]["language"]["iso"] = parsed[1][3]
    except (IndexError, TypeError):
        return result if detail else ""

    try:
        if parsed[0] and parsed[0][1] and parsed[0][1][0]:
            value = parsed[0][1][0][0][1]
            value = re.sub(r"<b>(<i>)?", "[", value)
            value = re.sub(r"(</i>)?</b>", "]", value)
            result["from"]["text"]["value"] = value
            if parsed[0][1][0][2] == 1:
                result["from"]["text"]["autoCorrected"] = True
            else:
                result["from"]["text"]["didYouMean"] = True
    except (IndexError, TypeError):
        return result if detail else ""

    result["text"] = _de_map(text, str_map, result["text"])
    return result if detail else result["text"]


def _apply_request_delay(options: TranslationOptions) -> None:
    delay_ms = options.request_delay_ms
    if delay_ms is None:
        return
    try:
        delay_value = int(delay_ms)
    except (TypeError, ValueError):
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

    def translate_batch(self, texts: List[str], options: TranslationOptions) -> List[str]:
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
            result: List[str] = []
            for idx in range(len(texts)):
                result.append(str(translated.get(str(idx), "")))
            return result
        if len(texts) == 1:
            return [str(translated)]
        raise TranslationError("Некорректный ответ от Google Translate.")

    def unload(self) -> None:
        pass


__all__ = ["GoogleWebTranslator"]
