from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any

from filelock import FileLock

from sub_translate.constants import SUBTITLE_CACHE_FILE
from sub_translate.dictionaries.languages import get_code
from sub_translate.enums import FileFormat
from sub_translate.utils.io_utils import atomic_write_json, atomic_write_text
from sub_translate.utils.validation_utils import (
    ass_validator,
    srt_start_validator,
    srt_time_validator,
    vtt_header_validator,
    vtt_time_validator,
)

OUTPUT_CACHE_SCHEMA_VERSION = 3
OUTPUT_CACHE_TYPE = "translated-subtitle-output"
OUTPUT_CACHE_ALGORITHM_VERSION = "subtitle-output-v4"
DEFAULT_CACHE_TTL_SECONDS = 30 * 24 * 60 * 60
DEFAULT_CACHE_MAX_ENTRIES = 512
DEFAULT_CACHE_LOCK_TIMEOUT_SECONDS = 60.0

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_SENSITIVE_SETTING_NAMES = {
    "api_key",
    "apikey",
    "auth_token",
    "password",
    "secret",
    "token",
}
_SENSITIVE_SETTING_SUFFIXES = ("_api_key", "_password", "_secret", "_token")
_SIGNATURE_KEYS = {
    "source_sha256",
    "format",
    "source_language",
    "target_language",
    "translator_id",
    "model_id",
    "model_revision",
    "model_content_fingerprint",
    "settings",
    "prompt_signatures",
    "algorithm_version",
}
_ENTRY_KEYS = {"created_at", "signature", "lines_sha256", "line_separator", "lines"}
_DOCUMENT_KEYS = {"schema_version", "cache_type", "entries"}

type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]


class CacheRestoreStatus(StrEnum):
    RESTORED = "restored"
    OUTPUT_EXISTS = "output_exists"
    MISS = "miss"
    CACHE_REJECTED = "cache_rejected"
    WRITE_FAILED = "write_failed"

    @property
    def handled(self) -> bool:
        return self in {self.RESTORED, self.OUTPUT_EXISTS}


@dataclass(frozen=True, slots=True)
class CachePolicy:
    ttl_seconds: int = DEFAULT_CACHE_TTL_SECONDS
    max_entries: int = DEFAULT_CACHE_MAX_ENTRIES
    lock_timeout_seconds: float = DEFAULT_CACHE_LOCK_TIMEOUT_SECONDS

    def __post_init__(self) -> None:
        if isinstance(self.ttl_seconds, bool) or not isinstance(self.ttl_seconds, int) or self.ttl_seconds <= 0:
            raise ValueError("Срок хранения кеша должен быть положительным целым числом секунд.")
        if isinstance(self.max_entries, bool) or not isinstance(self.max_entries, int) or self.max_entries <= 0:
            raise ValueError("Максимальное число записей кеша должно быть положительным.")
        if (
            isinstance(self.lock_timeout_seconds, bool)
            or not isinstance(self.lock_timeout_seconds, (int, float))
            or not math.isfinite(self.lock_timeout_seconds)
            or self.lock_timeout_seconds <= 0
        ):
            raise ValueError("Время ожидания блокировки кеша должно быть положительным.")


DEFAULT_CACHE_POLICY = CachePolicy()


@dataclass(frozen=True, slots=True)
class OutputCacheIdentity:
    source_path: Path
    file_format: FileFormat
    source_language: str
    target_language: str
    translator_id: str
    model_id: str | None = None
    model_revision: str | None = None
    model_content_fingerprint: str | None = None
    settings: Mapping[str, JsonValue] = field(default_factory=dict)
    prompt_signatures: Mapping[str, str] = field(default_factory=dict)
    algorithm_version: str = OUTPUT_CACHE_ALGORITHM_VERSION
    local_model_unavailable: bool = field(default=False, repr=False, compare=False)
    source_sha256: str = field(init=False, repr=False)
    _settings_json: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        source_path = Path(self.source_path).expanduser().resolve()
        if not source_path.is_file():
            raise FileNotFoundError(f"Исходный файл не найден: {source_path}")
        object.__setattr__(self, "source_path", source_path)
        object.__setattr__(self, "source_sha256", _sha256_file(source_path))
        object.__setattr__(self, "file_format", FileFormat(self.file_format))
        object.__setattr__(
            self,
            "source_language",
            _normalize_language(self.source_language, "исходный язык"),
        )
        object.__setattr__(
            self,
            "target_language",
            _normalize_language(self.target_language, "целевой язык"),
        )
        object.__setattr__(self, "translator_id", _normalize_id(self.translator_id, "переводчик"))
        object.__setattr__(self, "model_id", _normalize_optional_text(self.model_id))
        object.__setattr__(self, "model_revision", _normalize_optional_text(self.model_revision))
        object.__setattr__(
            self,
            "model_content_fingerprint",
            _normalize_optional_sha256(self.model_content_fingerprint),
        )
        if not isinstance(self.local_model_unavailable, bool):
            raise TypeError("Признак недоступности локальной модели должен быть логическим значением.")
        if self.local_model_unavailable and (
            self.model_id is None or self.model_revision is None or self.model_content_fingerprint is not None
        ):
            raise ValueError(
                "Недоступная локальная модель должна иметь идентификатор и ревизию без отпечатка содержимого."
            )
        object.__setattr__(self, "algorithm_version", _normalize_id(self.algorithm_version, "алгоритм"))
        normalized_settings = _normalize_json_mapping(self.settings)
        normalized_prompts = _normalize_prompt_signatures(self.prompt_signatures)
        settings_json = _canonical_json(normalized_settings).decode("utf-8")
        object.__setattr__(self, "settings", MappingProxyType(normalized_settings))
        object.__setattr__(self, "prompt_signatures", MappingProxyType(normalized_prompts))
        object.__setattr__(self, "_settings_json", settings_json)


@dataclass(frozen=True, slots=True)
class _CacheReadResult:
    document: dict[str, Any] | None


def prompt_signature(prompt: str) -> str:
    """Возвращает безопасную подпись подсказки без сохранения её текста в кеше."""
    if not isinstance(prompt, str):
        raise TypeError("Подсказка должна быть строкой.")
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _normalize_language(value: str, field_name: str) -> str:
    normalized = _normalize_id(value, field_name)
    return get_code(normalized) or normalized


def _normalize_id(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Поле «{field_name}» не должно быть пустым.")
    return value.strip().casefold()


def _normalize_optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("Идентификатор модели и ревизия должны быть строками.")
    normalized = value.strip()
    return normalized or None


def _normalize_optional_sha256(value: str | None) -> str | None:
    normalized = _normalize_optional_text(value)
    if normalized is not None and not _SHA256_PATTERN.fullmatch(normalized):
        raise ValueError("Отпечаток содержимого модели должен быть SHA-256 в нижнем регистре.")
    return normalized


def _contains_sensitive_part(key: str) -> bool:
    normalized = key.casefold().replace("-", "_")
    return normalized in _SENSITIVE_SETTING_NAMES or normalized.endswith(_SENSITIVE_SETTING_SUFFIXES)


def _normalize_json_value(value: Any, *, key_path: str) -> JsonValue:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"Настройка «{key_path}» содержит нечисловое значение.")
        return value
    if isinstance(value, Mapping):
        return _normalize_json_mapping(value, key_path=key_path)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_normalize_json_value(item, key_path=f"{key_path}[{index}]") for index, item in enumerate(value)]
    raise TypeError(f"Настройка «{key_path}» не является допустимым JSON-значением.")


def _normalize_json_mapping(
    value: Mapping[str, Any],
    *,
    key_path: str = "settings",
) -> dict[str, JsonValue]:
    if not isinstance(value, Mapping):
        raise TypeError("Значимые настройки должны быть отображением.")
    result: dict[str, JsonValue] = {}
    keys = list(value)
    if any(not isinstance(key, str) or not key for key in keys):
        raise TypeError("Имена значимых настроек должны быть непустыми строками.")
    for key in sorted(keys):
        if _contains_sensitive_part(key):
            raise ValueError(f"Секретное поле «{key_path}.{key}» нельзя включать в кеш.")
        result[key] = _normalize_json_value(value[key], key_path=f"{key_path}.{key}")
    return result


def _normalize_prompt_signatures(value: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise TypeError("Подписи подсказок должны быть отображением.")
    result: dict[str, str] = {}
    keys = list(value)
    if any(not isinstance(key, str) or not key for key in keys):
        raise TypeError("Имена подписей подсказок должны быть непустыми строками.")
    for key in sorted(keys):
        signature = value[key]
        if not isinstance(signature, str) or not _SHA256_PATTERN.fullmatch(signature.casefold()):
            raise ValueError(f"Подпись подсказки «{key}» должна быть SHA-256.")
        result[key] = signature.casefold()
    return result


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _build_signature(identity: OutputCacheIdentity) -> dict[str, JsonValue]:
    if not identity.source_path.is_file():
        raise FileNotFoundError(f"Исходный файл не найден: {identity.source_path}")
    if _sha256_file(identity.source_path) != identity.source_sha256:
        raise ValueError("Исходный файл изменился после построения идентификатора кеша.")
    return {
        "source_sha256": identity.source_sha256,
        "format": identity.file_format.value,
        "source_language": identity.source_language,
        "target_language": identity.target_language,
        "translator_id": identity.translator_id,
        "model_id": identity.model_id,
        "model_revision": identity.model_revision,
        "model_content_fingerprint": identity.model_content_fingerprint,
        "settings": json.loads(identity._settings_json),
        "prompt_signatures": dict(identity.prompt_signatures),
        "algorithm_version": identity.algorithm_version,
    }


def _fingerprint(signature: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(signature)).hexdigest()


def build_cache_fingerprint(identity: OutputCacheIdentity) -> str:
    """Строит полный отпечаток результата по источнику и значимому контракту перевода."""
    return _fingerprint(_build_signature(identity))


def _lines_digest(lines: Sequence[str], line_separator: str) -> str:
    return hashlib.sha256(line_separator.join(lines).encode("utf-8")).hexdigest()


def _split_blocks(lines: Sequence[str]) -> list[list[str]]:
    blocks: list[list[str]] = []
    current: list[str] = []
    for line in lines:
        if line == "":
            if current:
                blocks.append(current)
                current = []
            continue
        current.append(line)
    if current:
        blocks.append(current)
    return blocks


def _validate_srt(lines: Sequence[str]) -> bool:
    blocks = _split_blocks(lines)
    if not blocks:
        return False
    for block in blocks:
        if len(block) < 3 or not srt_start_validator(block[0]):
            return False
        if not srt_time_validator(block[1]) or not any(line.strip() for line in block[2:]):
            return False
    return True


def _validate_vtt(lines: Sequence[str]) -> bool:
    if not lines or not vtt_header_validator(lines[0]):
        return False
    body_start = 1
    while body_start < len(lines) and lines[body_start] != "":
        body_start += 1
    blocks = _split_blocks(lines[body_start + 1 :])
    if not blocks:
        return False
    for block in blocks:
        timing_index = 0 if vtt_time_validator(block[0]) else 1
        if len(block) <= timing_index + 1 or not vtt_time_validator(block[timing_index]):
            return False
        if not any(line.strip() for line in block[timing_index + 1 :]):
            return False
    return True


def _validate_ass(lines: Sequence[str]) -> bool:
    in_events = False
    has_format = False
    dialogue_count = 0
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            in_events = stripped.casefold() == "[events]"
            continue
        if in_events and stripped.casefold().startswith("format:"):
            has_format = True
            continue
        if stripped.casefold().startswith("dialogue:"):
            if not in_events or not ass_validator(stripped):
                return False
            dialogue_count += 1
    return has_format and dialogue_count > 0


def validate_cached_lines(lines: Sequence[str], file_format: FileFormat) -> bool:
    """Проверяет, что кеш содержит цельный документ заявленного формата."""
    if (
        isinstance(lines, (str, bytes, bytearray))
        or not isinstance(lines, Sequence)
        or not lines
        or any(not isinstance(line, str) or "\x00" in line or "\r" in line or "\n" in line for line in lines)
    ):
        return False
    validators = {
        FileFormat.ASS: _validate_ass,
        FileFormat.SRT: _validate_srt,
        FileFormat.VTT: _validate_vtt,
    }
    return validators[FileFormat(file_format)](lines)


def _valid_number(value: Any) -> bool:
    return (
        not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(float(value)) and value >= 0
    )


def _is_valid_json_value(value: Any) -> bool:
    if value is None or isinstance(value, (str, bool, int)):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(_is_valid_json_value(item) for item in value)
    if isinstance(value, dict):
        return all(
            isinstance(key, str) and bool(key) and not _contains_sensitive_part(key) and _is_valid_json_value(item)
            for key, item in value.items()
        )
    return False


def _is_valid_signature(signature: Any) -> bool:
    if not isinstance(signature, dict) or set(signature) != _SIGNATURE_KEYS:
        return False
    string_keys = (
        "source_sha256",
        "format",
        "source_language",
        "target_language",
        "translator_id",
        "algorithm_version",
    )
    if any(not isinstance(signature[key], str) or not signature[key] for key in string_keys):
        return False
    if not _SHA256_PATTERN.fullmatch(signature["source_sha256"]):
        return False
    try:
        FileFormat(signature["format"])
    except ValueError:
        return False
    if any(
        value is not None and (not isinstance(value, str) or not value.strip())
        for value in (signature["model_id"], signature["model_revision"])
    ):
        return False
    model_fingerprint = signature["model_content_fingerprint"]
    if model_fingerprint is not None and (
        not isinstance(model_fingerprint, str) or not _SHA256_PATTERN.fullmatch(model_fingerprint)
    ):
        return False
    settings = signature["settings"]
    prompts = signature["prompt_signatures"]
    return (
        isinstance(settings, dict)
        and _is_valid_json_value(settings)
        and isinstance(prompts, dict)
        and all(
            isinstance(key, str) and bool(key) and isinstance(value, str) and bool(_SHA256_PATTERN.fullmatch(value))
            for key, value in prompts.items()
        )
    )


def _is_valid_entry(fingerprint: str, entry: Any) -> bool:
    if not _SHA256_PATTERN.fullmatch(fingerprint):
        return False
    if not isinstance(entry, dict) or set(entry) != _ENTRY_KEYS:
        return False
    signature = entry["signature"]
    lines = entry["lines"]
    if not _valid_number(entry["created_at"]) or not _is_valid_signature(signature):
        return False
    if not isinstance(lines, list) or not all(isinstance(line, str) for line in lines):
        return False
    if not isinstance(entry["lines_sha256"], str):
        return False
    line_separator = entry["line_separator"]
    if line_separator not in {"\n", "\r\n"}:
        return False
    try:
        file_format = FileFormat(signature["format"])
    except ValueError:
        return False
    return (
        _fingerprint(signature) == fingerprint
        and _lines_digest(lines, line_separator) == entry["lines_sha256"]
        and validate_cached_lines(lines, file_format)
    )


def _is_valid_document(document: Any) -> bool:
    if not isinstance(document, dict) or set(document) != _DOCUMENT_KEYS:
        return False
    if document["schema_version"] != OUTPUT_CACHE_SCHEMA_VERSION:
        return False
    if document["cache_type"] != OUTPUT_CACHE_TYPE:
        return False
    entries = document["entries"]
    return isinstance(entries, dict) and all(
        _is_valid_entry(fingerprint, entry) for fingerprint, entry in entries.items()
    )


def _empty_document() -> dict[str, Any]:
    return {
        "schema_version": OUTPUT_CACHE_SCHEMA_VERSION,
        "cache_type": OUTPUT_CACHE_TYPE,
        "entries": {},
    }


def _read_document(cache_path: Path, logger: logging.Logger) -> _CacheReadResult:
    if not cache_path.exists():
        return _CacheReadResult(_empty_document())
    try:
        raw = cache_path.read_bytes()
        if raw.startswith(b"\xef\xbb\xbf"):
            raise ValueError("обнаружен BOM")
        document = json.loads(raw.decode("utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("Кеш готовых переводов отклонён: %s", exc)
        return _CacheReadResult(None)
    if not _is_valid_document(document):
        logger.warning("Кеш готовых переводов отклонён: несовместимая схема или содержимое.")
        return _CacheReadResult(None)
    return _CacheReadResult(document)


def _cache_lock_path(cache_path: Path) -> Path:
    return cache_path.with_name(f"{cache_path.name}.lock")


def _prune_entries(document: dict[str, Any], policy: CachePolicy, now: float) -> bool:
    entries = document["entries"]
    stale = [
        fingerprint for fingerprint, entry in entries.items() if now - float(entry["created_at"]) >= policy.ttl_seconds
    ]
    for fingerprint in stale:
        del entries[fingerprint]
    overflow = len(entries) - policy.max_entries
    if overflow > 0:
        oldest = sorted(
            entries,
            key=lambda fingerprint: (entries[fingerprint]["created_at"], fingerprint),
        )
        for fingerprint in oldest[:overflow]:
            del entries[fingerprint]
    return bool(stale or overflow > 0)


def _read_utf8_lines(path: Path) -> tuple[list[str], str]:
    raw = path.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        raise ValueError("Файл результата содержит BOM.")
    text = raw.decode("utf-8")
    without_crlf = text.replace("\r\n", "")
    if "\r" in without_crlf or ("\r\n" in text and "\n" in without_crlf):
        raise ValueError("Файл результата содержит смешанные или неподдерживаемые переводы строк.")
    line_separator = "\r\n" if "\r\n" in text else "\n"
    return text.split(line_separator), line_separator


def _signature_without_model_fingerprint(signature: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in signature.items() if key != "model_content_fingerprint"}


def _find_unavailable_model_entry(
    identity: OutputCacheIdentity,
    signature: Mapping[str, Any],
    entries: Mapping[str, Any],
    logger: logging.Logger,
) -> dict[str, Any] | None:
    if not identity.local_model_unavailable:
        return None
    expected = _signature_without_model_fingerprint(signature)
    candidates = [
        entry
        for entry in entries.values()
        if entry["signature"]["model_content_fingerprint"] is not None
        and _signature_without_model_fingerprint(entry["signature"]) == expected
    ]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        logger.warning(
            "Кеш готовых переводов содержит несколько результатов для недоступной локальной модели; "
            "автоматическое восстановление отменено."
        )
    return None


def _lookup_lines(
    identity: OutputCacheIdentity,
    logger: logging.Logger,
    cache_path: Path,
    policy: CachePolicy,
) -> tuple[CacheRestoreStatus, tuple[list[str], str] | None]:
    signature = _build_signature(identity)
    fingerprint = _fingerprint(signature)
    cache_path = Path(cache_path).expanduser().resolve()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with FileLock(str(_cache_lock_path(cache_path)), timeout=policy.lock_timeout_seconds):
            result = _read_document(cache_path, logger)
            if result.document is None:
                return CacheRestoreStatus.CACHE_REJECTED, None
            changed = _prune_entries(result.document, policy, time.time())
            entry = result.document["entries"].get(fingerprint)
            if entry is None:
                entry = _find_unavailable_model_entry(
                    identity,
                    signature,
                    result.document["entries"],
                    logger,
                )
            if changed:
                atomic_write_json(cache_path, result.document)
    except OSError as exc:
        logger.warning("Не удалось прочитать кеш готовых переводов: %s", exc)
        return CacheRestoreStatus.CACHE_REJECTED, None
    if entry is None:
        return CacheRestoreStatus.MISS, None
    return CacheRestoreStatus.RESTORED, (list(entry["lines"]), entry["line_separator"])


def has_cached_output(
    identity: OutputCacheIdentity,
    logger: logging.Logger,
    *,
    cache_path: Path = SUBTITLE_CACHE_FILE,
    policy: CachePolicy = DEFAULT_CACHE_POLICY,
) -> bool:
    status, _ = _lookup_lines(identity, logger, cache_path, policy)
    return status is CacheRestoreStatus.RESTORED


def restore_output_cache(
    identity: OutputCacheIdentity,
    output_path: Path,
    logger: logging.Logger,
    *,
    force: bool = False,
    cache_path: Path = SUBTITLE_CACHE_FILE,
    policy: CachePolicy = DEFAULT_CACHE_POLICY,
) -> CacheRestoreStatus:
    """Восстанавливает проверенный результат, не заменяя пользовательский файл без ``force``."""
    resolved_output = output_path.expanduser().resolve()
    resolved_cache = Path(cache_path).expanduser().resolve()
    if resolved_output == identity.source_path:
        raise ValueError("Исходный и выходной пути не должны совпадать.")
    if resolved_output == resolved_cache:
        raise ValueError("Выходной файл и файл кеша не должны совпадать.")
    if resolved_output.exists() and not force:
        return CacheRestoreStatus.OUTPUT_EXISTS
    status, cached_output = _lookup_lines(identity, logger, resolved_cache, policy)
    if status is not CacheRestoreStatus.RESTORED or cached_output is None:
        return status
    lines, line_separator = cached_output
    try:
        atomic_write_text(resolved_output, line_separator.join(lines), overwrite=force)
    except FileExistsError:
        return CacheRestoreStatus.OUTPUT_EXISTS
    except OSError as exc:
        logger.warning("Не удалось восстановить готовый перевод из кеша: %s", exc)
        return CacheRestoreStatus.WRITE_FAILED
    return CacheRestoreStatus.RESTORED


def store_output_cache(
    identity: OutputCacheIdentity,
    output_path: Path,
    logger: logging.Logger,
    *,
    cache_path: Path = SUBTITLE_CACHE_FILE,
    policy: CachePolicy = DEFAULT_CACHE_POLICY,
) -> bool:
    """Сохраняет проверенный готовый перевод без изменения пользовательского результата."""
    if identity.local_model_unavailable:
        logger.warning("Результат не добавлен в кеш: локальная модель недоступна для построения полной подписи.")
        return False
    resolved_output = output_path.expanduser().resolve()
    resolved_cache = Path(cache_path).expanduser().resolve()
    if resolved_output == identity.source_path:
        raise ValueError("Исходный и выходной пути не должны совпадать.")
    if resolved_output == resolved_cache or identity.source_path == resolved_cache:
        raise ValueError("Пользовательский файл и файл кеша не должны совпадать.")
    try:
        lines, line_separator = _read_utf8_lines(resolved_output)
    except (OSError, ValueError) as exc:
        logger.warning("Файл результата не добавлен в кеш: %s", exc)
        return False
    if not validate_cached_lines(lines, identity.file_format):
        logger.warning("Файл результата не добавлен в кеш: структура субтитров повреждена.")
        return False
    signature = _build_signature(identity)
    fingerprint = _fingerprint(signature)
    cache_path = resolved_cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    now = time.time()
    entry = {
        "created_at": now,
        "signature": signature,
        "lines_sha256": _lines_digest(lines, line_separator),
        "line_separator": line_separator,
        "lines": lines,
    }
    try:
        with FileLock(str(_cache_lock_path(cache_path)), timeout=policy.lock_timeout_seconds):
            result = _read_document(cache_path, logger)
            if result.document is None:
                return False
            result.document["entries"][fingerprint] = entry
            _prune_entries(result.document, policy, now)
            atomic_write_json(cache_path, result.document)
    except OSError as exc:
        logger.warning("Не удалось сохранить кеш готовых переводов: %s", exc)
        return False
    return True


__all__ = [
    "OUTPUT_CACHE_ALGORITHM_VERSION",
    "OUTPUT_CACHE_SCHEMA_VERSION",
    "CachePolicy",
    "CacheRestoreStatus",
    "OutputCacheIdentity",
    "build_cache_fingerprint",
    "has_cached_output",
    "prompt_signature",
    "restore_output_cache",
    "store_output_cache",
    "validate_cached_lines",
]
