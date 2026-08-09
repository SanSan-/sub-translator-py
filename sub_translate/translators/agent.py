from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass

from filelock import FileLock
from openai import OpenAI, OpenAIError

from sub_translate.constants import CACHE_DIR, PERSIST_DIR, TRANSLATOR_LOGS_DIR
from sub_translate.models import TranslationOptions
from sub_translate.translators.agent_prompts import (
    BATCH_PROMPT_VARIANT,
    DEFAULT_PROMPT_VARIANT,
    PROMPT_VARIANTS,
    get_prompt_template,
)
from sub_translate.translators.base import TranslationError
from sub_translate.utils.env_utils import load_env
from sub_translate.utils.io_utils import atomic_write_json
from sub_translate.utils.logging_utils import configure_rotating_logger
from sub_translate.utils.translation_utils import postprocess_translation, strip_source_wrapper
from sub_translate.utils.usage_tracker import log_usage_summary, record_usage

LOGS_DIR = TRANSLATOR_LOGS_DIR

REQUEST_CACHE_FILE = CACHE_DIR / "agent_translation_cache.json"
PROMPT_CACHE_META_FILE = CACHE_DIR / "agent_prompt_cache.json"
PRICING_FILE = PERSIST_DIR / "model_pricing.json"

PROMPT_CACHE_VERSION = 2
REQUEST_CACHE_VERSION = 3
REQUEST_CACHE_TYPE = "openai-translation-fragments"
REQUEST_CACHE_TTL_SECONDS = 30 * 24 * 60 * 60
REQUEST_CACHE_MAX_ENTRIES = 4096
REQUEST_CACHE_MAX_BYTES = 16 * 1024 * 1024
REQUEST_CACHE_MIN_BYTES = 256
REQUEST_CACHE_LOCK_TIMEOUT_SECONDS = 60.0

_REQUEST_CACHE_DOCUMENT_KEYS = {"cache_type", "entries", "schema_version"}
_REQUEST_CACHE_ENTRY_KEYS = {"created_at", "last_used_at", "translation"}
_REQUEST_CACHE_KEY_PATTERN = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class RequestCachePolicy:
    ttl_seconds: int = REQUEST_CACHE_TTL_SECONDS
    max_entries: int = REQUEST_CACHE_MAX_ENTRIES
    max_bytes: int = REQUEST_CACHE_MAX_BYTES
    lock_timeout_seconds: float = REQUEST_CACHE_LOCK_TIMEOUT_SECONDS

    def __post_init__(self) -> None:
        if isinstance(self.ttl_seconds, bool) or not isinstance(self.ttl_seconds, int) or self.ttl_seconds <= 0:
            raise ValueError("Срок хранения кеша запросов должен быть положительным целым числом секунд.")
        if isinstance(self.max_entries, bool) or not isinstance(self.max_entries, int) or self.max_entries <= 0:
            raise ValueError("Максимальное число записей кеша запросов должно быть положительным.")
        if (
            isinstance(self.max_bytes, bool)
            or not isinstance(self.max_bytes, int)
            or self.max_bytes < REQUEST_CACHE_MIN_BYTES
        ):
            raise ValueError(
                f"Максимальный размер кеша запросов должен быть не меньше {REQUEST_CACHE_MIN_BYTES} байт."
            )
        if (
            isinstance(self.lock_timeout_seconds, bool)
            or not isinstance(self.lock_timeout_seconds, (int, float))
            or not math.isfinite(self.lock_timeout_seconds)
            or self.lock_timeout_seconds <= 0
        ):
            raise ValueError("Время ожидания блокировки кеша запросов должно быть положительным.")


REQUEST_CACHE_POLICY = RequestCachePolicy()


@dataclass(frozen=True, slots=True)
class _RequestCacheEntry:
    translation: str
    created_at: float
    last_used_at: float

    def as_json(self) -> dict[str, str | float]:
        return {
            "translation": self.translation,
            "created_at": self.created_at,
            "last_used_at": self.last_used_at,
        }


SYSTEM_PROMPT = """Ты - профессиональный переводчик субтитров и разнопрофильных текстов.
    Вход - обычный текст без разметки и кода. Перевод должен быть точным, естественным и без добавлений.

    Правила:
    1. Формат
       * Сохраняй количество строк и переносы строк точно как во входе.
       * Не объединяй и не дели строки, не меняй их порядок.
    2. Имена и собственные
       * Имена персонажей передавай средствами целевого языка, минимум транслитерацией.
       * Если имя уже имеет устойчивую форму в целевом языке - оставляй её без изменений.
    3. Термины и факты
       * Переводи термины по смыслу, без упрощений; сохраняй точность фактов.
       * Если фраза неполная или обрывочная, не додумывай продолжение.
    4. Числа и символы
       * Сохраняй числа, даты, единицы и пунктуацию; не меняй формат.
    5. Стиль
       * Сохраняй стиль исходника (разговорный, академический, деловой и т.д.).
       * Не добавляй пояснения и не цензурируй лексику.

    Вывод: только переведённый текст без обёрток и комментариев.
    """
_user_prompt_variant = DEFAULT_PROMPT_VARIANT
_user_prompt_template = get_prompt_template(_user_prompt_variant)


class TranslatorLoadError(RuntimeError):
    """Фатальная ошибка инициализации переводчика-агента."""


load_env()

MODEL_NAME = os.getenv("TRANSLATOR_AGENT_MODEL", "gpt-5.6-luna")

_client: OpenAI | None = None
_logger: logging.Logger | None = None
_request_cache: dict[str, _RequestCacheEntry] = {}
_request_cache_loaded = False
_cache_dirty = False
_prompt_cache_key: str | None = None
_pricing_table: dict[str, dict[str, float]] = {}
_total_input_tokens = 0
_total_output_tokens = 0
_total_cost_usd = 0.0
_cache_lock = threading.RLock()
_cache_init_lock = threading.Lock()
_usage_lock = threading.Lock()
_init_lock = threading.Lock()
_model_notice_logged = False


def get_agent_model_name() -> str:
    return MODEL_NAME


def is_openai_configured() -> bool:
    return bool((os.getenv("OPENAI_API_KEY") or "").strip())


def _get_required_openai_api_key() -> str:
    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not api_key:
        raise TranslatorLoadError("Переменная окружения OPENAI_API_KEY не задана.")
    return api_key


def set_agent_model(model_name: str) -> None:
    global MODEL_NAME, _client, _pricing_table, _model_notice_logged
    candidate = model_name.strip()
    if not candidate or candidate == MODEL_NAME:
        return
    MODEL_NAME = candidate
    os.environ["TRANSLATOR_AGENT_MODEL"] = candidate
    _model_notice_logged = False
    _client = None
    _pricing_table = {}
    _recompute_prompt_cache_key()


def apply_agent_options(options: TranslationOptions) -> None:
    model_name = options.agent_model.strip() if options.agent_model else None
    if model_name:
        set_agent_model(model_name)


def _compute_prompt_cache_key(template: str, variant: str) -> str:
    signature = json.dumps(
        {
            "version": PROMPT_CACHE_VERSION,
            "model": MODEL_NAME,
            "variant": variant,
            "system_prompt": SYSTEM_PROMPT,
            "user_prompt_template": template,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    digest = hashlib.sha256(signature.encode("utf-8")).hexdigest()
    return f"at-{digest[:48]}"


def _persist_prompt_cache_meta(cache_key: str, template: str, variant: str) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    meta = {
        "key": cache_key,
        "model": MODEL_NAME,
        "version": PROMPT_CACHE_VERSION,
        "system_prompt_signature": hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        "variant": variant,
        "user_prompt_signature": hashlib.sha256(template.encode("utf-8")).hexdigest(),
    }
    atomic_write_json(PROMPT_CACHE_META_FILE, meta)


def _recompute_prompt_cache_key() -> None:
    global _prompt_cache_key
    _prompt_cache_key = _compute_prompt_cache_key(_user_prompt_template, _user_prompt_variant)
    _persist_prompt_cache_meta(_prompt_cache_key, _user_prompt_template, _user_prompt_variant)


def _set_prompt_template(template: str, variant: str) -> None:
    global _user_prompt_template, _user_prompt_variant
    _user_prompt_template = template
    _user_prompt_variant = variant
    _recompute_prompt_cache_key()


def _set_system_prompt(prompt: str) -> None:
    global SYSTEM_PROMPT
    SYSTEM_PROMPT = prompt
    _recompute_prompt_cache_key()


_prompt_cache_key = _compute_prompt_cache_key(_user_prompt_template, _user_prompt_variant)


def _get_logger() -> logging.Logger:
    global _logger
    if _logger is not None:
        return _logger

    logger = configure_rotating_logger(
        "agent_translator",
        LOGS_DIR / "agent_translator.log",
        verbose=False,
    )
    _logger = logger
    return logger


def _log_safe_error(logger: logging.Logger, message: str, *args: object) -> None:
    """Записывает ошибку без текста и трассировки ответа поставщика."""
    logger.error(message, *args)


def attach_log_handler(handler: logging.Handler) -> None:
    """Добавляет внешний обработчик к логам агента."""
    logger = _get_logger()
    if handler not in logger.handlers:
        logger.addHandler(handler)


def detach_log_handler(handler: logging.Handler) -> None:
    """Снимает внешний обработчик логов агента."""
    logger = _get_logger()
    if handler in logger.handlers:
        logger.removeHandler(handler)


def _request_cache_payload(entries: dict[str, _RequestCacheEntry]) -> dict[str, object]:
    return {
        "schema_version": REQUEST_CACHE_VERSION,
        "cache_type": REQUEST_CACHE_TYPE,
        "entries": {key: entries[key].as_json() for key in sorted(entries)},
    }


def _request_cache_payload_size(entries: dict[str, _RequestCacheEntry]) -> int:
    payload = _request_cache_payload(entries)
    return len(json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"))


def _valid_cache_timestamp(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value) and value >= 0


def _parse_request_cache_entry(value: object) -> _RequestCacheEntry | None:
    if not isinstance(value, dict) or set(value) != _REQUEST_CACHE_ENTRY_KEYS:
        return None
    translation = value.get("translation")
    created_at = value.get("created_at")
    last_used_at = value.get("last_used_at")
    if not isinstance(translation, str) or not translation:
        return None
    if not _valid_cache_timestamp(created_at) or not _valid_cache_timestamp(last_used_at):
        return None
    created = float(created_at)
    last_used = float(last_used_at)
    if last_used < created:
        return None
    return _RequestCacheEntry(
        translation=translation,
        created_at=created,
        last_used_at=last_used,
    )


def _parse_request_cache_document(value: object) -> dict[str, _RequestCacheEntry] | None:
    if not isinstance(value, dict) or set(value) != _REQUEST_CACHE_DOCUMENT_KEYS:
        return None
    if value.get("schema_version") != REQUEST_CACHE_VERSION or value.get("cache_type") != REQUEST_CACHE_TYPE:
        return None
    raw_entries = value.get("entries")
    if not isinstance(raw_entries, dict):
        return None
    entries: dict[str, _RequestCacheEntry] = {}
    for key, raw_entry in raw_entries.items():
        if not isinstance(key, str) or _REQUEST_CACHE_KEY_PATTERN.fullmatch(key) is None:
            return None
        entry = _parse_request_cache_entry(raw_entry)
        if entry is None:
            return None
        entries[key] = entry
    return entries


def _read_request_cache(logger: logging.Logger) -> dict[str, _RequestCacheEntry] | None:
    if not REQUEST_CACHE_FILE.exists():
        return {}
    try:
        raw = REQUEST_CACHE_FILE.read_bytes()
        if raw.startswith(b"\xef\xbb\xbf"):
            raise ValueError("BOM")
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning(
            "Кеш переводов агента отклонён при чтении (тип ошибки: %s).",
            type(exc).__name__,
        )
        return None
    entries = _parse_request_cache_document(payload)
    if entries is None:
        logger.warning("Кеш переводов агента отклонён: несовместимая схема или содержимое.")
    return entries


def _cache_entry_order(item: tuple[str, _RequestCacheEntry]) -> tuple[float, float, str]:
    key, entry = item
    return entry.last_used_at, entry.created_at, key


def _prune_request_cache(
    entries: dict[str, _RequestCacheEntry],
    policy: RequestCachePolicy,
    now: float,
) -> bool:
    changed = False
    stale_keys = sorted(key for key, entry in entries.items() if now - entry.last_used_at >= policy.ttl_seconds)
    for key in stale_keys:
        del entries[key]
        changed = True

    overflow = len(entries) - policy.max_entries
    if overflow > 0:
        oldest = sorted(entries.items(), key=_cache_entry_order)
        for key, _entry in oldest[:overflow]:
            del entries[key]
            changed = True

    if entries and _request_cache_payload_size(entries) > policy.max_bytes:
        oldest_keys = [key for key, _entry in sorted(entries.items(), key=_cache_entry_order)]
        lower = 1
        upper = len(oldest_keys)
        while lower < upper:
            removed_count = (lower + upper) // 2
            removed = set(oldest_keys[:removed_count])
            retained = {key: entry for key, entry in entries.items() if key not in removed}
            if _request_cache_payload_size(retained) <= policy.max_bytes:
                upper = removed_count
            else:
                lower = removed_count + 1
        for key in oldest_keys[:lower]:
            del entries[key]
        changed = True
    return changed


def _merge_request_cache_entries(
    persisted: dict[str, _RequestCacheEntry],
    current: dict[str, _RequestCacheEntry],
) -> dict[str, _RequestCacheEntry]:
    merged = dict(persisted)
    for key, entry in current.items():
        existing = merged.get(key)
        if existing is None:
            merged[key] = entry
            continue
        existing_rank = (
            existing.last_used_at,
            existing.created_at,
            hashlib.sha256(existing.translation.encode("utf-8")).digest(),
        )
        current_rank = (
            entry.last_used_at,
            entry.created_at,
            hashlib.sha256(entry.translation.encode("utf-8")).digest(),
        )
        if current_rank > existing_rank:
            merged[key] = entry
    return merged


def _load_request_cache(logger: logging.Logger) -> None:
    global _cache_dirty, _request_cache, _request_cache_loaded
    entries = _read_request_cache(logger)
    if entries is None:
        entries = {}
    changed = _prune_request_cache(entries, REQUEST_CACHE_POLICY, time.time())
    with _cache_lock:
        _request_cache = entries
        _request_cache_loaded = True
        _cache_dirty = changed
    if changed:
        _save_request_cache(logger)


def _ensure_request_cache_loaded(logger: logging.Logger) -> None:
    with _cache_lock:
        if _request_cache_loaded:
            return
    with _cache_init_lock:
        with _cache_lock:
            if _request_cache_loaded:
                return
        _load_request_cache(logger)


def _save_request_cache(logger: logging.Logger) -> None:
    global _cache_dirty, _request_cache
    try:
        with _cache_lock:
            if not _cache_dirty:
                return
            REQUEST_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
            cache_lock = FileLock(
                f"{REQUEST_CACHE_FILE}.lock",
                timeout=REQUEST_CACHE_POLICY.lock_timeout_seconds,
            )
            with cache_lock:
                persisted = _read_request_cache(logger) or {}
                merged = _merge_request_cache_entries(persisted, _request_cache)
                _prune_request_cache(merged, REQUEST_CACHE_POLICY, time.time())
                atomic_write_json(REQUEST_CACHE_FILE, _request_cache_payload(merged))
                _request_cache = merged
                _cache_dirty = False
    except OSError as exc:
        logger.warning(
            "Не удалось сохранить кеш переводов агента (тип ошибки: %s).",
            type(exc).__name__,
        )


def _load_pricing_table(logger: logging.Logger) -> dict[str, dict[str, float]]:
    if not PRICING_FILE.exists():
        logger.warning("Файл с тарифами моделей не найден: %s", PRICING_FILE)
        return {}
    try:
        data = json.loads(PRICING_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Не удалось прочитать тарифы моделей: %s", exc)
        return {}
    if not isinstance(data, dict):
        logger.warning("Неверный формат файла тарифов: ожидается объект JSON.")
        return {}
    result: dict[str, dict[str, float]] = {}
    for model, info in data.items():
        if isinstance(info, dict):
            result[str(model)] = info
    if MODEL_NAME not in result:
        logger.warning("Для модели %s отсутствуют тарифы в %s.", MODEL_NAME, PRICING_FILE)
    return result


def _calculate_request_cost(model: str, input_tokens: int, output_tokens: int) -> float | None:
    pricing = _pricing_table.get(model)
    if not pricing:
        return None
    input_rate = pricing.get("input_per_1k_tokens")
    output_rate = pricing.get("output_per_1k_tokens")
    if input_rate is None or output_rate is None:
        return None
    return (input_tokens / 1000.0) * float(input_rate) + (output_tokens / 1000.0) * float(output_rate)


def ensure_translator_ready() -> None:
    global _client, _pricing_table, _model_notice_logged
    logger = _get_logger()
    _ensure_request_cache_loaded(logger)
    if _client is not None:
        return
    with _init_lock:
        if _client is not None:
            return
        try:
            _client = OpenAI(api_key=_get_required_openai_api_key())
        except Exception:  # pragma: no cover - сетевые/конфигурационные ошибки
            raise TranslatorLoadError("Не удалось инициализировать клиента OpenAI для переводчика-агента.") from None

        if not _model_notice_logged:
            logger.info("Переводчик-агент использует модель: %s", MODEL_NAME)
            _model_notice_logged = True
        _pricing_table = _load_pricing_table(logger)


def unload_model() -> None:
    global _client, _model_notice_logged
    _client = None
    _model_notice_logged = False
    # Тяжёлый стек CUDA нужен только при явной выгрузке, а не при запуске интерфейса.
    from sub_translate.utils.local_utils import clear_gpu_memory

    clear_gpu_memory()


def set_system_prompt(prompt: str) -> None:
    """Устанавливает системный промпт переводчика-агента."""
    if not prompt or not prompt.strip():
        raise ValueError("Системный промпт не может быть пустым.")
    _set_system_prompt(prompt)


def get_user_prompt_variant() -> str:
    """Возвращает активный вариант шаблона подсказки."""
    return _user_prompt_variant


def use_user_prompt_variant(variant: str) -> None:
    """Переключает шаблон подсказки на один из предопределённых вариантов."""
    if variant not in PROMPT_VARIANTS:
        raise ValueError(f"Неизвестный вариант подсказки: {variant}")
    _set_prompt_template(get_prompt_template(variant), variant)


def set_user_prompt_template(template: str, *, variant: str = "custom") -> None:
    """Устанавливает произвольный шаблон подсказки (использовать осторожно)."""
    if not template:
        raise ValueError("Шаблон подсказки не может быть пустым.")
    _set_prompt_template(template, variant)


def _format_prompt(
    template: str,
    *,
    source: str,
    source_lang: str,
    target_lang: str,
    separator: str | None = None,
) -> str:
    return template.format(
        source=source,
        source_lang=source_lang,
        target_lang=target_lang,
        separator=separator or "",
    )


def _resolve_direction(options: TranslationOptions | None) -> tuple[str, str]:
    source_lang = (options.source_lang if options else None) or "auto"
    target_lang = (options.target_lang if options else None) or "ru"
    return source_lang.strip() or "auto", target_lang.strip() or "ru"


def _make_cache_key(text: str, options: TranslationOptions | None = None) -> str:
    source_lang, target_lang = _resolve_direction(options)
    signature = json.dumps(
        {
            "version": REQUEST_CACHE_VERSION,
            "model": MODEL_NAME,
            "prompt_cache_key": _prompt_cache_key,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "text": text,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(signature.encode("utf-8")).hexdigest()


def _get_cached_translation(cache_key: str, logger: logging.Logger) -> str | None:
    global _cache_dirty
    _ensure_request_cache_loaded(logger)
    now = time.time()
    with _cache_lock:
        entry = _request_cache.get(cache_key)
        if entry is None:
            return None
        if now - entry.last_used_at >= REQUEST_CACHE_POLICY.ttl_seconds:
            del _request_cache[cache_key]
            _cache_dirty = True
            return None
        last_used_at = max(now, entry.created_at, entry.last_used_at)
        _request_cache[cache_key] = _RequestCacheEntry(
            translation=entry.translation,
            created_at=entry.created_at,
            last_used_at=last_used_at,
        )
        _cache_dirty = True
        return entry.translation


def _store_cached_translation(cache_key: str, translation: str) -> None:
    global _cache_dirty
    now = time.time()
    with _cache_lock:
        existing = _request_cache.get(cache_key)
        created_at = existing.created_at if existing is not None else now
        last_used_at = max(now, created_at)
        _request_cache[cache_key] = _RequestCacheEntry(
            translation=translation,
            created_at=created_at,
            last_used_at=last_used_at,
        )
        _cache_dirty = True


def _normalize_cached_translation(cache_key: str, cached: str) -> str:
    global _cache_dirty
    fixed_cached = postprocess_translation(cached)
    if fixed_cached != cached:
        with _cache_lock:
            entry = _request_cache.get(cache_key)
            if entry is not None:
                _request_cache[cache_key] = _RequestCacheEntry(
                    translation=fixed_cached,
                    created_at=entry.created_at,
                    last_used_at=entry.last_used_at,
                )
                _cache_dirty = True
    return fixed_cached


def _apply_usage(usage, logger: logging.Logger) -> None:
    global _total_input_tokens, _total_output_tokens, _total_cost_usd
    input_tokens = 0
    output_tokens = 0
    total_tokens = 0
    cost_increment = None
    if usage is not None:

        def _safe_int(value) -> int:
            try:
                return int(value)
            except TypeError, ValueError:
                return 0

        input_tokens = _safe_int(getattr(usage, "input_tokens", 0))
        output_tokens = _safe_int(getattr(usage, "output_tokens", 0))
        total_tokens = _safe_int(getattr(usage, "total_tokens", 0)) or (input_tokens + output_tokens)
        cost_increment = _calculate_request_cost(MODEL_NAME, input_tokens, output_tokens)
    else:
        logger.debug("Для модели %s не заполнены данные usage.", MODEL_NAME)

    with _usage_lock:
        _total_input_tokens += input_tokens
        _total_output_tokens += output_tokens
        if cost_increment is not None:
            _total_cost_usd += cost_increment
        session_input = _total_input_tokens
        session_output = _total_output_tokens
        session_cost = _total_cost_usd
    cost_value = cost_increment if cost_increment is not None else 0.0
    logger.info(
        "Лимиты текущего запроса: input=%s, output=%s, total=%s токенов; стоимость текущего запроса ~ $%.4f",
        input_tokens,
        output_tokens,
        total_tokens,
        cost_value,
    )
    session_total_tokens = session_input + session_output
    logger.info(
        "Итог по сессии: input=%s, output=%s, total=%s токенов; стоимость~$%.4f",
        session_input,
        session_output,
        session_total_tokens,
        session_cost,
    )
    with _usage_lock:
        totals = record_usage(
            source="agent_translator",
            model=MODEL_NAME,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_increment or 0.0,
        )
    log_usage_summary(
        logger,
        totals,
    )


def _request_translation(user_prompt: str, prompt_cache_key: str | None, logger: logging.Logger) -> str:
    ensure_translator_ready()
    assert _client is not None
    try:
        response = _client.responses.create(
            model=MODEL_NAME,
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            prompt_cache_key=prompt_cache_key,
        )
    except OpenAIError as exc:
        _log_safe_error(
            logger,
            "Запрос к модели %s завершился ошибкой типа %s.",
            MODEL_NAME,
            type(exc).__name__,
        )
        raise TranslationError(f"Сервис OpenAI не выполнил запрос к модели {MODEL_NAME}.") from None

    output = response.output_text.strip()
    if not output:
        raise RuntimeError("Модель вернула пустой перевод.")
    _apply_usage(response.usage, logger)
    return output


def _build_batch_separator() -> str:
    return f"<<<SUB_TRANSLATE_BATCH_{uuid.uuid4().hex}>>>"


def _join_batch_texts(texts: list[str], separator: str) -> str:
    return f"\n{separator}\n".join(texts)


def _split_batch_output(text: str, separator: str, expected_count: int) -> list[str] | None:
    if expected_count <= 1:
        return [text]
    pattern = rf"(?:\r?\n)\s*{re.escape(separator)}\s*(?:\r?\n)"
    parts = re.split(pattern, text)
    if len(parts) != expected_count:
        fallback_pattern = rf"\s*{re.escape(separator)}\s*"
        parts = re.split(fallback_pattern, text)
        if len(parts) != expected_count:
            while parts and parts[0].strip() == "":
                parts.pop(0)
            while parts and parts[-1].strip() == "":
                parts.pop()
    if len(parts) != expected_count:
        return None
    return parts


def translate_text(text: str, options: TranslationOptions | None = None) -> str:
    if not text:
        return ""
    logger = _get_logger()

    cache_key = _make_cache_key(text, options)
    cached = _get_cached_translation(cache_key, logger)
    if cached is not None:
        logger.debug("Перевод найден в локальном кеше агента.")
        normalized = _normalize_cached_translation(cache_key, cached)
        _save_request_cache(logger)
        return normalized

    ensure_translator_ready()
    source_lang, target_lang = _resolve_direction(options)
    user_prompt = _format_prompt(
        _user_prompt_template,
        source=text,
        source_lang=source_lang,
        target_lang=target_lang,
    )
    output = _request_translation(user_prompt, _prompt_cache_key, logger)
    output = postprocess_translation(output)

    _store_cached_translation(cache_key, output)
    _save_request_cache(logger)
    return output


def _translate_texts_batch(
    texts: list[str],
    logger: logging.Logger,
    options: TranslationOptions,
) -> list[str]:
    if not texts:
        return []
    if len(texts) == 1:
        return [translate_text(texts[0], options)]

    separator = _build_batch_separator()
    joined_source = _join_batch_texts(texts, separator)
    template = get_prompt_template(BATCH_PROMPT_VARIANT)
    source_lang, target_lang = _resolve_direction(options)
    user_prompt = _format_prompt(
        template,
        source=joined_source,
        source_lang=source_lang,
        target_lang=target_lang,
        separator=separator,
    )
    prompt_cache_key = _compute_prompt_cache_key(template, BATCH_PROMPT_VARIANT)
    output = _request_translation(user_prompt, prompt_cache_key, logger)
    output = strip_source_wrapper(output)
    parts = _split_batch_output(output, separator, len(texts))
    if parts is None:
        logger.warning("Не удалось разделить результат батча агента, перехожу на поэлементный перевод.")
        return [translate_text(text, options) for text in texts]
    return [postprocess_translation(part) for part in parts]


def translate_batch(texts: list[str], options: TranslationOptions) -> list[str]:
    if not texts:
        return []
    apply_agent_options(options)
    logger = _get_logger()
    _ensure_request_cache_loaded(logger)
    result = [""] * len(texts)
    pending_texts: list[str] = []
    pending_indexes: list[int] = []
    pending_keys: list[str] = []

    for idx, text in enumerate(texts):
        if not text:
            result[idx] = ""
            continue
        cache_key = _make_cache_key(text, options)
        cached = _get_cached_translation(cache_key, logger)
        if cached is not None:
            logger.debug("Перевод найден в локальном кеше агента.")
            result[idx] = _normalize_cached_translation(cache_key, cached)
            continue
        pending_texts.append(text)
        pending_indexes.append(idx)
        pending_keys.append(cache_key)

    _save_request_cache(logger)
    if pending_texts:
        try:
            translated = _translate_texts_batch(pending_texts, logger, options)
        except TranslationError:
            raise
        except Exception as exc:
            _log_safe_error(
                logger,
                "Перевод через агент завершился внутренней ошибкой типа %s.",
                type(exc).__name__,
            )
            raise TranslationError("Перевод через агент завершился внутренней ошибкой.") from None
        if len(translated) != len(pending_texts):
            raise TranslationError("Ответ переводчика не совпадает с размером пачки.")
        for idx, cache_key, text in zip(pending_indexes, pending_keys, translated, strict=True):
            result[idx] = text
            _store_cached_translation(cache_key, text)
        _save_request_cache(logger)
    return result


class AgentTranslator:
    name = "agent"

    def translate_batch(self, texts: list[str], options: TranslationOptions) -> list[str]:
        return translate_batch(texts, options)

    @staticmethod
    def unload() -> None:
        unload_model()


__all__ = [
    "AgentTranslator",
    "TranslatorLoadError",
    "apply_agent_options",
    "attach_log_handler",
    "detach_log_handler",
    "ensure_translator_ready",
    "get_agent_model_name",
    "get_user_prompt_variant",
    "is_openai_configured",
    "set_agent_model",
    "set_system_prompt",
    "set_user_prompt_template",
    "translate_batch",
    "translate_text",
    "use_user_prompt_variant",
]
