from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import uuid
from typing import Dict, Optional

from dotenv import load_dotenv
from openai import OpenAI
from sub_translate.models import TranslationOptions
from sub_translate.translators.base import TranslationError
from sub_translate.constants import CACHE_DIR, PERSIST_DIR, TRANSLATOR_LOGS_DIR
from sub_translate.utils.logging_utils import configure_rotating_logger
from sub_translate.utils.translation_utils import postprocess_translation, strip_source_wrapper
from sub_translate.utils.usage_tracker import log_usage_summary, record_usage
from sub_translate.translators.agent_prompts import (
    BATCH_PROMPT_VARIANT,
    DEFAULT_PROMPT_VARIANT,
    PROMPT_VARIANTS,
    get_prompt_template,
)

LOGS_DIR = TRANSLATOR_LOGS_DIR

REQUEST_CACHE_FILE = CACHE_DIR / "agent_translation_cache.json"
PROMPT_CACHE_META_FILE = CACHE_DIR / "agent_prompt_cache.json"
PRICING_FILE = PERSIST_DIR / "model_pricing.json"

PROMPT_CACHE_VERSION = 1

SYSTEM_PROMPT = (
    """Ты - профессиональный переводчик субтитров и разнопрофильных текстов на русский язык.
    Вход - обычный текст без разметки и кода. Перевод должен быть точным, естественным и без добавлений.

    Правила:
    1. Формат
       * Сохраняй количество строк и переносы строк точно как во входе.
       * Не объединяй и не дели строки, не меняй их порядок.
    2. Имена и собственные
       * Имена персонажей переводить на русский, минимум транслитерацией.
       * Если имя уже русифицировано - оставляй как есть.
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
)
_user_prompt_variant = DEFAULT_PROMPT_VARIANT
_user_prompt_template = get_prompt_template(_user_prompt_variant)


class TranslatorLoadError(RuntimeError):
    """Фатальная ошибка инициализации переводчика-агента."""


load_dotenv()

MODEL_NAME = os.getenv("TRANSLATOR_AGENT_MODEL", "gpt-5.1-mini")

_client: Optional[OpenAI] = None
_logger: Optional[logging.Logger] = None
_request_cache: Dict[str, str] = {}
_cache_dirty = False
_prompt_cache_key: Optional[str] = None
_pricing_table: Dict[str, Dict[str, float]] = {}
_total_input_tokens = 0
_total_output_tokens = 0
_total_cost_usd = 0.0
_cache_lock = threading.RLock()
_usage_lock = threading.Lock()


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
        "system_prompt": SYSTEM_PROMPT,
        "variant": variant,
        "user_prompt_template": template,
    }
    PROMPT_CACHE_META_FILE.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


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


_set_prompt_template(_user_prompt_template, _user_prompt_variant)


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


def _load_request_cache(logger: logging.Logger) -> None:
    global _request_cache
    if not REQUEST_CACHE_FILE.exists():
        _request_cache = {}
        return
    try:
        data = json.loads(REQUEST_CACHE_FILE.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            with _cache_lock:
                _request_cache = {str(key): str(value) for key, value in data.items()}
        else:
            logger.warning("Формат кеша переводов агента не распознан, начинаю с пустого состояния.")
            with _cache_lock:
                _request_cache = {}
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Не удалось загрузить кеш переводов агента: %s", exc)
        with _cache_lock:
            _request_cache = {}


def _save_request_cache(logger: logging.Logger) -> None:
    global _cache_dirty
    try:
        with _cache_lock:
            if not _cache_dirty:
                return
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            REQUEST_CACHE_FILE.write_text(
                json.dumps(_request_cache, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            _cache_dirty = False
    except OSError as exc:
        logger.warning("Не удалось сохранить кеш переводов агента: %s", exc)


def _load_pricing_table(logger: logging.Logger) -> Dict[str, Dict[str, float]]:
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
    result: Dict[str, Dict[str, float]] = {}
    for model, info in data.items():
        if isinstance(info, dict):
            result[str(model)] = info
    if MODEL_NAME not in result:
        logger.warning("Для модели %s отсутствуют тарифы в %s.", MODEL_NAME, PRICING_FILE)
    return result


def _calculate_request_cost(model: str, input_tokens: int, output_tokens: int) -> Optional[float]:
    pricing = _pricing_table.get(model)
    if not pricing:
        return None
    input_rate = pricing.get("input_per_1k_tokens")
    output_rate = pricing.get("output_per_1k_tokens")
    if input_rate is None or output_rate is None:
        return None
    return (input_tokens / 1000.0) * float(input_rate) + (output_tokens / 1000.0) * float(output_rate)


def ensure_translator_ready() -> None:
    global _client, _pricing_table
    logger = _get_logger()
    if _client is not None:
        return
    try:
        _client = OpenAI()
    except Exception as exc:  # pragma: no cover - сетевые/конфигурационные ошибки
        raise TranslatorLoadError("Не удалось инициализировать клиента OpenAI для переводчика-агента.") from exc

    logger.info("Переводчик-агент использует модель: %s", MODEL_NAME)
    _load_request_cache(logger)
    _pricing_table = _load_pricing_table(logger)


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


def _format_prompt(template: str, *, source: str, separator: str | None = None) -> str:
    return template.format(source=source, separator=separator or "")


def _make_cache_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _normalize_cached_translation(cache_key: str, cached: str, logger: logging.Logger) -> str:
    global _cache_dirty
    fixed_cached = postprocess_translation(cached)
    if fixed_cached != cached:
        with _cache_lock:
            _request_cache[cache_key] = fixed_cached
            _cache_dirty = True
        _save_request_cache(logger)
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
            except (TypeError, ValueError):
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


def _request_translation(user_prompt: str, prompt_cache_key: Optional[str], logger: logging.Logger) -> str:
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
    except Exception as exc:
        logger.error("Ошибка при обращении к модели %s: %s", MODEL_NAME, exc)
        raise

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
        return None
    return parts


def translate_text(text: str) -> str:
    ensure_translator_ready()
    logger = _get_logger()
    if not text:
        return ""

    cache_key = _make_cache_key(text)
    with _cache_lock:
        cached = _request_cache.get(cache_key)
    if cached is not None:
        logger.debug("Перевод найден в локальном кеше агента (ключ %s).", cache_key)
        return _normalize_cached_translation(cache_key, cached, logger)

    user_prompt = _format_prompt(_user_prompt_template, source=text)
    output = _request_translation(user_prompt, _prompt_cache_key, logger)
    output = postprocess_translation(output)

    global _cache_dirty
    with _cache_lock:
        _request_cache[cache_key] = output
        _cache_dirty = True
    _save_request_cache(logger)
    return output


def _translate_texts_batch(texts: list[str], logger: logging.Logger) -> list[str]:
    if not texts:
        return []
    if len(texts) == 1:
        return [translate_text(texts[0])]

    separator = _build_batch_separator()
    joined_source = _join_batch_texts(texts, separator)
    template = get_prompt_template(BATCH_PROMPT_VARIANT)
    user_prompt = _format_prompt(template, source=joined_source, separator=separator)
    prompt_cache_key = _compute_prompt_cache_key(template, BATCH_PROMPT_VARIANT)
    output = _request_translation(user_prompt, prompt_cache_key, logger)
    output = strip_source_wrapper(output)
    parts = _split_batch_output(output, separator, len(texts))
    if parts is None:
        logger.warning("Не удалось разделить результат батча агента, перехожу на поэлементный перевод.")
        return [translate_text(text) for text in texts]
    return [postprocess_translation(part) for part in parts]


def translate_batch(texts: list[str], options: TranslationOptions) -> list[str]:
    if not texts:
        return []
    ensure_translator_ready()
    logger = _get_logger()
    result = [""] * len(texts)
    pending_texts: list[str] = []
    pending_indexes: list[int] = []
    pending_keys: list[str] = []

    for idx, text in enumerate(texts):
        if not text:
            result[idx] = ""
            continue
        cache_key = _make_cache_key(text)
        with _cache_lock:
            cached = _request_cache.get(cache_key)
        if cached is not None:
            logger.debug("Перевод найден в локальном кеше агента (ключ %s).", cache_key)
            result[idx] = _normalize_cached_translation(cache_key, cached, logger)
            continue
        pending_texts.append(text)
        pending_indexes.append(idx)
        pending_keys.append(cache_key)

    if pending_texts:
        try:
            translated = _translate_texts_batch(pending_texts, logger)
        except TranslationError:
            raise
        except Exception as exc:
            raise TranslationError(f"Ошибка перевода через агент: {exc}") from exc
        if len(translated) != len(pending_texts):
            raise TranslationError("Ответ переводчика не совпадает с размером пачки.")
        global _cache_dirty
        with _cache_lock:
            for idx, cache_key, text in zip(pending_indexes, pending_keys, translated):
                result[idx] = text
                _request_cache[cache_key] = text
                _cache_dirty = True
        _save_request_cache(logger)
    return result

class AgentTranslator:
    name = "agent"

    def translate_batch(self, texts: list[str], options: TranslationOptions) -> list[str]:
        return translate_batch(texts, options)

__all__ = [
    "translate_text",
    "translate_batch",
    "ensure_translator_ready",
    "TranslatorLoadError",
    "set_system_prompt",
    "get_user_prompt_variant",
    "use_user_prompt_variant",
    "set_user_prompt_template",
    "AgentTranslator",
]
