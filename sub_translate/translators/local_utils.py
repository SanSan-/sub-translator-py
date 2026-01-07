from __future__ import annotations

import gc
import re

import torch

from sub_translate.dictionaries.languages import get_code
from sub_translate.translators.base import TranslationError

try:
    from transformers import BitsAndBytesConfig
except ImportError:  # pragma: no cover - опциональная зависимость
    BitsAndBytesConfig = None  # type: ignore[misc,assignment]

MAX_MODEL_INPUT = 512
MAX_OUTPUT_LENGTH = 1024


def normalize_lang(
    value: str | None,
    *,
    aliases: dict[str, str],
    code_pattern: re.Pattern[str],
) -> str | None:
    """Нормализует код языка, возвращая алиас/код или None."""
    if not value:
        return None
    raw = value.strip()
    if not raw:
        return None
    if raw in aliases.values() or code_pattern.match(raw):
        return raw
    lowered = raw.lower()
    if lowered == "auto":
        return "auto"
    if lowered in aliases:
        return aliases[lowered]
    if "-" in lowered:
        lowered = lowered.split("-", 1)[0]
    if "_" in lowered:
        lowered = lowered.split("_", 1)[0]
    code = get_code(lowered)
    if code in aliases:
        return aliases[code]
    return code or lowered


def resolve_lang(
    value: str | None,
    *,
    default: str,
    role: str,
    aliases: dict[str, str],
    code_pattern: re.Pattern[str],
    model_label: str,
    hint: str,
) -> str:
    """Возвращает нормализованный код или дефолт, иначе вызывает TranslationError."""
    normalized = normalize_lang(value, aliases=aliases, code_pattern=code_pattern)
    if not normalized or normalized == "auto":
        return default
    if normalized in aliases.values() or code_pattern.match(normalized):
        return normalized
    raise TranslationError(
        f"Для {model_label} не поддерживается {role} язык '{value}'. {hint}"
    )


def resolve_device_and_quantization(
    *,
    allow_quantization: bool = True,
    enable_cpu_offload: bool = False,
    gpu_message: str = "Модель переводчика загружена в видеопамять (GPU).",
    cpu_message: str = "Модель переводчика загружена в оперативную память (CPU).",
) -> tuple[torch.device, BitsAndBytesConfig | None]:
    """Выбирает устройство и опциональную 8-битную квантовку (с поддержкой CPU offload)."""
    device_is_gpu = torch.cuda.is_available()
    quantization_config = None
    if device_is_gpu and allow_quantization:
        if BitsAndBytesConfig is not None:
            try:
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_enable_fp32_cpu_offload=enable_cpu_offload,
                )
            except TypeError:
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        else:
            print("Библиотека bitsandbytes недоступна - загружаем модель без 8-битной квантовки.", flush=True)

    if device_is_gpu:
        device = torch.device("cuda")
        print(gpu_message, flush=True)
    else:
        device = torch.device("cpu")
        print(cpu_message, flush=True)
    return device, quantization_config



def sanitize_generation_config(model: object) -> None:
    """Убирает конфликтующие параметры генерации."""
    config = getattr(model, "generation_config", None)
    if config is None:
        return
    max_new_tokens = getattr(config, "max_new_tokens", None)
    max_length = getattr(config, "max_length", None)
    if max_new_tokens is not None and max_length is not None:
        config.max_new_tokens = None


def clear_gpu_memory() -> None:
    """Принудительно очищает память GPU и запускает сборщик мусора Python."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


__all__ = [
    "MAX_MODEL_INPUT",
    "MAX_OUTPUT_LENGTH",
    "normalize_lang",
    "resolve_lang",
    "resolve_device_and_quantization",
    "sanitize_generation_config",
    "clear_gpu_memory",
]