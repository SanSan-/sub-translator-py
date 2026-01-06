from __future__ import annotations

import re

import torch

from sub_translate.dictionaries.languages import get_code
from sub_translate.translators.base import TranslationError

try:
    from transformers import BitsAndBytesConfig
except ImportError:  # pragma: no cover - опциональная зависимость
    BitsAndBytesConfig = None  # type: ignore[misc,assignment]


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
    gpu_message: str = "Модель переводчика загружена в видеопамять (GPU).",
    cpu_message: str = "Модель переводчика загружена в оперативную память (CPU).",
) -> tuple[torch.device, BitsAndBytesConfig | None]:
    """Выбирает устройство и опциональную 8-битную квантовку для локальной модели."""
    device_is_gpu = torch.cuda.is_available()
    quantization_config = None
    if device_is_gpu and allow_quantization:
        if BitsAndBytesConfig is not None:
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


__all__ = ["normalize_lang", "resolve_lang", "resolve_device_and_quantization"]
