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


class Int8UnavailableError(RuntimeError):
    """Ошибка строгого режима INT8, когда CUDA или bitsandbytes недоступны."""


def _build_bitsandbytes_config(*, enable_cpu_offload: bool) -> BitsAndBytesConfig:
    assert BitsAndBytesConfig is not None
    try:
        return BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=enable_cpu_offload,
        )
    except TypeError:
        return BitsAndBytesConfig(load_in_8bit=True)


def _resolve_quantization_config(
    *,
    device_is_gpu: bool,
    allow_quantization: bool,
    enable_cpu_offload: bool,
    require_int8: bool,
) -> BitsAndBytesConfig | None:
    if not device_is_gpu or not allow_quantization:
        return None
    if BitsAndBytesConfig is None:
        if require_int8:
            raise Int8UnavailableError("Для строгого режима INT8 требуется установленная библиотека bitsandbytes.")
        print(
            "Библиотека bitsandbytes недоступна - загружаем модель без 8-битной квантовки.",
            flush=True,
        )
        return None
    try:
        return _build_bitsandbytes_config(enable_cpu_offload=enable_cpu_offload)
    except Exception as exc:
        if require_int8:
            raise Int8UnavailableError("Не удалось создать конфигурацию INT8 библиотеки bitsandbytes.") from exc
        print("Квантование INT8 недоступно - модель будет загружена без него.", flush=True)
        return None


def _selected_device(
    *,
    device_is_gpu: bool,
    gpu_message: str,
    cpu_message: str,
) -> torch.device:
    if device_is_gpu:
        print(gpu_message, flush=True)
        return torch.device("cuda")
    print(cpu_message, flush=True)
    return torch.device("cpu")


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
    raise TranslationError(f"Для {model_label} не поддерживается {role} язык '{value}'. {hint}")


def resolve_device_and_quantization(
    *,
    allow_quantization: bool = True,
    enable_cpu_offload: bool = False,
    require_int8: bool = False,
    gpu_message: str = "Модель переводчика загружена в видеопамять (GPU).",
    cpu_message: str = "Модель переводчика загружена в оперативную память (CPU).",
) -> tuple[torch.device, BitsAndBytesConfig | None]:
    """Выбирает устройство и опциональную 8-битную квантовку (с поддержкой CPU offload)."""
    device_is_gpu = torch.cuda.is_available()
    if require_int8 and not allow_quantization:
        raise Int8UnavailableError("Строгий режим INT8 нельзя отключить параметром квантования.")
    if require_int8 and not device_is_gpu:
        raise Int8UnavailableError("Для строгого режима INT8 требуется доступное устройство CUDA.")
    quantization_config = _resolve_quantization_config(
        device_is_gpu=device_is_gpu,
        allow_quantization=allow_quantization,
        enable_cpu_offload=enable_cpu_offload,
        require_int8=require_int8,
    )
    if require_int8 and quantization_config is None:
        raise Int8UnavailableError("Не удалось включить обязательное квантование INT8.")
    device = _selected_device(
        device_is_gpu=device_is_gpu,
        gpu_message=gpu_message,
        cpu_message=cpu_message,
    )
    return device, quantization_config


def sanitize_generation_config(model: object, *, deterministic: bool = False) -> None:
    """Убирает конфликтующие параметры генерации."""
    config = getattr(model, "generation_config", None)
    if config is None:
        return
    max_new_tokens = getattr(config, "max_new_tokens", None)
    max_length = getattr(config, "max_length", None)
    if max_new_tokens is not None and max_length is not None:
        config.max_new_tokens = None
    if deterministic:
        config.do_sample = False
        config.num_beams = 1
        for attribute in (
            "temperature",
            "top_p",
            "min_p",
            "typical_p",
            "top_k",
            "epsilon_cutoff",
            "eta_cutoff",
        ):
            if hasattr(config, attribute):
                setattr(config, attribute, None)
        if hasattr(config, "early_stopping"):
            config.early_stopping = False
        if hasattr(config, "length_penalty"):
            config.length_penalty = 1.0


def clear_gpu_memory() -> None:
    """Принудительно очищает память GPU и запускает сборщик мусора Python."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


__all__ = [
    "MAX_MODEL_INPUT",
    "MAX_OUTPUT_LENGTH",
    "Int8UnavailableError",
    "clear_gpu_memory",
    "normalize_lang",
    "resolve_device_and_quantization",
    "resolve_lang",
    "sanitize_generation_config",
]
