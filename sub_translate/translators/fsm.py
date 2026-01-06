from __future__ import annotations

import torch
from transformers import FSMTForConditionalGeneration, FSMTTokenizer

from sub_translate.constants import MODELS_DIR
from sub_translate.dictionaries.languages import get_code
from sub_translate.models import TranslationOptions
from sub_translate.translators.base import TranslationError
from sub_translate.translators.local_utils import resolve_device_and_quantization
from sub_translate.utils.translation_utils import translate_text as translate_text_common

MODEL_NAME = "facebook/wmt19-en-ru"
MODEL_CACHE_DIR = MODELS_DIR
MAX_MODEL_INPUT = 512
MAX_OUTPUT_LENGTH = 1024

DEFAULT_SOURCE_LANG = "en"
DEFAULT_TARGET_LANG = "ru"


class TranslatorLoadError(RuntimeError):
    """Исключение, возникающее при невозможности загрузить legacy-переводчик."""


_tokenizer: FSMTTokenizer | None = None
_model: FSMTForConditionalGeneration | None = None
_device: torch.device | None = None


def _normalize_lang(value: str | None, default: str) -> str:
    if not value:
        return default
    value = value.strip()
    if not value:
        return default
    lowered = value.lower()
    if lowered == "auto":
        return default
    if "-" in lowered:
        lowered = lowered.split("-", 1)[0]
    if "_" in lowered:
        lowered = lowered.split("_", 1)[0]
    return get_code(lowered) or lowered


def _ensure_model_loaded() -> None:
    global _tokenizer, _model, _device
    if _tokenizer is not None and _model is not None and _device is not None:
        return

    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        _tokenizer = FSMTTokenizer.from_pretrained(MODEL_NAME, cache_dir=str(MODEL_CACHE_DIR))
        _model = FSMTForConditionalGeneration.from_pretrained(MODEL_NAME, cache_dir=str(MODEL_CACHE_DIR))
    except Exception as exc:  # pragma: no cover - сетевые или файловые ошибки
        _tokenizer = None
        _model = None
        _device = None
        raise TranslatorLoadError("Не удалось загрузить legacy-модель перевода.") from exc

    _device, _ = resolve_device_and_quantization(
        allow_quantization=False,
        gpu_message="Legacy-модель переводчика загружена в видеопамять (GPU).",
        cpu_message="Legacy-модель переводчика загружена в оперативную память (CPU).",
    )

    _model.to(_device)
    _model.eval()


def _token_count(text: str) -> int:
    _ensure_model_loaded()
    assert _tokenizer is not None
    encoded = _tokenizer(text, return_tensors="pt", truncation=False)
    return encoded["input_ids"].shape[-1]


def _translate_chunk(text: str) -> str:
    _ensure_model_loaded()
    assert _tokenizer is not None and _model is not None and _device is not None

    inputs = _tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_MODEL_INPUT,
    )
    inputs = {key: tensor.to(_device) for key, tensor in inputs.items()}

    with torch.no_grad():
        outputs = _model.generate(
            **inputs,
            max_length=min(MAX_OUTPUT_LENGTH, MAX_MODEL_INPUT * 2),
            num_beams=4,
            early_stopping=True,
        )
    return _tokenizer.decode(outputs[0], skip_special_tokens=True)


def translate_text(text: str) -> str:
    return translate_text_common(
        text=text,
        token_limit=MAX_MODEL_INPUT,
        token_counter=_token_count,
        chunk_translator=_translate_chunk,
    )


def ensure_translator_ready() -> None:
    _ensure_model_loaded()


def translate_batch(texts: list[str], options: TranslationOptions) -> list[str]:
    if not texts:
        return []
    source_lang = _normalize_lang(options.source_lang, DEFAULT_SOURCE_LANG)
    target_lang = _normalize_lang(options.target_lang, DEFAULT_TARGET_LANG)
    if source_lang != DEFAULT_SOURCE_LANG or target_lang != DEFAULT_TARGET_LANG:
        raise TranslationError(
            "FSM-переводчик поддерживает только en -> ru "
            f"(получено {source_lang} -> {target_lang})."
        )
    try:
        _ensure_model_loaded()
    except TranslatorLoadError as exc:
        raise TranslationError(str(exc)) from exc
    result: list[str] = []
    for text in texts:
        try:
            result.append(translate_text(text))
        except Exception as exc:
            raise TranslationError(f"Ошибка перевода через FSM: {exc}") from exc
    return result


class FsmTranslator:
    name = "fsm"

    @staticmethod
    def translate_batch(texts: list[str], options: TranslationOptions) -> list[str]:
        return translate_batch(texts, options)


__all__ = ["translate_text", "ensure_translator_ready", "TranslatorLoadError", "FsmTranslator"]
