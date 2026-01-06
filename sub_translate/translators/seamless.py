from __future__ import annotations

import re

import torch
from transformers import SeamlessM4TProcessor, SeamlessM4Tv2ForTextToText

from sub_translate.constants import MODELS_DIR
from sub_translate.models import TranslationOptions
from sub_translate.translators.base import TranslationError
from sub_translate.translators.local_utils import resolve_device_and_quantization, resolve_lang
from sub_translate.utils.translation_utils import translate_text as translate_text_common

MODEL_NAME = "facebook/seamless-m4t-v2-large"
MODEL_CACHE_DIR = MODELS_DIR
MAX_MODEL_INPUT = 512
MAX_OUTPUT_LENGTH = 1024

SEAMLESS_LANGUAGE_ALIASES = {
    "en": "eng",
    "ru": "rus",
}

_SEAMLESS_CODE_PATTERN = re.compile(r"^[a-z]{3}$")

DEFAULT_SOURCE_LANG = SEAMLESS_LANGUAGE_ALIASES["en"]
DEFAULT_TARGET_LANG = SEAMLESS_LANGUAGE_ALIASES["ru"]


class TranslatorLoadError(RuntimeError):
    """Исключение, возникающее при невозможности загрузить модель перевода."""


_processor: SeamlessM4TProcessor | None = None
_model: SeamlessM4Tv2ForTextToText | None = None
_device: torch.device | None = None


def _handle_model_load_error(exc: Exception) -> None:
    raise TranslatorLoadError("Не удалось загрузить модель перевода.") from exc


def _ensure_model_loaded() -> None:
    global _processor, _model, _device
    if _processor is not None and _model is not None and _device is not None:
        return

    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        _processor = SeamlessM4TProcessor.from_pretrained(MODEL_NAME, cache_dir=str(MODEL_CACHE_DIR))
    except Exception as exc:  # pragma: no cover - сетевые или файловые ошибки
        _handle_model_load_error(exc)

    _device, quantization_config = resolve_device_and_quantization()

    try:
        if quantization_config is not None:
            _model = SeamlessM4Tv2ForTextToText.from_pretrained(
                MODEL_NAME,
                cache_dir=str(MODEL_CACHE_DIR),
                quantization_config=quantization_config,
                device_map="auto",
            )
        else:
            _model = SeamlessM4Tv2ForTextToText.from_pretrained(
                MODEL_NAME,
                cache_dir=str(MODEL_CACHE_DIR),
            )
            _model.to(_device)
    except Exception as exc:  # pragma: no cover - защита от частично загруженных весов
        _processor = None
        _model = None
        _device = None
        _handle_model_load_error(exc)

    _model.eval()


def _token_count(text: str, source_lang: str) -> int:
    _ensure_model_loaded()
    assert _processor is not None
    encoded = _processor(text=text, src_lang=source_lang, return_tensors="pt")
    return encoded["input_ids"].shape[-1]


def _decode_output(tokens) -> str:
    assert _processor is not None
    if hasattr(_processor, "decode"):
        return _processor.decode(tokens, skip_special_tokens=True)
    return _processor.tokenizer.decode(tokens, skip_special_tokens=True)


def _translate_chunk(text: str, source_lang: str, target_lang: str) -> str:
    _ensure_model_loaded()
    assert _processor is not None and _model is not None and _device is not None

    inputs = _processor(text=text, src_lang=source_lang, return_tensors="pt")
    inputs = {key: tensor.to(_device) for key, tensor in inputs.items()}

    with torch.no_grad():
        outputs = _model.generate(
            **inputs,
            max_length=min(MAX_OUTPUT_LENGTH, MAX_MODEL_INPUT * 2),
            num_beams=4,
            early_stopping=True,
            tgt_lang=target_lang,
        )
    return _decode_output(outputs[0])


def translate_text(text: str, source_lang: str, target_lang: str) -> str:
    return translate_text_common(
        text=text,
        token_limit=MAX_MODEL_INPUT,
        token_counter=lambda value: _token_count(value, source_lang),
        chunk_translator=lambda value: _translate_chunk(value, source_lang, target_lang),
    )


def ensure_translator_ready() -> None:
    _ensure_model_loaded()


class SeamlessTranslator:
    name = "seamless"

    @staticmethod
    def translate_batch(texts: list[str], options: TranslationOptions) -> list[str]:
        if not texts:
            return []
        source_lang = resolve_lang(
            options.source_lang,
            default=DEFAULT_SOURCE_LANG,
            role="исходный",
            aliases=SEAMLESS_LANGUAGE_ALIASES,
            code_pattern=_SEAMLESS_CODE_PATTERN,
            model_label="SeamlessM4T",
            hint="Используйте код ISO-639-3 (например, eng/rus) или алиас en/ru.",
        )
        target_lang = resolve_lang(
            options.target_lang,
            default=DEFAULT_TARGET_LANG,
            role="целевой",
            aliases=SEAMLESS_LANGUAGE_ALIASES,
            code_pattern=_SEAMLESS_CODE_PATTERN,
            model_label="SeamlessM4T",
            hint="Используйте код ISO-639-3 (например, eng/rus) или алиас en/ru.",
        )
        try:
            _ensure_model_loaded()
        except TranslatorLoadError as exc:
            raise TranslationError(str(exc)) from exc
        result: list[str] = []
        for text in texts:
            try:
                result.append(translate_text(text, source_lang, target_lang))
            except Exception as exc:
                raise TranslationError(f"Ошибка перевода через SeamlessM4T: {exc}") from exc
        return result


__all__ = ["translate_text", "ensure_translator_ready", "TranslatorLoadError", "SeamlessTranslator"]
