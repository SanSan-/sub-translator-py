from __future__ import annotations

import re

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from sub_translate.constants import MODELS_DIR
from sub_translate.models import TranslationOptions
from sub_translate.translators.base import TranslationError
from sub_translate.utils.huggingface import TranslatorLoadError, load_model_components
from sub_translate.translators.local_utils import MAX_MODEL_INPUT, MAX_OUTPUT_LENGTH, clear_gpu_memory, resolve_lang
from sub_translate.utils.translation_utils import translate_text as translate_text_common

MODEL_NAME = "google/madlad400-7b-mt"
MODEL_CACHE_DIR = MODELS_DIR

MADLAD_LANGUAGE_ALIASES = {
    "en": "eng",
    "ru": "rus",
}

_MADLAD_CODE_PATTERN = re.compile(r"^[a-z]{3}$")

DEFAULT_SOURCE_LANG = MADLAD_LANGUAGE_ALIASES["en"]
DEFAULT_TARGET_LANG = MADLAD_LANGUAGE_ALIASES["ru"]


_tokenizer: AutoTokenizer | None = None
_model: AutoModelForSeq2SeqLM | None = None
_device: torch.device | None = None
_allow_cpu_fallback = False


def _ensure_model_loaded(allow_cpu_fallback: bool = False) -> None:
    global _tokenizer, _model, _device, _allow_cpu_fallback
    if (
        _tokenizer is not None
        and _model is not None
        and _device is not None
        and _allow_cpu_fallback == allow_cpu_fallback
    ):
        return

    _tokenizer, _model, _device = load_model_components(
        MODEL_NAME,
        MODEL_CACHE_DIR,
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        use_safetensors=True,
        allow_cpu_fallback=allow_cpu_fallback,
    )
    _allow_cpu_fallback = allow_cpu_fallback


def unload_model() -> None:
    global _tokenizer, _model, _device, _allow_cpu_fallback
    _tokenizer = None
    _model = None
    _device = None
    _allow_cpu_fallback = False
    clear_gpu_memory()


def _resolve_forced_bos(target_lang: str) -> int | None:
    if _tokenizer is None:
        return None
    if hasattr(_tokenizer, "lang_code_to_id"):
        return _tokenizer.lang_code_to_id.get(target_lang)
    return None


def _prepare_input(text: str, target_lang: str) -> tuple[str, int | None]:
    forced_bos = _resolve_forced_bos(target_lang)
    if forced_bos is not None:
        return text, forced_bos
    return f">>{target_lang}<< {text}", None


def _token_count(text: str, target_lang: str) -> int:
    _ensure_model_loaded(_allow_cpu_fallback)
    assert _tokenizer is not None
    prepared, _ = _prepare_input(text, target_lang)
    encoded = _tokenizer(prepared, return_tensors="pt", truncation=False)
    return encoded["input_ids"].shape[-1]


def _translate_chunk(text: str, target_lang: str) -> str:
    _ensure_model_loaded(_allow_cpu_fallback)
    assert _tokenizer is not None and _model is not None and _device is not None

    prepared, forced_bos = _prepare_input(text, target_lang)
    inputs = _tokenizer(
        prepared,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_MODEL_INPUT,
    )
    inputs = {key: tensor.to(_device) for key, tensor in inputs.items()}

    gen_kwargs = {
        "max_length": min(MAX_OUTPUT_LENGTH, MAX_MODEL_INPUT * 2),
        "num_beams": 4,
        "early_stopping": True,
    }
    if forced_bos is not None:
        gen_kwargs["forced_bos_token_id"] = forced_bos

    with torch.no_grad():
        outputs = _model.generate(**inputs, **gen_kwargs)
    return _tokenizer.decode(outputs[0], skip_special_tokens=True)


def translate_text(text: str, target_lang: str) -> str:
    return translate_text_common(
        text=text,
        token_limit=MAX_MODEL_INPUT,
        token_counter=lambda value: _token_count(value, target_lang),
        chunk_translator=lambda value: _translate_chunk(value, target_lang),
    )


def ensure_translator_ready() -> None:
    _ensure_model_loaded(False)


class MadladTranslator:
    name = "madlad"

    @staticmethod
    def translate_batch(texts: list[str], options: TranslationOptions) -> list[str]:
        if not texts:
            return []
        resolve_lang(
            options.source_lang,
            default=DEFAULT_SOURCE_LANG,
            role="исходный",
            aliases=MADLAD_LANGUAGE_ALIASES,
            code_pattern=_MADLAD_CODE_PATTERN,
            model_label="MADLAD",
            hint="Используйте код ISO-639-3 (например, eng/rus) или алиас en/ru.",
        )
        target_lang = resolve_lang(
            options.target_lang,
            default=DEFAULT_TARGET_LANG,
            role="целевой",
            aliases=MADLAD_LANGUAGE_ALIASES,
            code_pattern=_MADLAD_CODE_PATTERN,
            model_label="MADLAD",
            hint="Используйте код ISO-639-3 (например, eng/rus) или алиас en/ru.",
        )
        allow_cpu_fallback = bool(options.allow_cpu_fallback)
        try:
            _ensure_model_loaded(allow_cpu_fallback)
        except TranslatorLoadError as exc:
            raise TranslationError(str(exc)) from exc
        result: list[str] = []
        for text in texts:
            try:
                result.append(translate_text(text, target_lang))
            except Exception as exc:
                raise TranslationError(f"Ошибка перевода через MADLAD: {exc}") from exc
        return result

    @staticmethod
    def unload() -> None:
        unload_model()


__all__ = ["translate_text", "ensure_translator_ready", "TranslatorLoadError", "MadladTranslator"]
