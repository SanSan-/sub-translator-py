from __future__ import annotations

import re

import torch
from transformers import FSMTForConditionalGeneration, FSMTTokenizer

from sub_translate.constants import MODELS_DIR
from sub_translate.models import TranslationOptions
from sub_translate.translators.base import TranslationError
from sub_translate.utils.huggingface import TranslatorLoadError, load_model_components
from sub_translate.utils.local_utils import MAX_MODEL_INPUT, MAX_OUTPUT_LENGTH, clear_gpu_memory, resolve_lang
from sub_translate.utils.translation_utils import translate_text as translate_text_common

MODEL_NAME = "facebook/wmt19-en-ru"
MODEL_CACHE_DIR = MODELS_DIR

DEFAULT_SOURCE_LANG = "en"
DEFAULT_TARGET_LANG = "ru"

FSM_LANGUAGE_ALIASES = {
    "en": "en",
    "ru": "ru",
}

_FSM_CODE_PATTERN = re.compile(r"^[a-z]{2}$")


_tokenizer: FSMTTokenizer | None = None
_model: FSMTForConditionalGeneration | None = None
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
        FSMTForConditionalGeneration,
        FSMTTokenizer,
        use_safetensors=True,
        allow_quantization=False,
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


def _token_count(text: str) -> int:
    _ensure_model_loaded(_allow_cpu_fallback)
    assert _tokenizer is not None
    encoded = _tokenizer(text, return_tensors="pt", truncation=False)
    return encoded["input_ids"].shape[-1]


def _translate_chunk(text: str) -> str:
    _ensure_model_loaded(_allow_cpu_fallback)
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
    _ensure_model_loaded(False)


def translate_batch(texts: list[str], options: TranslationOptions) -> list[str]:
    if not texts:
        return []
    source_lang = resolve_lang(
        options.source_lang,
        default=DEFAULT_SOURCE_LANG,
        role="исходный",
        aliases=FSM_LANGUAGE_ALIASES,
        code_pattern=_FSM_CODE_PATTERN,
        model_label="FSM",
        hint="FSM поддерживает только en -> ru.",
    )
    target_lang = resolve_lang(
        options.target_lang,
        default=DEFAULT_TARGET_LANG,
        role="целевой",
        aliases=FSM_LANGUAGE_ALIASES,
        code_pattern=_FSM_CODE_PATTERN,
        model_label="FSM",
        hint="FSM поддерживает только en -> ru.",
    )
    if source_lang != DEFAULT_SOURCE_LANG or target_lang != DEFAULT_TARGET_LANG:
        raise TranslationError(
            "FSM-переводчик поддерживает только en -> ru "
            f"(получено {source_lang} -> {target_lang})."
        )
    allow_cpu_fallback = bool(options.allow_cpu_fallback)
    try:
        _ensure_model_loaded(allow_cpu_fallback)
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

    @staticmethod
    def unload() -> None:
        unload_model()


__all__ = ["translate_text", "ensure_translator_ready", "TranslatorLoadError", "FsmTranslator"]
