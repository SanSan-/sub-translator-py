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

DEFAULT_MODEL_NAME = "facebook/nllb-200-3.3B"
LITE_MODEL_NAME = "facebook/nllb-200-distilled-1.3B"
MODEL_CACHE_DIR = MODELS_DIR

NLLB_LANGUAGE_ALIASES = {
    "en": "eng_Latn",
    "ru": "rus_Cyrl",
}

_NLLB_CODE_PATTERN = re.compile(r"^[a-z]{3}_[A-Za-z]+$")

DEFAULT_SOURCE_LANG = NLLB_LANGUAGE_ALIASES["en"]
DEFAULT_TARGET_LANG = NLLB_LANGUAGE_ALIASES["ru"]


def _resolve_forced_bos(tokenizer: AutoTokenizer, target_lang: str) -> int:
    if hasattr(tokenizer, "lang_code_to_id"):
        try:
            return tokenizer.lang_code_to_id[target_lang]
        except KeyError as exc:  # pragma: no cover
            raise ValueError(f"Target language {target_lang!r} is not supported by the tokenizer.") from exc
    token_id = tokenizer.convert_tokens_to_ids(target_lang)
    if token_id == tokenizer.unk_token_id:
        raise ValueError(f"Target language {target_lang!r} is not supported by the tokenizer.")
    return token_id


class _NllbEngine:
    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._tokenizer: AutoTokenizer | None = None
        self._model: AutoModelForSeq2SeqLM | None = None
        self._device: torch.device | None = None
        self._forced_bos_token_id: int | None = None
        self._current_lang_pair: tuple[str, str] | None = None
        self._allow_cpu_fallback = False

    def ensure_loaded(self, source_lang: str, target_lang: str, *, allow_cpu_fallback: bool = False) -> None:
        if (
            self._tokenizer is None
            or self._model is None
            or self._device is None
            or self._allow_cpu_fallback != allow_cpu_fallback
        ):
            self._tokenizer, self._model, self._device = load_model_components(
                self._model_name,
                MODEL_CACHE_DIR,
                AutoModelForSeq2SeqLM,
                AutoTokenizer,
                use_safetensors=True,
                processor_kwargs={"use_fast": False},
                allow_cpu_fallback=allow_cpu_fallback,
            )
            self._allow_cpu_fallback = allow_cpu_fallback
            self._forced_bos_token_id = None
            self._current_lang_pair = None

        if self._current_lang_pair != (source_lang, target_lang) or self._forced_bos_token_id is None:
            if hasattr(self._tokenizer, "src_lang"):
                self._tokenizer.src_lang = source_lang
            if hasattr(self._tokenizer, "tgt_lang"):
                self._tokenizer.tgt_lang = target_lang
            self._forced_bos_token_id = _resolve_forced_bos(self._tokenizer, target_lang)
            self._current_lang_pair = (source_lang, target_lang)

    def _token_count(self, text: str, source_lang: str, target_lang: str) -> int:
        self.ensure_loaded(source_lang, target_lang, allow_cpu_fallback=self._allow_cpu_fallback)
        assert self._tokenizer is not None
        encoded = self._tokenizer(text, return_tensors="pt", truncation=False)
        return encoded["input_ids"].shape[-1]

    def _translate_chunk(self, text: str, source_lang: str, target_lang: str) -> str:
        self.ensure_loaded(source_lang, target_lang, allow_cpu_fallback=self._allow_cpu_fallback)
        assert (
            self._tokenizer is not None
            and self._model is not None
            and self._device is not None
            and self._forced_bos_token_id is not None
        )

        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_MODEL_INPUT,
        )
        inputs = {key: tensor.to(self._device) for key, tensor in inputs.items()}

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_length=min(MAX_OUTPUT_LENGTH, MAX_MODEL_INPUT * 2),
                num_beams=4,
                early_stopping=True,
                forced_bos_token_id=self._forced_bos_token_id,
            )
        return self._tokenizer.decode(outputs[0], skip_special_tokens=True)

    def translate_text(self, text: str, source_lang: str, target_lang: str) -> str:
        return translate_text_common(
            text=text,
            token_limit=MAX_MODEL_INPUT,
            token_counter=lambda value: self._token_count(value, source_lang, target_lang),
            chunk_translator=lambda value: self._translate_chunk(value, source_lang, target_lang),
        )

    def ensure_ready(self) -> None:
        self.ensure_loaded(DEFAULT_SOURCE_LANG, DEFAULT_TARGET_LANG, allow_cpu_fallback=False)

    def unload(self) -> None:
        self._tokenizer = None
        self._model = None
        self._device = None
        self._forced_bos_token_id = None
        self._current_lang_pair = None
        self._allow_cpu_fallback = False
        clear_gpu_memory()


_DEFAULT_ENGINE = _NllbEngine(DEFAULT_MODEL_NAME)
_LITE_ENGINE = _NllbEngine(LITE_MODEL_NAME)


def translate_text(text: str, source_lang: str, target_lang: str) -> str:
    return _DEFAULT_ENGINE.translate_text(text, source_lang, target_lang)


def ensure_translator_ready() -> None:
    _DEFAULT_ENGINE.ensure_ready()


def _translate_batch(texts: list[str], options: TranslationOptions, engine: _NllbEngine, label: str) -> list[str]:
    if not texts:
        return []
    source_lang = resolve_lang(
        options.source_lang,
        default=DEFAULT_SOURCE_LANG,
        role="исходный",
        aliases=NLLB_LANGUAGE_ALIASES,
        code_pattern=_NLLB_CODE_PATTERN,
        model_label="NLLB",
        hint="Используйте код NLLB (например, eng_Latn) или алиас en/ru.",
    )
    target_lang = resolve_lang(
        options.target_lang,
        default=DEFAULT_TARGET_LANG,
        role="целевой",
        aliases=NLLB_LANGUAGE_ALIASES,
        code_pattern=_NLLB_CODE_PATTERN,
        model_label="NLLB",
        hint="Используйте код NLLB (например, eng_Latn) или алиас en/ru.",
    )
    allow_cpu_fallback = bool(options.allow_cpu_fallback)
    try:
        engine.ensure_loaded(source_lang, target_lang, allow_cpu_fallback=allow_cpu_fallback)
    except TranslatorLoadError as exc:
        raise TranslationError(str(exc)) from exc
    result: list[str] = []
    for text in texts:
        try:
            result.append(engine.translate_text(text, source_lang, target_lang))
        except Exception as exc:
            raise TranslationError(f"Ошибка перевода через {label}: {exc}") from exc
    return result


class NllbTranslator:
    name = "nllb"

    @staticmethod
    def translate_batch(texts: list[str], options: TranslationOptions) -> list[str]:
        return _translate_batch(texts, options, _DEFAULT_ENGINE, "NLLB")

    @staticmethod
    def unload() -> None:
        _DEFAULT_ENGINE.unload()


class NllbLiteTranslator:
    name = "nllb-lite"

    @staticmethod
    def translate_batch(texts: list[str], options: TranslationOptions) -> list[str]:
        return _translate_batch(texts, options, _LITE_ENGINE, "NLLB Lite")

    @staticmethod
    def unload() -> None:
        _LITE_ENGINE.unload()


__all__ = [
    "translate_text",
    "ensure_translator_ready",
    "TranslatorLoadError",
    "NllbTranslator",
    "NllbLiteTranslator",
]
