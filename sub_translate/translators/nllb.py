from __future__ import annotations

import re

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from sub_translate.constants import MODELS_DIR
from sub_translate.models import TranslationOptions
from sub_translate.translators.base import TranslationError
from sub_translate.translators.local_utils import resolve_device_and_quantization, resolve_lang
from sub_translate.utils.translation_utils import translate_text as translate_text_common

DEFAULT_MODEL_NAME = "facebook/nllb-200-3.3B"
LITE_MODEL_NAME = "facebook/nllb-200-distilled-1.3B"
MODEL_CACHE_DIR = MODELS_DIR
MAX_MODEL_INPUT = 512
MAX_OUTPUT_LENGTH = 1024

NLLB_LANGUAGE_ALIASES = {
    "en": "eng_Latn",
    "ru": "rus_Cyrl",
}

_NLLB_CODE_PATTERN = re.compile(r"^[a-z]{3}_[A-Za-z]+$")

DEFAULT_SOURCE_LANG = NLLB_LANGUAGE_ALIASES["en"]
DEFAULT_TARGET_LANG = NLLB_LANGUAGE_ALIASES["ru"]


class TranslatorLoadError(RuntimeError):
    """Исключение, возникающее при невозможности загрузить модель перевода."""


def _describe_incomplete_download() -> str:
    incomplete = list(MODEL_CACHE_DIR.rglob("*.incomplete"))
    safetensors = list(MODEL_CACHE_DIR.rglob("model.safetensors"))

    if incomplete:
        return (
            "Обнаружены файлы с расширением '.incomplete' в кеше модели. "
            "Дождитесь завершения загрузки или удалите неполный кеш и повторите попытку."
        )

    if not safetensors:
        return (
            "Файл 'model.safetensors' отсутствует в каталоге кеша. "
            "Вероятно, загрузка не завершилась - скачайте модель повторно перед запуском перевода."
        )

    return ""


def _handle_model_load_error(exc: Exception) -> None:
    hint = _describe_incomplete_download()
    if not hint and isinstance(exc, RuntimeError):
        message = str(exc)
        if "torch.load" in message and "v2.6" in message:
            hint = (
                "Библиотека transformers пытается открыть бинарные веса через `torch.load`, "
                "что требует torch>=2.6. Убедитесь, что в кеше есть safetensors-веса."
            )

    context = "Не удалось загрузить модель перевода."
    if hint:
        context = f"{context}\n{hint}"

    raise TranslatorLoadError(context) from exc


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

    def ensure_loaded(self, source_lang: str, target_lang: str) -> None:
        if self._tokenizer is None or self._model is None or self._device is None:
            MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            self._tokenizer = AutoTokenizer.from_pretrained(
                self._model_name,
                cache_dir=str(MODEL_CACHE_DIR),
                use_fast=False,
            )

            self._device, quantization_config = resolve_device_and_quantization()

            try:
                if quantization_config is not None:
                    self._model = AutoModelForSeq2SeqLM.from_pretrained(
                        self._model_name,
                        cache_dir=str(MODEL_CACHE_DIR),
                        quantization_config=quantization_config,
                        device_map="auto",
                        use_safetensors=True,
                    )
                else:
                    self._model = AutoModelForSeq2SeqLM.from_pretrained(
                        self._model_name,
                        cache_dir=str(MODEL_CACHE_DIR),
                        use_safetensors=True,
                    )
                    self._model.to(self._device)
            except Exception as exc:  # pragma: no cover - защита от частично загруженных весов
                self._tokenizer = None
                self._model = None
                self._device = None
                self._forced_bos_token_id = None
                self._current_lang_pair = None
                _handle_model_load_error(exc)

            self._model.eval()

        if self._tokenizer is None or self._model is None:
            raise TranslatorLoadError("Не удалось инициализировать модель перевода.")

        if self._current_lang_pair != (source_lang, target_lang) or self._forced_bos_token_id is None:
            if hasattr(self._tokenizer, "src_lang"):
                self._tokenizer.src_lang = source_lang
            if hasattr(self._tokenizer, "tgt_lang"):
                self._tokenizer.tgt_lang = target_lang
            self._forced_bos_token_id = _resolve_forced_bos(self._tokenizer, target_lang)
            self._current_lang_pair = (source_lang, target_lang)

    def _token_count(self, text: str, source_lang: str, target_lang: str) -> int:
        self.ensure_loaded(source_lang, target_lang)
        assert self._tokenizer is not None
        encoded = self._tokenizer(text, return_tensors="pt", truncation=False)
        return encoded["input_ids"].shape[-1]

    def _translate_chunk(self, text: str, source_lang: str, target_lang: str) -> str:
        self.ensure_loaded(source_lang, target_lang)
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
        self.ensure_loaded(DEFAULT_SOURCE_LANG, DEFAULT_TARGET_LANG)


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
    try:
        engine.ensure_loaded(source_lang, target_lang)
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


class NllbLiteTranslator:
    name = "nllb-lite"

    @staticmethod
    def translate_batch(texts: list[str], options: TranslationOptions) -> list[str]:
        return _translate_batch(texts, options, _LITE_ENGINE, "NLLB Lite")


__all__ = [
    "translate_text",
    "ensure_translator_ready",
    "TranslatorLoadError",
    "NllbTranslator",
    "NllbLiteTranslator",
]
