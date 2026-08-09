from __future__ import annotations

import re

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from sub_translate.models import TranslationOptions
from sub_translate.translators.base import TranslationError
from sub_translate.translators.registry import resolve_registered_model
from sub_translate.utils.huggingface import (
    ModelLoadOptions,
    ResolvedLocalModel,
    TranslatorLoadError,
    load_model_components,
)
from sub_translate.utils.local_utils import (
    MAX_MODEL_INPUT,
    MAX_OUTPUT_LENGTH,
    clear_gpu_memory,
    resolve_lang,
)
from sub_translate.utils.translation_utils import (
    translate_text as translate_text_common,
)

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
    def __init__(self) -> None:
        self._tokenizer: AutoTokenizer | None = None
        self._model: AutoModelForSeq2SeqLM | None = None
        self._device: torch.device | None = None
        self._forced_bos_token_id: int | None = None
        self._current_lang_pair: tuple[str, str] | None = None
        self._allow_cpu_fallback = False
        self._model_source: ResolvedLocalModel | None = None

    def ensure_loaded(
        self,
        source_lang: str,
        target_lang: str,
        options: TranslationOptions | None = None,
    ) -> None:
        load_options = options or TranslationOptions()
        model_source = resolve_registered_model("nllb-600m", load_options)
        allow_cpu_fallback = bool(load_options.allow_cpu_fallback)
        if (
            self._tokenizer is None
            or self._model is None
            or self._device is None
            or self._allow_cpu_fallback != allow_cpu_fallback
            or self._model_source != model_source
        ):
            if self._tokenizer is not None or self._model is not None or self._device is not None:
                self.unload()
            self._tokenizer, self._model, self._device = load_model_components(
                str(model_source.path),
                model_source.path,
                AutoModelForSeq2SeqLM,
                AutoTokenizer,
                ModelLoadOptions(
                    use_safetensors=False,
                    allow_cpu_fallback=allow_cpu_fallback,
                    enable_cpu_offload=allow_cpu_fallback,
                    revision=model_source.revision,
                    local_files_only=True,
                ),
            )
            self._allow_cpu_fallback = allow_cpu_fallback
            self._model_source = model_source
            self._forced_bos_token_id = None
            self._current_lang_pair = None

        if self._current_lang_pair != (source_lang, target_lang) or self._forced_bos_token_id is None:
            if hasattr(self._tokenizer, "src_lang"):
                self._tokenizer.src_lang = source_lang
            if hasattr(self._tokenizer, "tgt_lang"):
                self._tokenizer.tgt_lang = target_lang
            self._forced_bos_token_id = _resolve_forced_bos(self._tokenizer, target_lang)
            self._current_lang_pair = (source_lang, target_lang)

    def _token_count(self, text: str) -> int:
        assert self._tokenizer is not None
        encoded = self._tokenizer(text, return_tensors="pt", truncation=False)
        return encoded["input_ids"].shape[-1]

    def _translate_chunk(self, text: str) -> str:
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

    def translate_text(self, text: str) -> str:
        return translate_text_common(
            text=text,
            token_limit=MAX_MODEL_INPUT,
            token_counter=self._token_count,
            chunk_translator=self._translate_chunk,
        )

    def unload(self) -> None:
        self._tokenizer = None
        self._model = None
        self._device = None
        self._forced_bos_token_id = None
        self._current_lang_pair = None
        self._allow_cpu_fallback = False
        self._model_source = None
        clear_gpu_memory()


_ENGINE = _NllbEngine()


def _translate_batch(texts: list[str], options: TranslationOptions) -> list[str]:
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
        _ENGINE.ensure_loaded(source_lang, target_lang, options)
    except TranslatorLoadError as exc:
        raise TranslationError(str(exc)) from exc
    result: list[str] = []
    for text in texts:
        try:
            result.append(_ENGINE.translate_text(text))
        except torch.cuda.OutOfMemoryError:
            _ENGINE.unload()
            raise TranslationError(
                "NLLB 600M не хватило видеопамяти; модель выгружена, скрытый переход на CPU не выполнялся."
            ) from None
        except Exception as exc:
            raise TranslationError(f"Ошибка перевода через NLLB 600M: {exc}") from exc
    return result


class Nllb600MTranslator:
    name = "nllb-600m"

    @staticmethod
    def translate_batch(texts: list[str], options: TranslationOptions) -> list[str]:
        return _translate_batch(texts, options)

    @staticmethod
    def unload() -> None:
        _ENGINE.unload()


__all__ = [
    "Nllb600MTranslator",
    "TranslatorLoadError",
]
