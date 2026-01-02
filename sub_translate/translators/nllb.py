from __future__ import annotations

import re

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from sub_translate.constants import MODELS_DIR
from sub_translate.dictionaries.languages import get_code
from sub_translate.models import TranslationOptions
from sub_translate.translators.base import TranslationError
from sub_translate.utils.translation_utils import translate_text as translate_text_common

try:
    from transformers import BitsAndBytesConfig
except ImportError:  # pragma: no cover - опциональная зависимость
    BitsAndBytesConfig = None  # type: ignore[misc,assignment]

MODEL_NAME = "facebook/nllb-200-distilled-1.3B"
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


_tokenizer: AutoTokenizer | None = None
_model: AutoModelForSeq2SeqLM | None = None
_device: torch.device | None = None
_forced_bos_token_id: int | None = None
_current_lang_pair: tuple[str, str] | None = None


def _normalize_lang(value: str | None) -> str | None:
    if not value:
        return None
    value = value.strip()
    if not value:
        return None
    if value in NLLB_LANGUAGE_ALIASES.values() or _NLLB_CODE_PATTERN.match(value):
        return value
    lowered = value.lower()
    if lowered == "auto":
        return "auto"
    if "-" in lowered:
        lowered = lowered.split("-", 1)[0]
    if "_" in lowered:
        lowered = lowered.split("_", 1)[0]
    return get_code(lowered) or lowered


def _resolve_nllb_lang(value: str | None, *, default: str, role: str) -> str:
    normalized = _normalize_lang(value)
    if not normalized or normalized == "auto":
        return default
    if normalized in NLLB_LANGUAGE_ALIASES:
        return NLLB_LANGUAGE_ALIASES[normalized]
    if normalized in NLLB_LANGUAGE_ALIASES.values() or _NLLB_CODE_PATTERN.match(normalized):
        return normalized
    raise TranslationError(
        f"Для NLLB не поддерживается {role} язык '{value}'. "
        "Используйте код NLLB (например, eng_Latn) или алиас en/ru."
    )


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


def _ensure_model_loaded(source_lang: str, target_lang: str) -> None:
    global _tokenizer, _model, _device, _forced_bos_token_id, _current_lang_pair
    if _tokenizer is None or _model is None or _device is None:
        MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            cache_dir=str(MODEL_CACHE_DIR),
            use_fast=False,
        )

        device_is_gpu = torch.cuda.is_available()
        quantization_config = None

        if device_is_gpu and BitsAndBytesConfig is not None:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        elif device_is_gpu:
            print("Библиотека bitsandbytes недоступна - загружаем модель без 8-битной квантовки.", flush=True)

        if device_is_gpu:
            _device = torch.device("cuda")
            print("Модель переводчика загружена в видеопамять (GPU).", flush=True)
        else:
            _device = torch.device("cpu")
            print("Модель переводчика загружена в оперативную память (CPU).", flush=True)

        try:
            if quantization_config is not None:
                _model = AutoModelForSeq2SeqLM.from_pretrained(
                    MODEL_NAME,
                    cache_dir=str(MODEL_CACHE_DIR),
                    quantization_config=quantization_config,
                    device_map="auto",
                    use_safetensors=True,
                )
            else:
                _model = AutoModelForSeq2SeqLM.from_pretrained(
                    MODEL_NAME,
                    cache_dir=str(MODEL_CACHE_DIR),
                    use_safetensors=True,
                )
                _model.to(_device)
        except Exception as exc:  # pragma: no cover - защита от частично загруженных весов
            _tokenizer = None
            _model = None
            _device = None
            _forced_bos_token_id = None
            _current_lang_pair = None
            _handle_model_load_error(exc)

        _model.eval()

    if _tokenizer is None or _model is None:
        raise TranslatorLoadError("Не удалось инициализировать модель перевода.")

    if _current_lang_pair != (source_lang, target_lang) or _forced_bos_token_id is None:
        _tokenizer.src_lang = source_lang
        _tokenizer.tgt_lang = target_lang
        _forced_bos_token_id = _resolve_forced_bos(_tokenizer, target_lang)
        _current_lang_pair = (source_lang, target_lang)


def _token_count(text: str, source_lang: str, target_lang: str) -> int:
    _ensure_model_loaded(source_lang, target_lang)
    assert _tokenizer is not None
    encoded = _tokenizer(text, return_tensors="pt", truncation=False)
    return encoded["input_ids"].shape[-1]


def _translate_chunk(text: str, source_lang: str, target_lang: str) -> str:
    _ensure_model_loaded(source_lang, target_lang)
    assert (
        _tokenizer is not None
        and _model is not None
        and _device is not None
        and _forced_bos_token_id is not None
    )

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
            forced_bos_token_id=_forced_bos_token_id,
        )
    return _tokenizer.decode(outputs[0], skip_special_tokens=True)


def translate_text(text: str, source_lang: str, target_lang: str) -> str:
    return translate_text_common(
        text=text,
        token_limit=MAX_MODEL_INPUT,
        token_counter=lambda value: _token_count(value, source_lang, target_lang),
        chunk_translator=lambda value: _translate_chunk(value, source_lang, target_lang),
    )


def ensure_translator_ready() -> None:
    _ensure_model_loaded(DEFAULT_SOURCE_LANG, DEFAULT_TARGET_LANG)


class NllbTranslator:
    name = "nllb"

    @staticmethod
    def translate_batch(texts: list[str], options: TranslationOptions) -> list[str]:
        if not texts:
            return []
        source_lang = _resolve_nllb_lang(options.source_lang, default=DEFAULT_SOURCE_LANG, role="исходный")
        target_lang = _resolve_nllb_lang(options.target_lang, default=DEFAULT_TARGET_LANG, role="целевой")
        try:
            _ensure_model_loaded(source_lang, target_lang)
        except TranslatorLoadError as exc:
            raise TranslationError(str(exc)) from exc
        result: list[str] = []
        for text in texts:
            try:
                result.append(translate_text(text, source_lang, target_lang))
            except Exception as exc:
                raise TranslationError(f"Ошибка перевода через NLLB: {exc}") from exc
        return result


__all__ = ["translate_text", "ensure_translator_ready", "TranslatorLoadError", "NllbTranslator"]
