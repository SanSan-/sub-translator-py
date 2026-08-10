"""Изолированная среда Seed-X-PPO-7B с NF4-квантованием при загрузке."""

from __future__ import annotations

import gc
import importlib.metadata
import json
import os
import re
import sys
from collections import Counter
from collections.abc import Mapping
from contextlib import redirect_stdout
from pathlib import Path
from types import MappingProxyType
from typing import Any

from sub_translate.translators.registry import SEEDX_MODEL_ID, SEEDX_MODEL_REVISION
from sub_translate.workers.common import (
    PublicWorkerError,
    model_uses_only_device,
    run_worker_loop,
    validate_model_content_fingerprint,
)

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

REQUIRED_RUNTIME_VERSIONS = MappingProxyType(
    {
        "accelerate": "1.14.0",
        "bitsandbytes": "0.50.0",
        "huggingface_hub": "1.27.0",
        "safetensors": "0.8.0",
        "tokenizers": "0.22.2",
        "torch": "2.13.0+cu130",
        "transformers": "5.14.1",
    }
)
MAX_BATCH_SIZE = 1
MAX_INPUT_TOKENS = 4_096
MAX_NEW_TOKENS = 512
NUM_BEAMS = 4
NO_REPEAT_NGRAM_SIZE = 3
DEVICE = "cuda:0"
SUPPORTED_DIRECTIONS = (("en", "ru"), ("ru", "en"))
LANGUAGE_NAMES = {"en": "English", "ru": "Russian"}
_HTML_BREAK_TAG_NAMES = frozenset(
    {
        "article",
        "aside",
        "blockquote",
        "body",
        "br",
        "div",
        "figcaption",
        "figure",
        "footer",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "head",
        "header",
        "hr",
        "html",
        "li",
        "main",
        "ol",
        "p",
        "pre",
        "section",
        "table",
        "tbody",
        "td",
        "tfoot",
        "th",
        "thead",
        "tr",
        "ul",
    }
)
_HTML_TAG_NAME_PATTERN = r"[A-Za-z][A-Za-z0-9:-]*"
_HTML_ATTRIBUTES_PATTERN = r"(?:[^<>\"']++|\"[^\"]*+\"|'[^']*+')*+"
_HTML_TAG_PATTERN = re.compile(
    rf"<\s*(?P<closing>/?)\s*(?P<name>{_HTML_TAG_NAME_PATTERN})(?=[\s/>]){_HTML_ATTRIBUTES_PATTERN}>",
    re.IGNORECASE,
)
_MALFORMED_HTML_PREFIX_PATTERN = re.compile(
    rf"(?m)^(?P<indent>[ \t]*)(?P<prefix><[ \t]*)"
    rf"(?=<\s*(?P<tag_closing>/?)\s*(?P<tag_name>{_HTML_TAG_NAME_PATTERN})(?=[\s/>]))",
    re.IGNORECASE,
)
_URL_PATTERN = re.compile(r"(?:(?:https?|ftp)://|www\.)[^\s<>\"']+", re.IGNORECASE)
_URL_TRAILING_PUNCTUATION = ".,!?;:)]}»”"
_SOURCE_LABEL_PATTERN = re.compile(r"(?:источник|source)\s*:", re.IGNORECASE)
_SOURCE_LABELS = ("источник", "source")
_REQUIRED_FILES = (
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "tokenizer.json",
)


class SeedXRuntimeError(PublicWorkerError):
    """Безопасная ошибка среды Seed-X."""


class SeedXOutOfMemoryError(PublicWorkerError):
    """Seed-X не поместилась в доступную видеопамять."""


class SeedXWorkerRuntime:
    """Хранит одну квантованную модель и не допускает переход на CPU или сеть."""

    def __init__(self) -> None:
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._torch: Any | None = None
        self._load_key: tuple[str, str, str | None] | None = None
        self._modules: tuple[Any, Any, Any, Any] | None = None

    def preflight(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Проверяет среду, CUDA и локальные файлы без загрузки весов."""
        model_path, revision, content_fingerprint = _validate_model_request(payload)
        torch, _tokenizer_class, _model_class, _quantization_class = self._runtime_modules()
        if not bool(torch.cuda.is_available()):
            raise SeedXRuntimeError("Для Seed-X требуется CUDA; переход на CPU отключён.")
        _require_bfloat16_cuda(torch)
        _validate_base_model_config(model_path)
        return _runtime_signature(model_path, revision, content_fingerprint, loaded=self._model is not None)

    def translate(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Переводит ровно один элемент, ограничивая пиковую видеопамять."""
        model_path, revision, content_fingerprint = _validate_model_request(payload)
        source_lang, target_lang = _validate_direction(payload)
        texts = _validate_texts(payload.get("texts"))
        if not texts:
            return {
                "translations": [],
                "runtime": _runtime_signature(
                    model_path,
                    revision,
                    content_fingerprint,
                    loaded=self._model is not None,
                ),
            }
        if not texts[0]:
            return {
                "translations": [""],
                "runtime": _runtime_signature(
                    model_path,
                    revision,
                    content_fingerprint,
                    loaded=self._model is not None,
                ),
            }

        self._ensure_loaded(model_path, revision, content_fingerprint)
        translation = self._translate_one(texts[0], source_lang, target_lang)
        return {
            "translations": [translation],
            "runtime": _runtime_signature(model_path, revision, content_fingerprint, loaded=True),
        }

    def unload(self) -> dict[str, Any]:
        """Освобождает веса и кеш CUDA внутри изолированного процесса."""
        self._model = None
        self._tokenizer = None
        self._load_key = None
        gc.collect()
        if self._torch is not None and bool(self._torch.cuda.is_available()):
            self._torch.cuda.empty_cache()
            self._torch.cuda.ipc_collect()
        return {"unloaded": True}

    def _ensure_loaded(
        self,
        model_path: Path,
        revision: str,
        content_fingerprint: str | None = None,
    ) -> None:
        load_key = (str(model_path).casefold(), revision, content_fingerprint)
        if self._model is not None and self._tokenizer is not None and self._load_key == load_key:
            return
        if self._model is not None or self._tokenizer is not None:
            self.unload()
        _validate_base_model_config(model_path)
        torch, tokenizer_class, model_class, quantization_class = self._runtime_modules()
        if not bool(torch.cuda.is_available()):
            raise SeedXRuntimeError("Для Seed-X требуется CUDA; переход на CPU отключён.")
        _require_bfloat16_cuda(torch)
        try:
            quantization = _quantization_config(torch, quantization_class)
            with redirect_stdout(sys.stderr):
                tokenizer = tokenizer_class.from_pretrained(
                    str(model_path),
                    local_files_only=True,
                    trust_remote_code=False,
                )
                model = model_class.from_pretrained(
                    str(model_path),
                    local_files_only=True,
                    trust_remote_code=False,
                    use_safetensors=True,
                    quantization_config=quantization,
                    device_map={"": 0},
                    dtype=torch.bfloat16,
                    low_cpu_mem_usage=True,
                )
                model.eval()
            _configure_eos_token(tokenizer, model)
            _require_nf4_quantization(model)
            _require_cuda_only(model)
        except torch.cuda.OutOfMemoryError as exc:
            self.unload()
            raise SeedXOutOfMemoryError("Недостаточно видеопамяти для Seed-X NF4; переход на CPU отключён.") from exc
        except PublicWorkerError:
            self.unload()
            raise
        except Exception as exc:
            self.unload()
            raise SeedXRuntimeError("Не удалось загрузить локальную Seed-X NF4 в CUDA.") from exc
        self._tokenizer = tokenizer
        self._model = model
        self._torch = torch
        self._load_key = load_key

    def _translate_one(self, text: str, source_lang: str, target_lang: str) -> str:
        assert self._tokenizer is not None and self._model is not None and self._torch is not None
        prompt = build_translation_prompt(text, source_lang, target_lang)
        try:
            with redirect_stdout(sys.stderr):
                encoded = self._tokenizer(
                    prompt,
                    add_special_tokens=True,
                    return_tensors="pt",
                    truncation=False,
                )
            input_ids = encoded.get("input_ids")
            if input_ids is None:
                raise SeedXRuntimeError("Токенизатор Seed-X не вернул input_ids.")
            input_length = int(input_ids.shape[-1])
            if input_length > MAX_INPUT_TOKENS:
                raise SeedXRuntimeError(
                    f"Запрос Seed-X содержит {input_length} токенов при пределе {MAX_INPUT_TOKENS}."
                )
            eos_token_id = _eos_token_id(self._tokenizer)
            cuda_inputs = {key: value.to(DEVICE) if hasattr(value, "to") else value for key, value in encoded.items()}
            with self._torch.inference_mode(), redirect_stdout(sys.stderr):
                generated = self._model.generate(
                    **cuda_inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    num_beams=NUM_BEAMS,
                    no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
                    use_cache=True,
                    eos_token_id=eos_token_id,
                    pad_token_id=eos_token_id,
                )
            result = self._tokenizer.decode(
                _completed_generated_tokens(
                    generated[0][input_length:],
                    eos_token_id,
                ),
                skip_special_tokens=True,
            ).strip()
        except self._torch.cuda.OutOfMemoryError as exc:
            self.unload()
            raise SeedXOutOfMemoryError(
                "Недостаточно видеопамяти для перевода Seed-X; процесс будет перезапущен."
            ) from exc
        except PublicWorkerError:
            raise
        except Exception as exc:
            self.unload()
            raise SeedXRuntimeError("Seed-X не выполнила локальный перевод.") from exc
        return _sanitize_translation_result(result, prompt, target_lang, text)

    def _runtime_modules(self) -> tuple[Any, Any, Any, Any]:
        if self._modules is None:
            self._modules = _import_runtime()
            self._torch = self._modules[0]
        return self._modules


def build_translation_prompt(text: str, source_lang: str, target_lang: str) -> str:
    """Строит официальный одноходовый запрос с обязательной целевой меткой."""
    if (source_lang, target_lang) not in SUPPORTED_DIRECTIONS:
        raise ValueError("Seed-X поддерживает только направления en -> ru и ru -> en.")
    source_name = LANGUAGE_NAMES[source_lang]
    target_name = LANGUAGE_NAMES[target_lang]
    return f"Translate the following {source_name} sentence into {target_name}:\n{text} <{target_lang}>"


def _validate_model_request(payload: Mapping[str, Any]) -> tuple[Path, str, str | None]:
    model_id = str(payload.get("model_id") or "").strip()
    if model_id != SEEDX_MODEL_ID:
        raise SeedXRuntimeError("Запрос относится не к закреплённой модели Seed-X.")
    revision = str(payload.get("model_revision") or "").strip().lower()
    if revision != SEEDX_MODEL_REVISION:
        raise SeedXRuntimeError("Seed-X требует закреплённую ревизию модели.")
    raw_path = str(payload.get("model_path") or "").strip()
    if not raw_path:
        raise SeedXRuntimeError("Не указан локальный каталог Seed-X.")
    model_path = Path(raw_path).expanduser().resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(f"Каталог Seed-X не найден: {model_path}")
    missing = [name for name in _REQUIRED_FILES if not (model_path / name).is_file()]
    if missing:
        raise SeedXRuntimeError(f"Каталог Seed-X неполон; отсутствуют файлы: {', '.join(missing)}.")
    content_fingerprint = validate_model_content_fingerprint(payload.get("model_content_fingerprint"))
    return model_path, revision, content_fingerprint


def _validate_base_model_config(model_path: Path) -> None:
    config_path = model_path / "config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(config, Mapping):
            raise SeedXRuntimeError("Корнем config.json Seed-X должен быть JSON-объект.")
        architectures = config.get("architectures")
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SeedXRuntimeError("Повреждён config.json локальной Seed-X.") from exc
    if architectures != ["MistralForCausalLM"] or config.get("model_type") != "mistral":
        raise SeedXRuntimeError("Seed-X должна использовать закреплённую архитектуру MistralForCausalLM.")
    if str(config.get("torch_dtype") or "").casefold() != "bfloat16":
        raise SeedXRuntimeError("Seed-X требует вычисления BF16 из закреплённой конфигурации.")
    if (
        config.get("hidden_size") != 4_096
        or config.get("num_hidden_layers") != 32
        or config.get("vocab_size") != 65_269
        or config.get("eos_token_id") != 2
    ):
        raise SeedXRuntimeError("config.json не соответствует закреплённой базовой Seed-X PPO 7B.")


def _quantization_config(torch: Any, quantization_class: Any) -> Any:
    return quantization_class(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def _require_bfloat16_cuda(torch: Any) -> None:
    checker = getattr(torch.cuda, "is_bf16_supported", None)
    if not callable(checker) or not bool(checker()):
        raise SeedXRuntimeError("Для Seed-X требуется CUDA с поддержкой BF16; переход на FP16 отключён.")


def _eos_token_id(tokenizer: Any) -> int:
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if not isinstance(eos_token_id, int) or isinstance(eos_token_id, bool) or eos_token_id < 0:
        raise SeedXRuntimeError("Токенизатор Seed-X не сообщает корректный токен завершения.")
    return eos_token_id


def _configure_eos_token(tokenizer: Any, model: Any) -> None:
    model_eos_token_id = getattr(getattr(model, "config", None), "eos_token_id", None)
    if not isinstance(model_eos_token_id, int) or isinstance(model_eos_token_id, bool) or model_eos_token_id < 0:
        raise SeedXRuntimeError("Seed-X не сообщает корректный токен завершения.")
    tokenizer_eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if isinstance(tokenizer_eos_token_id, int) and not isinstance(tokenizer_eos_token_id, bool):
        if tokenizer_eos_token_id != model_eos_token_id:
            raise SeedXRuntimeError("Токен завершения Seed-X не совпадает с закреплённой конфигурацией.")
        return
    converter = getattr(tokenizer, "convert_ids_to_tokens", None)
    if not callable(converter):
        raise SeedXRuntimeError("Seed-X не сообщает корректный токен завершения.")
    eos_token = converter(model_eos_token_id)
    if not isinstance(eos_token, str) or not eos_token:
        raise SeedXRuntimeError("Seed-X не удалось восстановить токен завершения из конфигурации.")
    tokenizer.eos_token = eos_token
    if _eos_token_id(tokenizer) != model_eos_token_id:
        raise SeedXRuntimeError("Токен завершения Seed-X не совпадает с закреплённой конфигурацией.")


def _completed_generated_tokens(tokens: Any, eos_token_id: int) -> Any:
    raw_ids = tokens.tolist() if callable(getattr(tokens, "tolist", None)) else list(tokens)
    if not raw_ids or not all(isinstance(token_id, int) for token_id in raw_ids):
        raise SeedXRuntimeError("Seed-X не вернула корректную последовательность токенов.")
    if eos_token_id not in raw_ids:
        raise SeedXRuntimeError("Seed-X не завершила перевод токеном EOS; незавершённый результат отклонён.")
    return tokens


def _html_tag_key(match: re.Match[str]) -> tuple[str, bool, str]:
    return (
        match.group("name").casefold(),
        bool(match.group("closing")),
        match.group(0).casefold(),
    )


def _malformed_prefix_key(match: re.Match[str]) -> tuple[str, str, bool]:
    return (
        match.group("prefix").casefold(),
        match.group("tag_name").casefold(),
        bool(match.group("tag_closing")),
    )


def _strip_generated_html(text: str, source_text: str) -> tuple[str, bool]:
    source_prefixes = Counter(
        _malformed_prefix_key(match) for match in _MALFORMED_HTML_PREFIX_PATTERN.finditer(source_text)
    )
    source_tags = Counter(_html_tag_key(match) for match in _HTML_TAG_PATTERN.finditer(source_text))
    result_tags = Counter(_html_tag_key(match) for match in _HTML_TAG_PATTERN.finditer(text))
    keep_quotas = Counter({key: min(count, result_tags[key]) for key, count in source_tags.items()})
    preserved_openings = Counter(
        {
            name: sum(
                count
                for (tag_name, closing, _signature), count in keep_quotas.items()
                if tag_name == name and not closing
            )
            for name, closing, _signature in source_tags
            if not closing
        }
    )
    for key in tuple(keep_quotas):
        name, closing, _signature = key
        source_has_opening = any(tag_name == name and not is_closing for tag_name, is_closing, _ in source_tags)
        if closing and source_has_opening:
            keep_quotas[key] = min(keep_quotas[key], preserved_openings[name])
    remaining = result_tags.copy()
    kept = Counter()
    changed = False

    def strip_malformed_prefix(match: re.Match[str]) -> str:
        nonlocal changed
        key = _malformed_prefix_key(match)
        if source_prefixes[key]:
            source_prefixes[key] -= 1
            return match.group(0)
        changed = True
        return match.group("indent")

    def strip_tag(match: re.Match[str]) -> str:
        nonlocal changed
        key = _html_tag_key(match)
        name, closing, _signature = key
        remaining[key] -= 1
        keep = remaining[key] < keep_quotas[key] if closing else kept[key] < keep_quotas[key]
        if keep:
            kept[key] += 1
            return match.group(0)
        changed = True
        return "\n" if name in _HTML_BREAK_TAG_NAMES else ""

    without_prefixes = _MALFORMED_HTML_PREFIX_PATTERN.sub(strip_malformed_prefix, text)
    return _HTML_TAG_PATTERN.sub(strip_tag, without_prefixes), changed


def _url_key(value: str) -> str:
    return value.rstrip(_URL_TRAILING_PUNCTUATION)


def _strip_foreign_urls(text: str, source_text: str) -> tuple[str, bool]:
    source_urls = Counter(_url_key(match.group(0)) for match in _URL_PATTERN.finditer(source_text))
    changed = False

    def strip_url(match: re.Match[str]) -> str:
        nonlocal changed
        key = _url_key(match.group(0))
        if source_urls[key]:
            source_urls[key] -= 1
            return match.group(0)
        changed = True
        return ""

    cleaned = _URL_PATTERN.sub(strip_url, text)
    source_has_label = bool(_SOURCE_LABEL_PATTERN.search(source_text))
    if changed and not source_has_label:
        cleaned = _strip_foreign_source_tail(cleaned)
    return cleaned, changed


def _strip_foreign_source_tail(text: str) -> str:
    stripped = text.rstrip()
    if not stripped.endswith(":"):
        return text
    before_colon = stripped[:-1].rstrip()
    folded = before_colon.casefold()
    for label in _SOURCE_LABELS:
        if not folded.endswith(label):
            continue
        label_start = len(before_colon) - len(label)
        if label_start == 0 or before_colon[label_start - 1].isspace():
            return before_colon[:label_start].rstrip()
    return text


def _normalize_cleaned_lines(text: str, source_text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    source_line_count = max(1, len(source_text.splitlines()))
    if len(lines) > source_line_count:
        tail = " ".join(lines[source_line_count - 1 :])
        lines = [*lines[: source_line_count - 1], tail]
    return "\n".join(lines).strip()


def _sanitize_translation_result(
    result: str,
    prompt: str,
    target_lang: str,
    source_text: str,
) -> str:
    cleaned = result.strip()
    if cleaned.startswith(prompt):
        cleaned = cleaned[len(prompt) :].strip()
    target_tag_pattern = re.compile(rf"<\s*{re.escape(target_lang)}\s*>", re.IGNORECASE)
    leading_target_tag = target_tag_pattern.match(cleaned)
    if leading_target_tag is not None:
        cleaned = cleaned[leading_target_tag.end() :].strip()
    target_tag_matches = tuple(target_tag_pattern.finditer(cleaned))
    if target_tag_matches and target_tag_matches[-1].end() == len(cleaned):
        cleaned = cleaned[: target_tag_matches[-1].start()].strip()
    if not cleaned:
        raise SeedXRuntimeError("Seed-X вернула пустой перевод.")
    if prompt in cleaned or target_tag_pattern.search(cleaned) is not None:
        raise SeedXRuntimeError("Seed-X вернула часть исходного запроса вместо перевода.")
    cleaned, html_changed = _strip_generated_html(cleaned, source_text)
    cleaned, url_changed = _strip_foreign_urls(cleaned, source_text)
    if html_changed or url_changed:
        cleaned = _normalize_cleaned_lines(cleaned, source_text)
    if not cleaned:
        raise SeedXRuntimeError("Seed-X вернула пустой перевод.")
    return cleaned


def _validate_direction(payload: Mapping[str, Any]) -> tuple[str, str]:
    source_lang = str(payload.get("source_lang") or "").strip().casefold()
    target_lang = str(payload.get("target_lang") or "").strip().casefold()
    if (source_lang, target_lang) not in SUPPORTED_DIRECTIONS:
        raise SeedXRuntimeError("Seed-X поддерживает только направления en -> ru и ru -> en.")
    return source_lang, target_lang


def _validate_texts(raw_texts: Any) -> list[str]:
    if not isinstance(raw_texts, list):
        raise SeedXRuntimeError("Поле texts должно быть списком строк.")
    if len(raw_texts) > MAX_BATCH_SIZE:
        raise SeedXRuntimeError(f"Пачка Seed-X содержит {len(raw_texts)} элементов при пределе {MAX_BATCH_SIZE}.")
    if not all(isinstance(text, str) for text in raw_texts):
        raise SeedXRuntimeError("Пачка Seed-X содержит значение, которое не является строкой.")
    return list(raw_texts)


def _require_nf4_quantization(model: Any) -> None:
    quantizer = getattr(model, "hf_quantizer", None)
    quantization = getattr(quantizer, "quantization_config", None)
    loaded = (
        getattr(model, "is_loaded_in_4bit", False) is True and getattr(quantization, "load_in_4bit", False) is True
    )
    quant_type = str(getattr(quantization, "bnb_4bit_quant_type", "")).casefold()
    double_quant = getattr(quantization, "bnb_4bit_use_double_quant", False) is True
    compute_dtype = str(getattr(quantization, "bnb_4bit_compute_dtype", "")).casefold()
    if not (loaded and quant_type == "nf4" and double_quant and compute_dtype in {"bfloat16", "torch.bfloat16"}):
        raise SeedXRuntimeError("Seed-X загружена без обязательного NF4 double/BF16-квантования.")


def _require_cuda_only(model: Any) -> None:
    if not model_uses_only_device(model, _is_cuda_device):
        raise SeedXRuntimeError("Часть Seed-X оказалась вне CUDA; переход на CPU отключён.")


def _is_cuda_device(value: Any) -> bool:
    if isinstance(value, int):
        return value == 0
    return str(value).strip().casefold() == DEVICE


def _runtime_signature(
    model_path: Path,
    revision: str,
    content_fingerprint: str | None,
    *,
    loaded: bool,
) -> dict[str, Any]:
    return {
        "backend": "seedx-bitsandbytes-nf4",
        "model_id": SEEDX_MODEL_ID,
        "model_revision": revision,
        "model_path": str(model_path),
        "model_content_fingerprint": content_fingerprint,
        "device": DEVICE,
        "quantization": "bitsandbytes-nf4-double",
        "decoding": f"beam-search-{NUM_BEAMS}-no-repeat-ngram-{NO_REPEAT_NGRAM_SIZE}",
        "batch_size": MAX_BATCH_SIZE,
        "loaded": loaded,
    }


def _import_runtime() -> tuple[Any, Any, Any, Any]:
    mismatches = [
        f"{package}=={required} (обнаружено {_package_version(package)})"
        for package, required in REQUIRED_RUNTIME_VERSIONS.items()
        if _package_version(package) != required
    ]
    if mismatches:
        raise SeedXRuntimeError("Среда Seed-X не совпадает с закреплённой: " + "; ".join(mismatches) + ".")
    try:
        with redirect_stdout(sys.stderr):
            import bitsandbytes  # noqa: F401
            import torch
            from transformers import AutoTokenizer, BitsAndBytesConfig, MistralForCausalLM
    except ImportError as exc:
        raise SeedXRuntimeError("В среде отсутствуют зависимости Seed-X.") from exc
    return torch, AutoTokenizer, MistralForCausalLM, BitsAndBytesConfig


def _package_version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "не установлено"


def main() -> None:
    """Запускает постоянный протокольный процесс Seed-X."""
    run_worker_loop(SeedXWorkerRuntime())


if __name__ == "__main__":
    main()


__all__ = [
    "LANGUAGE_NAMES",
    "MAX_BATCH_SIZE",
    "MAX_INPUT_TOKENS",
    "MAX_NEW_TOKENS",
    "NO_REPEAT_NGRAM_SIZE",
    "NUM_BEAMS",
    "REQUIRED_RUNTIME_VERSIONS",
    "SUPPORTED_DIRECTIONS",
    "SeedXOutOfMemoryError",
    "SeedXRuntimeError",
    "SeedXWorkerRuntime",
    "build_translation_prompt",
    "main",
]
