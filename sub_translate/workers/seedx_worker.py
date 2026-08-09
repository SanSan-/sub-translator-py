"""Изолированная среда Seed-X-PPO-7B-AWQ-Int4."""

from __future__ import annotations

import gc
import importlib.metadata
import json
import os
import sys
from collections.abc import Mapping
from contextlib import redirect_stdout
from pathlib import Path
from types import MappingProxyType
from typing import Any

from sub_translate.translators.registry import SEEDX_MODEL_ID, SEEDX_MODEL_REVISION
from sub_translate.workers.common import PublicWorkerError, run_worker_loop

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

REQUIRED_RUNTIME_VERSIONS = MappingProxyType(
    {
        "accelerate": "1.14.0",
        "compressed-tensors": "0.18.0",
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
DEVICE = "cuda:0"
SUPPORTED_DIRECTIONS = (("en", "ru"), ("ru", "en"))
LANGUAGE_NAMES = {"en": "English", "ru": "Russian"}
_REQUIRED_FILES = (
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
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
        self._load_key: tuple[str, str] | None = None
        self._modules: tuple[Any, Any, Any] | None = None

    def preflight(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Проверяет среду, CUDA и локальные файлы без загрузки весов."""
        model_path, revision = _validate_model_request(payload)
        torch, _tokenizer_class, _model_class = self._runtime_modules()
        if not bool(torch.cuda.is_available()):
            raise SeedXRuntimeError("Для Seed-X требуется CUDA; переход на CPU отключён.")
        _require_bfloat16_cuda(torch)
        _validate_quantization_config(model_path)
        return _runtime_signature(model_path, revision, loaded=self._model is not None)

    def translate(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Переводит ровно один элемент, ограничивая пиковую видеопамять."""
        model_path, revision = _validate_model_request(payload)
        source_lang, target_lang = _validate_direction(payload)
        texts = _validate_texts(payload.get("texts"))
        if not texts:
            return {
                "translations": [],
                "runtime": _runtime_signature(model_path, revision, loaded=self._model is not None),
            }
        if not texts[0]:
            return {
                "translations": [""],
                "runtime": _runtime_signature(model_path, revision, loaded=self._model is not None),
            }

        self._ensure_loaded(model_path, revision)
        translation = self._translate_one(texts[0], source_lang, target_lang)
        return {
            "translations": [translation],
            "runtime": _runtime_signature(model_path, revision, loaded=True),
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

    def _ensure_loaded(self, model_path: Path, revision: str) -> None:
        load_key = (str(model_path).casefold(), revision)
        if self._model is not None and self._tokenizer is not None and self._load_key == load_key:
            return
        if self._model is not None or self._tokenizer is not None:
            self.unload()
        _validate_quantization_config(model_path)
        torch, tokenizer_class, model_class = self._runtime_modules()
        if not bool(torch.cuda.is_available()):
            raise SeedXRuntimeError("Для Seed-X требуется CUDA; переход на CPU отключён.")
        _require_bfloat16_cuda(torch)
        try:
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
                    device_map={"": DEVICE},
                    dtype=torch.bfloat16,
                    low_cpu_mem_usage=True,
                )
                model.eval()
            _require_compressed_int4(model)
            _require_cuda_only(model)
        except torch.cuda.OutOfMemoryError as exc:
            self.unload()
            raise SeedXOutOfMemoryError("Недостаточно видеопамяти для Seed-X Int4; переход на CPU отключён.") from exc
        except PublicWorkerError:
            self.unload()
            raise
        except Exception as exc:
            self.unload()
            raise SeedXRuntimeError("Не удалось загрузить локальную Seed-X Int4 в CUDA.") from exc
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
        if not result:
            raise SeedXRuntimeError("Seed-X вернула пустой перевод непустого текста.")
        return result

    def _runtime_modules(self) -> tuple[Any, Any, Any]:
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


def _validate_model_request(payload: Mapping[str, Any]) -> tuple[Path, str]:
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
    return model_path, revision


def _validate_quantization_config(model_path: Path) -> None:
    config_path = model_path / "config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(config, Mapping):
            raise SeedXRuntimeError("Корнем config.json Seed-X должен быть JSON-объект.")
        architectures = config.get("architectures")
        quantization = config.get("quantization_config")
        groups = quantization.get("config_groups") if isinstance(quantization, Mapping) else None
        group = groups.get("group_0") if isinstance(groups, Mapping) else None
        weights = group.get("weights") if isinstance(group, Mapping) else None
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SeedXRuntimeError("Повреждён config.json локальной Seed-X.") from exc
    if not isinstance(quantization, Mapping) or quantization.get("quant_method") != "compressed-tensors":
        raise SeedXRuntimeError("Seed-X должна использовать закреплённый формат compressed-tensors.")
    if architectures != ["MistralForCausalLM"] or config.get("model_type") != "mistral":
        raise SeedXRuntimeError("Seed-X должна использовать закреплённую архитектуру MistralForCausalLM.")
    if str(config.get("torch_dtype") or "").casefold() != "bfloat16":
        raise SeedXRuntimeError("Seed-X требует вычисления BF16 из закреплённой конфигурации.")
    if quantization.get("format") != "pack-quantized" or quantization.get("quantization_status") != "compressed":
        raise SeedXRuntimeError("Seed-X должна содержать упакованные сжатые веса.")
    if (
        not isinstance(weights, Mapping)
        or weights.get("num_bits") != 4
        or weights.get("group_size") != 128
        or weights.get("type") != "int"
    ):
        raise SeedXRuntimeError("Seed-X должна содержать закреплённые 4-битные веса.")


def _require_bfloat16_cuda(torch: Any) -> None:
    checker = getattr(torch.cuda, "is_bf16_supported", None)
    if not callable(checker) or not bool(checker()):
        raise SeedXRuntimeError("Для Seed-X требуется CUDA с поддержкой BF16; переход на FP16 отключён.")


def _eos_token_id(tokenizer: Any) -> int:
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if not isinstance(eos_token_id, int) or isinstance(eos_token_id, bool) or eos_token_id < 0:
        raise SeedXRuntimeError("Токенизатор Seed-X не сообщает корректный токен завершения.")
    return eos_token_id


def _completed_generated_tokens(tokens: Any, eos_token_id: int) -> Any:
    raw_ids = tokens.tolist() if callable(getattr(tokens, "tolist", None)) else list(tokens)
    if not raw_ids or not all(isinstance(token_id, int) for token_id in raw_ids):
        raise SeedXRuntimeError("Seed-X не вернула корректную последовательность токенов.")
    if eos_token_id not in raw_ids:
        raise SeedXRuntimeError("Seed-X не завершила перевод токеном EOS; незавершённый результат отклонён.")
    return tokens


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


def _require_compressed_int4(model: Any) -> None:
    quantizer = getattr(model, "hf_quantizer", None)
    quantization = getattr(quantizer, "quantization_config", None)
    if quantization is None or not bool(getattr(quantization, "run_compressed", False)):
        raise SeedXRuntimeError("Seed-X загружена без обязательного сжатого 4-битного режима.")


def _require_cuda_only(model: Any) -> None:
    device_map = getattr(model, "hf_device_map", None)
    if isinstance(device_map, Mapping):
        devices = tuple(device_map.values())
        if not devices or any(not _is_cuda_device(value) for value in devices):
            raise SeedXRuntimeError("Часть Seed-X оказалась вне CUDA; переход на CPU отключён.")
        return
    parameters = getattr(model, "parameters", None)
    if not callable(parameters):
        raise SeedXRuntimeError("Seed-X не сообщает размещение параметров.")
    if any(getattr(parameter.device, "type", "") != "cuda" for parameter in parameters()):
        raise SeedXRuntimeError("Часть Seed-X оказалась вне CUDA; переход на CPU отключён.")


def _is_cuda_device(value: Any) -> bool:
    if isinstance(value, int):
        return value >= 0
    return str(value).casefold().startswith("cuda")


def _runtime_signature(model_path: Path, revision: str, *, loaded: bool) -> dict[str, Any]:
    return {
        "backend": "seedx-compressed-int4",
        "model_id": SEEDX_MODEL_ID,
        "model_revision": revision,
        "model_path": str(model_path),
        "device": DEVICE,
        "quantization": "compressed-tensors-int4",
        "decoding": f"beam-search-{NUM_BEAMS}",
        "batch_size": MAX_BATCH_SIZE,
        "loaded": loaded,
    }


def _import_runtime() -> tuple[Any, Any, Any]:
    mismatches = [
        f"{package}=={required} (обнаружено {_package_version(package)})"
        for package, required in REQUIRED_RUNTIME_VERSIONS.items()
        if _package_version(package) != required
    ]
    if mismatches:
        raise SeedXRuntimeError("Среда Seed-X не совпадает с закреплённой: " + "; ".join(mismatches) + ".")
    try:
        with redirect_stdout(sys.stderr):
            import compressed_tensors  # noqa: F401
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SeedXRuntimeError("В среде отсутствуют зависимости Seed-X.") from exc
    return torch, AutoTokenizer, AutoModelForCausalLM


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
    "NUM_BEAMS",
    "REQUIRED_RUNTIME_VERSIONS",
    "SUPPORTED_DIRECTIONS",
    "SeedXOutOfMemoryError",
    "SeedXRuntimeError",
    "SeedXWorkerRuntime",
    "build_translation_prompt",
    "main",
]
