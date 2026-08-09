"""Изолированная среда закреплённых профилей TranslateGemma в CUDA."""

from __future__ import annotations

import gc
import importlib.metadata
import os
import sys
from collections.abc import Mapping
from contextlib import redirect_stdout
from pathlib import Path
from types import MappingProxyType
from typing import Any

from sub_translate.translators.registry import (
    TRANSLATEGEMMA_PROFILE_IDS,
    TRANSLATEGEMMA_WORKER_REQUIREMENTS,
    TranslatorMetadata,
    get_translator_metadata,
)
from sub_translate.utils.translation_utils import translate_text as translate_text_common
from sub_translate.workers.common import PublicWorkerError, run_worker_loop

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _required_runtime_versions() -> Mapping[str, str]:
    versions: dict[str, str] = {}
    for requirement in TRANSLATEGEMMA_WORKER_REQUIREMENTS:
        package, version = requirement.split("==", maxsplit=1)
        versions[package] = version
    return MappingProxyType(versions)


REQUIRED_RUNTIME_VERSIONS = _required_runtime_versions()
MAX_BATCH_SIZE = 128
MAX_MODEL_INPUT = 512
MAX_OUTPUT_LENGTH = 1_024
DEVICE = "cuda:0"
SUPPORTED_DIRECTIONS = (("en", "ru"), ("ru", "en"))
_TOKENIZER_KWARGS = MappingProxyType(
    {
        "fix_mistral_regex": False,
        "use_fast": True,
    }
)


class TranslateGemmaRuntimeError(PublicWorkerError):
    """Безопасная ошибка среды TranslateGemma."""


class TranslateGemmaOutOfMemoryError(PublicWorkerError):
    """TranslateGemma не поместилась в доступную видеопамять."""


class TranslateGemmaWorkerRuntime:
    """Держит одну закреплённую TranslateGemma и не допускает перехода на CPU или к сети."""

    def __init__(self) -> None:
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._torch: Any | None = None
        self._load_key: tuple[str, str, str, str] | None = None
        self._modules: tuple[Any, Any, Any, Any] | None = None

    def preflight(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Проверяет среду, CUDA и локальный каталог без загрузки весов."""
        metadata, model_path, revision = _validate_model_request(payload)
        torch, _tokenizer_class, _model_class, _quantization_class = self._runtime_modules()
        if not bool(torch.cuda.is_available()):
            raise TranslateGemmaRuntimeError("Для TranslateGemma требуется CUDA; переход на CPU отключён.")
        return _runtime_signature(metadata, model_path, revision, loaded=self._model is not None)

    def translate(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Последовательно переводит пачку с проверкой её мощности."""
        metadata, model_path, revision = _validate_model_request(payload)
        source_lang, target_lang = _validate_direction(payload)
        texts = _validate_texts(payload.get("texts"))
        if not texts or not any(texts):
            return {
                "translations": ["" for _text in texts],
                "runtime": _runtime_signature(
                    metadata,
                    model_path,
                    revision,
                    loaded=self._model is not None,
                ),
            }
        self._ensure_loaded(metadata, model_path, revision)
        translations = [
            ""
            if not text
            else self._translate_text(
                text,
                source_lang,
                target_lang,
                metadata.max_input_tokens or MAX_MODEL_INPUT,
                metadata.max_output_tokens or MAX_OUTPUT_LENGTH,
            )
            for text in texts
        ]
        if len(translations) != len(texts):
            raise TranslateGemmaRuntimeError("TranslateGemma нарушила мощность входной пачки.")
        return {
            "translations": translations,
            "runtime": _runtime_signature(metadata, model_path, revision, loaded=True),
        }

    def unload(self) -> dict[str, Any]:
        """Освобождает веса и кеш CUDA внутри дочернего процесса."""
        self._model = None
        self._tokenizer = None
        self._load_key = None
        gc.collect()
        if self._torch is not None and bool(self._torch.cuda.is_available()):
            self._torch.cuda.empty_cache()
            self._torch.cuda.ipc_collect()
        return {"unloaded": True}

    def _ensure_loaded(self, metadata: TranslatorMetadata, model_path: Path, revision: str) -> None:
        load_key = (metadata.id, str(model_path).casefold(), revision, metadata.quantization or "")
        if self._model is not None and self._tokenizer is not None and self._load_key == load_key:
            return
        if self._model is not None or self._tokenizer is not None:
            self.unload()
        torch, tokenizer_class, model_class, quantization_class = self._runtime_modules()
        if not bool(torch.cuda.is_available()):
            raise TranslateGemmaRuntimeError("Для TranslateGemma требуется CUDA; переход на CPU отключён.")
        try:
            quantization = _quantization_config(metadata, torch, quantization_class)
            with redirect_stdout(sys.stderr):
                tokenizer = tokenizer_class.from_pretrained(
                    str(model_path),
                    revision=revision,
                    local_files_only=True,
                    trust_remote_code=False,
                    **dict(_TOKENIZER_KWARGS),
                )
                model = model_class.from_pretrained(
                    str(model_path),
                    revision=revision,
                    local_files_only=True,
                    trust_remote_code=False,
                    use_safetensors=True,
                    quantization_config=quantization,
                    device_map={"": 0},
                    dtype=torch.bfloat16,
                    low_cpu_mem_usage=True,
                )
                model.eval()
            _sanitize_generation_config(model)
            _require_cuda_quantization(model, metadata)
        except torch.cuda.OutOfMemoryError as exc:
            self.unload()
            raise TranslateGemmaOutOfMemoryError(
                "Недостаточно видеопамяти для выбранной TranslateGemma; CPU-переход отключён."
            ) from exc
        except PublicWorkerError:
            self.unload()
            raise
        except Exception as exc:
            self.unload()
            raise TranslateGemmaRuntimeError("Не удалось загрузить локальную TranslateGemma в CUDA.") from exc
        self._tokenizer = tokenizer
        self._model = model
        self._torch = torch
        self._load_key = load_key

    def _translate_text(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        token_limit: int,
        max_output_tokens: int,
    ) -> str:
        return translate_text_common(
            text=text,
            token_limit=token_limit,
            token_counter=lambda value: self._token_count(
                value,
                source_lang,
                target_lang,
            ),
            chunk_translator=lambda value: self._translate_chunk(
                value,
                source_lang,
                target_lang,
                max_output_tokens,
            ),
        )

    def _prepare_inputs(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
    ) -> dict[str, Any]:
        assert self._tokenizer is not None
        return self._tokenizer.apply_chat_template(
            translation_messages(text, source_lang, target_lang),
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

    def _token_count(self, text: str, source_lang: str, target_lang: str) -> int:
        inputs = self._prepare_inputs(text, source_lang, target_lang)
        input_ids = inputs.get("input_ids")
        if input_ids is None:
            raise TranslateGemmaRuntimeError("Токенизатор TranslateGemma не вернул input_ids.")
        return int(input_ids.shape[-1])

    def _translate_chunk(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        max_output_tokens: int,
    ) -> str:
        assert self._tokenizer is not None and self._model is not None and self._torch is not None
        inputs = self._prepare_inputs(text, source_lang, target_lang)
        input_ids = inputs.get("input_ids")
        if input_ids is None:
            raise TranslateGemmaRuntimeError("Токенизатор TranslateGemma не вернул input_ids.")
        input_length = int(input_ids.shape[-1])
        device_inputs = {key: value.to(DEVICE) if hasattr(value, "to") else value for key, value in inputs.items()}
        try:
            with self._torch.inference_mode(), redirect_stdout(sys.stderr):
                outputs = self._model.generate(
                    **device_inputs,
                    max_new_tokens=max_output_tokens,
                    do_sample=False,
                    num_beams=1,
                )
            generated_tokens = outputs[0][input_length:]
            result = self._tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True,
            ).strip()
        except self._torch.cuda.OutOfMemoryError as exc:
            self.unload()
            raise TranslateGemmaOutOfMemoryError(
                "Недостаточно видеопамяти для перевода TranslateGemma; процесс будет перезапущен."
            ) from exc
        except PublicWorkerError:
            raise
        except Exception as exc:
            self.unload()
            raise TranslateGemmaRuntimeError("TranslateGemma не выполнила локальный перевод.") from exc
        if not result:
            raise TranslateGemmaRuntimeError("TranslateGemma вернула пустой перевод непустого текста.")
        return result

    def _runtime_modules(self) -> tuple[Any, Any, Any, Any]:
        if self._modules is None:
            self._modules = _import_runtime()
            self._torch = self._modules[0]
        return self._modules


def translation_messages(
    text: str,
    source_lang: str,
    target_lang: str,
) -> list[dict[str, Any]]:
    """Строит официальный одноходовый запрос TranslateGemma."""
    if (source_lang, target_lang) not in SUPPORTED_DIRECTIONS:
        raise ValueError("TranslateGemma поддерживает только направления en -> ru и ru -> en.")
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "source_lang_code": source_lang,
                    "target_lang_code": target_lang,
                    "text": text,
                }
            ],
        }
    ]


def _validate_model_request(payload: Mapping[str, Any]) -> tuple[TranslatorMetadata, Path, str]:
    profile_id = str(payload.get("profile_id") or "").strip().casefold()
    if profile_id not in TRANSLATEGEMMA_PROFILE_IDS:
        raise TranslateGemmaRuntimeError("Запрос относится к неизвестному профилю TranslateGemma.")
    metadata = get_translator_metadata(profile_id)
    model_id = str(payload.get("model_id") or "").strip()
    if metadata.model_id is None or model_id != metadata.model_id:
        raise TranslateGemmaRuntimeError("Запрос относится не к закреплённой TranslateGemma.")
    revision = str(payload.get("model_revision") or "").strip().lower()
    if metadata.model_revision is None or revision != metadata.model_revision:
        raise TranslateGemmaRuntimeError("TranslateGemma требует закреплённую ревизию модели.")
    raw_path = str(payload.get("model_path") or "").strip()
    if not raw_path:
        raise TranslateGemmaRuntimeError("Не указан локальный каталог TranslateGemma.")
    model_path = Path(raw_path).expanduser().resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(f"Каталог TranslateGemma не найден: {model_path}")
    missing = [name for name in metadata.required_files if not (model_path / name).is_file()]
    if missing:
        raise TranslateGemmaRuntimeError(f"Каталог TranslateGemma неполон; отсутствуют файлы: {', '.join(missing)}.")
    if any(
        not any((model_path / name).is_file() for name in alternatives)
        for alternatives in metadata.required_file_groups
    ):
        raise TranslateGemmaRuntimeError("Каталог TranslateGemma не содержит safetensors-веса.")
    return metadata, model_path, revision


def _validate_direction(payload: Mapping[str, Any]) -> tuple[str, str]:
    source_lang = str(payload.get("source_lang") or "").strip().casefold()
    target_lang = str(payload.get("target_lang") or "").strip().casefold()
    if (source_lang, target_lang) not in SUPPORTED_DIRECTIONS:
        raise TranslateGemmaRuntimeError("TranslateGemma поддерживает только направления en -> ru и ru -> en.")
    return source_lang, target_lang


def _validate_texts(raw_texts: Any) -> list[str]:
    if not isinstance(raw_texts, list):
        raise TranslateGemmaRuntimeError("Поле texts должно быть списком строк.")
    if len(raw_texts) > MAX_BATCH_SIZE:
        raise TranslateGemmaRuntimeError(
            f"Пачка TranslateGemma содержит {len(raw_texts)} элементов при пределе {MAX_BATCH_SIZE}."
        )
    if not all(isinstance(text, str) for text in raw_texts):
        raise TranslateGemmaRuntimeError("Пачка TranslateGemma содержит значение, которое не является строкой.")
    return list(raw_texts)


def _sanitize_generation_config(model: Any) -> None:
    config = getattr(model, "generation_config", None)
    if config is None:
        return
    if (
        getattr(config, "max_new_tokens", None) is not None
        and getattr(
            config,
            "max_length",
            None,
        )
        is not None
    ):
        config.max_new_tokens = None
    config.do_sample = False
    config.num_beams = 1
    for attribute in (
        "temperature",
        "top_p",
        "min_p",
        "typical_p",
        "top_k",
        "epsilon_cutoff",
        "eta_cutoff",
    ):
        if hasattr(config, attribute):
            setattr(config, attribute, None)
    if hasattr(config, "early_stopping"):
        config.early_stopping = False
    if hasattr(config, "length_penalty"):
        config.length_penalty = 1.0


def _quantization_config(metadata: TranslatorMetadata, torch: Any, quantization_class: Any) -> Any:
    if metadata.quantization == "bitsandbytes-int8":
        return quantization_class(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=False,
        )
    if metadata.quantization == "bitsandbytes-nf4-double":
        return quantization_class(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    raise TranslateGemmaRuntimeError("Для профиля TranslateGemma не задано разрешённое квантование.")


def _require_cuda_quantization(model: Any, metadata: TranslatorMetadata) -> None:
    quantizer = getattr(model, "hf_quantizer", None)
    quantization = getattr(quantizer, "quantization_config", None)
    if metadata.quantization == "bitsandbytes-int8":
        loaded = bool(getattr(model, "is_loaded_in_8bit", False)) or bool(
            getattr(quantization, "load_in_8bit", False)
        )
    elif metadata.quantization == "bitsandbytes-nf4-double":
        loaded = bool(getattr(model, "is_loaded_in_4bit", False)) or bool(
            getattr(quantization, "load_in_4bit", False)
        )
        quant_type = str(getattr(quantization, "bnb_4bit_quant_type", "")).casefold()
        double_quant = bool(getattr(quantization, "bnb_4bit_use_double_quant", False))
        loaded = loaded and quant_type == "nf4" and double_quant
    else:
        loaded = False
    if not loaded:
        raise TranslateGemmaRuntimeError("TranslateGemma загружена без обязательного квантования профиля.")
    _require_cuda_placement(model)


def _require_cuda_placement(model: Any) -> None:
    device_map = getattr(model, "hf_device_map", None)
    if isinstance(device_map, Mapping):
        devices = tuple(device_map.values())
        if not devices or any(not _is_cuda_device(value) for value in devices):
            raise TranslateGemmaRuntimeError("Часть TranslateGemma оказалась вне CUDA; CPU-переход отключён.")
        return
    parameters = getattr(model, "parameters", None)
    if not callable(parameters):
        raise TranslateGemmaRuntimeError("TranslateGemma не сообщает размещение параметров.")
    if any(getattr(parameter.device, "type", "") != "cuda" for parameter in parameters()):
        raise TranslateGemmaRuntimeError("Часть TranslateGemma оказалась вне CUDA; CPU-переход отключён.")


def _is_cuda_device(value: Any) -> bool:
    if isinstance(value, int):
        return value >= 0
    return str(value).casefold().startswith("cuda")


def _runtime_signature(
    metadata: TranslatorMetadata,
    model_path: Path,
    revision: str,
    *,
    loaded: bool,
) -> dict[str, Any]:
    return {
        "backend": metadata.id,
        "profile_id": metadata.id,
        "model_id": metadata.model_id,
        "model_revision": revision,
        "model_path": str(model_path),
        "device": DEVICE,
        "quantization": metadata.quantization,
        "decoding": "greedy",
        "loaded": loaded,
    }


def _import_runtime() -> tuple[Any, Any, Any, Any]:
    installed = {package: _package_version(package) for package in REQUIRED_RUNTIME_VERSIONS}
    mismatches = [
        f"{package}=={required} (обнаружено {installed[package]})"
        for package, required in REQUIRED_RUNTIME_VERSIONS.items()
        if installed[package] != required
    ]
    if mismatches:
        raise TranslateGemmaRuntimeError(
            "Изолированная среда TranslateGemma не совпадает с закреплённой: " + "; ".join(mismatches) + "."
        )
    try:
        with redirect_stdout(sys.stderr):
            import bitsandbytes  # noqa: F401
            import torch
            from transformers import (
                AutoTokenizer,
                BitsAndBytesConfig,
                Gemma3ForConditionalGeneration,
            )
    except ImportError as exc:
        raise TranslateGemmaRuntimeError("В изолированной среде отсутствуют зависимости TranslateGemma.") from exc
    return torch, AutoTokenizer, Gemma3ForConditionalGeneration, BitsAndBytesConfig


def _package_version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "не установлено"


def main() -> None:
    """Запускает постоянный протокольный процесс TranslateGemma."""
    run_worker_loop(TranslateGemmaWorkerRuntime())


if __name__ == "__main__":
    main()


__all__ = [
    "MAX_MODEL_INPUT",
    "MAX_OUTPUT_LENGTH",
    "REQUIRED_RUNTIME_VERSIONS",
    "SUPPORTED_DIRECTIONS",
    "TranslateGemmaOutOfMemoryError",
    "TranslateGemmaRuntimeError",
    "TranslateGemmaWorkerRuntime",
    "main",
    "translation_messages",
]
