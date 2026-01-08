from __future__ import annotations

from pathlib import Path
from typing import Any, Type

import torch
from transformers import PreTrainedModel

from sub_translate.utils.local_utils import resolve_device_and_quantization, sanitize_generation_config


class TranslatorLoadError(RuntimeError):
    """Исключение, возникающее при невозможности загрузить модель перевода."""


def _describe_incomplete_download(cache_dir: Path, use_safetensors: bool) -> str:
    incomplete = list(cache_dir.rglob("*.incomplete"))

    if incomplete:
        return (
            "Обнаружены файлы с расширением '.incomplete' в кеше модели. "
            "Дождитесь завершения загрузки или удалите неполный кеш и повторите попытку."
        )

    if use_safetensors:
        safetensors = list(cache_dir.rglob("model.safetensors"))
        if not safetensors:
            return (
                "Файл 'model.safetensors' отсутствует в каталоге кеша. "
                "Вероятно, загрузка не завершилась - скачайте модель повторно перед запуском перевода."
            )

    return ""


def _handle_model_load_error(exc: Exception, cache_dir: Path, use_safetensors: bool) -> None:
    hint = _describe_incomplete_download(cache_dir, use_safetensors)
    if not hint and isinstance(exc, RuntimeError):
        message = str(exc)
        if "torch.load" in message and "v2.6" in message:
            hint = (
                "Библиотека transformers пытается открыть бинарные веса через `torch.load`, "
                "что требует torch>=2.6. Убедитесь, что в кеше есть safetensors-веса."
            )
        elif _is_meta_error(exc):
            hint = (
                "Часть параметров модели осталась на meta устройстве. "
                "Удалите кеш модели и скачайте веса заново."
            )

    context = "Не удалось загрузить модель перевода."
    if hint:
        context = f"{context}\n{hint}"

    raise TranslatorLoadError(context) from exc


def _is_meta_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "meta tensor" in message or "meta device" in message


def _is_cuda_oom(exc: Exception) -> bool:
    message = str(exc).lower()
    if "out of memory" not in message:
        return False
    return any(token in message for token in ("cuda", "cudnn", "cublas"))


def _should_cpu_fallback(exc: Exception) -> bool:
    return _is_meta_error(exc) or _is_cuda_oom(exc)


def _has_meta_parameters(model: PreTrainedModel) -> bool:
    parameters = getattr(model, "parameters", None)
    if callable(parameters):
        for param in parameters():
            if param.device.type == "meta":
                return True
    buffers = getattr(model, "buffers", None)
    if callable(buffers):
        for buffer in buffers():
            if buffer.device.type == "meta":
                return True
    return False


def load_model_components(
    model_name: str,
    cache_dir: Path,
    model_class: Type[PreTrainedModel],
    processor_class: Type[Any],
    use_safetensors: bool = False,
    processor_kwargs: dict[str, Any] | None = None,
    model_kwargs: dict[str, Any] | None = None,
    allow_quantization: bool = True,
    allow_cpu_fallback: bool = False,
    trust_remote_code: bool = False,
) -> tuple[Any, Any, torch.device]:
    """
    Загружает компоненты модели HuggingFace (процессор/токенизатор и модель)
    с учетом квантования и выбора устройства.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        processor = processor_class.from_pretrained(
            model_name,
            cache_dir=str(cache_dir),
            trust_remote_code=trust_remote_code,
            **(processor_kwargs or {}),
        )
    except Exception as exc:  # pragma: no cover
        _handle_model_load_error(exc, cache_dir, use_safetensors)

    device, quantization_config = resolve_device_and_quantization(
        enable_cpu_offload=True, allow_quantization=allow_quantization
    )

    def _ensure_no_meta(model: PreTrainedModel) -> None:
        if _has_meta_parameters(model):
            raise RuntimeError("Параметры модели остались на meta device.")

    def _build_fallback_kwargs(force_download: bool, *, use_dtype: bool) -> dict[str, Any]:
        fallback_kwargs: dict[str, Any] = {"low_cpu_mem_usage": False}
        if use_dtype:
            fallback_kwargs["dtype"] = torch.float16
        if force_download:
            fallback_kwargs["force_download"] = True
        return fallback_kwargs

    def _load_without_device_map(force_download: bool, *, use_dtype: bool) -> PreTrainedModel:
        fallback_kwargs = _build_fallback_kwargs(force_download, use_dtype=use_dtype)
        return model_class.from_pretrained(
            model_name,
            cache_dir=str(cache_dir),
            use_safetensors=use_safetensors,
            trust_remote_code=trust_remote_code,
            **fallback_kwargs,
            **(model_kwargs or {}),
        )

    def _move_to_device(model: PreTrainedModel, target_device: torch.device) -> None:
        _ensure_no_meta(model)
        if target_device.type != "cpu":
            model.to(target_device)
        _ensure_no_meta(model)

    def _load_with_device_map() -> PreTrainedModel:
        if quantization_config is not None:
            try:
                model = model_class.from_pretrained(
                    model_name,
                    cache_dir=str(cache_dir),
                    quantization_config=quantization_config,
                    device_map="auto",
                    use_safetensors=use_safetensors,
                    trust_remote_code=trust_remote_code,
                    **(model_kwargs or {}),
                )
            except Exception as exc:
                if not _is_meta_error(exc):
                    raise
                model = _load_without_device_map(force_download=False, use_dtype=device.type == "cuda")
                try:
                    _move_to_device(model, device)
                except Exception as exc_inner:
                    if not _is_meta_error(exc_inner):
                        raise
                    model = _load_without_device_map(force_download=True, use_dtype=device.type == "cuda")
                    _move_to_device(model, device)
            return model
        try:
            model = model_class.from_pretrained(
                model_name,
                cache_dir=str(cache_dir),
                use_safetensors=use_safetensors,
                trust_remote_code=trust_remote_code,
                device_map="auto",
                **(model_kwargs or {}),
            )
        except Exception as exc:
            if not _is_meta_error(exc):
                raise
            model = _load_without_device_map(force_download=False, use_dtype=device.type == "cuda")
            try:
                _move_to_device(model, device)
            except Exception as exc_inner:
                if not _is_meta_error(exc_inner):
                    raise
                model = _load_without_device_map(force_download=True, use_dtype=device.type == "cuda")
                _move_to_device(model, device)
        return model

    def _load_on_cpu() -> PreTrainedModel:
        cpu_device = torch.device("cpu")
        try:
            model = _load_without_device_map(force_download=False, use_dtype=False)
            _move_to_device(model, cpu_device)
        except Exception as exc:
            if not _should_cpu_fallback(exc):
                raise
            model = _load_without_device_map(force_download=True, use_dtype=False)
            _move_to_device(model, cpu_device)
        return model

    try:
        try:
            model = _load_with_device_map()
        except Exception as exc:
            if allow_cpu_fallback and device.type == "cuda" and _should_cpu_fallback(exc):
                print("Загрузка на GPU не удалась, пробую CPU.", flush=True)
                model = _load_on_cpu()
                device = torch.device("cpu")
            else:
                raise
    except Exception as exc:  # pragma: no cover
        _handle_model_load_error(exc, cache_dir, use_safetensors)

    model.eval()
    sanitize_generation_config(model)
    return processor, model, device
