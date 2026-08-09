from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Never

import torch
from transformers import PreTrainedModel

from sub_translate.utils.local_utils import (
    Int8UnavailableError,
    resolve_device_and_quantization,
    sanitize_generation_config,
)


class TranslatorLoadError(RuntimeError):
    """Исключение, возникающее при невозможности загрузить модель перевода."""


class ModelAcquisitionError(TranslatorLoadError):
    """Ошибка проверки или безопасного получения локальной модели."""


@dataclass(frozen=True, slots=True)
class ResolvedLocalModel:
    """Проверенный локальный каталог и закреплённая ревизия модели."""

    path: Path
    revision: str
    content_fingerprint: str | None = None
    revision_verified: bool = False


@dataclass(frozen=True, slots=True)
class ModelDirectoryInspection:
    """Результат локальной проверки обязательных файлов модели."""

    complete: bool
    issues: tuple[str, ...]
    content_fingerprint: str | None = None
    revision_verified: bool = False
    structurally_complete: bool = False


@dataclass(frozen=True, slots=True)
class ModelLoadOptions:
    """Необязательные параметры загрузки процессора и весов модели."""

    use_safetensors: bool = False
    processor_kwargs: dict[str, Any] | None = None
    model_kwargs: dict[str, Any] | None = None
    allow_quantization: bool = True
    allow_cpu_fallback: bool = False
    trust_remote_code: bool = False
    revision: str | None = None
    local_files_only: bool = False
    require_int8: bool = False
    enable_cpu_offload: bool = True
    device_map: str | dict[str, int | str] = "auto"


MODEL_MARKER_FILENAME = ".sub_translate_model.json"
MIN_DOWNLOAD_RESERVE_BYTES = 1024**3
DOWNLOAD_RESERVE_RATIO = 0.10
MODEL_DOWNLOAD_LOCK_TIMEOUT_SECONDS = 600
_PINNED_REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_FINGERPRINT_CHUNK_BYTES = 64 * 1024


def _is_nonempty_file(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _safe_model_child(model_path: Path, relative_path: str) -> Path | None:
    root = model_path.resolve()
    candidate = (root / relative_path).resolve()
    if not candidate.is_relative_to(root):
        return None
    return candidate


def _inspect_weight_index(model_path: Path, issues: list[str]) -> None:
    index_path = model_path / "model.safetensors.index.json"
    if not index_path.is_file():
        return
    try:
        index_data = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index_data.get("weight_map")
        shard_names = set(weight_map.values()) if isinstance(weight_map, dict) else set()
    except OSError, UnicodeError, json.JSONDecodeError:
        issues.append("повреждён файл model.safetensors.index.json")
        return
    if not shard_names:
        issues.append("индекс safetensors не содержит списка сегментов весов")
        return
    for shard_name in sorted(shard_names):
        if not isinstance(shard_name, str):
            issues.append("индекс safetensors содержит некорректное имя сегмента")
            continue
        shard_path = _safe_model_child(model_path, shard_name)
        if shard_path is None or not _is_nonempty_file(shard_path):
            issues.append(f"отсутствует сегмент весов {shard_name}")


def _inspect_model_marker(
    model_path: Path,
    model_id: str,
    revision: str,
    content_fingerprint: str,
    issues: list[str],
) -> bool:
    marker_path = model_path / MODEL_MARKER_FILENAME
    if not marker_path.exists():
        return False
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except OSError, UnicodeError, json.JSONDecodeError:
        issues.append(f"повреждён файл {MODEL_MARKER_FILENAME}")
        return False
    if not isinstance(marker, dict):
        issues.append(f"повреждён файл {MODEL_MARKER_FILENAME}")
        return False
    if marker.get("model_id") != model_id or marker.get("revision") != revision:
        issues.append("локальный каталог относится к другой модели или ревизии")
        return False
    marker_fingerprint = marker.get("content_fingerprint")
    if not isinstance(marker_fingerprint, str) or not _SHA256_PATTERN.fullmatch(marker_fingerprint):
        issues.append(f"повреждён файл {MODEL_MARKER_FILENAME}")
        return False
    if marker_fingerprint != content_fingerprint:
        issues.append("содержимое модели изменено после проверки закреплённой ревизии")
        return False
    return True


def model_content_fingerprint(model_path: Path) -> str | None:
    """Строит быстрый отпечаток локальных файлов без чтения весов целиком."""
    resolved_path = model_path.expanduser().resolve()
    if not resolved_path.is_dir():
        return None
    files = sorted(
        (
            path
            for path in resolved_path.rglob("*")
            if path.is_file()
            and path.name != MODEL_MARKER_FILENAME
            and not path.name.endswith(".incomplete")
            and ".cache" not in path.relative_to(resolved_path).parts
        ),
        key=lambda path: path.relative_to(resolved_path).as_posix(),
    )
    if not files:
        return None
    digest = hashlib.sha256()
    for path in files:
        relative_path = path.relative_to(resolved_path).as_posix()
        stat = path.stat()
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(b"\0")
        with path.open("rb") as model_file:
            digest.update(model_file.read(_FINGERPRINT_CHUNK_BYTES))
            if stat.st_size > _FINGERPRINT_CHUNK_BYTES:
                model_file.seek(max(0, stat.st_size - _FINGERPRINT_CHUNK_BYTES))
                digest.update(model_file.read(_FINGERPRINT_CHUNK_BYTES))
        digest.update(b"\0")
    return digest.hexdigest()


def inspect_model_directory(
    model_path: Path,
    *,
    model_id: str,
    revision: str,
    required_files: tuple[str, ...],
    required_file_groups: tuple[tuple[str, ...], ...],
) -> ModelDirectoryInspection:
    """Проверяет каталог локально, не импортируя клиент Hugging Face Hub."""
    if not model_path.is_dir():
        return ModelDirectoryInspection(False, ("каталог не существует",))

    issues: list[str] = []
    if next(model_path.rglob("*.incomplete"), None) is not None:
        issues.append("обнаружены незавершённые файлы *.incomplete")
    for relative_path in required_files:
        candidate = _safe_model_child(model_path, relative_path)
        if candidate is None or not _is_nonempty_file(candidate):
            issues.append(f"отсутствует обязательный файл {relative_path}")
    for alternatives in required_file_groups:
        if not any(
            (candidate := _safe_model_child(model_path, relative_path)) is not None and _is_nonempty_file(candidate)
            for relative_path in alternatives
        ):
            issues.append(f"отсутствует один из файлов: {', '.join(alternatives)}")

    _inspect_weight_index(model_path, issues)
    content_fingerprint = model_content_fingerprint(model_path)
    if content_fingerprint is None:
        issues.append("каталог модели не содержит проверяемых файлов")
        return ModelDirectoryInspection(False, tuple(issues))
    revision_verified = _inspect_model_marker(
        model_path,
        model_id,
        revision,
        content_fingerprint,
        issues,
    )
    structurally_complete = not issues
    return ModelDirectoryInspection(
        not issues,
        tuple(issues),
        content_fingerprint,
        revision_verified,
        structurally_complete,
    )


def _cached_snapshot_path(cache_root: Path, model_id: str, revision: str) -> Path:
    repository_dir = f"models--{model_id.replace('/', '--')}"
    return cache_root / repository_dir / "snapshots" / revision


def _format_bytes(value: int) -> str:
    gibibytes = value / 1024**3
    return f"{gibibytes:.2f} ГиБ"


def _missing_download_bytes(model_info: Any, target_path: Path) -> int:
    missing_bytes = 0
    for sibling in getattr(model_info, "siblings", ()):
        relative_path = getattr(sibling, "rfilename", None)
        file_size = getattr(sibling, "size", None)
        if not isinstance(relative_path, str) or not isinstance(file_size, int) or file_size < 0:
            raise ModelAcquisitionError("Hugging Face Hub не вернул надёжный размер всех файлов закреплённой ревизии.")
        local_file = _safe_model_child(target_path, relative_path)
        if local_file is None:
            raise ModelAcquisitionError(f"Hugging Face Hub вернул небезопасный путь файла: {relative_path}.")
        if local_file.is_file() and local_file.stat().st_size == file_size:
            continue
        missing_bytes += file_size
    return missing_bytes


def _ensure_download_space(target_parent: Path, missing_bytes: int) -> None:
    reserve_bytes = max(MIN_DOWNLOAD_RESERVE_BYTES, int(missing_bytes * DOWNLOAD_RESERVE_RATIO))
    required_bytes = missing_bytes + reserve_bytes
    free_bytes = shutil.disk_usage(target_parent).free
    if free_bytes < required_bytes:
        raise ModelAcquisitionError(
            "Недостаточно свободного места для модели: "
            f"нужно {_format_bytes(required_bytes)}, доступно {_format_bytes(free_bytes)}."
        )


def _write_model_marker(
    target_path: Path,
    model_id: str,
    revision: str,
    content_fingerprint: str,
) -> None:
    marker_path = target_path / MODEL_MARKER_FILENAME
    temporary_path = target_path / f"{MODEL_MARKER_FILENAME}.{os.getpid()}.tmp"
    try:
        with temporary_path.open("w", encoding="utf-8", newline="\n") as marker_file:
            json.dump(
                {
                    "model_id": model_id,
                    "revision": revision,
                    "content_fingerprint": content_fingerprint,
                },
                marker_file,
                ensure_ascii=False,
                indent=2,
            )
            marker_file.write("\n")
        os.replace(temporary_path, marker_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _huggingface_token_from_environment() -> str | bool:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN") or False


def _download_model_snapshot(
    *,
    model_id: str,
    revision: str,
    target_path: Path,
    required_files: tuple[str, ...],
    required_file_groups: tuple[tuple[str, ...], ...],
    requires_hf_token: bool,
) -> ResolvedLocalModel:
    target_parent = target_path.parent
    target_parent.mkdir(parents=True, exist_ok=True)

    from huggingface_hub.utils import WeakFileLock

    lock_path = target_parent / f".{target_path.name}.download.lock"
    try:
        with WeakFileLock(lock_path, timeout=MODEL_DOWNLOAD_LOCK_TIMEOUT_SECONDS):
            inspection = inspect_model_directory(
                target_path,
                model_id=model_id,
                revision=revision,
                required_files=required_files,
                required_file_groups=required_file_groups,
            )
            if inspection.complete and inspection.revision_verified:
                return ResolvedLocalModel(
                    target_path,
                    revision,
                    inspection.content_fingerprint,
                    True,
                )

            token = _huggingface_token_from_environment()
            if requires_hf_token and token is False:
                raise ModelAcquisitionError(
                    f"Для загрузки {model_id} задайте HF_TOKEN или HUGGINGFACE_TOKEN в окружении процесса."
                )

            from huggingface_hub import HfApi, snapshot_download

            try:
                model_info = HfApi(token=token).model_info(
                    model_id,
                    revision=revision,
                    files_metadata=True,
                )
            except Exception:
                raise ModelAcquisitionError(
                    f"Не удалось получить метаданные {model_id} для ревизии {revision}."
                ) from None

            missing_bytes = _missing_download_bytes(model_info, target_path)
            _ensure_download_space(target_parent, missing_bytes)
            try:
                snapshot_download(
                    repo_id=model_id,
                    revision=revision,
                    local_dir=str(target_path),
                    force_download=False,
                    token=token,
                )
            except Exception:
                raise ModelAcquisitionError(f"Не удалось загрузить {model_id} в выбранный каталог.") from None

            inspection = inspect_model_directory(
                target_path,
                model_id=model_id,
                revision=revision,
                required_files=required_files,
                required_file_groups=required_file_groups,
            )
            if not inspection.complete:
                details = "; ".join(inspection.issues)
                raise ModelAcquisitionError(f"Каталог модели остался неполным после загрузки: {details}.")
            assert inspection.content_fingerprint is not None
            _write_model_marker(
                target_path,
                model_id,
                revision,
                inspection.content_fingerprint,
            )
            verified_inspection = inspect_model_directory(
                target_path,
                model_id=model_id,
                revision=revision,
                required_files=required_files,
                required_file_groups=required_file_groups,
            )
            if not verified_inspection.complete or not verified_inspection.revision_verified:
                raise ModelAcquisitionError("Не удалось подтвердить закреплённую ревизию загруженной модели.")
            return ResolvedLocalModel(
                target_path,
                revision,
                verified_inspection.content_fingerprint,
                True,
            )
    except TimeoutError as exc:
        raise ModelAcquisitionError(f"Не удалось получить блокировку загрузки модели в {target_path}.") from exc


def acquire_local_model(
    *,
    model_id: str,
    revision: str,
    model_path: Path | None,
    default_model_path: Path,
    auto_download: bool,
    required_files: tuple[str, ...],
    required_file_groups: tuple[tuple[str, ...], ...],
    requires_hf_token: bool = False,
) -> ResolvedLocalModel:
    """Возвращает полную локальную модель либо явно и безопасно докачивает её."""
    normalized_revision = revision.strip().lower()
    if not _PINNED_REVISION_PATTERN.fullmatch(normalized_revision):
        raise ModelAcquisitionError("Для локальной модели требуется закреплённая 40-символьная SHA-ревизия.")

    default_path = default_model_path.expanduser().resolve()
    selected_path = (model_path or default_path).expanduser().resolve()
    if selected_path.exists() and not selected_path.is_dir():
        raise ModelAcquisitionError(f"Путь локальной модели не является каталогом: {selected_path}")

    inspection = inspect_model_directory(
        selected_path,
        model_id=model_id,
        revision=normalized_revision,
        required_files=required_files,
        required_file_groups=required_file_groups,
    )
    if inspection.structurally_complete:
        return ResolvedLocalModel(
            selected_path,
            normalized_revision,
            inspection.content_fingerprint,
            inspection.revision_verified,
        )

    if model_path is None:
        cached_snapshot = _cached_snapshot_path(
            default_path.parent,
            model_id,
            normalized_revision,
        ).resolve()
        cached_inspection = inspect_model_directory(
            cached_snapshot,
            model_id=model_id,
            revision=normalized_revision,
            required_files=required_files,
            required_file_groups=required_file_groups,
        )
        if cached_inspection.complete:
            return ResolvedLocalModel(
                cached_snapshot,
                normalized_revision,
                cached_inspection.content_fingerprint,
                True,
            )

    if not auto_download:
        details = "; ".join(inspection.issues)
        raise ModelAcquisitionError(
            f"Каталог модели {selected_path} неполон: {details}. "
            "Укажите полный model_path или явно включите auto_download_model."
        )

    return _download_model_snapshot(
        model_id=model_id,
        revision=normalized_revision,
        target_path=selected_path,
        required_files=required_files,
        required_file_groups=required_file_groups,
        requires_hf_token=requires_hf_token,
    )


def _describe_incomplete_download(cache_dir: Path, use_safetensors: bool) -> str:
    incomplete = list(cache_dir.rglob("*.incomplete"))

    if incomplete:
        return (
            "Обнаружены файлы с расширением '.incomplete' в кеше модели. "
            "Дождитесь завершения загрузки или удалите неполный кеш и повторите попытку."
        )

    if use_safetensors:
        safetensors = list(cache_dir.rglob("*.safetensors"))
        if not safetensors:
            return (
                "Файл 'model.safetensors' отсутствует в каталоге кеша. "
                "Вероятно, загрузка не завершилась - скачайте модель повторно перед запуском перевода."
            )

    return ""


def _handle_model_load_error(exc: Exception, cache_dir: Path, use_safetensors: bool) -> Never:
    hint = _describe_incomplete_download(cache_dir, use_safetensors)
    if not hint and isinstance(exc, RuntimeError):
        message = str(exc)
        if "torch.load" in message and "v2.6" in message:
            hint = (
                "Библиотека transformers пытается открыть бинарные веса через `torch.load`, "
                "что требует torch>=2.6. Убедитесь, что в кеше есть safetensors-веса."
            )
        elif _is_meta_error(exc):
            hint = "Часть параметров модели осталась на meta устройстве. Удалите кеш модели и скачайте веса заново."

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


def _has_cpu_or_disk_placement(model: PreTrainedModel) -> bool:
    device_map = getattr(model, "hf_device_map", None)
    if not isinstance(device_map, dict):
        return False
    for placement in device_map.values():
        normalized = str(placement).strip().casefold()
        if normalized == "disk" or normalized.startswith("cpu"):
            return True
    return False


class _ModelComponentLoader:
    def __init__(
        self,
        model_name: str,
        cache_dir: Path,
        model_class: type[PreTrainedModel],
        processor_class: type[Any],
        options: ModelLoadOptions,
    ) -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.model_class = model_class
        self.processor_class = processor_class
        self.options = options
        self.device = torch.device("cpu")
        self.quantization_config: Any | None = None

    def load(self) -> tuple[Any, Any, torch.device]:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._resolve_runtime()
        processor = self._load_processor()
        model = self._load_model()
        if not self.options.allow_cpu_fallback and _has_cpu_or_disk_placement(model):
            raise TranslatorLoadError(
                "Модель частично размещена на CPU или диске без явного разрешения перехода на CPU."
            )
        model.eval()
        sanitize_generation_config(model)
        return processor, model, self.device

    def _resolve_runtime(self) -> None:
        if self.options.require_int8 and self.options.allow_cpu_fallback:
            raise TranslatorLoadError("Строгий режим INT8 не поддерживает переход на CPU.")
        try:
            self.device, self.quantization_config = resolve_device_and_quantization(
                enable_cpu_offload=self.options.enable_cpu_offload,
                allow_quantization=self.options.allow_quantization,
                require_int8=self.options.require_int8,
            )
        except Int8UnavailableError as exc:
            raise TranslatorLoadError(str(exc)) from exc
        if self.device.type == "cpu" and not self.options.allow_cpu_fallback:
            raise TranslatorLoadError(
                "Устройство CUDA недоступно. Явно разрешите переход на CPU, если такой режим допустим."
            )

    def _common_pretrained_kwargs(self) -> dict[str, Any]:
        pretrained_kwargs: dict[str, Any] = {
            "cache_dir": str(self.cache_dir),
            "trust_remote_code": self.options.trust_remote_code,
        }
        if self.options.revision is not None:
            pretrained_kwargs["revision"] = self.options.revision
        if self.options.local_files_only:
            pretrained_kwargs["local_files_only"] = True
        return pretrained_kwargs

    def _load_processor(self) -> Any:
        processor_kwargs = dict(self.options.processor_kwargs or {})
        processor_kwargs.update(self._common_pretrained_kwargs())
        try:
            return self.processor_class.from_pretrained(
                self.model_name,
                **processor_kwargs,
            )
        except Exception as exc:  # pragma: no cover
            _handle_model_load_error(exc, self.cache_dir, self.options.use_safetensors)

    def _model_pretrained_kwargs(self, **overrides: Any) -> dict[str, Any]:
        pretrained_kwargs = dict(self.options.model_kwargs or {})
        pretrained_kwargs.update(self._common_pretrained_kwargs())
        pretrained_kwargs["use_safetensors"] = self.options.use_safetensors
        pretrained_kwargs.update(overrides)
        return pretrained_kwargs

    def _load_without_device_map(
        self,
        *,
        force_download: bool,
        use_dtype: bool,
    ) -> PreTrainedModel:
        fallback_kwargs: dict[str, Any] = {"low_cpu_mem_usage": False}
        if use_dtype:
            fallback_kwargs["dtype"] = torch.float16
        if force_download and not self.options.local_files_only:
            fallback_kwargs["force_download"] = True
        return self.model_class.from_pretrained(
            self.model_name,
            **self._model_pretrained_kwargs(**fallback_kwargs),
        )

    @staticmethod
    def _move_to_device(model: PreTrainedModel, target_device: torch.device) -> None:
        if _has_meta_parameters(model):
            raise RuntimeError("Параметры модели остались на meta device.")
        if target_device.type != "cpu":
            model.to(target_device)
        if _has_meta_parameters(model):
            raise RuntimeError("Параметры модели остались на meta device.")

    def _recover_meta_model(self) -> PreTrainedModel:
        use_dtype = self.device.type == "cuda"
        model = self._load_without_device_map(force_download=False, use_dtype=use_dtype)
        try:
            self._move_to_device(model, self.device)
            return model
        except Exception as exc:
            if not _is_meta_error(exc):
                raise
        model = self._load_without_device_map(force_download=True, use_dtype=use_dtype)
        self._move_to_device(model, self.device)
        return model

    def _load_quantized(self) -> PreTrainedModel:
        quantized_dtype = (self.options.model_kwargs or {}).get("dtype", torch.float16)
        try:
            return self.model_class.from_pretrained(
                self.model_name,
                **self._model_pretrained_kwargs(
                    quantization_config=self.quantization_config,
                    device_map=self.options.device_map,
                    dtype=quantized_dtype,
                ),
            )
        except Exception as exc:
            if self.options.require_int8 or not _is_meta_error(exc):
                raise
            return self._recover_meta_model()

    def _load_unquantized(self) -> PreTrainedModel:
        try:
            return self.model_class.from_pretrained(
                self.model_name,
                **self._model_pretrained_kwargs(device_map=self.options.device_map),
            )
        except Exception as exc:
            if not _is_meta_error(exc):
                raise
            return self._recover_meta_model()

    def _load_with_device_map(self) -> PreTrainedModel:
        if self.quantization_config is not None:
            return self._load_quantized()
        return self._load_unquantized()

    def _load_on_cpu(self) -> PreTrainedModel:
        cpu_device = torch.device("cpu")
        try:
            model = self._load_without_device_map(force_download=False, use_dtype=False)
            self._move_to_device(model, cpu_device)
            return model
        except Exception as exc:
            if not _should_cpu_fallback(exc):
                raise
        model = self._load_without_device_map(force_download=True, use_dtype=False)
        self._move_to_device(model, cpu_device)
        return model

    def _load_with_optional_cpu_fallback(self) -> PreTrainedModel:
        try:
            return self._load_with_device_map()
        except Exception as exc:
            can_fallback = self.options.allow_cpu_fallback and self.device.type == "cuda" and _should_cpu_fallback(exc)
            if not can_fallback:
                raise
        print("Загрузка на GPU не удалась, пробую CPU.", flush=True)
        model = self._load_on_cpu()
        self.device = torch.device("cpu")
        return model

    def _load_model(self) -> PreTrainedModel:
        try:
            return self._load_with_optional_cpu_fallback()
        except Exception as exc:  # pragma: no cover
            _handle_model_load_error(exc, self.cache_dir, self.options.use_safetensors)


def load_model_components(
    model_name: str,
    cache_dir: Path,
    model_class: type[PreTrainedModel],
    processor_class: type[Any],
    options: ModelLoadOptions | None = None,
) -> tuple[Any, Any, torch.device]:
    """Загружает процессор и веса на явно разрешённом устройстве."""
    loader = _ModelComponentLoader(
        model_name,
        cache_dir,
        model_class,
        processor_class,
        options or ModelLoadOptions(),
    )
    return loader.load()


__all__ = [
    "MODEL_MARKER_FILENAME",
    "ModelAcquisitionError",
    "ModelDirectoryInspection",
    "ModelLoadOptions",
    "ResolvedLocalModel",
    "TranslatorLoadError",
    "acquire_local_model",
    "inspect_model_directory",
    "load_model_components",
    "model_content_fingerprint",
]
