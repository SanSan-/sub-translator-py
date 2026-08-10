"""Отказобезопасный адаптер TranslateGemma через отдельный процесс."""

from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import Any, ClassVar

from sub_translate.models import TranslationOptions
from sub_translate.translators.base import TranslationError
from sub_translate.translators.local.common import normalize_language, validate_worker_translations
from sub_translate.translators.registry import (
    TRANSLATEGEMMA_MAX_BATCH_SIZE,
    TRANSLATEGEMMA_MODEL_ID,
    TRANSLATEGEMMA_MODEL_REVISION,
    TRANSLATEGEMMA_PROFILE_IDS,
    get_translator_metadata,
    resolve_registered_model,
)
from sub_translate.workers.external_worker import (
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
    ExternalWorkerError,
    PersistentNdjsonWorker,
)

MODEL_NAME = TRANSLATEGEMMA_MODEL_ID
MODEL_REVISION = TRANSLATEGEMMA_MODEL_REVISION
WORKER_MODULE = "sub_translate.workers.translategemma_worker"
SUPPORTED_DIRECTIONS = (("en", "ru"), ("ru", "en"))


class TranslateGemmaTranslator:
    """Сохраняет общий worker между файлами и изолирует сбои CUDA от приложения."""

    name = "translategemma"
    _worker: ClassVar[PersistentNdjsonWorker | None] = None
    _worker_python: ClassVar[Path | None] = None
    _worker_profile_id: ClassVar[str | None] = None
    _worker_lock: ClassVar[threading.RLock] = threading.RLock()

    def __init__(self, timeout: int = int(DEFAULT_REQUEST_TIMEOUT_SECONDS)) -> None:
        if timeout <= 0:
            raise ValueError("Таймаут TranslateGemma должен быть положительным.")
        if self.name not in TRANSLATEGEMMA_PROFILE_IDS:
            raise ValueError(f"Неизвестный профиль TranslateGemma: {self.name}")
        self._timeout = float(timeout)

    def translate_batch(self, texts: list[str], options: TranslationOptions) -> list[str]:
        if not texts:
            return []
        if not all(isinstance(text, str) for text in texts):
            raise TranslationError("Пачка TranslateGemma должна состоять только из строк.")
        source_lang, target_lang = _resolve_direction(options)
        metadata = get_translator_metadata(self.name)
        model_source = _resolve_model(self.name, metadata.model_revision, options)
        python_path = _resolve_worker_python(options)
        payload = {
            "profile_id": self.name,
            "model_id": metadata.model_id,
            "model_revision": model_source.revision,
            "model_path": str(model_source.path),
            "model_content_fingerprint": model_source.content_fingerprint,
            "source_lang": source_lang,
            "target_lang": target_lang,
        }
        with TranslateGemmaTranslator._worker_lock:
            worker = TranslateGemmaTranslator._worker_for(python_path, self.name)
            try:
                return _translate_chunks(worker, payload, texts, self._timeout)
            except ExternalWorkerError as exc:
                TranslateGemmaTranslator._discard_worker()
                raise TranslationError(_worker_error_message(exc)) from exc
            except TranslationError:
                TranslateGemmaTranslator._discard_worker()
                raise

    @classmethod
    def unload(cls) -> None:
        """Выгружает модель и завершает только принадлежащее ей дерево процессов."""
        owner = TranslateGemmaTranslator
        with owner._worker_lock:
            worker = owner._worker
            owner._worker = None
            owner._worker_python = None
            owner._worker_profile_id = None
            if worker is None:
                return
            worker.shutdown()

    @classmethod
    def _worker_for(cls, python_path: Path, profile_id: str) -> PersistentNdjsonWorker:
        if cls._worker is not None and cls._worker_python == python_path and cls._worker_profile_id == profile_id:
            return cls._worker
        cls._discard_worker()
        cls._worker = PersistentNdjsonWorker(
            python_path,
            WORKER_MODULE,
            timeout_seconds=DEFAULT_REQUEST_TIMEOUT_SECONDS,
        )
        cls._worker_python = python_path
        cls._worker_profile_id = profile_id
        return cls._worker

    @classmethod
    def _discard_worker(cls) -> None:
        worker = cls._worker
        cls._worker = None
        cls._worker_python = None
        cls._worker_profile_id = None
        if worker is not None:
            worker.abort()


def _resolve_model(
    profile_id: str,
    expected_revision: str | None,
    options: TranslationOptions,
) -> Any:
    try:
        model_source = resolve_registered_model(profile_id, options)
    except RuntimeError as exc:
        raise TranslationError(str(exc)) from exc
    if expected_revision is None or model_source.revision != expected_revision:
        raise TranslationError(
            f"TranslateGemma запускается только с закреплённой ревизией {expected_revision or 'не задана'}."
        )
    return model_source


def _translate_chunks(
    worker: PersistentNdjsonWorker,
    payload: dict[str, Any],
    texts: list[str],
    timeout_seconds: float,
) -> list[str]:
    translations: list[str] = []
    for start in range(0, len(texts), TRANSLATEGEMMA_MAX_BATCH_SIZE):
        chunk = texts[start : start + TRANSLATEGEMMA_MAX_BATCH_SIZE]
        response = worker.request(
            "translate",
            {**payload, "texts": chunk},
            timeout_seconds=timeout_seconds,
        )
        translations.extend(validate_worker_translations(response, chunk, "TranslateGemma"))
    return translations


def _resolve_worker_python(options: TranslationOptions) -> Path:
    python_path = Path(options.worker_python_path or sys.executable).expanduser().resolve()
    if not python_path.is_file():
        raise TranslationError(f"Python изолированной среды TranslateGemma не найден: {python_path}")
    return python_path


def _resolve_direction(options: TranslationOptions) -> tuple[str, str]:
    source_lang = normalize_language(options.source_lang, default="en")
    target_lang = normalize_language(options.target_lang, default="ru")
    if (source_lang, target_lang) not in SUPPORTED_DIRECTIONS:
        raise TranslationError(
            "TranslateGemma поддерживает только направления en -> ru и ru -> en "
            f"(получено {source_lang} -> {target_lang})."
        )
    return source_lang, target_lang


def _worker_error_message(error: ExternalWorkerError) -> str:
    if error.error_type in {"TranslateGemmaOutOfMemoryError", "OutOfMemoryError"}:
        return (
            "TranslateGemma не хватило видеопамяти. Изолированный процесс остановлен; "
            "следующий вызов запустит его заново."
        )
    if error.error_type == "TimeoutError":
        return (
            "TranslateGemma превысила таймаут. Изолированный процесс остановлен; следующий вызов запустит его заново."
        )
    if error.error_type in {"ProtocolError", "WorkerExited"}:
        return (
            "Изолированный процесс TranslateGemma завершился или нарушил протокол. "
            "Следующий вызов запустит его заново."
        )
    return f"Ошибка изолированной TranslateGemma: {error}"


class TranslateGemma12BTranslator(TranslateGemmaTranslator):
    """Профиль TranslateGemma 12B на общем изолированном адаптере."""

    name = "translategemma-12b"


__all__ = [
    "MODEL_NAME",
    "MODEL_REVISION",
    "SUPPORTED_DIRECTIONS",
    "WORKER_MODULE",
    "TranslateGemma12BTranslator",
    "TranslateGemmaTranslator",
]
