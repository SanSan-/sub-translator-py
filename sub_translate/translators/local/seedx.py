"""Отказобезопасный адаптер Seed-X через отдельный процесс."""

from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import Any, ClassVar

from sub_translate.models import TranslationOptions
from sub_translate.translators.base import TranslationError
from sub_translate.translators.local.common import normalize_language, validate_worker_translations
from sub_translate.translators.registry import (
    SEEDX_MODEL_ID,
    SEEDX_MODEL_REVISION,
    resolve_registered_model,
)
from sub_translate.workers.external_worker import (
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
    ExternalWorkerError,
    PersistentNdjsonWorker,
)

MODEL_NAME = SEEDX_MODEL_ID
MODEL_REVISION = SEEDX_MODEL_REVISION
WORKER_MODULE = "sub_translate.workers.seedx_worker"
SUPPORTED_DIRECTIONS = (("en", "ru"), ("ru", "en"))


class SeedXTranslator:
    """Переиспользует один процесс и изолирует сбои CUDA от приложения."""

    name = "seedx"
    _worker: ClassVar[PersistentNdjsonWorker | None] = None
    _worker_python: ClassVar[Path | None] = None
    _worker_lock: ClassVar[threading.RLock] = threading.RLock()

    def __init__(self, timeout: int = int(DEFAULT_REQUEST_TIMEOUT_SECONDS)) -> None:
        if timeout <= 0:
            raise ValueError("Таймаут Seed-X должен быть положительным.")
        self._timeout = float(timeout)

    def translate_batch(self, texts: list[str], options: TranslationOptions) -> list[str]:
        if not texts:
            return []
        if not all(isinstance(text, str) for text in texts):
            raise TranslationError("Пачка Seed-X должна состоять только из строк.")

        source_lang, target_lang = _resolve_direction(options)
        model_source = _resolve_model(options)
        python_path = _resolve_worker_python(options)
        common_payload = {
            "model_id": SEEDX_MODEL_ID,
            "model_revision": model_source.revision,
            "model_path": str(model_source.path),
            "model_content_fingerprint": model_source.content_fingerprint,
            "source_lang": source_lang,
            "target_lang": target_lang,
        }
        translations: list[str] = []
        with type(self)._worker_lock:
            worker = type(self)._worker_for(python_path)
            try:
                for text in texts:
                    response = worker.request(
                        "translate",
                        {**common_payload, "texts": [text]},
                        timeout_seconds=self._timeout,
                    )
                    translations.extend(validate_worker_translations(response, [text], "Seed-X"))
            except ExternalWorkerError as exc:
                type(self)._discard_worker()
                raise TranslationError(_worker_error_message(exc)) from exc
            except TranslationError:
                type(self)._discard_worker()
                raise
        return translations

    @classmethod
    def unload(cls) -> None:
        """Выгружает модель и завершает только принадлежащее ей дерево процессов."""
        with cls._worker_lock:
            worker = cls._worker
            cls._worker = None
            cls._worker_python = None
            if worker is None:
                return
            worker.shutdown()

    @classmethod
    def _worker_for(cls, python_path: Path) -> PersistentNdjsonWorker:
        if cls._worker is not None and cls._worker_python == python_path:
            return cls._worker
        cls._discard_worker()
        cls._worker = PersistentNdjsonWorker(
            python_path,
            WORKER_MODULE,
            timeout_seconds=DEFAULT_REQUEST_TIMEOUT_SECONDS,
        )
        cls._worker_python = python_path
        return cls._worker

    @classmethod
    def _discard_worker(cls) -> None:
        worker = cls._worker
        cls._worker = None
        cls._worker_python = None
        if worker is not None:
            worker.abort()


def _resolve_model(options: TranslationOptions) -> Any:
    try:
        model_source = resolve_registered_model("seedx", options)
    except RuntimeError as exc:
        raise TranslationError(str(exc)) from exc
    if model_source.revision != SEEDX_MODEL_REVISION:
        raise TranslationError(f"Seed-X запускается только с закреплённой ревизией {SEEDX_MODEL_REVISION}.")
    return model_source


def _resolve_worker_python(options: TranslationOptions) -> Path:
    python_path = Path(options.worker_python_path or sys.executable).expanduser().resolve()
    if not python_path.is_file():
        raise TranslationError(f"Python изолированной среды Seed-X не найден: {python_path}")
    return python_path


def _resolve_direction(options: TranslationOptions) -> tuple[str, str]:
    source_lang = normalize_language(options.source_lang, default="en")
    target_lang = normalize_language(options.target_lang, default="ru")
    if (source_lang, target_lang) not in SUPPORTED_DIRECTIONS:
        raise TranslationError(
            f"Seed-X поддерживает только направления en -> ru и ru -> en (получено {source_lang} -> {target_lang})."
        )
    return source_lang, target_lang


def _worker_error_message(error: ExternalWorkerError) -> str:
    if error.error_type in {"SeedXOutOfMemoryError", "OutOfMemoryError"}:
        return "Seed-X не хватило видеопамяти. Изолированный процесс остановлен; следующий вызов запустит его заново."
    if error.error_type == "TimeoutError":
        return "Seed-X превысила таймаут. Изолированный процесс остановлен; следующий вызов запустит его заново."
    if error.error_type in {"ProtocolError", "WorkerExited"}:
        return "Изолированный процесс Seed-X завершился или нарушил протокол. Следующий вызов запустит его заново."
    return f"Ошибка изолированной Seed-X: {error}"


__all__ = [
    "MODEL_NAME",
    "MODEL_REVISION",
    "SUPPORTED_DIRECTIONS",
    "WORKER_MODULE",
    "SeedXTranslator",
]
