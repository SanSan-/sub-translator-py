from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
import uuid
import warnings
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass, field
from functools import partial
from ipaddress import ip_address
from pathlib import Path
from typing import Any, Self
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from sub_translate import __version__
from sub_translate.constants import (
    LOGS_DIR,
    SMART_SPLIT_MAX_CHARS,
    SMART_SPLIT_MAX_DURATION_MS,
    SMART_SPLIT_MAX_GAP_MS,
    SMART_SPLIT_MAX_LINES,
    SMART_SPLIT_MAX_WORDS,
)
from sub_translate.dictionaries.languages import LANGS, get_code
from sub_translate.models import SmartSplitSettings
from sub_translate.service import (
    ProcessingResult,
    ProcessingSettings,
    ProcessingStatus,
    build_output_cache_identity,
    process_subtitle_batch,
    resolve_api,
    resolve_format,
    resolve_io_paths,
)
from sub_translate.translators.agent import (
    SYSTEM_PROMPT,
    attach_log_handler,
    detach_log_handler,
    get_agent_model_name,
    is_openai_configured,
    set_system_prompt,
    set_user_prompt_template,
)
from sub_translate.translators.agent_prompts import DEFAULT_PROMPT_VARIANT, get_prompt_template
from sub_translate.translators.base import TranslationError
from sub_translate.translators.google_web import normalize_google_tld
from sub_translate.translators.registry import (
    DEFAULT_TRANSLATOR_ID,
    TranslatorMetadata,
    get_translator_metadata,
    list_translator_metadata,
    resolve_translator_id,
    unload_all_local_translators,
)
from sub_translate.utils.env_utils import load_env
from sub_translate.utils.io_utils import read_text, split_lines
from sub_translate.utils.logging_utils import configure_rotating_logger
from sub_translate.utils.path_utils import split_lang_suffix
from sub_translate.utils.subtitle_cache import has_cached_output, prompt_signature, validate_cached_lines
from sub_translate.web.picker import (
    SUBTITLE_EXTENSIONS,
    PickerError,
    PickSelection,
    collect_subtitle_paths,
    filter_subtitle_paths,
    pick_paths,
)
from sub_translate.web.preparation import PreparationTracker

ROOT_DIR = Path(__file__).resolve().parent
STATIC_DIR = ROOT_DIR / "static"
load_env()

SUBTITLE_EXTS = SUBTITLE_EXTENSIONS
_MUTATING_HTTP_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})
_TEST_CLIENT_HOST = "testclient"
_TEST_CLIENT_EXTENSION = "http.response.debug"
_LOCAL_ACCESS_ERROR = "Доступ разрешён только через локальный интерфейс."
_ORIGIN_ACCESS_ERROR = "Источник запроса не совпадает с локальным интерфейсом."
_REQUEST_VALIDATION_ERROR = "Запрос не прошёл проверку."
_PREPARATION_IN_PROGRESS_ERROR = "Подготовка файлов уже выполняется."


class _SubtitlePreparationError(ValueError):
    """Безопасная ошибка подготовки одной карточки файла."""


MAX_STORED_JOBS = 32
MAX_EVENTS_PER_JOB = 512
TERMINAL_JOB_TTL_SECONDS = 15 * 60

app = FastAPI(title="Sub Translator", version=__version__)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def _is_loopback_address(value: str) -> bool:
    try:
        return ip_address(value).is_loopback
    except ValueError:
        return False


def _is_loopback_client(request: Request) -> bool:
    client = request.client
    if client is None:
        return False
    if client.host == _TEST_CLIENT_HOST:
        extensions = request.scope.get("extensions")
        return isinstance(extensions, dict) and _TEST_CLIENT_EXTENSION in extensions
    return _is_loopback_address(client.host)


def _single_header(request: Request, name: bytes) -> str | None:
    values = [value for key, value in request.scope.get("headers", []) if key.lower() == name]
    if len(values) != 1:
        return None
    try:
        decoded = values[0].decode("ascii")
    except UnicodeDecodeError:
        return None
    if not decoded or decoded != decoded.strip() or any(character.isspace() for character in decoded):
        return None
    return decoded


def _is_loopback_host(host_header: str) -> bool:
    if any(character in host_header for character in "/\\@?#%"):
        return False
    try:
        parsed = urlsplit(f"//{host_header}")
        _ = parsed.port
    except ValueError:
        return False
    if not parsed.hostname or parsed.path or parsed.query or parsed.fragment:
        return False
    hostname = parsed.hostname.lower()
    return hostname == "localhost" or _is_loopback_address(hostname)


def _has_matching_origin(request: Request, host_header: str) -> bool:
    origin_headers = [value for key, value in request.scope.get("headers", []) if key.lower() == b"origin"]
    if not origin_headers:
        return True
    if len(origin_headers) != 1:
        return False
    try:
        origin = origin_headers[0].decode("ascii")
    except UnicodeDecodeError:
        return False
    return origin == f"{request.scope.get('scheme', 'http')}://{host_header}"


@app.middleware("http")
async def restrict_to_local_interface(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    host_header = _single_header(request, b"host")
    if not _is_loopback_client(request) or host_header is None or not _is_loopback_host(host_header):
        return JSONResponse(status_code=403, content={"detail": _LOCAL_ACCESS_ERROR})
    if request.method in _MUTATING_HTTP_METHODS and not _has_matching_origin(request, host_header):
        return JSONResponse(status_code=403, content={"detail": _ORIGIN_ACCESS_ERROR})
    return await call_next(request)


@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(
    _request: Request,
    _exc: RequestValidationError,
) -> JSONResponse:
    """Возвращает ошибки схемы без исходных значений запроса."""
    return JSONResponse(status_code=422, content={"detail": _REQUEST_VALIDATION_ERROR})


_WEB_LOGGER = configure_rotating_logger(
    "sub_translate_web",
    LOGS_DIR / "sub_translate_web.log",
    verbose=False,
)
_WEB_LOGGER.propagate = False


class StrictRequestModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TranslationSettings(StrictRequestModel):
    source_lang: str = "en"
    target_lang: str = "ru"
    api: str = Field(...)
    batch_size: int = Field(default=21, ge=1)
    threads: int = Field(default=3, ge=1)
    allow_cpu_fallback: bool = False
    smart_split: bool = False
    force: bool = False
    smart_split_max_lines: int = Field(default=SMART_SPLIT_MAX_LINES, ge=1)
    smart_split_max_words: int = Field(default=SMART_SPLIT_MAX_WORDS, ge=1)
    smart_split_max_chars: int = Field(default=SMART_SPLIT_MAX_CHARS, ge=1)
    smart_split_max_gap_ms: int = Field(default=SMART_SPLIT_MAX_GAP_MS, ge=1)
    smart_split_max_duration_ms: int = Field(default=SMART_SPLIT_MAX_DURATION_MS, ge=1)
    tld: str = "com"
    timeout: int = Field(default=30, ge=1)
    request_delay_ms: int = Field(default=350, ge=0)
    model_path: str | None = None
    model_revision: str | None = None
    worker_python_path: str | None = None
    auto_download_model: bool = False
    agent_model: str | None = None
    agent_system_prompt: str | None = None
    agent_prompt: str | None = None
    agent_system_prompt_file: str | None = None
    agent_prompt_file: str | None = None
    verbose: bool = False

    @model_validator(mode="after")
    def apply_profile_timeout_default(self) -> Self:
        if "timeout" not in self.model_fields_set:
            self.timeout = get_translator_metadata(self.api).default_timeout_seconds
        return self

    @field_validator("api", mode="before")
    @classmethod
    def validate_api(cls, value: object) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Переводчик должен быть указан.")
        canonical_id = resolve_translator_id(value)
        try:
            metadata = get_translator_metadata(canonical_id)
        except TranslationError as exc:
            raise ValueError(str(exc)) from None
        if metadata.deprecated:
            raise ValueError(f"Переводчик {canonical_id} больше не поддерживается.")
        return canonical_id

    @field_validator("tld", mode="before")
    @classmethod
    def validate_tld(cls, value: object) -> str:
        return normalize_google_tld(value)

    @field_validator("model_path", "model_revision", "worker_python_path", mode="before")
    @classmethod
    def normalize_optional_model_value(cls, value: object) -> object:
        if not isinstance(value, str):
            return value
        normalized = value.strip()
        return normalized or None


class PickRequest(StrictRequestModel):
    kind: str = Field(..., pattern="^(file|folder)$")
    settings: TranslationSettings | None = None


class RefreshRequest(StrictRequestModel):
    paths: list[str]
    settings: TranslationSettings
    recursive: bool = False


class TranslateRequest(StrictRequestModel):
    paths: list[str]
    settings: TranslationSettings


class BoundedEventQueue(queue.Queue[dict[str, Any]]):
    """Ограниченная очередь, которая не блокирует производителя и сохраняет итог."""

    def __init__(self, maxsize: int) -> None:
        if maxsize <= 0:
            raise ValueError("Размер очереди событий должен быть положительным.")
        super().__init__(maxsize=maxsize)
        self._terminal_published = False

    @staticmethod
    def _is_terminal(event: dict[str, Any]) -> bool:
        return event.get("type") == "done"

    def _discard_oldest(self) -> None:
        self._get()
        if self.unfinished_tasks > 0:
            self.unfinished_tasks -= 1
            if self.unfinished_tasks == 0:
                self.all_tasks_done.notify_all()
        self.not_full.notify()

    def put(
        self,
        item: dict[str, Any],
        block: bool = True,
        timeout: float | None = None,
    ) -> None:
        del block, timeout
        with self.not_full:
            if self._terminal_published:
                return
            while self._qsize() >= self.maxsize:
                self._discard_oldest()
            self._put(item)
            if self._is_terminal(item):
                self._terminal_published = True
            self.unfinished_tasks += 1
            self.not_empty.notify()


class QueueLogHandler(logging.Handler):
    def __init__(
        self,
        events: queue.Queue[dict[str, Any]],
        on_emit: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        super().__init__()
        self._events = events
        self._on_emit = on_emit

    def emit(self, record: logging.LogRecord) -> None:
        message = self.format(record)
        event = {"type": "log", "message": message}
        self._events.put(event)
        if self._on_emit is not None:
            self._on_emit(event)


@dataclass
class TranslationJob:
    job_id: str
    paths: list[Path]
    settings: TranslationSettings
    events: queue.Queue[dict[str, Any]]
    thread: threading.Thread | None
    status: str = "running"
    completed: bool = False
    job_total: int = 0
    file_states: dict[str, dict[str, Any]] = field(default_factory=dict)
    log_lines: list[str] = field(default_factory=list)
    prompt_signatures: dict[str, str] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)
    created_at: float = field(default_factory=time.monotonic)
    completed_at: float | None = None


_jobs: dict[str, TranslationJob] = {}
_jobs_lock = threading.Lock()
_active_job_id: str | None = None
_active_job_lock = threading.Lock()
preparation_registry = PreparationTracker()
_picker_refresh_lock = threading.Lock()

LOG_HISTORY_LIMIT = 250
_TERMINAL_FILE_STATES = frozenset({"cached", "done", "error"})


def _terminal_timestamp(job: TranslationJob) -> float | None:
    if not job.completed:
        return None
    return job.completed_at if job.completed_at is not None else job.created_at


def _prune_jobs_locked(now: float) -> None:
    expired_ids = [
        job_id
        for job_id, job in _jobs.items()
        if (completed_at := _terminal_timestamp(job)) is not None and now - completed_at >= TERMINAL_JOB_TTL_SECONDS
    ]
    for job_id in expired_ids:
        del _jobs[job_id]

    overflow = len(_jobs) - max(1, MAX_STORED_JOBS)
    if overflow <= 0:
        return
    terminal_jobs = sorted(
        (
            (completed_at, job.created_at, job_id)
            for job_id, job in _jobs.items()
            if (completed_at := _terminal_timestamp(job)) is not None
        ),
    )
    for _completed_at, _created_at, job_id in terminal_jobs[:overflow]:
        del _jobs[job_id]


def _record_log_event(job: TranslationJob, event: dict[str, Any]) -> None:
    message = str(event.get("message", "")).rstrip()
    if not message:
        return
    job.log_lines.append(message)
    if len(job.log_lines) > LOG_HISTORY_LIMIT:
        job.log_lines = job.log_lines[-LOG_HISTORY_LIMIT:]


def _record_total_event(job: TranslationJob, event: dict[str, Any]) -> None:
    total = event.get("total")
    if isinstance(total, int):
        job.job_total = total


def _record_file_event(job: TranslationJob, event: dict[str, Any]) -> None:
    path = event.get("path")
    if not isinstance(path, str) or not path:
        return
    state = job.file_states.get(path, {})
    for key in ("state", "progress", "output", "error"):
        if key in event:
            state[key] = event[key]
    job.file_states[path] = state


_EVENT_RECORDERS: dict[str, Callable[[TranslationJob, dict[str, Any]], None]] = {
    "log": _record_log_event,
    "job": _record_total_event,
    "file": _record_file_event,
}


def _record_job_event(job: TranslationJob, event: dict[str, Any]) -> None:
    recorder = _EVENT_RECORDERS.get(str(event.get("type", "")))
    if recorder is None:
        return
    with job.lock:
        recorder(job, event)


def _emit_job_event(job: TranslationJob, event: dict[str, Any]) -> None:
    _record_job_event(job, event)
    job.events.put(event)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


def _get_job(job_id: str) -> TranslationJob | None:
    with _jobs_lock:
        _prune_jobs_locked(time.monotonic())
        return _jobs.get(job_id)


@app.get("/api/active-job")
def active_job() -> dict[str, Any]:
    with _active_job_lock:
        job_id = _active_job_id
    if job_id is None:
        return {"active": False}
    job = _get_job(job_id)
    if job is None:
        return {"active": False}
    return {"active": True, **_build_job_snapshot(job)}


def _build_job_snapshot(job: TranslationJob) -> dict[str, Any]:
    with job.lock:
        paths = list(job.paths)
        file_states = {path: dict(state) for path, state in job.file_states.items()}
        logs = list(job.log_lines)
        total = job.job_total or len(paths)
        completed = job.completed
        status = job.status
    items = [_build_snapshot_item(path, file_states.get(str(path), {})) for path in paths]
    done = sum(1 for item in items if item.get("state") in {"cached", "done", "error"})
    return {
        "job_id": job.job_id,
        "completed": completed,
        "status": status,
        "items": items,
        "logs": logs,
        "total": total,
        "done": done,
    }


def _build_snapshot_item(path: Path, state: Mapping[str, Any]) -> dict[str, Any]:
    item: dict[str, Any] = {
        "name": path.name,
        "path": str(path),
        "format": path.suffix.casefold().lstrip("."),
        "state": state.get("state", "queued"),
        "progress": state.get("progress", 0),
    }
    for key in ("output", "error"):
        if key in state:
            item[key] = state[key]
    if state.get("state") == "cached":
        item["cached"] = True
    return item


@app.get(
    "/api/jobs/{job_id}",
    responses={404: {"description": "Задача не найдена."}},
)
def job_snapshot(job_id: str) -> dict[str, Any]:
    job = _get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Задача не найдена.")
    return _build_job_snapshot(job)


@app.get("/api/preparation-status")
def preparation_status() -> dict[str, Any]:
    """Возвращает ход выбора и рекурсивного обхода каталога."""
    return preparation_registry.snapshot()


def _build_language_options(include_auto: bool) -> list[dict[str, str]]:
    options: list[dict[str, str]] = []
    if include_auto and "auto" in LANGS:
        options.append({"value": "auto", "label": LANGS["auto"]})
    for key, label in LANGS.items():
        if key == "auto":
            continue
        options.append({"value": key, "label": label})
    return options


def _language_option(code: str) -> dict[str, str]:
    labels = {
        "auto": "Автоматически",
        "en": "Английский",
        "ru": "Русский",
    }
    return {"value": code, "label": labels.get(code, LANGS.get(code, code))}


def _profile_languages(metadata: TranslatorMetadata) -> dict[str, list[dict[str, str]]]:
    if metadata.supported_directions:
        source_codes = tuple(dict.fromkeys(source for source, _target in metadata.supported_directions))
        target_codes = tuple(dict.fromkeys(target for _source, target in metadata.supported_directions))
        return {
            "source": [_language_option(code) for code in source_codes],
            "target": [_language_option(code) for code in target_codes],
        }
    if metadata.local:
        local_languages = [_language_option("en"), _language_option("ru")]
        return {"source": local_languages, "target": local_languages}
    return {
        "source": _build_language_options(include_auto=True),
        "target": _build_language_options(include_auto=False),
    }


def _huggingface_token_configured() -> bool:
    return bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN"))


def _safe_translator_metadata(metadata: TranslatorMetadata) -> dict[str, Any]:
    result: dict[str, Any] = {
        "id": metadata.id,
        "label": metadata.display_name,
        "local": metadata.local,
        "thread_safe": metadata.thread_safe,
        "runtime_kind": metadata.runtime_kind,
        "supports_cpu_fallback": metadata.supports_cpu_fallback,
        "default_timeout_seconds": metadata.default_timeout_seconds,
        "supported_directions": [
            {"source": source, "target": target} for source, target in metadata.supported_directions
        ],
    }
    if metadata.local:
        result["model"] = {
            "id": metadata.model_id,
            "revision": metadata.model_revision,
            "quantization": metadata.quantization,
            "requires_access_token": metadata.requires_hf_token,
            "access_token_configured": (_huggingface_token_configured() if metadata.requires_hf_token else True),
        }
    return result


def _product_translator_metadata() -> tuple[TranslatorMetadata, ...]:
    return tuple(metadata for metadata in list_translator_metadata() if not metadata.deprecated)


def _build_ui_config() -> dict[str, Any]:
    translators = _product_translator_metadata()
    return {
        "defaults": {
            "api": DEFAULT_TRANSLATOR_ID,
            "source_lang": "en",
            "target_lang": "ru",
            "batch_size": 21,
            "threads": 3,
            "allow_cpu_fallback": False,
            "force": False,
            "smart_split_max_lines": SMART_SPLIT_MAX_LINES,
            "smart_split_max_words": SMART_SPLIT_MAX_WORDS,
            "smart_split_max_chars": SMART_SPLIT_MAX_CHARS,
            "smart_split_max_gap_ms": SMART_SPLIT_MAX_GAP_MS,
            "smart_split_max_duration_ms": SMART_SPLIT_MAX_DURATION_MS,
            "tld": "com",
            "timeout": 30,
            "request_delay_ms": 350,
            "auto_download_model": False,
        },
        "translators": [_safe_translator_metadata(metadata) for metadata in translators],
        "languages": {metadata.id: _profile_languages(metadata) for metadata in translators},
        "agent_prompts": {
            "system": SYSTEM_PROMPT,
            "user": get_prompt_template(DEFAULT_PROMPT_VARIANT),
        },
        "agent_settings": {
            "model": get_agent_model_name(),
            "configured": is_openai_configured(),
        },
    }


@app.get("/api/ui-config")
def ui_config() -> dict[str, Any]:
    return _build_ui_config()


def _get_logger(name: str, verbose: bool) -> logging.Logger:
    logger = configure_rotating_logger(
        name,
        LOGS_DIR / f"{name}.log",
        verbose=verbose,
    )
    logger.propagate = False
    return logger


def _is_subtitle(path: Path) -> bool:
    return path.suffix.lower() in SUBTITLE_EXTS


def _has_target_suffix(path: Path, target_lang: str) -> bool:
    if not target_lang:
        return False
    resolved_target = (get_code(target_lang) or target_lang).lower()
    if not resolved_target:
        return False
    stem = path.stem
    if "." not in stem:
        return False
    _, suffix_lang = split_lang_suffix(stem)
    if suffix_lang:
        return suffix_lang.lower() == resolved_target
    candidate = stem.rsplit(".", 1)[1].lower()
    return candidate == resolved_target


def _is_folder_source(path: Path, target_lang: str) -> bool:
    return _is_subtitle(path) and not _has_target_suffix(path, target_lang)


def _iter_subtitles(
    folder: Path,
    _source_lang: str,
    target_lang: str,
    *,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[Path, ...]:
    return collect_subtitle_paths(
        folder,
        recursive=True,
        accept_path=lambda path: _is_folder_source(path, target_lang),
        progress_callback=progress_callback,
    )


def _build_items(
    paths: Iterable[Path],
    settings: TranslationSettings,
    *,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    processing = _to_processing_settings(settings)
    api = processing.api
    selected_paths = list(paths)
    total = len(selected_paths)
    items: list[dict[str, Any]] = []
    for index, path in enumerate(selected_paths, start=1):
        try:
            items.append(_build_item(path, processing, api))
        except Exception as exc:
            _WEB_LOGGER.exception(
                "Не удалось подготовить файл %s (%s).",
                path,
                type(exc).__name__,
            )
            items.append(_build_error_item(path, exc))
        if progress_callback is not None:
            progress_callback(
                {
                    "phase": "queueing",
                    "discovered": total,
                    "processed": index,
                    "total": total,
                }
            )
    return items


def _build_item(
    path: Path,
    processing: ProcessingSettings,
    api: str,
) -> dict[str, Any]:
    file_format = resolve_format(path, None)
    lines = split_lines(read_text(path))
    if not validate_cached_lines(lines, file_format):
        raise _SubtitlePreparationError(
            f"Содержимое не соответствует формату {file_format.value.upper()}.",
        )
    source_lang, output_path = resolve_io_paths(
        path,
        None,
        processing.source_lang,
        processing.target_lang,
        api,
        file_format,
    )
    cached = output_path.exists()
    if not cached:
        try:
            identity = build_output_cache_identity(
                path,
                file_format,
                processing,
                source_lang=source_lang,
            )
            cached = has_cached_output(identity, _WEB_LOGGER)
        except OSError, ValueError:
            path.stat()
            cached = False
    return {
        "name": path.name,
        "path": str(path),
        "format": file_format.value,
        "cached": cached,
        "output": str(output_path),
    }


def _build_error_item(path: Path, error: Exception) -> dict[str, Any]:
    if isinstance(error, FileNotFoundError):
        message = "Файл не найден или был перемещён."
    elif isinstance(error, PermissionError):
        message = "Нет доступа к файлу."
    elif isinstance(error, UnicodeError):
        message = "Файл не является корректным UTF-8."
    elif isinstance(error, _SubtitlePreparationError):
        message = str(error) or "Содержимое субтитров повреждено."
    elif isinstance(error, ValueError):
        message = "Не удалось проверить параметры файла."
    else:
        message = f"Не удалось подготовить файл ({type(error).__name__})."
    return {
        "name": path.name,
        "path": str(path),
        "format": path.suffix.casefold().lstrip("."),
        "cached": False,
        "state": "error",
        "progress": 100,
        "error": message,
    }


def _begin_preparation(operation: str, *, phase: str, message: str) -> str:
    operation_id = preparation_registry.begin(operation, phase=phase, message=message)
    _WEB_LOGGER.info(message)
    return operation_id


def _update_preparation(operation_id: str, event: Mapping[str, Any]) -> None:
    message_value = event.get("message")
    message = str(message_value) if message_value else None
    preparation_registry.update(
        operation_id,
        phase=str(event["phase"]) if event.get("phase") else None,
        discovered=_optional_int(event.get("discovered")),
        processed=_optional_int(event.get("processed")),
        total=_optional_int(event.get("total")),
        message=message,
    )
    if message:
        _WEB_LOGGER.info(message)


def _finish_preparation(operation_id: str, message: str) -> None:
    preparation_registry.finish(operation_id, message=message)
    _WEB_LOGGER.info(message)


def _fail_preparation(operation_id: str, error: object) -> None:
    message = str(error) or error.__class__.__name__
    preparation_registry.fail(operation_id, message=message)
    _WEB_LOGGER.error("Подготовка субтитров завершилась ошибкой: %s", message)


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except TypeError, ValueError:
        return None


def _selection_payload(
    selection: PickSelection,
    items: list[dict[str, Any]],
    *,
    recursive: bool,
) -> dict[str, Any]:
    return {
        "mode": selection.mode,
        "path": str(selection.folder) if selection.folder else "",
        "paths": [str(path) for path in selection.paths],
        "items": items,
        "cancelled": selection.folder is None and not selection.paths,
        "recursive": recursive,
    }


def _normalize_refresh_path(value: str) -> Path:
    return Path(os.path.abspath(Path(value).expanduser()))


def _expand_refresh_paths(
    payload: RefreshRequest,
    *,
    progress_callback: Callable[[dict[str, Any]], None] | None,
) -> tuple[tuple[Path, ...], Path | None, bool]:
    source_paths = [_normalize_refresh_path(value) for value in payload.paths]
    candidates: list[Path] = []
    folders: list[Path] = []
    for path in source_paths:
        if path.is_dir() or (payload.recursive and not _is_subtitle(path)):
            folders.append(path)
            candidates.extend(
                _iter_subtitles(
                    path,
                    payload.settings.source_lang,
                    payload.settings.target_lang,
                    progress_callback=progress_callback,
                )
            )
        else:
            candidates.append(path)
    paths = filter_subtitle_paths(candidates)
    folder = folders[0] if len(source_paths) == 1 and len(folders) == 1 else None
    return paths, folder, bool(folders) or payload.recursive


@app.post(
    "/api/pick",
    responses={500: {"description": "Системный диалог выбора недоступен."}},
)
def pick(payload: PickRequest) -> dict[str, Any]:
    settings = payload.settings or TranslationSettings(api=DEFAULT_TRANSLATOR_ID)
    recursive = payload.kind == "folder"
    with _picker_refresh_lock:
        operation_id = _begin_preparation(
            "pick",
            phase="dialog",
            message="Начат выбор локальных субтитров.",
        )
        try:
            selection = pick_paths(
                payload.kind,
                recursive=recursive,
                accept_path=(lambda path: _is_folder_source(path, settings.target_lang) if recursive else True),
                progress_callback=lambda event: _update_preparation(operation_id, event),
            )
            if selection.folder is None and not selection.paths:
                _finish_preparation(operation_id, "Выбор субтитров отменён.")
                return _selection_payload(selection, [], recursive=recursive)
            items = _build_items(
                selection.paths,
                settings,
                progress_callback=lambda event: _update_preparation(operation_id, event),
            )
        except PickerError as exc:
            _fail_preparation(operation_id, exc)
            _WEB_LOGGER.exception("Системный выбор субтитров завершился ошибкой.")
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        except Exception as exc:
            _fail_preparation(operation_id, "Не удалось завершить выбор локальных субтитров.")
            _WEB_LOGGER.exception("Необработанная ошибка системного выбора субтитров.")
            raise HTTPException(
                status_code=500,
                detail="Не удалось завершить выбор локальных субтитров.",
            ) from exc
        _finish_preparation(operation_id, f"Подготовка завершена: файлов — {len(items)}.")
        return _selection_payload(selection, items, recursive=recursive)


@app.post(
    "/api/refresh",
    responses={
        400: {"description": "Выбранный каталог недоступен или некорректен."},
        500: {"description": "Не удалось обновить выбранные субтитры."},
    },
)
def refresh(payload: RefreshRequest) -> dict[str, Any]:
    with _picker_refresh_lock:
        operation_id = _begin_preparation(
            "refresh",
            phase="collecting",
            message="Начато обновление выбранных субтитров.",
        )
        try:
            paths, folder, recursive = _expand_refresh_paths(
                payload,
                progress_callback=lambda event: _update_preparation(operation_id, event),
            )
            items = _build_items(
                paths,
                payload.settings,
                progress_callback=lambda event: _update_preparation(operation_id, event),
            )
        except PickerError as exc:
            _fail_preparation(operation_id, exc)
            _WEB_LOGGER.exception("Обновление каталога субтитров завершилось ошибкой.")
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            _fail_preparation(operation_id, "Не удалось обновить выбранные субтитры.")
            _WEB_LOGGER.exception("Необработанная ошибка обновления выбранных субтитров.")
            raise HTTPException(
                status_code=500,
                detail="Не удалось обновить выбранные субтитры.",
            ) from exc
        _finish_preparation(operation_id, f"Обновление завершено: файлов — {len(items)}.")
        return {
            "path": str(folder) if folder else "",
            "paths": [str(path) for path in paths],
            "items": items,
            "recursive": recursive,
        }


def _resolve_agent_prompts(settings: TranslationSettings) -> tuple[str, str, str]:
    system_prompt = SYSTEM_PROMPT
    user_prompt = get_prompt_template(DEFAULT_PROMPT_VARIANT)
    user_variant = DEFAULT_PROMPT_VARIANT
    if settings.agent_system_prompt:
        system_prompt = settings.agent_system_prompt
    elif settings.agent_system_prompt_file:
        system_path = Path(settings.agent_system_prompt_file).expanduser().resolve()
        if not system_path.is_file():
            raise FileNotFoundError(f"Системный промпт не найден: {system_path}")
        system_prompt = read_text(system_path)
    if settings.agent_prompt:
        user_prompt = settings.agent_prompt
        user_variant = "ui"
    elif settings.agent_prompt_file:
        user_path = Path(settings.agent_prompt_file).expanduser().resolve()
        if not user_path.is_file():
            raise FileNotFoundError(f"Промпт агента не найден: {user_path}")
        user_prompt = read_text(user_path)
        user_variant = "file"
    return system_prompt, user_prompt, user_variant


def _apply_agent_prompts(settings: TranslationSettings) -> dict[str, str]:
    system_prompt, user_prompt, user_variant = _resolve_agent_prompts(settings)
    set_system_prompt(system_prompt)
    set_user_prompt_template(user_prompt, variant=user_variant)
    return {
        "system": prompt_signature(system_prompt),
        "user": prompt_signature(user_prompt),
    }


@dataclass(frozen=True)
class WarningsCapture:
    logger: logging.Logger
    previous_level: int
    previous_propagate: bool
    previous_filters: list[Any]
    handler_attached: bool


def _start_warnings_capture(
    verbose: bool,
    queue_handler: QueueLogHandler,
) -> WarningsCapture | None:
    if not verbose:
        return None
    warnings_logger = logging.getLogger("py.warnings")
    state = WarningsCapture(
        logger=warnings_logger,
        previous_level=warnings_logger.level,
        previous_propagate=warnings_logger.propagate,
        previous_filters=warnings.filters[:],
        handler_attached=queue_handler not in warnings_logger.handlers,
    )
    try:
        logging.captureWarnings(True)
        warnings_logger.setLevel(logging.WARNING)
        warnings_logger.propagate = False
        warnings.simplefilter("always")
        if state.handler_attached:
            warnings_logger.addHandler(queue_handler)
    except Exception:
        _stop_warnings_capture(state, queue_handler)
        raise
    return state


def _stop_warnings_capture(
    capture: WarningsCapture | None,
    queue_handler: QueueLogHandler,
) -> None:
    if capture is None:
        return
    if capture.handler_attached:
        capture.logger.removeHandler(queue_handler)
    capture.logger.setLevel(capture.previous_level)
    capture.logger.propagate = capture.previous_propagate
    warnings.filters[:] = capture.previous_filters
    logging.captureWarnings(False)


def _emit_file_state(
    job: TranslationJob,
    path: Path,
    index: int,
    total: int,
    state: str,
    progress: int,
    *,
    output: Path | None = None,
    error: str | None = None,
) -> None:
    event: dict[str, Any] = {
        "type": "file",
        "path": str(path),
        "index": index,
        "total": total,
        "state": state,
        "progress": progress,
    }
    if output is not None:
        event["output"] = str(output)
    if error is not None:
        event["error"] = error
    _emit_job_event(job, event)


def _build_smart_split_settings(settings: TranslationSettings) -> SmartSplitSettings | None:
    if not settings.smart_split:
        return None
    return SmartSplitSettings(
        max_lines=settings.smart_split_max_lines,
        max_words=settings.smart_split_max_words,
        max_chars=settings.smart_split_max_chars,
        max_gap_ms=settings.smart_split_max_gap_ms,
        max_duration_ms=settings.smart_split_max_duration_ms,
    )


def _to_processing_settings(
    settings: TranslationSettings,
    prompt_signatures: dict[str, str] | None = None,
) -> ProcessingSettings:
    api = resolve_api(settings.api)
    signatures = prompt_signatures
    if api == "agent" and signatures is None:
        system_prompt, user_prompt, _variant = _resolve_agent_prompts(settings)
        signatures = {
            "system": prompt_signature(system_prompt),
            "user": prompt_signature(user_prompt),
        }
    model_path = settings.model_path.strip() if settings.model_path else None
    return ProcessingSettings(
        source_lang=settings.source_lang,
        target_lang=settings.target_lang,
        api=api,
        batch_size=settings.batch_size,
        thread_count=settings.threads,
        smart_split=settings.smart_split,
        smart_split_settings=_build_smart_split_settings(settings),
        timeout=settings.timeout,
        force=settings.force,
        tld=settings.tld,
        request_delay_ms=settings.request_delay_ms,
        allow_cpu_fallback=settings.allow_cpu_fallback,
        model_path=Path(model_path) if model_path else None,
        model_revision=settings.model_revision,
        worker_python_path=(Path(settings.worker_python_path) if settings.worker_python_path else None),
        auto_download_model=settings.auto_download_model,
        agent_model=(settings.agent_model or get_agent_model_name()) if api == "agent" else None,
        prompt_signatures=signatures or {},
    )


def _report_progress(
    job: TranslationJob,
    path: Path,
    index: int,
    total: int,
    processed: int,
    total_items: int,
) -> None:
    progress = 100 if total_items <= 0 else int(processed / total_items * 100)
    _emit_file_state(job, path, index, total, "started", min(100, max(0, progress)))


def _report_batch_started(
    job: TranslationJob,
    path: Path,
    index: int,
    total: int,
) -> None:
    _emit_file_state(job, path, index, total, "started", 0)
    _emit_job_event(
        job,
        {"type": "log", "message": f"Файл {index}/{total}: {path.name} - перевод."},
    )


def _report_batch_result(
    job: TranslationJob,
    result: ProcessingResult,
    index: int,
    total: int,
) -> None:
    path = result.input_path
    if result.status is ProcessingStatus.ERROR:
        _emit_file_state(
            job,
            path,
            index,
            total,
            "error",
            100,
            error="Не удалось перевести файл.",
        )
        _emit_job_event(job, {"type": "log", "message": f"Файл {index}/{total}: {path.name} - ошибка."})
        return
    if result.status in {ProcessingStatus.CACHED, ProcessingStatus.OUTPUT_EXISTS}:
        _emit_file_state(job, path, index, total, "cached", 100, output=result.output_path)
        message = "кеш" if result.status is ProcessingStatus.CACHED else "результат уже существует"
        _emit_job_event(
            job,
            {"type": "log", "message": f"Файл {index}/{total}: {path.name} - {message}."},
        )
        return
    _emit_file_state(job, path, index, total, "done", 100, output=result.output_path)
    _emit_job_event(job, {"type": "log", "message": f"Файл {index}/{total}: {path.name} - готово."})


def _finish_job(job: TranslationJob, status: str) -> None:
    global _active_job_id
    with job.lock:
        unresolved = []
        for path in job.paths:
            path_key = str(path)
            state = job.file_states.get(path_key, {})
            if state.get("state") not in _TERMINAL_FILE_STATES:
                unresolved.append(path_key)
                job.file_states[path_key] = {
                    **state,
                    "state": "error",
                    "progress": 100,
                    "error": "Файл не был обработан до завершения задачи.",
                }
        if unresolved:
            status = "error" if len(unresolved) == len(job.paths) else "partial"
        job.status = status
        job.completed = True
        job.completed_at = time.monotonic()
    try:
        _emit_job_event(job, {"type": "done", "status": status})
    except Exception:
        _safe_web_exception("Не удалось опубликовать итог веб-задачи %s.", job.job_id)
    finally:
        with _active_job_lock:
            if _active_job_id == job.job_id:
                _active_job_id = None


def _safe_web_exception(message: str, *args: object) -> None:
    try:
        _WEB_LOGGER.exception(message, *args)
    except Exception:
        return


def _safe_job_error(logger: logging.Logger | None, message: str) -> None:
    if logger is None:
        return
    try:
        logger.error(message)
    except Exception:
        _safe_web_exception("Не удалось записать ошибку веб-задачи в её журнал.")


def _safe_cleanup(action: Callable[[], None], message: str) -> None:
    try:
        action()
    except Exception:
        _safe_web_exception(message)


def _cleanup_job_logging(
    logger: logging.Logger | None,
    queue_handler: QueueLogHandler | None,
    warnings_capture: WarningsCapture | None,
    *,
    agent_mode: bool,
) -> None:
    if queue_handler is None:
        return
    if agent_mode:
        _safe_cleanup(
            partial(detach_log_handler, queue_handler),
            "Не удалось отсоединить журнал агента от веб-задачи.",
        )
    if warnings_capture is not None:
        _safe_cleanup(
            partial(_stop_warnings_capture, warnings_capture, queue_handler),
            "Не удалось восстановить обработчик предупреждений веб-задачи.",
        )
    if logger is not None:
        _safe_cleanup(
            partial(logger.removeHandler, queue_handler),
            "Не удалось отсоединить обработчик журнала веб-задачи.",
        )
    _safe_cleanup(queue_handler.close, "Не удалось закрыть обработчик журнала веб-задачи.")


def _run_job(job: TranslationJob) -> None:
    logger: logging.Logger | None = None
    queue_handler: QueueLogHandler | None = None
    warnings_capture: WarningsCapture | None = None
    agent_mode = job.settings.api == "agent"
    status = "error"
    try:
        logger = _get_logger(f"sub_translate_job_{job.job_id}", verbose=job.settings.verbose)
        queue_handler = QueueLogHandler(job.events, on_emit=partial(_record_job_event, job))
        queue_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(queue_handler)
        if agent_mode:
            attach_log_handler(queue_handler)
        warnings_capture = _start_warnings_capture(job.settings.verbose, queue_handler)
        if agent_mode:
            try:
                job.prompt_signatures = _apply_agent_prompts(job.settings)
            except Exception:
                _safe_web_exception("Не удалось загрузить промпты агента.")
                _safe_job_error(logger, "Не удалось загрузить промпты агента.")
                return
        total = len(job.paths)
        _emit_job_event(job, {"type": "job", "total": total})
        if job.settings.force:
            _emit_job_event(job, {"type": "log", "message": "Кеш перевода игнорируется (--force)."})
        processing = _to_processing_settings(job.settings, job.prompt_signatures or None)
        results = process_subtitle_batch(
            job.paths,
            processing,
            logger=logger,
            started_callback=partial(_report_batch_started, job),
            progress_callback=partial(_report_progress, job),
            result_callback=partial(_report_batch_result, job),
        )
        status = "ok" if all(result.successful for result in results) else "partial"
    except Exception:
        _safe_web_exception("Необработанная ошибка веб-задачи %s.", job.job_id)
        _safe_job_error(logger, "Задача перевода завершилась с ошибкой.")
    finally:
        _cleanup_job_logging(
            logger,
            queue_handler,
            warnings_capture,
            agent_mode=agent_mode,
        )
        _finish_job(job, status)


def _start_translation_job(paths: list[Path], settings: TranslationSettings) -> str:
    global _active_job_id
    job_id = uuid.uuid4().hex
    try:
        with _active_job_lock:
            if _active_job_id is not None:
                raise HTTPException(status_code=409, detail="Перевод уже выполняется.")

            events: queue.Queue[dict[str, Any]] = BoundedEventQueue(MAX_EVENTS_PER_JOB)
            job = TranslationJob(
                job_id=job_id,
                paths=paths,
                settings=settings,
                events=events,
                thread=None,
                job_total=len(paths),
            )
            thread = threading.Thread(
                target=_run_job,
                args=(job,),
                name=f"sub-translate-{job_id}",
                daemon=False,
            )
            job.thread = thread
            with _jobs_lock:
                _jobs[job_id] = job
                try:
                    _prune_jobs_locked(time.monotonic())
                except Exception:
                    _jobs.pop(job_id, None)
                    raise
            _active_job_id = job_id
            try:
                thread.start()
            except Exception:
                with _jobs_lock:
                    _jobs.pop(job_id, None)
                if _active_job_id == job_id:
                    _active_job_id = None
                raise
    except HTTPException:
        raise
    except Exception:
        _safe_web_exception("Не удалось запустить веб-задачу перевода %s.", job_id)
        raise HTTPException(
            status_code=503,
            detail="Не удалось запустить задачу перевода.",
        ) from None
    return job_id


@app.post(
    "/api/translate",
    responses={
        400: {"description": "Не передан ни один поддерживаемый файл."},
        409: {"description": "Подготовка файлов или другая задача перевода уже выполняется."},
        503: {"description": "Не удалось запустить задачу перевода."},
    },
)
def translate(payload: TranslateRequest) -> dict[str, Any]:
    paths = [Path(value) for value in payload.paths]
    paths = [path for path in paths if _is_subtitle(path)]
    if not paths:
        raise HTTPException(status_code=400, detail="Список файлов пуст.")
    if not _picker_refresh_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail=_PREPARATION_IN_PROGRESS_ERROR)
    try:
        if preparation_registry.snapshot()["active"]:
            raise HTTPException(status_code=409, detail=_PREPARATION_IN_PROGRESS_ERROR)
        return {"job_id": _start_translation_job(paths, payload.settings)}
    finally:
        _picker_refresh_lock.release()


@app.post(
    "/api/unload",
    responses={
        409: {"description": "Подготовка или перевод уже выполняется."},
        500: {"description": "Не удалось выгрузить локальную модель."},
    },
)
def unload_models() -> dict[str, str]:
    """Принудительно выгружает все модели из памяти."""
    if not _picker_refresh_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail=_PREPARATION_IN_PROGRESS_ERROR)
    try:
        if preparation_registry.snapshot()["active"]:
            raise HTTPException(status_code=409, detail=_PREPARATION_IN_PROGRESS_ERROR)
        with _active_job_lock:
            if _active_job_id is not None:
                raise HTTPException(status_code=409, detail="Перевод уже выполняется.")
            try:
                unload_all_local_translators()
            except Exception as exc:
                _WEB_LOGGER.error("Не удалось выгрузить локальную модель (%s).", type(exc).__name__)
                raise HTTPException(status_code=500, detail="Не удалось выгрузить локальную модель.") from None
    finally:
        _picker_refresh_lock.release()
    return {"status": "ok", "message": "Модели выгружены."}


@app.get(
    "/api/stream/{job_id}",
    responses={404: {"description": "Задача не найдена."}},
)
def stream(job_id: str) -> StreamingResponse:
    job = _get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Задача не найдена.")

    def _event_stream() -> Iterable[str]:
        while True:
            try:
                event = job.events.get(timeout=1.0)
            except queue.Empty:
                with job.lock:
                    completed = job.completed
                    status = job.status
                if completed:
                    payload = json.dumps(
                        {"type": "done", "status": status},
                        ensure_ascii=False,
                    )
                    yield f"data: {payload}\n\n"
                    break
                yield ": keep-alive\n\n"
                continue
            payload = json.dumps(event, ensure_ascii=False)
            yield f"data: {payload}\n\n"
            if event.get("type") == "done":
                break

    return StreamingResponse(_event_stream(), media_type="text/event-stream")
