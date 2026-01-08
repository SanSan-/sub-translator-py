from __future__ import annotations

import json
import logging
import queue
import re
import threading
import uuid
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from sub_translate.cli import resolve_io_paths
from sub_translate.constants import (
    LOGS_DIR,
    SMART_SPLIT_MAX_CHARS,
    SMART_SPLIT_MAX_DURATION_MS,
    SMART_SPLIT_MAX_GAP_MS,
    SMART_SPLIT_MAX_LINES,
    SMART_SPLIT_MAX_WORDS,
)
from sub_translate.models import SmartSplitSettings, TranslationOptions
from sub_translate.service import resolve_api, resolve_format, translate_subtitles
from sub_translate.dictionaries.languages import LANGS, get_code
from sub_translate.translators.agent import (
    SYSTEM_PROMPT,
    AgentTranslator,
    attach_log_handler,
    detach_log_handler,
    get_agent_model_name,
    get_openai_api_key,
    set_system_prompt,
    set_user_prompt_template,
)
from sub_translate.translators.agent_prompts import DEFAULT_PROMPT_VARIANT, get_prompt_template
from sub_translate.translators.google_web import GoogleWebTranslator
from sub_translate.translators.local.fsm import FsmTranslator
from sub_translate.translators.local.madlad import MadladTranslator
from sub_translate.translators.local.nllb import NllbTranslator, NllbLiteTranslator
from sub_translate.translators.local.seamless import SeamlessTranslator
from sub_translate.utils.io_utils import read_text
from sub_translate.utils.env_utils import load_env
from sub_translate.utils.logging_utils import configure_rotating_logger
from sub_translate.utils.subtitle_cache import (
    apply_output_cache,
    build_cache_key,
    get_cached_lines,
    load_cache_snapshot,
    update_output_cache,
)
from sub_translate.utils.path_utils import split_lang_suffix

ROOT_DIR = Path(__file__).resolve().parent
STATIC_DIR = ROOT_DIR / "static"
load_env()

SUBTITLE_EXTS = {".ass", ".srt", ".vtt"}
_LANG_SUFFIX_PATTERN = re.compile(r"^[a-z]{2,3}(?:[-_][a-z0-9]+)*$")

app = FastAPI(title="Sub Translator")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

_WEB_LOGGER = configure_rotating_logger(
    "sub_translate_web",
    LOGS_DIR / "sub_translate_web.log",
    verbose=False,
)
_WEB_LOGGER.propagate = False


class TranslationSettings(BaseModel):
    source_lang: str = "en"
    target_lang: str = "ru"
    api: str = "google"
    batch_size: int = 21
    threads: int = 3
    allow_cpu_fallback: bool = False
    smart_split: bool = False
    force: bool = False
    smart_split_max_lines: int = SMART_SPLIT_MAX_LINES
    smart_split_max_words: int = SMART_SPLIT_MAX_WORDS
    smart_split_max_chars: int = SMART_SPLIT_MAX_CHARS
    smart_split_max_gap_ms: int = SMART_SPLIT_MAX_GAP_MS
    smart_split_max_duration_ms: int = SMART_SPLIT_MAX_DURATION_MS
    tld: str = "com"
    timeout: int = 30
    request_delay_ms: int = 350
    agent_model: str | None = None
    openai_api_key: str | None = None
    agent_system_prompt: str | None = None
    agent_prompt: str | None = None
    agent_system_prompt_file: str | None = None
    agent_prompt_file: str | None = None
    verbose: bool = False


class PickRequest(BaseModel):
    kind: str = Field(..., pattern="^(file|folder)$")
    settings: TranslationSettings | None = None


class RefreshRequest(BaseModel):
    paths: list[str]
    settings: TranslationSettings


class TranslateRequest(BaseModel):
    paths: list[str]
    settings: TranslationSettings


class QueueLogHandler(logging.Handler):
    def __init__(self, events: queue.Queue, on_emit: Callable[[dict[str, Any]], None] | None = None) -> None:
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
    events: queue.Queue
    thread: threading.Thread | None
    completed: bool = False
    job_total: int = 0
    file_states: dict[str, dict[str, Any]] = field(default_factory=dict)
    log_lines: list[str] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)


_jobs: dict[str, TranslationJob] = {}
_jobs_lock = threading.Lock()
_active_job_id: str | None = None
_active_job_lock = threading.Lock()

LOG_HISTORY_LIMIT = 250


def _record_job_event(job: TranslationJob, event: dict[str, Any]) -> None:
    event_type = event.get("type")
    with job.lock:
        if event_type == "log":
            message = str(event.get("message", "")).rstrip()
            if message:
                job.log_lines.append(message)
                if len(job.log_lines) > LOG_HISTORY_LIMIT:
                    job.log_lines = job.log_lines[-LOG_HISTORY_LIMIT:]
            return
        if event_type == "job":
            total = event.get("total")
            if isinstance(total, int):
                job.job_total = total
            return
        if event_type == "file":
            path = event.get("path")
            if not path:
                return
            state = job.file_states.get(path, {})
            if "state" in event:
                state["state"] = event["state"]
            if "progress" in event:
                state["progress"] = event["progress"]
            if "output" in event:
                state["output"] = event["output"]
            if "error" in event:
                state["error"] = event["error"]
            job.file_states[path] = state


def _emit_job_event(job: TranslationJob, event: dict[str, Any]) -> None:
    _record_job_event(job, event)
    job.events.put(event)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/active-job")
def active_job() -> dict[str, Any]:
    with _active_job_lock:
        job_id = _active_job_id
    if job_id is None:
        return {"active": False}
    job = _jobs.get(job_id)
    if job is None:
        return {"active": False}
    items = _build_items(job.paths, job.settings)
    with job.lock:
        for item in items:
            state = job.file_states.get(item["path"])
            if state:
                item.update(state)
        logs = list(job.log_lines)
        total = job.job_total or len(items)
    done = sum(1 for item in items if item.get("state") in {"cached", "done", "error"})
    return {
        "active": True,
        "job_id": job_id,
        "items": items,
        "logs": logs,
        "total": total,
        "done": done,
    }


def _build_language_options(include_auto: bool) -> list[dict[str, str]]:
    options: list[dict[str, str]] = []
    if include_auto and "auto" in LANGS:
        options.append({"value": "auto", "label": LANGS["auto"]})
    for key, label in LANGS.items():
        if key == "auto":
            continue
        options.append({"value": key, "label": label})
    return options


def _build_ui_config() -> dict[str, Any]:
    google_source = _build_language_options(include_auto=True)
    google_target = _build_language_options(include_auto=False)
    local_langs = [
        {"value": "en", "label": "English"},
        {"value": "ru", "label": "Russian"},
    ]
    return {
        "defaults": {
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
        },
        "languages": {
            "google": {"source": google_source, "target": google_target},
            "agent": {"source": google_source, "target": google_target},
            "nllb": {"source": local_langs, "target": local_langs},
            "nllb-lite": {"source": local_langs, "target": local_langs},
            "seamless": {"source": local_langs, "target": local_langs},
            "madlad": {"source": local_langs, "target": local_langs},
            "fsm": {
                "source": [{"value": "en", "label": "English"}],
                "target": [{"value": "ru", "label": "Russian"}],
            },
        },
        "agent_prompts": {
            "system": SYSTEM_PROMPT,
            "user": get_prompt_template(DEFAULT_PROMPT_VARIANT),
        },
        "agent_settings": {
            "model": get_agent_model_name(),
            "openai_api_key": get_openai_api_key() or "",
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


def _is_language_suffix(candidate: str) -> bool:
    if not candidate:
        return False
    if get_code(candidate):
        return True
    return bool(_LANG_SUFFIX_PATTERN.match(candidate))


def _matches_source_suffix(path: Path, source_lang: str) -> bool:
    if not source_lang:
        return True
    resolved_source = (get_code(source_lang) or source_lang).lower()
    if not resolved_source:
        return True
    stem = path.stem
    if "." not in stem:
        return True
    candidate = stem.rsplit(".", 1)[1].lower()
    if not _is_language_suffix(candidate):
        return True
    return candidate == resolved_source


def _iter_subtitles(folder: Path, source_lang: str, target_lang: str) -> Iterable[Path]:
    if not folder.exists():
        return []
    results = []
    for item in sorted(folder.iterdir()):
        if not item.is_file() or not _is_subtitle(item):
            continue
        if not _matches_source_suffix(item, source_lang):
            continue
        if _has_target_suffix(item, target_lang):
            continue
        results.append(item)
    return results


def _build_items(paths: Iterable[Path], settings: TranslationSettings) -> list[dict[str, Any]]:
    api = resolve_api(settings.api)
    cache = load_cache_snapshot(_WEB_LOGGER)
    items: list[dict[str, Any]] = []
    for path in paths:
        file_format = resolve_format(path, None)
        _, output_path = resolve_io_paths(
            path,
            None,
            settings.source_lang,
            settings.target_lang,
            api,
            file_format,
        )
        cache_key = build_cache_key(path, api, settings.target_lang)
        cached_lines = get_cached_lines(cache, cache_key, file_format)
        cached = bool(cached_lines) or output_path.exists()
        items.append(
            {
                "name": path.name,
                "path": str(path),
                "format": file_format.value,
                "cached": cached,
                "output": str(output_path),
            }
        )
    return items


def _pick_with_tk(kind: str) -> dict[str, Any]:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as exc:  # pragma: no cover - зависит от окружения
        raise HTTPException(status_code=500, detail=f"Не удалось открыть диалог выбора: {exc}") from exc

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        if kind == "folder":
            path = filedialog.askdirectory()
            if not path:
                return {"mode": "folder", "path": "", "items": []}
            return {"mode": "folder", "path": path}
        paths = filedialog.askopenfilenames(
            filetypes=[
                ("Субтитры", "*.ass *.srt *.vtt"),
                ("Все файлы", "*.*"),
            ]
        )
        return {"mode": "files", "paths": list(paths)}
    finally:
        root.destroy()


@app.post("/api/pick")
def pick(payload: PickRequest) -> dict[str, Any]:
    settings = payload.settings or TranslationSettings()
    result = _pick_with_tk(payload.kind)
    if result.get("mode") == "folder":
        path_value = result.get("path") or ""
        if not path_value:
            return {"mode": "folder", "path": "", "items": []}
        folder = Path(path_value)
        items = _build_items(_iter_subtitles(folder, settings.source_lang, settings.target_lang), settings)
        return {"mode": "folder", "path": str(folder), "items": items}
    paths = [Path(value) for value in result.get("paths", [])]
    items = _build_items([path for path in paths if _is_subtitle(path)], settings)
    return {"mode": "files", "items": items}


@app.post("/api/refresh")
def refresh(payload: RefreshRequest) -> dict[str, Any]:
    paths = [Path(value) for value in payload.paths]
    return {"items": _build_items(paths, payload.settings)}


def _apply_agent_prompts(settings: TranslationSettings) -> None:
    if settings.agent_system_prompt:
        set_system_prompt(settings.agent_system_prompt)
    elif settings.agent_system_prompt_file:
        system_path = Path(settings.agent_system_prompt_file).expanduser().resolve()
        if not system_path.exists():
            raise FileNotFoundError(f"Системный промпт не найден: {system_path}")
        set_system_prompt(read_text(system_path))
    if settings.agent_prompt:
        set_user_prompt_template(settings.agent_prompt, variant="ui")
    elif settings.agent_prompt_file:
        user_path = Path(settings.agent_prompt_file).expanduser().resolve()
        if not user_path.exists():
            raise FileNotFoundError(f"Промпт агента не найден: {user_path}")
        set_user_prompt_template(read_text(user_path), variant="file")


def _run_job(job: TranslationJob) -> None:
    global _active_job_id
    logger = _get_logger(f"sub_translate_job_{job.job_id}", verbose=job.settings.verbose)
    logger.propagate = False
    queue_handler = QueueLogHandler(job.events, on_emit=lambda event: _record_job_event(job, event))
    queue_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(queue_handler)
    attach_log_handler(queue_handler)
    warnings_logger = None
    warnings_attached = False
    warnings_level = None
    warnings_propagate = None
    warnings_filters = None
    if job.settings.verbose:
        logging.captureWarnings(True)
        warnings_logger = logging.getLogger("py.warnings")
        warnings_level = warnings_logger.level
        warnings_propagate = warnings_logger.propagate
        warnings_logger.setLevel(logging.WARNING)
        warnings_logger.propagate = False
        warnings_filters = warnings.filters[:]
        warnings.simplefilter("always")
        if queue_handler not in warnings_logger.handlers:
            warnings_logger.addHandler(queue_handler)
            warnings_attached = True

    try:
        try:
            _apply_agent_prompts(job.settings)
        except Exception as exc:
            logger.error("Не удалось загрузить промпты агента: %s", exc)
            _emit_job_event(job, {"type": "done", "status": "error"})
            job.completed = True
            with _active_job_lock:
                _active_job_id = None
            return

        total = len(job.paths)
        _emit_job_event(job, {"type": "job", "total": total})
        if job.settings.force:
            _emit_job_event(job, {"type": "log", "message": "Кеш перевода игнорируется (--force)."})

        for idx, path in enumerate(job.paths, start=1):
            _emit_job_event(job,
                {
                    "type": "file",
                    "path": str(path),
                    "index": idx,
                    "total": total,
                    "state": "started",
                    "progress": 0,
                }
            )
            _emit_job_event(job,
                {
                    "type": "log",
                    "message": f"Файл {idx}/{total}: {path.name} - перевод.",
                }
            )
            try:
                api = resolve_api(job.settings.api)
                file_format = resolve_format(path, None)
                source_lang, output_path = resolve_io_paths(
                    path,
                    None,
                    job.settings.source_lang,
                    job.settings.target_lang,
                    api,
                    file_format,
                )
                options = TranslationOptions(
                    source_lang=source_lang,
                    target_lang=job.settings.target_lang,
                    api=api,
                    tld=job.settings.tld,
                    request_delay_ms=job.settings.request_delay_ms,
                    allow_cpu_fallback=job.settings.allow_cpu_fallback,
                    agent_model=job.settings.agent_model,
                    openai_api_key=job.settings.openai_api_key,
                )
                smart_split_settings = None
                if job.settings.smart_split:
                    smart_split_settings = SmartSplitSettings(
                        max_lines=job.settings.smart_split_max_lines,
                        max_words=job.settings.smart_split_max_words,
                        max_chars=job.settings.smart_split_max_chars,
                        max_gap_ms=job.settings.smart_split_max_gap_ms,
                        max_duration_ms=job.settings.smart_split_max_duration_ms,
                    )
                if not job.settings.force:
                    if apply_output_cache(
                        path,
                        output_path,
                        api,
                        job.settings.target_lang,
                        file_format,
                        logger,
                    ):
                        _emit_job_event(job,
                            {
                                "type": "file",
                                "path": str(path),
                                "index": idx,
                                "total": total,
                                "state": "cached",
                                "progress": 100,
                                "output": str(output_path),
                            }
                        )
                        _emit_job_event(job,
                            {
                                "type": "log",
                                "message": f"Файл {idx}/{total}: {path.name} - кеш.",
                            }
                        )
                        continue

                def report_progress(processed: int, total_items: int) -> None:
                    if total_items <= 0:
                        progress = 100
                    else:
                        progress = int(processed / total_items * 100)
                    progress = min(100, max(0, progress))
                    _emit_job_event(job,
                        {
                            "type": "file",
                            "path": str(path),
                            "index": idx,
                            "total": total,
                            "state": "started",
                            "progress": progress,
                        }
                    )

                translate_subtitles(
                    path,
                    output_path,
                    file_format,
                    options,
                    api,
                    thread_count=job.settings.threads,
                    batch_size=job.settings.batch_size,
                    smart_split=job.settings.smart_split,
                    smart_split_settings=smart_split_settings,
                    timeout=job.settings.timeout,
                    logger=logger,
                    progress_callback=report_progress,
                )
                update_output_cache(
                    path,
                    output_path,
                    api,
                    job.settings.target_lang,
                    file_format,
                    logger,
                )
                _emit_job_event(job,
                    {
                        "type": "file",
                        "path": str(path),
                        "index": idx,
                        "total": total,
                        "state": "done",
                        "progress": 100,
                        "output": str(output_path),
                    }
                )
                _emit_job_event(job,
                    {
                        "type": "log",
                        "message": f"Файл {idx}/{total}: {path.name} - готово.",
                    }
                )
            except Exception as exc:
                logger.error("Ошибка перевода %s: %s", path, exc)
                _emit_job_event(job,
                    {
                        "type": "file",
                        "path": str(path),
                        "index": idx,
                        "total": total,
                        "state": "error",
                        "progress": 100,
                        "error": str(exc),
                    }
                )

        _emit_job_event(job, {"type": "done", "status": "ok"})
        job.completed = True
        with _active_job_lock:
            _active_job_id = None
    finally:
        detach_log_handler(queue_handler)
        if warnings_attached and warnings_logger is not None:
            warnings_logger.removeHandler(queue_handler)
        if warnings_logger is not None:
            if warnings_level is not None:
                warnings_logger.setLevel(warnings_level)
            if warnings_propagate is not None:
                warnings_logger.propagate = warnings_propagate
        if warnings_filters is not None:
            warnings.filters[:] = warnings_filters
        if job.settings.verbose:
            logging.captureWarnings(False)


@app.post("/api/translate")
def translate(payload: TranslateRequest) -> dict[str, Any]:
    global _active_job_id
    paths = [Path(value) for value in payload.paths]
    paths = [path for path in paths if _is_subtitle(path)]
    if not paths:
        raise HTTPException(status_code=400, detail="Список файлов пуст.")

    with _active_job_lock:
        if _active_job_id is not None:
            raise HTTPException(status_code=409, detail="Перевод уже выполняется.")

        job_id = uuid.uuid4().hex
        events: queue.Queue = queue.Queue()
        job = TranslationJob(
            job_id=job_id,
            paths=paths,
            settings=payload.settings,
            events=events,
            thread=None,
            job_total=len(paths),
        )
        thread = threading.Thread(target=_run_job, args=(job,))
        job.thread = thread
        _jobs[job_id] = job
        _active_job_id = job_id

    assert job.thread is not None
    job.thread.start()
    return {"job_id": job_id}


@app.post("/api/unload")
def unload_models() -> dict[str, str]:
    """Принудительно выгружает все модели из памяти."""
    translators = [
        AgentTranslator,
        FsmTranslator,
        GoogleWebTranslator,
        MadladTranslator,
        NllbTranslator,
        NllbLiteTranslator,
        SeamlessTranslator,
    ]
    for translator in translators:
        try:
            translator.unload()
        except Exception as exc:
            _WEB_LOGGER.warning("Ошибка выгрузки %s: %s", translator.name, exc)
    return {"status": "ok", "message": "Модели выгружены."}


@app.get("/api/stream/{job_id}")
def stream(job_id: str) -> StreamingResponse:
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Задача не найдена.")

    def _event_stream() -> Iterable[str]:
        while True:
            try:
                event = job.events.get(timeout=1.0)
            except queue.Empty:
                yield ": keep-alive\n\n"
                continue
            payload = json.dumps(event, ensure_ascii=False)
            yield f"data: {payload}\n\n"
            if event.get("type") == "done":
                break

    return StreamingResponse(_event_stream(), media_type="text/event-stream")
