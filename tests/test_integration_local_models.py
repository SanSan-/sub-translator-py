"""Явно включаемая сквозная проверка локальных переводчиков без сети."""

from __future__ import annotations

import hashlib
import logging
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import psutil
import pytest
from fastapi.testclient import TestClient

from sub_translate import cli, service
from sub_translate.enums import FileFormat
from sub_translate.models import TranslationOptions
from sub_translate.service import ProcessingSettings, ProcessingStatus
from sub_translate.translators import registry
from sub_translate.translators.local.translategemma import TranslateGemmaTranslator
from sub_translate.utils.io_utils import split_lines
from sub_translate.utils.subtitle_cache import validate_cached_lines

if TYPE_CHECKING:
    from collections.abc import Iterator

    from _pytest.monkeypatch import MonkeyPatch

    from sub_translate.translators.base import Translator
    from sub_translate.web import app as web_app_module


RUN_INTEGRATION = os.environ.get("RUN_LOCAL_MODEL_INTEGRATION") == "1"
WORKER_MODULE = "sub_translate.workers.translategemma_worker"
CYRILLIC_PATTERN = re.compile(r"[А-Яа-яЁё]")
LATIN_PATTERN = re.compile(r"[A-Za-z]")
MODEL_TIMEOUT_SECONDS = 900

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not RUN_INTEGRATION,
        reason="реальные локальные модели включаются через RUN_LOCAL_MODEL_INTEGRATION=1",
    ),
]


@dataclass(frozen=True, slots=True)
class LocalRuntime:
    nllb_model_path: Path
    translategemma_model_path: Path
    worker_python: Path


def _required_path(name: str, *, directory: bool) -> Path:
    raw_value = os.environ.get(name, "").strip()
    if not raw_value:
        pytest.fail(f"Для интеграционного прогона требуется переменная {name}.")
    path = Path(raw_value).expanduser().resolve()
    exists = path.is_dir() if directory else path.is_file()
    if not exists:
        expected = "каталог" if directory else "файл"
        pytest.fail(f"Переменная {name} должна указывать на существующий {expected}: {path}")
    return path


@pytest.fixture
def local_runtime(monkeypatch: MonkeyPatch) -> Iterator[LocalRuntime]:
    """Запрещает сеть и возвращает только явно переданные полные каталоги моделей."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
    monkeypatch.setenv("HF_HUB_DISABLE_TELEMETRY", "1")
    monkeypatch.setenv("TOKENIZERS_PARALLELISM", "false")
    runtime = LocalRuntime(
        nllb_model_path=_required_path("SUB_TRANSLATOR_NLLB_MODEL_PATH", directory=True),
        translategemma_model_path=_required_path(
            "SUB_TRANSLATOR_TRANSLATEGEMMA_MODEL_PATH",
            directory=True,
        ),
        worker_python=_required_path("SUB_TRANSLATOR_WORKER_PYTHON", directory=False),
    )
    assert sys.version_info[:2] == (3, 14)
    assert Path(sys.executable).resolve().samefile(runtime.worker_python)
    yield runtime


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _isolated_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.NullHandler())
    return logger


def _write_subtitle_sources(directory: Path, profile_id: str) -> dict[FileFormat, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    contents = {
        FileFormat.ASS: (
            "[Script Info]\n"
            "Title: Проверка локальной модели\n"
            "ScriptType: v4.00+\n"
            "\n"
            "[V4+ Styles]\n"
            "Format: Name, Fontname, Fontsize, PrimaryColour\n"
            "Style: Default,Arial,42,&H00FFFFFF\n"
            "\n"
            "[Events]\n"
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
            "Dialogue: 2,0:00:00.00,0:00:02.50,Default,Анна,0010,0020,0030,"
            "Karaoke;1,{\\i1}Hello, friend!{\\i0}\n"
        ),
        FileFormat.SRT: "7\n00:00:00,000 --> 00:00:02,500\nGood morning, everyone!\n",
        FileFormat.VTT: (
            "WEBVTT - Проверка\n"
            "X-TIMESTAMP-MAP=LOCAL:00:00:00.000,MPEGTS:900000\n"
            "\n"
            "вступление-α\n"
            "00:00.000 --> 00:02.500 line:10% position:20% size:60% align:start\n"
            "Please open the door.\n"
        ),
    }
    sources: dict[FileFormat, Path] = {}
    for file_format, content in contents.items():
        path = directory / f"{profile_id}-source.en.{file_format.value}"
        path.write_text(content, encoding="utf-8", newline="")
        sources[file_format] = path
    return sources


def _profile_settings(
    profile_id: str,
    runtime: LocalRuntime,
    *,
    force: bool = False,
) -> ProcessingSettings:
    metadata = registry.get_translator_metadata(profile_id)
    model_path = runtime.nllb_model_path if profile_id == "nllb-600m" else runtime.translategemma_model_path
    return ProcessingSettings(
        source_lang="en",
        target_lang="ru",
        api=profile_id,
        batch_size=1,
        thread_count=1,
        timeout=MODEL_TIMEOUT_SECONDS,
        force=force,
        allow_cpu_fallback=False,
        model_path=model_path,
        model_revision=metadata.model_revision,
        worker_python_path=runtime.worker_python,
        auto_download_model=False,
    )


def _translation_options(
    profile_id: str,
    runtime: LocalRuntime,
    source_lang: str,
    target_lang: str,
) -> TranslationOptions:
    settings = _profile_settings(profile_id, runtime)
    return TranslationOptions(
        source_lang=source_lang,
        target_lang=target_lang,
        api=profile_id,
        allow_cpu_fallback=False,
        model_path=settings.model_path,
        model_revision=settings.model_revision,
        worker_python_path=runtime.worker_python,
        auto_download_model=False,
    )


def _assert_translation(
    translator: Translator,
    profile_id: str,
    runtime: LocalRuntime,
    source: str,
    source_lang: str,
    target_lang: str,
) -> str:
    result = translator.translate_batch(
        [source],
        _translation_options(profile_id, runtime, source_lang, target_lang),
    )
    assert len(result) == 1
    translated = result[0].strip()
    assert translated
    assert translated.casefold() != source.casefold()
    expected_script = CYRILLIC_PATTERN if target_lang == "ru" else LATIN_PATTERN
    assert expected_script.search(translated)
    return translated


def _assert_valid_output(path: Path, file_format: FileFormat) -> bytes:
    payload = path.read_bytes()
    assert payload
    assert not payload.startswith(b"\xef\xbb\xbf")
    text = payload.decode("utf-8", errors="strict")
    normalized_text = text.replace("\r\n", "\n")
    assert CYRILLIC_PATTERN.search(text)
    assert validate_cached_lines(split_lines(text), file_format)
    if file_format is FileFormat.ASS:
        assert "Dialogue: 2,0:00:00.00,0:00:02.50,Default,Анна,0010,0020,0030,Karaoke;1," in normalized_text
        assert "{\\i1}" in normalized_text and "{\\i0}" in normalized_text
    elif file_format is FileFormat.SRT:
        assert normalized_text.startswith("7\n00:00:00,000 --> 00:00:02,500\n")
    else:
        assert normalized_text.startswith("WEBVTT - Проверка\nX-TIMESTAMP-MAP=")
        assert "вступление-α\n" in normalized_text
        assert "line:10% position:20% size:60% align:start" in normalized_text
    return payload


def _run_service_and_cache_repeat(
    profile_id: str,
    runtime: LocalRuntime,
    source: Path,
    output: Path,
    cache_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    logger = _isolated_logger(f"integration.service.{profile_id}")
    settings = _profile_settings(profile_id, runtime)
    result = service.process_subtitle(
        source,
        settings,
        logger=logger,
        output_path=output,
        cache_path=cache_path,
    )
    assert result.status is ProcessingStatus.TRANSLATED
    expected_payload = _assert_valid_output(output, FileFormat.ASS)
    output.unlink()

    def reject_adapter(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("При совпадении кеша адаптер локальной модели вызываться не должен.")

    with monkeypatch.context() as cache_guard:
        cache_guard.setattr(service, "get_translator", reject_adapter)
        cached = service.process_subtitle(
            source,
            settings,
            logger=logger,
            output_path=output,
            cache_path=cache_path,
        )
    assert cached.status is ProcessingStatus.CACHED
    assert output.read_bytes() == expected_payload


def _run_cli(
    profile_id: str,
    runtime: LocalRuntime,
    source: Path,
    output: Path,
    cache_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    settings = _profile_settings(profile_id, runtime)
    original_process = service.process_subtitle

    def process_with_temporary_cache(*args: object, **kwargs: object):
        kwargs["cache_path"] = cache_path
        return original_process(*args, **kwargs)

    argv = [
        "sub-translate",
        "--input",
        str(source),
        "--output",
        str(output),
        "--api",
        profile_id,
        "--from",
        "en",
        "--to",
        "ru",
        "--batch-size",
        "1",
        "--threads",
        "1",
        "--timeout",
        str(MODEL_TIMEOUT_SECONDS),
        "--model-path",
        str(settings.model_path),
        "--model-revision",
        str(settings.model_revision),
        "--worker-python-path",
        str(runtime.worker_python),
    ]
    with monkeypatch.context() as cli_patch:
        cli_patch.setattr(sys, "argv", argv)
        cli_patch.setattr(cli, "configure_utf8_stdio", lambda: None)
        cli_patch.setattr(cli, "load_env", lambda: None)
        cli_patch.setattr(
            cli,
            "configure_rotating_logger",
            lambda *_args, **_kwargs: _isolated_logger(f"integration.cli.{profile_id}"),
        )
        cli_patch.setattr(cli, "process_subtitle", process_with_temporary_cache)
        assert cli.main() == cli.EXIT_SUCCESS
    _assert_valid_output(output, FileFormat.SRT)


def _reset_web_jobs(web_app: web_app_module) -> None:
    with web_app._jobs_lock:
        web_app._jobs.clear()
    with web_app._active_job_lock:
        web_app._active_job_id = None


def _run_web(
    profile_id: str,
    runtime: LocalRuntime,
    source: Path,
    cache_path: Path,
    monkeypatch: MonkeyPatch,
) -> Path:
    from sub_translate.web import app as web_app

    settings = _profile_settings(profile_id, runtime)
    original_batch = service.process_subtitle_batch

    def batch_with_temporary_cache(*args: object, **kwargs: object):
        kwargs["cache_path"] = cache_path
        return original_batch(*args, **kwargs)

    _reset_web_jobs(web_app)
    with monkeypatch.context() as web_patch:
        web_patch.setattr(web_app, "process_subtitle_batch", batch_with_temporary_cache)
        web_patch.setattr(
            web_app,
            "_get_logger",
            lambda *_args, **_kwargs: _isolated_logger(f"integration.web.{profile_id}"),
        )
        web_patch.setattr(web_app, "_WEB_LOGGER", _isolated_logger("integration.web.internal"))
        web_patch.setattr(web_app, "attach_log_handler", lambda *_args: None)
        web_patch.setattr(web_app, "detach_log_handler", lambda *_args: None)
        with TestClient(web_app.app, base_url="http://127.0.0.1") as client:
            response = client.post(
                "/api/translate",
                json={
                    "paths": [str(source)],
                    "settings": {
                        "source_lang": "en",
                        "target_lang": "ru",
                        "api": profile_id,
                        "batch_size": 1,
                        "threads": 1,
                        "timeout": MODEL_TIMEOUT_SECONDS,
                        "model_path": str(settings.model_path),
                        "model_revision": settings.model_revision,
                        "worker_python_path": str(runtime.worker_python),
                        "auto_download_model": False,
                        "allow_cpu_fallback": False,
                    },
                },
            )
            assert response.status_code == 200, response.text
            job_id = response.json()["job_id"]
            job = web_app._get_job(job_id)
            assert job is not None and job.thread is not None
            job.thread.join(timeout=MODEL_TIMEOUT_SECONDS)
            assert not job.thread.is_alive()
            assert job.completed
            stream = client.get(f"/api/stream/{job_id}")
            assert stream.status_code == 200
            assert '"type": "done"' in stream.text
            assert '"status": "ok"' in stream.text

    source_lang, output = service.resolve_io_paths(
        source,
        None,
        "en",
        "ru",
        profile_id,
        FileFormat.VTT,
    )
    assert source_lang == "en"
    _assert_valid_output(output, FileFormat.VTT)
    return output


def _worker_pids() -> set[int]:
    result: set[int] = set()
    for process in psutil.process_iter(("pid", "cmdline")):
        try:
            command = " ".join(process.info.get("cmdline") or ())
        except psutil.Error:
            continue
        if WORKER_MODULE in command:
            result.add(int(process.info["pid"]))
    return result


def _wait_for_worker_exit(pid: int) -> None:
    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        if not psutil.pid_exists(pid) and pid not in _worker_pids():
            return
        time.sleep(0.1)
    pytest.fail(f"Процесс TranslateGemma {pid} остался после явной выгрузки.")


def _unload_through_web(monkeypatch: MonkeyPatch) -> None:
    from sub_translate.web import app as web_app

    with monkeypatch.context() as web_patch:
        web_patch.setattr(web_app, "_WEB_LOGGER", _isolated_logger("integration.web.unload"))
        with TestClient(web_app.app, base_url="http://127.0.0.1") as client:
            response = client.post("/api/unload", json={})
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_real_local_models_end_to_end(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    local_runtime: LocalRuntime,
) -> None:
    """Проверяет оба локальных профиля одним последовательным GPU-прогоном."""
    import torch

    from sub_translate.translators.local import nllb as nllb_module
    from sub_translate.utils import huggingface as huggingface_utils

    assert torch.cuda.is_available()
    assert _worker_pids() == set()
    registry.unload_all_local_translators()

    def reject_network(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("Интеграционный прогон полных локальных моделей не должен обращаться к сети.")

    monkeypatch.setattr(huggingface_utils, "_download_model_snapshot", reject_network)
    work_dir = tmp_path / "приёмка локальных моделей"
    sources_by_profile = {
        profile_id: _write_subtitle_sources(work_dir / profile_id, profile_id)
        for profile_id in ("nllb-600m", "translategemma")
    }
    source_hashes = {path: _sha256(path) for sources in sources_by_profile.values() for path in sources.values()}

    try:
        nllb = registry.create_translator("nllb-600m", timeout=MODEL_TIMEOUT_SECONDS)
        assert registry.get_active_local_translator_id() == "nllb-600m"
        _assert_translation(nllb, "nllb-600m", local_runtime, "Hello, my friend!", "en", "ru")
        _assert_translation(nllb, "nllb-600m", local_runtime, "Доброе утро, друзья!", "ru", "en")
        nllb_sources = sources_by_profile["nllb-600m"]
        _run_service_and_cache_repeat(
            "nllb-600m",
            local_runtime,
            nllb_sources[FileFormat.ASS],
            work_dir / "nllb-service.ru.ass",
            work_dir / "nllb-output-cache.json",
            monkeypatch,
        )
        _run_cli(
            "nllb-600m",
            local_runtime,
            nllb_sources[FileFormat.SRT],
            work_dir / "nllb-cli.ru.srt",
            work_dir / "nllb-cli-cache.json",
            monkeypatch,
        )
        _run_web(
            "nllb-600m",
            local_runtime,
            nllb_sources[FileFormat.VTT],
            work_dir / "nllb-web-cache.json",
            monkeypatch,
        )

        translategemma = registry.create_translator("translategemma", timeout=MODEL_TIMEOUT_SECONDS)
        assert registry.get_active_local_translator_id() == "translategemma"
        assert nllb_module._ENGINE._model is None
        _assert_translation(
            translategemma,
            "translategemma",
            local_runtime,
            "Hello, my friend!",
            "en",
            "ru",
        )
        _assert_translation(
            translategemma,
            "translategemma",
            local_runtime,
            "Доброе утро, друзья!",
            "ru",
            "en",
        )
        translategemma_sources = sources_by_profile["translategemma"]
        _run_service_and_cache_repeat(
            "translategemma",
            local_runtime,
            translategemma_sources[FileFormat.ASS],
            work_dir / "translategemma-service.ru.ass",
            work_dir / "translategemma-output-cache.json",
            monkeypatch,
        )
        _run_cli(
            "translategemma",
            local_runtime,
            translategemma_sources[FileFormat.SRT],
            work_dir / "translategemma-cli.ru.srt",
            work_dir / "translategemma-cli-cache.json",
            monkeypatch,
        )
        _run_web(
            "translategemma",
            local_runtime,
            translategemma_sources[FileFormat.VTT],
            work_dir / "translategemma-web-cache.json",
            monkeypatch,
        )

        worker = TranslateGemmaTranslator._worker
        assert worker is not None and worker.is_running
        process = worker._process
        assert process is not None and isinstance(process.pid, int)
        worker_pid = process.pid
        assert worker_pid in _worker_pids()
        _unload_through_web(monkeypatch)
        _wait_for_worker_exit(worker_pid)
        assert TranslateGemmaTranslator._worker is None
        assert registry.get_active_local_translator_id() is None
        assert _worker_pids() == set()

        assert all(_sha256(path) == digest for path, digest in source_hashes.items())
        assert not list(tmp_path.rglob("*.tmp"))
    finally:
        registry.unload_all_local_translators()
        TranslateGemmaTranslator.unload()
        _reset_web_jobs(__import__("sub_translate.web.app", fromlist=["app"]))
