"""Явно включаемая приёмка реальных TranslateGemma 12B и Seed-X без сети.

Для запуска нужны:

- ``RUN_HEAVY_LOCAL_MODEL_INTEGRATION=1``;
- ``SUB_TRANSLATOR_TRANSLATEGEMMA_12B_MODEL_PATH``;
- ``SUB_TRANSLATOR_TRANSLATEGEMMA_12B_WORKER_PYTHON``;
- ``SUB_TRANSLATOR_SEEDX_MODEL_PATH``;
- ``SUB_TRANSLATOR_SEEDX_WORKER_PYTHON``.
"""

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

from sub_translate import cli, service
from sub_translate.enums import FileFormat
from sub_translate.service import ProcessingResult, ProcessingSettings, ProcessingStatus
from sub_translate.translators import registry
from sub_translate.translators.local.seedx import SeedXTranslator
from sub_translate.translators.local.translategemma import TranslateGemmaTranslator
from sub_translate.utils.io_utils import split_lines
from sub_translate.utils.subtitle_cache import validate_cached_lines

if TYPE_CHECKING:
    from collections.abc import Iterator

    from _pytest.monkeypatch import MonkeyPatch


OPT_IN_ENV = "RUN_HEAVY_LOCAL_MODEL_INTEGRATION"
RUN_INTEGRATION = os.environ.get(OPT_IN_ENV) == "1"
MODEL_TIMEOUT_SECONDS = 3_600
TRANSLATEGEMMA_WORKER_MODULE = "sub_translate.workers.translategemma_worker"
SEEDX_WORKER_MODULE = "sub_translate.workers.seedx_worker"
WORKER_MODULES = (TRANSLATEGEMMA_WORKER_MODULE, SEEDX_WORKER_MODULE)
CYRILLIC_PATTERN = re.compile(r"[А-Яа-яЁё]")
LATIN_PATTERN = re.compile(r"[A-Za-z]")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not RUN_INTEGRATION,
        reason=f"реальные тяжёлые модели включаются через {OPT_IN_ENV}=1",
    ),
]


@dataclass(frozen=True, slots=True)
class HeavyProfileRuntime:
    profile_id: str
    model_path: Path
    worker_python: Path
    worker_module: str


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
def heavy_profiles(monkeypatch: MonkeyPatch) -> Iterator[tuple[HeavyProfileRuntime, ...]]:
    """Передаёт дочерним процессам строгий автономный режим и явные локальные пути."""
    for name, value in (
        ("HF_HUB_OFFLINE", "1"),
        ("TRANSFORMERS_OFFLINE", "1"),
        ("HF_DATASETS_OFFLINE", "1"),
        ("HF_HUB_DISABLE_TELEMETRY", "1"),
        ("TOKENIZERS_PARALLELISM", "false"),
    ):
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_TOKEN", raising=False)
    assert sys.version_info[:2] == (3, 14)
    yield (
        HeavyProfileRuntime(
            profile_id="translategemma-12b",
            model_path=_required_path(
                "SUB_TRANSLATOR_TRANSLATEGEMMA_12B_MODEL_PATH",
                directory=True,
            ),
            worker_python=_required_path(
                "SUB_TRANSLATOR_TRANSLATEGEMMA_12B_WORKER_PYTHON",
                directory=False,
            ),
            worker_module=TRANSLATEGEMMA_WORKER_MODULE,
        ),
        HeavyProfileRuntime(
            profile_id="seedx",
            model_path=_required_path("SUB_TRANSLATOR_SEEDX_MODEL_PATH", directory=True),
            worker_python=_required_path(
                "SUB_TRANSLATOR_SEEDX_WORKER_PYTHON",
                directory=False,
            ),
            worker_module=SEEDX_WORKER_MODULE,
        ),
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _isolated_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.NullHandler())
    return logger


def _write_srt_source(directory: Path, profile_id: str, source_lang: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    text = "Hello, friend!" if source_lang == "en" else "Привет, друг!"
    path = directory / f"{profile_id}-{source_lang}.{source_lang}.srt"
    path.write_text(
        f"1\n00:00:00,000 --> 00:00:02,000\n{text}\n",
        encoding="utf-8",
        newline="",
    )
    return path


def _settings(
    profile: HeavyProfileRuntime,
    *,
    source_lang: str,
    target_lang: str,
) -> ProcessingSettings:
    metadata = registry.get_translator_metadata(profile.profile_id)
    assert metadata.model_revision is not None
    settings = ProcessingSettings(
        source_lang=source_lang,
        target_lang=target_lang,
        api=profile.profile_id,
        batch_size=1,
        thread_count=1,
        timeout=MODEL_TIMEOUT_SECONDS,
        allow_cpu_fallback=False,
        model_path=profile.model_path,
        model_revision=metadata.model_revision,
        worker_python_path=profile.worker_python,
        auto_download_model=False,
    )
    assert settings.auto_download_model is False
    return settings


def _assert_valid_srt(path: Path, target_lang: str) -> bytes:
    payload = path.read_bytes()
    assert payload and not payload.startswith(b"\xef\xbb\xbf")
    text = payload.decode("utf-8", errors="strict")
    normalized = text.replace("\r\n", "\n")
    assert normalized.startswith("1\n00:00:00,000 --> 00:00:02,000\n")
    assert validate_cached_lines(split_lines(text), FileFormat.SRT)
    expected_script = CYRILLIC_PATTERN if target_lang == "ru" else LATIN_PATTERN
    assert expected_script.search(text)
    return payload


def _worker_instance(profile_id: str):
    if profile_id == "translategemma-12b":
        return TranslateGemmaTranslator._worker
    if profile_id == "seedx":
        return SeedXTranslator._worker
    raise AssertionError(f"Неизвестный тяжёлый профиль: {profile_id}")


def _worker_pid(profile_id: str) -> int:
    worker = _worker_instance(profile_id)
    assert worker is not None and worker.is_running
    process = worker._process
    assert process is not None and isinstance(process.pid, int)
    return process.pid


def _worker_module_pids(module: str | None = None) -> set[int]:
    modules = (module,) if module is not None else WORKER_MODULES
    result: set[int] = set()
    for process in psutil.process_iter(("pid", "cmdline")):
        try:
            command = " ".join(process.info.get("cmdline") or ())
        except psutil.Error:
            continue
        if any(worker_module in command for worker_module in modules):
            result.add(int(process.info["pid"]))
    return result


def _worker_tree_pids(worker_pid: int) -> set[int]:
    process = psutil.Process(worker_pid)
    try:
        return {worker_pid, *(child.pid for child in process.children(recursive=True))}
    except psutil.Error:
        return {worker_pid}


def _wait_for_worker_exit(pids: set[int], module: str, label: str) -> None:
    deadline = time.monotonic() + 20.0
    while time.monotonic() < deadline:
        remaining = {pid for pid in pids if psutil.pid_exists(pid)}
        module_pids = _worker_module_pids(module)
        if not remaining and not module_pids:
            return
        time.sleep(0.1)
    remaining = {pid for pid in pids if psutil.pid_exists(pid)} | _worker_module_pids(module)
    pytest.fail(f"После выгрузки {label} остались процессы: {sorted(remaining)}")


def _run_service_and_cache_repeat(
    profile: HeavyProfileRuntime,
    source: Path,
    output: Path,
    cache_path: Path,
    monkeypatch: MonkeyPatch,
) -> int:
    source_digest = _sha256(source)
    settings = _settings(profile, source_lang="en", target_lang="ru")
    logger = _isolated_logger(f"integration.heavy.service.{profile.profile_id}")
    result = service.process_subtitle(
        source,
        settings,
        logger=logger,
        output_path=output,
        cache_path=cache_path,
    )
    assert result.status is ProcessingStatus.TRANSLATED
    expected_payload = _assert_valid_srt(output, "ru")
    worker_pid = _worker_pid(profile.profile_id)
    output.unlink()

    def reject_adapter(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("При совпадении кеша адаптер тяжёлой модели вызываться не должен.")

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
    assert _worker_pid(profile.profile_id) == worker_pid
    assert _sha256(source) == source_digest
    return worker_pid


def _run_cli(
    profile: HeavyProfileRuntime,
    source: Path,
    output: Path,
    cache_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    source_digest = _sha256(source)
    metadata = registry.get_translator_metadata(profile.profile_id)
    assert metadata.model_revision is not None
    original_process = service.process_subtitle

    def process_with_temporary_cache(
        input_path: Path,
        settings: ProcessingSettings,
        **kwargs: object,
    ) -> ProcessingResult:
        assert settings.auto_download_model is False
        kwargs["cache_path"] = cache_path
        return original_process(input_path, settings, **kwargs)

    argv = [
        "sub-translate",
        "--input",
        str(source),
        "--output",
        str(output),
        "--api",
        profile.profile_id,
        "--from",
        "ru",
        "--to",
        "en",
        "--batch-size",
        "1",
        "--threads",
        "1",
        "--timeout",
        str(MODEL_TIMEOUT_SECONDS),
        "--model-path",
        str(profile.model_path),
        "--model-revision",
        metadata.model_revision,
        "--worker-python-path",
        str(profile.worker_python),
    ]
    with monkeypatch.context() as cli_patch:
        cli_patch.setattr(sys, "argv", argv)
        cli_patch.setattr(cli, "configure_utf8_stdio", lambda: None)
        cli_patch.setattr(cli, "load_env", lambda: None)
        cli_patch.setattr(
            cli,
            "configure_rotating_logger",
            lambda *_args, **_kwargs: _isolated_logger(f"integration.heavy.cli.{profile.profile_id}"),
        )
        cli_patch.setattr(cli, "process_subtitle", process_with_temporary_cache)
        assert cli.main() == cli.EXIT_SUCCESS
    _assert_valid_srt(output, "en")
    assert _sha256(source) == source_digest


def test_real_heavy_models_sequential_acceptance(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    heavy_profiles: tuple[HeavyProfileRuntime, ...],
) -> None:
    """Проверяет два тяжёлых профиля строго последовательно и без повторной загрузки из кеша."""
    import torch

    from sub_translate.utils import huggingface as huggingface_utils

    assert torch.cuda.is_available()
    assert registry.get_active_local_translator_id() is None
    assert _worker_module_pids() == set()

    def reject_network(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("Полные локальные модели не должны обращаться к Hugging Face Hub.")

    monkeypatch.setattr(huggingface_utils, "_download_model_snapshot", reject_network)
    profiles = {profile.profile_id: profile for profile in heavy_profiles}
    translategemma = profiles["translategemma-12b"]
    seedx = profiles["seedx"]
    work_dir = tmp_path / "приёмка тяжёлых моделей"
    sources = {
        (profile.profile_id, source_lang): _write_srt_source(
            work_dir / profile.profile_id,
            profile.profile_id,
            source_lang,
        )
        for profile in heavy_profiles
        for source_lang in ("en", "ru")
    }
    source_hashes = {path: _sha256(path) for path in sources.values()}

    try:
        translategemma_pid = _run_service_and_cache_repeat(
            translategemma,
            sources[(translategemma.profile_id, "en")],
            work_dir / "translategemma-12b-service.ru.srt",
            work_dir / "translategemma-12b-service-cache.json",
            monkeypatch,
        )
        _run_cli(
            translategemma,
            sources[(translategemma.profile_id, "ru")],
            work_dir / "translategemma-12b-cli.en.srt",
            work_dir / "translategemma-12b-cli-cache.json",
            monkeypatch,
        )
        assert _worker_pid(translategemma.profile_id) == translategemma_pid
        assert registry.get_active_local_translator_id() == translategemma.profile_id
        translategemma_tree = _worker_tree_pids(translategemma_pid)

        seedx_pid = _run_service_and_cache_repeat(
            seedx,
            sources[(seedx.profile_id, "en")],
            work_dir / "seedx-service.ru.srt",
            work_dir / "seedx-service-cache.json",
            monkeypatch,
        )
        _wait_for_worker_exit(
            translategemma_tree,
            translategemma.worker_module,
            "TranslateGemma 12B",
        )
        assert TranslateGemmaTranslator._worker is None
        assert registry.get_active_local_translator_id() == seedx.profile_id
        _run_cli(
            seedx,
            sources[(seedx.profile_id, "ru")],
            work_dir / "seedx-cli.en.srt",
            work_dir / "seedx-cli-cache.json",
            monkeypatch,
        )
        assert _worker_pid(seedx.profile_id) == seedx_pid
        seedx_tree = _worker_tree_pids(seedx_pid)

        registry.unload_all_local_translators()
        _wait_for_worker_exit(seedx_tree, seedx.worker_module, "Seed-X")
        assert registry.get_active_local_translator_id() is None
        assert SeedXTranslator._worker is None
        assert _worker_module_pids() == set()
        assert all(_sha256(path) == digest for path, digest in source_hashes.items())
        assert not list(tmp_path.rglob("*.tmp"))
    finally:
        registry.unload_all_local_translators()
        TranslateGemmaTranslator.unload()
        SeedXTranslator.unload()
