"""Отказобезопасный клиент постоянного процесса локальной модели."""

from __future__ import annotations

import os
import queue
import subprocess
import threading
import time
import uuid
from collections import deque
from collections.abc import Mapping
from contextlib import suppress
from pathlib import Path
from typing import Any, Never, TextIO

try:
    import psutil
except ImportError:  # pragma: no cover - зависимость проверяется при установке приложения
    psutil = None  # type: ignore[assignment]

from sub_translate.workers.protocol import (
    MAX_FRAME_BYTES,
    PROTOCOL_VERSION,
    WorkerProtocolError,
    decode_frame,
    encode_frame,
)

DEFAULT_REQUEST_TIMEOUT_SECONDS = 3_600.0
SHUTDOWN_TIMEOUT_SECONDS = 5.0
_SECRET_ENVIRONMENT_NAMES = {
    "OPENAI_API_KEY",
    "HF_TOKEN",
    "HUGGINGFACE_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "SONAR_TOKEN",
}
_SECRET_ENVIRONMENT_SUFFIXES = (
    "_ACCESS_KEY",
    "_API_KEY",
    "_PASSWORD",
    "_PRIVATE_KEY",
    "_SECRET",
    "_TOKEN",
)


class ExternalWorkerError(RuntimeError):
    """Ошибка запуска, выполнения или протокола изолированного процесса."""

    def __init__(self, message: str, *, error_type: str = "WorkerError") -> None:
        super().__init__(message)
        self.error_type = error_type


_Response = dict[str, Any] | WorkerProtocolError | None


class PersistentNdjsonWorker:
    """Последовательно обслуживает запросы одним процессом до сбоя или выгрузки."""

    def __init__(
        self,
        python_path: Path,
        module: str,
        *,
        timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
        cwd: Path | None = None,
        max_frame_bytes: int = MAX_FRAME_BYTES,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("Таймаут процесса должен быть положительным.")
        if max_frame_bytes <= 0:
            raise ValueError("Предельный размер кадра должен быть положительным.")
        self.python_path = Path(python_path)
        self.module = module
        self.timeout_seconds = float(timeout_seconds)
        self.cwd = cwd or Path(__file__).resolve().parents[2]
        self.max_frame_bytes = int(max_frame_bytes)
        self._process: subprocess.Popen[str] | None = None
        self._lock = threading.RLock()
        self._responses: queue.Queue[tuple[subprocess.Popen[str], _Response]] = queue.Queue()

    @property
    def is_running(self) -> bool:
        """Сообщает, жив ли текущий процесс."""
        with self._lock:
            return self._process is not None and self._process.poll() is None

    def request(
        self,
        command: str,
        payload: Mapping[str, Any] | None = None,
        *,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        """Отправляет кадр и ждёт единственный ответ с тем же идентификатором."""
        normalized_command = command.strip().casefold()
        if not normalized_command:
            raise ValueError("Команда процесса не может быть пустой.")
        timeout = self.timeout_seconds if timeout_seconds is None else float(timeout_seconds)
        if timeout <= 0:
            raise ValueError("Таймаут процесса должен быть положительным.")
        with self._lock:
            return self._request_locked(normalized_command, payload or {}, timeout)

    def abort(self) -> None:
        """Безусловно завершает только дерево текущего процесса."""
        with self._lock:
            if self._process is not None:
                self._terminate_locked(self._process)

    def shutdown(self) -> None:
        """Просит процесс завершиться и ограниченно ждёт его остановки."""
        with self._lock:
            process = self._process
            if process is None:
                return
            if process.poll() is None:
                with suppress(ExternalWorkerError):
                    self._request_locked("shutdown", {}, SHUTDOWN_TIMEOUT_SECONDS)
            self._terminate_locked(process)

    def _request_locked(
        self,
        command: str,
        payload: Mapping[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        process = self._ensure_started_locked()
        request_id = uuid.uuid4().hex
        message = {
            "protocol": PROTOCOL_VERSION,
            "id": request_id,
            "command": command,
            "payload": dict(payload),
        }
        try:
            frame = encode_frame(message, max_frame_bytes=self.max_frame_bytes)
            assert process.stdin is not None
            process.stdin.write(frame)
            process.stdin.flush()
        except WorkerProtocolError as exc:
            self._terminate_locked(process)
            raise ExternalWorkerError(str(exc), error_type="ProtocolError") from exc
        except (OSError, ValueError) as exc:
            self._terminate_locked(process)
            raise ExternalWorkerError(
                f"Не удалось отправить команду '{command}' в процесс модели.",
                error_type=type(exc).__name__,
            ) from exc
        return self._wait_for_response(process, request_id, command, timeout_seconds)

    def _wait_for_response(
        self,
        process: subprocess.Popen[str],
        request_id: str,
        command: str,
        timeout_seconds: float,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_seconds
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._terminate_locked(process)
                raise ExternalWorkerError(
                    f"Процесс модели не ответил на команду '{command}' за {timeout_seconds:g} с.",
                    error_type="TimeoutError",
                )
            response = self._next_response(process, min(remaining, 0.1), command)
            if response is None:
                continue
            owner, value = response
            if owner is not process:
                continue
            if value is None:
                return self._raise_terminated(process, command)
            if isinstance(value, WorkerProtocolError):
                self._terminate_locked(process)
                raise ExternalWorkerError(str(value), error_type="ProtocolError") from value
            if str(value.get("id") or "") != request_id:
                self._terminate_locked(process)
                raise ExternalWorkerError(
                    "Процесс модели вернул неожиданный идентификатор ответа.",
                    error_type="ProtocolError",
                )
            return self._unwrap_response(value, command)

    def _next_response(
        self,
        process: subprocess.Popen[str],
        timeout_seconds: float,
        command: str,
    ) -> tuple[subprocess.Popen[str], _Response] | None:
        try:
            return self._responses.get(timeout=timeout_seconds)
        except queue.Empty:
            if process.poll() is not None:
                return self._raise_terminated(process, command)
            return None

    @staticmethod
    def _unwrap_response(response: Mapping[str, Any], command: str) -> dict[str, Any]:
        if response.get("protocol") != PROTOCOL_VERSION:
            raise ExternalWorkerError(
                "Процесс модели использует несовместимую версию протокола.",
                error_type="ProtocolError",
            )
        if response.get("ok") is not True:
            raw_error = response.get("error")
            error = dict(raw_error) if isinstance(raw_error, Mapping) else {}
            error_type = str(error.get("type") or "WorkerError")
            message = str(error.get("message") or "неизвестная ошибка")
            raise ExternalWorkerError(
                f"Команда процесса '{command}' завершилась ошибкой {error_type}: {message}",
                error_type=error_type,
            )
        result = response.get("result")
        if not isinstance(result, Mapping):
            raise ExternalWorkerError(
                f"Процесс модели вернул некорректный результат команды '{command}'.",
                error_type="ProtocolError",
            )
        return dict(result)

    def _ensure_started_locked(self) -> subprocess.Popen[str]:
        if self._process is not None and self._process.poll() is None:
            return self._process
        if self._process is not None:
            self._terminate_locked(self._process)
        environment = _worker_environment()
        creation_flags = int(getattr(subprocess, "CREATE_NO_WINDOW", 0))
        try:
            process = subprocess.Popen(
                [str(self.python_path), "-B", "-u", "-m", self.module],
                cwd=str(self.cwd),
                env=environment,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=creation_flags,
            )
        except OSError as exc:
            raise ExternalWorkerError(
                f"Не удалось запустить Python процесса модели: {self.python_path}",
                error_type=type(exc).__name__,
            ) from exc
        self._process = process
        assert process.stdout is not None and process.stderr is not None
        threading.Thread(
            target=self._read_stdout,
            args=(process, process.stdout),
            name="translation-worker-stdout",
            daemon=True,
        ).start()
        threading.Thread(
            target=self._drain_stderr,
            args=(process.stderr,),
            name="translation-worker-stderr",
            daemon=True,
        ).start()
        return process

    def _read_stdout(self, process: subprocess.Popen[str], stream: TextIO) -> None:
        try:
            for line in stream:
                try:
                    response = decode_frame(line, max_frame_bytes=self.max_frame_bytes)
                except WorkerProtocolError as exc:
                    self._responses.put((process, exc))
                    return
                self._responses.put((process, response))
        finally:
            self._responses.put((process, None))

    @staticmethod
    def _drain_stderr(stream: TextIO) -> None:
        """Не переносит потенциальную диагностику модели в журналы приложения."""
        deque(stream, maxlen=0)

    def _raise_terminated(self, process: subprocess.Popen[str], command: str) -> Never:
        code = process.poll()
        self._terminate_locked(process)
        raise ExternalWorkerError(
            f"Процесс модели завершился до ответа на команду '{command}' (код {code}).",
            error_type="WorkerExited",
        )

    def _terminate_locked(self, process: subprocess.Popen[str]) -> None:
        descendants = _worker_descendants(process)
        if self._process is process:
            self._process = None
        try:
            if process.stdin is not None:
                process.stdin.close()
        except OSError:
            pass
        _terminate_descendants(descendants)
        if process.poll() is None:
            with suppress(OSError):
                process.terminate()
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                with suppress(OSError):
                    process.kill()
                with suppress(OSError, subprocess.TimeoutExpired):
                    process.wait(timeout=2.0)


def _worker_environment() -> dict[str, str]:
    environment = {name: value for name, value in os.environ.items() if not _is_secret_environment_name(name)}
    environment.update(
        {
            "PYTHONUTF8": "1",
            "PYTHONIOENCODING": "utf-8",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    return environment


def _is_secret_environment_name(name: str) -> bool:
    normalized = name.upper()
    return normalized in _SECRET_ENVIRONMENT_NAMES or normalized.endswith(_SECRET_ENVIRONMENT_SUFFIXES)


def _worker_descendants(process: subprocess.Popen[str]) -> list[Any]:
    if psutil is None or not isinstance(getattr(process, "pid", None), int):
        return []
    try:
        return list(psutil.Process(process.pid).children(recursive=True))
    except psutil.Error:
        return []


def _terminate_descendants(processes: list[Any]) -> None:
    if psutil is None or not processes:
        return
    for process in reversed(processes):
        with suppress(psutil.Error):
            process.terminate()
    _gone, alive = psutil.wait_procs(processes, timeout=2.0)
    for process in alive:
        with suppress(psutil.Error):
            process.kill()


__all__ = [
    "DEFAULT_REQUEST_TIMEOUT_SECONDS",
    "SHUTDOWN_TIMEOUT_SECONDS",
    "ExternalWorkerError",
    "PersistentNdjsonWorker",
]
