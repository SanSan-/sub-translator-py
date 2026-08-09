from __future__ import annotations

import queue
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from sub_translate.workers import external_worker
from sub_translate.workers.external_worker import (
    ExternalWorkerError,
    PersistentNdjsonWorker,
)
from sub_translate.workers.protocol import (
    PROTOCOL_PREFIX,
    PROTOCOL_VERSION,
    WorkerProtocolError,
    decode_frame,
    encode_frame,
)


class _BlockingStream:
    def __init__(self) -> None:
        self._lines: queue.Queue[str | None] = queue.Queue()

    def push(self, line: str) -> None:
        self._lines.put(line)

    def close(self) -> None:
        self._lines.put(None)

    def __iter__(self) -> _BlockingStream:
        return self

    def __next__(self) -> str:
        line = self._lines.get(timeout=2.0)
        if line is None:
            raise StopIteration
        return line


class _FakeStdin:
    def __init__(self, process: _FakeProcess) -> None:
        self._process = process

    def write(self, value: str) -> int:
        self._process.receive(value)
        return len(value)

    def flush(self) -> None:
        return None

    def close(self) -> None:
        return None


class _FakeProcess:
    def __init__(self, mode: str = "ok") -> None:
        self.mode = mode
        self.stdin = _FakeStdin(self)
        self.stdout = _BlockingStream()
        self.stderr = _BlockingStream()
        self.returncode: int | None = None
        self.requests: list[dict[str, Any]] = []
        self.terminate_calls = 0
        self.kill_calls = 0
        self.pid = 12345

    def receive(self, line: str) -> None:
        request = decode_frame(line)
        self.requests.append(request)
        command = str(request["command"])
        if self.mode == "timeout" and command != "shutdown":
            return
        if self.mode == "crash" and command != "shutdown":
            self.returncode = 7
            self.stdout.close()
            self.stderr.close()
            return
        if self.mode == "malformed" and command != "shutdown":
            self.stdout.push(f"{PROTOCOL_PREFIX}{{broken\n")
            return
        if self.mode == "oversized" and command != "shutdown":
            self.stdout.push(
                encode_frame(
                    {
                        "protocol": PROTOCOL_VERSION,
                        "id": request["id"],
                        "ok": True,
                        "result": {"text": "x" * 2_000},
                    }
                )
            )
            return
        if self.mode == "wrong-id" and command != "shutdown":
            request = {**request, "id": "чужой"}
        if self.mode == "error" and command != "shutdown":
            response = {
                "protocol": PROTOCOL_VERSION,
                "id": request["id"],
                "ok": False,
                "error": {
                    "type": "TranslateGemmaOutOfMemoryError",
                    "message": "нет памяти",
                },
            }
        else:
            response = {
                "protocol": PROTOCOL_VERSION,
                "id": request["id"],
                "ok": True,
                "result": {"command": command, "echo": request["payload"]},
            }
        self.stdout.push(encode_frame(response))
        if command == "shutdown":
            self.returncode = 0
            self.stdout.close()
            self.stderr.close()

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        if self.returncode is None:
            raise subprocess.TimeoutExpired("fake", timeout)
        return self.returncode

    def terminate(self) -> None:
        self.terminate_calls += 1
        self.returncode = -15
        self.stdout.close()
        self.stderr.close()

    def kill(self) -> None:
        self.kill_calls += 1
        self.returncode = -9
        self.stdout.close()
        self.stderr.close()


def _install_processes(
    monkeypatch: pytest.MonkeyPatch,
    processes: list[_FakeProcess],
) -> tuple[list[list[str]], list[dict[str, Any]]]:
    launches: list[list[str]] = []
    kwargs_seen: list[dict[str, Any]] = []

    def fake_popen(args: list[str], **kwargs: Any) -> _FakeProcess:
        launches.append(args)
        kwargs_seen.append(kwargs)
        return processes[len(launches) - 1]

    monkeypatch.setattr(external_worker.subprocess, "Popen", fake_popen)
    return launches, kwargs_seen


def test_framing_accepts_unicode_and_rejects_foreign_or_broken_data() -> None:
    payload = {"protocol": 1, "id": "один", "payload": {"text": "Привет"}}

    assert decode_frame(encode_frame(payload)) == payload
    with pytest.raises(WorkerProtocolError, match="посторонние"):
        decode_frame("обычная строка\n")
    with pytest.raises(WorkerProtocolError, match="повреждённый"):
        decode_frame(f"{PROTOCOL_PREFIX}{{broken\n")
    with pytest.raises(WorkerProtocolError, match="превышает"):
        encode_frame(payload, max_frame_bytes=10)


def test_worker_is_persistent_and_supports_unload_and_shutdown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _FakeProcess()
    launches, _kwargs = _install_processes(monkeypatch, [process])
    worker = PersistentNdjsonWorker(tmp_path / "python.exe", "worker.module")

    assert worker.request("translate", {"value": 1})["echo"] == {"value": 1}
    assert worker.request("unload", {})["command"] == "unload"
    worker.shutdown()

    assert len(launches) == 1
    assert launches[0][-2:] == ["-m", "worker.module"]
    assert [request["command"] for request in process.requests] == [
        "translate",
        "unload",
        "shutdown",
    ]
    assert len({request["id"] for request in process.requests}) == 3
    assert worker.is_running is False


@pytest.mark.parametrize(
    ("mode", "error_type"),
    [
        ("malformed", "ProtocolError"),
        ("wrong-id", "ProtocolError"),
        ("oversized", "ProtocolError"),
    ],
)
def test_worker_rejects_protocol_failures(
    mode: str,
    error_type: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _FakeProcess(mode)
    _install_processes(monkeypatch, [process])
    worker = PersistentNdjsonWorker(
        tmp_path / "python.exe",
        "worker.module",
        max_frame_bytes=512,
    )

    with pytest.raises(ExternalWorkerError) as error:
        worker.request("translate")

    assert error.value.error_type == error_type
    assert process.terminate_calls == 1
    assert worker.is_running is False


def test_timeout_stops_process_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _FakeProcess("timeout")
    _install_processes(monkeypatch, [process])
    child = SimpleNamespace(terminate_calls=0, kill_calls=0)

    def terminate() -> None:
        child.terminate_calls += 1

    def kill() -> None:
        child.kill_calls += 1

    child.terminate = terminate
    child.kill = kill
    fake_parent = SimpleNamespace(children=lambda recursive: [child])
    fake_psutil = SimpleNamespace(
        Error=RuntimeError,
        Process=lambda _pid: fake_parent,
        wait_procs=lambda _items, timeout: ([], [child]),
    )
    monkeypatch.setattr(external_worker, "psutil", fake_psutil)
    worker = PersistentNdjsonWorker(
        tmp_path / "python.exe",
        "worker.module",
        timeout_seconds=0.02,
    )

    with pytest.raises(ExternalWorkerError, match="не ответил") as error:
        worker.request("translate")

    assert error.value.error_type == "TimeoutError"
    assert child.terminate_calls == 1
    assert child.kill_calls == 1
    assert process.terminate_calls == 1


def test_crashed_worker_restarts_on_next_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _FakeProcess("crash")
    second = _FakeProcess()
    launches, _kwargs = _install_processes(monkeypatch, [first, second])
    worker = PersistentNdjsonWorker(tmp_path / "python.exe", "worker.module")

    with pytest.raises(ExternalWorkerError) as error:
        worker.request("translate")
    result = worker.request("translate", {"attempt": 2})
    worker.shutdown()

    assert error.value.error_type == "WorkerExited"
    assert result["echo"] == {"attempt": 2}
    assert len(launches) == 2


def test_remote_error_is_typed_and_does_not_stop_main_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _FakeProcess("error")
    _install_processes(monkeypatch, [process])
    worker = PersistentNdjsonWorker(tmp_path / "python.exe", "worker.module")

    with pytest.raises(ExternalWorkerError, match="нет памяти") as error:
        worker.request("translate")

    assert error.value.error_type == "TranslateGemmaOutOfMemoryError"
    worker.abort()


def test_worker_environment_is_offline_and_does_not_inherit_tokens(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "secret-openai")
    monkeypatch.setenv("HF_TOKEN", "secret-hf")
    monkeypatch.setenv("SONAR_TOKEN", "secret-sonar")
    monkeypatch.setenv("EXAMPLE_PASSWORD", "secret-password")
    process = _FakeProcess()
    _launches, kwargs_seen = _install_processes(monkeypatch, [process])
    worker = PersistentNdjsonWorker(tmp_path / "python.exe", "worker.module")

    worker.request("ping")
    worker.shutdown()

    environment = kwargs_seen[0]["env"]
    assert "OPENAI_API_KEY" not in environment
    assert "HF_TOKEN" not in environment
    assert "SONAR_TOKEN" not in environment
    assert "EXAMPLE_PASSWORD" not in environment
    assert environment["HF_HUB_OFFLINE"] == "1"
    assert environment["TRANSFORMERS_OFFLINE"] == "1"
