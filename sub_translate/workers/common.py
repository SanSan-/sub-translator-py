"""Серверная часть протокола изолированных переводчиков."""

from __future__ import annotations

import re
import sys
from collections.abc import Callable, Mapping
from typing import Any, Protocol

from sub_translate.workers.protocol import (
    PROTOCOL_VERSION,
    WorkerProtocolError,
    decode_frame,
    encode_frame,
)


class PublicWorkerError(RuntimeError):
    """Ошибка с безопасным сообщением для вызывающего процесса."""


_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def validate_model_content_fingerprint(raw_value: Any) -> str | None:
    """Проверяет необязательный отпечаток содержимого модели на границе процесса."""
    if raw_value is None:
        return None
    if not isinstance(raw_value, str) or not _SHA256_PATTERN.fullmatch(raw_value):
        raise PublicWorkerError("Отпечаток содержимого модели должен быть 64-символьным SHA-256.")
    return raw_value


def model_uses_only_device(model: Any, is_allowed: Callable[[Any], bool]) -> bool:
    """Сверяет декларативную карту и фактические параметры модели с одним устройством."""
    device_map = getattr(model, "hf_device_map", None)
    if isinstance(device_map, Mapping):
        mapped_devices = tuple(device_map.values())
        if mapped_devices and any(not is_allowed(device) for device in mapped_devices):
            return False

    observed_tensor = False
    for provider_name in ("parameters", "buffers"):
        provider = getattr(model, provider_name, None)
        if not callable(provider):
            continue
        try:
            tensors = provider()
            for tensor in tensors:
                observed_tensor = True
                if not is_allowed(getattr(tensor, "device", None)):
                    return False
        except RuntimeError, TypeError:
            return False
    return observed_tensor


class WorkerRuntime(Protocol):
    """Минимальный контракт среды локальной модели."""

    def preflight(self, payload: Mapping[str, Any]) -> dict[str, Any]: ...

    def translate(self, payload: Mapping[str, Any]) -> dict[str, Any]: ...

    def unload(self) -> dict[str, Any]: ...


def handle_worker_request(
    runtime: WorkerRuntime,
    request: Mapping[str, Any],
) -> tuple[dict[str, Any], bool]:
    """Выполняет одну разрешённую команду и возвращает безопасный ответ."""
    request_id = str(request.get("id") or "")
    try:
        _validate_request(request)
        command = str(request.get("command") or "").strip().casefold()
        raw_payload = request.get("payload")
        payload = dict(raw_payload) if isinstance(raw_payload, Mapping) else {}
        result, should_stop = _execute_command(runtime, command, payload)
        response = {
            "protocol": PROTOCOL_VERSION,
            "id": request_id,
            "ok": True,
            "result": result,
        }
        return response, should_stop
    except Exception as exc:  # Процесс обязан вернуть доменную ошибку, а не завершиться.
        print(f"Ошибка команды изолированного процесса: {type(exc).__name__}", file=sys.stderr)
        response = {
            "protocol": PROTOCOL_VERSION,
            "id": request_id,
            "ok": False,
            "error": {
                "type": type(exc).__name__,
                "message": _public_error_message(exc),
            },
        }
        return response, False


def run_worker_loop(runtime: WorkerRuntime) -> None:
    """Читает кадры до команды завершения или конца входного потока."""
    _configure_streams()
    for line in sys.stdin:
        try:
            request = decode_frame(line)
        except WorkerProtocolError:
            print("Изолированный процесс отклонил повреждённый входной кадр.", file=sys.stderr)
            continue
        response, should_stop = handle_worker_request(runtime, request)
        sys.stdout.write(encode_frame(response))
        sys.stdout.flush()
        if should_stop:
            return
    runtime.unload()


def _validate_request(request: Mapping[str, Any]) -> None:
    if request.get("protocol") != PROTOCOL_VERSION:
        raise PublicWorkerError("Несовместимая версия протокола процесса модели.")
    if not str(request.get("id") or ""):
        raise PublicWorkerError("В запросе отсутствует идентификатор.")
    if not isinstance(request.get("payload"), Mapping):
        raise PublicWorkerError("Поле payload должно быть JSON-объектом.")


def _execute_command(
    runtime: WorkerRuntime,
    command: str,
    payload: Mapping[str, Any],
) -> tuple[dict[str, Any], bool]:
    if command == "ping":
        return {"protocol": PROTOCOL_VERSION}, False
    if command == "preflight":
        return runtime.preflight(payload), False
    if command == "translate":
        return runtime.translate(payload), False
    if command == "unload":
        return runtime.unload(), False
    if command == "shutdown":
        result = runtime.unload()
        result["shutdown"] = True
        return result, True
    raise PublicWorkerError(f"Неизвестная команда процесса модели: {command or '?'}.")


def _public_error_message(exc: Exception) -> str:
    if isinstance(exc, PublicWorkerError):
        return str(exc)
    if isinstance(exc, FileNotFoundError):
        return str(exc)
    return "Изолированная модель не выполнила команду; подробности оставлены в её stderr."


def _configure_streams() -> None:
    if hasattr(sys.stdin, "reconfigure"):
        sys.stdin.reconfigure(encoding="utf-8", errors="strict")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="strict", write_through=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace", write_through=True)


__all__ = [
    "PublicWorkerError",
    "WorkerRuntime",
    "handle_worker_request",
    "model_uses_only_device",
    "run_worker_loop",
    "validate_model_content_fingerprint",
]
