"""Ограниченный протокол обмена с изолированными процессами моделей."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

PROTOCOL_PREFIX = "@@SUB_TRANSLATE_WORKER_V1@@"
PROTOCOL_VERSION = 1
MAX_FRAME_BYTES = 16 * 1024 * 1024


class WorkerProtocolError(RuntimeError):
    """Повреждённый, посторонний либо слишком большой кадр протокола."""


def encode_frame(
    payload: Mapping[str, Any],
    *,
    max_frame_bytes: int = MAX_FRAME_BYTES,
) -> str:
    """Кодирует один JSON-кадр и проверяет его размер в UTF-8."""
    if max_frame_bytes <= 0:
        raise ValueError("Предельный размер кадра должен быть положительным.")
    body = json.dumps(dict(payload), ensure_ascii=False, separators=(",", ":"))
    frame = f"{PROTOCOL_PREFIX}{body}\n"
    size = len(frame.encode("utf-8"))
    if size > max_frame_bytes:
        raise WorkerProtocolError(f"Размер кадра {size} байт превышает предел {max_frame_bytes} байт.")
    return frame


def decode_frame(
    line: str,
    *,
    max_frame_bytes: int = MAX_FRAME_BYTES,
) -> dict[str, Any]:
    """Декодирует только корректный кадр текущей версии протокола."""
    if max_frame_bytes <= 0:
        raise ValueError("Предельный размер кадра должен быть положительным.")
    size = len(line.encode("utf-8"))
    if size > max_frame_bytes:
        raise WorkerProtocolError(f"Размер кадра {size} байт превышает предел {max_frame_bytes} байт.")
    normalized = line.rstrip("\r\n")
    if not normalized.startswith(PROTOCOL_PREFIX):
        raise WorkerProtocolError("Стандартный вывод процесса содержит посторонние данные.")
    try:
        value = json.loads(normalized[len(PROTOCOL_PREFIX) :])
    except json.JSONDecodeError as exc:
        raise WorkerProtocolError("Процесс вернул повреждённый JSON-кадр.") from exc
    if not isinstance(value, Mapping):
        raise WorkerProtocolError("Корнем кадра должен быть JSON-объект.")
    return dict(value)


__all__ = [
    "MAX_FRAME_BYTES",
    "PROTOCOL_PREFIX",
    "PROTOCOL_VERSION",
    "WorkerProtocolError",
    "decode_frame",
    "encode_frame",
]
