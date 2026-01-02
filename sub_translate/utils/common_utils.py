from __future__ import annotations

from typing import Any

from sub_translate.constants import EMPTY_STRING


def is_empty(value: Any) -> bool:
    return value is None or value == EMPTY_STRING


def is_empty_object(value: Any) -> bool:
    return is_empty(value) or (isinstance(value, dict) and len(value) == 0)


def is_empty_array(value: Any) -> bool:
    return not (value and isinstance(value, list) and len(value) > 0)


__all__ = ["is_empty", "is_empty_object", "is_empty_array"]