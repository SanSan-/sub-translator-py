from __future__ import annotations

import json
import logging
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

from filelock import FileLock

from sub_translate.constants import USAGE_STATS_FILE
from sub_translate.utils.io_utils import atomic_write_json

_DEFAULT_STATS: dict[str, Any] = {
    "version": 1,
    "totals": {
        "input_tokens": 0,
        "output_tokens": 0,
        "cost_usd": 0.0,
    },
    "limit_usd": 0.0,
    "by_model": {},
    "history": [],
    "last_updated": 0,
}

HISTORY_LIMIT = 200
USAGE_STATS_LOCK_TIMEOUT_SECONDS = 60.0


def _stats_lock_path(stats_path: Path) -> Path:
    return stats_path.with_name(f"{stats_path.name}.lock")


def _normalize_totals(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return deepcopy(_DEFAULT_STATS["totals"])
    result = deepcopy(_DEFAULT_STATS["totals"])
    for key in ("input_tokens", "output_tokens"):
        try:
            result[key] = int(value.get(key, 0))
        except TypeError, ValueError:
            result[key] = 0
    try:
        result["cost_usd"] = float(value.get("cost_usd", 0.0))
    except TypeError, ValueError:
        result["cost_usd"] = 0.0
    return result


def _read_stats_data(stats_path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(stats_path.read_text(encoding="utf-8"))
    except OSError, json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _normalize_limit(value: Any) -> float:
    try:
        return float(value)
    except TypeError, ValueError:
        return float(_DEFAULT_STATS["limit_usd"])


def _normalize_by_model(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {model: _normalize_totals(totals) for model, totals in value.items() if isinstance(model, str)}


def _normalize_history_item(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "timestamp": int(item.get("timestamp", 0)),
        "source": str(item.get("source", "")),
        "model": str(item.get("model", "")),
        "input_tokens": int(item.get("input_tokens", 0) or 0),
        "output_tokens": int(item.get("output_tokens", 0) or 0),
        "cost_usd": float(item.get("cost_usd", 0.0) or 0.0),
    }


def _normalize_history(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [_normalize_history_item(item) for item in value[-HISTORY_LIMIT:] if isinstance(item, dict)]


def _load_stats(stats_path: Path) -> dict[str, Any]:
    if not stats_path.exists():
        return deepcopy(_DEFAULT_STATS)
    data = _read_stats_data(stats_path)
    if data is None:
        return deepcopy(_DEFAULT_STATS)

    stats = deepcopy(_DEFAULT_STATS)
    stats["version"] = int(data.get("version", _DEFAULT_STATS["version"]))
    stats["totals"] = _normalize_totals(data.get("totals"))
    stats["limit_usd"] = _normalize_limit(data.get("limit_usd", _DEFAULT_STATS["limit_usd"]))
    stats["by_model"] = _normalize_by_model(data.get("by_model"))
    stats["history"] = _normalize_history(data.get("history"))
    stats["last_updated"] = int(data.get("last_updated", 0))
    return stats


def _save_stats(stats: dict[str, Any], stats_path: Path) -> None:
    atomic_write_json(stats_path, stats)


def record_usage(
    source: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    cost_usd: float | None = None,
) -> dict[str, Any]:
    stats_path = USAGE_STATS_FILE
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(_stats_lock_path(stats_path)), timeout=USAGE_STATS_LOCK_TIMEOUT_SECONDS):
        stats = _load_stats(stats_path)
        totals = stats["totals"]
        totals["input_tokens"] += max(0, int(input_tokens))
        totals["output_tokens"] += max(0, int(output_tokens))
        totals["cost_usd"] += float(cost_usd or 0.0)

        by_model = stats["by_model"]
        model_totals = by_model.setdefault(model, deepcopy(_DEFAULT_STATS["totals"]))
        model_totals["input_tokens"] += max(0, int(input_tokens))
        model_totals["output_tokens"] += max(0, int(output_tokens))
        model_totals["cost_usd"] += float(cost_usd or 0.0)

        history = stats["history"]
        history.append(
            {
                "timestamp": int(time.time()),
                "source": source,
                "model": model,
                "input_tokens": max(0, int(input_tokens)),
                "output_tokens": max(0, int(output_tokens)),
                "cost_usd": float(cost_usd or 0.0),
            }
        )
        if len(history) > HISTORY_LIMIT:
            del history[:-HISTORY_LIMIT]

        stats["last_updated"] = int(time.time())
        _save_stats(stats, stats_path)

    result = stats["totals"].copy()
    result["limit_usd"] = float(stats.get("limit_usd", 0.0))
    return result


def get_totals() -> dict[str, Any]:
    stats = _load_stats(USAGE_STATS_FILE)
    result = stats["totals"].copy()
    result["limit_usd"] = float(stats.get("limit_usd", 0.0))
    return result


def get_stats() -> dict[str, Any]:
    return _load_stats(USAGE_STATS_FILE)


def log_usage_summary(
    logger: logging.Logger,
    totals: dict[str, Any],
    *,
    combined_tokens: int | None = None,
    token_limit: int | None = None,
) -> None:
    """Выводит краткую сводку по токенам и бюджету."""
    input_tokens = int(totals.get("input_tokens", 0) or 0)
    output_tokens = int(totals.get("output_tokens", 0) or 0)
    spent = float(totals.get("cost_usd", 0.0) or 0.0)
    overall_tokens = combined_tokens if combined_tokens is not None else input_tokens + output_tokens
    suffix = f" (лимит {token_limit})" if token_limit else ""
    logger.info(
        "Итог по токенам: input=%s, output=%s, total=%s%s; стоимость~$%.4f",
        input_tokens,
        output_tokens,
        overall_tokens,
        suffix,
        spent,
    )

    limit_usd = float(totals.get("limit_usd", 0.0) or 0.0)
    remaining = max(0.0, limit_usd - spent)
    if limit_usd > 0:
        logger.info(
            "Счет: input=%s, output=%s, стоимость=$%.4f из лимита $%.2f, остаток=$%.2f",
            input_tokens,
            output_tokens,
            spent,
            limit_usd,
            remaining,
        )
    else:
        logger.info(
            "Счет: input=%s, output=%s, стоимость=$%.4f (лимит не задан)",
            input_tokens,
            output_tokens,
            spent,
        )
