from __future__ import annotations

import json
import logging
import time
from copy import deepcopy
from typing import Any, Dict, Optional

from sub_translate.constants import CACHE_DIR, USAGE_STATS_FILE

_DEFAULT_STATS: Dict[str, Any] = {
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


def _normalize_totals(value: Any) -> Dict[str, Any]:
    if not isinstance(value, dict):
        return deepcopy(_DEFAULT_STATS["totals"])
    result = deepcopy(_DEFAULT_STATS["totals"])
    for key in ("input_tokens", "output_tokens"):
        try:
            result[key] = int(value.get(key, 0))
        except (TypeError, ValueError):
            result[key] = 0
    try:
        result["cost_usd"] = float(value.get("cost_usd", 0.0))
    except (TypeError, ValueError):
        result["cost_usd"] = 0.0
    return result


def _load_stats() -> Dict[str, Any]:
    if not USAGE_STATS_FILE.exists():
        return deepcopy(_DEFAULT_STATS)
    try:
        data = json.loads(USAGE_STATS_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return deepcopy(_DEFAULT_STATS)
    if not isinstance(data, dict):
        return deepcopy(_DEFAULT_STATS)

    stats = deepcopy(_DEFAULT_STATS)
    stats["version"] = int(data.get("version", _DEFAULT_STATS["version"]))
    stats["totals"] = _normalize_totals(data.get("totals"))
    try:
        stats["limit_usd"] = float(data.get("limit_usd", _DEFAULT_STATS["limit_usd"]))
    except (TypeError, ValueError):
        stats["limit_usd"] = _DEFAULT_STATS["limit_usd"]

    by_model: Dict[str, Any] = {}
    raw_by_model = data.get("by_model")
    if isinstance(raw_by_model, dict):
        for model, totals in raw_by_model.items():
            if isinstance(model, str):
                by_model[model] = _normalize_totals(totals)
    stats["by_model"] = by_model

    history = []
    raw_history = data.get("history")
    if isinstance(raw_history, list):
        for item in raw_history[-HISTORY_LIMIT:]:
            if not isinstance(item, dict):
                continue
            history.append(
                {
                    "timestamp": int(item.get("timestamp", 0)),
                    "source": str(item.get("source", "")),
                    "model": str(item.get("model", "")),
                    "input_tokens": int(item.get("input_tokens", 0) or 0),
                    "output_tokens": int(item.get("output_tokens", 0) or 0),
                    "cost_usd": float(item.get("cost_usd", 0.0) or 0.0),
                }
            )
    stats["history"] = history
    stats["last_updated"] = int(data.get("last_updated", 0))
    return stats


def _save_stats(stats: Dict[str, Any]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    USAGE_STATS_FILE.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def record_usage(
    source: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    cost_usd: Optional[float] = None,
) -> Dict[str, Any]:
    stats = _load_stats()
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
    _save_stats(stats)

    result = stats["totals"].copy()
    result["limit_usd"] = float(stats.get("limit_usd", 0.0))
    return result


def get_totals() -> Dict[str, Any]:
    stats = _load_stats()
    result = stats["totals"].copy()
    result["limit_usd"] = float(stats.get("limit_usd", 0.0))
    return result


def get_stats() -> Dict[str, Any]:
    return _load_stats()


def log_usage_summary(
    logger: logging.Logger,
    totals: Dict[str, Any],
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