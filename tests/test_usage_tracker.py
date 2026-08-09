from __future__ import annotations

import json
import multiprocessing
from pathlib import Path

import pytest

from sub_translate.utils import io_utils, usage_tracker


def _record_usage_in_process(stats_path: str, lock_cache_path: str, repeats: int) -> None:
    from sub_translate.utils import io_utils as process_io_utils
    from sub_translate.utils import usage_tracker as process_usage_tracker

    process_usage_tracker.USAGE_STATS_FILE = Path(stats_path)
    process_io_utils.CACHE_DIR = Path(lock_cache_path)
    for _ in range(repeats):
        process_usage_tracker.record_usage("process-test", "test-model", 2, 3, 0.01)


@pytest.fixture
def isolated_usage_stats(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    stats_path = tmp_path / "usage" / "stats.json"
    monkeypatch.setattr(usage_tracker, "USAGE_STATS_FILE", stats_path)
    monkeypatch.setattr(io_utils, "CACHE_DIR", tmp_path / "atomic-locks")
    return stats_path


def test_record_usage_writes_utf8_atomically(isolated_usage_stats: Path) -> None:
    totals = usage_tracker.record_usage("test", "model", 4, 5, 0.25)

    assert totals["input_tokens"] == 4
    assert totals["output_tokens"] == 5
    assert totals["cost_usd"] == pytest.approx(0.25)
    assert not isolated_usage_stats.read_bytes().startswith(b"\xef\xbb\xbf")
    assert list(isolated_usage_stats.parent.glob(".*.tmp")) == []


def test_parallel_processes_do_not_lose_usage_updates(
    isolated_usage_stats: Path,
    tmp_path: Path,
) -> None:
    process_count = 4
    repeats = 20
    context = multiprocessing.get_context("spawn")
    processes = [
        context.Process(
            target=_record_usage_in_process,
            args=(str(isolated_usage_stats), str(tmp_path / "atomic-locks"), repeats),
        )
        for _ in range(process_count)
    ]

    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=30)
        assert process.exitcode == 0

    stats = json.loads(isolated_usage_stats.read_text(encoding="utf-8"))
    expected_calls = process_count * repeats
    assert stats["totals"]["input_tokens"] == expected_calls * 2
    assert stats["totals"]["output_tokens"] == expected_calls * 3
    assert stats["totals"]["cost_usd"] == pytest.approx(expected_calls * 0.01)
    assert stats["by_model"]["test-model"] == stats["totals"]
    assert len(stats["history"]) == expected_calls
