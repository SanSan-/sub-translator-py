from __future__ import annotations

from sub_translate.web.preparation import PREPARATION_LOG_LIMIT, PreparationTracker


def test_preparation_tracker_exposes_progress_and_terminal_error() -> None:
    tracker = PreparationTracker()
    operation_id = tracker.begin("pick", phase="dialog", message="Открыт диалог.")
    tracker.update(
        operation_id,
        phase="queueing",
        discovered=80,
        processed=31,
        total=80,
        message="Подготовлено файлов: 31 из 80.",
    )

    running = tracker.snapshot()

    assert running["generation"] == 1
    assert running["operation_id"] == operation_id
    assert running["operation"] == "pick"
    assert running["status"] == "running"
    assert running["phase"] == "queueing"
    assert running["active"] is True
    assert running["discovered"] == 80
    assert running["processed"] == 31
    assert running["total"] == 80
    assert running["error"] is None

    tracker.fail(operation_id, message="Каталог стал недоступен.")
    failed = tracker.snapshot()
    assert failed["status"] == "error"
    assert failed["phase"] == "error"
    assert failed["active"] is False
    assert failed["error"] == "Каталог стал недоступен."
    assert failed["finished_at"] is not None


def test_preparation_tracker_ignores_stale_updates_and_bounds_history() -> None:
    tracker = PreparationTracker()
    stale = tracker.begin("pick", phase="dialog", message="Старый выбор.")
    current = tracker.begin("refresh", phase="collecting", message="Новое обновление.")

    assert tracker.update(stale, phase="error", message="Запоздалая ошибка.") is False
    for index in range(PREPARATION_LOG_LIMIT + 20):
        assert tracker.update(
            current,
            phase="queueing",
            processed=index,
            message=f"Шаг {index}.",
        )

    snapshot = tracker.snapshot()
    assert snapshot["generation"] == 2
    assert snapshot["operation"] == "refresh"
    assert len(snapshot["logs"]) == PREPARATION_LOG_LIMIT
    assert all("Запоздалая" not in entry["message"] for entry in snapshot["logs"])


def test_preparation_snapshot_is_independent_and_counters_are_monotonic() -> None:
    tracker = PreparationTracker()
    operation_id = tracker.begin("refresh", phase="collecting", message="Начат сбор.")
    tracker.update(operation_id, discovered=1_000, processed=900, total=1_000)
    snapshot = tracker.snapshot()
    snapshot["logs"].clear()
    snapshot["message"] = "изменено снаружи"

    tracker.update(operation_id, discovered=1, processed=1, total=1)
    tracker.finish(operation_id, message="Сбор завершён.")
    actual = tracker.snapshot()

    assert actual["message"] == "Сбор завершён."
    assert actual["status"] == "done"
    assert actual["discovered"] == 1_000
    assert actual["processed"] == 900
    assert actual["total"] == 1_000
    assert len(actual["logs"]) == 2
