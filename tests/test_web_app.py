from __future__ import annotations

import asyncio
import logging
import queue
import re
import threading
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import Mock

import httpx
import pytest
from fastapi.testclient import TestClient

from sub_translate import __version__
from sub_translate.enums import FileFormat
from sub_translate.service import ProcessingResult, ProcessingSettings, ProcessingStatus
from sub_translate.web import __main__ as web_main
from sub_translate.web import app as web_app

VALID_SRT = "1\n00:00:00,000 --> 00:00:01,000\nТекст\n"
VALID_VTT = "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nТекст\n"


@pytest.fixture(autouse=True)
def reset_job_registry() -> Iterator[None]:
    web_app.preparation_registry.reset_for_tests()
    with web_app._jobs_lock:
        web_app._jobs.clear()
    with web_app._active_job_lock:
        web_app._active_job_id = None
    yield
    with web_app._jobs_lock:
        web_app._jobs.clear()
    with web_app._active_job_lock:
        web_app._active_job_id = None
    web_app.preparation_registry.reset_for_tests()


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(web_app.app, base_url="http://127.0.0.1") as test_client:
        yield test_client


def request_from(
    client_address: tuple[str, int],
    method: str,
    path: str,
    *,
    headers: dict[str, str] | None = None,
    json_body: dict[str, object] | None = None,
) -> httpx.Response:
    async def send() -> httpx.Response:
        transport = httpx.ASGITransport(app=web_app.app, client=client_address)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://127.0.0.1",
        ) as async_client:
            return await async_client.request(method, path, headers=headers, json=json_body)

    return asyncio.run(send())


def make_job(
    paths: list[Path] | None = None,
    settings: web_app.TranslationSettings | None = None,
    job_id: str = "a" * 32,
) -> web_app.TranslationJob:
    return web_app.TranslationJob(
        job_id=job_id,
        paths=paths or [Path("episode.srt")],
        settings=settings or web_app.TranslationSettings(api="google"),
        events=queue.Queue(),
        thread=None,
    )


def drain_events(job: web_app.TranslationJob) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    while not job.events.empty():
        events.append(job.events.get_nowait())
    return events


def isolated_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    return logger


def test_index_health_config_and_openapi(client: TestClient) -> None:
    index_response = client.get("/")
    health_response = client.get("/api/health")
    config_response = client.get("/api/ui-config")
    schema = client.get("/openapi.json").json()

    assert index_response.status_code == 200
    assert "Sub Translator" in index_response.text
    assert health_response.json() == {"status": "ok", "version": __version__}
    assert config_response.status_code == 200
    assert config_response.json()["defaults"]["batch_size"] == 21
    assert schema["info"]["version"] == __version__
    assert "500" in schema["paths"]["/api/pick"]["post"]["responses"]
    assert {"400", "500"} <= schema["paths"]["/api/refresh"]["post"]["responses"].keys()
    assert "/api/preparation-status" in schema["paths"]
    assert "409" in schema["paths"]["/api/translate"]["post"]["responses"]
    assert "503" in schema["paths"]["/api/translate"]["post"]["responses"]
    assert "404" in schema["paths"]["/api/jobs/{job_id}"]["get"]["responses"]
    assert "404" in schema["paths"]["/api/stream/{job_id}"]["get"]["responses"]
    assert "409" in schema["paths"]["/api/unload"]["post"]["responses"]


def test_ui_config_is_registry_driven_and_contains_no_local_paths() -> None:
    config = web_app._build_ui_config()
    profiles = config["translators"]
    expected_ids = [metadata.id for metadata in web_app._product_translator_metadata()]

    assert [profile["id"] for profile in profiles] == expected_ids
    assert expected_ids == ["google", "agent", "nllb-600m", "translategemma", "translategemma-12b", "seedx"]
    assert all("model_path" not in profile for profile in profiles)
    assert all("worker_requirements" not in profile for profile in profiles)
    assert config["languages"].keys() == set(expected_ids)
    nllb = next(profile for profile in profiles if profile["id"] == "nllb-600m")
    assert nllb["supports_cpu_fallback"] is True
    profile_12b = next(profile for profile in profiles if profile["id"] == "translategemma-12b")
    assert profile_12b["model"]["id"] == "google/translategemma-12b-it"
    assert profile_12b["model"]["quantization"] == "bitsandbytes-nf4-double"
    assert profile_12b["default_timeout_seconds"] == 3_600
    assert profile_12b["supports_cpu_fallback"] is False
    seedx = next(profile for profile in profiles if profile["id"] == "seedx")
    assert seedx["model"]["id"] == "ByteDance-Seed/Seed-X-PPO-7B"
    assert seedx["model"]["quantization"] == "bitsandbytes-nf4-double"
    assert seedx["default_timeout_seconds"] == 3_600
    assert seedx["supports_cpu_fallback"] is False


def test_translation_settings_normalize_registry_alias_and_optional_paths() -> None:
    settings = web_app.TranslationSettings(
        api="translate-gemma-4b",
        tld=" COM ",
        model_path="  ",
        model_revision="  ",
        worker_python_path="  ",
    )

    assert settings.api == "translategemma"
    assert settings.tld == "com"
    assert settings.model_path is None
    assert settings.model_revision is None
    assert settings.worker_python_path is None


def test_translation_settings_use_profile_timeout_unless_explicit() -> None:
    defaulted = web_app.TranslationSettings(api="seedx")
    explicit = web_app.TranslationSettings(api="seedx", timeout=45)

    assert defaulted.timeout == 3_600
    assert explicit.timeout == 45


def test_static_form_controls_have_labels_and_safe_stream_url() -> None:
    static_dir = Path(web_app.__file__).resolve().parent / "static"
    html = (static_dir / "index.html").read_text(encoding="utf-8")
    javascript = (static_dir / "app.js").read_text(encoding="utf-8")
    control_ids = set(re.findall(r'<(?:input|select|textarea)\b[^>]*\bid="([^"]+)"', html))
    label_targets = set(re.findall(r'<label\b[^>]*\bfor="([^"]+)"', html))

    assert control_ids <= label_targets
    assert 'role="button"' not in html
    assert 'placeholder="gpt-5.6-luna"' in html
    assert "JOB_ID_PATTERN.test(jobId)" in javascript
    assert "encodeURIComponent(jobId)" in javascript
    assert "new URL(" in javascript


def test_web_main_runs_local_uvicorn(monkeypatch) -> None:
    run = Mock()
    monkeypatch.setattr(web_main.uvicorn, "run", run)

    web_main.main()

    run.assert_called_once_with(
        "sub_translate.web.app:app",
        host="127.0.0.1",
        port=7860,
        reload=False,
        proxy_headers=False,
    )


def test_remote_get_is_rejected_even_with_forwarded_loopback_headers() -> None:
    response = request_from(
        ("203.0.113.10", 41000),
        "GET",
        "/api/health",
        headers={
            "Forwarded": "for=127.0.0.1;host=127.0.0.1;proto=http",
            "X-Forwarded-For": "127.0.0.1",
            "X-Forwarded-Host": "127.0.0.1",
            "X-Forwarded-Proto": "http",
        },
    )

    assert response.status_code == 403
    assert response.json() == {"detail": web_app._LOCAL_ACCESS_ERROR}


def test_testclient_literal_requires_test_transport_marker() -> None:
    response = request_from(("testclient", 50000), "GET", "/api/health")

    assert response.status_code == 403


def test_dns_rebinding_host_is_rejected(client: TestClient) -> None:
    response = client.get(
        "/api/health",
        headers={"Host": "127.0.0.1.attacker.example"},
    )

    assert response.status_code == 403
    assert response.json() == {"detail": web_app._LOCAL_ACCESS_ERROR}


def test_remote_post_is_rejected_before_handler(monkeypatch) -> None:
    unload = Mock()
    monkeypatch.setattr(web_app, "unload_all_local_translators", unload)

    response = request_from(
        ("198.51.100.20", 42000),
        "POST",
        "/api/unload",
        headers={"X-Forwarded-For": "127.0.0.1"},
        json_body={},
    )

    assert response.status_code == 403
    unload.assert_not_called()


def test_cross_origin_post_is_rejected_and_same_origin_is_allowed(
    client: TestClient,
    monkeypatch,
) -> None:
    unload = Mock()
    monkeypatch.setattr(web_app, "unload_all_local_translators", unload)

    rejected = client.post(
        "/api/unload",
        headers={"Origin": "http://attacker.example"},
        json={},
    )
    allowed = client.post(
        "/api/unload",
        headers={"Origin": "http://127.0.0.1"},
        json={},
    )

    assert rejected.status_code == 403
    assert rejected.json() == {"detail": web_app._ORIGIN_ACCESS_ERROR}
    assert allowed.status_code == 200
    unload.assert_called_once_with()


@pytest.mark.parametrize(
    "settings",
    [
        {"batch_size": 0},
        {"threads": 0},
        {"timeout": 0},
        {"request_delay_ms": -1},
        {"smart_split_max_lines": 0},
    ],
)
def test_settings_validation_rejects_invalid_numbers(
    client: TestClient,
    settings: dict[str, int],
) -> None:
    response = client.post(
        "/api/refresh",
        json={"paths": [], "settings": {"api": "google", **settings}},
    )

    assert response.status_code == 422


def test_settings_validation_rejects_unregistered_translator(client: TestClient) -> None:
    unknown_id = "https://attacker.example/not-registered-translator"

    response = client.post(
        "/api/refresh",
        json={"paths": [], "settings": {"api": unknown_id}},
    )

    assert response.status_code == 422
    assert response.json() == {"detail": web_app._REQUEST_VALIDATION_ERROR}
    assert unknown_id not in response.text
    assert all(key not in response.text for key in ('"input"', '"ctx"', '"msg"', '"url"'))


@pytest.mark.parametrize(
    "tld",
    ["com@attacker.example", "com/path", "com:443", "attacker.example"],
)
def test_settings_validation_rejects_unsafe_google_tld_without_echo(
    client: TestClient,
    tld: str,
) -> None:
    response = client.post(
        "/api/refresh",
        json={"paths": [], "settings": {"api": "google", "tld": tld}},
    )

    assert response.status_code == 422
    assert response.json() == {"detail": web_app._REQUEST_VALIDATION_ERROR}
    assert tld not in response.text


def test_request_models_reject_unknown_top_level_field(client: TestClient) -> None:
    unknown_field = "unexpected-secret-field"
    unknown_value = "secret-value"
    response = client.post(
        "/api/refresh",
        json={"paths": [], "settings": {"api": "google"}, unknown_field: unknown_value},
    )

    assert response.status_code == 422
    assert response.json() == {"detail": web_app._REQUEST_VALIDATION_ERROR}
    assert unknown_field not in response.text
    assert unknown_value not in response.text


def test_event_recording_tracks_state_and_limits_logs() -> None:
    job = make_job()
    job.log_lines = [f"строка-{index}" for index in range(web_app.LOG_HISTORY_LIMIT)]

    web_app._record_job_event(job, {"type": "log", "message": "новая строка\n"})
    web_app._record_job_event(job, {"type": "job", "total": 3})
    web_app._record_job_event(
        job,
        {
            "type": "file",
            "path": "episode.srt",
            "state": "done",
            "progress": 100,
            "output": "episode.ru.srt",
        },
    )
    web_app._record_job_event(job, {"type": "unknown"})
    web_app._record_job_event(job, {"type": "file", "path": ""})

    assert len(job.log_lines) == web_app.LOG_HISTORY_LIMIT
    assert job.log_lines[-1] == "новая строка"
    assert job.job_total == 3
    assert job.file_states["episode.srt"] == {
        "state": "done",
        "progress": 100,
        "output": "episode.ru.srt",
    }


def test_queue_log_handler_emits_queue_and_snapshot() -> None:
    events: queue.Queue[dict[str, object]] = queue.Queue()
    snapshots: list[dict[str, object]] = []
    handler = web_app.QueueLogHandler(events, snapshots.append)
    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))

    handler.emit(logging.LogRecord("test", logging.INFO, "", 0, "сообщение", (), None))

    assert events.get_nowait() == {"type": "log", "message": "INFO: сообщение"}
    assert snapshots == [{"type": "log", "message": "INFO: сообщение"}]


def test_bounded_event_queue_discards_oldest_events_and_preserves_done() -> None:
    job = make_job()
    job.events = web_app.BoundedEventQueue(maxsize=3)

    for index in range(5):
        web_app._emit_job_event(job, {"type": "log", "message": f"строка-{index}"})
    web_app._emit_job_event(job, {"type": "done", "status": "ok"})
    web_app._emit_job_event(job, {"type": "log", "message": "после завершения"})

    events = drain_events(job)
    assert events == [
        {"type": "log", "message": "строка-3"},
        {"type": "log", "message": "строка-4"},
        {"type": "done", "status": "ok"},
    ]


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("lecture.cc.srt", True),
        ("movie.web.srt", True),
        ("lesson.ru.srt", False),
        ("lesson.en.srt", True),
    ],
)
def test_folder_source_excludes_only_target_language(name: str, expected: bool) -> None:
    assert web_app._is_folder_source(Path(name), "ru") is expected


def test_iter_subtitles_filters_format_source_and_target(tmp_path: Path) -> None:
    expected = tmp_path / "lesson.en.srt"
    nested = tmp_path / "Вложенная папка"
    nested.mkdir()
    nested_expected = nested / "урок с пробелом.en.ASS"
    expected.write_text("1\n", encoding="utf-8")
    nested_expected.write_text("[Script Info]\n", encoding="utf-8")
    (tmp_path / "plain.vtt").write_text("WEBVTT\n", encoding="utf-8")
    (tmp_path / "lesson.ru.srt").write_text("1\n", encoding="utf-8")
    other_language = tmp_path / "lesson.ja.ass"
    other_language.write_text("[Script Info]\n", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("text", encoding="utf-8")

    result = list(web_app._iter_subtitles(tmp_path, "en", "ru"))

    assert result == sorted(
        [nested_expected, expected, other_language, tmp_path / "plain.vtt"],
        key=lambda path: (str(path).casefold(), str(path)),
    )
    with pytest.raises(web_app.PickerError, match="Выбранный каталог"):
        web_app._iter_subtitles(tmp_path / "missing", "en", "ru")


def test_build_items_marks_full_identity_or_existing_output_as_cached(
    tmp_path: Path,
    monkeypatch,
) -> None:
    first = tmp_path / "first.srt"
    second = tmp_path / "second.srt"
    first.write_text(VALID_SRT, encoding="utf-8")
    second.write_text(VALID_SRT, encoding="utf-8")
    outputs = {
        first: tmp_path / "first.ru.srt",
        second: tmp_path / "second.ru.srt",
    }
    outputs[second].write_text("1\n", encoding="utf-8")
    checked_identities: list[Path] = []

    def has_cached_output(identity: Path, _logger: logging.Logger) -> bool:
        checked_identities.append(identity)
        return identity == first

    monkeypatch.setattr(
        web_app,
        "build_output_cache_identity",
        lambda path, *_args, **_kwargs: path,
    )
    monkeypatch.setattr(
        web_app,
        "has_cached_output",
        has_cached_output,
    )
    monkeypatch.setattr(
        web_app,
        "resolve_io_paths",
        lambda path, *_args: ("en", outputs[path]),
    )

    items = web_app._build_items(
        [first, second],
        web_app.TranslationSettings(api="google"),
    )

    assert [item["cached"] for item in items] == [True, True]
    assert [item["format"] for item in items] == ["srt", "srt"]
    assert checked_identities == [first]


def test_pick_and_refresh_endpoints(client: TestClient, monkeypatch, tmp_path: Path) -> None:
    subtitle = tmp_path / "episode.srt"
    subtitle.write_text(VALID_SRT, encoding="utf-8")
    received: dict[str, object] = {}

    def fake_pick_paths(kind, *, recursive, accept_path, progress_callback):
        received.update(kind=kind, recursive=recursive, accepted=accept_path(subtitle))
        progress_callback({"phase": "collecting", "discovered": 1, "total": 1})
        return web_app.PickSelection(mode="files", paths=(subtitle,))

    monkeypatch.setattr(web_app, "pick_paths", fake_pick_paths)

    picked = client.post(
        "/api/pick",
        json={"kind": "file", "settings": {"api": "google"}},
    )
    refreshed = client.post(
        "/api/refresh",
        json={"paths": [str(subtitle)], "settings": {"api": "google"}},
    )

    assert picked.status_code == 200
    assert picked.json()["path"] == ""
    assert picked.json()["paths"] == [str(subtitle)]
    assert picked.json()["cancelled"] is False
    assert picked.json()["recursive"] is False
    assert len(picked.json()["items"]) == 1
    assert received == {"kind": "file", "recursive": False, "accepted": True}
    assert refreshed.status_code == 200
    assert refreshed.json()["paths"] == [str(subtitle)]
    assert refreshed.json()["recursive"] is False
    assert refreshed.json()["items"][0]["name"] == subtitle.name
    assert client.get("/api/preparation-status").json()["status"] == "done"


def test_pick_folder_handles_cancel_and_selected_folder(
    client: TestClient,
    monkeypatch,
    tmp_path: Path,
) -> None:
    subtitle = tmp_path / "episode.srt"
    subtitle.write_text(VALID_SRT, encoding="utf-8")
    monkeypatch.setattr(
        web_app,
        "pick_paths",
        lambda *_args, **_kwargs: web_app.PickSelection(mode="folder", paths=()),
    )
    cancelled = client.post("/api/pick", json={"kind": "folder"})

    monkeypatch.setattr(
        web_app,
        "pick_paths",
        lambda *_args, **_kwargs: web_app.PickSelection(
            mode="folder",
            paths=(subtitle,),
            folder=tmp_path,
        ),
    )
    selected = client.post("/api/pick", json={"kind": "folder"})

    assert cancelled.json() == {
        "mode": "folder",
        "path": "",
        "paths": [],
        "items": [],
        "cancelled": True,
        "recursive": True,
    }
    assert selected.status_code == 200
    assert selected.json()["path"] == str(tmp_path)
    assert selected.json()["paths"] == [str(subtitle)]
    assert selected.json()["items"][0]["name"] == "episode.srt"
    assert selected.json()["recursive"] is True


def test_refresh_reexpands_folder_recursively_and_excludes_translated_target(
    client: TestClient,
    tmp_path: Path,
) -> None:
    nested = tmp_path / "Вложенная папка"
    nested.mkdir()
    top = tmp_path / "верх.en.srt"
    child = nested / "низ с пробелом.en.VTT"
    translated = nested / "низ с пробелом.google.ru.vtt"
    top.write_text(VALID_SRT, encoding="utf-8")
    child.write_text(VALID_VTT, encoding="utf-8")
    translated.write_text(VALID_VTT, encoding="utf-8")

    response = client.post(
        "/api/refresh",
        json={
            "paths": [str(tmp_path)],
            "settings": {"api": "google", "source_lang": "en", "target_lang": "ru"},
            "recursive": True,
        },
    )

    assert response.status_code == 200
    assert response.json()["path"] == str(tmp_path)
    assert response.json()["recursive"] is True
    expected = sorted((str(child), str(top)), key=lambda value: (value.casefold(), value))
    assert response.json()["paths"] == expected
    assert [item["path"] for item in response.json()["items"]] == expected


def test_pick_error_remains_visible_in_preparation_status(
    client: TestClient,
    monkeypatch,
) -> None:
    raw_detail = r"RAW_OS_PRIVATE_DETAIL C:\private\folder"
    log_exception = Mock()

    def fail_pick(*_args, **_kwargs):
        try:
            raise PermissionError(raw_detail)
        except PermissionError as exc:
            raise web_app.PickerError("Не удалось прочитать выбранный каталог.") from exc

    monkeypatch.setattr(web_app, "pick_paths", fail_pick)
    monkeypatch.setattr(web_app._WEB_LOGGER, "exception", log_exception)

    response = client.post("/api/pick", json={"kind": "folder"})
    status = client.get("/api/preparation-status").json()

    assert response.status_code == 500
    assert status["status"] == "error"
    assert status["active"] is False
    assert "Не удалось прочитать" in status["error"]
    assert raw_detail not in response.text
    assert "Traceback" not in response.text
    log_exception.assert_called_once_with("Системный выбор субтитров завершился ошибкой.")


def test_file_pick_keeps_valid_neighbor_and_returns_safe_error_cards(
    client: TestClient,
    monkeypatch,
    tmp_path: Path,
) -> None:
    valid = tmp_path / "01-valid.srt"
    missing = tmp_path / "02-missing.srt"
    malformed = tmp_path / "03-malformed.vtt"
    valid.write_text(VALID_SRT, encoding="utf-8")
    malformed.write_text("WEBVTT\n", encoding="utf-8")
    monkeypatch.setattr(
        web_app,
        "pick_paths",
        lambda *_args, **_kwargs: web_app.PickSelection(
            mode="files",
            paths=(valid, missing, malformed),
        ),
    )

    response = client.post("/api/pick", json={"kind": "file", "settings": {"api": "google"}})
    by_name = {item["name"]: item for item in response.json()["items"]}

    assert response.status_code == 200
    assert "state" not in by_name[valid.name]
    assert by_name[missing.name]["state"] == "error"
    assert by_name[missing.name]["error"] == "Файл не найден или был перемещён."
    assert "output" not in by_name[missing.name]
    assert by_name[malformed.name]["state"] == "error"
    assert "не соответствует формату VTT" in by_name[malformed.name]["error"]
    assert "output" not in by_name[malformed.name]
    assert "Traceback" not in response.text
    assert client.get("/api/preparation-status").json()["status"] == "done"


def test_refresh_isolates_permission_error_without_exposing_raw_os_message(
    client: TestClient,
    monkeypatch,
    tmp_path: Path,
) -> None:
    valid = tmp_path / "01-valid.srt"
    blocked = tmp_path / "02-blocked.srt"
    valid.write_text(VALID_SRT, encoding="utf-8")
    blocked.write_text(VALID_SRT, encoding="utf-8")
    real_read_text = web_app.read_text
    log_exception = Mock()

    def selective_read(path: Path) -> str:
        if path == blocked:
            raise PermissionError("RAW_OS_PRIVATE_DETAIL")
        return real_read_text(path)

    monkeypatch.setattr(web_app, "read_text", selective_read)
    monkeypatch.setattr(web_app._WEB_LOGGER, "exception", log_exception)

    response = client.post(
        "/api/refresh",
        json={"paths": [str(valid), str(blocked)], "settings": {"api": "google"}},
    )
    by_name = {item["name"]: item for item in response.json()["items"]}

    assert response.status_code == 200
    assert "state" not in by_name[valid.name]
    assert by_name[blocked.name]["error"] == "Нет доступа к файлу."
    assert "output" not in by_name[blocked.name]
    assert "RAW_OS_PRIVATE_DETAIL" not in response.text
    assert "Traceback" not in response.text
    log_exception.assert_called_once_with(
        "Не удалось подготовить файл %s (%s).",
        blocked,
        "PermissionError",
    )


def test_apply_agent_prompts_uses_inline_values(monkeypatch) -> None:
    system = Mock()
    user = Mock()
    monkeypatch.setattr(web_app, "set_system_prompt", system)
    monkeypatch.setattr(web_app, "set_user_prompt_template", user)
    settings = web_app.TranslationSettings(
        api="agent",
        agent_system_prompt="system",
        agent_prompt="user",
    )

    web_app._apply_agent_prompts(settings)

    system.assert_called_once_with("system")
    user.assert_called_once_with("user", variant="ui")


def test_apply_agent_prompts_reads_files_and_rejects_missing(monkeypatch, tmp_path: Path) -> None:
    system_path = tmp_path / "system.txt"
    user_path = tmp_path / "user.txt"
    system_path.write_text("система", encoding="utf-8")
    user_path.write_text("пользователь", encoding="utf-8")
    system = Mock()
    user = Mock()
    monkeypatch.setattr(web_app, "set_system_prompt", system)
    monkeypatch.setattr(web_app, "set_user_prompt_template", user)

    web_app._apply_agent_prompts(
        web_app.TranslationSettings(
            api="agent",
            agent_system_prompt_file=str(system_path),
            agent_prompt_file=str(user_path),
        )
    )

    system.assert_called_once_with("система")
    user.assert_called_once_with("пользователь", variant="file")
    missing_settings = web_app.TranslationSettings(
        api="agent",
        agent_system_prompt_file=str(tmp_path / "missing.txt"),
    )
    with pytest.raises(FileNotFoundError, match="Системный промпт не найден"):
        web_app._apply_agent_prompts(missing_settings)


def test_smart_split_settings_and_progress_are_bounded() -> None:
    settings = web_app.TranslationSettings(
        api="google",
        smart_split=True,
        smart_split_max_chars=123,
    )
    job = make_job(settings=settings)

    result = web_app._build_smart_split_settings(settings)
    web_app._report_progress(job, Path("episode.srt"), 1, 1, 5, 0)
    web_app._report_progress(job, Path("episode.srt"), 1, 1, 200, 100)

    assert result is not None
    assert result.max_chars == 123
    assert web_app._build_smart_split_settings(web_app.TranslationSettings(api="google")) is None
    assert job.file_states["episode.srt"]["progress"] == 100


def test_batch_callbacks_report_started_cached_and_translated(tmp_path: Path) -> None:
    path = tmp_path / "episode.srt"
    output = tmp_path / "episode.ru.srt"
    job = make_job(paths=[path])

    web_app._report_batch_started(job, path, 1, 1)
    web_app._report_batch_result(
        job,
        ProcessingResult(path, output, ProcessingStatus.CACHED, FileFormat.SRT),
        1,
        1,
    )

    assert job.file_states[str(path)]["state"] == "cached"
    assert job.file_states[str(path)]["output"] == str(output)


def test_batch_result_callback_reports_isolated_error(tmp_path: Path) -> None:
    path = tmp_path / "broken.srt"
    job = make_job(paths=[path])

    web_app._report_batch_result(
        job,
        ProcessingResult(path, None, ProcessingStatus.ERROR, FileFormat.SRT, "RuntimeError"),
        1,
        1,
    )

    assert job.file_states[str(path)] == {
        "state": "error",
        "progress": 100,
        "error": "Не удалось перевести файл.",
    }


def test_run_job_reports_partial_result_and_cleans_handlers(monkeypatch, tmp_path: Path) -> None:
    paths = [tmp_path / "ok.srt", tmp_path / "broken.srt"]
    settings = web_app.TranslationSettings(api="google", force=True)
    job = make_job(paths=paths, settings=settings)
    logger = isolated_logger("test_run_job_reports_partial_result_and_cleans_handlers")
    monkeypatch.setattr(web_app, "_get_logger", lambda *_args, **_kwargs: logger)
    monkeypatch.setattr(web_app, "attach_log_handler", Mock())
    monkeypatch.setattr(web_app, "detach_log_handler", Mock())
    apply_agent_prompts = Mock(return_value={})
    monkeypatch.setattr(web_app, "_apply_agent_prompts", apply_agent_prompts)
    captured: dict[str, object] = {}

    def process_batch_stub(input_paths, settings, **kwargs):
        captured.update(input_paths=list(input_paths), settings=settings, kwargs=kwargs)
        results = [
            ProcessingResult(paths[0], paths[0].with_suffix(".ru.srt"), ProcessingStatus.TRANSLATED),
            ProcessingResult(paths[1], None, ProcessingStatus.ERROR, error_type="RuntimeError"),
        ]
        for index, result in enumerate(results, start=1):
            kwargs["started_callback"](result.input_path, index, len(results))
            kwargs["result_callback"](result, index, len(results))
        return results

    monkeypatch.setattr(web_app, "process_subtitle_batch", process_batch_stub)
    with web_app._active_job_lock:
        web_app._active_job_id = job.job_id

    web_app._run_job(job)

    events = drain_events(job)
    assert job.completed is True
    assert events[-1] == {"type": "done", "status": "partial"}
    assert any(event.get("type") == "job" for event in events)
    assert any("--force" in str(event.get("message")) for event in events)
    assert isinstance(captured["settings"], ProcessingSettings)
    assert captured["input_paths"] == paths
    assert logger.handlers == []
    assert web_app._active_job_id is None
    apply_agent_prompts.assert_not_called()


def test_finish_job_downgrades_ok_when_some_files_are_unresolved() -> None:
    paths = [Path("done.srt"), Path("failed.srt"), Path("started.srt")]
    job = make_job(paths=paths)
    job.file_states[str(paths[0])] = {
        "state": "done",
        "progress": 100,
        "output": "done.ru.srt",
    }
    job.file_states[str(paths[1])] = {
        "state": "error",
        "progress": 100,
        "error": "Исходная ошибка.",
    }
    job.file_states[str(paths[2])] = {"state": "started", "progress": 40}

    web_app._finish_job(job, "ok")

    assert job.status == "partial"
    assert job.file_states[str(paths[0])]["state"] == "done"
    assert job.file_states[str(paths[0])]["output"] == "done.ru.srt"
    assert job.file_states[str(paths[1])]["error"] == "Исходная ошибка."
    assert job.file_states[str(paths[2])] == {
        "state": "error",
        "progress": 100,
        "error": "Файл не был обработан до завершения задачи.",
    }


@pytest.mark.parametrize("api", ["google", "nllb-600m", "translategemma", "translategemma-12b", "seedx"])
def test_run_job_ignores_stale_agent_settings_for_non_agent_profiles(
    api: str,
    monkeypatch,
    tmp_path: Path,
) -> None:
    settings = web_app.TranslationSettings(
        api=api,
        agent_model="stale-model",
        agent_system_prompt="stale-system",
        agent_prompt="stale-user",
        agent_system_prompt_file=str(tmp_path / "missing-system.txt"),
        agent_prompt_file=str(tmp_path / "missing-user.txt"),
    )
    job = make_job(settings=settings)
    logger = isolated_logger(f"test_run_job_ignores_agent_settings.{api}")
    apply_agent_prompts = Mock(side_effect=AssertionError("agent prompts must be ignored"))
    read_prompt_file = Mock(side_effect=AssertionError("agent files must be ignored"))
    attach_agent_log = Mock()
    detach_agent_log = Mock()

    def successful_batch(input_paths, _settings, **kwargs):
        path = input_paths[0]
        result = ProcessingResult(
            path,
            path.with_suffix(".ru.srt"),
            ProcessingStatus.TRANSLATED,
        )
        kwargs["result_callback"](result, 1, 1)
        return [result]

    process_batch = Mock(side_effect=successful_batch)
    monkeypatch.setattr(web_app, "_get_logger", lambda *_args, **_kwargs: logger)
    monkeypatch.setattr(web_app, "_apply_agent_prompts", apply_agent_prompts)
    monkeypatch.setattr(web_app, "read_text", read_prompt_file)
    monkeypatch.setattr(web_app, "attach_log_handler", attach_agent_log)
    monkeypatch.setattr(web_app, "detach_log_handler", detach_agent_log)
    monkeypatch.setattr(web_app, "process_subtitle_batch", process_batch)

    web_app._run_job(job)

    assert drain_events(job)[-1] == {"type": "done", "status": "ok"}
    assert job.prompt_signatures == {}
    processing = process_batch.call_args.args[1]
    assert processing.api == api
    assert processing.agent_model is None
    assert dict(processing.prompt_signatures) == {}
    apply_agent_prompts.assert_not_called()
    read_prompt_file.assert_not_called()
    attach_agent_log.assert_not_called()
    detach_agent_log.assert_not_called()


def test_run_job_empty_result_is_terminal_error(monkeypatch) -> None:
    paths = [Path("first.srt"), Path("second.srt")]
    job = make_job(paths=paths)
    logger = isolated_logger("test_run_job_empty_result_is_terminal_error")
    monkeypatch.setattr(web_app, "_get_logger", lambda *_args, **_kwargs: logger)
    monkeypatch.setattr(web_app, "process_subtitle_batch", Mock(return_value=[]))

    web_app._run_job(job)

    assert job.completed is True
    assert job.status == "error"
    assert drain_events(job)[-1] == {"type": "done", "status": "error"}
    assert {state["state"] for state in job.file_states.values()} == {"error"}
    assert set(job.file_states) == {str(path) for path in paths}


def test_run_job_reports_prompt_error(monkeypatch) -> None:
    job = make_job(settings=web_app.TranslationSettings(api="agent"))
    logger = isolated_logger("test_run_job_reports_prompt_error")
    monkeypatch.setattr(web_app, "_get_logger", lambda *_args, **_kwargs: logger)
    monkeypatch.setattr(web_app, "attach_log_handler", Mock())
    monkeypatch.setattr(web_app, "detach_log_handler", Mock())
    monkeypatch.setattr(web_app, "_apply_agent_prompts", Mock(side_effect=ValueError("bad")))
    monkeypatch.setattr(web_app._WEB_LOGGER, "exception", Mock())

    web_app._run_job(job)

    assert drain_events(job)[-1] == {"type": "done", "status": "error"}
    assert job.completed is True


@pytest.mark.parametrize("failure_site", ["logger", "handler"])
def test_run_job_setup_failure_is_terminal_and_cleans_active_job(
    client: TestClient,
    monkeypatch,
    failure_site: str,
) -> None:
    raw_error = "RAW_DISK_FULL_PRIVATE_PATH"
    web_exception = Mock()
    logger = isolated_logger(f"test_run_job_setup_failure.{failure_site}")
    monkeypatch.setattr(web_app._WEB_LOGGER, "exception", web_exception)
    if failure_site == "logger":
        monkeypatch.setattr(
            web_app,
            "_get_logger",
            Mock(side_effect=OSError(raw_error)),
        )
    else:
        monkeypatch.setattr(web_app, "_get_logger", lambda *_args, **_kwargs: logger)
        real_add_handler = logger.addHandler
        failure_pending = True

        def add_handler_then_fail(handler: logging.Handler) -> None:
            nonlocal failure_pending
            real_add_handler(handler)
            if failure_pending:
                failure_pending = False
                raise OSError(raw_error)

        monkeypatch.setattr(logger, "addHandler", add_handler_then_fail)

    response = client.post(
        "/api/translate",
        json={"paths": ["episode.srt"], "settings": {"api": "google"}},
    )

    assert response.status_code == 200
    job = web_app._get_job(response.json()["job_id"])
    assert job is not None
    assert job.thread is not None
    job.thread.join(timeout=2)
    assert job.thread.is_alive() is False
    assert job.completed is True
    assert job.completed_at is not None
    assert drain_events(job)[-1] == {"type": "done", "status": "error"}
    assert web_app._active_job_id is None
    assert raw_error not in response.text
    assert "Traceback" not in response.text
    assert logger.handlers == []
    snapshot = client.get(f"/api/jobs/{job.job_id}").json()
    assert snapshot["completed"] is True
    assert snapshot["status"] == "error"
    assert snapshot["done"] == 1
    assert snapshot["items"][0]["state"] == "error"
    web_exception.assert_called()


def test_warnings_capture_restores_logger() -> None:
    handler = web_app.QueueLogHandler(queue.Queue())
    warnings_logger = logging.getLogger("py.warnings")
    original_level = warnings_logger.level
    original_propagate = warnings_logger.propagate

    capture = web_app._start_warnings_capture(True, handler)
    web_app._stop_warnings_capture(capture, handler)

    assert capture is not None
    assert handler not in warnings_logger.handlers
    assert warnings_logger.level == original_level
    assert warnings_logger.propagate == original_propagate
    assert web_app._start_warnings_capture(False, handler) is None


def test_warnings_capture_setup_failure_restores_logger(monkeypatch) -> None:
    handler = web_app.QueueLogHandler(queue.Queue())
    warnings_logger = logging.getLogger("py.warnings")
    original_level = warnings_logger.level
    original_propagate = warnings_logger.propagate
    original_filters = web_app.warnings.filters[:]
    real_add_handler = warnings_logger.addHandler
    failure_pending = True

    def add_handler_then_fail(candidate: logging.Handler) -> None:
        nonlocal failure_pending
        real_add_handler(candidate)
        if failure_pending:
            failure_pending = False
            raise OSError("RAW_WARNINGS_HANDLER_PRIVATE_DETAIL")

    monkeypatch.setattr(warnings_logger, "addHandler", add_handler_then_fail)

    with pytest.raises(OSError, match="RAW_WARNINGS_HANDLER_PRIVATE_DETAIL"):
        web_app._start_warnings_capture(True, handler)

    assert handler not in warnings_logger.handlers
    assert warnings_logger.level == original_level
    assert warnings_logger.propagate == original_propagate
    assert web_app.warnings.filters == original_filters


class FakeThread:
    def __init__(self, *, target, args, name: str, daemon: bool) -> None:
        self.target = target
        self.args = args
        self.name = name
        self.daemon = daemon
        self.started = False

    def start(self) -> None:
        self.started = True


class FailingStartThread(FakeThread):
    def start(self) -> None:
        raise OSError("RAW_THREAD_START_PRIVATE_DETAIL")


@pytest.mark.parametrize("settings", [{}, {"api": None}, {"api": ""}, {"api": "   "}])
def test_translate_endpoint_requires_explicit_translator(
    client: TestClient,
    settings: dict[str, object],
) -> None:
    response = client.post(
        "/api/translate",
        json={"paths": ["episode.srt"], "settings": settings},
    )

    assert response.status_code == 422
    assert response.json() == {"detail": web_app._REQUEST_VALIDATION_ERROR}


def test_translate_endpoint_starts_one_filtered_job(client: TestClient, monkeypatch) -> None:
    monkeypatch.setattr(web_app.threading, "Thread", FakeThread)

    response = client.post(
        "/api/translate",
        json={
            "paths": ["episode.srt", "notes.txt"],
            "settings": {"api": "google"},
        },
    )

    assert response.status_code == 200
    job_id = response.json()["job_id"]
    assert len(job_id) == 32
    job = web_app._get_job(job_id)
    assert job is not None
    assert job.paths == [Path("episode.srt")]
    assert isinstance(job.thread, FakeThread)
    assert job.thread.started is True
    assert job.thread.daemon is False
    assert isinstance(job.events, web_app.BoundedEventQueue)
    assert job.events.maxsize == web_app.MAX_EVENTS_PER_JOB
    assert web_app._active_job_id == job_id


def test_translate_thread_start_failure_rolls_back_registry(
    client: TestClient,
    monkeypatch,
) -> None:
    web_exception = Mock()
    monkeypatch.setattr(web_app.threading, "Thread", FailingStartThread)
    monkeypatch.setattr(web_app._WEB_LOGGER, "exception", web_exception)

    response = client.post(
        "/api/translate",
        json={"paths": ["episode.srt"], "settings": {"api": "google"}},
    )

    assert response.status_code == 503
    assert response.json() == {"detail": "Не удалось запустить задачу перевода."}
    assert "RAW_THREAD_START_PRIVATE_DETAIL" not in response.text
    assert "Traceback" not in response.text
    assert web_app._active_job_id is None
    with web_app._jobs_lock:
        assert web_app._jobs == {}
    web_exception.assert_called_once()


def test_translate_rejects_active_preparation_without_creating_job(
    client: TestClient,
    monkeypatch,
) -> None:
    monkeypatch.setattr(web_app.threading, "Thread", FakeThread)
    operation_id = web_app.preparation_registry.begin(
        "pick",
        phase="collecting",
        message="Идёт подготовка.",
    )
    try:
        response = client.post(
            "/api/translate",
            json={"paths": ["episode.srt"], "settings": {"api": "google"}},
        )
    finally:
        web_app.preparation_registry.finish(operation_id, message="Подготовка завершена.")

    assert response.status_code == 409
    assert response.json() == {"detail": web_app._PREPARATION_IN_PROGRESS_ERROR}
    assert web_app._active_job_id is None
    with web_app._jobs_lock:
        assert web_app._jobs == {}


def test_translate_rejects_occupied_picker_lock_without_waiting(
    client: TestClient,
) -> None:
    response_holder: list[httpx.Response] = []
    acquired = web_app._picker_refresh_lock.acquire(blocking=False)
    assert acquired is True

    worker = threading.Thread(
        target=lambda: response_holder.append(
            client.post(
                "/api/translate",
                json={"paths": ["episode.srt"], "settings": {"api": "google"}},
            )
        ),
    )
    try:
        worker.start()
        worker.join(timeout=1)
        completed_while_locked = not worker.is_alive()
    finally:
        web_app._picker_refresh_lock.release()
        worker.join(timeout=2)

    assert completed_while_locked is True
    assert response_holder[0].status_code == 409
    assert response_holder[0].json() == {"detail": web_app._PREPARATION_IN_PROGRESS_ERROR}
    assert web_app._active_job_id is None


def test_translate_rejects_concurrent_folder_pick(
    client: TestClient,
    monkeypatch,
) -> None:
    pick_entered = threading.Event()
    release_pick = threading.Event()
    pick_responses: list[httpx.Response] = []

    def blocking_pick(*_args, **_kwargs) -> web_app.PickSelection:
        pick_entered.set()
        assert release_pick.wait(timeout=2)
        return web_app.PickSelection(mode="folder", paths=())

    monkeypatch.setattr(web_app, "pick_paths", blocking_pick)
    worker = threading.Thread(
        target=lambda: pick_responses.append(
            client.post("/api/pick", json={"kind": "folder"}),
        ),
    )
    worker.start()
    assert pick_entered.wait(timeout=1)

    started_at = web_app.time.monotonic()
    translate_response = client.post(
        "/api/translate",
        json={"paths": ["episode.srt"], "settings": {"api": "google"}},
    )
    elapsed = web_app.time.monotonic() - started_at
    release_pick.set()
    worker.join(timeout=2)

    assert worker.is_alive() is False
    assert pick_responses[0].status_code == 200
    assert translate_response.status_code == 409
    assert elapsed < 1
    assert web_app._active_job_id is None


def test_translate_creation_prunes_oldest_terminal_job_at_limit_plus_one(
    client: TestClient,
    monkeypatch,
) -> None:
    monkeypatch.setattr(web_app.threading, "Thread", FakeThread)
    monkeypatch.setattr(web_app, "MAX_STORED_JOBS", 2)
    monkeypatch.setattr(web_app, "TERMINAL_JOB_TTL_SECONDS", 1_000)
    now = web_app.time.monotonic()
    oldest = make_job(job_id="1" * 32)
    oldest.completed = True
    oldest.completed_at = now - 20
    recent = make_job(job_id="2" * 32)
    recent.completed = True
    recent.completed_at = now - 10
    with web_app._jobs_lock:
        web_app._jobs[oldest.job_id] = oldest
        web_app._jobs[recent.job_id] = recent

    response = client.post(
        "/api/translate",
        json={"paths": ["episode.srt"], "settings": {"api": "nllb-600m"}},
    )

    assert response.status_code == 200
    new_job_id = response.json()["job_id"]
    with web_app._jobs_lock:
        assert len(web_app._jobs) == web_app.MAX_STORED_JOBS
        assert oldest.job_id not in web_app._jobs
        assert recent.job_id in web_app._jobs
        assert new_job_id in web_app._jobs


def test_job_access_removes_expired_terminal_jobs_but_keeps_active_jobs(monkeypatch) -> None:
    monkeypatch.setattr(web_app, "TERMINAL_JOB_TTL_SECONDS", 10)
    now = web_app.time.monotonic()
    expired = make_job(job_id="3" * 32)
    expired.completed = True
    expired.completed_at = now - 11
    fresh = make_job(job_id="4" * 32)
    fresh.completed = True
    fresh.completed_at = now - 9
    active = make_job(job_id="5" * 32)
    active.created_at = now - 100
    with web_app._jobs_lock:
        web_app._jobs[expired.job_id] = expired
        web_app._jobs[fresh.job_id] = fresh
        web_app._jobs[active.job_id] = active

    assert web_app._get_job(expired.job_id) is None
    assert web_app._get_job(fresh.job_id) is fresh
    assert web_app._get_job(active.job_id) is active


def test_translate_endpoint_rejects_empty_and_concurrent_jobs(client: TestClient) -> None:
    empty = client.post(
        "/api/translate",
        json={"paths": ["notes.txt"], "settings": {"api": "google"}},
    )
    with web_app._active_job_lock:
        web_app._active_job_id = "busy"
    concurrent = client.post(
        "/api/translate",
        json={"paths": ["episode.srt"], "settings": {"api": "google"}},
    )

    assert empty.status_code == 400
    assert concurrent.status_code == 409


def test_active_job_returns_snapshot(client: TestClient, monkeypatch) -> None:
    assert client.get("/api/active-job").json() == {"active": False}
    with web_app._active_job_lock:
        web_app._active_job_id = "missing"
    assert client.get("/api/active-job").json() == {"active": False}

    job = make_job(paths=[Path("one.srt"), Path("two.srt")])
    job.job_total = 2
    job.log_lines = ["строка"]
    job.file_states["one.srt"] = {"state": "done", "progress": 100}
    with web_app._jobs_lock:
        web_app._jobs[job.job_id] = job
    with web_app._active_job_lock:
        web_app._active_job_id = job.job_id
    monkeypatch.setattr(
        web_app,
        "_build_items",
        Mock(side_effect=AssertionError("snapshot must not build items")),
    )

    snapshot = client.get("/api/active-job").json()

    assert snapshot["active"] is True
    assert snapshot["done"] == 1
    assert snapshot["logs"] == ["строка"]
    assert snapshot["items"][0]["state"] == "done"
    assert snapshot["items"][1]["state"] == "queued"


def test_repeated_large_job_snapshots_use_only_memory(
    client: TestClient,
    monkeypatch,
) -> None:
    paths = [Path(f"folder-{index // 100}/subtitle-{index:05d}.srt") for index in range(10_000)]
    job = make_job(paths=paths)
    with web_app._jobs_lock:
        web_app._jobs[job.job_id] = job
    with web_app._active_job_lock:
        web_app._active_job_id = job.job_id
    build_items = Mock(side_effect=AssertionError("snapshot must not build items"))
    read_source = Mock(side_effect=AssertionError("snapshot must not read files"))
    monkeypatch.setattr(web_app, "_build_items", build_items)
    monkeypatch.setattr(web_app, "read_text", read_source)

    active = client.get("/api/active-job")
    first = client.get(f"/api/jobs/{job.job_id}")
    second = client.get(f"/api/jobs/{job.job_id}")

    assert active.status_code == 200
    assert first.status_code == 200
    assert second.status_code == 200
    assert len(first.json()["items"]) == 10_000
    assert first.json()["items"][0] == {
        "name": "subtitle-00000.srt",
        "path": str(paths[0]),
        "format": "srt",
        "state": "queued",
        "progress": 0,
    }
    assert second.json()["done"] == 0
    build_items.assert_not_called()
    read_source.assert_not_called()


def test_active_job_keeps_recorded_state_when_source_disappears(
    client: TestClient,
    tmp_path: Path,
) -> None:
    missing = tmp_path / "исчезнувший.srt"
    output = tmp_path / "исчезнувший.google.ru.srt"
    job = make_job(paths=[missing])
    job.job_total = 1
    job.file_states[str(missing)] = {
        "state": "done",
        "progress": 100,
        "output": str(output),
    }
    with web_app._jobs_lock:
        web_app._jobs[job.job_id] = job
    with web_app._active_job_lock:
        web_app._active_job_id = job.job_id

    snapshot = client.get("/api/active-job").json()
    item = snapshot["items"][0]

    assert snapshot["done"] == 1
    assert item["state"] == "done"
    assert item["output"] == str(output)
    assert "error" not in item


def test_completed_job_snapshot_survives_consumed_terminal_event(
    client: TestClient,
    tmp_path: Path,
) -> None:
    first = tmp_path / "первый файл.srt"
    second = tmp_path / "второй файл.srt"
    first.write_text(VALID_SRT, encoding="utf-8")
    second.write_text(VALID_SRT, encoding="utf-8")
    job = make_job(paths=[first, second])
    with web_app._jobs_lock:
        web_app._jobs[job.job_id] = job
    with web_app._active_job_lock:
        web_app._active_job_id = job.job_id
    web_app._emit_job_event(job, {"type": "job", "total": 2})
    web_app._emit_job_event(job, {"type": "log", "message": "Файл 1 готов."})
    web_app._emit_file_state(job, first, 1, 2, "done", 100, output=first.with_suffix(".ru.srt"))
    web_app._emit_file_state(job, second, 2, 2, "error", 100, error="Ошибка перевода.")
    web_app._finish_job(job, "partial")
    drain_events(job)

    response = client.get(f"/api/jobs/{job.job_id}")

    assert response.status_code == 200
    snapshot = response.json()
    assert snapshot["job_id"] == job.job_id
    assert snapshot["completed"] is True
    assert snapshot["status"] == "partial"
    assert snapshot["logs"] == ["Файл 1 готов."]
    assert snapshot["total"] == 2
    assert snapshot["done"] == 2
    assert [item["state"] for item in snapshot["items"]] == ["done", "error"]
    assert web_app._active_job_id is None


def test_job_snapshot_returns_safe_404_for_unknown_job(client: TestClient) -> None:
    response = client.get("/api/jobs/RAW_UNKNOWN_PRIVATE_JOB")

    assert response.status_code == 404
    assert response.json() == {"detail": "Задача не найдена."}
    assert "RAW_UNKNOWN_PRIVATE_JOB" not in response.text


def test_unload_endpoint_uses_shared_local_lifecycle(client: TestClient, monkeypatch) -> None:
    unload = Mock()
    monkeypatch.setattr(web_app, "unload_all_local_translators", unload)

    response = client.post("/api/unload", json={})

    assert response.json() == {"status": "ok", "message": "Модели выгружены."}
    unload.assert_called_once_with()


def test_unload_endpoint_rejects_active_job(client: TestClient, monkeypatch) -> None:
    unload = Mock()
    monkeypatch.setattr(web_app, "unload_all_local_translators", unload)
    with web_app._active_job_lock:
        web_app._active_job_id = "busy"

    response = client.post("/api/unload", json={})

    assert response.status_code == 409
    assert response.json() == {"detail": "Перевод уже выполняется."}
    unload.assert_not_called()


def test_unload_endpoint_returns_safe_error(client: TestClient, monkeypatch) -> None:
    monkeypatch.setattr(
        web_app,
        "unload_all_local_translators",
        Mock(side_effect=RuntimeError("внутренняя диагностика")),
    )
    monkeypatch.setattr(web_app._WEB_LOGGER, "error", Mock())

    response = client.post("/api/unload", json={})

    assert response.status_code == 500
    assert response.json() == {"detail": "Не удалось выгрузить локальную модель."}
    assert "внутренняя диагностика" not in response.text


class EventQueue:
    def __init__(self) -> None:
        self.calls = 0

    def get(self, timeout: float) -> dict[str, str]:
        assert timeout == 1.0
        self.calls += 1
        if self.calls == 1:
            raise queue.Empty
        return {"type": "done", "status": "ok"}


def test_stream_returns_keepalive_and_done_event(client: TestClient) -> None:
    missing = client.get("/api/stream/" + "f" * 32)
    assert missing.status_code == 404

    job = make_job()
    job.events = EventQueue()  # type: ignore[assignment]
    with web_app._jobs_lock:
        web_app._jobs[job.job_id] = job

    response = client.get(f"/api/stream/{job.job_id}")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert ": keep-alive" in response.text
    assert 'data: {"type": "done", "status": "ok"}' in response.text


def test_two_stream_consumers_receive_terminal_event(client: TestClient) -> None:
    job = make_job()
    with web_app._jobs_lock:
        web_app._jobs[job.job_id] = job
    web_app._finish_job(job, "error")

    first = client.get(f"/api/stream/{job.job_id}")
    second = client.get(f"/api/stream/{job.job_id}")

    expected = 'data: {"type": "done", "status": "error"}'
    assert first.status_code == 200
    assert second.status_code == 200
    assert expected in first.text
    assert expected in second.text
