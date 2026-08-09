from __future__ import annotations

import threading
from types import SimpleNamespace
from typing import Any

import pytest

from sub_translate.models import TranslationOptions
from sub_translate.translators import registry
from sub_translate.translators.lifecycle import LocalTranslatorLifecycleCoordinator


@pytest.fixture
def isolated_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
) -> LocalTranslatorLifecycleCoordinator:
    coordinator = LocalTranslatorLifecycleCoordinator()
    monkeypatch.setattr(registry, "_LOCAL_TRANSLATOR_LIFECYCLE", coordinator)
    return coordinator


def _install_fake_adapters(
    monkeypatch: pytest.MonkeyPatch,
    events: list[str],
    state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    behavior = state or {}

    class FakeNllbTranslator:
        name = "nllb-600m"

        def __init__(self) -> None:
            events.append("create:nllb")

        def translate_batch(self, texts: list[str], _options: TranslationOptions) -> list[str]:
            events.append("translate-enter:nllb")
            started = behavior.get("nllb_started")
            if started is not None:
                started.set()
            release = behavior.get("nllb_release")
            if release is not None:
                assert release.wait(timeout=2)
            if behavior.get("nllb_translate_error"):
                raise ValueError("исходная ошибка перевода")
            events.append("translate-exit:nllb")
            return [f"nllb:{text}" for text in texts]

        def unload(self) -> None:
            events.append("unload:nllb")
            if behavior.get("nllb_unload_error"):
                raise RuntimeError("ошибка выгрузки NLLB")

    class FakeTranslateGemmaTranslator:
        name = "translategemma"

        def __init__(self, *, timeout: int) -> None:
            events.append(f"create:translategemma:{timeout}")
            created = behavior.get("translategemma_created")
            if created is not None:
                created.set()

        def translate_batch(self, texts: list[str], _options: TranslationOptions) -> list[str]:
            events.append("translate:translategemma")
            return [f"translategemma:{text}" for text in texts]

        def unload(self) -> None:
            events.append("unload:translategemma")

    class FakeTranslateGemma12BTranslator:
        name = "translategemma-12b"

        def __init__(self, *, timeout: int) -> None:
            events.append(f"create:translategemma-12b:{timeout}")

        def translate_batch(self, texts: list[str], _options: TranslationOptions) -> list[str]:
            events.append("translate:translategemma-12b")
            return [f"translategemma-12b:{text}" for text in texts]

        def unload(self) -> None:
            events.append("unload:translategemma-12b")

    class FakeGoogleTranslator:
        name = "google"

        def __init__(self, *, timeout: int) -> None:
            events.append(f"create:google:{timeout}")

        @staticmethod
        def translate_batch(texts: list[str], _options: TranslationOptions) -> list[str]:
            return texts

        @staticmethod
        def unload() -> None:
            events.append("unload:google")

    class FakeAgentTranslator:
        name = "agent"

        def __init__(self) -> None:
            events.append("create:agent")

        @staticmethod
        def translate_batch(texts: list[str], _options: TranslationOptions) -> list[str]:
            return texts

        @staticmethod
        def unload() -> None:
            events.append("unload:agent")

    modules = {
        "sub_translate.translators.local.nllb": SimpleNamespace(
            Nllb600MTranslator=FakeNllbTranslator,
        ),
        "sub_translate.translators.local.translategemma": SimpleNamespace(
            TranslateGemmaTranslator=FakeTranslateGemmaTranslator,
            TranslateGemma12BTranslator=FakeTranslateGemma12BTranslator,
        ),
        "sub_translate.translators.google_web": SimpleNamespace(
            GoogleWebTranslator=FakeGoogleTranslator,
        ),
        "sub_translate.translators.agent": SimpleNamespace(
            AgentTranslator=FakeAgentTranslator,
        ),
    }
    monkeypatch.setattr(registry, "import_module", modules.__getitem__)
    return behavior


def test_switches_nllb_to_translategemma_and_back_without_redundant_unload(
    monkeypatch: pytest.MonkeyPatch,
    isolated_lifecycle: LocalTranslatorLifecycleCoordinator,
) -> None:
    events: list[str] = []
    _install_fake_adapters(monkeypatch, events)
    options = TranslationOptions()

    first_nllb = registry.create_translator("nllb-600m", timeout=10)
    assert first_nllb.translate_batch(["one"], options) == ["nllb:one"]
    registry.create_translator("nllb-600m", timeout=11)
    gemma = registry.create_translator("translategemma", timeout=12)
    assert gemma.translate_batch(["two"], options) == ["translategemma:two"]
    second_nllb = registry.create_translator("nllb-600m", timeout=13)
    assert second_nllb.translate_batch(["three"], options) == ["nllb:three"]

    assert events == [
        "create:nllb",
        "translate-enter:nllb",
        "translate-exit:nllb",
        "create:nllb",
        "unload:nllb",
        "create:translategemma:12",
        "translate:translategemma",
        "unload:translategemma",
        "create:nllb",
        "translate-enter:nllb",
        "translate-exit:nllb",
    ]
    assert isolated_lifecycle.active_profile_id == "nllb-600m"


def test_switches_between_translategemma_profiles_with_unload(
    monkeypatch: pytest.MonkeyPatch,
    isolated_lifecycle: LocalTranslatorLifecycleCoordinator,
) -> None:
    events: list[str] = []
    _install_fake_adapters(monkeypatch, events)
    options = TranslationOptions()

    profile_4b = registry.create_translator("translategemma", timeout=10)
    assert profile_4b.translate_batch(["four"], options) == ["translategemma:four"]
    profile_12b = registry.create_translator("translategemma-12b", timeout=20)
    assert profile_12b.translate_batch(["twelve"], options) == ["translategemma-12b:twelve"]

    assert events == [
        "create:translategemma:10",
        "translate:translategemma",
        "unload:translategemma",
        "create:translategemma-12b:20",
        "translate:translategemma-12b",
    ]
    assert isolated_lifecycle.active_profile_id == "translategemma-12b"


def test_network_translators_do_not_unload_active_local_profile(
    monkeypatch: pytest.MonkeyPatch,
    isolated_lifecycle: LocalTranslatorLifecycleCoordinator,
) -> None:
    events: list[str] = []
    _install_fake_adapters(monkeypatch, events)

    registry.create_translator("nllb-600m", timeout=10)
    registry.create_translator("google", timeout=11)
    registry.create_translator("agent", timeout=12)

    assert "unload:nllb" not in events
    assert isolated_lifecycle.active_profile_id == "nllb-600m"


def test_switch_waits_for_concurrent_local_translation(
    monkeypatch: pytest.MonkeyPatch,
    isolated_lifecycle: LocalTranslatorLifecycleCoordinator,
) -> None:
    events: list[str] = []
    started = threading.Event()
    release = threading.Event()
    gemma_created = threading.Event()
    _install_fake_adapters(
        monkeypatch,
        events,
        {
            "nllb_started": started,
            "nllb_release": release,
            "translategemma_created": gemma_created,
        },
    )
    translator = registry.create_translator("nllb-600m", timeout=10)
    errors: list[BaseException] = []

    def translate() -> None:
        try:
            translator.translate_batch(["one"], TranslationOptions())
        except BaseException as exc:  # pragma: no cover - защита диагностики потока
            errors.append(exc)

    def switch() -> None:
        try:
            registry.create_translator("translategemma", timeout=20)
        except BaseException as exc:  # pragma: no cover - защита диагностики потока
            errors.append(exc)

    translate_thread = threading.Thread(target=translate)
    switch_thread = threading.Thread(target=switch)
    translate_thread.start()
    assert started.wait(timeout=1)
    switch_thread.start()
    assert not gemma_created.wait(timeout=0.1)
    release.set()
    translate_thread.join(timeout=2)
    switch_thread.join(timeout=2)

    assert not translate_thread.is_alive()
    assert not switch_thread.is_alive()
    assert errors == []
    assert events.index("translate-exit:nllb") < events.index("unload:nllb")
    assert events.index("unload:nllb") < events.index("create:translategemma:20")
    assert isolated_lifecycle.active_profile_id == "translategemma"


def test_unload_failure_preserves_active_state_and_blocks_next_profile(
    monkeypatch: pytest.MonkeyPatch,
    isolated_lifecycle: LocalTranslatorLifecycleCoordinator,
) -> None:
    events: list[str] = []
    state = _install_fake_adapters(
        monkeypatch,
        events,
        {"nllb_unload_error": True},
    )
    registry.create_translator("nllb-600m", timeout=10)

    with pytest.raises(RuntimeError, match="ошибка выгрузки NLLB"):
        registry.create_translator("translategemma", timeout=20)

    assert "create:translategemma:20" not in events
    assert isolated_lifecycle.active_profile_id == "nllb-600m"
    state["nllb_unload_error"] = False
    registry.unload_all_local_translators()
    registry.unload_all_local_translators()
    assert isolated_lifecycle.active_profile_id is None
    assert events.count("unload:nllb") == 2


def test_translation_error_is_not_hidden_by_failing_unload(
    monkeypatch: pytest.MonkeyPatch,
    isolated_lifecycle: LocalTranslatorLifecycleCoordinator,
) -> None:
    events: list[str] = []
    state = _install_fake_adapters(
        monkeypatch,
        events,
        {
            "nllb_translate_error": True,
            "nllb_unload_error": True,
        },
    )
    translator = registry.create_translator("nllb-600m", timeout=10)
    options = TranslationOptions()

    with pytest.raises(ValueError, match="исходная ошибка перевода"):
        translator.translate_batch(["one"], options)

    assert "unload:nllb" not in events
    assert isolated_lifecycle.active_profile_id == "nllb-600m"
    state["nllb_unload_error"] = False
    registry.unload_all_local_translators()


def test_stale_wrapper_cannot_unload_current_profile(
    monkeypatch: pytest.MonkeyPatch,
    isolated_lifecycle: LocalTranslatorLifecycleCoordinator,
) -> None:
    events: list[str] = []
    _install_fake_adapters(monkeypatch, events)
    stale_nllb = registry.create_translator("nllb-600m", timeout=10)
    registry.create_translator("translategemma", timeout=20)

    stale_nllb.unload()

    assert events.count("unload:nllb") == 1
    assert "unload:translategemma" not in events
    assert isolated_lifecycle.active_profile_id == "translategemma"


def test_listing_metadata_does_not_import_local_adapters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_import(_module_name: str) -> None:
        raise AssertionError("Листинг не должен импортировать адаптеры.")

    monkeypatch.setattr(registry, "import_module", fail_import)

    assert {item.id for item in registry.list_translator_metadata()} >= {
        "nllb-600m",
        "translategemma",
    }
