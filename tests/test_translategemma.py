from __future__ import annotations

import sys
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from sub_translate.models import TranslationOptions
from sub_translate.translators.base import TranslationError
from sub_translate.translators.local import translategemma
from sub_translate.translators.local.translategemma import (
    TranslateGemma12BTranslator,
    TranslateGemmaTranslator,
)
from sub_translate.translators.registry import (
    TRANSLATEGEMMA_12B_MODEL_ID,
    TRANSLATEGEMMA_12B_MODEL_REVISION,
    TRANSLATEGEMMA_MODEL_ID,
    TRANSLATEGEMMA_MODEL_REVISION,
    TRANSLATEGEMMA_WORKER_REQUIREMENTS,
    get_translator_metadata,
)
from sub_translate.workers import translategemma_worker
from sub_translate.workers.external_worker import SHUTDOWN_TIMEOUT_SECONDS, ExternalWorkerError
from sub_translate.workers.translategemma_worker import (
    REQUIRED_RUNTIME_VERSIONS,
    TranslateGemmaWorkerRuntime,
    translation_messages,
)


@pytest.fixture(autouse=True)
def reset_shared_worker() -> None:
    TranslateGemmaTranslator._worker = None
    TranslateGemmaTranslator._worker_python = None
    TranslateGemmaTranslator._worker_profile_id = None
    yield
    worker = TranslateGemmaTranslator._worker
    TranslateGemmaTranslator._worker = None
    TranslateGemmaTranslator._worker_python = None
    TranslateGemmaTranslator._worker_profile_id = None
    if worker is not None:
        worker.abort()


def _model_directory(tmp_path: Path) -> Path:
    model_path = tmp_path / "translategemma"
    model_path.mkdir()
    for name in (
        "config.json",
        "model.safetensors",
        "preprocessor_config.json",
        "processor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ):
        (model_path / name).write_text("test\n", encoding="utf-8")
    return model_path


def _model_directory_12b(tmp_path: Path) -> Path:
    model_path = tmp_path / "translategemma-12b"
    model_path.mkdir()
    metadata = get_translator_metadata("translategemma-12b")
    for name in metadata.required_files:
        (model_path / name).write_text("test\n", encoding="utf-8")
    (model_path / "model.safetensors.index.json").write_text("{}\n", encoding="utf-8")
    return model_path


class _FakeWorker:
    def __init__(
        self,
        python_path: Path,
        module: str,
        *,
        timeout_seconds: float,
        response: dict[str, Any] | ExternalWorkerError,
    ) -> None:
        self.python_path = python_path
        self.module = module
        self.timeout_seconds = timeout_seconds
        self.response = response
        self.requests: list[tuple[str, dict[str, Any], float | None]] = []
        self.abort_calls = 0
        self.shutdown_calls = 0

    def request(
        self,
        command: str,
        payload: dict[str, Any],
        *,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        self.requests.append((command, payload, timeout_seconds))
        if isinstance(self.response, ExternalWorkerError):
            raise self.response
        if command == "unload":
            return {"unloaded": True}
        return self.response

    def abort(self) -> None:
        self.abort_calls += 1

    def shutdown(self) -> None:
        self.shutdown_calls += 1


def _install_worker_factory(
    monkeypatch: pytest.MonkeyPatch,
    responses: list[dict[str, Any] | ExternalWorkerError],
) -> list[_FakeWorker]:
    workers: list[_FakeWorker] = []

    def factory(
        python_path: Path,
        module: str,
        *,
        timeout_seconds: float,
    ) -> _FakeWorker:
        worker = _FakeWorker(
            python_path,
            module,
            timeout_seconds=timeout_seconds,
            response=responses[len(workers)],
        )
        workers.append(worker)
        return worker

    monkeypatch.setattr(translategemma, "PersistentNdjsonWorker", factory)
    return workers


def _patch_model_resolution(
    monkeypatch: pytest.MonkeyPatch,
    model_path: Path,
    revision: str = TRANSLATEGEMMA_MODEL_REVISION,
) -> None:
    monkeypatch.setattr(
        translategemma,
        "resolve_registered_model",
        lambda _identifier, _options: SimpleNamespace(
            path=model_path,
            revision=revision,
        ),
    )


def test_registry_declares_pinned_isolated_translategemma() -> None:
    metadata = get_translator_metadata("translate-gemma-4b")

    assert metadata.id == "translategemma"
    assert metadata.model_id == TRANSLATEGEMMA_MODEL_ID
    assert metadata.model_revision == TRANSLATEGEMMA_MODEL_REVISION
    assert metadata.runtime_kind == "isolated_worker"
    assert metadata.worker_requirements == TRANSLATEGEMMA_WORKER_REQUIREMENTS
    assert REQUIRED_RUNTIME_VERSIONS["torch"] == "2.13.0+cu130"
    assert REQUIRED_RUNTIME_VERSIONS["bitsandbytes"] == "0.50.0"


def test_runtime_import_uses_text_tokenizer(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_torch = ModuleType("torch")
    fake_bitsandbytes = ModuleType("bitsandbytes")
    fake_transformers = ModuleType("transformers")
    tokenizer_class = object()
    model_class = object()
    quantization_class = object()
    fake_transformers.AutoTokenizer = tokenizer_class
    fake_transformers.Gemma3ForConditionalGeneration = model_class
    fake_transformers.BitsAndBytesConfig = quantization_class
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "bitsandbytes", fake_bitsandbytes)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setattr(
        translategemma_worker.importlib.metadata,
        "version",
        lambda package: REQUIRED_RUNTIME_VERSIONS[package],
    )

    runtime = translategemma_worker._import_runtime()

    assert runtime == (
        fake_torch,
        tokenizer_class,
        model_class,
        quantization_class,
    )


@pytest.mark.parametrize(
    ("source", "target"),
    [("en", "ru"), ("ru", "en")],
)
def test_official_processor_message_is_preserved(source: str, target: str) -> None:
    assert translation_messages("Текст", source, target) == [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "source_lang_code": source,
                    "target_lang_code": target,
                    "text": "Текст",
                }
            ],
        }
    ]


def test_worker_uses_official_template_and_deterministic_generation() -> None:
    runtime = TranslateGemmaWorkerRuntime()
    tokenizer = MagicMock()
    input_ids = MagicMock()
    input_ids.shape = (1, 12)
    tokenizer.apply_chat_template.return_value = {
        "input_ids": input_ids,
        "attention_mask": MagicMock(),
    }
    tokenizer.decode.return_value = "Перевод"
    generated = MagicMock()
    model = MagicMock()
    model.generate.return_value = [generated]
    fake_torch = SimpleNamespace(
        inference_mode=lambda: nullcontext(),
        cuda=SimpleNamespace(OutOfMemoryError=MemoryError),
    )
    runtime._tokenizer = tokenizer
    runtime._model = model
    runtime._torch = fake_torch

    result = runtime._translate_chunk("Hello", "en", "ru", 1_024)

    assert result == "Перевод"
    template_call = tokenizer.apply_chat_template.call_args
    assert template_call.args[0] == translation_messages("Hello", "en", "ru")
    assert template_call.kwargs == {
        "tokenize": True,
        "add_generation_prompt": True,
        "return_dict": True,
        "return_tensors": "pt",
    }
    generation = model.generate.call_args.kwargs
    assert generation["max_new_tokens"] == 1_024
    assert generation["do_sample"] is False
    assert generation["num_beams"] == 1


def test_worker_loads_only_local_cuda_int8(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = _model_directory(tmp_path)
    runtime = TranslateGemmaWorkerRuntime()
    tokenizer = MagicMock()
    model = MagicMock()
    model.is_loaded_in_8bit = True
    model.hf_device_map = {"": 0}
    tokenizer_class = SimpleNamespace(from_pretrained=MagicMock(return_value=tokenizer))
    model_class = SimpleNamespace(from_pretrained=MagicMock(return_value=model))
    quantization_class = MagicMock(return_value="int8-config")
    fake_torch = SimpleNamespace(
        bfloat16="bfloat16",
        cuda=SimpleNamespace(
            OutOfMemoryError=MemoryError,
            is_available=lambda: True,
        ),
    )
    monkeypatch.setattr(
        runtime,
        "_runtime_modules",
        lambda: (fake_torch, tokenizer_class, model_class, quantization_class),
    )

    runtime._ensure_loaded(get_translator_metadata("translategemma"), model_path, TRANSLATEGEMMA_MODEL_REVISION)

    quantization_class.assert_called_once_with(
        load_in_8bit=True,
        llm_int8_enable_fp32_cpu_offload=False,
    )
    tokenizer_kwargs = tokenizer_class.from_pretrained.call_args.kwargs
    assert tokenizer_kwargs["local_files_only"] is True
    assert tokenizer_kwargs["trust_remote_code"] is False
    assert tokenizer_kwargs["fix_mistral_regex"] is False
    assert tokenizer_kwargs["use_fast"] is True
    model_kwargs = model_class.from_pretrained.call_args.kwargs
    assert model_kwargs["local_files_only"] is True
    assert model_kwargs["trust_remote_code"] is False
    assert model_kwargs["quantization_config"] == "int8-config"
    assert model_kwargs["device_map"] == {"": 0}
    assert model_kwargs["dtype"] == "bfloat16"


def test_worker_loads_12b_with_nf4_double_quantization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = _model_directory_12b(tmp_path)
    metadata = get_translator_metadata("translategemma-12b")
    runtime = TranslateGemmaWorkerRuntime()
    tokenizer = MagicMock()
    quantization_state = SimpleNamespace(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = MagicMock()
    model.is_loaded_in_4bit = True
    model.hf_quantizer = SimpleNamespace(quantization_config=quantization_state)
    model.hf_device_map = {"": 0}
    tokenizer_class = SimpleNamespace(from_pretrained=MagicMock(return_value=tokenizer))
    model_class = SimpleNamespace(from_pretrained=MagicMock(return_value=model))
    quantization_class = MagicMock(return_value="nf4-config")
    fake_torch = SimpleNamespace(
        bfloat16="bfloat16",
        cuda=SimpleNamespace(
            OutOfMemoryError=MemoryError,
            is_available=lambda: True,
        ),
    )
    monkeypatch.setattr(
        runtime,
        "_runtime_modules",
        lambda: (fake_torch, tokenizer_class, model_class, quantization_class),
    )

    runtime._ensure_loaded(metadata, model_path, TRANSLATEGEMMA_12B_MODEL_REVISION)

    quantization_class.assert_called_once_with(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype="bfloat16",
    )
    model_kwargs = model_class.from_pretrained.call_args.kwargs
    assert model_kwargs["quantization_config"] == "nf4-config"
    assert model_kwargs["device_map"] == {"": 0}
    assert model_kwargs["dtype"] == "bfloat16"
    assert "max_memory" not in model_kwargs
    assert "offload_folder" not in model_kwargs


@pytest.mark.parametrize("device", ["cpu", "disk"])
def test_12b_rejects_cpu_and_disk_placement(device: str) -> None:
    metadata = get_translator_metadata("translategemma-12b")
    quantization = SimpleNamespace(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = SimpleNamespace(
        is_loaded_in_4bit=True,
        hf_quantizer=SimpleNamespace(quantization_config=quantization),
        hf_device_map={"model.layers.0": device},
    )

    with pytest.raises(translategemma_worker.TranslateGemmaRuntimeError, match="вне CUDA"):
        translategemma_worker._require_cuda_quantization(model, metadata)


def test_12b_rejects_incomplete_nf4_configuration() -> None:
    metadata = get_translator_metadata("translategemma-12b")
    quantization = SimpleNamespace(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=False,
    )
    model = SimpleNamespace(
        is_loaded_in_4bit=True,
        hf_quantizer=SimpleNamespace(quantization_config=quantization),
        hf_device_map={"": 0},
    )

    with pytest.raises(translategemma_worker.TranslateGemmaRuntimeError, match="квантования"):
        translategemma_worker._require_cuda_quantization(model, metadata)


def test_worker_uses_12b_limits_and_processes_texts_sequentially(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = _model_directory_12b(tmp_path)
    runtime = TranslateGemmaWorkerRuntime()
    calls: list[tuple[str, int, int]] = []
    monkeypatch.setattr(
        TranslateGemmaWorkerRuntime,
        "_ensure_loaded",
        lambda self, _metadata, _path, _revision: None,
    )

    def translate_text(
        _self: TranslateGemmaWorkerRuntime,
        text: str,
        _source: str,
        _target: str,
        input_limit: int,
        output_limit: int,
    ) -> str:
        calls.append((text, input_limit, output_limit))
        return f"ru:{text}"

    monkeypatch.setattr(TranslateGemmaWorkerRuntime, "_translate_text", translate_text)
    result = runtime.translate(
        {
            "profile_id": "translategemma-12b",
            "model_id": TRANSLATEGEMMA_12B_MODEL_ID,
            "model_revision": TRANSLATEGEMMA_12B_MODEL_REVISION,
            "model_path": str(model_path),
            "source_lang": "en",
            "target_lang": "ru",
            "texts": ["one", "two"],
        }
    )

    assert result["translations"] == ["ru:one", "ru:two"]
    assert calls == [("one", 512, 512), ("two", 512, 512)]


def test_worker_preserves_batch_cardinality(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = _model_directory(tmp_path)
    runtime = TranslateGemmaWorkerRuntime()
    monkeypatch.setattr(
        TranslateGemmaWorkerRuntime,
        "_ensure_loaded",
        lambda self, _metadata, _path, _revision: None,
    )
    monkeypatch.setattr(
        TranslateGemmaWorkerRuntime,
        "_translate_text",
        lambda self, text, source, target, _input_limit, _output_limit: f"{source}-{target}:{text}",
    )
    payload = {
        "profile_id": "translategemma",
        "model_id": TRANSLATEGEMMA_MODEL_ID,
        "model_revision": TRANSLATEGEMMA_MODEL_REVISION,
        "model_path": str(model_path),
        "source_lang": "en",
        "target_lang": "ru",
        "texts": ["one", "", "three"],
    }

    result = runtime.translate(payload)

    assert result["translations"] == ["en-ru:one", "", "en-ru:three"]
    assert len(result["translations"]) == len(payload["texts"])


def test_adapter_sends_parent_resolved_model_and_worker_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = _model_directory(tmp_path)
    python_path = tmp_path / "runtime" / "python.exe"
    python_path.parent.mkdir()
    python_path.write_bytes(b"")
    workers = _install_worker_factory(
        monkeypatch,
        [{"translations": ["Первый", "Второй"]}],
    )
    _patch_model_resolution(monkeypatch, model_path)
    translator = TranslateGemmaTranslator(timeout=77)
    options = TranslationOptions(
        source_lang="English",
        target_lang="Russian",
        worker_python_path=python_path,
    )

    result = translator.translate_batch(["First", "Second"], options)

    assert result == ["Первый", "Второй"]
    worker = workers[0]
    assert worker.python_path == python_path.resolve()
    assert worker.module == "sub_translate.workers.translategemma_worker"
    command, payload, request_timeout = worker.requests[0]
    assert command == "translate"
    assert request_timeout == 77
    assert payload["profile_id"] == "translategemma"
    assert payload["model_id"] == TRANSLATEGEMMA_MODEL_ID
    assert payload["model_revision"] == TRANSLATEGEMMA_MODEL_REVISION
    assert payload["model_path"] == str(model_path)
    assert payload["texts"] == ["First", "Second"]


def test_12b_adapter_sends_own_canonical_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = _model_directory_12b(tmp_path)
    workers = _install_worker_factory(monkeypatch, [{"translations": ["Перевод"]}])
    _patch_model_resolution(monkeypatch, model_path, revision=TRANSLATEGEMMA_12B_MODEL_REVISION)
    translator = TranslateGemma12BTranslator(timeout=90)

    assert translator.translate_batch(["Translation"], TranslationOptions(source_lang="en", target_lang="ru")) == [
        "Перевод"
    ]
    command, payload, timeout = workers[0].requests[0]
    assert command == "translate"
    assert timeout == 90
    assert payload["profile_id"] == "translategemma-12b"
    assert payload["model_id"] == TRANSLATEGEMMA_12B_MODEL_ID
    assert payload["model_revision"] == TRANSLATEGEMMA_12B_MODEL_REVISION


def test_switching_4b_and_12b_restarts_shared_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_4b = _model_directory(tmp_path)
    model_12b = _model_directory_12b(tmp_path)
    workers = _install_worker_factory(
        monkeypatch,
        [
            {"translations": ["Четыре"]},
            {"translations": ["Двенадцать"]},
        ],
    )

    def resolve_model(identifier: str, _options: TranslationOptions) -> SimpleNamespace:
        metadata = get_translator_metadata(identifier)
        path = model_4b if identifier == "translategemma" else model_12b
        return SimpleNamespace(path=path, revision=metadata.model_revision)

    monkeypatch.setattr(translategemma, "resolve_registered_model", resolve_model)
    options = TranslationOptions(source_lang="en", target_lang="ru")

    assert TranslateGemmaTranslator().translate_batch(["Four"], options) == ["Четыре"]
    assert TranslateGemma12BTranslator().translate_batch(["Twelve"], options) == ["Двенадцать"]
    TranslateGemma12BTranslator.unload()

    assert len(workers) == 2
    assert workers[0].abort_calls == 1
    assert workers[1].shutdown_calls == 1
    assert TranslateGemmaTranslator._worker is None


def test_different_instances_share_one_worker_and_class_unload_stops_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = _model_directory(tmp_path)
    workers = _install_worker_factory(monkeypatch, [{"translations": ["Готово"]}])
    _patch_model_resolution(monkeypatch, model_path)
    first = TranslateGemmaTranslator(timeout=10)
    second = TranslateGemmaTranslator(timeout=20)
    options = TranslationOptions(source_lang="en", target_lang="ru")

    assert first.translate_batch(["Done"], options) == ["Готово"]
    assert second.translate_batch(["Done"], options) == ["Готово"]
    second.unload()

    assert len(workers) == 1
    commands = [command for command, _payload, _timeout in workers[0].requests]
    assert commands == ["translate", "translate", "unload"]
    assert [timeout for _command, _payload, timeout in workers[0].requests[:2]] == [10, 20]
    assert workers[0].requests[-1][2] == SHUTDOWN_TIMEOUT_SECONDS
    assert workers[0].shutdown_calls == 1
    assert TranslateGemmaTranslator._worker is None


@pytest.mark.parametrize(
    "response",
    [
        {},
        {"translations": ["только один"]},
        {"translations": ["", "два"]},
        {"translations": [1, "два"]},
    ],
)
def test_adapter_rejects_malformed_response_and_discards_worker(
    response: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = _model_directory(tmp_path)
    workers = _install_worker_factory(monkeypatch, [response])
    _patch_model_resolution(monkeypatch, model_path)
    translator = TranslateGemmaTranslator()
    options = TranslationOptions(source_lang="en", target_lang="ru")

    with pytest.raises(TranslationError):
        translator.translate_batch(["one", "two"], options)

    assert workers[0].abort_calls == 1
    assert TranslateGemmaTranslator._worker is None


@pytest.mark.parametrize(
    ("error_type", "message"),
    [
        ("TranslateGemmaOutOfMemoryError", "видеопамяти"),
        ("TimeoutError", "таймаут"),
        ("ProtocolError", "протокол"),
        ("WorkerExited", "завершился"),
    ],
)
def test_worker_failure_is_wrapped_and_next_call_restarts(
    error_type: str,
    message: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = _model_directory(tmp_path)
    workers = _install_worker_factory(
        monkeypatch,
        [
            ExternalWorkerError("внутренняя ошибка", error_type=error_type),
            {"translations": ["Успех"]},
        ],
    )
    _patch_model_resolution(monkeypatch, model_path)
    translator = TranslateGemmaTranslator()
    options = TranslationOptions(source_lang="en", target_lang="ru")

    with pytest.raises(TranslationError, match=message):
        translator.translate_batch(["First"], options)
    result = translator.translate_batch(["First"], options)

    assert workers[0].abort_calls == 1
    assert result == ["Успех"]
    assert len(workers) == 2


@pytest.mark.parametrize(
    ("source_lang", "target_lang"),
    [("en", "en"), ("ru", "ru"), ("en", "de")],
)
def test_unsupported_direction_is_rejected_before_worker(
    source_lang: str,
    target_lang: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workers = _install_worker_factory(monkeypatch, [])
    translator = TranslateGemmaTranslator()
    options = TranslationOptions(source_lang=source_lang, target_lang=target_lang)

    with pytest.raises(TranslationError, match="TranslateGemma"):
        translator.translate_batch(["Text"], options)

    assert workers == []


def test_cpu_fallback_and_unpinned_revision_are_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = _model_directory(tmp_path)
    translator = TranslateGemmaTranslator()
    cpu_options = TranslationOptions(allow_cpu_fallback=True)
    with pytest.raises(TranslationError, match="CPU"):
        translator.translate_batch(["Text"], cpu_options)

    _patch_model_resolution(monkeypatch, model_path, revision="0" * 40)
    default_options = TranslationOptions()
    with pytest.raises(TranslationError, match="закреплённой ревизией"):
        translator.translate_batch(["Text"], default_options)


def test_generation_config_is_normalized_for_greedy_decoding() -> None:
    config = SimpleNamespace(
        do_sample=True,
        num_beams=4,
        top_p=0.95,
        top_k=64,
        max_new_tokens=32,
        max_length=20,
    )
    model = SimpleNamespace(generation_config=config)

    translategemma_worker._sanitize_generation_config(model)

    assert config.do_sample is False
    assert config.num_beams == 1
    assert config.top_p is None
    assert config.top_k is None
    assert config.max_new_tokens is None
