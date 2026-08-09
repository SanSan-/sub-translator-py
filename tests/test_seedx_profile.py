from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from sub_translate.models import TranslationOptions
from sub_translate.translators.base import TranslationError
from sub_translate.translators.local import seedx
from sub_translate.translators.local.seedx import SeedXTranslator
from sub_translate.translators.registry import SEEDX_MODEL_ID, SEEDX_MODEL_REVISION
from sub_translate.workers.external_worker import SHUTDOWN_TIMEOUT_SECONDS, ExternalWorkerError
from sub_translate.workers.seedx_worker import (
    MAX_BATCH_SIZE,
    MAX_NEW_TOKENS,
    NUM_BEAMS,
    SeedXRuntimeError,
    SeedXWorkerRuntime,
    _require_compressed_int4,
    _validate_quantization_config,
    build_translation_prompt,
)


@pytest.fixture(autouse=True)
def reset_seedx_worker() -> Any:
    SeedXTranslator._worker = None
    SeedXTranslator._worker_python = None
    yield
    worker = SeedXTranslator._worker
    SeedXTranslator._worker = None
    SeedXTranslator._worker_python = None
    if worker is not None:
        worker.abort()


def _model_directory(tmp_path: Path) -> Path:
    model_path = tmp_path / "seed-x"
    model_path.mkdir()
    config = (
        '{"architectures":["MistralForCausalLM"],"model_type":"mistral",'
        '"torch_dtype":"bfloat16","quantization_config":{'
        '"quant_method":"compressed-tensors","format":"pack-quantized",'
        '"quantization_status":"compressed","config_groups":{"group_0":{'
        '"weights":{"num_bits":4,"group_size":128,"type":"int"}}}}}'
    )
    (model_path / "config.json").write_text(config, encoding="utf-8")
    for name in (
        "generation_config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
    ):
        (model_path / name).write_text("test\n", encoding="utf-8")
    return model_path


class _FakeWorker:
    def __init__(
        self,
        python_path: Path,
        module: str,
        *,
        timeout_seconds: float,
        responses: list[dict[str, Any] | ExternalWorkerError],
    ) -> None:
        self.python_path = python_path
        self.module = module
        self.timeout_seconds = timeout_seconds
        self.responses = list(responses)
        self.requests: list[tuple[str, dict[str, Any], float | None]] = []
        self.abort_calls = 0
        self.shutdown_calls = 0

    def request(
        self,
        command: str,
        payload: dict[str, Any] | None = None,
        *,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        self.requests.append((command, dict(payload or {}), timeout_seconds))
        if command == "unload":
            return {"unloaded": True}
        response = self.responses.pop(0)
        if isinstance(response, ExternalWorkerError):
            raise response
        return response

    def abort(self) -> None:
        self.abort_calls += 1

    def shutdown(self) -> None:
        self.shutdown_calls += 1


def _install_worker_factory(
    monkeypatch: pytest.MonkeyPatch,
    worker_responses: list[list[dict[str, Any] | ExternalWorkerError]],
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
            responses=worker_responses[len(workers)],
        )
        workers.append(worker)
        return worker

    monkeypatch.setattr(seedx, "PersistentNdjsonWorker", factory)
    return workers


def _patch_model_resolution(
    monkeypatch: pytest.MonkeyPatch,
    model_path: Path,
    revision: str = SEEDX_MODEL_REVISION,
) -> None:
    monkeypatch.setattr(
        seedx,
        "resolve_registered_model",
        lambda _identifier, _options: SimpleNamespace(path=model_path, revision=revision),
    )


@pytest.mark.parametrize(
    ("source", "target", "text", "expected"),
    [
        (
            "en",
            "ru",
            "The child found the missing key under the old wooden table.",
            "Translate the following English sentence into Russian:\n"
            "The child found the missing key under the old wooden table. <ru>",
        ),
        (
            "ru",
            "en",
            "Ребёнок нашёл пропавший ключ.",
            "Translate the following Russian sentence into English:\nРебёнок нашёл пропавший ключ. <en>",
        ),
    ],
)
def test_prompt_matches_official_single_turn_format(
    source: str,
    target: str,
    text: str,
    expected: str,
) -> None:
    prompt = build_translation_prompt(text, source, target)

    assert prompt == expected
    assert "chat" not in prompt.casefold()


def test_adapter_sends_sequential_single_item_requests_and_reuses_worker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    python_path = tmp_path / "python.exe"
    python_path.write_bytes(b"")
    model_path = _model_directory(tmp_path)
    workers = _install_worker_factory(
        monkeypatch,
        [[{"translations": ["Первый"]}, {"translations": ["Второй"]}, {"translations": ["Третий"]}]],
    )
    _patch_model_resolution(monkeypatch, model_path)
    options = TranslationOptions(
        source_lang="English",
        target_lang="Russian",
        worker_python_path=python_path,
    )

    first = SeedXTranslator(timeout=4_321).translate_batch(["First", "Second"], options)
    second = SeedXTranslator(timeout=4_321).translate_batch(["Third"], options)

    assert first == ["Первый", "Второй"]
    assert second == ["Третий"]
    assert len(workers) == 1
    worker = workers[0]
    assert worker.python_path == python_path.resolve()
    assert worker.module == "sub_translate.workers.seedx_worker"
    assert [request[1]["texts"] for request in worker.requests] == [["First"], ["Second"], ["Third"]]
    assert all(request[2] == 4_321 for request in worker.requests)
    assert all(request[1]["model_id"] == SEEDX_MODEL_ID for request in worker.requests)
    assert all(request[1]["model_revision"] == SEEDX_MODEL_REVISION for request in worker.requests)
    assert all(request[1]["model_path"] == str(model_path) for request in worker.requests)


@pytest.mark.parametrize(
    ("error_type", "message"),
    [
        ("SeedXOutOfMemoryError", "видеопамяти"),
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
            [ExternalWorkerError("внутренняя ошибка", error_type=error_type)],
            [{"translations": ["Успех"]}],
        ],
    )
    _patch_model_resolution(monkeypatch, model_path)
    translator = SeedXTranslator()
    options = TranslationOptions(source_lang="en", target_lang="ru")

    with pytest.raises(TranslationError, match=message):
        translator.translate_batch(["First"], options)
    result = translator.translate_batch(["First"], options)

    assert workers[0].abort_calls == 1
    assert result == ["Успех"]
    assert len(workers) == 2


def test_malformed_response_discards_worker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model_path = _model_directory(tmp_path)
    workers = _install_worker_factory(monkeypatch, [[{"translations": []}]])
    _patch_model_resolution(monkeypatch, model_path)

    with pytest.raises(TranslationError, match="другое число"):
        SeedXTranslator().translate_batch(
            ["Text"],
            TranslationOptions(source_lang="en", target_lang="ru"),
        )

    assert workers[0].abort_calls == 1
    assert SeedXTranslator._worker is None


def test_unload_releases_model_and_stops_worker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model_path = _model_directory(tmp_path)
    workers = _install_worker_factory(monkeypatch, [[{"translations": ["Готово"]}]])
    _patch_model_resolution(monkeypatch, model_path)
    translator = SeedXTranslator()
    translator.translate_batch(
        ["Done"],
        TranslationOptions(source_lang="en", target_lang="ru"),
    )

    translator.unload()

    worker = workers[0]
    assert [command for command, _payload, _timeout in worker.requests] == ["translate", "unload"]
    assert worker.requests[-1][2] == SHUTDOWN_TIMEOUT_SECONDS
    assert worker.shutdown_calls == 1
    assert SeedXTranslator._worker is None


def test_cpu_fallback_and_unpinned_revision_are_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = _model_directory(tmp_path)
    translator = SeedXTranslator()
    with pytest.raises(TranslationError, match="CPU"):
        translator.translate_batch(["Text"], TranslationOptions(allow_cpu_fallback=True))

    _patch_model_resolution(monkeypatch, model_path, revision="0" * 40)
    with pytest.raises(TranslationError, match="закреплённой ревизией"):
        translator.translate_batch(["Text"], TranslationOptions())


def test_worker_enforces_single_item_batch_and_compressed_int4(tmp_path: Path) -> None:
    model_path = _model_directory(tmp_path)
    _validate_quantization_config(model_path)
    runtime = SeedXWorkerRuntime()
    payload = {
        "model_id": SEEDX_MODEL_ID,
        "model_path": str(model_path),
        "model_revision": SEEDX_MODEL_REVISION,
        "source_lang": "en",
        "target_lang": "ru",
        "texts": ["one", "two"],
    }

    assert MAX_BATCH_SIZE == 1
    with pytest.raises(SeedXRuntimeError, match="пределе 1"):
        runtime.translate(payload)
    _require_compressed_int4(
        SimpleNamespace(
            hf_quantizer=SimpleNamespace(
                quantization_config=SimpleNamespace(run_compressed=True),
            )
        )
    )
    with pytest.raises(SeedXRuntimeError, match="4-битного"):
        _require_compressed_int4(SimpleNamespace(hf_quantizer=None))

    config_path = model_path / "config.json"
    config_path.write_text(
        config_path.read_text(encoding="utf-8").replace("bfloat16", "float16"),
        encoding="utf-8",
    )
    with pytest.raises(SeedXRuntimeError, match="BF16"):
        _validate_quantization_config(model_path)


def test_loading_preserves_bfloat16_and_compressed_cuda_placement(tmp_path: Path) -> None:
    model_path = _model_directory(tmp_path)
    bfloat16 = object()
    float16 = object()
    load_kwargs: dict[str, Any] = {}

    class FakeTokenizerClass:
        @staticmethod
        def from_pretrained(_path: str, **_kwargs: Any) -> object:
            return object()

    class FakeModel:
        def __init__(self) -> None:
            self.hf_quantizer = SimpleNamespace(
                quantization_config=SimpleNamespace(run_compressed=True),
            )
            self.hf_device_map = {"": "cuda:0"}

        @staticmethod
        def eval() -> None:
            return None

    class FakeModelClass:
        @staticmethod
        def from_pretrained(_path: str, **kwargs: Any) -> FakeModel:
            load_kwargs.update(kwargs)
            return FakeModel()

    fake_torch = SimpleNamespace(
        bfloat16=bfloat16,
        float16=float16,
        cuda=SimpleNamespace(
            OutOfMemoryError=MemoryError,
            is_available=lambda: True,
            is_bf16_supported=lambda: True,
        ),
    )
    runtime = SeedXWorkerRuntime()
    runtime._modules = (fake_torch, FakeTokenizerClass, FakeModelClass)

    runtime._ensure_loaded(model_path, SEEDX_MODEL_REVISION)

    assert load_kwargs["dtype"] is bfloat16
    assert load_kwargs["dtype"] is not float16
    assert load_kwargs["device_map"] == {"": "cuda:0"}
    assert load_kwargs["local_files_only"] is True
    assert load_kwargs["use_safetensors"] is True


def test_generation_uses_quality_oriented_beam_search() -> None:
    class FakeTensor:
        shape = (1, 2)

        def to(self, _device: str) -> FakeTensor:
            return self

    class FakeTokenizer:
        eos_token_id = 2

        def __init__(self) -> None:
            self.decoded_tokens: list[int] | None = None

        def __call__(self, _prompt: str, **_kwargs: Any) -> dict[str, FakeTensor]:
            return {"input_ids": FakeTensor()}

        def decode(self, tokens: Any, **_kwargs: Any) -> str:
            self.decoded_tokens = list(tokens)
            return "Перевод"

    class FakeModel:
        def __init__(self) -> None:
            self.kwargs: dict[str, Any] = {}

        def generate(self, **kwargs: Any) -> list[list[int]]:
            self.kwargs = kwargs
            return [[1, 2, 3, 2]]

    model = FakeModel()
    tokenizer = FakeTokenizer()
    runtime = SeedXWorkerRuntime()
    runtime._tokenizer = tokenizer
    runtime._model = model
    runtime._torch = SimpleNamespace(
        inference_mode=nullcontext,
        cuda=SimpleNamespace(OutOfMemoryError=MemoryError),
    )

    result = runtime._translate_one("Text", "en", "ru")

    assert result == "Перевод"
    assert NUM_BEAMS == 4
    assert model.kwargs["num_beams"] == 4
    assert model.kwargs["do_sample"] is False
    assert model.kwargs["eos_token_id"] == 2
    assert model.kwargs["pad_token_id"] == 2
    assert model.kwargs["max_new_tokens"] == MAX_NEW_TOKENS
    assert tokenizer.decoded_tokens == [3, 2]


def test_generation_without_eos_is_rejected_before_decode() -> None:
    class FakeTensor:
        shape = (1, 2)

        def to(self, _device: str) -> FakeTensor:
            return self

    class FakeTokenizer:
        eos_token_id = 2

        def __call__(self, _prompt: str, **_kwargs: Any) -> dict[str, FakeTensor]:
            return {"input_ids": FakeTensor()}

        @staticmethod
        def decode(_tokens: Any, **_kwargs: Any) -> str:
            raise AssertionError("Незавершённую генерацию нельзя декодировать.")

    class FakeModel:
        @staticmethod
        def generate(**kwargs: Any) -> list[list[int]]:
            return [[1, 2, *([3] * int(kwargs["max_new_tokens"]))]]

    runtime = SeedXWorkerRuntime()
    runtime._tokenizer = FakeTokenizer()
    runtime._model = FakeModel()
    runtime._torch = SimpleNamespace(
        inference_mode=nullcontext,
        cuda=SimpleNamespace(OutOfMemoryError=MemoryError),
    )

    with pytest.raises(SeedXRuntimeError, match="EOS"):
        runtime._translate_one("Text", "en", "ru")
