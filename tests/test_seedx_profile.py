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
from sub_translate.workers.external_worker import ExternalWorkerError
from sub_translate.workers.seedx_worker import (
    MAX_BATCH_SIZE,
    MAX_NEW_TOKENS,
    NO_REPEAT_NGRAM_SIZE,
    NUM_BEAMS,
    SeedXRuntimeError,
    SeedXWorkerRuntime,
    _configure_eos_token,
    _require_cuda_only,
    _require_nf4_quantization,
    _sanitize_translation_result,
    _validate_base_model_config,
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
        '"torch_dtype":"bfloat16","hidden_size":4096,"num_hidden_layers":32,'
        '"vocab_size":65269,"eos_token_id":2}'
    )
    (model_path / "config.json").write_text(config, encoding="utf-8")
    for name in (
        "generation_config.json",
        "model.safetensors",
        "tokenizer.json",
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
    content_fingerprint: str | None = "b" * 64,
) -> None:
    monkeypatch.setattr(
        seedx,
        "resolve_registered_model",
        lambda _identifier, _options: SimpleNamespace(
            path=model_path,
            revision=revision,
            content_fingerprint=content_fingerprint,
        ),
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
    assert all(request[1]["model_content_fingerprint"] == "b" * 64 for request in worker.requests)


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
    translator = SeedXTranslator()
    options = TranslationOptions(source_lang="en", target_lang="ru")

    with pytest.raises(TranslationError, match="другое число"):
        translator.translate_batch(["Text"], options)

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
    assert [command for command, _payload, _timeout in worker.requests] == ["translate"]
    assert worker.shutdown_calls == 1
    assert SeedXTranslator._worker is None


def test_cpu_fallback_flag_is_ignored_for_cuda_only_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = _model_directory(tmp_path)
    workers = _install_worker_factory(monkeypatch, [[{"translations": ["Перевод"]}]])
    _patch_model_resolution(monkeypatch, model_path)

    result = SeedXTranslator().translate_batch(
        ["Text"],
        TranslationOptions(source_lang="en", target_lang="ru", allow_cpu_fallback=True),
    )

    assert result == ["Перевод"]
    assert len(workers) == 1


def test_unpinned_revision_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = _model_directory(tmp_path)
    translator = SeedXTranslator()
    options = TranslationOptions()

    _patch_model_resolution(monkeypatch, model_path, revision="0" * 40)
    with pytest.raises(TranslationError, match="закреплённой ревизией"):
        translator.translate_batch(["Text"], options)


def test_worker_enforces_single_item_batch_and_base_model_contract(tmp_path: Path) -> None:
    model_path = _model_directory(tmp_path)
    _validate_base_model_config(model_path)
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
    _require_nf4_quantization(
        SimpleNamespace(
            is_loaded_in_4bit=True,
            hf_quantizer=SimpleNamespace(
                quantization_config=SimpleNamespace(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype="bfloat16",
                ),
            ),
        )
    )
    unquantized_model = SimpleNamespace(hf_quantizer=None)
    with pytest.raises(SeedXRuntimeError, match="NF4"):
        _require_nf4_quantization(unquantized_model)

    config_path = model_path / "config.json"
    config_path.write_text(
        config_path.read_text(encoding="utf-8").replace("bfloat16", "float16"),
        encoding="utf-8",
    )
    with pytest.raises(SeedXRuntimeError, match="BF16"):
        _validate_base_model_config(model_path)


def test_loading_preserves_bfloat16_nf4_and_cuda_placement(tmp_path: Path) -> None:
    model_path = _model_directory(tmp_path)
    bfloat16 = "bfloat16"
    float16 = object()
    load_kwargs: dict[str, Any] = {}

    class FakeTokenizerClass:
        @staticmethod
        def from_pretrained(_path: str, **_kwargs: Any) -> object:
            return SimpleNamespace(eos_token_id=2)

    class FakeQuantizationConfig:
        def __init__(self, **kwargs: Any) -> None:
            vars(self).update(kwargs)

    class FakeModel:
        def __init__(self) -> None:
            self.is_loaded_in_4bit = True
            self.hf_quantizer: Any = None
            self.hf_device_map = {"": "cuda:0"}
            self.config = SimpleNamespace(eos_token_id=2)

        @staticmethod
        def eval() -> None:
            return None

        @staticmethod
        def parameters() -> list[SimpleNamespace]:
            return [SimpleNamespace(device="cuda:0")]

        @staticmethod
        def buffers() -> list[SimpleNamespace]:
            return []

    class FakeModelClass:
        @staticmethod
        def from_pretrained(_path: str, **kwargs: Any) -> FakeModel:
            load_kwargs.update(kwargs)
            model = FakeModel()
            model.hf_quantizer = SimpleNamespace(quantization_config=kwargs["quantization_config"])
            return model

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
    runtime._modules = (fake_torch, FakeTokenizerClass, FakeModelClass, FakeQuantizationConfig)

    runtime._ensure_loaded(model_path, SEEDX_MODEL_REVISION)

    assert load_kwargs["dtype"] is bfloat16
    assert load_kwargs["dtype"] is not float16
    assert load_kwargs["device_map"] == {"": 0}
    quantization = load_kwargs["quantization_config"]
    assert quantization.load_in_4bit is True
    assert quantization.bnb_4bit_quant_type == "nf4"
    assert quantization.bnb_4bit_use_double_quant is True
    assert quantization.bnb_4bit_compute_dtype is bfloat16
    assert load_kwargs["local_files_only"] is True
    assert load_kwargs["use_safetensors"] is True


def test_worker_reloads_when_model_content_fingerprint_changes(tmp_path: Path) -> None:
    model_path = _model_directory(tmp_path)
    model_loads = 0

    class FakeTokenizerClass:
        @staticmethod
        def from_pretrained(_path: str, **_kwargs: Any) -> object:
            return SimpleNamespace(eos_token_id=2)

    class FakeQuantizationConfig:
        def __init__(self, **kwargs: Any) -> None:
            vars(self).update(kwargs)

    class FakeModel:
        def __init__(self) -> None:
            self.is_loaded_in_4bit = True
            self.hf_quantizer: Any = None
            self.hf_device_map = {"": "cuda:0"}
            self.config = SimpleNamespace(eos_token_id=2)

        @staticmethod
        def eval() -> None:
            return None

        @staticmethod
        def parameters() -> list[SimpleNamespace]:
            return [SimpleNamespace(device="cuda:0")]

        @staticmethod
        def buffers() -> list[SimpleNamespace]:
            return []

    class FakeModelClass:
        @staticmethod
        def from_pretrained(_path: str, **_kwargs: Any) -> FakeModel:
            nonlocal model_loads
            model_loads += 1
            model = FakeModel()
            model.hf_quantizer = SimpleNamespace(quantization_config=_kwargs["quantization_config"])
            return model

    fake_torch = SimpleNamespace(
        bfloat16="bfloat16",
        cuda=SimpleNamespace(
            OutOfMemoryError=MemoryError,
            is_available=lambda: True,
            is_bf16_supported=lambda: True,
            empty_cache=lambda: None,
            ipc_collect=lambda: None,
        ),
    )
    runtime = SeedXWorkerRuntime()
    runtime._modules = (fake_torch, FakeTokenizerClass, FakeModelClass, FakeQuantizationConfig)

    runtime._ensure_loaded(model_path, SEEDX_MODEL_REVISION, "a" * 64)
    runtime._ensure_loaded(model_path, SEEDX_MODEL_REVISION, "a" * 64)
    runtime._ensure_loaded(model_path, SEEDX_MODEL_REVISION, "b" * 64)

    assert model_loads == 2
    assert runtime._load_key[-1] == "b" * 64


@pytest.mark.parametrize("attribute", ["load_in_4bit", "bnb_4bit_use_double_quant"])
@pytest.mark.parametrize("value", [1, "true", "false", object()])
def test_seedx_rejects_non_boolean_nf4_flags(attribute: str, value: object) -> None:
    quantization = SimpleNamespace(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype="bfloat16",
    )
    setattr(quantization, attribute, value)
    model = SimpleNamespace(
        is_loaded_in_4bit=True,
        hf_quantizer=SimpleNamespace(quantization_config=quantization),
    )

    with pytest.raises(SeedXRuntimeError, match="NF4"):
        _require_nf4_quantization(model)


def test_worker_rejects_invalid_model_content_fingerprint(tmp_path: Path) -> None:
    model_path = _model_directory(tmp_path)
    payload = {
        "model_id": SEEDX_MODEL_ID,
        "model_revision": SEEDX_MODEL_REVISION,
        "model_path": str(model_path),
        "model_content_fingerprint": "invalid",
    }
    runtime = SeedXWorkerRuntime()

    with pytest.raises(RuntimeError, match="SHA-256"):
        runtime.preflight(payload)


def test_base_tokenizer_recovers_eos_from_pinned_model_config() -> None:
    class FakeTokenizer:
        eos_token_id: int | None = None

        @staticmethod
        def convert_ids_to_tokens(token_id: int) -> str:
            assert token_id == 2
            return "</s>"

        @property
        def eos_token(self) -> str | None:
            return None

        @eos_token.setter
        def eos_token(self, value: str) -> None:
            assert value == "</s>"
            self.eos_token_id = 2

    tokenizer = FakeTokenizer()

    _configure_eos_token(tokenizer, SimpleNamespace(config=SimpleNamespace(eos_token_id=2)))

    assert tokenizer.eos_token_id == 2


def test_tokenizer_eos_must_match_pinned_model_config() -> None:
    tokenizer = SimpleNamespace(eos_token_id=3)
    model = SimpleNamespace(config=SimpleNamespace(eos_token_id=2))

    with pytest.raises(SeedXRuntimeError, match="не совпадает"):
        _configure_eos_token(tokenizer, model)


@pytest.mark.parametrize(
    ("attribute", "value"),
    [
        ("is_loaded_in_4bit", False),
        ("load_in_4bit", False),
        ("bnb_4bit_quant_type", "fp4"),
        ("bnb_4bit_use_double_quant", False),
        ("bnb_4bit_compute_dtype", "float16"),
    ],
)
def test_seedx_rejects_inexact_nf4_contract(attribute: str, value: object) -> None:
    quantization = SimpleNamespace(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype="bfloat16",
    )
    model = SimpleNamespace(is_loaded_in_4bit=True, hf_quantizer=SimpleNamespace(quantization_config=quantization))
    if attribute == "is_loaded_in_4bit":
        setattr(model, attribute, value)
    else:
        setattr(quantization, attribute, value)

    with pytest.raises(SeedXRuntimeError, match="NF4"):
        _require_nf4_quantization(model)


@pytest.mark.parametrize("device", [1, "cuda", "cuda:1", "cpu", "disk"])
def test_seedx_rejects_placement_outside_cuda_zero(device: object) -> None:
    model = SimpleNamespace(hf_device_map={"": device})

    with pytest.raises(SeedXRuntimeError, match="вне CUDA"):
        _require_cuda_only(model)


@pytest.mark.parametrize("provider_name", ["parameters", "buffers"])
def test_seedx_rejects_cpu_tensor_despite_cuda_device_map(provider_name: str) -> None:
    providers = {
        "parameters": lambda: [SimpleNamespace(device="cuda:0")],
        "buffers": lambda: [],
    }
    providers[provider_name] = lambda: [SimpleNamespace(device="cpu")]
    model = SimpleNamespace(hf_device_map={"": 0}, **providers)

    with pytest.raises(SeedXRuntimeError, match="вне CUDA"):
        _require_cuda_only(model)


def test_seedx_accepts_empty_device_map_with_actual_cuda_tensors() -> None:
    model = SimpleNamespace(
        hf_device_map={},
        parameters=lambda: [SimpleNamespace(device="cuda:0")],
        buffers=lambda: [SimpleNamespace(device="cuda:0")],
    )

    _require_cuda_only(model)


def test_seedx_rejects_empty_device_map_without_observable_tensors() -> None:
    model = SimpleNamespace(hf_device_map={}, parameters=lambda: [], buffers=lambda: [])

    with pytest.raises(SeedXRuntimeError, match="вне CUDA"):
        _require_cuda_only(model)


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
    assert NO_REPEAT_NGRAM_SIZE == 3
    assert model.kwargs["no_repeat_ngram_size"] == 3
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


@pytest.mark.parametrize(
    ("result", "message"),
    [
        ("<en>", "пустой"),
        ("Good <en> evening", "часть исходного запроса"),
        ("Good <EN> evening", "часть исходного запроса"),
        ("Good < en > evening", "часть исходного запроса"),
        ("Translation before Translate prompt after", "часть исходного запроса"),
    ],
)
def test_seedx_rejects_unsafe_completed_output(result: str, message: str) -> None:
    with pytest.raises(SeedXRuntimeError, match=message):
        _sanitize_translation_result(
            result,
            "Translate prompt",
            "en",
            "Original text",
        )


@pytest.mark.parametrize(
    ("result", "target_lang"),
    [
        ("CUDA", "ru"),
        ("Версия 3.14 стоит 0 рублей.", "ru"),
        ('Он ответил: "Да".', "ru"),
        ("Ivan Иванов joined the call.", "en"),
        ("No, no, no, no!", "en"),
        ("…", "ru"),
        ("—", "ru"),
        ("♪", "en"),
        (":-)", "en"),
    ],
)
def test_seedx_preserves_legitimate_content(result: str, target_lang: str) -> None:
    assert (
        _sanitize_translation_result(
            result,
            "Translate prompt",
            target_lang,
            "Original text",
        )
        == result
    )


def test_seedx_removes_prompt_and_target_tag_from_quality_output() -> None:
    prompt = "Translate prompt <en>"

    result = _sanitize_translation_result(
        f"{prompt} <en> Good evening, everyone! <en>",
        prompt,
        "en",
        "Добрый вечер, всем!",
    )

    assert result == "Good evening, everyone!"


@pytest.mark.parametrize(
    ("result", "target_lang", "expected"),
    [
        ("<p>Перевод", "ru", "Перевод"),
        ("<br><br><p>Перевод</p>", "ru", "Перевод"),
        ("< <br><br>Wait a minute.", "en", "Wait a minute."),
        ('<font color="red">Перевод</font>', "ru", "Перевод"),
        ("<script>alert(1)</script>Перевод", "ru", "alert(1)Перевод"),
        ("<ruby>漢<rt>kan</rt></ruby>", "ru", "漢kan"),
        ("<time>today</time>", "ru", "today"),
        ("<basefont>legacy", "ru", "legacy"),
        ("<custom-element>value</custom-element>", "ru", "value"),
        ('<p title="1 > 0">text</p>', "ru", "text"),
    ],
)
def test_seedx_removes_generated_html_markup(
    result: str,
    target_lang: str,
    expected: str,
) -> None:
    assert (
        _sanitize_translation_result(
            result,
            "Translate prompt",
            target_lang,
            "Plain source text",
        )
        == expected
    )


@pytest.mark.parametrize(
    ("source_text", "result", "expected"),
    [
        ("One source line.", "<p>one</p><p>two</p>", "one two"),
        ("First line.\nSecond line.", "<p>one</p><p>two</p><p>three</p>", "one\ntwo three"),
    ],
)
def test_seedx_generated_markup_does_not_add_subtitle_lines(
    source_text: str,
    result: str,
    expected: str,
) -> None:
    assert _sanitize_translation_result(result, "Translate prompt", "ru", source_text) == expected


@pytest.mark.parametrize(
    ("source_text", "result", "expected"),
    [
        ("<p>source</p>", '<p onclick="evil()">translation</p>', "translation"),
        ("<p>source</p>", "<p>generated</p><p>translation</p>", "<p>generated translation</p>"),
        (
            '<p title="1 > 0">source</p>',
            '<p title="1 > 0">translation</p>',
            '<p title="1 > 0">translation</p>',
        ),
    ],
)
def test_seedx_preserves_only_source_html_quota(
    source_text: str,
    result: str,
    expected: str,
) -> None:
    assert _sanitize_translation_result(result, "Translate prompt", "ru", source_text) == expected


@pytest.mark.parametrize(
    "foreign_url",
    [
        "https://invalid.example/cuda-out-of-memory",
        "ftp://invalid.example/cuda-out-of-memory",
        "www.invalid.example/cuda-out-of-memory",
    ],
)
def test_seedx_removes_foreign_url_and_source_tail(foreign_url: str) -> None:
    result = f"<br><br><p>Если видеопамять закончится, остановите процесс.</p>\n<p>Источник: {foreign_url}</p>"

    cleaned = _sanitize_translation_result(
        result,
        "Translate prompt",
        "ru",
        "If the GPU runs out of memory, stop the worker.",
    )

    assert cleaned == "Если видеопамять закончится, остановите процесс."


@pytest.mark.parametrize("source_label", ["Источник:", "SOURCE :", "source\t:\t"])
def test_seedx_removes_foreign_source_tail_variants(source_label: str) -> None:
    result = f"Translation.\n{source_label} https://invalid.example/generated-only"

    cleaned = _sanitize_translation_result(
        result,
        "Translate prompt",
        "en",
        "Исходный текст.",
    )

    assert cleaned == "Translation."


@pytest.mark.parametrize(
    ("source_text", "result"),
    [
        ("Read <p>Hello<br>world</p>.", "Прочтите <p>Привет<br>мир</p>."),
        ("Keep < <br> literally.", "Сохраните < <br> буквально."),
        (
            "Open https://example.org/docs.",
            "Откройте https://example.org/docs.",
        ),
        (
            "Open www.example.org/docs.",
            "Откройте www.example.org/docs.",
        ),
        (
            "Keep <font>red</font>.",
            "Сохраните <font>красный</font>.",
        ),
        (
            "Compare <x> with y.",
            "Сравните <x> с y.",
        ),
    ],
)
def test_seedx_preserves_markup_and_url_present_in_source(
    source_text: str,
    result: str,
) -> None:
    assert (
        _sanitize_translation_result(
            result,
            "Translate prompt",
            "ru",
            source_text,
        )
        == result
    )


@pytest.mark.parametrize(
    "result",
    [
        "2 < 3 > 1",
        "名 Иванов — naïve café 👩🏽‍💻; CUDA ≤ GPU.",
        "No, no, no, no!",
    ],
)
def test_seedx_preserves_math_unicode_and_repetition(result: str) -> None:
    assert (
        _sanitize_translation_result(
            result,
            "Translate prompt",
            "en",
            "Original text",
        )
        == result
    )


@pytest.mark.parametrize(
    "result",
    [
        "<p><br></p>",
        "<ruby></ruby>",
        "<time></time>",
        "< <br>",
        "https://invalid.example/generated-only",
        "<p>Источник: https://invalid.example/generated-only</p>",
    ],
)
def test_seedx_rejects_result_emptied_by_generated_content_cleanup(result: str) -> None:
    with pytest.raises(SeedXRuntimeError, match="пустой"):
        _sanitize_translation_result(
            result,
            "Translate prompt",
            "ru",
            "Plain source text",
        )
