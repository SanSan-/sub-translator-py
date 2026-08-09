from unittest.mock import MagicMock, patch

import pytest
import torch

from sub_translate.utils.huggingface import (
    ModelLoadOptions,
    TranslatorLoadError,
    load_model_components,
)


class FakeModel:
    def __init__(self) -> None:
        self.device = None

    def eval(self):
        return self

    def to(self, device):
        self.device = device
        return self


def test_load_model_components_fallback_on_meta_device(tmp_path):
    calls = []
    model_instance = FakeModel()

    def fake_from_pretrained(*_args, **kwargs):
        calls.append(kwargs)
        if kwargs.get("device_map") == "auto":
            raise ValueError("weight is on the meta device, we need a value to put it on 0.")
        return model_instance

    model_class = MagicMock()
    model_class.from_pretrained.side_effect = fake_from_pretrained
    processor_class = MagicMock()
    processor_class.from_pretrained.return_value = MagicMock()

    with patch("sub_translate.utils.huggingface.resolve_device_and_quantization") as resolve_mock:
        resolve_mock.return_value = (torch.device("cuda"), None)
        processor, model, device = load_model_components(
            "test-model",
            tmp_path,
            model_class,
            processor_class,
            ModelLoadOptions(use_safetensors=True),
        )

    assert processor is processor_class.from_pretrained.return_value
    assert model is model_instance
    assert device.type == "cuda"
    assert calls[0].get("device_map") == "auto"
    assert "device_map" not in calls[1]
    assert calls[1].get("dtype") == torch.float16
    assert calls[1].get("low_cpu_mem_usage") is False
    assert model_instance.device.type == "cuda"


def test_load_model_components_cpu_fallback_on_meta_device(tmp_path):
    calls = []
    model_instance = FakeModel()

    def fake_from_pretrained(*_args, **kwargs):
        calls.append(kwargs)
        if kwargs.get("device_map") == "auto":
            raise ValueError("weight is on the meta device, we need a value to put it on 0.")
        if kwargs.get("dtype") is not None:
            raise ValueError("weight is on the meta device, we need a value to put it on 0.")
        return model_instance

    model_class = MagicMock()
    model_class.from_pretrained.side_effect = fake_from_pretrained
    processor_class = MagicMock()
    processor_class.from_pretrained.return_value = MagicMock()

    with patch("sub_translate.utils.huggingface.resolve_device_and_quantization") as resolve_mock:
        resolve_mock.return_value = (torch.device("cuda"), None)
        processor, model, device = load_model_components(
            "test-model",
            tmp_path,
            model_class,
            processor_class,
            ModelLoadOptions(
                use_safetensors=True,
                allow_cpu_fallback=True,
            ),
        )

    assert processor is processor_class.from_pretrained.return_value
    assert model is model_instance
    assert device.type == "cpu"
    assert calls[0].get("device_map") == "auto"
    assert "device_map" not in calls[1]
    assert calls[1].get("dtype") == torch.float16
    assert calls[2].get("low_cpu_mem_usage") is False
    assert calls[2].get("dtype") is None


def test_load_model_components_passes_revision_and_local_only(tmp_path):
    model_instance = FakeModel()
    model_class = MagicMock()
    model_class.from_pretrained.return_value = model_instance
    processor_class = MagicMock()
    processor_class.from_pretrained.return_value = MagicMock()

    with patch("sub_translate.utils.huggingface.resolve_device_and_quantization") as resolve_mock:
        resolve_mock.return_value = (torch.device("cpu"), None)
        load_model_components(
            "test-model",
            tmp_path,
            model_class,
            processor_class,
            ModelLoadOptions(
                revision="pinned-revision",
                local_files_only=True,
                allow_cpu_fallback=True,
            ),
        )

    assert processor_class.from_pretrained.call_args.kwargs["revision"] == "pinned-revision"
    assert processor_class.from_pretrained.call_args.kwargs["local_files_only"] is True
    assert model_class.from_pretrained.call_args.kwargs["revision"] == "pinned-revision"
    assert model_class.from_pretrained.call_args.kwargs["local_files_only"] is True


def test_load_model_components_rejects_implicit_cpu(tmp_path):
    model_class = MagicMock()
    processor_class = MagicMock()

    with patch("sub_translate.utils.huggingface.resolve_device_and_quantization") as resolve_mock:
        resolve_mock.return_value = (torch.device("cpu"), None)
        with pytest.raises(TranslatorLoadError, match="CUDA"):
            load_model_components(
                "test-model",
                tmp_path,
                model_class,
                processor_class,
            )

    processor_class.from_pretrained.assert_not_called()
    model_class.from_pretrained.assert_not_called()


@pytest.mark.parametrize("placement", ["cpu", "disk"])
def test_load_model_components_rejects_hidden_offload(tmp_path, placement):
    model_instance = FakeModel()
    model_instance.hf_device_map = {"encoder": 0, "decoder": placement}
    model_class = MagicMock()
    model_class.from_pretrained.return_value = model_instance
    processor_class = MagicMock()
    processor_class.from_pretrained.return_value = MagicMock()

    with patch("sub_translate.utils.huggingface.resolve_device_and_quantization") as resolve_mock:
        resolve_mock.return_value = (torch.device("cuda"), None)
        with pytest.raises(TranslatorLoadError, match="CPU или диске"):
            load_model_components(
                "test-model",
                tmp_path,
                model_class,
                processor_class,
            )


def test_load_model_components_allows_explicit_cpu_offload(tmp_path):
    model_instance = FakeModel()
    model_instance.hf_device_map = {"encoder": 0, "decoder": "cpu"}
    model_class = MagicMock()
    model_class.from_pretrained.return_value = model_instance
    processor_class = MagicMock()
    processor_class.from_pretrained.return_value = MagicMock()

    with patch("sub_translate.utils.huggingface.resolve_device_and_quantization") as resolve_mock:
        resolve_mock.return_value = (torch.device("cuda"), None)
        _processor, model, device = load_model_components(
            "test-model",
            tmp_path,
            model_class,
            processor_class,
            ModelLoadOptions(allow_cpu_fallback=True),
        )

    assert model is model_instance
    assert device.type == "cuda"


def test_required_int8_does_not_fallback_to_unquantized_model(tmp_path):
    model_class = MagicMock()
    model_class.from_pretrained.side_effect = ValueError("weight is on the meta device")
    processor_class = MagicMock()
    processor_class.from_pretrained.return_value = MagicMock()
    quantization_config = object()
    load_options = ModelLoadOptions(
        use_safetensors=True,
        require_int8=True,
        enable_cpu_offload=False,
        device_map={"": 0},
    )

    with patch("sub_translate.utils.huggingface.resolve_device_and_quantization") as resolve_mock:
        resolve_mock.return_value = (torch.device("cuda"), quantization_config)
        with pytest.raises(TranslatorLoadError):
            load_model_components(
                "test-model",
                tmp_path,
                model_class,
                processor_class,
                load_options,
            )

    resolve_mock.assert_called_once_with(
        enable_cpu_offload=False,
        allow_quantization=True,
        require_int8=True,
    )
    assert model_class.from_pretrained.call_count == 1
    model_kwargs = model_class.from_pretrained.call_args.kwargs
    assert model_kwargs["quantization_config"] is quantization_config
    assert model_kwargs["device_map"] == {"": 0}
    assert model_kwargs["dtype"] == torch.float16
