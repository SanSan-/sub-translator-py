from unittest.mock import MagicMock, patch

import torch

from sub_translate.utils.huggingface import load_model_components


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
            use_safetensors=True,
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
            use_safetensors=True,
            allow_cpu_fallback=True,
        )

    assert processor is processor_class.from_pretrained.return_value
    assert model is model_instance
    assert device.type == "cpu"
    assert calls[0].get("device_map") == "auto"
    assert "device_map" not in calls[1]
    assert calls[1].get("dtype") == torch.float16
    assert calls[2].get("low_cpu_mem_usage") is False
    assert calls[2].get("dtype") is None
