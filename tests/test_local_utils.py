from unittest.mock import MagicMock, patch

import pytest

from sub_translate.utils.local_utils import (
    Int8UnavailableError,
    resolve_device_and_quantization,
    sanitize_generation_config,
)


def test_required_int8_rejects_cpu() -> None:
    with (
        patch("sub_translate.utils.local_utils.torch.cuda.is_available", return_value=False),
        pytest.raises(Int8UnavailableError, match="CUDA"),
    ):
        resolve_device_and_quantization(require_int8=True)


def test_required_int8_rejects_missing_bitsandbytes() -> None:
    with (
        patch("sub_translate.utils.local_utils.torch.cuda.is_available", return_value=True),
        patch("sub_translate.utils.local_utils.BitsAndBytesConfig", None),
        pytest.raises(Int8UnavailableError, match="bitsandbytes"),
    ):
        resolve_device_and_quantization(require_int8=True)


def test_required_int8_builds_configuration_without_cpu_offload() -> None:
    bitsandbytes_config = MagicMock(return_value=object())
    with (
        patch("sub_translate.utils.local_utils.torch.cuda.is_available", return_value=True),
        patch("sub_translate.utils.local_utils.BitsAndBytesConfig", bitsandbytes_config),
    ):
        device, quantization = resolve_device_and_quantization(
            require_int8=True,
            enable_cpu_offload=False,
        )

    assert device.type == "cuda"
    assert quantization is bitsandbytes_config.return_value
    bitsandbytes_config.assert_called_once_with(
        load_in_8bit=True,
        llm_int8_enable_fp32_cpu_offload=False,
    )


def test_required_int8_wraps_bitsandbytes_initialization_error() -> None:
    bitsandbytes_config = MagicMock(side_effect=RuntimeError("broken"))
    with (
        patch("sub_translate.utils.local_utils.torch.cuda.is_available", return_value=True),
        patch("sub_translate.utils.local_utils.BitsAndBytesConfig", bitsandbytes_config),
        pytest.raises(Int8UnavailableError, match="bitsandbytes"),
    ):
        resolve_device_and_quantization(require_int8=True)


def test_deterministic_generation_config_removes_sampling_only_settings() -> None:
    config = MagicMock(
        max_new_tokens=128,
        max_length=256,
        do_sample=True,
        num_beams=4,
        temperature=0.8,
        top_p=0.95,
        min_p=0.1,
        typical_p=0.9,
        top_k=64,
        epsilon_cutoff=0.1,
        eta_cutoff=0.1,
        early_stopping=True,
        length_penalty=0.8,
    )
    model = MagicMock(generation_config=config)

    sanitize_generation_config(model, deterministic=True)

    assert config.max_new_tokens is None
    assert config.do_sample is False
    assert config.num_beams == 1
    assert config.temperature is None
    assert config.top_p is None
    assert config.min_p is None
    assert config.typical_p is None
    assert config.top_k is None
    assert config.epsilon_cutoff is None
    assert config.eta_cutoff is None
    assert config.early_stopping is False
    assert config.length_penalty == 1.0
