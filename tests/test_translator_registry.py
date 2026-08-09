from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from sub_translate.models import TranslationOptions
from sub_translate.translators.base import TranslationError
from sub_translate.translators.registry import (
    create_translator,
    get_translator_metadata,
    list_translator_metadata,
    resolve_registered_model,
    resolve_translator_id,
)
from sub_translate.utils.huggingface import ResolvedLocalModel

EXPECTED_TRANSLATOR_IDS = {
    "agent",
    "google",
    "nllb-600m",
    "seedx",
    "translategemma",
    "translategemma-12b",
}


def test_registry_contains_all_supported_translators() -> None:
    metadata = list_translator_metadata()

    assert {item.id for item in metadata} == EXPECTED_TRANSLATOR_IDS
    assert len({alias for item in metadata for alias in (item.id, *item.aliases)}) == sum(
        1 + len(item.aliases) for item in metadata
    )


@pytest.mark.parametrize(
    ("alias", "expected"),
    [
        (" google-translate-api ", "google"),
        ("OPENAI", "agent"),
        ("nllb-200-distilled-600m", "nllb-600m"),
        ("seed-x-ppo-7b-awq-int4", "seedx"),
        ("translate-gemma-4b", "translategemma"),
        ("translate-gemma-12b-it", "translategemma-12b"),
    ],
)
def test_registry_resolves_aliases(alias: str, expected: str) -> None:
    assert resolve_translator_id(alias) == expected


def test_translategemma_metadata_is_pinned_and_local() -> None:
    metadata = get_translator_metadata("translategemma-4b-it")

    assert metadata.id == "translategemma"
    assert metadata.local is True
    assert metadata.thread_safe is False
    assert metadata.runtime_kind == "isolated_worker"
    assert metadata.worker_requirements == (
        "torch==2.13.0+cu130",
        "transformers==5.14.1",
        "accelerate==1.14.0",
        "bitsandbytes==0.50.0",
        "huggingface_hub==1.27.0",
        "protobuf==7.35.1",
        "safetensors==0.8.0",
        "sentencepiece==0.2.2",
        "tokenizers==0.22.2",
    )
    assert metadata.model_id == "google/translategemma-4b-it"
    assert metadata.model_revision == "10042cb0e6e7fdce748996a71dc3dc432a4e0c89"
    assert metadata.model_path is not None
    assert metadata.requires_hf_token is True
    assert "config.json" in metadata.required_files
    assert metadata.supported_directions == (("en", "ru"), ("ru", "en"))
    assert metadata.license_name == "Gemma Terms of Use"
    assert metadata.primary_source_url == "https://huggingface.co/google/translategemma-4b-it"
    assert metadata.estimated_disk_bytes == 9 * 1024**3
    assert metadata.estimated_ram_bytes == 11 * 1024**3
    assert metadata.estimated_vram_bytes == 6 * 1024**3
    assert metadata.max_input_tokens == 512
    assert metadata.max_output_tokens == 1_024
    assert metadata.quantization == "bitsandbytes-int8"
    assert metadata.default_timeout_seconds == 3_600


def test_translategemma_12b_metadata_is_pinned_and_nf4() -> None:
    metadata = get_translator_metadata("translate-gemma-12b")

    assert metadata.id == "translategemma-12b"
    assert metadata.model_id == "google/translategemma-12b-it"
    assert metadata.model_revision == "d1b225e1caa17f1ddc7e62065d8637d0923f34e2"
    assert metadata.model_path is not None
    assert metadata.model_path.name == "translategemma-12b-it"
    assert metadata.runtime_kind == "isolated_worker"
    assert metadata.required_file_groups == (("model.safetensors", "model.safetensors.index.json"),)
    assert "chat_template.jinja" in metadata.required_files
    assert metadata.requires_hf_token is True
    assert metadata.supported_directions == (("en", "ru"), ("ru", "en"))
    assert metadata.primary_source_url == "https://huggingface.co/google/translategemma-12b-it"
    assert metadata.max_input_tokens == 512
    assert metadata.max_output_tokens == 512
    assert metadata.quantization == "bitsandbytes-nf4-double"
    assert metadata.default_timeout_seconds == 3_600


def test_nllb_600m_has_exact_pinned_metadata() -> None:
    metadata = get_translator_metadata("nllb-600m")

    assert metadata.model_id == "facebook/nllb-200-distilled-600M"
    assert metadata.model_revision == "f8d333a098d19b4fd9a8b18f94170487ad3f821d"
    assert metadata.required_file_groups == (("pytorch_model.bin",),)
    assert metadata.supported_directions == (("en", "ru"), ("ru", "en"))
    assert metadata.license_name == "CC-BY-NC-4.0"
    assert metadata.primary_source_url == "https://huggingface.co/facebook/nllb-200-distilled-600M"
    assert metadata.deprecated is False


def test_seedx_metadata_is_pinned_and_isolated() -> None:
    metadata = get_translator_metadata("seed-x-ppo")

    assert metadata.id == "seedx"
    assert metadata.display_name == "Seed-X PPO 7B Int4"
    assert metadata.model_id == "ByteDance-Seed/Seed-X-PPO-7B-AWQ-Int4"
    assert metadata.model_revision == "64a72a40045ac345005795f703a8ba627e99b48e"
    assert metadata.runtime_kind == "isolated_worker"
    assert metadata.required_file_groups == (("model.safetensors",),)
    assert metadata.license_name == "OpenMDW"
    assert metadata.estimated_vram_bytes == 8 * 1024**3
    assert metadata.max_input_tokens == 4_096
    assert metadata.quantization == "compressed-tensors-awq-int4"
    assert metadata.default_timeout_seconds == 3_600
    assert "compressed-tensors==0.18.0" in metadata.worker_requirements


@pytest.mark.parametrize(
    "identifier",
    [
        "fsm",
        "fsmt",
        "madlad",
        "madlad-3b",
        "nllb",
        "nllb-lite",
        "seamless",
    ],
)
def test_retired_local_profiles_are_not_registered(identifier: str) -> None:
    with pytest.raises(TranslationError, match="Неизвестный переводчик"):
        get_translator_metadata(identifier)


def test_every_local_model_has_pinned_acquisition_metadata() -> None:
    local_models = [item for item in list_translator_metadata() if item.local]

    assert local_models
    for metadata in local_models:
        assert metadata.model_id
        assert metadata.model_revision
        assert len(metadata.model_revision) == 40
        assert metadata.model_path is not None
        assert metadata.required_files
        assert metadata.required_file_groups
        assert metadata.license_name
        assert metadata.primary_source_url
        assert metadata.estimated_disk_bytes
        assert metadata.estimated_ram_bytes
        assert metadata.estimated_vram_bytes
        assert metadata.max_input_tokens


@pytest.mark.parametrize(
    ("translator_id", "class_name"),
    [
        ("google", "GoogleWebTranslator"),
        ("translategemma", "TranslateGemmaTranslator"),
        ("translategemma-12b", "TranslateGemma12BTranslator"),
        ("seedx", "SeedXTranslator"),
    ],
)
def test_registry_factory_passes_timeout_to_supported_translators(
    translator_id: str,
    class_name: str,
) -> None:
    class FakeGoogleTranslator:
        name = translator_id

        def __init__(self, *, timeout: int) -> None:
            self.timeout = timeout

        def unload(self) -> None:
            pass

    module = SimpleNamespace(**{class_name: FakeGoogleTranslator})
    with patch("sub_translate.translators.registry.import_module", return_value=module):
        translator = create_translator(translator_id, timeout=17)

    assert translator.name == translator_id
    assert translator.timeout == 17


def test_registry_rejects_unknown_translator() -> None:
    with pytest.raises(TranslationError, match="Неизвестный переводчик"):
        create_translator("unknown", timeout=30)


def test_registered_model_forwards_explicit_acquisition_options(tmp_path: Path) -> None:
    selected_path = tmp_path / "selected-model"
    custom_revision = "abcdef0123456789abcdef0123456789abcdef01"
    expected = ResolvedLocalModel(selected_path, custom_revision)
    options = TranslationOptions(
        model_path=selected_path,
        model_revision=custom_revision,
        auto_download_model=True,
    )

    with patch(
        "sub_translate.utils.huggingface.acquire_local_model",
        return_value=expected,
    ) as acquire:
        result = resolve_registered_model("translategemma", options)

    assert result is expected
    call_kwargs = acquire.call_args.kwargs
    assert call_kwargs["model_path"] == selected_path
    assert call_kwargs["revision"] == custom_revision
    assert call_kwargs["auto_download"] is True
    assert call_kwargs["requires_hf_token"] is True
