from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, cast

from sub_translate.constants import MODELS_DIR
from sub_translate.translators.base import TranslationError, Translator
from sub_translate.translators.lifecycle import LocalTranslatorLifecycleCoordinator

if TYPE_CHECKING:
    from sub_translate.models import TranslationOptions
    from sub_translate.utils.huggingface import ResolvedLocalModel

DEFAULT_TRANSLATOR_ID = "google"
NLLB_600M_MODEL_ID = "facebook/nllb-200-distilled-600M"
NLLB_600M_MODEL_REVISION = "f8d333a098d19b4fd9a8b18f94170487ad3f821d"
SEEDX_MODEL_ID = "ByteDance-Seed/Seed-X-PPO-7B"
SEEDX_MODEL_REVISION = "6ef78fc034ec86c0036d7a7ca2bfc24607f48050"
SEEDX_WORKER_REQUIREMENTS = (
    "accelerate==1.14.0",
    "bitsandbytes==0.50.0",
    "huggingface_hub==1.27.0",
    "safetensors==0.8.0",
    "tokenizers==0.22.2",
    "torch==2.13.0+cu130",
    "transformers==5.14.1",
)
TRANSLATEGEMMA_MODEL_ID = "google/translategemma-4b-it"
TRANSLATEGEMMA_MODEL_REVISION = "10042cb0e6e7fdce748996a71dc3dc432a4e0c89"
TRANSLATEGEMMA_12B_MODEL_ID = "google/translategemma-12b-it"
TRANSLATEGEMMA_12B_MODEL_REVISION = "d1b225e1caa17f1ddc7e62065d8637d0923f34e2"
TRANSLATEGEMMA_MAX_BATCH_SIZE = 128
TRANSLATEGEMMA_PROFILE_IDS = ("translategemma", "translategemma-12b")
TRANSLATEGEMMA_WORKER_REQUIREMENTS = (
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
_SAFETENSORS_FILES = (("model.safetensors", "model.safetensors.index.json"),)
_CONFIG_FILE = "config.json"
_GENERATION_CONFIG_FILE = "generation_config.json"
_TOKENIZER_CONFIG_FILE = "tokenizer_config.json"
_TOKENIZER_FILE = "tokenizer.json"


@dataclass(frozen=True, slots=True)
class TranslatorMetadata:
    """Стабильные сведения о движке для интерфейсов и диспетчеризации."""

    id: str
    display_name: str
    aliases: tuple[str, ...]
    local: bool
    thread_safe: bool
    runtime_kind: str = "in_process"
    worker_requirements: tuple[str, ...] = ()
    model_id: str | None = None
    model_revision: str | None = None
    model_path: Path | None = None
    required_files: tuple[str, ...] = ()
    required_file_groups: tuple[tuple[str, ...], ...] = ()
    requires_hf_token: bool = False
    supported_directions: tuple[tuple[str, str], ...] = ()
    license_name: str | None = None
    primary_source_url: str | None = None
    estimated_disk_bytes: int | None = None
    estimated_ram_bytes: int | None = None
    estimated_vram_bytes: int | None = None
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None
    quantization: str | None = None
    supports_cpu_fallback: bool = False
    default_timeout_seconds: int = 30
    deprecated: bool = False


@dataclass(frozen=True, slots=True)
class TranslatorRegistration:
    """Декларативная регистрация адаптера без его преждевременного импорта."""

    metadata: TranslatorMetadata
    module_name: str
    class_name: str
    accepts_timeout: bool = False

    def create(self, timeout: int) -> Translator:
        translator_class = getattr(import_module(self.module_name), self.class_name)
        if self.accepts_timeout:
            return cast(Translator, translator_class(timeout=timeout))
        return cast(Translator, translator_class())


TRANSLATOR_REGISTRY = (
    TranslatorRegistration(
        metadata=TranslatorMetadata(
            id="google",
            display_name="Google Translate",
            aliases=("google-translate-api",),
            local=False,
            thread_safe=True,
        ),
        module_name="sub_translate.translators.google_web",
        class_name="GoogleWebTranslator",
        accepts_timeout=True,
    ),
    TranslatorRegistration(
        metadata=TranslatorMetadata(
            id="agent",
            display_name="OpenAI Agent",
            aliases=("openai",),
            local=False,
            thread_safe=True,
        ),
        module_name="sub_translate.translators.agent",
        class_name="AgentTranslator",
    ),
    TranslatorRegistration(
        metadata=TranslatorMetadata(
            id="nllb-600m",
            display_name="NLLB 600M",
            aliases=("nllb-200-distilled-600m",),
            local=True,
            thread_safe=False,
            model_id=NLLB_600M_MODEL_ID,
            model_revision=NLLB_600M_MODEL_REVISION,
            model_path=MODELS_DIR / "nllb-200-distilled-600m",
            required_files=(
                _CONFIG_FILE,
                _GENERATION_CONFIG_FILE,
                _TOKENIZER_CONFIG_FILE,
                _TOKENIZER_FILE,
                "sentencepiece.bpe.model",
            ),
            required_file_groups=(("pytorch_model.bin",),),
            supported_directions=(("en", "ru"), ("ru", "en")),
            license_name="CC-BY-NC-4.0",
            primary_source_url="https://huggingface.co/facebook/nllb-200-distilled-600M",
            estimated_disk_bytes=3 * 1024**3,
            estimated_ram_bytes=4 * 1024**3,
            estimated_vram_bytes=4 * 1024**3,
            max_input_tokens=512,
            supports_cpu_fallback=True,
        ),
        module_name="sub_translate.translators.local.nllb",
        class_name="Nllb600MTranslator",
    ),
    TranslatorRegistration(
        metadata=TranslatorMetadata(
            id="translategemma",
            display_name="TranslateGemma 4B",
            aliases=(
                "translate-gemma",
                "translate-gemma-4b",
                "translategemma-4b",
                "translategemma-4b-it",
            ),
            local=True,
            thread_safe=False,
            runtime_kind="isolated_worker",
            worker_requirements=TRANSLATEGEMMA_WORKER_REQUIREMENTS,
            model_id=TRANSLATEGEMMA_MODEL_ID,
            model_revision=TRANSLATEGEMMA_MODEL_REVISION,
            model_path=MODELS_DIR / "translategemma-4b-it",
            required_files=(
                _CONFIG_FILE,
                _TOKENIZER_CONFIG_FILE,
                _TOKENIZER_FILE,
                "preprocessor_config.json",
                "processor_config.json",
            ),
            required_file_groups=_SAFETENSORS_FILES,
            requires_hf_token=True,
            supported_directions=(("en", "ru"), ("ru", "en")),
            license_name="Gemma Terms of Use",
            primary_source_url="https://huggingface.co/google/translategemma-4b-it",
            estimated_disk_bytes=9 * 1024**3,
            estimated_ram_bytes=11 * 1024**3,
            estimated_vram_bytes=6 * 1024**3,
            max_input_tokens=512,
            max_output_tokens=1_024,
            quantization="bitsandbytes-int8",
            default_timeout_seconds=3_600,
        ),
        module_name="sub_translate.translators.local.translategemma",
        class_name="TranslateGemmaTranslator",
        accepts_timeout=True,
    ),
    TranslatorRegistration(
        metadata=TranslatorMetadata(
            id="translategemma-12b",
            display_name="TranslateGemma 12B (NF4)",
            aliases=(
                "translate-gemma-12b",
                "translate-gemma-12b-it",
                "translategemma-12b-it",
            ),
            local=True,
            thread_safe=False,
            runtime_kind="isolated_worker",
            worker_requirements=TRANSLATEGEMMA_WORKER_REQUIREMENTS,
            model_id=TRANSLATEGEMMA_12B_MODEL_ID,
            model_revision=TRANSLATEGEMMA_12B_MODEL_REVISION,
            model_path=MODELS_DIR / "translategemma-12b-it",
            required_files=(
                "chat_template.jinja",
                _CONFIG_FILE,
                _GENERATION_CONFIG_FILE,
                _TOKENIZER_CONFIG_FILE,
                _TOKENIZER_FILE,
                "preprocessor_config.json",
                "processor_config.json",
            ),
            required_file_groups=_SAFETENSORS_FILES,
            requires_hf_token=True,
            supported_directions=(("en", "ru"), ("ru", "en")),
            license_name="Gemma Terms of Use",
            primary_source_url="https://huggingface.co/google/translategemma-12b-it",
            estimated_disk_bytes=27 * 1024**3,
            estimated_ram_bytes=32 * 1024**3,
            estimated_vram_bytes=12 * 1024**3,
            max_input_tokens=512,
            max_output_tokens=512,
            quantization="bitsandbytes-nf4-double",
            default_timeout_seconds=3_600,
        ),
        module_name="sub_translate.translators.local.translategemma",
        class_name="TranslateGemma12BTranslator",
        accepts_timeout=True,
    ),
    TranslatorRegistration(
        metadata=TranslatorMetadata(
            id="seedx",
            display_name="Seed-X PPO 7B (NF4)",
            aliases=(
                "seed-x",
                "seed-x-ppo",
                "seed-x-ppo-7b",
            ),
            local=True,
            thread_safe=False,
            runtime_kind="isolated_worker",
            worker_requirements=SEEDX_WORKER_REQUIREMENTS,
            model_id=SEEDX_MODEL_ID,
            model_revision=SEEDX_MODEL_REVISION,
            model_path=MODELS_DIR / "seed-x-ppo-7b",
            required_files=(
                _CONFIG_FILE,
                _GENERATION_CONFIG_FILE,
                _TOKENIZER_FILE,
            ),
            required_file_groups=(("model.safetensors",),),
            supported_directions=(("en", "ru"), ("ru", "en")),
            license_name="OpenMDW",
            primary_source_url="https://huggingface.co/ByteDance-Seed/Seed-X-PPO-7B",
            estimated_disk_bytes=15 * 1024**3,
            estimated_ram_bytes=18 * 1024**3,
            estimated_vram_bytes=6 * 1024**3,
            max_input_tokens=4_096,
            max_output_tokens=512,
            quantization="bitsandbytes-nf4-double",
            default_timeout_seconds=3_600,
        ),
        module_name="sub_translate.translators.local.seedx",
        class_name="SeedXTranslator",
        accepts_timeout=True,
    ),
)


def _normalize_identifier(value: str) -> str:
    return value.strip().lower()


def _build_registry_indexes() -> tuple[
    dict[str, TranslatorRegistration],
    MappingProxyType[str, str],
]:
    registrations: dict[str, TranslatorRegistration] = {}
    aliases: dict[str, str] = {}
    for registration in TRANSLATOR_REGISTRY:
        metadata = registration.metadata
        canonical_id = _normalize_identifier(metadata.id)
        if canonical_id != metadata.id or canonical_id in registrations:
            raise RuntimeError(f"Некорректный идентификатор переводчика: {metadata.id}")
        registrations[canonical_id] = registration
        for alias in (canonical_id, *metadata.aliases):
            normalized_alias = _normalize_identifier(alias)
            if normalized_alias in aliases:
                raise RuntimeError(f"Алиас переводчика зарегистрирован повторно: {alias}")
            aliases[normalized_alias] = canonical_id
    return registrations, MappingProxyType(aliases)


_REGISTRATIONS_BY_ID, TRANSLATOR_ALIASES = _build_registry_indexes()
_LOCAL_TRANSLATOR_LIFECYCLE = LocalTranslatorLifecycleCoordinator()


def resolve_translator_id(value: str | None) -> str:
    """Разрешает алиас, сохраняя прежнее поведение для неизвестного значения."""
    if not value or not value.strip():
        return DEFAULT_TRANSLATOR_ID
    identifier = _normalize_identifier(value)
    return TRANSLATOR_ALIASES.get(identifier, identifier)


def get_translator_metadata(identifier: str | None) -> TranslatorMetadata:
    canonical_id = resolve_translator_id(identifier)
    registration = _REGISTRATIONS_BY_ID.get(canonical_id)
    if registration is None:
        raise TranslationError(f"Неизвестный переводчик: {canonical_id}")
    return registration.metadata


def list_translator_metadata() -> tuple[TranslatorMetadata, ...]:
    return tuple(registration.metadata for registration in TRANSLATOR_REGISTRY)


def create_translator(identifier: str | None, timeout: int) -> Translator:
    canonical_id = resolve_translator_id(identifier)
    registration = _REGISTRATIONS_BY_ID.get(canonical_id)
    if registration is None:
        raise TranslationError(f"Неизвестный переводчик: {canonical_id}")
    if not registration.metadata.local:
        return registration.create(timeout)
    return _LOCAL_TRANSLATOR_LIFECYCLE.create(
        canonical_id,
        lambda: registration.create(timeout),
    )


def get_active_local_translator_id() -> str | None:
    """Возвращает идентификатор владельца единственного локального слота."""
    return _LOCAL_TRANSLATOR_LIFECYCLE.active_profile_id


def unload_all_local_translators() -> None:
    """Выгружает активный локальный движок через общую точку жизненного цикла."""
    _LOCAL_TRANSLATOR_LIFECYCLE.unload_all()


def resolve_registered_model(
    identifier: str,
    options: TranslationOptions,
) -> ResolvedLocalModel:
    metadata = get_translator_metadata(identifier)
    if (
        not metadata.local
        or metadata.model_id is None
        or metadata.model_revision is None
        or metadata.model_path is None
    ):
        raise TranslationError(f"Для переводчика {metadata.id} не задана локальная модель.")
    if options.model_revision is not None and options.model_revision.strip().lower() != metadata.model_revision:
        raise TranslationError(
            f"Переводчик {metadata.id} запускается только с закреплённой ревизией {metadata.model_revision}."
        )

    from sub_translate.utils.huggingface import acquire_local_model

    return acquire_local_model(
        model_id=metadata.model_id,
        revision=metadata.model_revision,
        model_path=options.model_path,
        default_model_path=metadata.model_path,
        auto_download=options.auto_download_model,
        required_files=metadata.required_files,
        required_file_groups=metadata.required_file_groups,
        requires_hf_token=metadata.requires_hf_token,
    )


__all__ = [
    "DEFAULT_TRANSLATOR_ID",
    "NLLB_600M_MODEL_ID",
    "NLLB_600M_MODEL_REVISION",
    "SEEDX_MODEL_ID",
    "SEEDX_MODEL_REVISION",
    "SEEDX_WORKER_REQUIREMENTS",
    "TRANSLATEGEMMA_12B_MODEL_ID",
    "TRANSLATEGEMMA_12B_MODEL_REVISION",
    "TRANSLATEGEMMA_MAX_BATCH_SIZE",
    "TRANSLATEGEMMA_MODEL_ID",
    "TRANSLATEGEMMA_MODEL_REVISION",
    "TRANSLATEGEMMA_PROFILE_IDS",
    "TRANSLATEGEMMA_WORKER_REQUIREMENTS",
    "TRANSLATOR_ALIASES",
    "TRANSLATOR_REGISTRY",
    "TranslatorMetadata",
    "TranslatorRegistration",
    "create_translator",
    "get_active_local_translator_id",
    "get_translator_metadata",
    "list_translator_metadata",
    "resolve_registered_model",
    "resolve_translator_id",
    "unload_all_local_translators",
]
