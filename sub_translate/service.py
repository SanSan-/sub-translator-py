from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import StrEnum
from functools import partial
from pathlib import Path
from types import MappingProxyType

from sub_translate.constants import DEFAULT_BATCH_SIZE, DEFAULT_THREAD_COUNT, SUBTITLE_CACHE_FILE
from sub_translate.dictionaries.languages import get_code
from sub_translate.enums import FileFormat
from sub_translate.models import (
    PrepareToTranslateItem,
    SmartSplitSettings,
    TranslatedItem,
    TranslationOptions,
)
from sub_translate.translators.base import TranslationError, Translator
from sub_translate.translators.google_web import normalize_google_tld
from sub_translate.translators.registry import (
    TRANSLATOR_ALIASES,
    TranslatorMetadata,
    create_translator,
    get_translator_metadata,
    resolve_registered_model,
    resolve_translator_id,
)
from sub_translate.utils.io_utils import read_text, split_lines, write_lines
from sub_translate.utils.line_utils import (
    analyse_lines,
    build_export_lines,
    build_prepare,
    build_translated_dialogs,
    parse_ass_dialogs,
    parse_srt_dialogs,
    parse_vtt_dialogs,
)
from sub_translate.utils.path_utils import split_lang_suffix
from sub_translate.utils.subtitle_cache import (
    DEFAULT_CACHE_POLICY,
    CachePolicy,
    CacheRestoreStatus,
    OutputCacheIdentity,
    build_cache_fingerprint,
    restore_output_cache,
    store_output_cache,
    validate_cached_lines,
)

API_ALIASES = TRANSLATOR_ALIASES

FileProgressCallback = Callable[[int, int], None]
BatchStartedCallback = Callable[[Path, int, int], None]
BatchProgressCallback = Callable[[Path, int, int, int, int], None]
BatchResultCallback = Callable[["ProcessingResult", int, int], None]


class ProcessingStatus(StrEnum):
    """Итог обработки одного файла общим прикладным конвейером."""

    TRANSLATED = "translated"
    CACHED = "cached"
    OUTPUT_EXISTS = "output_exists"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class ProcessingSettings:
    """Единые прикладные настройки для командного и веб-интерфейсов."""

    source_lang: str = "auto"
    target_lang: str = "ru"
    api: str | None = None
    batch_size: int = DEFAULT_BATCH_SIZE
    thread_count: int = DEFAULT_THREAD_COUNT
    smart_split: bool = False
    smart_split_settings: SmartSplitSettings | None = None
    timeout: int | None = None
    force: bool = False
    tld: str = "com"
    request_delay_ms: int | None = None
    allow_cpu_fallback: bool = False
    model_path: Path | None = None
    model_revision: str | None = None
    worker_python_path: Path | None = None
    auto_download_model: bool = False
    agent_model: str | None = None
    prompt_signatures: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        source_lang = self.source_lang.strip()
        target_lang = self.target_lang.strip()
        if not source_lang or not target_lang:
            raise ValueError("Исходный и целевой языки не должны быть пустыми.")
        if not isinstance(self.api, str) or not self.api.strip():
            raise ValueError("Переводчик должен быть выбран явно.")
        resolved_api = resolve_api(self.api)
        resolved_timeout = self.timeout
        if resolved_timeout is None:
            resolved_timeout = get_translator_metadata(resolved_api).default_timeout_seconds
        for value, label in (
            (self.batch_size, "Размер пачки"),
            (self.thread_count, "Число потоков"),
            (resolved_timeout, "Время ожидания"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{label} должно быть положительным целым числом.")
        if self.request_delay_ms is not None and (
            isinstance(self.request_delay_ms, bool)
            or not isinstance(self.request_delay_ms, int)
            or self.request_delay_ms < 0
        ):
            raise ValueError("Задержка между запросами не должна быть отрицательной.")
        object.__setattr__(self, "source_lang", source_lang)
        object.__setattr__(self, "target_lang", target_lang)
        object.__setattr__(self, "api", resolved_api)
        object.__setattr__(self, "timeout", resolved_timeout)
        object.__setattr__(self, "tld", normalize_google_tld(self.tld))
        if self.model_path is not None:
            object.__setattr__(self, "model_path", Path(self.model_path).expanduser().resolve())
        if self.worker_python_path is not None:
            object.__setattr__(
                self,
                "worker_python_path",
                Path(self.worker_python_path).expanduser().resolve(),
            )
        object.__setattr__(self, "prompt_signatures", MappingProxyType(dict(self.prompt_signatures)))


@dataclass(frozen=True, slots=True)
class ProcessingResult:
    input_path: Path
    output_path: Path | None
    status: ProcessingStatus
    file_format: FileFormat | None = None
    error_type: str | None = None

    @property
    def successful(self) -> bool:
        return self.status is not ProcessingStatus.ERROR


def resolve_api(api: str | None) -> str:
    return resolve_translator_id(api)


def resolve_format(input_path: Path, format_arg: str | None) -> FileFormat:
    if format_arg:
        return FileFormat(format_arg.lower())
    ext = input_path.suffix.lower().lstrip(".")
    if ext:
        return FileFormat(ext)
    return FileFormat.ASS


def resolve_io_paths(
    input_path: Path,
    output_arg: str | Path | None,
    source_lang: str,
    target_lang: str,
    api: str,
    file_format: FileFormat,
) -> tuple[str, Path]:
    """Единообразно определяет язык источника и путь результата."""
    resolved_input = input_path.expanduser().resolve()
    base_stem, suffix_lang = split_lang_suffix(resolved_input.stem)
    resolved_source = suffix_lang or source_lang
    resolved_target = get_code(target_lang) or target_lang
    if output_arg:
        return resolved_source, Path(output_arg).expanduser().resolve()
    output_stem = f"{base_stem}.{api}.{resolved_target}"
    return resolved_source, resolved_input.with_name(f"{output_stem}.{file_format.value}")


def get_translator(api: str, timeout: int) -> Translator:
    return create_translator(api, timeout)


def _validate_batch_response(source_texts: list[str], response: object) -> list[str]:
    """Проверяет единый контракт пакетного ответа любого переводчика."""
    if not isinstance(response, list):
        raise TranslationError("Переводчик вернул пакет в неподдерживаемом формате.")
    if len(response) != len(source_texts):
        raise TranslationError("Ответ переводчика не совпадает с размером пачки.")
    for index, (source_text, translated_text) in enumerate(
        zip(source_texts, response, strict=True),
        start=1,
    ):
        if not isinstance(translated_text, str):
            raise TranslationError(f"Переводчик вернул не строку для элемента пачки {index}.")
        if source_text.strip() and not translated_text.strip():
            raise TranslationError(f"Переводчик вернул пустой перевод для непустого элемента пачки {index}.")
    return response


def _iter_batch_results(
    batches: list[list[PrepareToTranslateItem]],
    translator: Translator,
    options: TranslationOptions,
    thread_count: int,
) -> Iterable[tuple[list[PrepareToTranslateItem], object]]:
    """Возвращает ответы пачек последовательно или по мере завершения потоков."""
    if thread_count <= 1 or len(batches) <= 1:
        for batch in batches:
            source_texts = [item.to_translate for item in batch]
            yield batch, translator.translate_batch(source_texts, options)
        return

    with ThreadPoolExecutor(max_workers=thread_count) as executor:
        future_map = {
            executor.submit(
                translator.translate_batch,
                [item.to_translate for item in batch],
                options,
            ): batch
            for batch in batches
        }
        for future in as_completed(future_map):
            yield future_map[future], future.result()


def _apply_batch_result(
    batch: list[PrepareToTranslateItem],
    response: object,
    translated: dict[int, TranslatedItem],
    processed_count: int,
    total_count: int,
    logger: logging.Logger,
    progress_callback: FileProgressCallback | None,
) -> int:
    """Проверяет ответ пачки и сохраняет элементы в исходном порядке."""
    source_texts = [item.to_translate for item in batch]
    translations = _validate_batch_response(source_texts, response)
    for item, text in zip(batch, translations, strict=True):
        translated[item.idx] = TranslatedItem(idx=item.idx, text=text, lines=item.lines)
    processed_count += len(batch)
    logger.info("Переведено %s/%s", processed_count, total_count)
    if progress_callback:
        progress_callback(processed_count, total_count)
    return processed_count


def _translate_batches(
    prepare: list[PrepareToTranslateItem],
    translator: Translator,
    options: TranslationOptions,
    thread_count: int,
    batch_size: int,
    logger: logging.Logger,
    progress_callback: FileProgressCallback | None = None,
    *,
    thread_safe: bool = True,
) -> list[TranslatedItem]:
    total = len(prepare)
    if total == 0:
        return []
    translated: dict[int, TranslatedItem] = {}
    batch_size = max(1, int(batch_size or DEFAULT_BATCH_SIZE))
    thread_count = max(1, int(thread_count or DEFAULT_THREAD_COUNT))
    if not thread_safe:
        thread_count = 1

    processed = 0
    for start in range(0, total, batch_size * thread_count):
        chunk = prepare[start : start + batch_size * thread_count]
        batches = [chunk[index : index + batch_size] for index in range(0, len(chunk), batch_size)]
        for batch, response in _iter_batch_results(batches, translator, options, thread_count):
            processed = _apply_batch_result(
                batch,
                response,
                translated,
                processed,
                total,
                logger,
                progress_callback,
            )
    return [translated[index] for index in sorted(translated)]


def _parse_dialogs(origins: list[str], file_format: FileFormat):
    parsers = {
        FileFormat.ASS: parse_ass_dialogs,
        FileFormat.SRT: parse_srt_dialogs,
        FileFormat.VTT: parse_vtt_dialogs,
    }
    try:
        return parsers[file_format](origins)
    except KeyError as exc:
        raise TranslationError(f"Формат не поддерживается: {file_format}") from exc


def translate_subtitles(
    input_path: Path,
    output_path: Path,
    file_format: FileFormat,
    options: TranslationOptions,
    api: str,
    *,
    thread_count: int,
    batch_size: int,
    smart_split: bool,
    smart_split_settings: SmartSplitSettings | None = None,
    timeout: int,
    logger: logging.Logger,
    progress_callback: FileProgressCallback | None = None,
    overwrite: bool = True,
) -> None:
    """Переводит один файл после решения вопросов путей, кеша и перезаписи вызывающим кодом."""
    origins = split_lines(read_text(input_path))
    dialogs = _parse_dialogs(origins, file_format)
    prepare = build_prepare(dialogs, smart_split, smart_split_settings)
    analysis = analyse_lines(dialogs)

    translator_metadata = get_translator_metadata(api)
    translator = get_translator(api, timeout)
    translated_items = _translate_batches(
        prepare,
        translator,
        options,
        thread_count,
        batch_size,
        logger,
        progress_callback,
        thread_safe=translator_metadata.thread_safe,
    )
    translated_dialogs = build_translated_dialogs(translated_items, analysis)
    export_lines = build_export_lines(origins, file_format.value, dialogs, translated_dialogs)
    if not validate_cached_lines(export_lines, file_format):
        raise TranslationError("Сформированный файл субтитров не прошёл проверку структуры.")
    write_lines(output_path, export_lines, overwrite=overwrite)


def _translation_options(settings: ProcessingSettings, source_lang: str) -> TranslationOptions:
    return TranslationOptions(
        source_lang=source_lang,
        target_lang=settings.target_lang,
        api=settings.api,
        tld=settings.tld,
        request_delay_ms=settings.request_delay_ms,
        allow_cpu_fallback=settings.allow_cpu_fallback,
        model_path=settings.model_path,
        model_revision=settings.model_revision,
        worker_python_path=settings.worker_python_path,
        auto_download_model=settings.auto_download_model,
        agent_model=settings.agent_model,
    )


def _significant_settings(settings: ProcessingSettings) -> dict[str, object]:
    values: dict[str, object] = {
        "batch_size": settings.batch_size,
        "smart_split": settings.smart_split,
    }
    if settings.smart_split:
        split = settings.smart_split_settings or SmartSplitSettings()
        values["smart_split_settings"] = {
            "max_lines": split.max_lines,
            "max_words": split.max_words,
            "max_chars": split.max_chars,
            "max_gap_ms": split.max_gap_ms,
            "max_duration_ms": split.max_duration_ms,
        }
    if settings.api == "google":
        values["tld"] = settings.tld
    return values


@dataclass(frozen=True, slots=True)
class _LocalModelCacheState:
    content_fingerprint: str | None = None
    unavailable: bool = False


def _local_model_cache_state(
    settings: ProcessingSettings,
    metadata: TranslatorMetadata,
) -> _LocalModelCacheState:
    """Разрешает локальное состояние модели без сети и загрузки весов."""
    if not metadata.local:
        return _LocalModelCacheState()

    inspection_options = TranslationOptions(
        model_path=settings.model_path,
        model_revision=settings.model_revision,
        auto_download_model=False,
    )
    try:
        resolved_model = resolve_registered_model(settings.api, inspection_options)
    except OSError, RuntimeError:
        selected_path = settings.model_path or metadata.model_path
        revision = settings.model_revision or metadata.model_revision
        if selected_path is None or revision is None or metadata.model_id is None:
            return _LocalModelCacheState(unavailable=True)
        from sub_translate.utils.huggingface import inspect_model_directory

        try:
            inspection = inspect_model_directory(
                selected_path,
                model_id=metadata.model_id,
                revision=revision,
                required_files=metadata.required_files,
                required_file_groups=metadata.required_file_groups,
            )
        except OSError:
            return _LocalModelCacheState(unavailable=True)
        if not inspection.structurally_complete:
            return _LocalModelCacheState(unavailable=True)
        content_fingerprint = None if inspection.revision_verified else inspection.content_fingerprint
        return _LocalModelCacheState(content_fingerprint)

    if resolved_model.revision_verified:
        return _LocalModelCacheState()
    return _LocalModelCacheState(resolved_model.content_fingerprint)


def build_output_cache_identity(
    input_path: Path,
    file_format: FileFormat,
    settings: ProcessingSettings,
    *,
    source_lang: str | None = None,
) -> OutputCacheIdentity:
    """Строит полную подпись результата без загрузки адаптера или модели."""
    metadata = get_translator_metadata(settings.api)
    resolved_source = source_lang or settings.source_lang
    model_id = settings.agent_model if settings.api == "agent" else metadata.model_id
    model_revision = settings.model_revision or metadata.model_revision
    model_state = _local_model_cache_state(settings, metadata)
    return OutputCacheIdentity(
        source_path=input_path,
        file_format=file_format,
        source_language=resolved_source,
        target_language=settings.target_lang,
        translator_id=settings.api,
        model_id=model_id,
        model_revision=model_revision,
        model_content_fingerprint=model_state.content_fingerprint,
        settings=_significant_settings(settings),
        prompt_signatures=settings.prompt_signatures if settings.api == "agent" else {},
        local_model_unavailable=model_state.unavailable,
    )


def process_subtitle(
    input_path: Path,
    settings: ProcessingSettings,
    *,
    logger: logging.Logger,
    output_path: str | Path | None = None,
    format_arg: str | None = None,
    cache_path: Path = SUBTITLE_CACHE_FILE,
    cache_policy: CachePolicy = DEFAULT_CACHE_POLICY,
    progress_callback: FileProgressCallback | None = None,
) -> ProcessingResult:
    """Обрабатывает один файл через единые правила кеша, перевода и публикации."""
    resolved_input = input_path.expanduser().resolve()
    if not resolved_input.is_file():
        raise FileNotFoundError(f"Файл не найден: {resolved_input}")
    file_format = resolve_format(resolved_input, format_arg)
    source_lang, resolved_output = resolve_io_paths(
        resolved_input,
        output_path,
        settings.source_lang,
        settings.target_lang,
        settings.api,
        file_format,
    )
    if resolved_output == resolved_input:
        raise ValueError("Исходный и выходной пути не должны совпадать.")
    identity = build_output_cache_identity(
        resolved_input,
        file_format,
        settings,
        source_lang=source_lang,
    )
    if not settings.force:
        cache_status = restore_output_cache(
            identity,
            resolved_output,
            logger,
            cache_path=cache_path,
            policy=cache_policy,
        )
        if cache_status is CacheRestoreStatus.RESTORED:
            return ProcessingResult(resolved_input, resolved_output, ProcessingStatus.CACHED, file_format)
        if cache_status is CacheRestoreStatus.OUTPUT_EXISTS:
            return ProcessingResult(
                resolved_input,
                resolved_output,
                ProcessingStatus.OUTPUT_EXISTS,
                file_format,
            )

    translate_subtitles(
        resolved_input,
        resolved_output,
        file_format,
        _translation_options(settings, source_lang),
        settings.api,
        thread_count=settings.thread_count,
        batch_size=settings.batch_size,
        smart_split=settings.smart_split,
        smart_split_settings=settings.smart_split_settings,
        timeout=settings.timeout,
        logger=logger,
        progress_callback=progress_callback,
        overwrite=settings.force,
    )
    current_identity = build_output_cache_identity(
        resolved_input,
        file_format,
        settings,
        source_lang=source_lang,
    )
    if build_cache_fingerprint(current_identity) != build_cache_fingerprint(identity):
        logger.warning("Входные данные или модель изменились во время перевода; результат не добавлен в кеш.")
    else:
        store_output_cache(
            current_identity,
            resolved_output,
            logger,
            cache_path=cache_path,
            policy=cache_policy,
        )
    return ProcessingResult(resolved_input, resolved_output, ProcessingStatus.TRANSLATED, file_format)


def _error_result(input_path: Path, settings: ProcessingSettings, error: Exception) -> ProcessingResult:
    resolved_input = input_path.expanduser().resolve()
    try:
        file_format = resolve_format(resolved_input, None)
        _, output_path = resolve_io_paths(
            resolved_input,
            None,
            settings.source_lang,
            settings.target_lang,
            settings.api,
            file_format,
        )
    except OSError, ValueError:
        file_format = None
        output_path = None
    return ProcessingResult(
        resolved_input,
        output_path,
        ProcessingStatus.ERROR,
        file_format,
        type(error).__name__,
    )


def process_subtitle_batch(
    input_paths: Iterable[Path],
    settings: ProcessingSettings,
    *,
    logger: logging.Logger,
    cache_path: Path = SUBTITLE_CACHE_FILE,
    cache_policy: CachePolicy = DEFAULT_CACHE_POLICY,
    started_callback: BatchStartedCallback | None = None,
    progress_callback: BatchProgressCallback | None = None,
    result_callback: BatchResultCallback | None = None,
) -> list[ProcessingResult]:
    """Обрабатывает пачку последовательно и изолирует ошибку каждого файла."""
    paths = [Path(path) for path in input_paths]
    total = len(paths)
    results: list[ProcessingResult] = []
    for index, path in enumerate(paths, start=1):
        if started_callback:
            started_callback(path, index, total)
        file_progress = partial(progress_callback, path, index, total) if progress_callback else None
        try:
            result = process_subtitle(
                path,
                settings,
                logger=logger,
                cache_path=cache_path,
                cache_policy=cache_policy,
                progress_callback=file_progress,
            )
        except Exception as exc:
            logger.exception("Не удалось обработать файл %s (%s).", path.name, type(exc).__name__)
            result = _error_result(path, settings, exc)
        results.append(result)
        if result_callback:
            result_callback(result, index, total)
    return results


__all__ = [
    "API_ALIASES",
    "ProcessingResult",
    "ProcessingSettings",
    "ProcessingStatus",
    "build_output_cache_identity",
    "get_translator",
    "process_subtitle",
    "process_subtitle_batch",
    "resolve_api",
    "resolve_format",
    "resolve_io_paths",
    "translate_subtitles",
]
