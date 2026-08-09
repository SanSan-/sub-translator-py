from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import threading
import time
import unicodedata
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psutil
import sacrebleu
import torch
from sacrebleu.dataset import DATASETS
from sacrebleu.metrics import BLEU, CHRF

from sub_translate.constants import BASE_DIR
from sub_translate.models import TranslationOptions
from sub_translate.translators.registry import create_translator, get_translator_metadata
from sub_translate.utils.io_utils import atomic_write_json, configure_utf8_stdio

DEFAULT_TEST_SET = "wmt20"
DEFAULT_LIMIT = 100
DEFAULT_BATCH_SIZE = 4
DEFAULT_TIMEOUT_SECONDS = 1_800
BENCHMARK_SCHEMA_VERSION = 3
GPU_SAMPLE_INTERVAL_SECONDS = 0.5


@dataclass(frozen=True, slots=True)
class ResourceSnapshot:
    wall_seconds: float
    peak_rss_bytes: int
    peak_cuda_allocated_bytes: int
    peak_cuda_reserved_bytes: int
    peak_gpu_used_delta_bytes: int


@dataclass(frozen=True, slots=True)
class CorpusSample:
    source: str
    reference: str
    required_terms: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class CorpusSelection:
    name: str
    samples: tuple[CorpusSample, ...]
    indexes: tuple[int, ...]


class ProcessMemoryMonitor:
    """Измеряет память приложения, его дочерних процессов и видеокарты."""

    def __init__(self, interval_seconds: float = 0.05) -> None:
        self._interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.peak_rss_bytes = 0
        self.peak_gpu_used_delta_bytes = 0
        self._baseline_gpu_used_bytes: int | None = None
        self._next_gpu_sample = 0.0

    def __enter__(self) -> ProcessMemoryMonitor:
        process = psutil.Process()
        self._baseline_gpu_used_bytes = _gpu_used_bytes()
        self._next_gpu_sample = time.monotonic()

        def sample() -> None:
            while not self._stop.wait(self._interval_seconds):
                self._sample(process)

        self._sample(process)
        self._thread = threading.Thread(target=sample, name="benchmark-memory", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, _exc_type: object, _exc: object, _traceback: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        self._sample(psutil.Process(), force_gpu=True)

    def _sample(self, process: psutil.Process, *, force_gpu: bool = False) -> None:
        self.peak_rss_bytes = max(self.peak_rss_bytes, _process_tree_rss(process))
        now = time.monotonic()
        if not force_gpu and now < self._next_gpu_sample:
            return
        self._next_gpu_sample = now + GPU_SAMPLE_INTERVAL_SECONDS
        current_gpu_used = _gpu_used_bytes()
        if current_gpu_used is None or self._baseline_gpu_used_bytes is None:
            return
        self.peak_gpu_used_delta_bytes = max(
            self.peak_gpu_used_delta_bytes,
            current_gpu_used - self._baseline_gpu_used_bytes,
        )


def _process_tree_rss(process: psutil.Process) -> int:
    processes = [process]
    with suppress(psutil.Error):
        processes.extend(process.children(recursive=True))
    total = 0
    for item in processes:
        try:
            total += item.memory_info().rss
        except psutil.Error:
            continue
    return total


def _gpu_used_bytes() -> int | None:
    if not torch.cuda.is_available():
        return None
    nvidia_smi_value = _nvidia_smi_used_bytes()
    if nvidia_smi_value is not None:
        return nvidia_smi_value
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info()
    except RuntimeError, OSError:
        return None
    return int(total_bytes - free_bytes)


def _nvidia_smi_used_bytes() -> int | None:
    creation_flags = int(getattr(subprocess, "CREATE_NO_WINDOW", 0))
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=2,
            creationflags=creation_flags,
        )
    except OSError, subprocess.SubprocessError:
        return None
    if result.returncode != 0:
        return None
    first_line = result.stdout.splitlines()[0].strip() if result.stdout else ""
    try:
        used_mib = int(first_line)
    except ValueError:
        return None
    return used_mib * 1024 * 1024


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Сравнение локального переводчика на закреплённом наборе")
    parser.add_argument("--translator", required=True, help="Идентификатор локального переводчика")
    parser.add_argument("--direction", required=True, choices=("en-ru", "ru-en"))
    parser.add_argument("--model-path", type=Path, help="Каталог уже загруженной модели")
    parser.add_argument("--model-revision", help="Закреплённая ревизия модели")
    parser.add_argument("--worker-python-path", type=Path, help="Python изолированного процесса модели")
    parser.add_argument("--test-set", default=DEFAULT_TEST_SET)
    parser.add_argument("--corpus", type=Path, help="Переносимый JSON-набор вместо SacreBLEU")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--output", type=Path, required=True, help="Выходной JSON в UTF-8 без BOM")
    return parser


def _select_evenly(values: list[Any], limit: int) -> tuple[list[Any], list[int]]:
    if limit <= 0:
        raise ValueError("Предел числа примеров должен быть положительным.")
    if limit >= len(values):
        return list(values), list(range(len(values)))
    indexes = [(index * len(values)) // limit for index in range(limit)]
    return [values[index] for index in indexes], indexes


def load_corpus(
    test_set: str,
    direction: str,
    limit: int,
    corpus_path: Path | None = None,
) -> CorpusSelection:
    if corpus_path is not None:
        return _load_json_corpus(corpus_path, direction, limit)
    try:
        dataset = DATASETS[test_set]
    except KeyError as exc:
        raise ValueError(f"Неизвестный набор SacreBLEU: {test_set}") from exc
    sources = list(dataset.source(direction))
    reference_rows = [list(row) for row in dataset.references(direction)]
    if len(sources) != len(reference_rows) or any(len(row) != 1 for row in reference_rows):
        raise RuntimeError("Набор сравнения должен содержать ровно один эталон для каждой строки.")
    samples = [
        CorpusSample(source=source, reference=row[0]) for source, row in zip(sources, reference_rows, strict=True)
    ]
    selected, indexes = _select_evenly(samples, limit)
    return CorpusSelection(name=test_set, samples=tuple(selected), indexes=tuple(indexes))


def _load_json_corpus(
    corpus_path: Path,
    direction: str,
    limit: int,
) -> CorpusSelection:
    try:
        payload = json.loads(corpus_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Не удалось прочитать набор сравнения: {corpus_path}") from exc
    corpus_name = payload.get("name") if isinstance(payload, Mapping) else None
    if not isinstance(corpus_name, str) or not corpus_name.strip():
        raise ValueError("Пользовательский набор должен содержать непустое имя.")
    directions = payload.get("directions") if isinstance(payload, Mapping) else None
    rows = directions.get(direction) if isinstance(directions, Mapping) else None
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"В наборе нет непустого направления {direction}.")
    samples: list[CorpusSample] = []
    for row in rows:
        source = row.get("source") if isinstance(row, Mapping) else None
        reference = row.get("reference") if isinstance(row, Mapping) else None
        if not isinstance(source, str) or not isinstance(reference, str) or not source or not reference:
            raise ValueError("Каждая строка набора должна содержать непустые source и reference.")
        samples.append(
            CorpusSample(
                source=source,
                reference=reference,
                required_terms=_required_terms(row),
            )
        )
    selected, indexes = _select_evenly(samples, limit)
    return CorpusSelection(
        name=corpus_name.strip(),
        samples=tuple(selected),
        indexes=tuple(indexes),
    )


def _required_terms(row: Mapping[str, Any]) -> tuple[str, ...]:
    value = row.get("required_terms", [])
    if not isinstance(value, list) or any(not isinstance(term, str) or not term.strip() for term in value):
        raise ValueError("required_terms должен быть списком непустых строк.")
    terms = tuple(term.strip() for term in value)
    normalized = tuple(_normalize_for_match(term) for term in terms)
    if len(set(normalized)) != len(normalized):
        raise ValueError("required_terms не должен содержать повторы.")
    return terms


def _dataset_digest(selection: CorpusSelection) -> str:
    payload = json.dumps(
        {
            "indexes": selection.indexes,
            "samples": [asdict(sample) for sample in selection.samples],
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _translate(
    translator_id: str,
    sources: list[str],
    options: TranslationOptions,
    batch_size: int,
    timeout: int,
) -> tuple[list[str], ResourceSnapshot]:
    if batch_size <= 0:
        raise ValueError("Размер пачки должен быть положительным.")
    if timeout <= 0:
        raise ValueError("Таймаут должен быть положительным.")
    translator = create_translator(translator_id, timeout=timeout)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    translations: list[str] = []
    monitor = ProcessMemoryMonitor()
    try:
        with monitor:
            for start in range(0, len(sources), batch_size):
                batch = sources[start : start + batch_size]
                translated = translator.translate_batch(batch, options)
                if len(translated) != len(batch):
                    raise RuntimeError("Переводчик вернул другое число строк.")
                translations.extend(translated)
    finally:
        translator.unload()
    wall_seconds = time.perf_counter() - started
    cuda_allocated = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    cuda_reserved = torch.cuda.max_memory_reserved() if torch.cuda.is_available() else 0
    return translations, ResourceSnapshot(
        wall_seconds=wall_seconds,
        peak_rss_bytes=monitor.peak_rss_bytes,
        peak_cuda_allocated_bytes=cuda_allocated,
        peak_cuda_reserved_bytes=cuda_reserved,
        peak_gpu_used_delta_bytes=monitor.peak_gpu_used_delta_bytes,
    )


def _score(translations: list[str], references: list[str]) -> dict[str, Any]:
    bleu = BLEU(tokenize="13a")
    chrf = CHRF(word_order=2)
    bleu_result = bleu.corpus_score(translations, [references])
    chrf_result = chrf.corpus_score(translations, [references])
    return {
        "bleu": bleu_result.score,
        "bleu_signature": str(bleu.get_signature()),
        "chrf_pp": chrf_result.score,
        "chrf_signature": str(chrf.get_signature()),
        "empty_translations": sum(not value.strip() for value in translations),
    }


def _quality_metrics(
    translations: list[str],
    samples: tuple[CorpusSample, ...],
) -> tuple[dict[str, int | float | None], list[dict[str, Any]]]:
    if len(translations) != len(samples):
        raise ValueError("Число переводов не совпадает с числом примеров набора.")
    checks = [_quality_check(translation, sample) for translation, sample in zip(translations, samples, strict=True)]
    required_term_total = sum(len(sample.required_terms) for sample in samples)
    missing_term_total = sum(len(check["missing_required_terms"]) for check in checks)
    line_break_checks = [check for check in checks if check["line_breaks_preserved"] is not None]
    line_break_matches = sum(check["line_breaks_preserved"] is True for check in line_break_checks)
    metrics: dict[str, int | float | None] = {
        "required_term_total": required_term_total,
        "required_term_matches": required_term_total - missing_term_total,
        "required_term_preservation_percent": _percent(
            required_term_total - missing_term_total,
            required_term_total,
        ),
        "required_term_examples": sum(bool(sample.required_terms) for sample in samples),
        "required_term_examples_fully_matched": sum(
            bool(sample.required_terms) and not check["missing_required_terms"]
            for sample, check in zip(samples, checks, strict=True)
        ),
        "line_break_examples": len(line_break_checks),
        "line_break_examples_matched": line_break_matches,
        "line_break_preservation_percent": _percent(line_break_matches, len(line_break_checks)),
    }
    return metrics, checks


def _quality_check(translation: str, sample: CorpusSample) -> dict[str, Any]:
    normalized_translation = _normalize_for_match(translation)
    missing_terms = [
        term for term in sample.required_terms if _normalize_for_match(term) not in normalized_translation
    ]
    expected_line_breaks = sample.reference.count("\n")
    actual_line_breaks = translation.count("\n")
    has_only_nonempty_lines = all(line.strip() for line in translation.split("\n"))
    return {
        "required_terms": list(sample.required_terms),
        "missing_required_terms": missing_terms,
        "expected_line_breaks": expected_line_breaks,
        "actual_line_breaks": actual_line_breaks,
        "line_breaks_preserved": (
            actual_line_breaks == expected_line_breaks and has_only_nonempty_lines if expected_line_breaks else None
        ),
    }


def _normalize_for_match(value: str) -> str:
    return unicodedata.normalize("NFKC", value).casefold()


def _percent(matches: int, total: int) -> float | None:
    return (matches / total) * 100 if total else None


def _environment() -> dict[str, Any]:
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": importlib.metadata.version("transformers"),
        "sacrebleu": sacrebleu.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "gpu": gpu_name,
    }


def _metadata_payload(metadata: Any) -> dict[str, Any]:
    payload = asdict(metadata)
    model_path = payload.get("model_path")
    if isinstance(model_path, Path):
        payload["model_path"] = _portable_model_path(model_path)
    return payload


def _portable_model_path(model_path: Path | None) -> str | None:
    """Не записывает путь машины оператора в переносимый отчёт."""
    if model_path is None:
        return None
    resolved = model_path.expanduser().resolve()
    try:
        relative = resolved.relative_to(BASE_DIR.resolve())
    except ValueError:
        return "<local-model-path>"
    return f"<repo-root>/{relative.as_posix()}"


def run(args: argparse.Namespace) -> dict[str, Any]:
    metadata = get_translator_metadata(args.translator)
    if not metadata.local:
        raise ValueError("Сравнение допускает только явно выбранный локальный переводчик.")
    selection = load_corpus(
        args.test_set,
        args.direction,
        args.limit,
        args.corpus,
    )
    sources = [sample.source for sample in selection.samples]
    references = [sample.reference for sample in selection.samples]
    source_lang, target_lang = args.direction.split("-", 1)
    options = TranslationOptions(
        source_lang=source_lang,
        target_lang=target_lang,
        api=metadata.id,
        allow_cpu_fallback=False,
        model_path=args.model_path,
        model_revision=args.model_revision or metadata.model_revision,
        auto_download_model=False,
        worker_python_path=args.worker_python_path,
    )
    translations, resources = _translate(
        metadata.id,
        sources,
        options,
        args.batch_size,
        args.timeout,
    )
    quality_metrics, quality_checks = _quality_metrics(translations, selection.samples)
    report = {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "translator": _metadata_payload(metadata),
        "requested_model_path": _portable_model_path(args.model_path),
        "worker_python_path": _portable_worker_path(args.worker_python_path),
        "test_set": selection.name,
        "corpus": _portable_corpus_path(args.corpus),
        "direction": args.direction,
        "sample_indexes": selection.indexes,
        "sample_count": len(sources),
        "dataset_sha256": _dataset_digest(selection),
        "metrics": {**_score(translations, references), **quality_metrics},
        "resources": asdict(resources),
        "environment": _environment(),
        "items": [
            {
                "index": index,
                "source": sample.source,
                "reference": sample.reference,
                "translation": translation,
                "checks": checks,
            }
            for index, sample, translation, checks in zip(
                selection.indexes,
                selection.samples,
                translations,
                quality_checks,
                strict=True,
            )
        ],
    }
    return report


def _portable_worker_path(worker_python_path: Path | None) -> str | None:
    if worker_python_path is None:
        return None
    return "<worker-python-path>"


def _portable_corpus_path(corpus_path: Path | None) -> str | None:
    if corpus_path is None:
        return None
    resolved = corpus_path.expanduser().resolve()
    try:
        relative = resolved.relative_to(BASE_DIR.resolve())
    except ValueError:
        return "<local-corpus-path>"
    return f"<repo-root>/{relative.as_posix()}"


def main() -> int:
    configure_utf8_stdio()
    args = build_parser().parse_args()
    try:
        report = run(args)
        atomic_write_json(args.output, report)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Ошибка сравнения: {exc}", file=sys.stderr)
        return 1
    print(f"Отчёт сохранён: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
