from __future__ import annotations

import argparse
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from sub_translate.translators.registry import TranslatorMetadata
from tools import benchmark_translators as benchmark

_REPORT_NAMES = (
    "nllb-600m-en-ru.json",
    "nllb-600m-ru-en.json",
    "translategemma-en-ru.json",
    "translategemma-ru-en.json",
)
_QUALITY_METRIC_NAMES = (
    "required_term_total",
    "required_term_matches",
    "required_term_preservation_percent",
    "required_term_examples",
    "required_term_examples_fully_matched",
    "line_break_examples",
    "line_break_examples_matched",
    "line_break_preservation_percent",
)


def test_select_evenly_is_deterministic() -> None:
    selected, indexes = benchmark._select_evenly(list(range(10)), 4)

    assert selected == [0, 2, 5, 7]
    assert indexes == [0, 2, 5, 7]


def test_score_is_perfect_for_reference_translation() -> None:
    metrics = benchmark._score(["Привет, мир!"], ["Привет, мир!"])

    assert metrics["bleu"] == pytest.approx(100.0)
    assert metrics["chrf_pp"] == pytest.approx(100.0)
    assert metrics["empty_translations"] == 0


def test_process_tree_rss_includes_live_children() -> None:
    parent = Mock()
    parent.memory_info.return_value.rss = 100
    child = Mock()
    child.memory_info.return_value.rss = 40
    disappeared = Mock()
    disappeared.memory_info.side_effect = benchmark.psutil.NoSuchProcess(123)
    parent.children.return_value = [child, disappeared]

    assert benchmark._process_tree_rss(parent) == 140


def test_nvidia_smi_used_bytes_parses_first_gpu(monkeypatch) -> None:
    monkeypatch.setattr(
        benchmark.subprocess,
        "run",
        lambda *_args, **_kwargs: Mock(returncode=0, stdout="512\n768\n"),
    )

    assert benchmark._nvidia_smi_used_bytes() == 512 * 1024 * 1024


def test_load_json_corpus_preserves_newlines_and_uses_stable_selection(tmp_path: Path) -> None:
    corpus_path = tmp_path / "corpus.json"
    corpus_path.write_text(
        json.dumps(
            {
                "name": "subtitle-quality-v1",
                "directions": {
                    "en-ru": [
                        {"source": "One", "reference": "Один"},
                        {
                            "source": "Two\nnow",
                            "reference": "Два\nсейчас",
                            "required_terms": ["сейчас"],
                        },
                        {"source": "Three", "reference": "Три"},
                    ]
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    selection = benchmark.load_corpus("ignored", "en-ru", 2, corpus_path)

    assert [sample.source for sample in selection.samples] == ["One", "Two\nnow"]
    assert [sample.reference for sample in selection.samples] == ["Один", "Два\nсейчас"]
    assert selection.samples[1].required_terms == ("сейчас",)
    assert selection.indexes == (0, 1)
    assert selection.name == "subtitle-quality-v1"


def test_load_json_corpus_rejects_repeated_required_terms(tmp_path: Path) -> None:
    corpus_path = tmp_path / "corpus.json"
    corpus_path.write_text(
        json.dumps(
            {
                "name": "fixture",
                "directions": {
                    "en-ru": [
                        {
                            "source": "API",
                            "reference": "API",
                            "required_terms": ["API", "api"],
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="повторы"):
        benchmark.load_corpus("ignored", "en-ru", 1, corpus_path)


def test_quality_metrics_measure_terms_and_semantic_line_breaks() -> None:
    samples = (
        benchmark.CorpusSample(
            source="Names",
            reference="Анна Францева и API",
            required_terms=("Анна Францева", "API"),
        ),
        benchmark.CorpusSample(
            source="First\nSecond",
            reference="Первая\nВторая",
            required_terms=("Вторая",),
        ),
    )

    metrics, checks = benchmark._quality_metrics(
        ["анна францева без сокращения", " \nВторая"],
        samples,
    )

    assert metrics == {
        "required_term_total": 3,
        "required_term_matches": 2,
        "required_term_preservation_percent": pytest.approx(200 / 3),
        "required_term_examples": 2,
        "required_term_examples_fully_matched": 1,
        "line_break_examples": 1,
        "line_break_examples_matched": 0,
        "line_break_preservation_percent": 0.0,
    }
    assert checks[0]["missing_required_terms"] == ["API"]
    assert checks[0]["line_breaks_preserved"] is None
    assert checks[1]["missing_required_terms"] == []
    assert checks[1]["line_breaks_preserved"] is False


def test_run_records_reproducible_local_report(tmp_path: Path, monkeypatch) -> None:
    metadata = TranslatorMetadata(
        id="fake-local",
        display_name="Имитация",
        aliases=(),
        local=True,
        thread_safe=False,
        model_id="example/model",
        model_revision="a" * 40,
        model_path=tmp_path / "model",
        supported_directions=(("en", "ru"),),
    )
    unloaded = False

    class FakeTranslator:
        @staticmethod
        def translate_batch(texts, _options):
            return ["Один" if text == "One" else "Два" for text in texts]

        @staticmethod
        def unload() -> None:
            nonlocal unloaded
            unloaded = True

    monkeypatch.setattr(benchmark, "get_translator_metadata", lambda _identifier: metadata)
    monkeypatch.setattr(benchmark, "create_translator", lambda _identifier, timeout: FakeTranslator())
    monkeypatch.setattr(
        benchmark,
        "load_corpus",
        lambda _test_set, _direction, _limit, _corpus: benchmark.CorpusSelection(
            name="fixture",
            samples=(
                benchmark.CorpusSample("One", "Один", ("Один",)),
                benchmark.CorpusSample("Two", "Два"),
            ),
            indexes=(1, 3),
        ),
    )
    args = argparse.Namespace(
        translator="fake-local",
        direction="en-ru",
        model_path=tmp_path / "model",
        model_revision=None,
        worker_python_path=tmp_path / "worker-python.exe",
        test_set="fixture",
        corpus=None,
        limit=2,
        batch_size=1,
        timeout=30,
        output=tmp_path / "report.json",
    )

    report = benchmark.run(args)

    assert report["schema_version"] == benchmark.BENCHMARK_SCHEMA_VERSION
    assert report["metrics"]["chrf_pp"] == pytest.approx(100.0)
    assert report["metrics"]["required_term_preservation_percent"] == pytest.approx(100.0)
    assert report["sample_indexes"] == (1, 3)
    assert report["test_set"] == "fixture"
    assert report["translator"]["model_path"] == "<local-model-path>"
    assert report["requested_model_path"] == "<local-model-path>"
    assert report["worker_python_path"] == "<worker-python-path>"
    assert str(tmp_path) not in json.dumps(report, ensure_ascii=False)
    assert report["items"][0]["checks"]["missing_required_terms"] == []
    assert unloaded is True
    json.dumps(report, ensure_ascii=False)


@pytest.mark.parametrize("report_name", _REPORT_NAMES)
def test_saved_report_matches_current_quality_schema(report_name: str) -> None:
    report_path = Path(__file__).parents[1] / "docs" / "benchmarks" / "results" / report_name
    raw_report = report_path.read_bytes()
    report = json.loads(raw_report.decode("utf-8"))
    samples = tuple(
        benchmark.CorpusSample(
            source=item["source"],
            reference=item["reference"],
            required_terms=tuple(item["checks"]["required_terms"]),
        )
        for item in report["items"]
    )
    selection = benchmark.CorpusSelection(
        name=report["test_set"],
        samples=samples,
        indexes=tuple(report["sample_indexes"]),
    )
    quality_metrics, checks = benchmark._quality_metrics(
        [item["translation"] for item in report["items"]],
        samples,
    )

    assert not raw_report.startswith(b"\xef\xbb\xbf")
    assert report["schema_version"] == benchmark.BENCHMARK_SCHEMA_VERSION
    assert report["dataset_sha256"] == benchmark._dataset_digest(selection)
    assert {name: report["metrics"][name] for name in _QUALITY_METRIC_NAMES} == quality_metrics
    assert [item["checks"] for item in report["items"]] == checks
