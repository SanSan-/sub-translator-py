import pytest

from sub_translate.utils.translation_utils import (
    postprocess_translation,
    split_long_text,
    strip_source_wrapper,
    translate_text,
)


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("<SOURCE>Hello</SOURCE>", "Hello"),
        ("  <SOURCE>\n Hello world \n</SOURCE>  ", "Hello world"),
        ("Text without wrapper", "Text without wrapper"),
    ],
)
def test_strip_source_wrapper(source: str, expected: str) -> None:
    assert strip_source_wrapper(source) == expected


def test_postprocess_translation_normalizes_supported_markers() -> None:
    assert postprocess_translation("Text *value* \t\n2^3^") == "Text  *value*\n2^3"


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("12^34^56^", "12^3456^"),
        ("2^3^ 4^5^", "2^3 4^5"),
        ("2^^3^", "2^^3^"),
        ("²^3^", "²^3^"),
        ("٢^٣^", "٢^٣"),
    ],
)
def test_postprocess_translation_preserves_exponent_marker_semantics(source: str, expected: str) -> None:
    assert postprocess_translation(source) == expected


def test_postprocess_translation_handles_long_exponent_input() -> None:
    digits = "9" * 100_000

    assert postprocess_translation(f"{digits}^{digits}^") == f"{digits}^{digits}"


@pytest.mark.parametrize(
    ("source", "limit", "expected"),
    [
        ("One two. Three four.", 2, ["One two.", "Three four."]),
        ("one two three", 2, ["one two", "three"]),
        ("single", 0, ["single"]),
    ],
)
def test_split_long_text_preserves_order_and_respects_word_boundaries(
    source: str,
    limit: int,
    expected: list[str],
) -> None:
    assert split_long_text(source, limit, lambda value: len(value.split())) == expected


def test_translate_text_preserves_empty_lines_and_joins_translated_chunks() -> None:
    result = translate_text(
        "one two three\n\nfour",
        token_limit=2,
        token_counter=lambda value: len(value.split()),
        chunk_translator=str.upper,
        postprocess=None,
    )

    assert result == "ONE TWO THREE\n\nFOUR"
