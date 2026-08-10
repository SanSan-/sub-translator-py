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
    ],
)
def test_split_long_text_preserves_order_and_respects_word_boundaries(
    source: str,
    limit: int,
    expected: list[str],
) -> None:
    assert split_long_text(source, limit, lambda value: len(value.split())) == expected


@pytest.mark.parametrize("length", [511, 512, 513])
def test_split_long_text_respects_limit_for_single_long_token(length: int) -> None:
    source = "x" * length

    chunks = split_long_text(source, 512, len)

    assert "".join(chunks) == source
    assert all(0 < len(chunk) <= 512 for chunk in chunks)


def test_split_long_text_preserves_unicode_token_without_spaces() -> None:
    source = "перевод🙂" * 100

    chunks = split_long_text(source, 37, len)

    assert "".join(chunks) == source
    assert all(0 < len(chunk) <= 37 for chunk in chunks)


def test_split_long_text_rejects_impossible_counter_limit() -> None:
    with pytest.raises(ValueError, match="Невозможно выделить"):
        split_long_text("abc", 1, lambda _value: 2)


@pytest.mark.parametrize("token_limit", [0, -1])
def test_split_long_text_rejects_non_positive_limit(token_limit: int) -> None:
    with pytest.raises(ValueError, match="положительным"):
        split_long_text("single", token_limit, len)


def test_translate_text_preserves_empty_lines_and_joins_translated_chunks() -> None:
    result = translate_text(
        "one two three\n\nfour",
        token_limit=2,
        token_counter=lambda value: len(value.split()),
        chunk_translator=str.upper,
        postprocess=None,
    )

    assert result == "ONE TWO THREE\n\nFOUR"
