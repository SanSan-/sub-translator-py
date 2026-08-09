from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from sub_translate import service
from sub_translate.enums import FileFormat
from sub_translate.models import TranslatedItem, TranslationOptions
from sub_translate.translators.base import TranslationError
from sub_translate.utils.line_utils import (
    analyse_lines,
    build_export_lines,
    build_prepare,
    build_translated_dialogs,
    parse_ass_dialogs,
    parse_ass_events,
    parse_srt_dialogs,
    parse_vtt_dialogs,
)
from sub_translate.utils.subtitle_cache import validate_cached_lines
from sub_translate.utils.validation_utils import (
    srt_time_validator,
    vtt_header_validator,
    vtt_time_validator,
)

LOGGER = logging.getLogger("test.format.acceptance")


def test_ass_preserves_document_fields_and_all_override_tags() -> None:
    dialogue = (
        "Dialogue: 7,9:59:59.99,10:00:00.01,Стиль верх,Анна,0012,0034,0056,"
        "Banner;5;0;20,{\\an8}{\\i1}Hello,\\Nworld.{\\i0}"
    )
    origins = [
        "[Script Info]",
        "Title: Проверка Unicode",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize",
        "Style: Стиль верх,Arial,48",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
        dialogue,
        "Comment: 1,0:00:00.00,0:00:01.00,Стиль верх,Анна,0,0,0,,Не менять",
    ]
    dialogue_index = origins.index(dialogue)
    dialogs = parse_ass_dialogs(origins)
    events = parse_ass_events(origins)
    dialog = dialogs[dialogue_index]
    comment = events[dialogue_index + 1]

    assert (
        dialog.layer,
        dialog.start_time,
        dialog.end_time,
        dialog.style,
        dialog.actor,
        dialog.margin_l,
        dialog.margin_r,
        dialog.margin_v,
        dialog.effect,
    ) == (7, "9:59:59.99", "10:00:00.01", "Стиль верх", "Анна", 12, 34, 56, "Banner;5;0;20")
    assert dialog.event_type == "Dialogue"
    assert dialog.translatable is True
    assert comment.event_type == "Comment"
    assert comment.translatable is False
    assert comment.text == "Не менять"
    assert dialogue_index + 1 not in dialogs

    prepare = build_prepare(dialogs)
    translated = [
        TranslatedItem(
            idx=prepare[0].idx,
            text="Здравствуйте, мир.",
            lines=prepare[0].lines,
        )
    ]
    restored = build_translated_dialogs(translated, analyse_lines(dialogs))
    exported = build_export_lines(origins, FileFormat.ASS.value, dialogs, restored)

    assert exported[:dialogue_index] == origins[:dialogue_index]
    assert exported[dialogue_index + 1 :] == origins[dialogue_index + 1 :]
    original_fields = dialogue.split(",", 9)
    exported_fields = exported[dialogue_index].split(",", 9)
    assert exported_fields[:9] == original_fields[:9]
    assert exported_fields[9] == "{\\an8}{\\i1}Здравствуйте,\\Nмир.{\\i0}"


def test_srt_preserves_identifiers_timings_multiline_unicode_and_numeric_text() -> None:
    origins = [
        "7",
        "00:00:00,000 --> 00:00:02,345",
        "Привет, мир 🌍",
        "2026",
        "第二行",
        "",
        "42",
        "99:59:58,999 --> 99:59:59,999",
        "До свидания",
        "",
    ]
    dialogs = parse_srt_dialogs(origins)

    assert list(dialogs) == [7, 42]
    assert dialogs[7].start_time == "00:00:00,000"
    assert dialogs[7].end_time == "00:00:02,345"
    assert dialogs[7].text == "Привет, мир 🌍\\N2026\\N第二行"

    translated = {7: dialogs[7].text or "", 42: "До свидания"}
    exported = build_export_lines(origins, FileFormat.SRT.value, dialogs, translated)

    assert exported == [
        "7",
        "00:00:00,000 --> 00:00:02,345",
        "Привет, мир 🌍",
        "2026",
        "第二行",
        "",
        "42",
        "99:59:58,999 --> 99:59:59,999",
        "До свидания",
        "",
    ]
    assert validate_cached_lines(exported, FileFormat.SRT)


def test_srt_preserves_source_order_and_duplicate_identifiers() -> None:
    origins = [
        "42",
        "00:00:00,000 --> 00:00:01,000",
        "Первый",
        "",
        "7",
        "00:00:01,000 --> 00:00:02,000",
        "Второй",
        "",
        "42",
        "00:00:02,000 --> 00:00:03,000",
        "Третий",
        "",
    ]

    dialogs = parse_srt_dialogs(origins)
    keys = list(dialogs)
    translated = {
        keys[0]: "Один",
        keys[1]: "Два",
        keys[2]: "Три",
    }

    assert keys[:2] == [42, 7]
    assert keys[2] < 0
    assert [dialog.cue_id for dialog in dialogs.values()] == ["42", "7", "42"]
    assert build_export_lines(origins, FileFormat.SRT.value, dialogs, translated) == [
        "42",
        "00:00:00,000 --> 00:00:01,000",
        "Один",
        "",
        "7",
        "00:00:01,000 --> 00:00:02,000",
        "Два",
        "",
        "42",
        "00:00:02,000 --> 00:00:03,000",
        "Три",
        "",
    ]


def test_vtt_preserves_header_identifier_timing_and_cue_settings() -> None:
    timing = "00:00.000 --> 00:02.500 line:10%,start position:20%,line-left size:35% align:start vertical:rl"
    origins = [
        "WEBVTT - Русские субтитры",
        "X-TIMESTAMP-MAP=LOCAL:00:00:00.000,MPEGTS:900000",
        "",
        "вступление-α",
        timing,
        "Привет",
        "мир 🌍",
        "",
    ]
    dialogs = parse_vtt_dialogs(origins)
    dialog = dialogs[1]

    assert dialog.cue_id == "вступление-α"
    assert dialog.start_time == "00:00.000"
    assert dialog.end_time == timing.split(" --> ", 1)[1]
    assert dialog.text == "Привет\\Nмир 🌍"

    exported = build_export_lines(
        origins,
        FileFormat.VTT.value,
        dialogs,
        {1: dialog.text or ""},
    )

    assert exported == origins
    assert validate_cached_lines(exported, FileFormat.VTT)


def test_vtt_rules_do_not_weaken_srt_or_accept_unknown_settings() -> None:
    vtt_timing = "00:00.000 --> 00:02.500 align:start"

    assert vtt_header_validator("WEBVTT - Заголовок")
    assert not vtt_header_validator("WEBVTT --> неверный заголовок")
    assert not vtt_header_validator("webvtt")
    assert vtt_time_validator(vtt_timing)
    assert not srt_time_validator(vtt_timing)
    assert not vtt_time_validator("00:00.000 --> 00:02.500 unknown:value")


class _BrokenTranslator:
    name = "broken"

    def __init__(self, response: list[str]) -> None:
        self.response = response

    def translate_batch(self, _texts: list[str], _options: TranslationOptions) -> list[str]:
        return self.response

    def unload(self) -> None:
        return None


def _write_source_srt(path: Path) -> None:
    path.write_text(
        "1\n00:00:00,000 --> 00:00:01,000\nHello\n",
        encoding="utf-8",
        newline="",
    )


def test_mismatched_translator_response_does_not_publish_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "исходник.srt"
    output = tmp_path / "результат.srt"
    _write_source_srt(source)
    monkeypatch.setattr(service, "get_translator", lambda *_args: _BrokenTranslator([]))
    monkeypatch.setattr(service, "get_translator_metadata", lambda *_args: SimpleNamespace(thread_safe=True))
    options = TranslationOptions(source_lang="en", target_lang="ru")

    with pytest.raises(TranslationError, match="не совпадает с размером пачки"):
        service.translate_subtitles(
            source,
            output,
            FileFormat.SRT,
            options,
            "broken",
            thread_count=1,
            batch_size=1,
            smart_split=False,
            timeout=30,
            logger=LOGGER,
        )

    assert not output.exists()


def test_invalid_translator_text_preserves_existing_output_atomically(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "исходник.srt"
    output = tmp_path / "результат.srt"
    _write_source_srt(source)
    output.write_text("результат владельца", encoding="utf-8", newline="")
    monkeypatch.setattr(service, "get_translator", lambda *_args: _BrokenTranslator([""]))
    monkeypatch.setattr(service, "get_translator_metadata", lambda *_args: SimpleNamespace(thread_safe=True))
    options = TranslationOptions(source_lang="en", target_lang="ru")

    with pytest.raises(TranslationError, match="пустой перевод для непустого элемента"):
        service.translate_subtitles(
            source,
            output,
            FileFormat.SRT,
            options,
            "broken",
            thread_count=1,
            batch_size=1,
            smart_split=False,
            timeout=30,
            logger=LOGGER,
            overwrite=True,
        )

    assert output.read_text(encoding="utf-8") == "результат владельца"
    assert list(tmp_path.glob(".*.tmp")) == []
