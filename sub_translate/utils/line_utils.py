from __future__ import annotations

from typing import Dict, List

from sub_translate.constants import (
    BIG_NEW_LINE_SIGN,
    EMPTY_STRING,
    FIRST_GROUP,
    SPACE_SIGN,
    WEBVTT,
    NOT_VTT_ERROR,
    ZERO_INT_SIGN,
    correct_sort,
)
from sub_translate.dictionaries.bad_symbols import BAD_SYMBOLS
from sub_translate.dictionaries.filters import (
    only_end_symbols,
    without_dashes,
    without_non_end_symbols,
    without_word_and_dashes,
    without_word_count,
)
from sub_translate.dictionaries.regex import (
    ASS_COMMENTS_MASK,
    ASS_EFFECTS_MASK,
    BRACKET_MASK,
    COLON_MASK,
    COMMA_MASK,
    DASH_MASK,
    DOT_MASK,
    DOUBLE_SPACES_MASK,
    DRAW_MASK,
    EXCLAMATION_MARK_MASK,
    GOOD_END_SYMBOLS_MASK,
    ITALIAN_MASK,
    NEXT_LINE_MASK,
    NO_SPACE_NEXT_LINE_MASK,
    QUESTION_MARK_MASK,
    QUOTE_MASK,
    SEMICOLON_MASK,
    SRT_EFFECTS_MASK,
)
from sub_translate.enums import FileFormat
from sub_translate.models import (
    AnalysedDialog,
    AnalysedItem,
    AnalysedLine,
    AssSubtitlesItem,
    PrepareToTranslateItem,
    SrtSubtitlesItem,
    TranslatedDialogItem,
    TranslatedItem,
)
from sub_translate.utils.common_utils import is_empty, is_empty_array
from sub_translate.utils.validation_utils import (
    ass_separator,
    ass_validator,
    count_regexp_entry,
    srt_start_validator,
    srt_time_extract,
    srt_time_validator,
)


def replace_all_words(input_text: str, source: str, target: str) -> str:
    source_len = len(source)
    output = EMPTY_STRING
    pos = 0
    while True:
        match_pos = input_text.find(source, pos)
        if match_pos == -1:
            output += input_text[pos:]
            break
        output += input_text[pos:match_pos]
        output += target
        pos = match_pos + source_len
    return output


def replace_all(text: str, pattern, replacement: str) -> str:
    temp = str(text)
    while pattern.search(temp):
        temp = pattern.sub(replacement, temp)
    return temp


def format_line(line: str, dictionary: List[Dict[str, str]]) -> str:
    temp = str(line)
    for item in dictionary:
        temp = replace_all_words(temp, item["key"], item["val"])
    return temp


def clean_line(text: str) -> str:
    temp = replace_all(format_line(text, BAD_SYMBOLS), NO_SPACE_NEXT_LINE_MASK, FIRST_GROUP)
    temp = NEXT_LINE_MASK.sub(SPACE_SIGN, temp)
    temp = DRAW_MASK.sub(EMPTY_STRING, temp)
    temp = ASS_EFFECTS_MASK.sub(EMPTY_STRING, temp)
    temp = ASS_COMMENTS_MASK.sub(EMPTY_STRING, temp)
    temp = SRT_EFFECTS_MASK.sub(EMPTY_STRING, temp)
    temp = DOUBLE_SPACES_MASK.sub(SPACE_SIGN, temp)
    return temp.strip()


def parse_ass_dialogs(origins: List[str]) -> Dict[int, AssSubtitlesItem]:
    dialogs: Dict[int, AssSubtitlesItem] = {}
    for idx, text in enumerate(origins):
        if ass_validator(text):
            dialogs.update(ass_separator(idx, text))
    return dialogs


def parse_srt_dialogs(origins: List[str]) -> Dict[int, SrtSubtitlesItem]:
    i = 0
    while i < len(origins) and not srt_start_validator(origins[i]):
        i += 1
    dialogs: Dict[int, SrtSubtitlesItem] = {}
    while i < len(origins):
        line_index = int(origins[i]) if srt_start_validator(origins[i]) else 0
        temp: List[str] = []
        times: List[str] = []
        i += 1
        while i < len(origins) and not srt_start_validator(origins[i]):
            if srt_time_validator(origins[i]):
                times = srt_time_extract(origins[i])
            elif not is_empty(origins[i]):
                temp.append(origins[i])
            i += 1
        dialogs[line_index] = SrtSubtitlesItem(
            start_time=times[0] if times else None,
            end_time=times[1] if len(times) > 1 else None,
            text=(SPACE_SIGN + BIG_NEW_LINE_SIGN).join(temp),
        )
    return dialogs


def parse_vtt_dialogs(origins: List[str]) -> Dict[int, SrtSubtitlesItem]:
    if not origins or origins[0] != WEBVTT:
        raise ValueError(NOT_VTT_ERROR)
    i = 1
    dialogs: Dict[int, SrtSubtitlesItem] = {}
    while i < len(origins):
        line_index = len(dialogs) + 1
        temp: List[str] = []
        times: List[str] = []
        i += 1
        while i < len(origins) and clean_line(origins[i]) != EMPTY_STRING:
            if srt_time_validator(origins[i]):
                times = srt_time_extract(origins[i])
            elif not is_empty(origins[i]):
                temp.append(origins[i])
            i += 1
        if not is_empty_array(temp):
            dialogs[line_index] = SrtSubtitlesItem(
                start_time=times[0] if times else None,
                end_time=times[1] if len(times) > 1 else None,
                text=(SPACE_SIGN + BIG_NEW_LINE_SIGN).join(temp),
            )
    return dialogs


def build_prepare(
    dialogs: Dict[int, SrtSubtitlesItem], use_smart_dialog_splitter: bool = False
) -> List[PrepareToTranslateItem]:
    result: List[PrepareToTranslateItem] = []
    tempo_lines: List[int] = []
    tempo_text = EMPTY_STRING
    keys = correct_sort([int(key) for key in dialogs.keys()])
    for idx, key in enumerate(keys):
        tempo_lines.append(key)
        tempo_text += f" {dialogs[key].text or EMPTY_STRING}"
        if GOOD_END_SYMBOLS_MASK.search(dialogs[key].text or EMPTY_STRING) or not use_smart_dialog_splitter:
            to_translate = clean_line(tempo_text)
            if not is_empty(to_translate):
                result.append(PrepareToTranslateItem(idx=len(result), lines=tempo_lines, to_translate=to_translate))
            tempo_lines = []
            tempo_text = EMPTY_STRING
        elif idx == len(keys) - 1:
            to_translate = clean_line(tempo_text)
            if not is_empty(to_translate):
                result.append(PrepareToTranslateItem(idx=len(result), lines=tempo_lines, to_translate=to_translate))
    return result


def _calc_effect_index(line: str, match: str) -> int:
    before = clean_line(line[: line.index(match)])
    return 0 if is_empty(before) else len(before.split(SPACE_SIGN))


def build_effects(line: str) -> Dict[int, str]:
    result: Dict[int, str] = {}
    matches = ASS_EFFECTS_MASK.findall(line)
    for match in matches:
        result[_calc_effect_index(line, match)] = match
    srt_matches = SRT_EFFECTS_MASK.findall(line)
    for match in srt_matches:
        result[_calc_effect_index(line, match)] = match
    return result


def _count_symbols(text: str) -> AnalysedLine:
    return {
        "dotCount": count_regexp_entry(text, DOT_MASK),
        "commaCount": count_regexp_entry(text, COMMA_MASK),
        "quoteCount": count_regexp_entry(text, QUOTE_MASK),
        "bracketCount": count_regexp_entry(text, BRACKET_MASK),
        "dashCount": count_regexp_entry(text, DASH_MASK),
        "colonCount": count_regexp_entry(text, COLON_MASK),
        "semicolonCount": count_regexp_entry(text, SEMICOLON_MASK),
        "questionMarkCount": count_regexp_entry(text, QUESTION_MARK_MASK),
        "exclamationMarkCount": count_regexp_entry(text, EXCLAMATION_MARK_MASK),
    }


def analyse_line(dialog_line: str) -> AnalysedDialog:
    result: AnalysedDialog = {
        "wordCount": len(clean_line(dialog_line).split(SPACE_SIGN)),
        "dotCount": 0,
        "commaCount": 0,
        "quoteCount": 0,
        "bracketCount": 0,
        "dashCount": 0,
        "colonCount": 0,
        "semicolonCount": 0,
        "questionMarkCount": 0,
        "exclamationMarkCount": 0,
    }
    lines = NEXT_LINE_MASK.split(dialog_line)
    analysed: List[AnalysedLine] = []
    if DRAW_MASK.search(dialog_line):
        analysed.append({**result, "effects": {0: dialog_line}})
    else:
        for line in lines:
            raw_line = ITALIAN_MASK.sub(EMPTY_STRING, line)
            raw_line = DRAW_MASK.sub(EMPTY_STRING, raw_line)
            pure_line = clean_line(line)
            counted = _count_symbols(pure_line)
            analysed.append(
                {
                    **counted,
                    "wordCount": len(pure_line.split(SPACE_SIGN)) if pure_line else 0,
                    "effects": build_effects(raw_line)
                    if ASS_EFFECTS_MASK.search(raw_line) or SRT_EFFECTS_MASK.search(raw_line)
                    else {},
                }
            )
            result = {
                **result,
                "dotCount": result["dotCount"] + counted["dotCount"],
                "commaCount": result["commaCount"] + counted["commaCount"],
                "quoteCount": result["quoteCount"] + counted["quoteCount"],
                "bracketCount": result["bracketCount"] + counted["bracketCount"],
                "dashCount": result["dashCount"] + counted["dashCount"],
                "colonCount": result["colonCount"] + counted["colonCount"],
                "semicolonCount": result["semicolonCount"] + counted["semicolonCount"],
                "questionMarkCount": result["questionMarkCount"] + counted["questionMarkCount"],
                "exclamationMarkCount": result["exclamationMarkCount"] + counted["exclamationMarkCount"],
            }
    result["lines"] = analysed
    return result


def analyse_lines(dialogs: Dict[int, AssSubtitlesItem]) -> AnalysedItem:
    result: AnalysedItem = {}
    for key, dialog in dialogs.items():
        analysed = analyse_line(dialog.text or EMPTY_STRING)
        analysed_lines = analysed.get("lines", [])
        if (
            not is_empty_array(list(analysed_lines))
            or not (len(analysed_lines) == 1 and analysed_lines[0].get("wordCount") == 0)
            or (analysed and analysed.get("wordCount", 0) > 0)
        ):
            result[int(key)] = analysed
    return result


def _add_effects(words: List[str], effects: Dict[int, str]) -> None:
    for key in sorted(effects.keys()):
        if key < len(words):
            words[key] = effects[key] + words[key]
        elif words:
            words[-1] = words[-1] + effects[key]


def _any_less(from_obj: Dict[str, int], diff: Dict[str, int], filter_func=lambda key: True) -> bool:
    for key in filter(filter_func, from_obj.keys()):
        if from_obj.get(key, 0) < diff.get(key, 0):
            return True
    return False


def _all_zeros(obj: Dict[str, int], filter_func=lambda key: True) -> bool:
    total = 0
    for key in filter(filter_func, obj.keys()):
        value = obj.get(key)
        if value is None:
            continue
        try:
            total += int(value)
        except (TypeError, ValueError):
            continue
    return total == 0


def _split_chars_by_line(
    words: List[str],
    lines_per_word: int,
    lines: List[AnalysedLine],
    result: List[str],
) -> None:
    for idx, word in enumerate(words):
        characters = list(word)
        chars_per_lines = int((len(characters) + lines_per_word - 1) / lines_per_word)
        temp: List[str] = []
        for j in range(idx * lines_per_word, idx * lines_per_word + lines_per_word):
            cur = lines[j]
            if cur.get("wordCount", 0) > 0:
                head = [characters[0]]
                n = 1
                while (
                    _any_less(_count_symbols(EMPTY_STRING.join(head)), cur)
                    or (_all_zeros(cur, without_word_count) and n < chars_per_lines)
                ) and n < len(characters):
                    head.append(characters[n])
                    n += 1
                _add_effects(head, cur.get("effects", {}))
                temp.append(EMPTY_STRING.join(head))
                characters = characters[n:]
            else:
                temp.append(BIG_NEW_LINE_SIGN)
        result.append(BIG_NEW_LINE_SIGN.join(temp))


def _split_words_by_line(
    lines_count: int,
    lines: List[AnalysedLine],
    words: List[str],
    result: List[str],
) -> None:
    temp = list(words)
    for i in range(lines_count):
        cur = lines[i]
        if cur.get("wordCount", 0) > 0:
            head = [temp[0]]
            j = 1
            while (
                _any_less(_count_symbols(SPACE_SIGN.join(head)), cur, without_dashes)
                or (_all_zeros(cur, without_word_count) and j < cur.get("wordCount", 0))
                or (
                    _all_zeros(cur, without_word_and_dashes)
                    and _any_less(_count_symbols(SPACE_SIGN.join(head)), cur)
                )
            ) and cur.get("wordCount", 0) > 1 and j < len(temp):
                head.append(temp[j])
                j += 1
            _add_effects(head, cur.get("effects", {}))
            result.append(SPACE_SIGN.join(head))
            temp = temp[j:]
        else:
            result.append(EMPTY_STRING)


def _restore_line(text: str, analysis: AnalysedDialog) -> str:
    result: List[str] = []
    words = text.split(SPACE_SIGN)
    lines = analysis.get("lines", [])
    lines_count = len(lines)
    if lines_count > 1:
        if lines_count > len(words):
            lines_per_word = max(1, lines_count // max(1, len(words)))
            _split_chars_by_line(words, lines_per_word, lines, result)
        else:
            _split_words_by_line(lines_count, lines, words, result)
    else:
        if lines:
            _add_effects(words, lines[0].get("effects", {}))
        return SPACE_SIGN.join(words)
    return BIG_NEW_LINE_SIGN.join(result)


def build_translated_dialogs(
    translated: List[TranslatedItem],
    analysis: AnalysedItem,
) -> TranslatedDialogItem:
    result: TranslatedDialogItem = {}
    for item in translated:
        line_idxs = item.lines
        lines_count = len(line_idxs)
        text = clean_line(item.text)
        if lines_count > 1:
            words = text.split(SPACE_SIGN)
            for line_idx in line_idxs:
                cur = analysis.get(line_idx)
                if not cur:
                    continue
                temp = [words[0]] if words else [EMPTY_STRING]
                k = 1
                while (
                    _any_less(_count_symbols(SPACE_SIGN.join(temp)), cur, only_end_symbols)
                    or (_all_zeros(cur, without_word_count) and k < cur.get("wordCount", 0))
                    or (
                        _all_zeros(cur, without_non_end_symbols)
                        and _any_less(_count_symbols(SPACE_SIGN.join(temp)), cur)
                    )
                ) and cur.get("wordCount", 0) > 1 and k < len(words):
                    temp.append(words[k])
                    k += 1
                result[line_idx] = _restore_line(SPACE_SIGN.join(temp), cur)
                words = words[k:]
        else:
            if not line_idxs:
                continue
            cur = analysis.get(line_idxs[0])
            if not cur:
                continue
            result[line_idxs[0]] = _restore_line(text, cur)
    return result


def _build_ass_dialog_line(dialog: AssSubtitlesItem, text: str) -> str:
    return (
        f"Dialogue: {dialog.layer if dialog.layer is not None else ZERO_INT_SIGN},"
        f"{dialog.start_time or EMPTY_STRING},{dialog.end_time or EMPTY_STRING},"
        f"{dialog.style or EMPTY_STRING},{dialog.actor or EMPTY_STRING},"
        f"{dialog.margin_l if dialog.margin_l is not None else ZERO_INT_SIGN},"
        f"{dialog.margin_r if dialog.margin_r is not None else ZERO_INT_SIGN},"
        f"{dialog.margin_v if dialog.margin_v is not None else ZERO_INT_SIGN},"
        f"{dialog.effect or EMPTY_STRING},{text or EMPTY_STRING}"
    )


def build_export_lines(
    origin: List[str],
    file_format: str,
    dialogs: Dict[int, AssSubtitlesItem],
    translated_dialogs: TranslatedDialogItem,
) -> List[str]:
    result: List[str] = []
    if file_format == FileFormat.ASS.value:
        for idx, line in enumerate(origin):
            if idx in translated_dialogs:
                result.append(_build_ass_dialog_line(dialogs[idx], translated_dialogs[idx]))
            else:
                result.append(line)
    else:
        if file_format == FileFormat.VTT.value:
            result.append(WEBVTT)
            result.append(EMPTY_STRING)
        for key in correct_sort([int(k) for k in translated_dialogs.keys()]):
            if file_format == FileFormat.SRT.value:
                result.append(str(key))
            dialog = dialogs.get(key)
            result.append(
                f"{dialog.start_time if dialog else EMPTY_STRING} --> {dialog.end_time if dialog else EMPTY_STRING}"
            )
            for text in translated_dialogs[key].split(BIG_NEW_LINE_SIGN):
                result.append(text)
            result.append(EMPTY_STRING)
    return result


__all__ = [
    "format_line",
    "clean_line",
    "parse_ass_dialogs",
    "parse_srt_dialogs",
    "parse_vtt_dialogs",
    "build_prepare",
    "analyse_line",
    "analyse_lines",
    "build_translated_dialogs",
    "build_export_lines",
]