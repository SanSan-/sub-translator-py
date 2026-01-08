from __future__ import annotations

import re
from typing import Dict, List

from sub_translate.constants import (
    BIG_NEW_LINE_SIGN,
    EMPTY_STRING,
    FIRST_GROUP,
    SPACE_SIGN,
    WEBVTT,
    NOT_VTT_ERROR,
    ZERO_INT_SIGN,
    SMART_SPLIT_MAX_CHARS,
    SMART_SPLIT_MAX_DURATION_MS,
    SMART_SPLIT_MAX_GAP_MS,
    SMART_SPLIT_MAX_LINES,
    SMART_SPLIT_MAX_WORDS,
    SMART_SPLIT_COMMA_WINDOW,
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
    SmartSplitSettings,
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

_SENTENCE_SPACE_DOT_MASK = re.compile(
    r"([.])(?=(?:[\"')\]]?)[A-Z\u0410-\u042F\u0401])"
)
_SENTENCE_SPACE_QE_MASK = re.compile(
    r"([!?])(?=(?:[\"')\]]?)[A-Z\u0410-\u042F\u0401])"
)
_SENTENCE_SKIP_CHARS = {'"', "'", ")", "]"}
_SENTENCE_START_SKIP_CHARS = {'"', "'", "(", ")", "[", "]", "{", "}", "«", "»", "-", "—", "–"}
_COMMA_CHARS = {",", "，"}


def _is_inside_parentheses(text: str, pos: int) -> bool:
    segment = text[:pos]
    return segment.count("(") > segment.count(")")


def _should_insert_qe_space(text: str, pos: int) -> bool:
    if pos <= 0:
        return True
    prev_char = text[pos - 1]
    next_idx = pos + 1
    while next_idx < len(text) and text[next_idx] in _SENTENCE_SKIP_CHARS:
        next_idx += 1
    if next_idx >= len(text):
        return False
    next_char = text[next_idx]
    if prev_char.isascii() and prev_char.isalpha() and next_char.isascii() and next_char.isalpha():
        if _is_inside_parentheses(text, pos):
            return False
        return True
    return True


def _ensure_sentence_spacing(text: str) -> str:
    temp = _SENTENCE_SPACE_DOT_MASK.sub(r"\1 ", text)

    def _replace_qe(match: re.Match[str]) -> str:
        if _should_insert_qe_space(match.string, match.start()):
            return f"{match.group(1)} "
        return match.group(1)

    return _SENTENCE_SPACE_QE_MASK.sub(_replace_qe, temp)


def _starts_with_upper(text: str) -> bool:
    cleaned = clean_line(text)
    if not cleaned:
        return False
    idx = 0
    while idx < len(cleaned) and cleaned[idx] in _SENTENCE_START_SKIP_CHARS:
        idx += 1
    if idx >= len(cleaned):
        return False
    return cleaned[idx].isupper()


def _starts_with_lower(text: str) -> bool:
    cleaned = clean_line(text)
    if not cleaned:
        return False
    idx = 0
    while idx < len(cleaned) and cleaned[idx] in _SENTENCE_START_SKIP_CHARS:
        idx += 1
    if idx >= len(cleaned):
        return False
    return cleaned[idx].islower()


def _word_has_comma(word: str) -> bool:
    return any(char in word for char in _COMMA_CHARS)


def _adjust_split_by_comma(words: List[str], start_idx: int, count: int, max_count: int) -> int:
    if SMART_SPLIT_COMMA_WINDOW <= 0 or count <= 0 or max_count <= 0:
        return count
    preferred = start_idx + count - 1
    max_index = start_idx + max_count - 1
    for offset in range(0, SMART_SPLIT_COMMA_WINDOW + 1):
        idx = preferred + offset
        if idx > max_index:
            break
        if _word_has_comma(words[idx]):
            return idx - start_idx + 1
    for offset in range(1, SMART_SPLIT_COMMA_WINDOW + 1):
        idx = preferred - offset
        if idx < start_idx:
            break
        if _word_has_comma(words[idx]):
            return idx - start_idx + 1
    return count


def _split_words_by_weights(words: List[str], weights: List[int]) -> List[int]:
    if not weights:
        return []
    total_words = len(words)
    if total_words == 0:
        return [0] * len(weights)
    suffix_weights: List[int] = []
    running = 0
    for weight in reversed(weights):
        running += max(0, int(weight))
        suffix_weights.append(running)
    suffix_weights.reverse()
    counts: List[int] = []
    offset = 0
    for idx, weight in enumerate(weights):
        remaining_words = total_words - offset
        remaining_lines = len(weights) - idx
        if remaining_lines == 1:
            count = remaining_words
        else:
            remaining_weight = suffix_weights[idx]
            if remaining_weight > 0:
                raw = int(round(remaining_words * weight / remaining_weight))
            else:
                raw = max(1, remaining_words // remaining_lines)
            max_count = remaining_words - (remaining_lines - 1)
            if max_count <= 0:
                count = 0
            else:
                count = min(max(raw, 1), max_count)
                count = _adjust_split_by_comma(words, offset, count, max_count)
        counts.append(count)
        offset += count
    if offset < total_words and counts:
        counts[-1] += total_words - offset
    return counts


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
    temp = _ensure_sentence_spacing(temp)
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
        if clean_line(origins[i]) == EMPTY_STRING:
            i += 1
            continue
        line_index = len(dialogs) + 1
        temp: List[str] = []
        times: List[str] = []
        cue_id: str | None = None
        if not srt_time_validator(origins[i]):
            candidate = origins[i].strip()
            cue_id = candidate if candidate else None
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
                cue_id=cue_id,
            )
    return dialogs


def _parse_timecode_ms(value: str | None) -> int | None:
    if not value:
        return None
    raw = value.strip()
    if not raw:
        return None
    raw = raw.replace(",", ".")
    parts = raw.split(":")
    if len(parts) < 2:
        return None
    try:
        seconds_part = parts[-1]
        if "." in seconds_part:
            seconds_str, fraction_str = seconds_part.split(".", 1)
        else:
            seconds_str, fraction_str = seconds_part, ""
        seconds = int(seconds_str)
        minutes = int(parts[-2])
        hours = int(parts[-3]) if len(parts) >= 3 else 0
    except ValueError:
        return None
    milliseconds = 0
    if fraction_str:
        digits = "".join(char for char in fraction_str if char.isdigit())
        if digits:
            if len(digits) >= 3:
                milliseconds = int(digits[:3])
            elif len(digits) == 2:
                milliseconds = int(digits) * 10
            else:
                milliseconds = int(digits) * 100
    return ((hours * 60 + minutes) * 60 + seconds) * 1000 + milliseconds


def build_prepare(
    dialogs: Dict[int, SrtSubtitlesItem],
    use_smart_dialog_splitter: bool = False,
    smart_split_settings: SmartSplitSettings | None = None,
) -> List[PrepareToTranslateItem]:
    def _normalize_setting(value: int | None, default: int) -> int:
        if value is None:
            return default
        try:
            numeric = int(value)
        except (TypeError, ValueError):
            return default
        return numeric if numeric > 0 else default

    max_lines = _normalize_setting(
        smart_split_settings.max_lines if smart_split_settings else None, SMART_SPLIT_MAX_LINES
    )
    max_words = _normalize_setting(
        smart_split_settings.max_words if smart_split_settings else None, SMART_SPLIT_MAX_WORDS
    )
    max_chars = _normalize_setting(
        smart_split_settings.max_chars if smart_split_settings else None, SMART_SPLIT_MAX_CHARS
    )
    max_gap_ms = _normalize_setting(
        smart_split_settings.max_gap_ms if smart_split_settings else None, SMART_SPLIT_MAX_GAP_MS
    )
    max_duration_ms = _normalize_setting(
        smart_split_settings.max_duration_ms if smart_split_settings else None, SMART_SPLIT_MAX_DURATION_MS
    )

    result: List[PrepareToTranslateItem] = []
    tempo_lines: List[int] = []
    tempo_text = EMPTY_STRING
    tempo_word_count = 0
    tempo_char_count = 0
    tempo_start_ms: int | None = None
    tempo_end_ms: int | None = None
    prev_text: str | None = None

    def _flush_tempo() -> None:
        nonlocal tempo_lines, tempo_text, tempo_word_count, tempo_char_count, tempo_start_ms, tempo_end_ms
        to_translate = clean_line(tempo_text)
        if not is_empty(to_translate):
            result.append(PrepareToTranslateItem(idx=len(result), lines=tempo_lines, to_translate=to_translate))
        tempo_lines = []
        tempo_text = EMPTY_STRING
        tempo_word_count = 0
        tempo_char_count = 0
        tempo_start_ms = None
        tempo_end_ms = None

    def _should_split_before(current_text: str, previous_text: str | None) -> bool:
        if not use_smart_dialog_splitter or not tempo_lines:
            return False
        if not _starts_with_upper(current_text):
            return False
        if previous_text and GOOD_END_SYMBOLS_MASK.search(previous_text):
            return False
        if previous_text and not _starts_with_lower(previous_text):
            return False
        return True
    keys = correct_sort([int(key) for key in dialogs.keys()])
    for idx, key in enumerate(keys):
        dialog = dialogs[key]
        dialog_text = dialog.text or EMPTY_STRING
        if _should_split_before(dialog_text, prev_text):
            _flush_tempo()
        tempo_lines.append(key)
        tempo_text += f" {dialog_text}"
        cleaned = clean_line(dialog_text)
        if cleaned:
            tempo_word_count += len([word for word in cleaned.split(SPACE_SIGN) if word])
            tempo_char_count += len(cleaned)
        if tempo_start_ms is None:
            tempo_start_ms = _parse_timecode_ms(dialog.start_time)
        end_time_ms = _parse_timecode_ms(dialog.end_time)
        if end_time_ms is not None:
            tempo_end_ms = end_time_ms
        should_split = False
        if not use_smart_dialog_splitter:
            should_split = True
        else:
            if GOOD_END_SYMBOLS_MASK.search(dialog_text):
                should_split = True
            elif len(tempo_lines) >= max_lines:
                should_split = True
            elif tempo_word_count >= max_words:
                should_split = True
            elif tempo_char_count >= max_chars:
                should_split = True
            else:
                if tempo_start_ms is not None and tempo_end_ms is not None:
                    duration = tempo_end_ms - tempo_start_ms
                    if duration >= max_duration_ms:
                        should_split = True
                if not should_split and idx < len(keys) - 1:
                    next_dialog = dialogs.get(keys[idx + 1])
                    if next_dialog is not None:
                        next_start = _parse_timecode_ms(next_dialog.start_time)
                        if (
                            end_time_ms is not None
                            and next_start is not None
                            and next_start - end_time_ms >= max_gap_ms
                        ):
                            should_split = True
        if should_split or idx == len(keys) - 1:
            _flush_tempo()
        prev_text = dialog_text
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
                if not characters:
                    head = [EMPTY_STRING]
                    _add_effects(head, cur.get("effects", {}))
                    temp.append(EMPTY_STRING.join(head))
                    continue
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
        if not temp:
            if cur.get("wordCount", 0) > 0:
                head = [EMPTY_STRING]
                _add_effects(head, cur.get("effects", {}))
                result.append(SPACE_SIGN.join(head))
            else:
                result.append(EMPTY_STRING)
            continue
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
            words = [word for word in text.split(SPACE_SIGN) if word]
            line_meta: List[tuple[int, AnalysedDialog, int]] = []
            for line_idx in line_idxs:
                cur = analysis.get(line_idx)
                if not cur:
                    continue
                word_count = int(cur.get("wordCount", 0) or 0)
                line_meta.append((line_idx, cur, word_count))
            lines_with_words = [meta for meta in line_meta if meta[2] > 0]
            if words and lines_with_words:
                weights = [meta[2] for meta in lines_with_words]
                counts = _split_words_by_weights(words, weights)
                offset = 0
                for (line_idx, cur, _), count in zip(lines_with_words, counts):
                    segment = words[offset : offset + count] if count > 0 else []
                    offset += count
                    result[line_idx] = _restore_line(SPACE_SIGN.join(segment), cur)
            for line_idx, cur, word_count in line_meta:
                if word_count <= 0 and line_idx not in result:
                    result[line_idx] = _restore_line(EMPTY_STRING, cur)
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
            dialog = dialogs.get(key)
            if file_format == FileFormat.VTT.value and dialog and dialog.cue_id:
                result.append(dialog.cue_id)
            if file_format == FileFormat.SRT.value:
                result.append(str(key))
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
