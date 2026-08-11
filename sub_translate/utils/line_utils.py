from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

from sub_translate.constants import (
    BIG_NEW_LINE_SIGN,
    EMPTY_STRING,
    FIRST_GROUP,
    NOT_VTT_ERROR,
    SMART_SPLIT_COMMA_WINDOW,
    SMART_SPLIT_MAX_CHARS,
    SMART_SPLIT_MAX_DURATION_MS,
    SMART_SPLIT_MAX_GAP_MS,
    SMART_SPLIT_MAX_LINES,
    SMART_SPLIT_MAX_WORDS,
    SPACE_SIGN,
    WEBVTT,
)
from sub_translate.dictionaries.bad_symbols import BAD_SYMBOLS
from sub_translate.dictionaries.filters import (
    without_dashes,
    without_word_and_dashes,
    without_word_count,
)
from sub_translate.dictionaries.regex import (
    ASS_COMMENTS_MASK,
    ASS_EFFECTS_MASK,
    DOUBLE_SPACES_MASK,
    DRAW_MASK,
    GOOD_END_SYMBOLS_MASK,
    NEXT_LINE_MASK,
    NO_SPACE_NEXT_LINE_MASK,
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
    srt_start_validator,
    srt_time_extract,
    srt_time_validator,
    vtt_header_validator,
    vtt_time_extract,
    vtt_time_validator,
)

_SENTENCE_SPACE_DOT_MASK = re.compile(r"([.])(?=[\"')\]]?[A-Z\u0410-\u042F\u0401])")
_SENTENCE_SPACE_QE_MASK = re.compile(r"([!?])(?=[\"')\]]?[A-Z\u0410-\u042F\u0401])")
_SENTENCE_SKIP_CHARS = {'"', "'", ")", "]"}
_SENTENCE_START_SKIP_CHARS = {'"', "'", "(", ")", "[", "]", "{", "}", "«", "»", "-", "—", "–"}
_COMMA_CHARS = {",", "，"}
_PHYSICAL_LINE_BREAKS = str.maketrans({"\r": SPACE_SIGN, "\n": SPACE_SIGN})


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
        return not _is_inside_parentheses(text, pos)
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


def _adjust_split_by_comma(words: list[str], start_idx: int, count: int, max_count: int) -> int:
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


def _calculate_weighted_count(
    words: list[str],
    weights: list[int],
    suffix_weights: list[int],
    offset: int,
    idx: int,
) -> int:
    remaining_words = len(words) - offset
    remaining_lines = len(weights) - idx
    if remaining_lines == 1:
        return remaining_words
    max_count = remaining_words - (remaining_lines - 1)
    if max_count <= 0:
        return 0
    remaining_weight = suffix_weights[idx]
    raw = (
        round(remaining_words * weights[idx] / remaining_weight)
        if remaining_weight > 0
        else max(1, remaining_words // remaining_lines)
    )
    count = min(max(raw, 1), max_count)
    return _adjust_split_by_comma(words, offset, count, max_count)


def _split_words_by_weights(words: list[str], weights: list[int]) -> list[int]:
    if not weights:
        return []
    total_words = len(words)
    if total_words == 0:
        return [0] * len(weights)
    suffix_weights: list[int] = []
    running = 0
    for weight in reversed(weights):
        running += max(0, int(weight))
        suffix_weights.append(running)
    suffix_weights.reverse()
    counts: list[int] = []
    offset = 0
    for idx in range(len(weights)):
        count = _calculate_weighted_count(words, weights, suffix_weights, offset, idx)
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


def format_line(line: str, dictionary: list[dict[str, str]]) -> str:
    temp = str(line)
    for item in dictionary:
        temp = replace_all_words(temp, item["key"], item["val"])
    return temp


def clean_line(text: str) -> str:
    normalized = str(text).translate(_PHYSICAL_LINE_BREAKS)
    temp = replace_all(format_line(normalized, BAD_SYMBOLS), NO_SPACE_NEXT_LINE_MASK, FIRST_GROUP)
    temp = NEXT_LINE_MASK.sub(SPACE_SIGN, temp)
    temp = DRAW_MASK.sub(EMPTY_STRING, temp)
    temp = ASS_EFFECTS_MASK.sub(EMPTY_STRING, temp)
    temp = ASS_COMMENTS_MASK.sub(EMPTY_STRING, temp)
    temp = SRT_EFFECTS_MASK.sub(EMPTY_STRING, temp)
    temp = _ensure_sentence_spacing(temp)
    temp = DOUBLE_SPACES_MASK.sub(SPACE_SIGN, temp)
    return temp.strip()


def parse_ass_events(origins: list[str]) -> dict[int, AssSubtitlesItem]:
    events: dict[int, AssSubtitlesItem] = {}
    for idx, text in enumerate(origins):
        if ass_validator(text):
            events.update(ass_separator(idx, text))
    return events


def parse_ass_dialogs(origins: list[str]) -> dict[int, AssSubtitlesItem]:
    return {key: event for key, event in parse_ass_events(origins).items() if event.translatable}


def _read_dialog_body(
    origins: list[str],
    start: int,
    is_end: Callable[[str], bool],
    time_validator: Callable[[str], bool] = srt_time_validator,
    time_extract: Callable[[str], list[str]] = srt_time_extract,
) -> tuple[int, list[str], list[str]]:
    text_lines: list[str] = []
    times: list[str] = []
    index = start
    while index < len(origins) and not is_end(origins[index]):
        line = origins[index]
        if time_validator(line):
            times = time_extract(line)
        elif not is_empty(line):
            text_lines.append(line)
        index += 1
    return index, text_lines, times


def _is_srt_cue_start(origins: list[str], index: int) -> bool:
    return index + 1 < len(origins) and srt_start_validator(origins[index]) and srt_time_validator(origins[index + 1])


def _select_srt_dialog_key(
    cue_id: str,
    dialogs: dict[int, SrtSubtitlesItem],
    duplicate_key: int,
) -> tuple[int, int]:
    line_index = int(cue_id)
    if line_index not in dialogs:
        return line_index, duplicate_key
    while duplicate_key in dialogs:
        duplicate_key -= 1
    return duplicate_key, duplicate_key - 1


def _read_srt_text(origins: list[str], start: int) -> tuple[int, list[str]]:
    text_lines: list[str] = []
    index = start
    while index < len(origins) and not _is_srt_cue_start(origins, index):
        if not is_empty(origins[index]):
            text_lines.append(origins[index])
        index += 1
    return index, text_lines


def parse_srt_dialogs(origins: list[str]) -> dict[int, SrtSubtitlesItem]:
    dialogs: dict[int, SrtSubtitlesItem] = {}
    duplicate_key = -1
    i = 0
    while i < len(origins):
        if not _is_srt_cue_start(origins, i):
            i += 1
            continue
        cue_id = origins[i]
        line_index, duplicate_key = _select_srt_dialog_key(cue_id, dialogs, duplicate_key)
        times = srt_time_extract(origins[i + 1])
        i, text_lines = _read_srt_text(origins, i + 2)
        if text_lines:
            dialogs[line_index] = SrtSubtitlesItem(
                start_time=times[0],
                end_time=times[1],
                text=BIG_NEW_LINE_SIGN.join(text_lines),
                cue_id=cue_id,
            )
    return dialogs


def _read_vtt_dialog(origins: list[str], start: int) -> tuple[int, SrtSubtitlesItem | None]:
    cue_id: str | None = None
    index = start
    if not vtt_time_validator(origins[index]):
        candidate = origins[index].strip()
        cue_id = candidate if candidate else None
        index += 1
    index, text_lines, times = _read_dialog_body(
        origins,
        index,
        lambda line: clean_line(line) == EMPTY_STRING,
        vtt_time_validator,
        vtt_time_extract,
    )
    if len(times) != 2 or is_empty_array(text_lines):
        return index, None
    return index, SrtSubtitlesItem(
        start_time=times[0],
        end_time=times[1],
        text=BIG_NEW_LINE_SIGN.join(text_lines),
        cue_id=cue_id,
    )


def parse_vtt_dialogs(origins: list[str]) -> dict[int, SrtSubtitlesItem]:
    if not origins or not vtt_header_validator(origins[0]):
        raise ValueError(NOT_VTT_ERROR)
    i = 1
    dialogs: dict[int, SrtSubtitlesItem] = {}
    while i < len(origins):
        if clean_line(origins[i]) == EMPTY_STRING:
            i += 1
            continue
        i, dialog = _read_vtt_dialog(origins, i)
        if dialog is not None:
            dialogs[len(dialogs) + 1] = dialog
    return dialogs


def _parse_timecode_ms(value: str | None) -> int | None:
    if not value:
        return None
    raw = value.strip().split(maxsplit=1)[0]
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


@dataclass(frozen=True, slots=True)
class _SmartSplitLimits:
    max_lines: int
    max_words: int
    max_chars: int
    max_gap_ms: int
    max_duration_ms: int


@dataclass(slots=True)
class _PrepareBuffer:
    lines: list[int] = field(default_factory=list)
    text: str = EMPTY_STRING
    word_count: int = 0
    char_count: int = 0
    start_ms: int | None = None
    end_ms: int | None = None

    def add(self, key: int, dialog: SrtSubtitlesItem) -> int | None:
        dialog_text = dialog.text or EMPTY_STRING
        self.lines.append(key)
        self.text += f" {dialog_text}"
        cleaned = clean_line(dialog_text)
        if cleaned:
            self.word_count += len([word for word in cleaned.split(SPACE_SIGN) if word])
            self.char_count += len(cleaned)
        if self.start_ms is None:
            self.start_ms = _parse_timecode_ms(dialog.start_time)
        current_end_ms = _parse_timecode_ms(dialog.end_time)
        if current_end_ms is not None:
            self.end_ms = current_end_ms
        return current_end_ms

    def flush(self, result: list[PrepareToTranslateItem]) -> None:
        to_translate = clean_line(self.text)
        if not is_empty(to_translate):
            result.append(
                PrepareToTranslateItem(
                    idx=len(result),
                    lines=self.lines,
                    to_translate=to_translate,
                )
            )
        self.lines = []
        self.text = EMPTY_STRING
        self.word_count = 0
        self.char_count = 0
        self.start_ms = None
        self.end_ms = None


def _normalize_setting(value: int | None, default: int) -> int:
    if value is None:
        return default
    try:
        numeric = int(value)
    except TypeError, ValueError:
        return default
    return numeric if numeric > 0 else default


def _smart_split_limits(settings: SmartSplitSettings | None) -> _SmartSplitLimits:
    return _SmartSplitLimits(
        max_lines=_normalize_setting(settings.max_lines if settings else None, SMART_SPLIT_MAX_LINES),
        max_words=_normalize_setting(settings.max_words if settings else None, SMART_SPLIT_MAX_WORDS),
        max_chars=_normalize_setting(settings.max_chars if settings else None, SMART_SPLIT_MAX_CHARS),
        max_gap_ms=_normalize_setting(settings.max_gap_ms if settings else None, SMART_SPLIT_MAX_GAP_MS),
        max_duration_ms=_normalize_setting(
            settings.max_duration_ms if settings else None,
            SMART_SPLIT_MAX_DURATION_MS,
        ),
    )


def _should_start_new_group(
    current_text: str,
    previous_text: str | None,
    use_smart_dialog_splitter: bool,
    buffer: _PrepareBuffer,
) -> bool:
    if not use_smart_dialog_splitter or not buffer.lines or not _starts_with_upper(current_text):
        return False
    if previous_text and GOOD_END_SYMBOLS_MASK.search(previous_text):
        return False
    return not previous_text or _starts_with_lower(previous_text)


def _duration_limit_reached(buffer: _PrepareBuffer, limits: _SmartSplitLimits) -> bool:
    if buffer.start_ms is None or buffer.end_ms is None:
        return False
    return buffer.end_ms - buffer.start_ms >= limits.max_duration_ms


def _gap_limit_reached(
    current_end_ms: int | None,
    next_dialog: SrtSubtitlesItem | None,
    limits: _SmartSplitLimits,
) -> bool:
    if current_end_ms is None or next_dialog is None:
        return False
    next_start_ms = _parse_timecode_ms(next_dialog.start_time)
    return next_start_ms is not None and next_start_ms - current_end_ms >= limits.max_gap_ms


def _should_flush_group(
    dialog_text: str,
    current_end_ms: int | None,
    next_dialog: SrtSubtitlesItem | None,
    use_smart_dialog_splitter: bool,
    is_last: bool,
    buffer: _PrepareBuffer,
    limits: _SmartSplitLimits,
) -> bool:
    if is_last or not use_smart_dialog_splitter:
        return True
    if GOOD_END_SYMBOLS_MASK.search(dialog_text):
        return True
    size_limit_reached = (
        len(buffer.lines) >= limits.max_lines
        or buffer.word_count >= limits.max_words
        or buffer.char_count >= limits.max_chars
    )
    return (
        size_limit_reached
        or _duration_limit_reached(buffer, limits)
        or _gap_limit_reached(current_end_ms, next_dialog, limits)
    )


def build_prepare(
    dialogs: dict[int, SrtSubtitlesItem],
    use_smart_dialog_splitter: bool = False,
    smart_split_settings: SmartSplitSettings | None = None,
) -> list[PrepareToTranslateItem]:
    result: list[PrepareToTranslateItem] = []
    buffer = _PrepareBuffer()
    limits = _smart_split_limits(smart_split_settings)
    previous_text: str | None = None
    keys = list(dialogs)
    for idx, key in enumerate(keys):
        dialog = dialogs[key]
        dialog_text = dialog.text or EMPTY_STRING
        if _should_start_new_group(dialog_text, previous_text, use_smart_dialog_splitter, buffer):
            buffer.flush(result)
        current_end_ms = buffer.add(key, dialog)
        is_last = idx == len(keys) - 1
        next_dialog = None if is_last else dialogs.get(keys[idx + 1])
        if _should_flush_group(
            dialog_text,
            current_end_ms,
            next_dialog,
            use_smart_dialog_splitter,
            is_last,
            buffer,
            limits,
        ):
            buffer.flush(result)
        previous_text = dialog_text
    return result


def _calc_effect_index(line: str, match: str) -> int:
    before = clean_line(line[: line.index(match)])
    return 0 if is_empty(before) else len(before.split(SPACE_SIGN))


def build_effects(line: str) -> dict[int, str]:
    result: dict[int, str] = {}
    matches = ASS_EFFECTS_MASK.findall(line)
    for match in matches:
        effect_index = _calc_effect_index(line, match)
        result[effect_index] = f"{result.get(effect_index, EMPTY_STRING)}{match}"
    srt_matches = SRT_EFFECTS_MASK.findall(line)
    for match in srt_matches:
        effect_index = _calc_effect_index(line, match)
        result[effect_index] = f"{result.get(effect_index, EMPTY_STRING)}{match}"
    return result


_SYMBOL_COUNT_KEYS = {
    ".": "dotCount",
    ",": "commaCount",
    '"': "quoteCount",
    "<": "quoteCount",
    ">": "quoteCount",
    "(": "bracketCount",
    ")": "bracketCount",
    "-": "dashCount",
    ":": "colonCount",
    ";": "semicolonCount",
    "?": "questionMarkCount",
    "!": "exclamationMarkCount",
}


def _empty_symbol_counts() -> AnalysedLine:
    return {
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


def _count_symbols(text: str) -> AnalysedLine:
    counts = _empty_symbol_counts()
    for char in text:
        if char == ":":
            counts["dotCount"] += 1
        count_key = _SYMBOL_COUNT_KEYS.get(char)
        if count_key:
            counts[count_key] += 1
    return counts


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
    analysed: list[AnalysedLine] = []
    if DRAW_MASK.search(dialog_line):
        analysed.append({**result, "effects": {0: dialog_line}})
    else:
        for line in lines:
            raw_line = DRAW_MASK.sub(EMPTY_STRING, line)
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


def analyse_lines(dialogs: dict[int, AssSubtitlesItem]) -> AnalysedItem:
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


def _add_effects(words: list[str], effects: dict[int, str]) -> None:
    for key in sorted(effects.keys()):
        if key < len(words):
            words[key] = effects[key] + words[key]
        elif words:
            words[-1] = words[-1] + effects[key]


def _any_less(from_obj: dict[str, int], diff: dict[str, int], filter_func=lambda key: True) -> bool:
    return any(from_obj.get(key, 0) < diff.get(key, 0) for key in filter(filter_func, from_obj))


def _all_zeros(obj: dict[str, int], filter_func=lambda key: True) -> bool:
    total = 0
    for key in filter(filter_func, obj.keys()):
        value = obj.get(key)
        if value is None:
            continue
        try:
            total += int(value)
        except TypeError, ValueError:
            continue
    return total == 0


def _take_characters_for_line(
    characters: list[str],
    current_line: AnalysedLine,
    chars_per_line: int,
) -> tuple[str, list[str]]:
    if not characters:
        head = [EMPTY_STRING]
        _add_effects(head, current_line.get("effects", {}))
        return EMPTY_STRING.join(head), characters
    head = [characters[0]]
    count = 1
    while count < len(characters):
        symbol_counts = _count_symbols(EMPTY_STRING.join(head))
        needs_symbols = _any_less(symbol_counts, current_line)
        needs_length = _all_zeros(current_line, without_word_count) and count < chars_per_line
        if not needs_symbols and not needs_length:
            break
        head.append(characters[count])
        count += 1
    _add_effects(head, current_line.get("effects", {}))
    return EMPTY_STRING.join(head), characters[count:]


def _split_chars_by_line(
    words: list[str],
    lines_per_word: int,
    lines: list[AnalysedLine],
    result: list[str],
) -> None:
    for idx, word in enumerate(words):
        characters = list(word)
        chars_per_lines = int((len(characters) + lines_per_word - 1) / lines_per_word)
        temp: list[str] = []
        for j in range(idx * lines_per_word, idx * lines_per_word + lines_per_word):
            cur = lines[j]
            if cur.get("wordCount", 0) <= 0:
                temp.append(BIG_NEW_LINE_SIGN)
                continue
            restored, characters = _take_characters_for_line(characters, cur, chars_per_lines)
            temp.append(restored)
        result.append(BIG_NEW_LINE_SIGN.join(temp))


def _needs_more_words(head: list[str], current_line: AnalysedLine, count: int) -> bool:
    symbol_counts = _count_symbols(SPACE_SIGN.join(head))
    needs_symbols = _any_less(symbol_counts, current_line, without_dashes)
    needs_word_count = _all_zeros(current_line, without_word_count) and count < current_line.get("wordCount", 0)
    needs_symbols_without_words = _all_zeros(
        current_line,
        without_word_and_dashes,
    ) and _any_less(symbol_counts, current_line)
    return needs_symbols or needs_word_count or needs_symbols_without_words


def _take_words_for_line(
    words: list[str],
    current_line: AnalysedLine,
) -> tuple[str, list[str]]:
    head = [words[0]]
    count = 1
    while count < len(words) and current_line.get("wordCount", 0) > 1 and _needs_more_words(head, current_line, count):
        head.append(words[count])
        count += 1
    _add_effects(head, current_line.get("effects", {}))
    return SPACE_SIGN.join(head), words[count:]


def _restore_empty_line(current_line: AnalysedLine) -> str:
    if current_line.get("wordCount", 0) <= 0:
        return EMPTY_STRING
    head = [EMPTY_STRING]
    _add_effects(head, current_line.get("effects", {}))
    return SPACE_SIGN.join(head)


def _split_words_by_line(
    lines_count: int,
    lines: list[AnalysedLine],
    words: list[str],
    result: list[str],
) -> None:
    temp = list(words)
    last_text_idx: int | None = None
    for i in range(lines_count):
        cur = lines[i]
        if not temp:
            result.append(_restore_empty_line(cur))
            continue
        if cur.get("wordCount", 0) > 0:
            restored, temp = _take_words_for_line(temp, cur)
            result.append(restored)
            last_text_idx = i
        else:
            result.append(EMPTY_STRING)
    if temp and result:
        tail = SPACE_SIGN.join(temp)
        target_idx = last_text_idx if last_text_idx is not None else len(result) - 1
        if result[target_idx]:
            result[target_idx] = f"{result[target_idx]}{SPACE_SIGN}{tail}"
        else:
            result[target_idx] = tail


def _restore_line(text: str, analysis: AnalysedDialog) -> str:
    result: list[str] = []
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


def _line_metadata(
    line_idxs: list[int],
    analysis: AnalysedItem,
) -> list[tuple[int, AnalysedDialog, int]]:
    result: list[tuple[int, AnalysedDialog, int]] = []
    for line_idx in line_idxs:
        current = analysis.get(line_idx)
        if current:
            result.append((line_idx, current, int(current.get("wordCount", 0) or 0)))
    return result


def _restore_weighted_lines(
    result: TranslatedDialogItem,
    words: list[str],
    lines_with_words: list[tuple[int, AnalysedDialog, int]],
) -> None:
    counts = _split_words_by_weights(words, [metadata[2] for metadata in lines_with_words])
    offset = 0
    for (line_idx, current, _), count in zip(lines_with_words, counts, strict=True):
        segment = words[offset : offset + count] if count > 0 else []
        offset += count
        result[line_idx] = _restore_line(SPACE_SIGN.join(segment), current)


def _restore_multiple_dialogs(
    result: TranslatedDialogItem,
    text: str,
    line_idxs: list[int],
    analysis: AnalysedItem,
) -> None:
    metadata = _line_metadata(line_idxs, analysis)
    words = [word for word in text.split(SPACE_SIGN) if word]
    lines_with_words = [line for line in metadata if line[2] > 0]
    if words and lines_with_words:
        _restore_weighted_lines(result, words, lines_with_words)
    for line_idx, current, word_count in metadata:
        if word_count <= 0 and line_idx not in result:
            result[line_idx] = _restore_line(EMPTY_STRING, current)


def _restore_translated_item(
    result: TranslatedDialogItem,
    item: TranslatedItem,
    analysis: AnalysedItem,
) -> None:
    if not item.lines:
        return
    text = clean_line(item.text)
    if len(item.lines) > 1:
        _restore_multiple_dialogs(result, text, item.lines, analysis)
        return
    line_idx = item.lines[0]
    current = analysis.get(line_idx)
    if current:
        result[line_idx] = _restore_line(text, current)


def build_translated_dialogs(
    translated: list[TranslatedItem],
    analysis: AnalysedItem,
) -> TranslatedDialogItem:
    result: TranslatedDialogItem = {}
    for item in translated:
        _restore_translated_item(result, item, analysis)
    return result


def _replace_ass_dialog_text(line: str, text: str) -> str:
    fields = line.split(",", 9)
    if len(fields) != 10:
        return line
    return ",".join([*fields[:9], text or EMPTY_STRING])


def _build_ass_export_lines(
    origin: list[str],
    dialogs: dict[int, AssSubtitlesItem],
    translated_dialogs: TranslatedDialogItem,
) -> list[str]:
    return [
        _replace_ass_dialog_text(line, translated_dialogs[idx])
        if idx in dialogs and idx in translated_dialogs
        else line
        for idx, line in enumerate(origin)
    ]


def _append_timed_dialog(
    result: list[str],
    key: int,
    dialog: AssSubtitlesItem | None,
    translated_text: str,
    file_format: str,
) -> None:
    if file_format == FileFormat.VTT.value and dialog and dialog.cue_id:
        result.append(dialog.cue_id)
    if file_format == FileFormat.SRT.value:
        result.append(dialog.cue_id if dialog and dialog.cue_id is not None else str(key))
    result.append(f"{dialog.start_time if dialog else EMPTY_STRING} --> {dialog.end_time if dialog else EMPTY_STRING}")
    result.extend(translated_text.split(BIG_NEW_LINE_SIGN))
    result.append(EMPTY_STRING)


def _build_vtt_preamble(origin: list[str]) -> list[str]:
    if not origin or not vtt_header_validator(origin[0]):
        return [WEBVTT, EMPTY_STRING]
    result = [origin[0]]
    for line in origin[1:]:
        if line == EMPTY_STRING:
            break
        result.append(line)
    result.append(EMPTY_STRING)
    return result


def _vtt_timing_index(block: list[str]) -> int | None:
    if block and vtt_time_validator(block[0]):
        return 0
    if len(block) > 1 and vtt_time_validator(block[1]):
        return 1
    return None


def _build_vtt_export_from_origin(
    origin: list[str],
    dialogs: dict[int, AssSubtitlesItem],
    translated_dialogs: TranslatedDialogItem,
) -> list[str]:
    result: list[str] = []
    dialog_keys = iter(dialogs)
    next_key = next(dialog_keys, None)
    index = 0
    while index < len(origin):
        if origin[index] == EMPTY_STRING:
            result.append(EMPTY_STRING)
            index += 1
            continue
        block_end = index
        while block_end < len(origin) and origin[block_end] != EMPTY_STRING:
            block_end += 1
        block = origin[index:block_end]
        timing_index = _vtt_timing_index(block)
        if timing_index is None or next_key is None:
            result.extend(block)
        else:
            current_key = next_key
            next_key = next(dialog_keys, None)
            if current_key not in translated_dialogs:
                result.extend(block)
            else:
                result.extend(block[: timing_index + 1])
                result.extend(translated_dialogs[current_key].split(BIG_NEW_LINE_SIGN))
        index = block_end
    return result


def _build_timed_export_lines(
    origin: list[str],
    file_format: str,
    dialogs: dict[int, AssSubtitlesItem],
    translated_dialogs: TranslatedDialogItem,
) -> list[str]:
    result = _build_vtt_preamble(origin) if file_format == FileFormat.VTT.value else []
    for key, dialog in dialogs.items():
        if key not in translated_dialogs:
            continue
        _append_timed_dialog(
            result,
            key,
            dialog,
            translated_dialogs[key],
            file_format,
        )
    return result


def build_export_lines(
    origin: list[str],
    file_format: str,
    dialogs: dict[int, AssSubtitlesItem],
    translated_dialogs: TranslatedDialogItem,
) -> list[str]:
    if file_format == FileFormat.ASS.value:
        return _build_ass_export_lines(origin, dialogs, translated_dialogs)
    if file_format == FileFormat.VTT.value and origin and vtt_header_validator(origin[0]):
        return _build_vtt_export_from_origin(origin, dialogs, translated_dialogs)
    return _build_timed_export_lines(origin, file_format, dialogs, translated_dialogs)


__all__ = [
    "analyse_line",
    "analyse_lines",
    "build_export_lines",
    "build_prepare",
    "build_translated_dialogs",
    "clean_line",
    "format_line",
    "parse_ass_dialogs",
    "parse_ass_events",
    "parse_srt_dialogs",
    "parse_vtt_dialogs",
]
