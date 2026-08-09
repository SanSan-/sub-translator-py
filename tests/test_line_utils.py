import json
from pathlib import Path

import pytest

from sub_translate.constants import BIG_NEW_LINE_SIGN, EMPTY_STRING, SMART_SPLIT_MAX_GAP_MS
from sub_translate.models import SmartSplitSettings, SrtSubtitlesItem, TranslatedItem
from sub_translate.utils.common_utils import is_empty_array, is_empty_object
from sub_translate.utils.line_utils import (
    analyse_lines,
    build_prepare,
    build_translated_dialogs,
    clean_line,
    parse_ass_dialogs,
    parse_srt_dialogs,
)


def _load_fixture(name: str) -> list[str]:
    path = Path(__file__).parent / "data" / name
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            "For a sex crime, we use\\Na doll like this to reenact it.",
            "For a sex crime, we use a doll like this to reenact it.",
        ),
        (
            "--And, sure enough,\\Nthings didn't look right.\\N--It's 50 meters from the scene of\\Nthe crime to Hamaguchi Dental.",
            "- And, sure enough, things didn't look right. - It's 50 meters from the scene of the crime to Hamaguchi Dental.",
        ),
        (
            '"Ahh! The sergeants are\\Nabout to go at it again."\\N"Oh!" "Yamada-{\\i1}san {\\i0}is timing\\Nwhen to present his head!"',
            '"Ahh! The sergeants are about to go at it again." "Oh!" "Yamada-san is timing when to present his head!"',
        ),
        (
            "{\\move(320,140,303,140)\\bord0\\c&H705E5B&\\fs11\\frx14\\fry306\\frz356.1}Fukumoto Miki \\N\\NMusic Note Interior",
            "Fukumoto Miki Music Note Interior",
        ),
        (
            "{\\fs8\\pos(150,55)\\fay-.3\\frz2.928\\bord3}K\\Ni\\Nz\\Na\\Nk\\Ni\\N \\NE\\Nr\\Ni\\Nk\\Na",
            "Kizaki Erika",
        ),
        (
            "{\\pos(240,105)}W\\Na\\Ns\\Nh\\Ni\\No\\N \\NH\\Ni\\Nt\\No\\Nm\\Ni",
            "Washio Hitomi",
        ),
        (
            "{\\pos(325,100)}N\\Na\\Nw\\Na\\Ns\\Nh\\Ni\\Nr\\No\\N \\NY\\Na\\Ns\\Nu\\Nk\\No",
            "Nawashiro Yasuko",
        ),
        (
            "{\\pos(325,100)}\\NY\\Na\\Ns\\Nu\\Nk\\No N\\Na\\Nw\\Na\\Ns\\Nh\\Ni\\Nr\\No\\N",
            "Yasuko Nawashiro",
        ),
        (
            "I'm gonna be in the cheer squad \\Nalong with Akebi-chan, you know...",
            "I'm gonna be in the cheer squad along with Akebi-chan, you know:",
        ),
        (
            "What's this about? \\NThere's no need to be so polite.",
            "What's this about? There's no need to be so polite.",
        ),
        (
            "What's this about?\\NThere's no need to be so polite.",
            "What's this about? There's no need to be so polite.",
        ),
        (
            "But that doesn't matter if you\\N can't keep the ball in the air.",
            "But that doesn't matter if you can't keep the ball in the air.",
        ),
        (
            "H-Hey, I'm just wishing out loud!\\N Don't be mad, okay?",
            "H-Hey, I'm just wishing out loud! Don't be mad, okay?",
        ),
        (
            "H-Hey, I'm just wishing out loud!\\NDon't be mad, okay?",
            "H-Hey, I'm just wishing out loud! Don't be mad, okay?",
        ),
        (
            '{\\fs8}2. Acoustic Guitar and Singing (Class 2-3 Soumi Ruri) \\NShort Film "As We Accelerate" (Film Research Club) \\N4. Band Performance (Radical?Pop) \\N5. Dance (Drama Club: Akebi Komichi) \\N6. Catch the Rhythm! Medley of Popular Songs (Brass Band Club) \\NClosing Speech \\NAfter-Party Planning Committee President (Class 3-2 Hanasato Yuka) \\N\\N{\\fs12}Time: After the Athletic Festival\'s Closing Ceremony \\NPlace: Roubai Auditorium',
            '2. Acoustic Guitar and Singing (Class 2-3 Soumi Ruri) Short Film "As We Accelerate" (Film Research Club) 4. Band Performance (Radical?Pop) 5. Dance (Drama Club: Akebi Komichi) 6. Catch the Rhythm! Medley of Popular Songs (Brass Band Club) Closing Speech After-Party Planning Committee President (Class 3-2 Hanasato Yuka) Time: After the Athletic Festival\'s Closing Ceremony Place: Roubai Auditorium',
        ),
        (
            'Roubai Academy Athletic Festival \\N\\N{\\fs16}After-Party \\N\\N{\\fs12}Program \\N\\N{\\fs8}\x07 Opening Speech \x07\\NAfter-Party Planning Committee Vice President\\N (Class 3-4 Takayama Haru) \\N\\N\x07 Stage Program \x07\\N1. Quiz Competition:\\N "How Many Do You Know?! Secrets of Roubai Academy" \\N(After-Party Planning Committee) \\N2. Acoustic Guitar and Singing (Class 2-3 Soumi Ruri) \\NShort Film "As We Accelerate" (Film Research Club) \\N4. Band Performance (Radical?Pop) \\N5. Dance (Drama Club: Akebi Komichi)',
            'Roubai Academy Athletic Festival After-Party Program \x07 Opening Speech \x07 After-Party Planning Committee Vice President (Class 3-4 Takayama Haru) \x07 Stage Program \x07 1. Quiz Competition: "How Many Do You Know?! Secrets of Roubai Academy" (After-Party Planning Committee) 2. Acoustic Guitar and Singing (Class 2-3 Soumi Ruri) Short Film "As We Accelerate" (Film Research Club) 4. Band Performance (Radical?Pop) 5. Dance (Drama Club: Akebi Komichi)',
        ),
        (
            '{\\an8}"First Investigation Division"\\NWhat is this, an invitational\\Nhard-boiled tournament?!',
            '"First Investigation Division" What is this, an invitational hard-boiled tournament?!',
        ),
        (
            "--I'm outta here.\\N--It does get tense, when it comes\\Nto HQ's First Investigation Division.",
            "- I'm outta here. - It does get tense, when it comes to HQ's First Investigation Division.",
        ),
        (
            "Hold on! Hold on! Hold on, dummy!\\NI'm doing my best here, too!",
            "Hold on! Hold on! Hold on, dummy! I'm doing my best here, too!",
        ),
        (
            "{\\an8\\b1\\blur0.4\\shad0\\fs26\\bord1.5\\c&H2400D3&\\3c&HDDD9F5&\\pos(542.667,21.334)}Declares \\NIndependence \\Nfrom the \\NJapanese\\NGovernment",
            "Declares Independence from the Japanese Government",
        ),
        (
            "{\\an8\\b1\\blur0.4\\fax-0.07\\shad0\\bord1.2\\c&H2400D3&\\3c&HDDD9F5&\\frz7.875\\pos(172,142.666)}New Governor\\N{\\fs42}Nekoyanagi\\N Pawoo",
            "New Governor Nekoyanagi Pawoo",
        ),
        ("Hey, Mako- <i>chan</i> , do you have a wish?", "Hey, Mako- chan , do you have a wish?"),
        (
            "It's all right. Let's give it our all, Rei- <i>chan</i> !",
            "It's all right. Let's give it our all, Rei- chan !",
        ),
    ],
)
def test_clean_line_smart_wrapping(source: str, expected: str) -> None:
    assert clean_line(source) == expected


def test_clean_line_sentence_spacing() -> None:
    assert clean_line("Если нам повезет, интервьюер перечислит все функции.Например, так.") == (
        "Если нам повезет, интервьюер перечислит все функции. Например, так."
    )


def test_clean_line_sentence_spacing_questions() -> None:
    assert clean_line("Что?Пойдём!") == "Что? Пойдём!"


def test_clean_line_sentence_spacing_questions_en() -> None:
    assert clean_line("What?No!") == "What? No!"


def test_clean_line_non_breaking_hyphen() -> None:
    assert clean_line("request\u2011response") == "request-response"


def test_clean_line_url_preserved() -> None:
    assert clean_line("Сайт example.com работает.") == "Сайт example.com работает."


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ('          "Victim"            "Suspect"        ', '"Victim" "Suspect"'),
        (
            "{\\an7\\fs85\\1c&H433A3C&\\pos(102,344)\\frz345.7}Вас ждёт решающий день\\N{\\fs25} \\N{\\fs85} Постарайтесь!",
            "Вас ждёт решающий день Постарайтесь!",
        ),
        (
            "{\\move(602,686,602,576,0,398)\\clip(572,552,1168,692)}Нисиката повержен!\\N\\N\\N Нисиката всё ещё смущается.",
            "Нисиката повержен! Нисиката всё ещё смущается.",
        ),
    ],
)
def test_clean_line_double_spaces(source: str, expected: str) -> None:
    assert clean_line(source) == expected


def test_clean_line_bold() -> None:
    assert clean_line("There is a {\\b1}bold {\\b0}word here") == "There is a bold word here"


def test_clean_line_italicised() -> None:
    assert clean_line("There is an {\\i1}italicised {\\i0}word here") == "There is an italicised word here"


def test_clean_line_underline() -> None:
    assert clean_line("There is an {\\u1}underline {\\u0}word here") == "There is an underline word here"


def test_clean_line_strikeout() -> None:
    assert clean_line("There is an {\\s1}strikeout {\\s0}word here") == "There is an strikeout word here"


def test_clean_line_border() -> None:
    assert clean_line("{\\bord1.5}Телепередачи часто нелегально загружают в интернет.") == (
        "Телепередачи часто нелегально загружают в интернет."
    )


def test_clean_line_shadow() -> None:
    assert clean_line("{\\shad.15}{SSA logo#текст под лого}Brand new SSA.") == "Brand new SSA."


def test_clean_line_comments() -> None:
    assert (
        clean_line(
            "{\\fad(0,650)\\blur5\\an5\\pos(640,510)\\alpha&HFFF\\t(0,450,\\alpha&H777)}На {\\c&H9e55e8&}Зад{\\c}ней {\\c&H9e55e8&}Пар{\\c}те {Нет, серьёзно, это же Зазнайкина и Бармалейкин в японской реальности, спустя 40 лет}"
        )
        == "На Задней Парте"
    )


def test_clean_line_blur_edges() -> None:
    assert clean_line("{\\be1}Я умственно переутомляюсь от этих ролей за массовку.{\\be0}") == (
        "Я умственно переутомляюсь от этих ролей за массовку."
    )


def test_clean_line_font_name() -> None:
    assert clean_line("Here is some {\\fnCourier New}fixed space text") == "Here is some fixed space text"


def test_clean_line_font_size() -> None:
    assert clean_line("{\\fs16}This is small text. {\\fs28}This is large text") == (
        "This is small text. This is large text"
    )


def test_clean_line_scale() -> None:
    assert clean_line("{\\fscx14.05\\fscy13.75\\t(0,620,(\\fscx100\\fscy100))}Donna") == "Donna"


def test_clean_line_spacing() -> None:
    assert clean_line("changes the distance {\\fsp24}between letters") == "changes the distance between letters"


def test_clean_line_rotation() -> None:
    assert clean_line("{\\frz359.9\\fax-0.1\\frx.354\\fry2}Mebaeru to amaku omotteta") == ("Mebaeru to amaku omotteta")


def test_clean_line_color() -> None:
    assert clean_line("{\\c&HFF&}This is pure, full intensity red") == "This is pure, full intensity red"
    assert clean_line("{\\c&HFF00&}This is pure, full intensity Green") == "This is pure, full intensity Green"
    assert clean_line("{\\c&HFF0000&}This is pure, full intensity Blue") == "This is pure, full intensity Blue"
    assert clean_line("{\\c&HFFFFFF&}This is White") == "This is White"
    assert clean_line("{\\c&HA0A0A&}This is dark grey") == "This is dark grey"


def test_clean_line_specific_colors() -> None:
    assert clean_line("{\\1c&H2C374A&}год") == "год"
    assert clean_line("{\\2c&H2C374A&}год") == "год"
    assert clean_line("{\\3c&H2C374A&}год") == "год"
    assert clean_line("{\\4c&H2C374A&}год") == "год"


def test_clean_line_specific_alpha_channels() -> None:
    assert clean_line("{\\1a&HFF&}год") == "год"
    assert clean_line("{\\2a&HEE&}год") == "год"
    assert clean_line("{\\3a&HAA&}год") == "год"
    assert clean_line("{\\4a&H07E&}год") == "год"


def test_clean_line_alpha_channel() -> None:
    assert clean_line("Что случилось?\\N{\\alpha&HFFF&}`{\\alpha}") == "Что случилось? `"


def test_clean_line_alignment() -> None:
    assert clean_line("{\\a1}This is a left-justified subtitle") == "This is a left-justified subtitle"
    assert clean_line("{\\a2}This is a centered subtitle") == "This is a centered subtitle"
    assert clean_line("  {\\a3}This is a right-justified subtitle") == "This is a right-justified subtitle"
    assert clean_line("{\\a5}This is a left-justified toptitle") == "This is a left-justified toptitle"
    assert clean_line("{\\a11}This is a right-justified midtitle") == "This is a right-justified midtitle"


def test_clean_line_numpad_layout() -> None:
    assert clean_line('{\\an8}"Police in a Pod"') == '"Police in a Pod"'


def test_clean_line_karaoke_words() -> None:
    assert clean_line(" {\\k94}This {\\k48}is {\\k24}a {\\k150}karaoke {\\k94}line") == ("This is a karaoke line")


def test_clean_line_karaoke_fill() -> None:
    assert (
        clean_line(
            "{\\K50}Kan{\\K25}chi{\\K45}ga{\\K40}i  {\\kf}sa{\\kf}re{\\kf}cha{\\kf}tta{\\kf}tte  {\\K75}ii  {\\K35}yo"
        )
        == "Kanchigai sarechattatte ii yo"
    )


def test_clean_line_karaoke_outline() -> None:
    assert (
        clean_line(
            "{\\ko30}Ko{\\ko30}tei  {\\ko20}de  {\\ko35}ki{\\ko35}mi  {\\ko30}no  {\\ko30}ko{\\ko30}to{\\ko30}ba"
        )
        == "Kotei de kimi no kotoba"
    )


def test_clean_line_wrapping_style() -> None:
    assert clean_line("{\\q2\\blur.8\\pos(66.667,416.667)}Результат Поиска") == "Результат Поиска"


def test_clean_line_cancel_overrides() -> None:
    assert (
        clean_line(
            "{\\q2\\blur1.1\\frx18\\fry358\\org(428.4,167.2)\\frz0.6719\\pos(319.6,285.6)\\clip(184,296,428.8,328)}П{\\blur.9}р{\\blur.6}ямая {\\rDefault}трансляция"
        )
        == "Прямая трансляция"
    )


def test_clean_line_wrong_symbol_in_effects() -> None:
    assert clean_line("{\\frx18\\fry24|}Gojo Dolls") == "Gojo Dolls"


def test_clean_line_animation_1() -> None:
    assert (
        clean_line(
            "{\\c&HD5C5F5&\\3c&H9A60E4&\\t(355,355,(\\c&H524C48&))\\t(355,355,(\\3c&HFFFFFF&))\\t(1523,1523,(\\c&HE7B592&))\\t(1523,1523,(\\3c&HB86F62&))\\t(1857,1857,(\\c&H524C48&))\\t(1857,1857,(\\3c&HFFFFFF&))}Чтобы зазвучал (на весь белый свет)"
        )
        == "Чтобы зазвучал (на весь белый свет)"
    )


def test_clean_line_animation_2() -> None:
    assert (
        clean_line(
            "Fure! {\\alpha&HFF\\t(290,-15,\\alpha&H00)}Fure! {\\alpha&HFF\\t(580,-15,\\alpha&H00)}Fure! {\\alpha&HFF\\t(880,-15,\\alpha&H00)}?"
        )
        == "Fure! Fure! Fure! ?"
    )


def test_clean_line_animation_3() -> None:
    assert clean_line("{\\fad(0,2000)\\c&H8E61C3&\\t(4071,0,(\\fs44)}Koete saku darou") == ("Koete saku darou")


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            "{\\move(640,60,639.822,20.178,26,2528)}(Kizuita n da ne? Hajimaru yo)",
            "(Kizuita n da ne? Hajimaru yo)",
        ),
        (
            "{\\q2\\blur1\\frz270\\move(115,83,115,-573)}Очарован и в состоянии\\Nстать кем-угодно и чем-угодно",
            "Очарован и в состоянии стать кем-угодно и чем-угодно",
        ),
        (
            "{\\q2\\bord1.5\\fs42\\blur1.5\\move(2994,658,-1700,658,27,26701)\\c&HF9F8F7&\\3c&H111111&}Первый том Blu-ray&DVD выйдет 21 декабря",
            "Первый том Blu-ray&DVD выйдет 21 декабря",
        ),
    ],
)
def test_clean_line_movement(source: str, expected: str) -> None:
    assert clean_line(source) == expected


def test_clean_line_position() -> None:
    assert clean_line("{\\pos(160,260)}Перевод") == "Перевод"


def test_clean_line_move_origin() -> None:
    assert clean_line("{\\blur1\\c&fafafa&\\pos(636,677)\\org(636,679)}Yumemiyo otome") == ("Yumemiyo otome")


def test_clean_line_fade_1() -> None:
    assert clean_line("{\\fad(150,0)}Ichiban ni naritai") == "Ichiban ni naritai"


def test_clean_line_fade_2() -> None:
    assert clean_line("{\\fad(200,220)\\fax-0.3\\t(1770,2062,(\\fax0)}Chiisana yume no tane wo maku") == (
        "Chiisana yume no tane wo maku"
    )


def test_clean_line_fade_3() -> None:
    assert (
        clean_line("{\\fade(&HFF, &HEE, &H00, 0, 500, 350, 2000)\\c&H8E61C3&\\t(4071,0,(\\fs44)}И надеждой расцветём.")
        == "И надеждой расцветём."
    )


def test_clean_line_clip_1() -> None:
    assert (
        clean_line("{\\move(602,570,602,525,571,738)\\clip(572,552,1168,692)}Такаги-сан дразнит его.")
        == "Такаги-сан дразнит его."
    )


def test_clean_line_clip_2() -> None:
    assert clean_line("{\\blur1\\1c&H171719&\\pos(96,244)\\iclip(m 286 236 l 248 284 192 230)}Нисиката появился!") == (
        "Нисиката появился!"
    )


def test_clean_line_drawings_1() -> None:
    assert (
        clean_line(
            "{\\fad(250,250)\\an7\\pos(783,114)\\1c&Hfafafa&\\p1\\fscx25\\fscy25}{SSA logo#рамка}m 294 294.02 l 455.98 294.02 455.98 456 294 456  m 312.34 312.36 l 312.34 437.66 437.64 437.66 437.64 312.36"
        )
        == EMPTY_STRING
    )


def test_clean_line_drawings_2() -> None:
    assert clean_line("{\\blur3\\1c&HFDFDFD&\\p4\\frz345.7\\pos(100,340)}m 0 0 l 1100 0 1100 195 0 195{\\p0}") == (
        EMPTY_STRING
    )


def test_clean_line_baseline_offset_up() -> None:
    assert (
        clean_line(
            "{\\blur1\\1c&H171719&\\pbo-34\\pos(96,244)\\iclip(m 286 236 l 248 284 192 230)}m 0 0 l 190 0 190 55 0 55"
        )
        == EMPTY_STRING
    )


def test_clean_line_baseline_offset_down() -> None:
    assert clean_line("{\\blur1\\1c&H131313&\\pbo12\\pos(228,357)}m 0 0 l 150 0 150 250 0 250") == EMPTY_STRING


def test_parse_ass_dialogs_akebi() -> None:
    dialogs = parse_ass_dialogs(_load_fixture("akebi11.ass.json"))
    assert not is_empty_object(dialogs)
    assert len(dialogs) == 299


def test_parse_ass_dialogs_bisco() -> None:
    dialogs = parse_ass_dialogs(_load_fixture("bisco12.ass.json"))
    assert not is_empty_object(dialogs)
    assert len(dialogs) == 310


def test_parse_ass_dialogs_girlish() -> None:
    dialogs = parse_ass_dialogs(_load_fixture("girlishNum10.ass.json"))
    assert not is_empty_object(dialogs)
    assert len(dialogs) == 653


def test_parse_ass_dialogs_hakozume() -> None:
    dialogs = parse_ass_dialogs(_load_fixture("hakozume12.ass.json"))
    assert not is_empty_object(dialogs)
    assert len(dialogs) == 403


def test_parse_ass_dialogs_sono() -> None:
    dialogs = parse_ass_dialogs(_load_fixture("sonoB12.ass.json"))
    assert not is_empty_object(dialogs)
    assert len(dialogs) == 376


def test_parse_srt_dialogs_free_movie() -> None:
    dialogs = parse_srt_dialogs(_load_fixture("freeMovie.srt.json"))
    assert not is_empty_object(dialogs)
    assert len(dialogs) == 1724


def test_build_prepare_smart_splitter_off_akebi() -> None:
    prepare = build_prepare(parse_ass_dialogs(_load_fixture("akebi11.ass.json")))
    assert not is_empty_array(prepare)
    assert len(prepare) == 299


def test_build_prepare_smart_splitter_off_bisco() -> None:
    prepare = build_prepare(parse_ass_dialogs(_load_fixture("bisco12.ass.json")))
    assert not is_empty_array(prepare)
    assert len(prepare) == 306


def test_build_prepare_smart_splitter_off_girlish() -> None:
    prepare = build_prepare(parse_ass_dialogs(_load_fixture("girlishNum10.ass.json")))
    assert not is_empty_array(prepare)
    assert len(prepare) == 581


def test_build_prepare_smart_splitter_off_hakozume() -> None:
    prepare = build_prepare(parse_ass_dialogs(_load_fixture("hakozume12.ass.json")))
    assert not is_empty_array(prepare)
    assert len(prepare) == 403


def test_build_prepare_smart_splitter_off_sono() -> None:
    prepare = build_prepare(parse_ass_dialogs(_load_fixture("sonoB12.ass.json")))
    assert not is_empty_array(prepare)
    assert len(prepare) == 375


def test_build_prepare_smart_splitter_off_free_movie() -> None:
    prepare = build_prepare(parse_srt_dialogs(_load_fixture("freeMovie.srt.json")))
    assert not is_empty_array(prepare)
    assert len(prepare) == 1724


def test_build_prepare_smart_splitter_on_akebi() -> None:
    prepare = build_prepare(parse_ass_dialogs(_load_fixture("akebi11.ass.json")), True)
    assert not is_empty_array(prepare)
    assert len(prepare) == 280


def test_build_prepare_smart_splitter_on_bisco() -> None:
    prepare = build_prepare(parse_ass_dialogs(_load_fixture("bisco12.ass.json")), True)
    assert not is_empty_array(prepare)
    assert len(prepare) == 254


def test_build_prepare_smart_splitter_on_girlish() -> None:
    prepare = build_prepare(parse_ass_dialogs(_load_fixture("girlishNum10.ass.json")), True)
    assert not is_empty_array(prepare)
    assert len(prepare) == 403


def test_build_prepare_smart_splitter_on_hakozume() -> None:
    prepare = build_prepare(parse_ass_dialogs(_load_fixture("hakozume12.ass.json")), True)
    assert not is_empty_array(prepare)
    assert len(prepare) == 375


def test_build_prepare_smart_splitter_on_sono() -> None:
    prepare = build_prepare(parse_ass_dialogs(_load_fixture("sonoB12.ass.json")), True)
    assert not is_empty_array(prepare)
    assert len(prepare) == 333


def test_build_prepare_smart_splitter_on_free_movie() -> None:
    prepare = build_prepare(parse_srt_dialogs(_load_fixture("freeMovie.srt.json")), True)
    assert not is_empty_array(prepare)
    assert len(prepare) == 1633


def test_build_prepare_smart_splitter_line_limit() -> None:
    dialogs = {
        1: SrtSubtitlesItem(start_time="00:00:00,000", end_time="00:00:00,500", text="Hello"),
        2: SrtSubtitlesItem(start_time="00:00:00,600", end_time="00:00:01,000", text="World"),
        3: SrtSubtitlesItem(start_time="00:00:01,100", end_time="00:00:01,500", text="Again"),
        4: SrtSubtitlesItem(start_time="00:00:01,600", end_time="00:00:02,000", text="And"),
        5: SrtSubtitlesItem(start_time="00:00:02,100", end_time="00:00:02,500", text="More"),
    }
    prepare = build_prepare(dialogs, True)
    assert len(prepare) == 2
    assert prepare[0].lines == [1, 2, 3, 4]
    assert prepare[1].lines == [5]


def test_build_prepare_smart_splitter_end_symbol_split() -> None:
    dialogs = {
        1: SrtSubtitlesItem(start_time="00:00:00,000", end_time="00:00:01,000", text="Hello world."),
        2: SrtSubtitlesItem(start_time="00:00:01,100", end_time="00:00:02,000", text="Next line"),
    }
    prepare = build_prepare(dialogs, True)
    assert len(prepare) == 2
    assert prepare[0].lines == [1]
    assert prepare[1].lines == [2]


def test_build_prepare_smart_splitter_uppercase_split() -> None:
    dialogs = {
        1: SrtSubtitlesItem(start_time="00:00:00,000", end_time="00:00:00,500", text="we define requirements"),
        2: SrtSubtitlesItem(start_time="00:00:00,600", end_time="00:00:01,000", text="still continue"),
        3: SrtSubtitlesItem(start_time="00:00:01,100", end_time="00:00:01,600", text="For example we design"),
        4: SrtSubtitlesItem(start_time="00:00:01,700", end_time="00:00:02,200", text="the system"),
    }
    settings = SmartSplitSettings(
        max_lines=10,
        max_words=100,
        max_chars=1000,
        max_gap_ms=9999,
        max_duration_ms=99999,
    )
    prepare = build_prepare(dialogs, True, settings)
    assert len(prepare) == 2
    assert prepare[0].lines == [1, 2]
    assert prepare[1].lines == [3, 4]


def test_build_prepare_smart_splitter_word_limit() -> None:
    dialogs = {
        1: SrtSubtitlesItem(start_time="00:00:00,000", end_time="00:00:01,000", text="one two"),
        2: SrtSubtitlesItem(start_time="00:00:01,100", end_time="00:00:02,000", text="three four"),
        3: SrtSubtitlesItem(start_time="00:00:02,100", end_time="00:00:03,000", text="five"),
    }
    settings = SmartSplitSettings(
        max_lines=10,
        max_words=3,
        max_chars=1000,
        max_gap_ms=9999,
        max_duration_ms=99999,
    )
    prepare = build_prepare(dialogs, True, settings)
    assert len(prepare) == 2
    assert prepare[0].lines == [1, 2]
    assert prepare[1].lines == [3]


def test_build_prepare_smart_splitter_char_limit() -> None:
    dialogs = {
        1: SrtSubtitlesItem(start_time="00:00:00,000", end_time="00:00:01,000", text="Hello"),
        2: SrtSubtitlesItem(start_time="00:00:01,100", end_time="00:00:02,000", text="World!"),
        3: SrtSubtitlesItem(start_time="00:00:02,100", end_time="00:00:03,000", text="Again"),
    }
    settings = SmartSplitSettings(
        max_lines=10,
        max_words=100,
        max_chars=10,
        max_gap_ms=9999,
        max_duration_ms=99999,
    )
    prepare = build_prepare(dialogs, True, settings)
    assert len(prepare) == 2
    assert prepare[0].lines == [1, 2]
    assert prepare[1].lines == [3]


def test_build_prepare_smart_splitter_duration_limit() -> None:
    dialogs = {
        1: SrtSubtitlesItem(start_time="00:00:00,000", end_time="00:00:00,900", text="Hello"),
        2: SrtSubtitlesItem(start_time="00:00:00,950", end_time="00:00:02,200", text="World"),
        3: SrtSubtitlesItem(start_time="00:00:02,300", end_time="00:00:03,000", text="Again"),
    }
    settings = SmartSplitSettings(
        max_lines=10,
        max_words=100,
        max_chars=1000,
        max_gap_ms=9999,
        max_duration_ms=1000,
    )
    prepare = build_prepare(dialogs, True, settings)
    assert len(prepare) == 2
    assert prepare[0].lines == [1, 2]
    assert prepare[1].lines == [3]


def test_build_translated_dialogs_comma_split() -> None:
    dialogs = {
        1: SrtSubtitlesItem(text="First part here"),
        2: SrtSubtitlesItem(text="Second part here"),
    }
    analysis = analyse_lines(dialogs)
    translated = [
        TranslatedItem(
            idx=0,
            text="Alpha beta, gamma delta",
            lines=[1, 2],
        )
    ]
    result = build_translated_dialogs(translated, analysis)
    assert result[1] == "Alpha beta,"
    assert result[2] == "gamma delta"


def test_build_translated_dialogs_weight_split() -> None:
    dialogs = {
        1: SrtSubtitlesItem(text="one two three"),
        2: SrtSubtitlesItem(text="four"),
    }
    analysis = analyse_lines(dialogs)
    translated = [
        TranslatedItem(
            idx=0,
            text="a b c d e f g h",
            lines=[1, 2],
        )
    ]
    result = build_translated_dialogs(translated, analysis)
    assert len(result[1].split()) == 6
    assert len(result[2].split()) == 2


def test_build_translated_dialogs_no_empty_lines() -> None:
    dialogs = {
        8: SrtSubtitlesItem(text="The system must allow application to exchange messages."),
        9: SrtSubtitlesItem(text="non-functional requirements defined qualities of a system"),
        10: SrtSubtitlesItem(text="how a System supposed to be"),
        11: SrtSubtitlesItem(text="For example non-functional requirements for messaging system"),
        12: SrtSubtitlesItem(text="may look like the following."),
    }
    analysis = analyse_lines(dialogs)
    translated = [
        TranslatedItem(
            idx=0,
            text=(
                "Система должна разрешить приложению обмениваться сообщениями. "
                "нефункциональные требования определяют качества системы, какой система должна быть"
            ),
            lines=[8, 9, 10],
        ),
        TranslatedItem(
            idx=1,
            text="Например, нефункциональные требования к системе обмена сообщениями могут выглядеть следующим образом.",
            lines=[11, 12],
        ),
    ]
    result = build_translated_dialogs(translated, analysis)
    assert result[10].strip() != ""
    assert result[12].strip() != ""


def test_build_prepare_smart_splitter_gap_limit() -> None:
    gap_ms = SMART_SPLIT_MAX_GAP_MS + 200
    start_ms = 1000 + gap_ms
    start_seconds = start_ms // 1000
    start_millis = start_ms % 1000
    end_ms = start_ms + 500
    end_seconds = end_ms // 1000
    end_millis = end_ms % 1000
    dialogs = {
        1: SrtSubtitlesItem(start_time="00:00:00,000", end_time="00:00:01,000", text="Hello"),
        2: SrtSubtitlesItem(
            start_time=f"00:00:{start_seconds:02d},{start_millis:03d}",
            end_time=f"00:00:{end_seconds:02d},{end_millis:03d}",
            text="World",
        ),
    }
    prepare = build_prepare(dialogs, True)
    assert len(prepare) == 2
    assert prepare[0].lines == [1]
    assert prepare[1].lines == [2]


def test_build_translated_dialogs_agent_cache_not_truncated() -> None:
    cached_text = (
        "При проектировании системы, чтобы достичь качеств, о которых мы только что говорили, и программное, и"
    )
    dialogs = {
        1: SrtSubtitlesItem(
            text=(
                "When designing a system, to achieve qualities"
                f"{BIG_NEW_LINE_SIGN}we’ve just discussed, both software and"
            )
        ),
    }
    analysis = analyse_lines(dialogs)
    translated = [
        TranslatedItem(
            idx=0,
            text=cached_text,
            lines=[1],
        )
    ]
    result = build_translated_dialogs(translated, analysis)
    assert clean_line(result[1]) == clean_line(cached_text)
