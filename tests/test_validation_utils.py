import json
from pathlib import Path

from sub_translate.constants import EMPTY_STRING
from sub_translate.models import AssSubtitlesItem
from sub_translate.utils.common_utils import is_empty_object
from sub_translate.utils.validation_utils import ass_separator


def _load_fixture(name: str) -> list[str]:
    path = Path(__file__).parent / "data" / name
    return json.loads(path.read_text(encoding="utf-8"))


def _check_dialog_line(
    line: AssSubtitlesItem,
    layer: int,
    start_time: str,
    end_time: str,
    style: str,
    actor: str,
    margin_l: int,
    margin_r: int,
    margin_v: int,
    effect: str,
    text: str,
) -> None:
    assert line.layer == layer
    assert line.start_time == start_time
    assert line.end_time == end_time
    assert line.style == style
    assert line.actor == actor
    assert line.margin_l == margin_l
    assert line.margin_r == margin_r
    assert line.margin_v == margin_v
    assert line.effect == effect
    assert line.text == text


def test_ass_separator_incorrect() -> None:
    idx = 155
    result = ass_separator(
        idx,
        "Dialogue: ,0:00:36.72,0:00:37.97,Карасума_Годзё_23:28(Ep 10),Wakana,0,0,0,Ba750[;0;25],\"Ahh! The sergeants are\\\\Nabout to go at it again.\"\\\\N\"Oh!\" \"Yamada-{\\\\i1}san {\\\\i0}is timing\\\\Nwhen to present his head!\"",
    )
    assert is_empty_object(result)


def test_ass_separator_layer() -> None:
    idx = 1
    result = ass_separator(
        idx,
        "Dialogue: 1,0:12:27.56,0:12:29.59,main,Touko,0000,0000,0000,,You really went to all that trouble?",
    )
    assert not is_empty_object(result)
    _check_dialog_line(
        result[idx],
        1,
        "0:12:27.56",
        "0:12:29.59",
        "main",
        "Touko",
        0,
        0,
        0,
        EMPTY_STRING,
        "You really went to all that trouble?",
    )


def test_ass_separator_style_with_undergrounds() -> None:
    idx = 1
    result = ass_separator(
        idx,
        "Dialogue: 0,0:12:25.59,0:12:29.60,sign_17877_164_Request_for_Perm,Text,0000,0000,0000,,{\\move(240,285,240,375)\\fnTrebuchet MS\\b1\\frz14.04\\an1}To practice for the athletic festival volleyball event",
    )
    assert not is_empty_object(result)
    _check_dialog_line(
        result[idx],
        0,
        "0:12:25.59",
        "0:12:29.60",
        "sign_17877_164_Request_for_Perm",
        "Text",
        0,
        0,
        0,
        EMPTY_STRING,
        "{\\move(240,285,240,375)\\fnTrebuchet MS\\b1\\frz14.04\\an1}To practice for the athletic festival volleyball event",
    )


def test_ass_separator_style_with_space_and_minus_splitter() -> None:
    idx = 2
    result = ass_separator(
        idx,
        "Dialogue: 0,0:02:45.10,0:02:47.18,Default - Top,Все,0,0,0,,Отлично потрудились.",
    )
    assert not is_empty_object(result)
    _check_dialog_line(
        result[idx],
        0,
        "0:02:45.10",
        "0:02:47.18",
        "Default - Top",
        "Все",
        0,
        0,
        0,
        EMPTY_STRING,
        "Отлично потрудились.",
    )


def test_ass_separator_style_with_brackets() -> None:
    idx = 3
    result = ass_separator(
        idx,
        "Dialogue: 0,0:17:17.43,0:17:23.18,Performance_date(Ep 12),Надпись,0,0,0,,{\\q2\\blur0.95\\frx27\\fry4\\pos(801,735)\\frz333.6}[Дата проведения]",
    )
    assert not is_empty_object(result)
    _check_dialog_line(
        result[idx],
        0,
        "0:17:17.43",
        "0:17:23.18",
        "Performance_date(Ep 12)",
        "Надпись",
        0,
        0,
        0,
        EMPTY_STRING,
        "{\\q2\\blur0.95\\frx27\\fry4\\pos(801,735)\\frz333.6}[Дата проведения]",
    )


def test_ass_separator_style_with_colons() -> None:
    idx = 4
    result = ass_separator(
        idx,
        "Dialogue: 0,0:22:42.21,0:22:43.21,Text/Messages(green)_22:33(Ep 12),Надпись,0,0,0,,{\\q2\\blur1.2\\pos(126,294)}Отличная работа сегодня! \\NМне можно пить",
    )
    assert not is_empty_object(result)
    _check_dialog_line(
        result[idx],
        0,
        "0:22:42.21",
        "0:22:43.21",
        "Text/Messages(green)_22:33(Ep 12)",
        "Надпись",
        0,
        0,
        0,
        EMPTY_STRING,
        "{\\q2\\blur1.2\\pos(126,294)}Отличная работа сегодня! \\NМне можно пить",
    )


def test_ass_separator_style_with_colons_variant() -> None:
    idx = 5
    result = ass_separator(
        idx,
        "Dialogue: 0,0:22:42.21,0:22:43.21,Text_Messages!(green)_22:33(Ep 12),Надпись,0,0,0,,{\\q2\\blur1.2\\pos(126,294)}Отличная работа сегодня! \\NМне можно пить",
    )
    assert not is_empty_object(result)
    _check_dialog_line(
        result[idx],
        0,
        "0:22:42.21",
        "0:22:43.21",
        "Text_Messages!(green)_22:33(Ep 12)",
        "Надпись",
        0,
        0,
        0,
        EMPTY_STRING,
        "{\\q2\\blur1.2\\pos(126,294)}Отличная работа сегодня! \\NМне можно пить",
    )


def test_ass_separator_style_unicode() -> None:
    idx = 6
    result = ass_separator(
        idx,
        "Dialogue: 0,0:22:42.21,0:22:45.67,Нанами_22:33(Ep 12),Надпись,0,0,0,,{\\q2\\blur1.6\\pos(663,-17)}Нанами",
    )
    assert not is_empty_object(result)
    _check_dialog_line(
        result[idx],
        0,
        "0:22:42.21",
        "0:22:45.67",
        "Нанами_22:33(Ep 12)",
        "Надпись",
        0,
        0,
        0,
        EMPTY_STRING,
        "{\\q2\\blur1.6\\pos(663,-17)}Нанами",
    )


def test_ass_separator_actor_unicode() -> None:
    idx = 10
    result = ass_separator(
        idx,
        "Dialogue: 0,0:23:39.68,0:23:41.19,Italics - Top,Карасума_Годзё_23:28(Ep 10),0,0,0,,{\\q2\\blur.6\\pos(95.667,90.333)}Карасума Годзё",
    )
    assert not is_empty_object(result)
    _check_dialog_line(
        result[idx],
        0,
        "0:23:39.68",
        "0:23:41.19",
        "Italics - Top",
        "Карасума_Годзё_23:28(Ep 10)",
        0,
        0,
        0,
        EMPTY_STRING,
        "{\\q2\\blur.6\\pos(95.667,90.333)}Карасума Годзё",
    )


def test_ass_separator_effects_karaoke() -> None:
    idx = 28
    result = ass_separator(
        idx,
        "Dialogue: 0,0:00:36.72,0:00:37.97,Italics,Wakana,0,0,0,Karaoke,Мы так близко.",
    )
    assert not is_empty_object(result)
    assert result[idx].effect == "Karaoke"
    assert result[idx].text == "Мы так близко."


def test_ass_separator_effects_scroll_up() -> None:
    idx = 25
    result = ass_separator(
        idx,
        "Dialogue: 0,0:00:36.72,0:00:37.97,Italics - Top,Карасума_Годзё_23:28(Ep 10),0,0,0,Scroll up;125;350;99[;345],\"Ahh! The sergeants are\\Nabout to go at it again.\"\\N\"Oh!\" \"Yamada-{\\i1}san {\\i0}is timing\\Nwhen to present his head!\"",
    )
    assert not is_empty_object(result)
    assert result[idx].effect == "Scroll up;125;350;99[;345]"
    assert (
        result[idx].text
        == "\"Ahh! The sergeants are\\Nabout to go at it again.\"\\N\"Oh!\" \"Yamada-{\\i1}san {\\i0}is timing\\Nwhen to present his head!\""
    )


def test_ass_separator_effects_banner() -> None:
    idx = 11
    result = ass_separator(
        idx,
        "Dialogue: 0,0:03:01.49,0:03:02.74,Japanese_Vocals_02:51_(Ep 10),Надпись,0,0,0,Banner;750[;0;125],{\\q2\\blur1\\frz359.5\\pos(642,304)}Японский Словарь Вокала и Актёрского Мастерства",
    )
    assert not is_empty_object(result)
    assert result[idx].effect == "Banner;750[;0;125]"
    assert (
        result[idx].text
        == "{\\q2\\blur1\\frz359.5\\pos(642,304)}Японский Словарь Вокала и Актёрского Мастерства"
    )


def test_ass_separator_effects_none() -> None:
    idx = 33
    result = ass_separator(
        idx,
        "Dialogue: 0,0:01:40.37,0:01:40.62,OP_Rom,,0,0,0,,Mebaeru to amaku omotteta",
    )
    assert not is_empty_object(result)
    assert result[idx].effect == EMPTY_STRING
    assert result[idx].text == "Mebaeru to amaku omotteta"


def test_ass_separator_text_commas() -> None:
    idx = 44
    result = ass_separator(
        idx,
        "Dialogue: 0,0:08:54.53,0:08:57.91,Default,,0,0,0,,On days we do reenactments,\\Neven though it's just a doll,",
    )
    assert not is_empty_object(result)
    assert result[idx].text == "On days we do reenactments,\\Neven though it's just a doll,"