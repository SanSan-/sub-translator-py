import pytest

from sub_translate.enums import FileFormat


@pytest.mark.parametrize(
    "member",
    [FileFormat.ASS, FileFormat.SRT, FileFormat.VTT],
)
def test_string_enums_preserve_legacy_representation(member: FileFormat) -> None:
    expected = f"{type(member).__name__}.{member.name}"

    assert isinstance(member, str)
    assert member == member.value
    assert str(member) == expected
    assert f"{member}" == expected
    assert format(member, ">30") == f"{expected:>30}"
