from __future__ import annotations

from sub_translate.dictionaries.languages import get_code


def split_lang_suffix(stem: str) -> tuple[str, str | None]:
    if "." not in stem:
        return stem, None
    base, candidate = stem.rsplit(".", 1)
    code = get_code(candidate)
    if code:
        return base, code
    return stem, None


__all__ = ["split_lang_suffix"]
