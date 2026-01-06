from sub_translate.service import resolve_api


def test_resolve_api_defaults_to_google() -> None:
    assert resolve_api(None) == "google"


def test_resolve_api_nllb_variants() -> None:
    assert resolve_api("nllb") == "nllb"
    assert resolve_api("nllb-lite") == "nllb-lite"
    assert resolve_api("nllb-3.3b") == "nllb"


def test_resolve_api_seamless_variants() -> None:
    assert resolve_api("seamless") == "seamless"
    assert resolve_api("seamless-m4t") == "seamless"


def test_resolve_api_madlad_variants() -> None:
    assert resolve_api("madlad") == "madlad"
    assert resolve_api("madlad-400") == "madlad"
