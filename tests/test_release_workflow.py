import re
from pathlib import Path

import pytest

_WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "python-publish.yml"
_ACTION_REFS = {
    "actions/checkout": "v4",
    "actions/setup-python": "v5",
    "astral-sh/setup-uv": "v7",
    "actions/upload-artifact": "v4",
    "actions/download-artifact": "v4",
    "pypa/gh-action-pypi-publish": "release/v1",
}
_PINNED_ACTION = re.compile(r"\s*(?:-\s+)?uses:\s+([\w-]+/[\w-]+)@([0-9a-f]{40})\s+#\s+(\S+)\s*")


def _assert_pinned_release_actions(workflow: str) -> None:
    actions = []
    for line in workflow.splitlines():
        if not re.search(r"\buses\b", line.partition("#")[0]):
            continue
        match = _PINNED_ACTION.fullmatch(line)
        assert match is not None, f"Неподдержанное или незакреплённое объявление Action: {line}"
        action, _, original_ref = match.groups()
        assert action in _ACTION_REFS, f"Неучтённая Action: {action}"
        assert original_ref == _ACTION_REFS[action], f"Изменена исходная версия {action}"
        actions.append(action)
    assert sorted(actions) == sorted(_ACTION_REFS), "Ожидается ровно шесть Actions выпуска без повторов"


def test_release_workflow_pins_all_actions() -> None:
    _assert_pinned_release_actions(_WORKFLOW.read_text(encoding="utf-8"))


@pytest.mark.parametrize("action", _ACTION_REFS)
@pytest.mark.parametrize("mutable_ref", ["v4", "main", "a" * 12])
def test_release_workflow_rejects_mutable_or_short_refs(action: str, mutable_ref: str) -> None:
    workflow = _WORKFLOW.read_text(encoding="utf-8")
    changed, count = re.subn(re.escape(action) + r"@[0-9a-f]{40}", f"{action}@{mutable_ref}", workflow)
    assert count == 1
    with pytest.raises(AssertionError, match="Неподдержанное или незакреплённое"):
        _assert_pinned_release_actions(changed)


@pytest.mark.parametrize(
    "declaration",
    [
        '- "uses": actions/checkout@{sha} # v4',
        "- {{uses: actions/checkout@{sha}}} # v4",
        "- uses: >\n    actions/checkout@{sha} # v4",
        "- uses: 'actions/checkout@{sha}' # v4",
        "- uses: actions/checkout@{sha}",
        "- uses: actions/checkout@{sha} # v5",
        "- uses: other/checkout@{sha} # v4",
    ],
)
def test_release_workflow_rejects_unsupported_declarations(declaration: str) -> None:
    workflow = _WORKFLOW.read_text(encoding="utf-8")
    changed, count = re.subn(
        r"(?m)^\s*- uses: actions/checkout@[^\n]+",
        declaration.format(sha="a" * 40),
        workflow,
    )
    assert count == 1
    with pytest.raises(AssertionError):
        _assert_pinned_release_actions(changed)


@pytest.mark.parametrize("change", ["remove", "duplicate", "extra"])
def test_release_workflow_rejects_changed_action_inventory(change: str) -> None:
    workflow = _WORKFLOW.read_text(encoding="utf-8")
    declaration = next(line for line in workflow.splitlines() if "uses: actions/checkout@" in line)
    if change == "remove":
        workflow = workflow.replace(declaration, "", 1)
    elif change == "duplicate":
        workflow += f"\n{declaration}\n"
    else:
        workflow += "\n- uses: other/action@main\n"
    with pytest.raises(AssertionError):
        _assert_pinned_release_actions(workflow)
