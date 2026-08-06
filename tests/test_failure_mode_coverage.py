"""Enforces the README's locked constraint: a test for every documented failure mode.

The registry in tools/planted_defects.py is the machine-readable failure-mode list;
this test pins its three-way sync: each entry's anchor exists in its documenting file
(README or check docstring), its catching tests exist in the suite, and its target
snippet exists exactly once in the source. tools/planted_defects.py replays each
defect against a throwaway copy and proves the named tests catch it.
"""

from __future__ import annotations

import importlib.util
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _registry():
    path = ROOT / "tools" / "planted_defects.py"
    assert path.exists(), "tools/planted_defects.py is missing: no failure-mode registry"
    spec = importlib.util.spec_from_file_location("planted_defects", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclass annotation resolution needs the module registered
    spec.loader.exec_module(module)
    return module.REGISTRY


def _suite_test_names() -> set[str]:
    names: set[str] = set()
    for p in sorted((ROOT / "tests").glob("test_*.py")):
        names.update(re.findall(r"^def (test_\w+)\(", p.read_text(encoding="utf-8"), re.M))
    return names


def test_every_registered_failure_mode_is_documented_and_tested():
    registry = _registry()
    assert registry, "the failure-mode registry is empty"

    suite = _suite_test_names()
    problems: list[str] = []

    for entry in registry:
        doc_text = (ROOT / entry.doc_file).read_text(encoding="utf-8")
        if entry.doc_anchor not in doc_text:
            problems.append(
                f"{entry.defect_id}: anchor not found in {entry.doc_file}: {entry.doc_anchor!r}"
            )
        for node in entry.caught_by:
            test_name = node.rsplit("::", 1)[-1]
            if test_name not in suite:
                problems.append(f"{entry.defect_id}: catching test does not exist: {node}")
        source = (ROOT / entry.target_file).read_text(encoding="utf-8")
        count = source.count(entry.old)
        if count != 1:
            problems.append(
                f"{entry.defect_id}: target snippet occurs {count} times in "
                f"{entry.target_file} (need exactly 1)"
            )

    assert not problems, "failure-mode registry out of sync:\n  " + "\n  ".join(problems)


def test_readme_states_the_registry_size():
    """The README quotes the mode count in four places. Registering a mode without
    updating them leaves the published figure describing a smaller registry."""
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    size = len(_registry())
    counted = re.findall(r"(?:registry of|all|each of the|floor for the) (\d+)", readme)
    quoted = {int(n) for n in counted}
    assert quoted == {size}, f"README quotes {sorted(quoted)} modes; the registry holds {size}"


def test_ci_runs_the_replay_on_a_version_in_the_matrix():
    """The replay runs on one interpreter, selected by an `if:` naming a version.
    Retire that version from the matrix and the step stops running while the job
    still reports success -- a gate that replays nothing, which is the failure the
    replay exists to catch, one level up."""
    workflow = (ROOT / ".github" / "workflows" / "python-package.yml").read_text(encoding="utf-8")
    assert "python tools/planted_defects.py" in workflow, "CI does not run the replay"

    matrix = re.search(r"python-version:\s*\[([^\]]+)\]", workflow)
    assert matrix, "no python-version matrix found"
    versions = set(re.findall(r"[\d.]+t?", matrix.group(1)))

    guard = re.search(r"if:\s*matrix\.python-version\s*==\s*'([^']+)'", workflow)
    assert guard, "the replay step has no matrix guard to check"
    gated_on = guard.group(1)
    assert gated_on in versions, (
        f"the replay is gated on python {gated_on}, absent from the matrix {sorted(versions)}"
    )


def test_registry_covers_all_three_public_checks():
    checks = {entry.check for entry in _registry()}
    assert checks == {"check_leakage", "check_schema", "check_stateless"}, (
        f"registry covers {sorted(checks)}; every public check must have at least one "
        "registered failure mode"
    )
