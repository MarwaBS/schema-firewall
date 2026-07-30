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


def test_registry_covers_all_three_public_checks():
    checks = {entry.check for entry in _registry()}
    assert checks == {"check_leakage", "check_schema", "check_stateless"}, (
        f"registry covers {sorted(checks)}; every public check must have at least one "
        "registered failure mode"
    )
