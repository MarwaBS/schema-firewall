"""Pins the README's quoted demo R^2 values against drift.

The README's "Verified invariants under execution" section documents that
`examples/leakage_demo.py` produces a specific leaky R^2 and honest R^2.
Without a test, sklearn or numpy version drift could silently move those
numbers, leaving the README stale. This module reads the claimed values from
README.md itself, runs the demo as a subprocess, and asserts the printed R^2
values stay within +/-0.005 of the claim, so README and demo can't diverge.

A failure means EITHER drift the README to the new numbers OR pin a tighter
dependency floor to lock the old ones -- an explicit decision, not staleness.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
DEMO_PATH = REPO_ROOT / "examples" / "leakage_demo.py"
README_PATH = REPO_ROOT / "README.md"

DRIFT_TOLERANCE = 0.005


def _readme_claimed_r2() -> tuple[float, float]:
    """Extract the (leaky, honest) R^2 values the README claims for the demo.

    Reads the claim from where it lives rather than copying it here. Each
    pattern must match exactly once; zero or multiple matches mean the README
    wording changed and this parser must be updated deliberately.
    """
    text = README_PATH.read_text(encoding="utf-8")
    leaky = re.findall(r"R\S* = (\d+\.\d+) \(leaky\)", text)
    honest = re.findall(r"R\S* collapses to (\d+\.\d+) \(honest\)", text)
    assert len(leaky) == 1, f"expected exactly one leaky-R^2 claim in README, found {leaky}"
    assert len(honest) == 1, f"expected exactly one honest-R^2 claim in README, found {honest}"
    return float(leaky[0]), float(honest[0])


def _run_demo() -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"  # decode demo output regardless of console codepage
    return subprocess.run(
        [sys.executable, str(DEMO_PATH)],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )


def _parse_r2(stdout: str, label: str) -> float:
    # Demo prints "leaky   R^2 = 0.9495"; R\S* spans the "^2" without hardcoding it.
    pattern = rf"{label}\s+R\S*\s*=\s*(\d+\.\d+)"
    match = re.search(pattern, stdout)
    assert match, f"R-squared for {label!r} not found in demo output:\n{stdout}"
    return float(match.group(1))


def test_demo_runs_and_R2_matches_README_claim():
    """Demo must exit 0 and both quoted R^2 values must be within +/-0.005.

    Bundled assertion (single subprocess invocation) covers three things:
    1. Demo completes -- proves catch_leak_via_leakage_check still detects
       the leak (it raises AssertionError if check_leakage stops tripping).
    2. Leaky R^2 matches the README's quoted value.
    3. Honest R^2 matches the README's quoted value.
    """
    readme_leaky_r2, readme_honest_r2 = _readme_claimed_r2()
    result = _run_demo()

    assert result.returncode == 0, (
        f"demo exited with code {result.returncode}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )

    leaky_r2 = _parse_r2(result.stdout, "leaky")
    honest_r2 = _parse_r2(result.stdout, "honest")

    assert abs(leaky_r2 - readme_leaky_r2) < DRIFT_TOLERANCE, (
        f"leaky R^2 drift: {leaky_r2:.4f} vs README {readme_leaky_r2}"
    )
    assert abs(honest_r2 - readme_honest_r2) < DRIFT_TOLERANCE, (
        f"honest R^2 drift: {honest_r2:.4f} vs README {readme_honest_r2}"
    )
