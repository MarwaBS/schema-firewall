"""Pins the README's quoted demo R² values against drift.

The README documents `examples/leakage_demo.py` produces leaky R² = 0.9495
and honest R² = 0.4384. Without a test, sklearn or numpy version drift could
silently move those numbers, leaving the README stale. This module runs the
demo as a subprocess (the actual user experience) and asserts the printed
R² values stay within ±0.005 of the documented numbers.

A failure here means EITHER (a) drift the README to the new numbers, OR
(b) pin a tighter dependency floor to lock the old numbers. Both are
explicit decisions, not silent staleness.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
DEMO_PATH = REPO_ROOT / "examples" / "leakage_demo.py"

# README claims at README.md:87
README_LEAKY_R2 = 0.9495
README_HONEST_R2 = 0.4384
DRIFT_TOLERANCE = 0.005


def _run_demo() -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"  # demo uses unicode box-drawing chars
    return subprocess.run(
        [sys.executable, str(DEMO_PATH)],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )


def _parse_r2(stdout: str, label: str) -> float:
    # Demo prints e.g. "  leaky                R² = 0.9495" — `R\S*` munches
    # the "²" superscript without depending on its codepoint.
    pattern = rf"{label}\s+R\S*\s*=\s*(\d+\.\d+)"
    match = re.search(pattern, stdout)
    assert match, f"R-squared for {label!r} not found in demo output:\n{stdout}"
    return float(match.group(1))


def test_demo_runs_and_R2_matches_README_claim():
    """Demo must exit 0 and both quoted R² values must be within ±0.005.

    Bundled assertion (single subprocess invocation) covers three things:
    1. Demo completes — proves catch_leak_via_leakage_check still detects
       the leak (it raises AssertionError if check_leakage stops tripping).
    2. Leaky R² ≈ 0.9495 (README.md:87).
    3. Honest R² ≈ 0.4384 (README.md:87).
    """
    result = _run_demo()

    assert result.returncode == 0, (
        f"demo exited with code {result.returncode}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )

    leaky_r2 = _parse_r2(result.stdout, "leaky")
    honest_r2 = _parse_r2(result.stdout, "honest")

    assert abs(leaky_r2 - README_LEAKY_R2) < DRIFT_TOLERANCE, (
        f"leaky R² drift: {leaky_r2:.4f} vs README {README_LEAKY_R2}"
    )
    assert abs(honest_r2 - README_HONEST_R2) < DRIFT_TOLERANCE, (
        f"honest R² drift: {honest_r2:.4f} vs README {README_HONEST_R2}"
    )
