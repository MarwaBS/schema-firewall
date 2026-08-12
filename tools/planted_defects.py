"""Replay planted defects and prove the suite catches each one.

For every documented failure mode in REGISTRY, this script copies the project to a
throwaway directory, plants the defect there (never in the real tree), and runs the
named tests against the copy: they must FAIL with the defect planted and PASS on a
pristine copy. Exit 0 only if every defect is caught and every control is green.

Usage: python tools/planted_defects.py [--defect ID]
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclasses.dataclass(frozen=True)
class PlantedDefect:
    defect_id: str
    check: str  # public check whose documented failure mode this blinds
    failure_mode: str
    doc_file: str  # where the failure mode is documented
    doc_anchor: str  # exact phrase that must exist in doc_file
    target_file: str
    old: str  # unique source snippet to sabotage
    new: str
    caught_by: tuple[str, ...]  # test node ids that must go RED


_CHECKS = "src/schema_firewall/_checks.py"

# Every child bounded. Slowest measured child is the 25-node control run at
# 7.8s, so 300 fires only on a hang. It has to stay under the tightest job
# ceiling, `timeout-minutes: 10`, or CI is cancelled before the bound can fire.
_CHILD_TIMEOUT_S = 300

REGISTRY: tuple[PlantedDefect, ...] = (
    PlantedDefect(
        defect_id="leakage-raise-disabled",
        check="check_leakage",
        failure_mode="feature is a direct or monotonic copy of the target",
        doc_file="README.md",
        doc_anchor="Pearson catches linear copies",
        target_file=_CHECKS,
        old='if violations:\n        raise LeakageError("target-correlated',
        new='if False:\n        raise LeakageError("target-correlated',
        caught_by=(
            "tests/test_checks.py::test_leakage_catches_direct_target_copy",
            "tests/test_checks.py::test_leakage_catches_monotonic_transform",
            "tests/test_checks.py::test_leakage_catches_near_duplicate_with_small_noise",
        ),
    ),
    PlantedDefect(
        defect_id="mi-binning-quantile-restored",
        check="check_leakage",
        failure_mode="zero-inflated feature collapses every observation into one bin",
        doc_file="README.md",
        doc_anchor="NON-monotone and discrete deterministic leakage",
        target_file=_CHECKS,
        old="return ((ranks * n_bins) // uniq.size).astype(np.int64)",
        new="return ((ranks * n_bins) // x.size).astype(np.int64)",
        caught_by=(
            "tests/test_checks.py::test_leakage_catches_zero_inflated_square_collapse_region",
        ),
    ),
    PlantedDefect(
        defect_id="output-tail-targeting-disabled",
        check="check_stateless",
        failure_mode="global transform edits only the tails of a column the pipeline derives",
        doc_file="README.md",
        doc_anchor="min/max rows of every numeric column",
        target_file=_CHECKS,
        old="for frame in (kept, out):",
        new="for frame in (kept,):",
        caught_by=(
            "tests/test_checks.py::test_stateless_catches_a_winsorise_on_a_column_the_pipeline_derives",
            "tests/test_checks.py::test_stateless_catches_a_duplicate_flag_read_across_rows",
        ),
    ),
    PlantedDefect(
        defect_id="mi-detector-disabled",
        check="check_leakage",
        failure_mode="non-monotone deterministic dependency (squares, abs, k-class encodings)",
        doc_file="README.md",
        doc_anchor="NON-monotone and discrete deterministic leakage",
        target_file=_CHECKS,
        old="mi_norm = _normalised_mi(feat_m, y_m)",
        new="mi_norm = 0.0",
        caught_by=("tests/test_checks.py::test_leakage_catches_non_monotone_dependence",),
    ),
    PlantedDefect(
        defect_id="min-samples-guard-disabled",
        check="check_leakage",
        failure_mode="too few samples for reliable detection (noise-dominated estimate)",
        doc_file="README.md",
        doc_anchor=">=100 rows",
        target_file=_CHECKS,
        old="if int(y_mask.sum()) < _MIN_SAMPLES:",
        new="if False:",
        caught_by=("tests/test_checks.py::test_leakage_small_sample_raises_clear_precondition",),
    ),
    PlantedDefect(
        defect_id="constant-target-guard-disabled",
        check="check_leakage",
        failure_mode="constant target makes the leakage check undefined",
        doc_file=_CHECKS,
        doc_anchor="the target is non-numeric or constant",
        target_file=_CHECKS,
        old="if y_arr[y_mask].std() == 0:",
        new="if False:",
        caught_by=("tests/test_checks.py::test_leakage_raises_on_constant_target",),
    ),
    PlantedDefect(
        defect_id="target-realign-disabled",
        check="check_leakage",
        failure_mode="permuted target index compared against the wrong rows",
        doc_file=_CHECKS,
        doc_anchor="each feature value is compared against its own target",
        target_file=_CHECKS,
        old="y = y.reindex(X.index)",
        new="y = y",
        caught_by=("tests/test_checks.py::test_leakage_realigns_permuted_target_index",),
    ),
    PlantedDefect(
        defect_id="skipped-column-warning-disabled",
        check="check_leakage",
        failure_mode="stringified target copy hides in an uninspected non-numeric column",
        doc_file=_CHECKS,
        doc_anchor="reported with a warning, not a raise",
        target_file=_CHECKS,
        old="if skipped:",
        new="if False:",
        caught_by=(
            "tests/test_checks.py::test_leakage_warns_on_skipped_non_numeric_columns",
            "tests/test_checks.py::test_leakage_warns_when_stringified_target_copy_is_unchecked",
        ),
    ),
    PlantedDefect(
        defect_id="forbidden-column-check-disabled",
        check="check_schema",
        failure_mode="forbidden / post-outcome column present in the input",
        doc_file="README.md",
        doc_anchor="Catches ICD-code-style post-outcome features and schema drift",
        target_file=_CHECKS,
        old="if present_forbidden:",
        new="if False:",
        caught_by=("tests/test_checks.py::test_schema_rejects_forbidden_column",),
    ),
    PlantedDefect(
        defect_id="required-column-check-disabled",
        check="check_schema",
        failure_mode="required column missing from the input",
        doc_file="README.md",
        doc_anchor="forbidden columns, required columns, dtypes",
        target_file=_CHECKS,
        old="if missing_required:",
        new="if False:",
        caught_by=("tests/test_checks.py::test_schema_rejects_missing_required_column",),
    ),
    PlantedDefect(
        defect_id="dtype-check-disabled",
        check="check_schema",
        failure_mode="column dtype drifts from the contract",
        doc_file="README.md",
        doc_anchor="forbidden columns, required columns, dtypes",
        target_file=_CHECKS,
        old="if dtype_violations:",
        new="if False:",
        caught_by=("tests/test_checks.py::test_schema_rejects_dtype_mismatch",),
    ),
    PlantedDefect(
        defect_id="determinism-check-disabled",
        check="check_stateless",
        failure_mode="non-deterministic transform (unseeded random, dict-order dependency)",
        doc_file="README.md",
        doc_anchor="Determinism check catches non-deterministic transforms.",
        target_file=_CHECKS,
        old="pd.testing.assert_frame_equal(first, second, check_exact=True)",
        new="pass",
        caught_by=("tests/test_checks.py::test_stateless_catches_nondeterministic_pipeline",),
    ),
    PlantedDefect(
        defect_id="determinism-tolerance-relaxed",
        check="check_stateless",
        failure_mode="drift below pandas' default 1e-5 relative tolerance reads as equal",
        doc_file="README.md",
        doc_anchor="check_exact=True",
        target_file=_CHECKS,
        old="pd.testing.assert_frame_equal(first, second, check_exact=True)",
        new="pd.testing.assert_frame_equal(first, second)",
        caught_by=(
            "tests/test_checks.py::test_stateless_catches_drift_below_the_default_float_tolerance",
        ),
    ),
    PlantedDefect(
        defect_id="row-spot-check-disabled",
        check="check_stateless",
        failure_mode="per-row output depends on other rows (mean/frequency/target encoders)",
        doc_file="README.md",
        doc_anchor="mean encoders, frequency encoders, target encoders",
        target_file=_CHECKS,
        old="for idx in sample_indices:",
        new="for idx in []:",
        caught_by=(
            "tests/test_checks.py::test_stateless_catches_mean_encoder",
            "tests/test_checks.py::test_stateless_catches_frequency_encoder",
        ),
    ),
    PlantedDefect(
        defect_id="tail-row-targeting-disabled",
        check="check_stateless",
        failure_mode="global transform edits only the tail rows (winsorise/clip)",
        doc_file="README.md",
        doc_anchor="min/max rows of every numeric column",
        target_file=_CHECKS,
        old="picks.append(s.idxmin())\n                    picks.append(s.idxmax())",
        new="pass",
        caught_by=(
            "tests/test_checks.py::test_stateless_catches_tail_edit_on_low_variance_wide_frame",
        ),
    ),
    PlantedDefect(
        defect_id="nan-row-targeting-disabled",
        check="check_stateless",
        failure_mode="global-mean imputation edits only the NaN-bearing rows",
        doc_file="README.md",
        doc_anchor="of each column separately",
        target_file=_CHECKS,
        old="picks.extend(kept.index[kept[col].isna()][:per_column])",
        new="picks.extend(kept.index[kept[col].isna()][:0])",
        caught_by=("tests/test_checks.py::test_stateless_catches_global_mean_imputation",),
    ),
    PlantedDefect(
        defect_id="lone-dirty-column-not-widened",
        check="check_stateless",
        failure_mode="a frame with one dirty column is sampled only three rows deep",
        doc_file=_CHECKS,
        doc_anchor="Widen the per-column pick until the total reaches this",
        target_file=_CHECKS,
        old="per_column = max(_NAN_ROWS_PER_COLUMN, _NAN_ROWS_BUDGET // max(1, dirty))",
        new="per_column = _NAN_ROWS_PER_COLUMN",
        caught_by=("tests/test_checks.py::test_stateless_samples_a_lone_dirty_column_more_deeply",),
    ),
    PlantedDefect(
        defect_id="row-tolerance-relaxed-to-default",
        check="check_stateless",
        failure_mode="a frame statistic on a large carrier hides under the default tolerance",
        doc_file=_CHECKS,
        doc_anchor="stops being separable from float noise",
        target_file=_CHECKS,
        old="rtol=_ROW_RELATIVE_TOLERANCE,",
        new="rtol=1e-5,",
        caught_by=(
            "tests/test_checks.py::test_stateless_catches_a_frame_statistic_on_a_large_carrier",
        ),
    ),
    PlantedDefect(
        defect_id="nan-budget-shared-across-columns",
        check="check_stateless",
        failure_mode="a dirtier column spends the NaN budget the leaking column needed",
        doc_file="README.md",
        doc_anchor="a budget shared across columns is spent by whichever column is dirtiest",
        target_file=_CHECKS,
        old=(
            "for col in kept.columns:\n"
            "            picks.extend(kept.index[kept[col].isna()][:per_column])"
        ),
        new="picks.extend(kept.index[kept.isna().any(axis=1)][:per_column])",
        caught_by=(
            "tests/test_checks.py::test_stateless_catches_imputation_behind_a_dirtier_column",
        ),
    ),
    PlantedDefect(
        defect_id="duplicate-index-guard-disabled",
        check="check_stateless",
        failure_mode="duplicate index labels make the one-row spot-check vacuous",
        doc_file=_CHECKS,
        doc_anchor="raw.index has duplicate labels",
        target_file=_CHECKS,
        old="if not raw.index.is_unique:",
        new="if False:",
        caught_by=("tests/test_checks.py::test_stateless_rejects_duplicate_index",),
    ),
)


def _copy_project(dst: Path) -> None:
    for item in ("src", "tests", "pyproject.toml"):
        src = ROOT / item
        if src.is_dir():
            shutil.copytree(src, dst / item)
        else:
            shutil.copy2(src, dst / item)


def _run_tests(project: Path, nodes: tuple[str, ...]) -> subprocess.CompletedProcess:
    env = dict(os.environ, PYTHONPATH=str(project / "src"))
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--no-cov", "-p", "no:cacheprovider", *nodes],
        cwd=project,
        env=env,
        capture_output=True,
        text=True,
        timeout=_CHILD_TIMEOUT_S,
    )


def _assert_isolation(project: Path) -> None:
    env = dict(os.environ, PYTHONPATH=str(project / "src"))
    probe = subprocess.run(
        [sys.executable, "-c", "import schema_firewall; print(schema_firewall.__file__)"],
        cwd=project,
        env=env,
        capture_output=True,
        text=True,
        timeout=_CHILD_TIMEOUT_S,
    )
    resolved = Path(probe.stdout.strip()).resolve()
    if not resolved.is_relative_to(project.resolve()):
        sys.exit(f"ISOLATION FAILURE: tests would import {resolved}, not the throwaway copy")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--defect", help="run a single defect by id")
    args = parser.parse_args()

    selected = [d for d in REGISTRY if args.defect in (None, d.defect_id)]
    if not selected:
        sys.exit(f"unknown defect id {args.defect!r}")

    failures: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        pristine = Path(tmp) / "pristine"
        _copy_project(pristine)
        _assert_isolation(pristine)

        all_nodes = tuple(dict.fromkeys(n for d in selected for n in d.caught_by))
        control = _run_tests(pristine, all_nodes)
        if control.returncode != 0:
            print(control.stdout[-2000:])
            sys.exit("CONTROL FAILURE: named tests do not pass on the pristine tree")
        print(f"control: {len(all_nodes)} named tests green on pristine copy")

        for defect in selected:
            mutated = Path(tmp) / defect.defect_id
            _copy_project(mutated)
            target = mutated / defect.target_file
            source = target.read_text(encoding="utf-8")
            if source.count(defect.old) != 1:
                failures.append(f"{defect.defect_id}: snippet not unique in source")
                continue
            target.write_text(source.replace(defect.old, defect.new), encoding="utf-8")
            result = _run_tests(mutated, defect.caught_by)
            caught = result.returncode != 0
            print(f"{'CAUGHT  ' if caught else 'SURVIVED'} {defect.defect_id}")
            if not caught:
                failures.append(f"{defect.defect_id}: planted defect survived {defect.caught_by}")
            shutil.rmtree(mutated, ignore_errors=True)

    if failures:
        print("\nFAILURES:\n  " + "\n  ".join(failures))
        return 1
    print(f"\nall {len(selected)} planted defects caught; controls green")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
