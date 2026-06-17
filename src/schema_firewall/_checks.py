"""Three public checks: leakage, schema, statelessness.

Each function either returns None (pass) or raises a specific
exception (fail). No check returns a truthy/falsy value; failures
carry diagnostic context in the exception message.

All logic is deterministic, stateless, and row-wise where applicable.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Sequence
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression

from ._exceptions import LeakageError, SchemaError, StatelessnessError
from ._schema import SchemaContract

# --- Public: leakage detection ---------------------------------------


def check_leakage(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    max_abs_corr: float = 0.95,
    mi_threshold: float = 0.8,
) -> None:
    """Fail if any column in X shows suspicious dependency with y.

    Runs three complementary detectors per numeric column:

    - Pearson |r|  > ``max_abs_corr`` -> linear leakage
    - Spearman |rho| > ``max_abs_corr`` -> monotonic leakage (catches
      log-transforms, rank re-encodings, expm1-of-log, etc.)
    - normalised mutual information > ``mi_threshold`` -> general,
      including non-monotonic, dependency

    MI is normalised by the target's *self-information* ``MI(y; y)`` computed
    with the SAME estimator (sklearn ``mutual_info_regression``), giving a
    scale-free ratio in [0, 1] where 1.0 means "as informative about y as y is
    about itself" (a copy). Normalising by a histogram Shannon entropy instead
    (as a prior version did) mixes a continuous MI estimate with a binned
    entropy and lets the ratio exceed 1, with semantics that drift with the bin
    count. NaNs are dropped pairwise per column before every metric, so a single
    missing value no longer crashes the check. The default thresholds are
    conservative -- they flag features that are almost certainly target-derived.
    Tune for your domain.

    The target ``y`` must be numeric: all three detectors are defined on a
    continuous target. Encode classification labels (e.g. ``LabelEncoder``)
    before calling.

    Raises:
        LeakageError: one or more columns crossed at least one detector's
            threshold, or the target is unusable (non-numeric, constant, or
            with fewer than two finite values). The message lists every
            violating column with all three metrics.
    """
    numeric = X.select_dtypes(include=[np.number])
    if numeric.empty:
        return

    try:
        y_arr = np.asarray(y, dtype=float)
    except (TypeError, ValueError) as exc:
        raise LeakageError(
            "check_leakage requires a numeric target; got a non-numeric y. "
            "Encode classification labels (e.g. with a LabelEncoder) first."
        ) from exc

    y_mask = np.isfinite(y_arr)
    if y_mask.sum() < 2:
        raise LeakageError("target has fewer than two finite values; leakage check undefined")
    y_finite = y_arr[y_mask]
    if y_finite.std() == 0:
        raise LeakageError("target is constant or all-NaN; leakage check undefined")

    # Self-information baseline MI(y; y), same estimator as the per-column MI,
    # so mi / mi_self is a consistent ratio in [0, 1] (1.0 == a copy of y).
    mi_self = float(mutual_info_regression(y_finite.reshape(-1, 1), y_finite, random_state=0)[0])
    if mi_self <= 0:
        raise LeakageError("target self-information is non-positive; leakage check undefined")

    violations: list[str] = []
    for col in numeric.columns:
        feat = numeric[col].to_numpy(dtype=float)
        # Drop rows where the feature OR the target is non-finite, so a single
        # NaN does not crash mutual_info_regression (which rejects NaN).
        mask = np.isfinite(feat) & y_mask
        if mask.sum() < 2:
            continue
        feat_m, y_m = feat[mask], y_arr[mask]

        pearson = abs(_safe_corr(feat, y_arr, method="pearson"))
        spearman = abs(_safe_corr(feat, y_arr, method="spearman"))
        mi = float(mutual_info_regression(feat_m.reshape(-1, 1), y_m, random_state=0)[0])
        mi_norm = min(1.0, max(0.0, mi / mi_self))

        if pearson > max_abs_corr or spearman > max_abs_corr or mi_norm > mi_threshold:
            violations.append(
                f"{col}: pearson={pearson:.3f} spearman={spearman:.3f} mi_norm={mi_norm:.3f}"
            )

    if violations:
        raise LeakageError("target-correlated feature(s) detected:\n  " + "\n  ".join(violations))


# --- Public: schema contract validation ------------------------------


def check_schema(X: pd.DataFrame, contract: SchemaContract) -> None:
    """Validate X against ``contract``.

    Failure modes, in order:

    1. Any ``forbidden_columns`` entry is present in X -> SchemaError.
    2. Any ``required_columns`` entry is missing from X -> SchemaError.
    3. Any column listed in ``contract.dtypes`` has a mismatched
       dtype -> SchemaError.
    """
    present_forbidden = sorted(set(X.columns) & set(contract.forbidden_columns))
    if present_forbidden:
        raise SchemaError(f"forbidden column(s) present in X: {present_forbidden}")

    missing_required = sorted(set(contract.required_columns) - set(X.columns))
    if missing_required:
        raise SchemaError(f"required column(s) missing from X: {missing_required}")

    if contract.dtypes:
        dtype_violations: list[str] = []
        for col, expected in contract.dtypes.items():
            if col not in X.columns:
                continue  # covered by required_columns check above
            actual_dtype = X[col].dtype
            # Compare resolved dtypes, not raw strings, so equivalent spellings
            # match: "int", "i8", "<i8" all resolve to int64. A raw string
            # compare rejected them against the actual "int64".
            try:
                expected_dtype = pd.api.types.pandas_dtype(expected)
                matches = expected_dtype == actual_dtype
            except TypeError:
                # `expected` is not a recognised dtype string; fall back to exact
                # string match so a typo'd contract still fails loudly.
                matches = str(actual_dtype) == expected
            if not matches:
                dtype_violations.append(f"{col}: expected {expected!r}, got {str(actual_dtype)!r}")
        if dtype_violations:
            raise SchemaError("dtype violation(s):\n  " + "\n  ".join(dtype_violations))


# --- Public: statelessness check -------------------------------------


def check_stateless(
    pipeline_fn: Callable[[pd.DataFrame], pd.DataFrame],
    raw: pd.DataFrame,
    *,
    sample_indices: Sequence[Hashable] | None = None,
) -> None:
    """Fail if ``pipeline_fn`` is not deterministic or not stateless.

    Two invariants are tested:

    1. **Determinism.** ``pipeline_fn(raw)`` produces identical output
       when called twice on the same input.
    2. **Row-wise statelessness.** For each row in ``sample_indices``
       (default: first kept row), ``pipeline_fn`` applied to a
       one-row subset must produce the same output for that row as
       ``pipeline_fn(raw)`` did. This is a strictly harder constraint
       than shuffling -- mean-encoders, rank transforms, and
       frequency encoders all fail it because their output for a
       given row depends on the rest of the dataset.

    Catches:

    - Santander-style frequency encoding across (train union test).
    - Target encoding fit on full data instead of per-fold.
    - ComBat / global normalisation applied outside cross-validation.
    - Any non-deterministic transform (unseeded random, dict-order
      dependency, etc.).

    Args:
        pipeline_fn: callable that takes a frame and returns a frame.
            Must preserve the input index for state-independence
            checking.
        raw: the input frame to exercise.
        sample_indices: which rows to spot-check for state-independence.
            Defaults to the first row of the deterministic output.

    Raises:
        StatelessnessError: pipeline is non-deterministic or
            state-dependent. Message identifies which invariant
            failed.
    """
    first = pipeline_fn(raw)
    second = pipeline_fn(raw)

    try:
        pd.testing.assert_frame_equal(first, second)
    except AssertionError as exc:
        raise StatelessnessError(
            f"pipeline is non-deterministic (two runs differ):\n  {exc}"
        ) from exc

    n = len(first)
    if n == 0:
        return

    # Index-preservation precondition. The spot-check aligns rows by index, so
    # pipeline_fn(raw) must keep rows under their ORIGINAL labels. A row-wise
    # filter that drops rows is fine (it returns a subset of raw.index); a
    # transform that resets/relabels the index is not. Name this precondition
    # explicitly here, rather than letting it surface later as a confusing
    # "sample_indices ... not in raw.index" error the caller never caused.
    if not first.index.isin(raw.index).all():
        raise StatelessnessError(
            "pipeline_fn must preserve the input index: pipeline_fn(raw) returned "
            "index labels not present in raw.index (the pipeline reset or "
            "relabelled the index). Re-emit each row under its original index so "
            "per-row spot-checks can be aligned."
        )

    if sample_indices is None:
        # Spread five spot-checks across the frame so the check isn't
        # fooled by the first row happening to sit in a singleton group.
        step = max(1, n // 5)
        sample_indices = [first.index[i] for i in range(0, n, step)][:5]

    for idx in sample_indices:
        if idx not in raw.index:
            raise ValueError(
                f"sample_indices contains {idx!r}, which is not in raw.index; "
                f"every sample index must appear in raw so the spot-check has "
                f"something to compare against"
            )
        if idx not in first.index:
            raise ValueError(
                f"sample index {idx!r} is in raw.index but was dropped by "
                f"pipeline_fn from the full-frame output; spot-check a row the "
                f"pipeline keeps (one in pipeline_fn(raw).index)"
            )
        single_out = pipeline_fn(raw.loc[[idx]].copy())
        if len(single_out) == 0:
            # The full frame KEEPS this row but the one-row subset DROPS it: the
            # keep/drop decision depends on the other rows. That is exactly the
            # state-dependence this check exists to catch — the old code skipped
            # it with `continue`, a false negative on global-statistic filters.
            raise StatelessnessError(
                f"pipeline is state-dependent at index {idx!r}: the row is kept "
                f"when the full frame is processed but dropped when that row is "
                f"processed alone. A stateless row-wise transform keeps it either "
                f"way; a global-statistic filter (e.g. df[df.x > df.x.median()]) "
                f"drops it."
            )
        try:
            pd.testing.assert_frame_equal(
                first.loc[[idx]].reset_index(drop=True),
                single_out.reset_index(drop=True),
            )
        except AssertionError as exc:
            raise StatelessnessError(
                f"pipeline is state-dependent at index {idx!r} "
                f"(one-row subset != same row from full frame):\n  {exc}"
            ) from exc


# --- Internal helpers ------------------------------------------------


def _safe_corr(a: np.ndarray, b: np.ndarray, *, method: Literal["pearson", "spearman"]) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 2:
        return 0.0
    a, b = a[mask], b[mask]
    if a.std() == 0 or b.std() == 0:
        return 0.0
    if method == "pearson":
        return float(np.corrcoef(a, b)[0, 1])
    if method == "spearman":
        return float(pd.Series(a).corr(pd.Series(b), method="spearman"))
    # Reachable only from an untyped caller passing a bogus method name;
    # typed callers are constrained by Literal at static time.
    raise ValueError(f"unknown correlation method {method!r}; expected 'pearson' or 'spearman'")


__all__ = ["check_leakage", "check_schema", "check_stateless"]
