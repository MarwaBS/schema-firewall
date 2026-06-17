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
from sklearn.metrics import adjusted_mutual_info_score

from ._exceptions import LeakageError, SchemaError, StatelessnessError
from ._schema import SchemaContract

# Below this many finite paired samples, the binned adjusted-MI estimate is
# noise-dominated: independent/legit features can score as high as a real
# non-monotone leak, so there is no honest threshold (and a 2-point correlation
# is always |r|=1). Empirically, at >= 100 samples deterministic leaks —
# including binary/k-class targets and non-monotone transforms — separate
# cleanly from noise (0% miss, 0 false positives across seeds). Below it, refuse
# rather than guess.
_MIN_SAMPLES = 100

# --- Public: leakage detection ---------------------------------------


def check_leakage(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    max_abs_corr: float = 0.95,
    mi_threshold: float = 0.2,
) -> None:
    """Fail if any column in X shows suspicious dependency with y.

    Runs three complementary detectors per numeric column:

    - Pearson |r|  > ``max_abs_corr`` -> linear leakage
    - Spearman |rho| > ``max_abs_corr`` -> monotonic leakage (catches
      log-transforms, rank re-encodings, expm1-of-log, etc.)
    - normalised mutual information > ``mi_threshold`` -> general dependency,
      INCLUDING non-monotone relationships (``y = x**2``, ``y = |x|``,
      ``y = cos(3x)``) that Pearson and Spearman both miss

    The MI is a *discrete normalised mutual information*: both the feature and
    the target are quantile-binned and scored with sklearn's
    ``normalized_mutual_info_score(..., average_method="min")``. The result is a
    single consistent estimator genuinely bounded in [0, 1] -- 1.0 when either
    variable determines the other (a copy OR a deterministic non-monotone
    transform), ~0 under independence. (A previous version divided a continuous
    kNN MI estimate by a self-MI baseline; that estimator was so deflated it
    only fired on an exact copy, so the MI detector caught nothing the linear/
    rank detectors didn't -- the non-monotone case ``y = x**2`` slipped through.
    The discrete NMI fixes that and also removes the small-``n`` crash the kNN
    estimator raised for ``n <= n_neighbors``.) NaNs are dropped pairwise per
    column before every metric, so a single missing value never crashes the
    check.

    The target ``y`` must be numeric: all three detectors are defined on a
    continuous (or integer-encoded) target. Encode classification labels (e.g.
    ``LabelEncoder``) before calling.

    ``mi_threshold`` is an adjusted-MI threshold in [0, 1] (default 0.2):
    deterministic dependence — copies, k-class/binary target encodings, and
    non-monotone transforms — lands well above it, while honest noisy predictors
    and independent columns land near 0. At least ``100`` rows are required; bins
    scale with sample size to keep the estimate stable.

    Raises:
        ValueError: fewer than ``100`` finite paired samples (leakage detection
            is noise-dominated below that).
        LeakageError: one or more columns crossed at least one detector's
            threshold, or the target is non-numeric or constant. The message
            lists every violating column with all three metrics.
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
    if int(y_mask.sum()) < _MIN_SAMPLES:
        raise ValueError(
            f"check_leakage needs at least {_MIN_SAMPLES} finite samples for "
            f"reliable detection; got {int(y_mask.sum())}. With fewer, sampling "
            f"noise dominates correlation and mutual information."
        )
    if y_arr[y_mask].std() == 0:
        raise LeakageError("target is constant or all-NaN; leakage check undefined")

    violations: list[str] = []
    for col in numeric.columns:
        feat = numeric[col].to_numpy(dtype=float)
        # Drop rows where the feature OR the target is non-finite, so a single
        # NaN does not poison a metric. A column with too few finite paired rows
        # can't be assessed reliably, so skip it rather than risk a noise-driven
        # false positive.
        mask = np.isfinite(feat) & y_mask
        if int(mask.sum()) < _MIN_SAMPLES:
            continue
        feat_m, y_m = feat[mask], y_arr[mask]

        pearson = abs(_safe_corr(feat, y_arr, method="pearson"))
        spearman = abs(_safe_corr(feat, y_arr, method="spearman"))
        mi_norm = _normalised_mi(feat_m, y_m)

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

    # Unique-index precondition. The spot-check selects a single row by label
    # (raw.loc[[label]]); with duplicate labels that pulls EVERY row sharing the
    # label, so the "one-row subset" is the whole frame and the check compares it
    # to itself -- a global transform would sail through. Duplicate indices are
    # routine after pd.concat / repeated timestamps, so refuse rather than give a
    # false pass.
    if not raw.index.is_unique:
        raise StatelessnessError(
            "raw.index has duplicate labels; the per-row spot-check selects by "
            "label, so a one-row subset would pull every row sharing that label "
            "and the check would be vacuous. Pass a unique index "
            "(e.g. raw.reset_index(drop=True))."
        )

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
        # Spot-check the EXTREME-value rows of every numeric column plus an even
        # spread. A global-statistic transform (winsorise/clip/robust-scale,
        # quantile filter) only edits tail rows, so a fixed-stride sample
        # routinely misses it; the rows holding each column's min and max are
        # exactly the ones such a transform touches. (Pass an explicit
        # `sample_indices` to check more rows; checking every row is the
        # strongest, at one pipeline call per row.)
        picks: list[Hashable] = []
        kept_numeric = raw.loc[first.index].select_dtypes(include=[np.number])
        for col in kept_numeric.columns:
            s = kept_numeric[col].dropna()
            if not s.empty:
                picks.append(s.idxmin())
                picks.append(s.idxmax())
        step = max(1, n // 5)
        picks.extend(first.index[i] for i in range(0, n, step))
        seen: set = set()
        deduped: list[Hashable] = []
        for i in picks:
            if i not in seen:
                seen.add(i)
                deduped.append(i)
        sample_indices = deduped

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


def _discretise(x: np.ndarray, n_bins: int) -> np.ndarray:
    """Discretise a finite 1-D array into integer bin labels.

    Low-cardinality / discrete data (<= ``n_bins`` distinct values, e.g. a
    binary or k-class target) keeps each distinct value as its own bin -- using
    quantile edges there collapses a binary 0/1 target to a single bin and makes
    it invisible to the MI detector. Continuous data is split into ``n_bins``
    equal-frequency (quantile) bins.
    """
    uniq = np.unique(x)
    if uniq.size < 2:
        return np.zeros(x.shape, dtype=np.int64)
    if uniq.size <= n_bins:
        # Discrete / low-cardinality: one bin per distinct value.
        return np.searchsorted(uniq, x).astype(np.int64)
    edges = np.unique(np.quantile(x, np.linspace(0.0, 1.0, n_bins + 1)))
    # Interior edges only; np.digitize then maps values to bin indices.
    return np.digitize(x, edges[1:-1]).astype(np.int64)


def _normalised_mi(feat: np.ndarray, y: np.ndarray) -> float:
    """Adjusted (chance-corrected) mutual information, clamped to [0, 1].

    Both arrays are discretised and scored with sklearn's
    ``adjusted_mutual_info_score`` -- mutual information corrected for the
    agreement expected by chance. Adjustment matters: a plain NMI has a positive
    finite-sample bias, so independent variables score well above 0 at small n
    (a false positive). AMI ~= 0 under independence regardless of n or bin
    count, ~= 1 when one variable determines the other -- INCLUDING a
    non-monotone deterministic transform such as ``y = feat**2`` (each feature
    bin maps to a single target bin). It can be slightly negative by chance, so
    clamp at 0. This is what makes the MI detector catch non-monotone leakage
    that Pearson/Spearman miss, without false-positiving on noise.

    The bin count scales with sample size (sqrt rule, capped at 16): too many
    bins on few samples gives ~1 sample/bin and unstable estimates, so smaller
    frames use coarser bins to keep the estimate stable.
    """
    n = int(feat.shape[0])
    n_bins = min(16, max(4, round(n**0.5)))
    fb = _discretise(feat, n_bins)
    yb = _discretise(y, n_bins)
    if np.unique(fb).size < 2 or np.unique(yb).size < 2:
        # One side is effectively constant after binning -> no dependence.
        return 0.0
    return max(0.0, float(adjusted_mutual_info_score(fb, yb)))


__all__ = ["check_leakage", "check_schema", "check_stateless"]
