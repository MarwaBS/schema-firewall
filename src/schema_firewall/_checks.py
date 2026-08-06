"""Three public checks: leakage, schema, statelessness.

Each function either returns None (pass) or raises a specific
exception (fail). No check returns a truthy/falsy value; failures
carry diagnostic context in the exception message.

All logic is deterministic, stateless, and row-wise where applicable.
"""

from __future__ import annotations

import warnings
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
# is always |r|=1). Empirically, at >= 100 samples deterministic leaks --
# including binary/k-class targets and non-monotone transforms -- separate
# cleanly from noise (0% miss, 0 false positives across seeds). Below it, refuse
# rather than guess.
_MIN_SAMPLES = 100

# NaN rows are spot-checked per column rather than across their union: a single
# budget shared by every column lets one dirty column spend it all, leaving the
# column that actually leaks unchecked. One row per column already catches a fill
# that edits every NaN row the same way (60/60 seeds) -- what fillna(df.mean())
# does. Three roughly doubles detection when only half a column's NaN rows are
# edited (34/60 -> 58/60) and costs ~30% more pipeline calls on a 60-column frame;
# past three the curve flattens. A fill touching a small subset stays a coverage
# floor, not a guarantee.
_NAN_ROWS_PER_COLUMN = 3

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
    - adjusted mutual information > ``mi_threshold`` -> non-monotone dependency
      that Pearson and Spearman both miss: squares (``y = x**2``), absolute
      values (``y = |x|``), low-order oscillations (``y = cos(3x)``), bucketing,
      and BINARY / k-class target encodings (the most common ML target shapes)

    The MI is sklearn's ``adjusted_mutual_info_score`` -- mutual information
    corrected for chance -- on discretised values (low-cardinality values get one
    bin each so binary/k-class targets are not collapsed; higher-cardinality
    values are dense-rank binned into ``sqrt(n)`` bins, bounded to [4, 16]).
    Chance correction matters: it is ~0
    under independence regardless of sample size or bin count, and ~1 when one
    variable determines the other. This is what makes the MI detector catch
    non-monotone leakage without false-positiving on honest noisy predictors,
    which a plain (uncorrected) NMI does not. NaNs are dropped pairwise per
    column, so a single missing value never crashes the check.

    LIMITATION: the binned estimator resolves dependence up to a few oscillation
    periods. A pathological *high-frequency* encoding (e.g. ``y = cos(5x)`` or a
    sawtooth) puts several target values inside each feature bin and can evade
    the MI detector. Such encodings are not a realistic leakage pattern and are
    statistically indistinguishable from a strong honest predictor at finite
    samples; do not rely on this detector for them.

    Detection is per-column: a target reconstructable only from a COMBINATION of
    columns (e.g. ``y = x1 XOR x2``) is not caught -- each column alone looks
    independent.

    The target ``y`` must be numeric: all three detectors are defined on a
    continuous (or integer-encoded) target. Encode classification labels (e.g.
    ``LabelEncoder``) before calling.

    Bool feature columns are inspected (True/False as 1/0). Non-numeric
    (object/string) FEATURE columns are NOT: a stringified target copy in one is
    reported with a warning, not a raise. Encode such columns before calling, or
    filter that warning to an error to fail closed on them.

    ``mi_threshold`` is an adjusted-MI threshold in [0, 1] (default 0.2):
    deterministic dependence -- copies, k-class/binary target encodings, and
    non-monotone transforms -- lands well above it. Independent and weakly
    correlated columns land near 0, but the detector scores raw target
    dependence, not its *source*: a genuinely strong honest predictor crosses
    the threshold too. Measured (honest linear feature, 20 seeds, n=400): flags
    0 up to ``|r| <= 0.75``, at most 5% at ``|r| = 0.80``, a minority (~15%) by
    ``|r| = 0.83``, and the majority (~75%) by ``|r| = 0.85``. That is
    deliberate -- a leakage firewall should surface any feature that nearly
    determines the target -- so read a
    flag as "audit this column", not "proven leak". At least ``100`` rows are
    required; bins scale with sample size to keep the estimate stable.

    Composite operating point: a column fails if it crosses ANY detector, so the
    effective gate is the STRICTEST of the three. On a *linear* feature the MI
    detector (flagging by ``|r| ~ 0.85``) therefore binds well before the Pearson
    / Spearman cap (``max_abs_corr = 0.95``) ever would -- i.e. the practical
    leakage threshold for a linear predictor is ``|r| ~ 0.85``, not 0.95. Raise
    ``mi_threshold`` if you want strong-but-honest linear features to pass. The
    Pearson/Spearman caps are kept even though MI usually binds first: they are
    near-free, they name which pillar tripped in the error message, and they
    remain a backstop if a future bin-count change lifts the MI band above 0.95.

    Raises:
        ValueError: a malformed call -- duplicate column names, a non-1-D target,
            an X/y length mismatch, an unalignable ``y.index``, or fewer than
            ``100`` finite paired samples (detection is noise-dominated below that).
        LeakageError: one or more columns crossed at least one detector's
            threshold, or the target is non-numeric or constant. The message
            lists every violating column with all three metrics.
    """
    # Preconditions up front: otherwise these surface deep in the loop as a raw
    # numpy broadcast error or IndexError that doesn't name the cause.
    if not X.columns.is_unique:
        dupes = X.columns[X.columns.duplicated()].unique().tolist()
        raise ValueError(f"check_leakage: X has duplicate column names {dupes}.")
    if np.ndim(y) != 1:
        raise ValueError("check_leakage: y must be 1-dimensional (a Series or 1-D array).")
    if len(y) != len(X):
        raise ValueError(f"check_leakage: length mismatch -- X has {len(X)} rows, y has {len(y)}.")

    # Include bool: True/False is 1/0, so a bool copy of a binary target must be
    # inspected like any numeric copy. Only object/string columns are left out.
    numeric = X.select_dtypes(include=[np.number, "bool"])

    # Warn on uninspected non-numeric columns, so a stringified target copy hiding
    # in one can't pass the firewall unnoticed.
    skipped = [c for c in X.columns if c not in numeric.columns]
    if skipped:
        warnings.warn(
            "check_leakage inspects numeric columns only; NOT checked for "
            f"leakage: {skipped}. A stringified/encoded target copy in a "
            "non-numeric column is invisible to this check -- encode such "
            "columns (e.g. LabelEncoder) before calling if they must be audited.",
            stacklevel=2,
        )

    if numeric.columns.empty:
        # No numeric columns to inspect. Gate on the absence of columns, not on
        # `numeric.empty` (also true for 0 rows), so a 0-row frame falls through
        # to the min-samples precondition instead of a silent clean pass.
        return

    # Realign the target to X by index before reading it positionally, or a
    # differently-ordered y (e.g. a post-split shuffle) is compared against the
    # wrong rows. Same labels reorder; a different label set can't align, so refuse.
    if isinstance(y, pd.Series) and not y.index.equals(X.index):
        same_labels = (
            X.index.is_unique
            and y.index.is_unique
            and X.index.sort_values().equals(y.index.sort_values())
        )
        if not same_labels:
            raise ValueError(
                "check_leakage: y.index does not align with X.index. Pass a target "
                "whose index matches X row-for-row (same labels), or reset both "
                "indices, so each feature value is compared against its own target."
            )
        y = y.reindex(X.index)

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
    under_min: list[str] = []
    for col in numeric.columns:
        feat = numeric[col].to_numpy(dtype=float)
        # Drop rows where the feature OR the target is non-finite, so a single
        # NaN does not poison a metric. A column with too few finite paired rows
        # can't be assessed reliably, so skip it -- but surface the skip (below),
        # or a target copy hiding behind heavy missingness passes unexamined.
        mask = np.isfinite(feat) & y_mask
        n_finite = int(mask.sum())
        if n_finite < _MIN_SAMPLES:
            under_min.append(f"{col} ({n_finite} finite paired rows)")
            continue
        feat_m, y_m = feat[mask], y_arr[mask]

        pearson = abs(_safe_corr(feat, y_arr, method="pearson"))
        spearman = abs(_safe_corr(feat, y_arr, method="spearman"))
        mi_norm = _normalised_mi(feat_m, y_m)

        if pearson > max_abs_corr or spearman > max_abs_corr or mi_norm > mi_threshold:
            violations.append(
                f"{col}: pearson={pearson:.3f} spearman={spearman:.3f} mi_norm={mi_norm:.3f}"
            )

    if under_min:
        warnings.warn(
            f"check_leakage skipped column(s) with fewer than {_MIN_SAMPLES} "
            f"finite paired samples (estimate would be noise-dominated): "
            f"{under_min}. These columns were NOT inspected.",
            stacklevel=2,
        )

    if violations:
        raise LeakageError("target-correlated feature(s) detected:\n  " + "\n  ".join(violations))


# --- Public: schema contract validation ------------------------------


def check_schema(X: pd.DataFrame, contract: SchemaContract) -> None:
    """Validate X against ``contract``.

    Failure modes, in order:

    0. X has duplicate column names -> ValueError (malformed input).
    1. Any ``forbidden_columns`` entry is present in X -> SchemaError.
       Name matching is exact and case-sensitive: ``"sale price"`` does
       not match a ``"SALE PRICE"`` contract entry.
    2. Any ``required_columns`` entry is missing from X -> SchemaError.
    3. Any column listed in ``contract.dtypes`` has a mismatched
       dtype -> SchemaError.
    """
    if not X.columns.is_unique:
        dupes = X.columns[X.columns.duplicated()].unique().tolist()
        raise ValueError(f"check_schema: X has duplicate column names {dupes}.")

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
                continue  # absent columns are not dtype-checked; presence is required_columns' job
            actual_dtype = X[col].dtype
            # Compare resolved dtypes, not raw strings, so equivalent spellings
            # ("int", "i8", "<i8") all match the actual int64.
            try:
                expected_dtype = pd.api.types.pandas_dtype(expected)
            except TypeError:
                # `expected` is not a recognised dtype string; fall back to exact
                # string match so a typo'd contract still fails loudly.
                matches = str(actual_dtype) == expected
            else:
                if (
                    isinstance(expected_dtype, pd.CategoricalDtype)
                    and expected_dtype.categories is None
                ):
                    # Bare "category" means "any categorical": equality against a
                    # column with specific categories is always False, so match on
                    # the dtype kind instead of exact categories.
                    matches = isinstance(actual_dtype, pd.CategoricalDtype)
                else:
                    matches = expected_dtype == actual_dtype
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
       (default: see below), ``pipeline_fn`` applied to a
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
        sample_indices: rows to spot-check for state-independence. Default: every
            numeric column's min/max tail rows, each column's first few NaN rows,
            and a fixed stride -- so a tail or imputation edit on any column is
            covered regardless of that column's scale or how dirty its neighbours
            are. Cost scales with column count, not frame length; pass an explicit
            list to bound it on very wide frames, or every index for full cover.

    Raises:
        StatelessnessError: pipeline is non-deterministic or state-dependent
            (message identifies which invariant failed), or ``raw`` is too small
            or its output empty to verify.
        ValueError: empty ``sample_indices``, or an index not present in ``raw``.
        TypeError: ``pipeline_fn`` returned a non-DataFrame.
    """
    if sample_indices is not None and len(sample_indices) == 0:
        raise ValueError(
            "sample_indices is empty; pass at least one row to spot-check, or None "
            "for the default row selection. An empty list checks nothing and would "
            "pass vacuously."
        )

    if len(raw) < 2:
        raise StatelessnessError(
            f"raw has {len(raw)} row(s): with fewer than 2 rows a one-row subset "
            "equals the full frame (or there is nothing to run), so a "
            "state-dependent transform would pass vacuously. Provide at least 2 rows."
        )

    # Copy per call: an in-place pipeline that returns its input would otherwise
    # alias first/second/raw, so the determinism check compares a frame to itself
    # and the caller's frame is mutated.
    first = pipeline_fn(raw.copy())
    if not isinstance(first, pd.DataFrame):
        raise TypeError(f"pipeline_fn must return a pandas DataFrame; got {type(first).__name__}.")
    second = pipeline_fn(raw.copy())

    try:
        # check_exact: pandas defaults to rtol=1e-5, which passes an unseeded RNG
        # perturbing values below that. Two runs of a deterministic pipeline agree
        # bit-for-bit, so anything else is the defect this invariant names.
        pd.testing.assert_frame_equal(first, second, check_exact=True)
    except AssertionError as exc:
        raise StatelessnessError(
            f"pipeline is non-deterministic (two runs differ):\n  {exc}"
        ) from exc

    n = len(first)
    if n == 0:
        # raw has >= 2 rows (checked above), so an empty output means the pipeline
        # dropped everything -- nothing left to spot-check.
        raise StatelessnessError(
            "pipeline_fn(raw) returned 0 rows, so the per-row spot-check has "
            "nothing to compare and statelessness cannot be verified. If this is "
            "a global-statistic filter, that is the state-dependence to fix; if it "
            "is row-wise, exercise it on an input it keeps rows for."
        )

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
        # Spot-check the rows a global transform is most likely to touch:
        #  - each numeric column's MIN and MAX rows -- winsorise/clip/robust-scale
        #    and quantile filters edit the tails;
        #  - the first few NaN rows OF EACH COLUMN -- global-mean/median
        #    imputation (df.fillna(df.mean())) edits exactly those, and a
        #    one-row subset can't reconstruct the global statistic;
        #  - an even fixed-stride spread for everything else: catch-all padding on
        #    top of the targeted picks, bounded at ~5 extra pipeline calls.
        # A plain stride sample routinely misses tail- or NaN-only edits. (Pass
        # an explicit `sample_indices` to check more rows; checking every row is
        # the strongest, at one pipeline call per row.)
        picks: list[Hashable] = []
        kept = raw.loc[first.index]
        kept_numeric = kept.select_dtypes(include=[np.number])
        for col in kept_numeric.columns:
            s = kept_numeric[col].dropna()
            if not s.empty:
                picks.append(s.idxmin())
                picks.append(s.idxmax())
        for col in kept.columns:
            picks.extend(kept.index[kept[col].isna()][:_NAN_ROWS_PER_COLUMN])
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
            # Full frame KEEPS this row but the one-row subset DROPS it: the
            # keep/drop decision depends on the other rows -- state-dependence.
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
                check_exact=True,
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
    """Discretise a finite 1-D array into integer bin labels by dense rank.

    Each value takes its rank among the distinct sorted values; ranks are then
    folded into ``n_bins`` equal-width bins. Ranking by distinct value rather than
    by observation quantile keeps a heavily repeated value (e.g. a zero-inflated
    feature, 90%+ zeros) from collapsing every observation into one bin and hiding
    a non-monotone leak. Low-cardinality data (<= ``n_bins`` distinct values, e.g.
    a binary or k-class target) keeps one bin per distinct value.
    """
    uniq = np.unique(x)
    ranks = np.searchsorted(uniq, x)
    if uniq.size <= n_bins:
        return ranks.astype(np.int64)
    return ((ranks * n_bins) // uniq.size).astype(np.int64)


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

    The bin count is ``sqrt(n)`` bounded to [4, 16]: too many bins on few samples
    give ~1 sample/bin (unstable), too few flatten a non-monotone fold like
    ``y = x**2`` into apparent independence.
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
