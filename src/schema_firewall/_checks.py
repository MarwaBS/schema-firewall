"""Three public checks: leakage, schema, statelessness.

Each function either returns None (pass) or raises a specific
exception (fail). No check returns a truthy/falsy value; failures
carry diagnostic context in the exception message.

All logic is deterministic, stateless, and row-wise where applicable.
"""
from __future__ import annotations

from typing import Callable, Literal

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

    MI is normalised by the target's histogram-based Shannon entropy,
    giving a scale-free ratio in [0, 1]. The default thresholds are
    conservative -- they flag features that are almost certainly
    target-derived. Tune for your domain.

    Raises:
        LeakageError: one or more columns crossed at least one
            detector's threshold. The message lists every violating
            column with all three metrics.
    """
    numeric = X.select_dtypes(include=[np.number])
    if numeric.empty:
        return

    y_arr = np.asarray(y, dtype=float)
    y_entropy = _shannon_entropy(y_arr)
    if y_entropy <= 0:
        raise LeakageError(
            "target is constant or all-NaN; leakage check undefined"
        )

    violations: list[str] = []
    for col in numeric.columns:
        feat = numeric[col].to_numpy(dtype=float)
        if not np.isfinite(feat).any():
            continue

        pearson = abs(_safe_corr(feat, y_arr, method="pearson"))
        spearman = abs(_safe_corr(feat, y_arr, method="spearman"))
        mi = float(
            mutual_info_regression(
                feat.reshape(-1, 1), y_arr, random_state=0
            )[0]
        )
        mi_norm = mi / y_entropy

        if (
            pearson > max_abs_corr
            or spearman > max_abs_corr
            or mi_norm > mi_threshold
        ):
            violations.append(
                f"{col}: pearson={pearson:.3f} "
                f"spearman={spearman:.3f} mi_norm={mi_norm:.3f}"
            )

    if violations:
        raise LeakageError(
            "target-correlated feature(s) detected:\n  "
            + "\n  ".join(violations)
        )


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
        raise SchemaError(
            f"forbidden column(s) present in X: {present_forbidden}"
        )

    missing_required = sorted(set(contract.required_columns) - set(X.columns))
    if missing_required:
        raise SchemaError(
            f"required column(s) missing from X: {missing_required}"
        )

    if contract.dtypes:
        dtype_violations: list[str] = []
        for col, expected in contract.dtypes.items():
            if col not in X.columns:
                continue  # covered by required_columns check above
            actual = str(X[col].dtype)
            if actual != expected:
                dtype_violations.append(
                    f"{col}: expected {expected!r}, got {actual!r}"
                )
        if dtype_violations:
            raise SchemaError(
                "dtype violation(s):\n  " + "\n  ".join(dtype_violations)
            )


# --- Public: statelessness check -------------------------------------

def check_stateless(
    pipeline_fn: Callable[[pd.DataFrame], pd.DataFrame],
    raw: pd.DataFrame,
    *,
    sample_indices: list | None = None,
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
        single_out = pipeline_fn(raw.loc[[idx]].copy())
        if len(single_out) == 0:
            continue
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

def _safe_corr(
    a: np.ndarray, b: np.ndarray, *, method: Literal["pearson", "spearman"]
) -> float:
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
    raise ValueError(
        f"unknown correlation method {method!r}; expected 'pearson' or 'spearman'"
    )


def _shannon_entropy(x: np.ndarray, *, bins: int = 64) -> float:
    x = x[np.isfinite(x)]
    if x.size < 2 or x.std() == 0:
        return 0.0
    hist, _ = np.histogram(x, bins=bins, density=False)
    p = hist / hist.sum()
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


__all__ = ["check_leakage", "check_schema", "check_stateless"]
