"""Adversarial test suite for schema-firewall.

Each test either injects a known failure and asserts the firewall
raises, or exercises a legitimate path and asserts the firewall
does not raise. Parameterised where failure has multiple shapes.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

from schema_firewall import (
    LeakageError,
    SchemaContract,
    SchemaError,
    StatelessnessError,
    __version__,
    check_leakage,
    check_schema,
    check_stateless,
)

# ───────────────────────────────────────────────────────────────────
# Fixtures
# ───────────────────────────────────────────────────────────────────


@pytest.fixture
def clean_frame() -> tuple[pd.DataFrame, pd.Series]:
    """Feature frame + target with no statistical dependency by construction.

    Each column is an independent random draw; the target is a separate
    independent random draw. Leakage tests that want a real relationship
    must inject one explicitly (they all do).
    """
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame(
        {
            "sqft": rng.uniform(500, 3000, n),
            "year_built": rng.integers(1900, 2020, n),
            "rooms": rng.integers(1, 6, n),
        }
    )
    y = pd.Series(rng.lognormal(mean=13, sigma=0.5, size=n), name="price")
    return df, y


# ───────────────────────────────────────────────────────────────────
# 1. check_leakage
# ───────────────────────────────────────────────────────────────────


def test_leakage_accepts_independent_features(clean_frame):
    x, y = clean_frame
    check_leakage(x, y)  # no raise


def test_leakage_catches_direct_target_copy(clean_frame):
    x, y = clean_frame
    x = x.copy()
    x["leaked"] = y.to_numpy()
    with pytest.raises(LeakageError, match="leaked"):
        check_leakage(x, y)


def test_leakage_catches_monotonic_transform(clean_frame):
    x, y = clean_frame
    x = x.copy()
    x["logged"] = np.log1p(y.to_numpy())
    with pytest.raises(LeakageError, match="logged"):
        check_leakage(x, y)


def test_leakage_catches_nonlinear_transform_via_spearman_or_mi(clean_frame):
    x, y = clean_frame
    x = x.copy()
    x["sigmoid_of_target"] = 1.0 / (1.0 + np.exp(-y.to_numpy() / y.std()))
    with pytest.raises(LeakageError, match="sigmoid_of_target"):
        check_leakage(x, y)


def test_leakage_catches_near_duplicate_with_small_noise(clean_frame):
    x, y = clean_frame
    x = x.copy()
    rng = np.random.default_rng(1)
    x["affluence_index"] = y.to_numpy() + rng.normal(0, 1e-6, len(y))
    with pytest.raises(LeakageError, match="affluence_index"):
        check_leakage(x, y)


def test_leakage_raises_on_constant_target(clean_frame):
    x, _ = clean_frame
    y = pd.Series(np.full(len(x), 42.0))
    with pytest.raises(LeakageError, match="constant"):
        check_leakage(x, y)


def test_leakage_passes_with_non_numeric_columns(clean_frame):
    x, y = clean_frame
    x = x.copy()
    x["category"] = pd.Series(["a", "b", "c"] * (len(x) // 3 + 1))[: len(x)]
    check_leakage(x, y)  # no raise — non-numerics ignored


def test_leakage_error_message_contains_all_three_metrics(clean_frame):
    x, y = clean_frame
    x = x.copy()
    x["obvious_leak"] = y.to_numpy()
    try:
        check_leakage(x, y)
    except LeakageError as exc:
        msg = str(exc)
        assert "pearson=" in msg
        assert "spearman=" in msg
        assert "mi_norm=" in msg
    else:
        pytest.fail("expected LeakageError")


def test_leakage_respects_custom_thresholds(clean_frame):
    x, y = clean_frame
    x = x.copy()
    # Inject a mildly target-correlated column that the default
    # thresholds miss but a lowered threshold catches.
    rng = np.random.default_rng(7)
    x["mild"] = 0.6 * y.to_numpy() + rng.normal(0, y.std(), len(y))
    check_leakage(x, y)  # default thresholds: passes
    with pytest.raises(LeakageError, match="mild"):
        check_leakage(x, y, max_abs_corr=0.3, mi_threshold=0.1)


def test_leakage_empty_numeric_frame_passes():
    x = pd.DataFrame({"cat": ["a", "b", "c"]})
    y = pd.Series([1.0, 2.0, 3.0])
    check_leakage(x, y)  # no raise


def test_leakage_handles_low_cardinality_classification_target():
    """A low-cardinality *numeric* classification target (<= 5 classes) must
    not false-positive on independent features, yet still catch a target copy.

    The MI normalisation now divides by the target's self-information MI(y; y)
    (same estimator as the per-column MI), so a 3-class integer target gives a
    well-defined baseline: independent features land far below threshold, and a
    copy normalises to ~1.0. This pins that behaviour so any future change to
    the MI normalisation that breaks classification targets fails loudly."""
    rng = np.random.default_rng(0)
    n = 200

    # 3-class balanced target.
    y3 = pd.Series(rng.integers(0, 3, n), name="class")

    # Independent features: no real dependency with y3.
    x_independent = pd.DataFrame(
        {
            "a": rng.normal(0, 1, n),
            "b": rng.uniform(0, 100, n),
            "c": rng.integers(0, 1000, n),
        }
    )
    check_leakage(x_independent, y3)  # no raise — was the bug class

    # Target-copy feature: must still be caught even for low-cardinality y.
    x_leaky = x_independent.copy()
    x_leaky["target_copy"] = y3.to_numpy()
    with pytest.raises(LeakageError, match="target_copy"):
        check_leakage(x_leaky, y3)


# ───────────────────────────────────────────────────────────────────
# 2. check_schema
# ───────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "forbidden",
    ["SALE PRICE", "PRICE_PER_SQFT", "target", "TARGET", "log_price"],
)
def test_schema_rejects_forbidden_column(forbidden):
    x = pd.DataFrame({forbidden: [1.0, 2.0], "safe": [3.0, 4.0]})
    contract = SchemaContract(forbidden_columns=frozenset({forbidden}))
    with pytest.raises(SchemaError, match=forbidden):
        check_schema(x, contract)


def test_schema_accepts_clean_frame():
    x = pd.DataFrame({"sqft": [100, 200], "rooms": [2, 3]})
    contract = SchemaContract(
        forbidden_columns=frozenset({"SALE PRICE"}),
        required_columns=frozenset({"sqft"}),
    )
    check_schema(x, contract)  # no raise


def test_schema_rejects_missing_required_column():
    x = pd.DataFrame({"sqft": [100, 200]})
    contract = SchemaContract(required_columns=frozenset({"sqft", "rooms"}))
    with pytest.raises(SchemaError, match="rooms"):
        check_schema(x, contract)


def test_schema_rejects_dtype_mismatch():
    x = pd.DataFrame({"age": ["30", "40"]})  # should be int
    contract = SchemaContract(dtypes={"age": "int64"})
    with pytest.raises(SchemaError, match="age"):
        check_schema(x, contract)


def test_schema_dtype_check_skips_absent_columns():
    x = pd.DataFrame({"sqft": [100, 200]})
    contract = SchemaContract(dtypes={"missing_col": "float64"})
    check_schema(x, contract)  # no raise — missing cols ignored here


# ───────────────────────────────────────────────────────────────────
# 3. check_stateless
# ───────────────────────────────────────────────────────────────────


def _row_wise_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["sqft_doubled"] = out["sqft"] * 2
    return out[["sqft_doubled"]]


def _mean_encoding_leak(df: pd.DataFrame) -> pd.DataFrame:
    """Dataset-wide mean → per-row encoded feature (classic leak)."""
    mean_sqft = df["sqft"].mean()
    out = df.copy()
    out["sqft_relative"] = out["sqft"] - mean_sqft
    return out[["sqft_relative"]]


def _frequency_encoding_leak(df: pd.DataFrame) -> pd.DataFrame:
    """Santander-style frequency feature (state-dependent)."""
    counts = df["rooms"].value_counts()
    out = df.copy()
    out["rooms_freq"] = out["rooms"].map(counts)
    return out[["rooms_freq"]]


def _non_deterministic_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng()  # no seed
    out = df.copy()
    out["noise"] = rng.normal(0, 1, len(out))
    return out[["noise"]]


def test_stateless_accepts_row_wise_pipeline(clean_frame):
    x, _ = clean_frame
    check_stateless(_row_wise_pipeline, x)  # no raise


def test_stateless_catches_mean_encoder(clean_frame):
    x, _ = clean_frame
    with pytest.raises(StatelessnessError, match="state-dependent"):
        check_stateless(_mean_encoding_leak, x)


def test_stateless_catches_frequency_encoder(clean_frame):
    x, _ = clean_frame
    with pytest.raises(StatelessnessError, match="state-dependent"):
        check_stateless(_frequency_encoding_leak, x)


def test_stateless_catches_nondeterministic_pipeline(clean_frame):
    x, _ = clean_frame
    with pytest.raises(StatelessnessError, match="non-deterministic"):
        check_stateless(_non_deterministic_pipeline, x)


def test_stateless_empty_frame_returns_silently():
    empty = pd.DataFrame({"sqft": []})
    check_stateless(lambda df: df, empty)  # no raise


def test_stateless_propagates_pipeline_exceptions(clean_frame):
    x, _ = clean_frame

    def broken(df):
        raise ValueError("pipeline died")

    with pytest.raises(ValueError, match="pipeline died"):
        check_stateless(broken, x)


def test_stateless_raises_on_unknown_sample_index(clean_frame):
    """0.1.2 behavioral change: unknown sample_indices is a caller bug,
    not something to silently skip. Previously the loop did `continue`
    and the check could pass on zero actual spot-checks if every index
    was typo'd."""
    x, _ = clean_frame

    def stateless_pipeline(df):
        out = df.copy()
        out["sqft_doubled"] = out["sqft"] * 2
        return out[["sqft_doubled"]]

    bogus_index = 99999  # not in clean_frame's range (0..199)
    with pytest.raises(ValueError, match="not in raw.index"):
        check_stateless(stateless_pipeline, x, sample_indices=[bogus_index])


# ───────────────────────────────────────────────────────────────────
# 4. Package-level
# ───────────────────────────────────────────────────────────────────


def test_version_is_string():
    assert isinstance(__version__, str)
    assert len(__version__) >= 5  # at least "X.Y.Z"


def test_public_api_shape():
    import schema_firewall as sf

    # exactly three check functions exported
    check_names = {n for n in dir(sf) if n.startswith("check_") and callable(getattr(sf, n))}
    assert check_names == {"check_leakage", "check_schema", "check_stateless"}

    # schema + exceptions importable from top-level
    assert sf.SchemaContract is not None
    assert issubclass(sf.LeakageError, sf.SchemaFirewallError)
    assert issubclass(sf.SchemaError, sf.SchemaFirewallError)
    assert issubclass(sf.StatelessnessError, sf.SchemaFirewallError)


# ───────────────────────────────────────────────────────────────────
# 5. Audit regressions (2026-06)
# ───────────────────────────────────────────────────────────────────


def test_leakage_single_nan_does_not_crash(clean_frame):
    """A single NaN in a numeric feature or in the target used to raise a raw
    sklearn ValueError from mutual_info_regression, escaping the exception
    hierarchy. NaNs are now dropped pairwise per column."""
    x, y = clean_frame
    x = x.copy()
    x.loc[5, "sqft"] = np.nan
    check_leakage(x, y)  # no crash; independent feature -> no leak

    y_nan = y.copy()
    y_nan.iloc[3] = np.nan
    check_leakage(x, y_nan)  # NaN in target also handled


def test_leakage_mi_norm_is_bounded_in_unit_interval(clean_frame):
    """The 'mi_norm in [0, 1]' claim must hold: a perfect copy normalises to
    1.0, not >1 (the old histogram-entropy denominator gave ~1.33)."""
    x, y = clean_frame
    x = x.copy()
    x["copy"] = y.to_numpy()
    try:
        check_leakage(x, y)
        pytest.fail("expected LeakageError on a target copy")
    except LeakageError as exc:
        values = [float(v) for v in re.findall(r"mi_norm=([0-9.]+)", str(exc))]
        assert values, str(exc)
        assert all(0.0 <= v <= 1.0 for v in values), f"mi_norm out of [0,1]: {values}"


def test_leakage_rejects_non_numeric_target(clean_frame):
    """A string classification target raised a raw 'could not convert string to
    float'; it now fails with a clear LeakageError."""
    x, _ = clean_frame
    y_str = pd.Series((["a", "b", "c"] * (len(x) // 3 + 1))[: len(x)])
    with pytest.raises(LeakageError, match="numeric target"):
        check_leakage(x, y_str)


def test_stateless_catches_global_statistic_row_filter(clean_frame):
    """A filter on a FULL-frame statistic (median) is state-dependent: a kept
    row processed alone has the row as its own median and is dropped, producing
    an empty one-row output. The old code skipped that (`continue`) — a false
    negative on exactly the leak class this check targets."""
    x, _ = clean_frame

    def global_median_filter(df):
        out = df.copy()
        return out[out["sqft"] > out["sqft"].median()]

    with pytest.raises(StatelessnessError, match="state-dependent"):
        check_stateless(global_median_filter, x)


def test_stateless_names_index_preservation_precondition(clean_frame):
    """An index-resetting pipeline used to surface as a confusing
    'sample_indices not in raw.index' error the caller never caused. It now
    fails naming the index-preservation precondition."""
    x, _ = clean_frame
    x = x.copy()
    x.index = [f"row_{i}" for i in range(len(x))]

    def index_resetting(df):
        out = df.copy().reset_index(drop=True)  # relabels to 0..n-1
        out["sqft_doubled"] = out["sqft"] * 2
        return out[["sqft_doubled"]]

    with pytest.raises(StatelessnessError, match="preserve the input index"):
        check_stateless(index_resetting, x)


def test_stateless_clear_error_when_sample_index_dropped_from_output(clean_frame):
    """Spot-checking a row the pipeline drops from its full output used to raise
    a raw pandas KeyError; it now gives a clear message."""
    x, _ = clean_frame

    def drop_label_zero(df):
        out = df[df.index != 0].copy()  # row-wise drop of label 0
        out["sqft_doubled"] = out["sqft"] * 2
        return out[["sqft_doubled"]]

    with pytest.raises(ValueError, match="dropped by pipeline_fn"):
        check_stateless(drop_label_zero, x, sample_indices=[0])


@pytest.mark.parametrize("spec", ["int64", "i8", "<i8"])
def test_schema_dtype_accepts_equivalent_int_spellings(spec):
    """'int64', 'i8', '<i8' all resolve to the same dtype; a raw string compare
    rejected them against the actual 'int64'."""
    x = pd.DataFrame({"age": pd.Series([30, 40], dtype="int64")})
    check_schema(x, SchemaContract(dtypes={"age": spec}))  # no raise
