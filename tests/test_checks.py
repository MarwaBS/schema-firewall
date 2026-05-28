"""Adversarial test suite for schema-firewall.

Each test either injects a known failure and asserts the firewall
raises, or exercises a legitimate path and asserts the firewall
does not raise. Parameterised where failure has multiple shapes.
"""
from __future__ import annotations

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
    check_names = {
        n for n in dir(sf)
        if n.startswith("check_") and callable(getattr(sf, n))
    }
    assert check_names == {"check_leakage", "check_schema", "check_stateless"}

    # schema + exceptions importable from top-level
    assert sf.SchemaContract is not None
    assert issubclass(sf.LeakageError, sf.SchemaFirewallError)
    assert issubclass(sf.SchemaError, sf.SchemaFirewallError)
    assert issubclass(sf.StatelessnessError, sf.SchemaFirewallError)
