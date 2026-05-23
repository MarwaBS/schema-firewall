"""
Target-encoding leak demo — California housing
================================================

A short, self-contained script that reproduces one of the most common
ML production bugs: computing a group-mean target encoding on the FULL
dataset (train + test) before splitting. The model's reported R²
looks solid. One call to ``check_stateless`` catches the leak.

Run directly:

    python examples/leakage_demo.py

Or open the matching notebook:

    examples/leakage_demo.ipynb
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from schema_firewall import (
    LeakageError,
    StatelessnessError,
    check_leakage,
    check_stateless,
)


def _load() -> tuple[pd.DataFrame, pd.Series]:
    """California housing as a single raw frame + target series."""
    raw = fetch_california_housing(as_frame=True)
    df = raw.data.copy()
    df["lat_bin"] = df["Latitude"].round(2)
    df["lon_bin"] = df["Longitude"].round(2)
    df["region"] = df["lat_bin"].astype(str) + "_" + df["lon_bin"].astype(str)
    return df, raw.target.rename("price")


# ─────────────────────────────────────────────────────────────────
# Step 1 — the leaky pipeline.
#
# Compute "region mean price" using the FULL dataset (including rows
# that will later be the test split). Attach it as a feature. This
# pattern appears all over production pipelines under names like
# "neighborhood affluence index" or "area target encoding".
# ─────────────────────────────────────────────────────────────────


def leaky_feature_engineering(df: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
    out = df.copy()
    region_means = target.groupby(df["region"]).mean()
    out["region_mean_price"] = out["region"].map(region_means)
    return out.drop(columns=["region", "lat_bin", "lon_bin"])


def _train_and_score(x_tr, x_te, y_tr, y_te, label: str) -> float:
    model = Ridge(alpha=1.0).fit(x_tr, y_tr)
    score = r2_score(y_te, model.predict(x_te))
    print(f"  {label:<20s} R² = {score:.4f}")
    return score


def run_leaky() -> float:
    print("── leaky pipeline (region mean computed on full dataset) ──")
    df, y = _load()
    x_leaky = leaky_feature_engineering(df, y)
    x_tr, x_te, y_tr, y_te = train_test_split(
        x_leaky, y, test_size=0.25, random_state=0
    )
    return _train_and_score(x_tr, x_te, y_tr, y_te, "leaky")


# ─────────────────────────────────────────────────────────────────
# Step 2 — the schema-firewall catches it.
#
# check_stateless asserts that applying the feature pipeline to a
# single-row subset produces the same output for that row as applying
# it to the full frame. A target-mean-encoded feature fails this
# invariant because the one-row mean is that row's own target.
# ─────────────────────────────────────────────────────────────────


def catch_leak() -> None:
    print("── schema-firewall running check_stateless ──")
    df, y = _load()

    def pipeline_fn(frame: pd.DataFrame) -> pd.DataFrame:
        # closure captures y-on-full-frame; that's the bug.
        return leaky_feature_engineering(frame, y.loc[frame.index])

    try:
        check_stateless(pipeline_fn, df)
    except StatelessnessError as exc:
        print("  CAUGHT: check_stateless raised StatelessnessError")
        print(f"  detail: {str(exc).splitlines()[0][:110]}")
        return
    print("  FAIL: check_stateless did not raise — firewall missed the leak")


def catch_leak_via_leakage_check() -> None:
    print("── schema-firewall running check_leakage on the leaky X ──")
    df, y = _load()
    x_leaky = leaky_feature_engineering(df, y)
    x_leaky_numeric = x_leaky.select_dtypes(include=[np.number])
    try:
        check_leakage(x_leaky_numeric, y)
    except LeakageError as exc:
        print("  CAUGHT: check_leakage raised LeakageError")
        print(f"  detail: {str(exc).splitlines()[1][:110]}")
        return
    raise AssertionError(
        "README documents that check_leakage raises LeakageError on the "
        "leaky region-mean feature, but it did not. Check sklearn version "
        "drift or default threshold values in _checks.py."
    )


# ─────────────────────────────────────────────────────────────────
# Step 3 — the correct pipeline.
#
# Compute the region mean from train ONLY, then map onto both splits.
# Unseen regions in the test split become NaN and must be handled
# explicitly; the model's score is honest rather than inflated.
# ─────────────────────────────────────────────────────────────────


def run_correct() -> float:
    print("── correct pipeline (region mean computed on train only) ──")
    df, y = _load()
    tr_idx, te_idx = train_test_split(df.index, test_size=0.25, random_state=0)

    region_means = y.loc[tr_idx].groupby(df.loc[tr_idx, "region"]).mean()
    df["region_mean_price"] = df["region"].map(region_means)
    df["region_mean_price"] = df["region_mean_price"].fillna(
        region_means.mean()
    )

    features = df.drop(columns=["region", "lat_bin", "lon_bin"])
    x_tr, x_te = features.loc[tr_idx], features.loc[te_idx]
    y_tr, y_te = y.loc[tr_idx], y.loc[te_idx]
    return _train_and_score(x_tr, x_te, y_tr, y_te, "honest")


# ─────────────────────────────────────────────────────────────────

def main() -> None:
    leaky_r2 = run_leaky()
    print()
    catch_leak()
    print()
    catch_leak_via_leakage_check()
    print()
    honest_r2 = run_correct()
    print()
    print("── summary ──")
    print(f"  leaky R²  = {leaky_r2:.4f}  (inflated by train+test target mean leak)")
    print(f"  honest R² = {honest_r2:.4f}  (train-only region mean, unseen regions imputed)")
    print(f"  gap       = {leaky_r2 - honest_r2:+.4f}")
    print()
    print("  If you have applied .mean() / .groupby().transform('mean') /")
    print("  TargetEncoder / fit_transform on the full dataset before")
    print("  cross-validation — you have probably shipped this bug.")


if __name__ == "__main__":
    main()
