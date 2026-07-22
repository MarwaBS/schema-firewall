"""Adversarial test suite for schema-firewall.

Each test either injects a known failure and asserts the firewall
raises, or exercises a legitimate path and asserts the firewall
does not raise. Parameterised where failure has multiple shapes.
"""

from __future__ import annotations

import re
import warnings

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

# -------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------


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


# -------------------------------------------------------------------
# 1. check_leakage
# -------------------------------------------------------------------


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


def test_leakage_warns_on_skipped_non_numeric_columns(clean_frame):
    """Non-numeric columns are not inspected -- but that must be surfaced, not
    silent. A caller with an object column gets a warning naming it, so a
    stringified target copy hiding there can't pass the firewall unnoticed."""
    x, y = clean_frame
    x = x.copy()
    x["category"] = pd.Series(["a", "b", "c"] * (len(x) // 3 + 1))[: len(x)]
    with pytest.warns(UserWarning, match=r"NOT checked for leakage.*category"):
        check_leakage(x, y)  # no raise, but warns that 'category' was skipped


def test_leakage_warns_when_stringified_target_copy_is_unchecked(clean_frame):
    """The dangerous case the warning exists for: a string copy of the target in
    an object column. The detector can't inspect it (non-numeric), so it can't
    catch the leak -- but it MUST warn that this column went unchecked."""
    x, y = clean_frame
    x = x.copy()
    x["y_str"] = pd.Series(y).astype(str).to_numpy()  # object-dtype target copy
    with pytest.warns(UserWarning, match=r"NOT checked for leakage.*y_str"):
        check_leakage(x, y)


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
    # No numeric columns to inspect -> no raise, but the object column is warned
    # about (it was not checked), consistent with the skipped-column contract.
    with pytest.warns(UserWarning, match=r"NOT checked for leakage.*cat"):
        check_leakage(x, y)


def test_leakage_handles_low_cardinality_classification_target():
    """A low-cardinality *numeric* classification target (<= 5 classes) must
    not false-positive on independent features, yet still catch a target copy.

    The MI detector uses adjusted (chance-corrected) mutual information, so a
    3-class integer target is handled cleanly: independent features land near 0
    (AMI corrects for chance regardless of cardinality) and a copy lands at ~1.0."""
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
    check_leakage(x_independent, y3)  # independent features: no raise

    # Target-copy feature: must still be caught even for low-cardinality y.
    x_leaky = x_independent.copy()
    x_leaky["target_copy"] = y3.to_numpy()
    with pytest.raises(LeakageError, match="target_copy"):
        check_leakage(x_leaky, y3)


# -------------------------------------------------------------------
# 2. check_schema
# -------------------------------------------------------------------


@pytest.mark.parametrize(
    "forbidden",
    # The first five are the exact forbidden set the README documents for the
    # flagship integration; the README points here as the verifiable proof.
    ["SALE PRICE", "SALE DATE", "PRICE_PER_SQFT", "TARGET", "log_price", "target"],
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


def test_schema_rejects_duplicate_columns():
    """Duplicate column names in X fail a clear precondition."""
    x = pd.DataFrame([[1, 2, 3]], columns=["a", "a", "b"])
    with pytest.raises(ValueError, match="duplicate"):
        check_schema(x, SchemaContract(dtypes={"a": "int64"}))


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
    check_schema(x, contract)  # no raise -- missing cols ignored here


# -------------------------------------------------------------------
# 3. check_stateless
# -------------------------------------------------------------------


def _row_wise_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["sqft_doubled"] = out["sqft"] * 2
    return out[["sqft_doubled"]]


def _mean_encoding_leak(df: pd.DataFrame) -> pd.DataFrame:
    """Dataset-wide mean -> per-row encoded feature (classic leak)."""
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


def test_stateless_refuses_empty_frame():
    """A 0-row frame has nothing to spot-check, so it is refused like a 1-row
    frame -- not passed vacuously."""
    empty = pd.DataFrame({"sqft": []})
    with pytest.raises(StatelessnessError, match="at least 2 rows"):
        check_stateless(lambda df: df, empty)


def test_stateless_propagates_pipeline_exceptions(clean_frame):
    x, _ = clean_frame

    def broken(df):
        raise ValueError("pipeline died")

    with pytest.raises(ValueError, match="pipeline died"):
        check_stateless(broken, x)


def test_stateless_catches_tail_edit_on_low_variance_wide_frame():
    """A cross-row edit on a low-variance column must be caught even on a wide
    frame: variance ranking is scale-dependent, so a standardised column
    (variance ~1) must not be skipped in favour of higher-variance ones."""
    for seed in range(20):
        rng = np.random.default_rng(seed)
        n = 400
        data = {f"c{i}": rng.normal(0, 10, n) for i in range(25)}
        data["victim"] = rng.normal(0, 1, n)  # lowest variance of the 26 columns
        raw = pd.DataFrame(data)

        def clip_victim(df):
            out = df.copy()
            hi = out["victim"].quantile(0.995)  # cross-row statistic -> stateful
            out["victim"] = out["victim"].clip(upper=hi)
            return out

        with pytest.raises(StatelessnessError):
            check_stateless(clip_victim, raw)


def test_stateless_refuses_empty_output_from_nonempty_input(clean_frame):
    """A pipeline that drops every row leaves no output rows to spot-check, so it
    is refused rather than passed vacuously."""
    x, _ = clean_frame

    def drop_everything(df):
        out = df.copy()
        return out[out["sqft"] > out["sqft"].max() + 1]  # global stat -> empty

    with pytest.raises(StatelessnessError, match="0 rows"):
        check_stateless(drop_everything, x)


def test_stateless_refuses_single_row_input():
    """With a single-row input the only spot-check subset equals the full frame,
    so even a mean-encoder passes vacuously. Refuse frames too small to check."""
    df = pd.DataFrame({"x": [5.0]})

    def mean_encode(frame):
        out = frame.copy()
        out["xr"] = out["x"] - out["x"].mean()
        return out[["xr"]]

    with pytest.raises(StatelessnessError, match="at least 2 rows"):
        check_stateless(mean_encode, df)


def test_stateless_clear_error_on_non_dataframe_return(clean_frame):
    """A pipeline returning a non-DataFrame fails naming the return type, not
    'non-deterministic'."""
    x, _ = clean_frame
    with pytest.raises(TypeError, match="DataFrame"):
        check_stateless(lambda df: df["sqft"].to_numpy(), x)


def test_stateless_raises_on_empty_sample_indices(clean_frame):
    """An empty sample_indices list skips the default row selection and would run
    zero spot-checks -- a vacuous pass -- so it is a caller error."""
    x, _ = clean_frame
    with pytest.raises(ValueError, match="empty"):
        check_stateless(_row_wise_pipeline, x, sample_indices=[])


def test_stateless_raises_on_unknown_sample_index(clean_frame):
    """Unknown sample_indices is a caller error, not silently skipped."""
    x, _ = clean_frame

    def stateless_pipeline(df):
        out = df.copy()
        out["sqft_doubled"] = out["sqft"] * 2
        return out[["sqft_doubled"]]

    bogus_index = 99999  # not in clean_frame's range (0..199)
    with pytest.raises(ValueError, match="not in raw.index"):
        check_stateless(stateless_pipeline, x, sample_indices=[bogus_index])


# -------------------------------------------------------------------
# 4. Package-level
# -------------------------------------------------------------------


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


# -------------------------------------------------------------------
# 5. Regression tests
# -------------------------------------------------------------------


def test_leakage_single_nan_does_not_crash(clean_frame):
    """A single NaN in a feature or the target is dropped pairwise, not crashed on."""
    x, y = clean_frame
    x = x.copy()
    x.loc[5, "sqft"] = np.nan
    check_leakage(x, y)  # no crash; independent feature -> no leak

    y_nan = y.copy()
    y_nan.iloc[3] = np.nan
    check_leakage(x, y_nan)  # NaN in target also handled


def test_leakage_mi_norm_is_bounded_in_unit_interval(clean_frame):
    """The 'mi_norm in [0, 1]' claim must hold: a perfect copy normalises to
    1.0, not >1 (a histogram-entropy denominator puts it at ~1.33)."""
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
    """A non-numeric (string) target fails with a clear LeakageError."""
    x, _ = clean_frame
    y_str = pd.Series((["a", "b", "c"] * (len(x) // 3 + 1))[: len(x)])
    with pytest.raises(LeakageError, match="numeric target"):
        check_leakage(x, y_str)


@pytest.mark.parametrize("seed", [0, 1, 2, 7])
@pytest.mark.parametrize(
    "transform",
    [
        pytest.param(lambda v: v**2, id="square"),
        pytest.param(np.abs, id="abs"),
        pytest.param(lambda v: np.cos(3 * v), id="cos"),
        pytest.param(lambda v: (v > 0).astype(float), id="binary"),
        pytest.param(lambda v: np.digitize(v, [-0.5, 0.5]).astype(float), id="three-class"),
    ],
)
def test_leakage_catches_non_monotone_dependence(transform, seed):
    """The MI detector catches non-monotone/discrete deterministic leakage that
    Pearson and Spearman miss -- y = x**2, |x|, cos(3x), and binary/k-class target
    encodings. Parametrised over seeds because per-seed variance can mask it."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=300)
    leak = transform(x)
    # The point: ordinary correlation stays below the 0.95 default for all of
    # these, so detection rests on the MI pillar.
    assert abs(np.corrcoef(x, leak)[0, 1]) < 0.95
    with pytest.raises(LeakageError, match="leak"):
        check_leakage(pd.DataFrame({"safe": rng.normal(size=300), "leak": x}), pd.Series(leak))


def test_leakage_passes_legitimate_noisy_predictor():
    """A genuinely predictive but noisy feature (real signal + real noise, not a
    deterministic encoding of the target) must NOT be flagged -- adjusted MI
    measures shared information, so noise keeps an honest predictor well below
    threshold. Without it, the firewall would flag honest, noisy predictors
    (r~0.45 here) and decay from a leak detector into a feature selector."""
    rng = np.random.default_rng(1)
    n = 300
    y = rng.lognormal(13, 0.5, n)
    predictor = 0.5 * (y - y.mean()) / y.std() + rng.normal(0, 1, n)  # corr ~0.45, noisy
    check_leakage(pd.DataFrame({"predictor": predictor}), pd.Series(y))  # no raise


def test_leakage_small_sample_raises_clear_precondition():
    """Below the minimum sample size the binned MI is noise-dominated (and a
    2-point correlation is always |r|=1), so there is no honest threshold. The
    check raises a clear ValueError precondition instead of a raw sklearn crash
    or a false positive/negative."""
    rng = np.random.default_rng(0)
    for n in (2, 3, 4, 25, 50, 99):
        with pytest.raises(ValueError, match="at least 100 finite samples"):
            check_leakage(pd.DataFrame({"f": rng.normal(size=n)}), pd.Series(rng.normal(size=n)))


def test_stateless_rejects_duplicate_index():
    """A non-unique index makes the per-row spot-check vacuous (raw.loc[[label]]
    pulls every row sharing the label), so a global transform would pass. The
    check refuses a duplicate index instead of giving a false pass."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"sqft": rng.uniform(500, 3000, 200)}, index=[7] * 200)

    def global_zscore(frame):
        out = frame.copy()
        out["z"] = (out["sqft"] - out["sqft"].mean()) / out["sqft"].std()
        return out[["z"]]

    with pytest.raises(StatelessnessError, match="duplicate"):
        check_stateless(global_zscore, df)


def test_stateless_catches_global_statistic_row_filter(clean_frame):
    """A filter on a full-frame statistic (median) is state-dependent: a kept row
    processed alone is its own median and gets dropped, so its one-row output is
    empty."""
    x, _ = clean_frame

    def global_median_filter(df):
        out = df.copy()
        return out[out["sqft"] > out["sqft"].median()]

    with pytest.raises(StatelessnessError, match="state-dependent"):
        check_stateless(global_median_filter, x)


def test_stateless_catches_global_winsorizer(clean_frame):
    """A global-quantile clip (winsorise) only edits TAIL rows, which
    fixed-stride spot-checks miss ~79% of the time. Spot-checking each
    numeric column's extreme rows catches it deterministically (the modified
    row, processed alone, has itself as its own quantile and is left unclipped,
    diverging from the full-frame output)."""
    x, _ = clean_frame

    def winsorize(df):
        out = df.copy()
        out["sqft_clip"] = out["sqft"].clip(upper=out["sqft"].quantile(0.95))
        return out[["sqft_clip"]]

    with pytest.raises(StatelessnessError, match="state-dependent"):
        check_stateless(winsorize, x)


def test_stateless_catches_global_mean_imputation():
    """Global-mean imputation `df.fillna(df.mean())` is the canonical leakage
    bug (fit on full data). The NaN row processed alone can't reconstruct the
    global mean, so it diverges from the full-frame output -- caught because the
    default spot-check now includes NaN-bearing rows, even when the NaN sits off
    the min/max/stride sample."""
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 100, 200)
    x[1] = np.nan  # a NaN at a non-extreme, off-stride row
    df = pd.DataFrame({"x": x})

    def mean_impute(frame):
        out = frame.copy()
        out["x"] = out["x"].fillna(out["x"].mean())
        return out[["x"]]

    with pytest.raises(StatelessnessError, match="state-dependent"):
        check_stateless(mean_impute, df)


def test_stateless_names_index_preservation_precondition(clean_frame):
    """An index-resetting pipeline fails naming the index-preservation precondition."""
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
    """Spot-checking a row the pipeline drops from its output gives a clear error."""
    x, _ = clean_frame

    def drop_label_zero(df):
        out = df[df.index != 0].copy()  # row-wise drop of label 0
        out["sqft_doubled"] = out["sqft"] * 2
        return out[["sqft_doubled"]]

    with pytest.raises(ValueError, match="dropped by pipeline_fn"):
        check_stateless(drop_label_zero, x, sample_indices=[0])


def test_schema_contract_dtypes_is_immutable():
    """The dtypes mapping is read-only: item assignment on a 'frozen' contract
    raises."""
    c = SchemaContract(dtypes={"age": "int64"})
    with pytest.raises(TypeError):
        c.dtypes["age"] = "object"


def test_schema_contract_does_not_alias_caller_dtypes():
    """The contract holds a copy of the input dict, so mutating the caller's dict
    does not change it."""
    d = {"age": "int64"}
    c = SchemaContract(dtypes=d)
    d["age"] = "object"
    assert c.dtypes is not None and c.dtypes["age"] == "int64"


def test_schema_contract_pickle_round_trip():
    """A contract survives pickling (joblib/multiprocessing) and comes back both
    equal and still immutable."""
    import pickle

    c = SchemaContract(
        forbidden_columns=frozenset({"TARGET"}),
        dtypes={"age": "int64"},
    )
    restored = pickle.loads(pickle.dumps(c))
    assert restored == c
    with pytest.raises(TypeError):
        restored.dtypes["age"] = "object"
    assert pickle.loads(pickle.dumps(SchemaContract())) == SchemaContract()


def test_schema_contract_attributes_are_frozen():
    """The documented 'frozen' promise: attribute rebinding raises."""
    from dataclasses import FrozenInstanceError

    c = SchemaContract(required_columns=frozenset({"a"}))
    with pytest.raises(FrozenInstanceError):
        c.required_columns = frozenset({"b"})


def test_schema_category_dtype_accepts_any_categorical():
    """An unparameterised `{"c": "category"}` contract accepts any CategoricalDtype
    (exact-categories equality would always be False)."""
    x = pd.DataFrame({"c": pd.Series(["a", "b", "a"], dtype="category")})
    check_schema(x, SchemaContract(dtypes={"c": "category"}))  # no raise


def test_schema_category_dtype_still_rejects_non_categorical():
    """The relaxed category match must not over-accept: an object column is not
    categorical and must still fail a `{"c": "category"}` contract."""
    x = pd.DataFrame({"c": ["a", "b", "a"]})  # object dtype, not categorical
    with pytest.raises(SchemaError, match="c"):
        check_schema(x, SchemaContract(dtypes={"c": "category"}))


@pytest.mark.parametrize("spec", ["int64", "i8", "<i8"])
def test_schema_dtype_accepts_equivalent_int_spellings(spec):
    """'int64', 'i8', '<i8' all resolve to the same dtype; a raw string compare
    rejected them against the actual 'int64'."""
    x = pd.DataFrame({"age": pd.Series([30, 40], dtype="int64")})
    check_schema(x, SchemaContract(dtypes={"age": spec}))  # no raise


def test_leakage_false_positive_operating_point_is_pinned():
    """Characterise the MI detector's false-positive boundary so the
    docstring/CHANGELOG claims stay honest. The detector scores raw target
    dependence, not its source, so an HONEST strong linear predictor trips it
    too. Pin the measured band (seeded, deterministic): independent and weak
    columns never flag; a near-deterministic predictor always does; the
    documented transition is majority-flagged by |r|=0.85. Retuning
    mi_threshold shifts these rates and fails this test, forcing the docs to be
    re-measured rather than silently drifting.
    """

    def flag_rate(r: float, seeds: int = 20, n: int = 400) -> float:
        hits = 0
        for seed in range(seeds):
            rng = np.random.default_rng(seed)
            y = rng.standard_normal(n)
            x = r * y + np.sqrt(max(0.0, 1.0 - r * r)) * rng.standard_normal(n)
            try:
                check_leakage(pd.DataFrame({"feat": x}), pd.Series(y))
            except LeakageError:
                hits += 1
        return hits / seeds

    # 0-FP floor: independent + weak honest predictors are never flagged.
    assert flag_rate(0.0) == 0.0
    assert flag_rate(0.75) == 0.0
    # Docstring's edge claim "at most 5% at |r| = 0.80" (measured 0.05).
    assert flag_rate(0.80) <= 0.05
    # A near-deterministic predictor is always flagged (the detector's job).
    assert flag_rate(0.95) == 1.0
    # Documented transition: a nonzero minority by |r|=0.83 (measured ~0.15),
    # the majority by |r|=0.85 (measured ~0.75). Pins both operating points.
    assert 0.0 < flag_rate(0.83) < 0.5
    assert flag_rate(0.85) >= 0.5


def test_leakage_passes_nullable_integer_with_na():
    """A pandas nullable-integer (Int64) feature/target carrying pd.NA is inspected
    as numeric, not skipped with a warning, and its NAs are dropped pairwise. An
    independent pair passes cleanly across the pandas>=2.0,<3.0 range."""
    rng = np.random.default_rng(0)
    n = 300
    feat = pd.array(rng.integers(0, 100, n), dtype="Int64")
    feat[0] = pd.NA
    tgt = pd.array(rng.integers(0, 100, n), dtype="Int64")
    tgt[1] = pd.NA
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check_leakage(pd.DataFrame({"feat": feat}), pd.Series(tgt))


def test_leakage_catches_nullable_boolean_target_copy():
    """A nullable-boolean (BooleanDtype) target copied into a feature is caught:
    the extension dtype is inspected like a plain bool, so the copy trips the
    correlation cap rather than slipping through as a non-numeric column."""
    rng = np.random.default_rng(0)
    b = pd.array(rng.integers(0, 2, 300).astype(bool), dtype="boolean")
    with pytest.raises(LeakageError, match="target_copy"):
        check_leakage(pd.DataFrame({"target_copy": b}), pd.Series(b))


def test_leakage_warns_on_columns_skipped_for_high_missingness(clean_frame):
    """A numeric column with too few finite paired rows to assess is surfaced as a
    warning naming it, not silently skipped."""
    x, y = clean_frame
    x = x.copy()
    leak = y.to_numpy().astype(float).copy()
    leak[:150] = np.nan  # 50 finite paired rows < the 100-sample floor
    x["sparse_leak"] = leak
    with pytest.warns(UserWarning, match=r"finite paired.*sparse_leak"):
        check_leakage(x, y)


def test_leakage_rejects_2d_target(clean_frame):
    """A 2-D target (one-column DataFrame) fails an up-front 1-D precondition."""
    x, y = clean_frame
    with pytest.raises(ValueError, match="1-dimensional"):
        check_leakage(x, y.to_frame())


def test_leakage_rejects_length_mismatch(clean_frame):
    """X and y of different lengths fail a clear length precondition."""
    x, y = clean_frame
    with pytest.raises(ValueError, match="length"):
        check_leakage(x, y.iloc[:-5])


def test_leakage_rejects_duplicate_columns(clean_frame):
    """Duplicate column names in X fail a clear precondition."""
    x, y = clean_frame
    x = x.copy()
    x.columns = ["sqft", "sqft", "rooms"]  # duplicate label
    with pytest.raises(ValueError, match="duplicate"):
        check_leakage(x, y)


def test_leakage_catches_bool_target_copy():
    """A bool column is numeric (True/False == 1/0), so an exact bool copy of a
    binary target is caught like any numeric copy."""
    rng = np.random.default_rng(0)
    n = 200
    y = pd.Series(rng.integers(0, 2, n))
    x = pd.DataFrame({"safe": rng.normal(size=n), "leaked": y.astype(bool).to_numpy()})
    with pytest.raises(LeakageError, match="leaked"):
        check_leakage(x, y)


def test_leakage_zero_row_frame_raises_min_samples(clean_frame):
    """A 0-row frame with numeric columns hits the min-samples precondition, not a
    silent pass."""
    x, y = clean_frame
    with pytest.raises(ValueError, match="at least 100 finite samples"):
        check_leakage(x.iloc[:0], y.iloc[:0])


def test_leakage_realigns_permuted_target_index(clean_frame):
    """A target with a permuted (same-labels) index is realigned to X before
    comparison, so a shuffled y still catches a verbatim copy."""
    x, y = clean_frame
    x = x.copy()
    x["leaked"] = y.to_numpy()
    y_shuffled = y.sample(frac=1, random_state=1)  # same labels, different order
    with pytest.raises(LeakageError, match="leaked"):
        check_leakage(x, y_shuffled)


def test_leakage_raises_on_unalignable_target_index(clean_frame):
    """A target whose index is a DIFFERENT label set than X cannot be aligned
    row-for-row; positional comparison would be a silent guess. Refuse instead."""
    x, y = clean_frame
    y_disjoint = y.copy()
    y_disjoint.index = y_disjoint.index + 10_000
    with pytest.raises(ValueError, match="align"):
        check_leakage(x, y_disjoint)


def _zero_inflated_square(sparsity: float, negfrac: float, n: int = 1000):
    """Zero-inflated feature: (1-sparsity) nonzero rows, negfrac of them negative,
    with a deterministic y = x**2 leak. `rng.random` is always consumed so the
    draw order is stable regardless of negfrac."""
    rng = np.random.default_rng(0)
    k = int(round((1.0 - sparsity) * n))
    idx = rng.choice(n, size=k, replace=False)
    mags = rng.uniform(1.0, 10.0, k)
    signs = np.where(rng.random(k) < negfrac, -1.0, 1.0)
    x = np.zeros(n)
    x[idx] = signs * mags
    return x, x**2


@pytest.mark.parametrize("sparsity", [0.95, 0.96, 0.97])
def test_leakage_catches_zero_inflated_square_collapse_region(sparsity):
    """At 95-97% zeros with a mixed-sign tail, y = x**2 is non-monotone (Pearson
    and Spearman ~0) so detection rests on the MI pillar; quantile binning scored
    this deterministic leak at adjusted MI 0.0, dense-rank binning scores ~0.9+."""
    x, y = _zero_inflated_square(sparsity, negfrac=0.5)
    with pytest.raises(LeakageError, match="leak"):
        check_leakage(pd.DataFrame({"leak": x}), pd.Series(y))


@pytest.mark.parametrize("sparsity", [0.5, 0.8, 0.9, 0.93, 0.99])
@pytest.mark.parametrize("negfrac", [0.0, 0.5])
def test_leakage_catches_zero_inflated_square_across_sparsity(sparsity, negfrac):
    """The zero-inflated non-monotone leak must be caught across the sparsity
    range, signed and unsigned -- a guard around the collapse band above."""
    x, y = _zero_inflated_square(sparsity, negfrac)
    with pytest.raises(LeakageError, match="leak"):
        check_leakage(pd.DataFrame({"leak": x}), pd.Series(y))


def test_stateless_catches_inplace_global_winsorizer(clean_frame):
    """An in-place winsoriser that returns its input aliases first/second/raw, so
    the determinism check would compare a frame to itself. The tail-clip leak is
    caught via the spot-check."""
    x, _ = clean_frame

    def winsorize_inplace(df):
        df["sqft"] = df["sqft"].clip(upper=df["sqft"].quantile(0.95))
        return df

    with pytest.raises(StatelessnessError, match="state-dependent"):
        check_stateless(winsorize_inplace, x)


def test_stateless_does_not_mutate_caller_frame(clean_frame):
    """A verification tool must not modify what it verifies: the caller's frame is
    unchanged after an in-place pipeline."""
    x, _ = clean_frame
    snapshot = x.copy()

    def add_column_inplace(df):
        df["sqft_doubled"] = df["sqft"] * 2
        return df

    check_stateless(add_column_inplace, x)  # stateless + deterministic: no raise
    pd.testing.assert_frame_equal(x, snapshot)


def test_checks_return_none_on_pass(clean_frame):
    """The documented contract: each check returns None (not a truthy value) on a
    clean input."""
    x, y = clean_frame
    assert check_leakage(x, y) is None
    assert check_schema(x, SchemaContract(required_columns=frozenset({"sqft"}))) is None
    assert check_stateless(_row_wise_pipeline, x) is None


def test_schema_dtype_unrecognised_string_falls_back_to_exact_match():
    """A dtype string pandas can't resolve falls back to an exact string compare
    against the actual dtype, so a typo'd contract still fails loudly."""
    x = pd.DataFrame({"a": [1, 2]})
    with pytest.raises(SchemaError, match="a"):
        check_schema(x, SchemaContract(dtypes={"a": "not_a_real_dtype"}))


def test_safe_corr_rejects_unknown_method():
    """_safe_corr guards against a bogus method name reaching it from an untyped
    caller."""
    from schema_firewall._checks import _safe_corr

    with pytest.raises(ValueError, match="unknown correlation method"):
        _safe_corr(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]), method="kendall")


def test_leakage_spearman_exceeds_pearson_on_monotone_nonlinear(clean_frame):
    """The Spearman pillar adds monotonic sensitivity beyond Pearson: a strictly
    monotone but nonlinear function of the target scores Spearman > Pearson."""
    x, y = clean_frame
    x = x.copy()
    yv = y.to_numpy()
    x["monotone"] = np.exp((yv - yv.mean()) / yv.std())
    try:
        check_leakage(x, y)
        pytest.fail("expected LeakageError")
    except LeakageError as exc:
        m = re.search(r"monotone: pearson=([0-9.]+) spearman=([0-9.]+)", str(exc))
        assert m, str(exc)
        assert float(m.group(2)) > float(m.group(1))


def test_exactly_three_runtime_dependencies():
    """The locked 'three dependencies, nothing else' promise."""
    from importlib import metadata

    runtime = {
        r.split(";")[0].split(">")[0].split("<")[0].split("=")[0].strip()
        for r in metadata.requires("schema-firewall") or []
        if "; extra" not in r
    }
    assert runtime == {"numpy", "pandas", "scikit-learn"}, runtime


def test_exactly_four_exceptions():
    """The locked 'four exceptions' promise: the base and exactly three subclasses."""
    from schema_firewall import LeakageError, SchemaError, SchemaFirewallError, StatelessnessError

    assert set(SchemaFirewallError.__subclasses__()) == {
        LeakageError,
        SchemaError,
        StatelessnessError,
    }


def test_core_loc_within_budget():
    """The locked <= 500 LoC design budget, enforced so it can't silently rot.

    Counts code lines the way the README documents (non-blank, non-comment)."""
    import pathlib

    src = pathlib.Path(__file__).resolve().parents[1] / "src" / "schema_firewall"
    code_lines = 0
    for p in sorted(src.rglob("*.py")):
        for line in p.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                code_lines += 1
    assert code_lines <= 500, f"core is {code_lines} LoC, over the 500 budget"


def test_source_is_ascii_only():
    """CONTRIBUTING promises ASCII-only source: a non-ASCII char in a docstring
    or comment crashes import on a cp1252 (Windows) console, and this repo hit
    exactly that. Enforce the promise mechanically -- every .py under src/ and
    tests/ must be pure ASCII -- so it can never silently drift back."""
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1]
    offenders = []
    for p in sorted((root / "src").rglob("*.py")) + sorted((root / "tests").rglob("*.py")):
        for lineno, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            bad = next((c for c in line if ord(c) > 127), None)
            if bad is not None:
                offenders.append(f"{p.relative_to(root)}:{lineno} U+{ord(bad):04X}")
                break
    assert not offenders, "non-ASCII source (breaks cp1252 import): " + "; ".join(offenders)
