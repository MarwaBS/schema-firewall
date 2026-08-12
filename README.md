# schema-firewall

**Three checks that catch the leakage and schema bugs that slip past peer review.**

```bash
pip install schema-firewall
```

[![CI](https://github.com/MarwaBS/schema-firewall/actions/workflows/python-package.yml/badge.svg)](https://github.com/MarwaBS/schema-firewall/actions/workflows/python-package.yml)
[![PyPI](https://img.shields.io/pypi/v/schema-firewall.svg)](https://pypi.org/project/schema-firewall/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

> **Downstream usage.** Extracted from the firewall layer of [`nyc-real-estate-predictor`](https://github.com/MarwaBS/nyc-real-estate-predictor) — the flagship pins `schema-firewall==0.1.3` in `requirements.txt` and re-validates the integration in its `External Benchmark` CI job, which runs weekly and on pushes/PRs touching the benchmark's paths (path-filtered, not every push). It shows the library is used and CI-exercised downstream — not that every contract is enforced there. The pin is deliberate, not deferred. `0.1.3` predates the `0.2.0` variance-cap tail-sampling regression entirely — it samples every numeric column exhaustively — so the flagship was never exposed to that fail-open; and the flagship calls only `check_leakage` and `check_schema`, never `check_stateless`, so the affected surface is unreachable downstream regardless. `0.2.x` also switched the leakage-MI binning from quantile to dense-rank, which shifts the MI scale, so moving the pin would require re-measuring the flagship's `mi_threshold` (calibrated against `0.1.3`'s MI) — a measured change, not a version bump.

---

## The problem

In the last five years, published and competition-grade ML systems have repeatedly shipped with one of these three bugs:

| Bug | Real example | Impact |
|---|---|---|
| **Feature statistically mirrors the target** | [COVID-19 chest X-ray classifiers learned hospital-ID confounders](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0274098), not pulmonary features | Internal AUC 0.99, external-hospital AUC near-chance |
| **Forbidden / post-outcome feature in the input** | [JAMA Network Open 2024](https://jamanetwork.com/journals/jamanetworkopen/fullarticle/2843179): 40.2% of MIMIC same-admission prediction studies fed in ICD codes finalised at discharge | AUROC 0.97 from leaky codes alone |
| **Transform that reads across the whole dataset** | [Kaggle Santander 2019 "magic" leak](https://www.kaggle.com/c/santander-customer-transaction-prediction/discussion/84614): frequency features computed on (train ∪ real-test) | Public AUC jumped 0.90 → 0.92 |

Each one escaped peer review, code review, or competition scrutiny — because the bug isn't a type error. It's a statistical / semantic contract violation.

`schema-firewall` provides three drop-in checks, one per bug class.

---

## Usage

```python
import pandas as pd
from schema_firewall import (
    check_leakage,
    check_schema,
    check_stateless,
    SchemaContract,
    LeakageError,
)

X: pd.DataFrame  # your feature frame
y: pd.Series     # your target

# 1. Statistical leakage — Pearson + Spearman + adjusted mutual information.
#    Pearson catches linear copies, Spearman monotonic transforms, and the
#    chance-corrected MI catches NON-monotone and discrete deterministic leakage
#    (y=x**2, |x|, low-order oscillations, binary/k-class target encodings) that
#    both correlations miss — while leaving honest noisy predictors alone.
#    Detection is per-column (a multi-column combination like y = x1 XOR x2 is a
#    documented non-goal, as are high-frequency oscillatory encodings). Needs
#    >=100 rows. Raises LeakageError on fail.
check_leakage(X, y)

# 2. Schema contract — forbidden columns, required columns, dtypes.
#    Catches ICD-code-style post-outcome features and schema drift.
contract = SchemaContract(
    forbidden_columns=frozenset({"SALE PRICE", "PRICE_PER_SQFT"}),
    required_columns=frozenset({"sqft", "year_built"}),
)
check_schema(X, contract)

# 3. Statelessness — runs your feature pipeline on the full frame vs a
#    single-row subset. Flags any transform whose per-row output depends
#    on other rows: mean encoders, frequency encoders, target encoders
#    applied outside CV, ComBat/global normalisation, etc.
check_stateless(my_pipeline_fn, raw_frame)
```

Each function raises on failure and returns `None` on pass. No silent
degradation.

---

## The demo notebook

> [**`examples/leakage_demo.ipynb`**](examples/leakage_demo.ipynb) — 60 seconds, California housing dataset, one deliberate leak, one library call.

Open it. It reproduces the target-encoding bug that sits in real production pipelines, shows an R² that looks impressive, then one call to `check_stateless` catches the leak before the model ships.

If you've ever applied `.mean()`, `.value_counts()`, `TargetEncoder`, or ComBat/`fit_transform` to your full dataset before cross-validation, the notebook is pointed at you.

---

## Verified invariants under execution

The library is consumed in downstream CI today as a pinned dep of [`nyc-real-estate-predictor`](https://github.com/MarwaBS/nyc-real-estate-predictor). The flagship's `External Benchmark` CI job re-checks these invariants against the published wheel on a weekly schedule and on pushes/PRs touching the benchmark's paths (the job is path-filtered):

- **Statistical leakage detection triggers on the bundled California housing demo.** Build a target-mean-encoded feature on rounded lat/lon buckets — Ridge regression returns R² = 0.9495 (leaky). Apply the same target encoding per train fold only — R² collapses to 0.4384 (honest). Both `check_leakage` and `check_stateless` raise on the leaky pipeline. Reproducible in 60 seconds via [`examples/leakage_demo.ipynb`](examples/leakage_demo.ipynb).

- **Statelessness holds under subset perturbation.** `check_stateless` runs the user pipeline on the full frame, then on a one-row subset. Any transform whose per-row output depends on other rows (frequency encoders, target-mean encoders, ComBat-style global normalisation) fails this invariant by construction. The default spot-check deliberately targets the rows a global transform is most likely to edit — the min/max rows of every numeric column, in the input **and** in the frame the pipeline returns (winsorise/clip/quantile filters touch the tails; a low-variance standardised column is as likely a target as a high-variance one, and a column the pipeline derives has its tails nowhere in the input, so no column is skipped), the first few NaN rows **of each column separately** (a budget shared across columns is spent by whichever column is dirtiest, leaving the column that leaks unchecked), and a fixed-stride spread across the rest — rather than being fooled by a plain stride sample that misses a tail- or NaN-only edit. One row per column is enough for a fill that edits every NaN row the same way, which is what `fillna(df.mean())` does; a fill touching only a subset of a column's NaN rows is a coverage floor, not a guarantee, and the default samples at least three per column, widened toward a ten-row total when few columns carry NaNs. Cost scales with column count, not frame length; pass an explicit `sample_indices` to bound it on very wide frames or to check every row (the strongest guarantee).

- **Forbidden-column gate raises on the documented set.** `nyc-real-estate-predictor` configures `SchemaContract(forbidden_columns=frozenset({"SALE PRICE", "SALE DATE", "PRICE_PER_SQFT", "TARGET", "log_price"}))`. Verifiable from this repo: the parametrized `tests/test_checks.py::test_schema_rejects_forbidden_column` asserts `check_schema` raises on each of those names. The flagship additionally re-validates the integration in its own CI (see the downstream-usage note above); its internal test suite is that repo's claim, not verified here.

- **Determinism check catches non-deterministic transforms.** Two consecutive `pipeline_fn(raw)` calls must produce identical frames. Unseeded random initialisation, dict-order dependency, and side-effecting transforms all fail. Internal `pd.testing.assert_frame_equal` with `check_exact=True`, so a perturbation below pandas' default 1e-5 relative tolerance still fails rather than passing as "close enough".

These hold across the test matrix; numbers (test counts, coverage %) age — the invariants don't.

---

## What this is NOT

- Not a replacement for train/test splitting, cross-validation, or sklearn `Pipeline`.
- Not a feature-importance tool.
- Not a drift-monitoring service.
- Not a validation framework with its own DSL.

Three checks. One contract class. Four exceptions. That's the whole library.

---

## Design constraints (locked)

- **≤ 500 LoC** of core implementation across `src/schema_firewall/`, enforced by a test so the budget can't silently rot. Count the code lines yourself: `find src/schema_firewall -name '*.py' -exec grep -vhE '^\s*(#|$)' {} + | wc -l`.
- **3 public check functions** — `check_leakage`, `check_schema`, `check_stateless`. No more.
- **A hostile-input test for every documented failure mode** (and a regression test for each fixed bug).
- **Three dependencies:** `numpy`, `pandas`, `scikit-learn`. Nothing else.

If `schema-firewall` is missing a check you need, the library is wrong for your use case. Build the check in-line. Its surface will not grow to absorb it.

---

## How the failure-mode constraint is checked

The constraint above is backed by a check you can re-run yourself.

`tools/planted_defects.py` carries a registry of 20 failure modes — one per documented behaviour across the three checks. Each entry names the documentation that describes the behaviour, the exact source edit that disables it, and the tests that must fail once it is disabled. Running the script copies the project to a throwaway directory, verifies that copy is what gets imported (not an installed build), runs the named tests untouched as a control, then plants each defect in turn and requires its tests to go red. A defect that no test catches exits non-zero.

```console
$ python tools/planted_defects.py
control: 25 named tests green on pristine copy
CAUGHT   leakage-raise-disabled
CAUGHT   mi-binning-quantile-restored
...
all 20 planted defects caught; controls green
```

`tests/test_failure_mode_coverage.py` keeps the registry from rotting inside the ordinary suite: every entry's documentation anchor must still exist, every named test must still exist, every planted edit must still match exactly one place in the source, and the registry must cover all three public checks. Delete a test or reword the docs and the suite fails.

What this establishes: each of the 20 registered failure modes has at least one test that fails when the behaviour is removed. What it does not establish: that the registry is every way this library can break, that the suite is strong against defects nobody planted, or that the registry is complete — the check runs from the registry outwards, so documenting a new failure mode without registering it goes unnoticed. It is a coverage floor for the 20 registered modes, not a mutation score.

---

## When to use each check

| You did this | Run this |
|---|---|
| Built any feature-engineering function that reads the full frame | `check_stateless(pipeline_fn, raw)` |
| Joined multiple datasets with different origins / schemas / timestamps | `check_schema(X, SchemaContract(forbidden_columns=…))` |
| Want a fast sanity gate before training | `check_leakage(X, y)` on the final feature frame |

---

## What it caught in downstream usage

The `schema-firewall` checks are the same ones used by the [NYC Real Estate Predictor external benchmark](https://github.com/MarwaBS/nyc-real-estate-predictor) against NYC.gov 2024 Rolling Sales data. The flagship benchmark uses `schema-firewall` as a dependency, not a vendored copy. When the library breaks, the benchmark breaks. This is by design.

---

## Attribution

Extracted from the firewall layer of the NYC Real Estate Predictor's external benchmark. The scoring-determinism pattern comes from the Protocol-based core of the Job Decision Engine project. Credit for the underlying problem classes goes to:

- DeGrave et al. (*Nature Machine Intelligence*, 2021) — COVID X-ray shortcut learning
- Rosenblatt et al. (*Nature Communications*, 2024) — connectome leakage
- Ramadan et al. (*JAMIA*, 2024) — clinical label-leakage framework
- YaG320 — Santander "magic" competition kernel

---

## License

MIT. See [LICENSE](LICENSE).
