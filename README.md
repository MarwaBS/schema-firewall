# schema-firewall

**Three checks that catch the leakage and schema bugs that slip past peer review.**

```bash
pip install schema-firewall
```

[![CI](https://github.com/MarwaBS/schema-firewall/actions/workflows/python-package.yml/badge.svg)](https://github.com/MarwaBS/schema-firewall/actions/workflows/python-package.yml)
[![PyPI](https://img.shields.io/pypi/v/schema-firewall.svg)](https://pypi.org/project/schema-firewall/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

> **Production usage.** Extracted from the firewall layer of [`nyc-real-estate-predictor`](https://github.com/MarwaBS/nyc-real-estate-predictor) — the flagship pins `schema-firewall==0.1.0` in `requirements.txt` and re-validates the firewall integration in its `External Benchmark` CI job on every push. Directional coupling signal (pinned dep + consuming CI), not a semantic contract invariant.

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

# 1. Statistical leakage — Pearson + Spearman + normalised mutual info.
#    Catches target-copies, monotonic transforms, sigmoid/rank re-encodings,
#    and strong confounders. Raises LeakageError on fail.
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

The library is in production use today as a pinned dep of [`nyc-real-estate-predictor`](https://github.com/MarwaBS/nyc-real-estate-predictor). The flagship's `External Benchmark` CI job re-checks these invariants against the published wheel on every push to `main`:

- **Statistical leakage detection triggers on the bundled California housing demo.** Build a target-mean-encoded feature on rounded lat/lon buckets — Ridge regression returns R² = 0.9495 (leaky). Apply the same target encoding per train fold only — R² collapses to 0.4384 (honest). Both `check_leakage` and `check_stateless` raise on the leaky pipeline. Reproducible in 60 seconds via [`examples/leakage_demo.ipynb`](examples/leakage_demo.ipynb).

- **Statelessness holds under subset perturbation.** `check_stateless` runs the user pipeline on the full frame, then on a one-row subset. Any transform whose per-row output depends on other rows (frequency encoders, target-mean encoders, ComBat-style global normalisation) fails this invariant by construction. Default samples five spread indices to avoid being fooled by a singleton-group row 0.

- **Forbidden-column gate raises on the documented set.** `nyc-real-estate-predictor` configures `SchemaContract(forbidden_columns=frozenset({"SALE PRICE", "SALE DATE", "PRICE_PER_SQFT", "TARGET", "log_price"}))`. The 18-test adversarial suite in the flagship asserts that `check_schema` raises on each of these columns presented under several disguises.

- **Determinism check catches non-deterministic transforms.** Two consecutive `pipeline_fn(raw)` calls must produce identical frames. Unseeded random initialisation, dict-order dependency, and side-effecting transforms all fail. Internal `pd.testing.assert_frame_equal`.

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

- **≤ 500 LoC** of core implementation across `src/schema_firewall/` — currently ~300 lines of code (~400 including blanks and comments), comfortably within budget.
- **3 public check functions** — `check_leakage`, `check_schema`, `check_stateless`. No more.
- **An adversarial test for every documented failure mode** (and a regression test for each fixed bug).
- **Three dependencies:** `numpy`, `pandas`, `scikit-learn`. Nothing else.

If `schema-firewall` v0.1 is missing a check you need, the library is wrong for your use case. Build the check in-line. v0.1 will not grow to absorb it.

---

## When to use each check

| You did this | Run this |
|---|---|
| Built any feature-engineering function that reads the full frame | `check_stateless(pipeline_fn, raw)` |
| Joined multiple datasets with different origins / schemas / timestamps | `check_schema(X, SchemaContract(forbidden_columns=…))` |
| Want a fast sanity gate before training | `check_leakage(X, y)` on the final feature frame |

---

## What it caught in production (dogfood)

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
