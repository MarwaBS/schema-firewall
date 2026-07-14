# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.3] - 2026-07-06

Audit-driven hardening (2026-06): fixes a crash, two incorrect/ineffective
checks, and several error-handling and tooling gaps. Each fix carries a
regression test.

### Fixed
- `check_leakage` no longer crashes with a raw sklearn `ValueError` on a single
  NaN in any numeric feature or in the target; NaNs are dropped pairwise per
  column before every metric.
- `check_stateless` now catches global-statistic row filters in two ways: an
  empty one-row output for a kept row fails (was silently skipped via
  `continue`), AND the default spot-check now includes each numeric column's
  extreme-value rows — so tail-only transforms (winsorise/clip/robust-scale,
  quantile filters) are caught deterministically instead of ~79% of the time.
- `check_stateless` now spot-checks NaN-bearing rows too, so global-mean/median
  imputation (`df.fillna(df.mean())`) — the canonical fit-on-full-data leak — is
  caught even when the NaN sits off the min/max/stride sample.
- `check_stateless` rejects a non-unique index (the per-row spot-check selects
  by label, so duplicate labels made it vacuous and a global transform passed).
- `check_leakage` raises a clear `ValueError` below 100 finite samples (was a raw
  sklearn crash / a |r|=1 two-point false positive). Below ~100 rows the binned
  MI is noise-dominated, so there is no honest threshold.
- Clear errors replace raw pandas `KeyError`s: spot-checking a row the pipeline
  drops, and pipelines that reset/relabel the index, now raise messages naming
  the actual precondition.
- `check_schema` dtype comparison resolves equivalent spellings (`int64`, `i8`,
  `<i8`) instead of rejecting them via raw string equality.
- A non-numeric target to `check_leakage` raises a clear `LeakageError` rather
  than a raw "could not convert string to float".

### Changed
- **MI detector rebuilt as adjusted (chance-corrected) mutual information** on
  sample-size-adaptive bins. The previous design (continuous kNN MI from
  `sklearn.feature_selection.mutual_info_regression`, normalised by the target's
  64-bin histogram Shannon entropy) was deflated on continuous targets and
  **missed non-monotone leakage** (`y = x**2` slipped through until a fix forced
  it). A later rebuild then silently collapsed
  **binary/low-cardinality targets** to one bin (AMI ≡ 0, every 0/1 target
  invisible). The final detector discretises low-cardinality values one-per-bin
  and continuous values into sqrt(n) quantile bins, and scores adjusted MI:
  genuinely in `[0, 1]`, ~0 under independence regardless of sample size, ~1
  when a feature determines the target — copy, binary/k-class encoding, OR
  non-monotone transform. Verified 0% miss on those leak fixtures and 0 false
  positives on independent / weakly-correlated columns, across seeds at n >= 100.
  It scores raw target dependence, not its source, so a *strong honest linear*
  predictor also crosses the threshold (measured: majority-flagged by `|r| >=
  0.85`) — deliberate for a leakage firewall; see the `check_leakage` docstring.
  Default `mi_threshold` recalibrated to `0.2`.
- The example demo's `check_stateless` branch now exits non-zero if the check
  misses the leak, so the README's "both checks raise" claim is enforced by
  running the demo (and by `tests/test_demo.py`).

### CI / tooling
- Removed the hardcoded `python_version = "3.10"` from `[tool.mypy]`: it made
  the gate fail on any fresh install under Python >= 3.12 (which resolves
  numpy >= 2.5, whose stubs use PEP 695 `type` statements — a syntax error in
  3.10 mode). mypy now checks at the running interpreter, so each CI matrix
  job validates its own resolved dependency universe; the 3.10 job still
  enforces the 3.10 floor.
- `release.yml` now refuses to publish when the pushed tag does not match the
  `pyproject.toml` version, so a mistagged push cannot upload a wheel whose
  version contradicts its tag.
- `tests/test_demo.py` now parses the claimed R² values out of README.md
  instead of hardcoding a copy next to a (previously drifted) line-number
  citation — the README is the single source of truth and the reference
  cannot drift again.
- CI runs `ruff check`, `ruff format --check`, and `mypy src` (matching the
  CONTRIBUTING promise) and adds Python 3.13 to the matrix and classifiers.
- Linting now covers `tests/` and `examples/` (previously excluded).
- Coverage is now **gated, not just mentioned**: `pytest-cov` is a dev
  dependency and `pytest` runs branch coverage with `--cov-fail-under=90`
  (current: 92%). A regression that drops a check's branch coverage now fails
  CI instead of sliding through behind a green badge. The floor sits below
  actual so it absorbs churn without being vacuous.

### Docs
- README "Default samples five spread indices" corrected — it described the
  pre-0.1.3 stride-only sampler. `check_stateless` now spot-checks each numeric
  column's min/max row, every NaN-bearing row, and a fixed-stride spread; the
  README now says so.
- README LoC figure refreshed to the verified current count (372 code / 515
  raw) with a one-line reproduce command, so the "≤ 500 LoC" budget claim can't
  silently rot again.
- README's "18-test adversarial suite in the flagship" citation replaced with
  what is verifiable from this repo: the parametrized forbidden-column test
  here (now including `SALE DATE`) plus the flagship's consuming CI job.

## [0.1.2] - 2026-05-28

Patch release closing 11 audit findings deferred from the 2026-05-23
ledger. Includes one behavioral change (see below).

### Added
- Regression test pinning `check_leakage` behavior on low-cardinality
  classification targets (3-class balanced y, N=200). Asserts the
  function passes on independent features and raises `LeakageError`
  on a target-copy feature. Closes the bug class that motivated
  retracted Finding #1 in the audit. (#5)
- Regression test asserting `check_stateless` raises `ValueError` when
  a `sample_indices` entry is not in `raw.index`. (#9)
- `CONTRIBUTING.md` at repo root: test-suite invocation, demo
  invocation, issue routing, the minimalism-lock as a contribution
  constraint, ASCII-source style note. (#25)

### Changed
- `_safe_corr` method parameter is now `Literal["pearson", "spearman"]`
  rather than `str`, so mypy catches typo'd call sites at
  static-analysis time. Runtime `ValueError` message tightened to
  quote the bogus method and list valid options for the residual
  untyped-caller case. (#27)
- `mutual_info_regression` import moved from inside `check_leakage`
  to the module-level imports of `_checks.py`. scikit-learn is a hard
  dependency; the function-local form saved no install cost and added
  an indirection per call. (#20)
- `pyproject.toml` runtime dependencies gained upper bounds:
  `numpy>=1.24,<3.0`, `pandas>=2.0,<3.0`, `scikit-learn>=1.3,<2.0`.
  Protects downstream users from numpy 2.x ABI breaks and future
  pandas/sklearn major bumps. Current dev versions remain in-range. (#11)
- Demo `examples/leakage_demo.py` no longer slices the exception
  message via `str(exc).splitlines()[N][:110]`. Now uses
  `str(exc)[:200]` — decouples the demo output from `_checks.py`
  exception-message line structure. (#13)
- `examples/leakage_demo.ipynb` honest-path cell now operates on
  `df_honest = df.copy()` so the leaky-path cell can be re-run
  without contamination from a previously-executed honest cell. (#14)
- README adversarial-test count: 27 -> 30 (verified via
  `pytest --collect-only -q`).

### Fixed
- `examples/leakage_demo.ipynb` is now committed with executed
  outputs. GitHub renders the demo's R^2 narrative inline
  (leaky=0.9495 / honest=0.4384 / gap=+0.5111). (#21)

### Behavioral changes
- `check_stateless` raises `ValueError` on unknown `sample_indices`
  entries, rather than silently skipping them. Previous behavior
  let a caller's typo result in zero actual spot-checks and a
  vacuous "pass". Callers who pass `sample_indices` should verify
  every index is in `raw.index`. (#9)

## [0.1.1] - 2026-05-23

### Added
- CI workflow running pytest on Ubuntu across Python 3.10, 3.11, and 3.12.
  Hardened with 10-minute timeout, least-privilege `contents: read`
  permissions, pip cache keyed on `pyproject.toml`, and concurrency
  cancellation on new pushes to the same ref.
- CI workflow status badge in README.
- Subprocess test pinning the leakage demo's documented R-squared values
  (0.9495 leaky, 0.4384 honest) within +/- 0.005. Surfaces sklearn or
  numpy version drift before it reaches a stale README.
- Default mypy + ruff configuration sections in `pyproject.toml`. mypy
  scoped to `src/`, ignoring missing imports for pandas and scikit-learn
  (no upstream stubs). ruff scoped to `src/` with conservative E/F/W/I
  rules.
- `CHANGELOG.md` and `SECURITY.md`.

### Changed
- `__version__` is now sourced from installed package metadata via
  `importlib.metadata` rather than duplicated as a literal in
  `__init__.py` and `pyproject.toml`. Bumping the version is now a
  single-file edit to `pyproject.toml`.
- README "Actual: ~305 LoC" claim replaced with verified counts
  (344 raw lines / 270 excluding blanks and comments).
- Leakage demo (`examples/leakage_demo.py`) output formatting: Unicode
  box-drawing, superscript-two, and em-dash characters replaced with
  ASCII (`-`, `^2`, `--`) so the demo renders under cp1252.

### Fixed
- Demo `examples/leakage_demo.py` crashed with `UnicodeEncodeError`
  on default Windows Python because print statements contained
  U+2500 (box-drawing), U+00B2 (superscript-2), and U+2014 (em-dash).
  All replaced with ASCII equivalents.
- Library `StatelessnessError` message contained U+2260 (not-equal
  sign). Any user code that printed the exception on a cp1252 stream
  crashed before showing the diagnostic. Replaced with the ASCII `!=`
  operator.
- Library docstrings and section-divider comments in
  `src/schema_firewall/_checks.py` contained additional non-ASCII
  (arrows, Greek letters, set operators). Code that printed the
  docstrings directly (e.g. `print(check_leakage.__doc__)`) crashed on
  cp1252 streams; `help()` degraded silently by showing `→`-style
  escape sequences instead of the actual character. All replaced with
  ASCII transliterations (`->` for arrows, `rho` for Greek rho, `union`
  for set-union, `--` for em-dashes).
- Demo's `catch_leak_via_leakage_check` previously printed a soft
  "did not trip" note when `check_leakage` failed to catch the leak.
  Now raises `AssertionError` so the demo and CI fail loud if sklearn's
  mutual-information estimator ever drifts below the 0.8 threshold.

## [0.1.0] - 2026-04-20

### Added
- Initial public release on PyPI.
- `check_leakage(X, y)` -- Pearson, Spearman, and normalized mutual
  information detectors combined by logical OR. Raises `LeakageError`
  on threshold violation.
- `check_schema(X, contract)` -- forbidden columns, required columns,
  and dtype validation against a frozen `SchemaContract`. Raises
  `SchemaError`.
- `check_stateless(pipeline_fn, raw)` -- determinism + row-wise
  state-independence check. Catches mean encoders, frequency encoders,
  and other Santander-style cross-row leaks. Raises `StatelessnessError`.
- Adversarial test suite (27 collected tests at release time).
- MIT license.

[0.1.3]: https://github.com/MarwaBS/schema-firewall/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/MarwaBS/schema-firewall/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/MarwaBS/schema-firewall/compare/ef5021df74e030cf90073d77089f9839677a154a...v0.1.1
[0.1.0]: https://github.com/MarwaBS/schema-firewall/tree/ef5021df74e030cf90073d77089f9839677a154a
