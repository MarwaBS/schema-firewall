# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.3] - 2026-06-16

Audit-driven hardening (2026-06): fixes a crash, two incorrect/ineffective
checks, and several error-handling and tooling gaps. Each fix carries a
regression test.

### Fixed
- `check_leakage` no longer crashes with a raw sklearn `ValueError` on a single
  NaN in any numeric feature or in the target; NaNs are dropped pairwise per
  column before every metric.
- `check_stateless` now fails (instead of silently skipping via `continue`) when
  a row kept in the full-frame output is dropped when processed alone — the
  false negative on global-statistic row filters like `df[df.x > df.x.median()]`.
- Clear errors replace raw pandas `KeyError`s: spot-checking a row the pipeline
  drops, and pipelines that reset/relabel the index, now raise messages naming
  the actual precondition.
- `check_schema` dtype comparison resolves equivalent spellings (`int64`, `i8`,
  `<i8`) instead of rejecting them via raw string equality.
- A non-numeric target to `check_leakage` raises a clear `LeakageError` rather
  than a raw "could not convert string to float".

### Changed
- MI is normalised by the target's self-information `MI(y; y)` under the same
  estimator, so `mi_norm` is genuinely in `[0, 1]` (a perfect copy → 1.0). The
  previous histogram-Shannon-entropy denominator let it exceed 1 (~1.33) and
  drift with the bin count; the `_shannon_entropy` helper was removed.
- The example demo's `check_stateless` branch now exits non-zero if the check
  misses the leak, so the README's "both checks raise" claim is enforced by
  running the demo (and by `tests/test_demo.py`).

### CI / tooling
- CI runs `ruff check`, `ruff format --check`, and `mypy src` (matching the
  CONTRIBUTING promise) and adds Python 3.13 to the matrix and classifiers.
- Linting now covers `tests/` and `examples/` (previously excluded).

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

[0.1.2]: https://github.com/MarwaBS/schema-firewall/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/MarwaBS/schema-firewall/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/MarwaBS/schema-firewall/releases/tag/v0.1.0
