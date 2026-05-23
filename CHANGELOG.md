# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1] - Unreleased

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
  (arrows, Greek letters, set operators). `help(check_leakage)` and
  similar introspection on cp1252 streams crashed. All replaced with
  ASCII transliterations.
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

[0.1.1]: https://github.com/MarwaBS/schema-firewall/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/MarwaBS/schema-firewall/releases/tag/v0.1.0
