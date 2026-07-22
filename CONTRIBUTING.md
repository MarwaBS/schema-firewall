# Contributing to schema-firewall

Thanks for your interest. This file covers how to run the suite locally,
how to file issues, and the scope contract that binds new contributions.

## Running the test suite

The dev extras pull in everything you need to run tests, type-check,
and lint:

```bash
pip install -e .[dev]
pytest
```

Expected: the full suite passes in under 20 seconds. (No fixed count here on
purpose — it drifts every time a test is added; `pytest` reports the current
number.)

To also run static analysis (the same commands CI runs):

```bash
ruff check .
ruff format --check .
mypy src tests
```

Both should be green on `main`. If they aren't on a fresh checkout,
that's a real bug — please open an issue.

## Running the demo

```bash
python examples/leakage_demo.py
```

Expected output ends with `leaky R^2 = 0.9495` and `honest R^2 = 0.4384`
on the California-housing dataset. If the R^2 values drift outside
[0.94, 0.96] / [0.43, 0.45], the subprocess test in `tests/test_demo.py`
will fail in CI before review.

## Filing issues

Bug reports, feature requests, and questions all go to
[GitHub Issues](https://github.com/MarwaBS/schema-firewall/issues).
For security-sensitive reports, see `SECURITY.md` — those route
through GitHub Security Advisories, not the public issue tracker.

## Scope contract — the minimalism lock

The README states the lock explicitly:

> Three checks. One contract class. Four exceptions. That's the whole library.

Pull requests that add to the public API surface — a fourth `check_*`
function, a second contract dataclass, a fifth exception type — will
be rejected on principle, not on quality. The public surface is locked at
this shape so users can audit the entire surface in 5 minutes.

If you find a real bug class the existing three checks can't catch,
open an issue first to discuss whether it warrants expanding the locked
surface or whether it fits inside the existing three.

In-scope PRs:

- Bug fixes that preserve the public API.
- Performance improvements to internal helpers.
- Better diagnostic messages on existing exceptions.
- Test additions covering documented invariants.
- Documentation, type-hint refinements, and tooling.

Out-of-scope without prior issue discussion:

- New public functions, classes, or exceptions.
- Changes to function signatures of `check_leakage`, `check_schema`,
  `check_stateless`, or `SchemaContract` that would break existing
  callers.
- New runtime dependencies (the library is locked at 3:
  numpy, pandas, scikit-learn).

## Code style

- `ruff check .` and `ruff format --check .` must be green. Conservative
  rule set (E, F, W, I) is already configured in `pyproject.toml`.
- `mypy src tests` must be green. Non-strict mode; `pandas-stubs` (a dev
  dependency) supplies pandas types, and scikit-learn imports are ignored
  (no upstream stubs).
- Line length: 100 characters (also from `pyproject.toml`).
- ASCII source. Non-ASCII characters in docstrings and comments
  caused cp1252 console crashes on Windows in 0.1.1, so the 0.1.x
  series keeps every `.py` under `src/` and `tests/` ASCII-only.
  This is **enforced**, not just requested: `test_source_is_ascii_only`
  fails the build on any non-ASCII byte. Use `->` for arrows,
  `rho` / `union` for Greek letters, `--` for em-dashes.

## Release process (maintainers only)

The full release flow is documented in `CHANGELOG.md` entries and
recorded in the audit ledger. Versions are single-sourced from
`pyproject.toml`; bumping the patch version is a one-file edit
because `__init__.py` reads from package metadata via
`importlib.metadata`.
