# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.2.x   | Yes       |
| < 0.2   | No        |

## Reporting a Vulnerability

Please report security vulnerabilities through GitHub's private
[security advisory feature](https://github.com/MarwaBS/schema-firewall/security/advisories/new).

Do **not** report security vulnerabilities through public GitHub
issues, discussions, or pull requests.

Expected initial response: within 7 days.

## Scope

`schema-firewall` is a validation library -- it inspects DataFrames
and pipeline functions you pass to it. It does not execute arbitrary
code, open network sockets, or persist data. Plausible vulnerability
classes are limited to:

- Crashes or hangs on hostile input frames.
- False negatives where a known-leaky pipeline passes a check.
- Dependency-chain issues inherited from numpy, pandas, or scikit-learn.

Anything outside that surface is out of scope.
