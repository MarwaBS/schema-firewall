"""Public exceptions raised by schema-firewall checks.

Each exception class corresponds to one check function. Catching a
specific subclass lets callers distinguish *what* failed, without
string-matching on error messages.
"""
from __future__ import annotations


class SchemaFirewallError(Exception):
    """Base class for all schema-firewall failures."""


class LeakageError(SchemaFirewallError):
    """A feature in X shows suspicious statistical dependency with the target."""


class SchemaError(SchemaFirewallError):
    """The input frame violates the declared SchemaContract."""


class StatelessnessError(SchemaFirewallError):
    """A transformation is not deterministic or not row-independent."""


__all__ = [
    "SchemaFirewallError",
    "LeakageError",
    "SchemaError",
    "StatelessnessError",
]
