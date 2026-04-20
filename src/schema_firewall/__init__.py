"""schema-firewall — a tiny, adversarially-tested library that catches the
three leakage + schema bugs that keep sneaking into published ML results.

Public API (three checks, one contract, four exceptions):

    from schema_firewall import (
        check_leakage,          # Pearson + Spearman + normalised MI
        check_schema,           # forbidden / required columns + dtypes
        check_stateless,        # determinism + row-wise state-independence
        SchemaContract,         # declarative input contract
        LeakageError,           # raised by check_leakage
        SchemaError,            # raised by check_schema
        StatelessnessError,     # raised by check_stateless
        SchemaFirewallError,    # common base
    )

Each check raises on failure and returns None on pass. No truthy/falsy
return values, no silent degradation, no runtime config side effects.
"""
from ._checks import check_leakage, check_schema, check_stateless
from ._exceptions import (
    LeakageError,
    SchemaError,
    SchemaFirewallError,
    StatelessnessError,
)
from ._schema import SchemaContract

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "check_leakage",
    "check_schema",
    "check_stateless",
    "SchemaContract",
    "LeakageError",
    "SchemaError",
    "StatelessnessError",
    "SchemaFirewallError",
]
