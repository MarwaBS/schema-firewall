"""schema-firewall -- a tiny, adversarially-tested library that catches the
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

from importlib import metadata as _metadata

from ._checks import check_leakage, check_schema, check_stateless
from ._exceptions import (
    LeakageError,
    SchemaError,
    SchemaFirewallError,
    StatelessnessError,
)
from ._schema import SchemaContract

try:
    __version__ = _metadata.version("schema-firewall")
except _metadata.PackageNotFoundError:
    # Editable source clone without `pip install -e .`. Should not occur
    # in released wheels; the literal here is a "not a real release"
    # sentinel rather than a stale version that drifts from pyproject.
    __version__ = "0.0.0+local"

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
