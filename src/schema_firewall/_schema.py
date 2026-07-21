"""SchemaContract -- declarative input contract consumed by check_schema."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType


@dataclass(frozen=True)
class SchemaContract:
    """Frozen schema contract for a feature frame.

    Attributes:
        forbidden_columns: columns that must not appear in X. Use this
            to block known target-derived or post-outcome features
            (e.g. ``{"SALE PRICE", "PRICE_PER_SQFT"}`` or
            ``{"ICD_CODE_AT_DISCHARGE"}``).
        required_columns: columns that must be present in X. A missing
            required column is a schema violation.
        dtypes: optional per-column dtype assertion. Keys are column
            names; values are numpy/pandas dtype strings (e.g.
            ``"int64"``, ``"float64"``, ``"object"``). Only columns
            listed here are dtype-checked.
    """

    forbidden_columns: frozenset[str] = field(default_factory=frozenset)
    required_columns: frozenset[str] = field(default_factory=frozenset)
    dtypes: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        # Copy then wrap read-only, so the contract cannot be rewritten after
        # construction -- neither by item assignment nor through the caller's dict.
        if self.dtypes is not None:
            object.__setattr__(self, "dtypes", MappingProxyType(dict(self.dtypes)))

    # A mappingproxy cannot be pickled; ship a plain dict across the boundary and
    # re-freeze on load, so contracts survive joblib/multiprocessing.
    def __getstate__(self) -> dict[str, object]:
        state = dict(self.__dict__)
        if self.dtypes is not None:
            state["dtypes"] = dict(self.dtypes)
        return state

    def __setstate__(self, state: dict[str, object]) -> None:
        for key, value in state.items():
            object.__setattr__(self, key, value)
        self.__post_init__()


__all__ = ["SchemaContract"]
