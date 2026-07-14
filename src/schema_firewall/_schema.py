"""SchemaContract -- declarative input contract consumed by check_schema."""

from __future__ import annotations

from dataclasses import dataclass, field


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
    dtypes: dict[str, str] | None = None


__all__ = ["SchemaContract"]
