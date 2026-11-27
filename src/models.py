"""Core data models shared across pipeline modules."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Flag:
    """A single rule/model finding attached to a transaction."""

    tx_id: str
    rule: str           # e.g. "statistical_outlier"
    threshold: str      # human-readable, e.g. "|z| > 3.0"
    evidence: dict = field(default_factory=dict)  # rule-specific key/value pairs


@dataclass
class ValidationReport:
    """Outcome of Transaction_Loader cleaning and validation."""

    rows_loaded: int = 0
    rows_coerced: int = 0    # rows where >=1 field was coerced with raw value preserved
    rows_excluded: int = 0    # rows dropped due to missing required fields
    duplicates_removed: int = 0
    coercion_detail: list[dict] = field(default_factory=list)   # [{row_index, field, raw_value, coerced_value}]
    exclusion_detail: list[dict] = field(default_factory=list)  # [{row_index, reason}]
