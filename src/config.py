"""Centralised configuration for the AI Audit and Risk Analytics Tool."""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field, fields, asdict
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Config:
    # Statistical outlier
    zscore_threshold: float = 3.0
    min_account_tx: int = 10

    # Benford
    benford_min_count: int = 100
    benford_significance: float = 0.05

    # Duplicate payment
    duplicate_window_days: int = 7

    # Timing
    business_start: datetime.time = datetime.time(8, 0)
    business_end: datetime.time = datetime.time(18, 0)

    # Round number
    round_threshold: float = 1000.0

    # ML
    min_ml_tx: int = 100
    random_state: int = 42

    # Risk scoring weights (rule -> base points)
    rule_weights: dict[str, int] = field(default_factory=lambda: {
        "statistical_outlier": 30,
        "benford_deviation": 20,
        "duplicate_payment": 35,
        "weekend_posting": 10,
        "after_hours_posting": 15,
        "round_amount": 10,
        "ml_isolation_forest": 25,
    })

    # OpenAI / Summary
    openai_model: str = "gpt-4o-mini"
    openai_max_tokens: int = 600
    openai_timeout: int = 30
    top_risks_in_summary: int = 5
    summary_max_words: int = 400


def _parse_time(value: Any) -> datetime.time:
    """Parse a time from 'HH:MM', 'HH:MM:SS', or an existing time object."""
    if isinstance(value, datetime.time):
        return value
    if isinstance(value, str):
        parts = value.strip().split(":")
        if len(parts) == 2:
            return datetime.time(int(parts[0]), int(parts[1]))
        if len(parts) == 3:
            return datetime.time(int(parts[0]), int(parts[1]), int(parts[2]))
    raise ValueError(f"Invalid time value: {value!r}")


def load_config(path: str | Path = "config.yaml") -> Config:
    """Load configuration from a YAML file, merging over dataclass defaults.

    Missing keys, or a missing file, fall back to defaults gracefully.
    """
    defaults = Config()
    path = Path(path)
    overrides: dict[str, Any] = {}
    if path.exists():
        with path.open("r", encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"Config file {path} must contain a YAML mapping")
        overrides = loaded

    valid_names = {f.name for f in fields(Config)}
    kwargs: dict[str, Any] = {}
    for key, value in overrides.items():
        if key not in valid_names:
            # Unknown keys are ignored so users can keep extra metadata in the file.
            continue
        if key in ("business_start", "business_end"):
            kwargs[key] = _parse_time(value)
        elif key == "rule_weights":
            merged = dict(defaults.rule_weights)
            if isinstance(value, dict):
                merged.update(value)
            kwargs[key] = merged
        else:
            kwargs[key] = value
    return Config(**kwargs)


def config_snapshot(config: Config) -> dict[str, Any]:
    """Serialise a Config into a JSON-safe dict for the run log."""
    snapshot = asdict(config)
    snapshot["business_start"] = config.business_start.isoformat()
    snapshot["business_end"] = config.business_end.isoformat()
    return snapshot
