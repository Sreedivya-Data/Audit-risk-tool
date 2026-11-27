"""Synthetic transaction generator with planted-anomaly support.

Produces reproducible DataFrames (identical seed + count + anomalies ->
identical output) that feed detection-accuracy tests and demos.
"""

from __future__ import annotations

import argparse
import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

# Type of anomaly that can be planted in a generated dataset.
AnomalyType = Literal[
    "statistical_outlier", "benford_violation", "duplicate_payment",
    "weekend_posting", "after_hours_posting", "round_amount",
    "ml_pattern",
]

VALID_ANOMALY_TYPES: tuple[str, ...] = (
    "statistical_outlier", "benford_violation", "duplicate_payment",
    "weekend_posting", "after_hours_posting", "round_amount",
    "ml_pattern",
)

REQUIRED_COLUMNS: tuple[str, ...] = (
    "date", "time", "account", "amount", "approver", "vendor", "description",
)

DEFAULT_WINDOW_START = dt.date(2024, 1, 1)
DEFAULT_WINDOW_DAYS = 365  # 12-month window


@dataclass
class AnomalySpec:
    """Specification for a set of planted anomalies.

    ``params`` is type-specific, e.g. for ``statistical_outlier`` one may
    pass {"multiplier": 10.0}; unused params are ignored.
    """

    type: AnomalyType
    count: int = 1                 # number of anomalies to plant
    params: dict = field(default_factory=dict)


def _make_time(rng: np.random.Generator) -> dt.time:
    """Sample a posting time centred in the business day (09:00-17:00)."""
    # Normal around 13:00 (46,800 s) with 2.5 h std, clipped to 09:00-17:00.
    seconds = int(np.clip(rng.normal(46_800, 9_000), 32_400, 61_200))
    return dt.time(seconds // 3600, (seconds % 3600) // 60, seconds % 60)


def _base_data(
    count: int,
    rng: np.random.Generator,
    window_start: dt.date,
    window_days: int,
) -> pd.DataFrame:
    """Generate the clean (pre-anomaly) transaction population."""
    n_accounts = 20
    n_vendors = 50
    n_approvers = 10

    accounts = [f"ACC-{i + 1:02d}" for i in range(n_accounts)]
    vendors = [f"VEN-{i + 1:03d}" for i in range(n_vendors)]
    approvers = [f"APR-{i + 1:02d}" for i in range(n_approvers)]

    # Per-account amount profile: mean 5,000-50,000, std 10-20% of mean.
    means = rng.uniform(5_000, 50_000, size=n_accounts)
    stds = means * rng.uniform(0.10, 0.20, size=n_accounts)

    account_idx = rng.integers(0, n_accounts, size=count)
    amounts = rng.normal(means[account_idx], stds[account_idx])
    amounts = np.round(np.clip(amounts, 1.0, None), 2)  # keep positive cents

    # Dates: uniform across a 12-month window.
    all_days = [window_start + dt.timedelta(days=i) for i in range(window_days)]
    weekdays = [d for d in all_days if d.weekday() < 5]
    day_offsets = rng.integers(0, len(weekdays), size=count)
    dates = [weekdays[int(i)] for i in day_offsets]

    # Times: normally distributed around mid-business-day.
    times = [_make_time(rng) for _ in range(count)]

    vendor_idx = rng.integers(0, n_vendors, size=count)
    approver_idx = rng.integers(0, n_approvers, size=count)

    descriptions = [
        f"Payment for services batch {i + 1}" for i in range(count)
    ]

    return pd.DataFrame(
        {
            "date": dates,
            "time": times,
            "account": [accounts[i] for i in account_idx],
            "amount": amounts.astype(float),
            "approver": [approvers[i] for i in approver_idx],
            "vendor": [vendors[i] for i in vendor_idx],
            "description": descriptions,
        }
    )


def _nearest_saturday(d: dt.date) -> dt.date:
    """Saturday of the week containing ``d`` (or the following Saturday)."""
    days_ahead = 5 - d.weekday()  # Saturday == weekday 5
    if days_ahead < 0:
        days_ahead += 7
    return d + dt.timedelta(days=days_ahead)


def _plant_statistical_outlier(
    df: pd.DataFrame, rng: np.random.Generator, spec: AnomalySpec
) -> pd.DataFrame:
    """Replace ``count`` amounts with ~10x their account mean."""
    multiplier = float(spec.params.get("multiplier", 10.0))
    rows = rng.choice(len(df), size=spec.count, replace=False)
    for row in rows:
        acct = df.at[row, "account"]
        acct_rows = df[df["account"] == acct]
        acct_mean = float(acct_rows["amount"].mean())
        df.at[row, "amount"] = round(acct_mean * multiplier, 2)
    return df


def _plant_benford_violation(
    df: pd.DataFrame, rng: np.random.Generator, spec: AnomalySpec
) -> pd.DataFrame:
    """Replace ``count`` amounts with 9-leading-digit values (9,000-9,999)."""
    rows = rng.choice(len(df), size=spec.count, replace=False)
    for row in rows:
        value = rng.uniform(9_000, 9_999)
        df.at[row, "amount"] = round(value, 2)
    return df


def _plant_duplicate_payment(
    df: pd.DataFrame, rng: np.random.Generator, spec: AnomalySpec
) -> pd.DataFrame:
    """Copy ``count`` rows with dates 1-3 days after the original."""
    rows = rng.choice(len(df), size=spec.count, replace=False)
    copies = []
    for row in rows:
        copy = df.loc[row].copy()
        gap = int(rng.integers(1, 4))  # 1-3 days, inside the 7-day window
        copy["date"] = copy["date"] + dt.timedelta(days=gap)
        copies.append(copy)
    if copies:
        df = pd.concat([df, pd.DataFrame(copies)], ignore_index=True)
    return df


def _plant_weekend_posting(
    df: pd.DataFrame, rng: np.random.Generator, spec: AnomalySpec
) -> pd.DataFrame:
    """Move ``count`` row dates to the nearest Saturday."""
    rows = rng.choice(len(df), size=spec.count, replace=False)
    for row in rows:
        df.at[row, "date"] = _nearest_saturday(df.at[row, "date"])
    return df


def _plant_after_hours_posting(
    df: pd.DataFrame, rng: np.random.Generator, spec: AnomalySpec
) -> pd.DataFrame:
    """Set ``count`` row times to 22:00 (outside 08:00-18:00)."""
    rows = rng.choice(len(df), size=spec.count, replace=False)
    for row in rows:
        df.at[row, "time"] = dt.time(22, 0, 0)
    return df


def _plant_round_amount(
    df: pd.DataFrame, rng: np.random.Generator, spec: AnomalySpec
) -> pd.DataFrame:
    """Set ``count`` amounts to exact multiples of the threshold (1,000)."""
    threshold = float(spec.params.get("threshold", 1_000.0))
    rows = rng.choice(len(df), size=spec.count, replace=False)
    for row in rows:
        multiple = int(rng.integers(1, 51))  # 1,000 - 50,000
        df.at[row, "amount"] = float(multiple * int(threshold))
    return df


def _plant_ml_pattern(
    df: pd.DataFrame, rng: np.random.Generator, spec: AnomalySpec
) -> pd.DataFrame:
    """Combine high amount + weekend + 22:00 time on ``count`` rows.

    Each condition alone is rule-detectable, but the co-occurrence pattern
    is the target of the ML layer.
    """
    rows = rng.choice(len(df), size=spec.count, replace=False)
    for row in rows:
        acct = df.at[row, "account"]
        acct_mean = float(df[df["account"] == acct]["amount"].mean())
        df.at[row, "amount"] = round(acct_mean * 5, 2)
        df.at[row, "date"] = _nearest_saturday(df.at[row, "date"])
        df.at[row, "time"] = dt.time(22, 0, 0)
    return df


_PLANTERS = {
    "statistical_outlier": _plant_statistical_outlier,
    "benford_violation": _plant_benford_violation,
    "duplicate_payment": _plant_duplicate_payment,
    "weekend_posting": _plant_weekend_posting,
    "after_hours_posting": _plant_after_hours_posting,
    "round_amount": _plant_round_amount,
    "ml_pattern": _plant_ml_pattern,
}


def generate(
    count: int,
    seed: int,
    anomalies: list[AnomalySpec] | None = None,
) -> pd.DataFrame:
    """Generate a synthetic transaction DataFrame.

    Identical seed + count + anomalies produces identical output. The seed
    is applied before any random generation.

    Args:
        count: number of base transactions to generate.
        seed: RNG seed for reproducibility.
        anomalies: optional list of anomaly specifications to plant.

    Returns:
        DataFrame with columns date, time, account, amount, approver,
        vendor, description (plus planted rows for duplicates).
    """
    if count < 1:
        raise ValueError(f"count must be >= 1, got {count}")

    rng = np.random.default_rng(seed)
    df = _base_data(count, rng, DEFAULT_WINDOW_START, DEFAULT_WINDOW_DAYS)

    for spec in anomalies or []:
        if spec.type not in VALID_ANOMALY_TYPES:
            raise ValueError(
                f"Unknown anomaly type: {spec.type!r}. "
                f"Valid types: {', '.join(VALID_ANOMALY_TYPES)}"
            )
        if spec.count < 1:
            raise ValueError(f"AnomalySpec.count must be >= 1, got {spec.count}")
        base_len = count  # duplicate planting grows df, guard against original count
        if spec.count > base_len:
            raise ValueError(
                f"Cannot plant {spec.count} anomalies in a dataset of {base_len} rows"
            )
        df = _PLANTERS[spec.type](df, rng, spec)

    df = df.reset_index(drop=True)
    df = df[list(REQUIRED_COLUMNS)]
    return df


def main(argv: list[str] | None = None) -> int:
    """CLI: python data_generator.py --count N --seed S --output path.csv"""
    parser = argparse.ArgumentParser(
        description="Generate synthetic audit transactions."
    )
    parser.add_argument("--count", type=int, default=1_000,
                        help="number of transactions to generate")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed for reproducibility")
    parser.add_argument("--output", type=str, default="data/synthetic.csv",
                        help="output CSV path")
    parser.add_argument("--anomalies", type=str, default="",
                        help="comma-separated type:count pairs, e.g. "
                             "'statistical_outlier:20,round_amount:10'")
    args = parser.parse_args(argv)

    anomalies: list[AnomalySpec] = []
    if args.anomalies:
        for chunk in args.anomalies.split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            type_name, _, cnt = chunk.partition(":")
            anomalies.append(AnomalySpec(type=type_name, count=int(cnt or 1)))

    df = generate(args.count, args.seed, anomalies)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} rows to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
