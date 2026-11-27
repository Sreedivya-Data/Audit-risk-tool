"""Unit tests for src/data_generator.py.

Covers seed reproducibility, schema correctness, anomaly planting,
and CLI behaviour.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import pytest

from src.data_generator import (
    REQUIRED_COLUMNS,
    AnomalySpec,
    generate,
    main,
)


# ---------------------------------------------------------------------------
# Schema and basic contract
# ---------------------------------------------------------------------------


def test_required_columns_present():
    df = generate(50, seed=0)
    for col in REQUIRED_COLUMNS:
        assert col in df.columns, f"Column {col!r} missing from output"


def test_exact_count_no_anomalies():
    df = generate(100, seed=1)
    assert len(df) == 100


def test_date_dtype_is_date():
    df = generate(50, seed=2)
    for d in df["date"]:
        assert isinstance(d, dt.date), f"Expected date, got {type(d)}"


def test_time_dtype_is_time():
    df = generate(50, seed=3)
    for t in df["time"]:
        assert isinstance(t, dt.time), f"Expected time, got {type(t)}"


def test_amounts_positive():
    df = generate(200, seed=4)
    assert (df["amount"] > 0).all(), "All base amounts should be positive"


def test_accounts_subset_of_expected():
    df = generate(200, seed=5)
    expected = {f"ACC-{i:02d}" for i in range(1, 21)}
    assert set(df["account"]).issubset(expected)


def test_vendors_subset_of_expected():
    df = generate(200, seed=6)
    expected = {f"VEN-{i:03d}" for i in range(1, 51)}
    assert set(df["vendor"]).issubset(expected)


def test_approvers_subset_of_expected():
    df = generate(200, seed=7)
    expected = {f"APR-{i:02d}" for i in range(1, 11)}
    assert set(df["approver"]).issubset(expected)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def test_same_seed_produces_identical_output():
    df1 = generate(100, seed=42)
    df2 = generate(100, seed=42)
    # Compare all columns element-by-element
    assert list(df1["amount"]) == list(df2["amount"])
    assert list(df1["account"]) == list(df2["account"])
    assert list(df1["vendor"]) == list(df2["vendor"])
    assert list(df1["date"]) == list(df2["date"])


def test_different_seeds_produce_different_output():
    df1 = generate(100, seed=1)
    df2 = generate(100, seed=2)
    # With high probability the amount lists differ
    assert list(df1["amount"]) != list(df2["amount"])


def test_same_seed_different_count_differ():
    df1 = generate(50, seed=42)
    df2 = generate(100, seed=42)
    assert len(df1) != len(df2)


# ---------------------------------------------------------------------------
# Anomaly: statistical_outlier
# ---------------------------------------------------------------------------


def test_statistical_outlier_count():
    specs = [AnomalySpec(type="statistical_outlier", count=5)]
    df = generate(200, seed=10, anomalies=specs)
    assert len(df) == 200  # no rows added


def test_statistical_outlier_amounts_extreme():
    """Planted outlier amounts should be much larger than the typical range."""
    specs = [AnomalySpec(type="statistical_outlier", count=10, params={"multiplier": 10.0})]
    df_base = generate(500, seed=11)
    df_anom = generate(500, seed=11, anomalies=specs)
    max_base = df_base["amount"].max()
    # At least one amount should exceed the clean max
    assert df_anom["amount"].max() > max_base


def test_statistical_outlier_reproduces():
    specs = [AnomalySpec(type="statistical_outlier", count=5)]
    df1 = generate(100, seed=20, anomalies=specs)
    df2 = generate(100, seed=20, anomalies=specs)
    assert list(df1["amount"]) == list(df2["amount"])


# ---------------------------------------------------------------------------
# Anomaly: benford_violation
# ---------------------------------------------------------------------------


def test_benford_violation_leading_digit():
    specs = [AnomalySpec(type="benford_violation", count=50)]
    df = generate(500, seed=30, anomalies=specs)
    # At least 50 amounts should start with digit '9'
    leading_9 = sum(1 for a in df["amount"] if str(a).lstrip("0")[0] == "9")
    assert leading_9 >= 50


def test_benford_violation_no_extra_rows():
    specs = [AnomalySpec(type="benford_violation", count=10)]
    df = generate(100, seed=31, anomalies=specs)
    assert len(df) == 100


# ---------------------------------------------------------------------------
# Anomaly: duplicate_payment
# ---------------------------------------------------------------------------


def test_duplicate_payment_adds_rows():
    specs = [AnomalySpec(type="duplicate_payment", count=5)]
    df = generate(100, seed=40, anomalies=specs)
    assert len(df) == 105  # 5 copies appended


def test_duplicate_payment_pairs_within_window():
    """Each copy should share vendor/amount with an original row within 7 days."""
    specs = [AnomalySpec(type="duplicate_payment", count=3)]
    df = generate(100, seed=41, anomalies=specs)
    # For each of the last 3 rows (the copies), find a matching (vendor, amount) row
    copies = df.tail(3)
    originals = df.head(100)
    for _, copy_row in copies.iterrows():
        partners = originals[
            (originals["vendor"] == copy_row["vendor"])
            & (originals["amount"] == copy_row["amount"])
        ]
        assert len(partners) > 0, "Copy has no matching original row"
        # Date gap should be <= 7 days
        gap = abs((copy_row["date"] - partners.iloc[0]["date"]).days)
        assert gap <= 7, f"Date gap {gap} exceeds 7-day window"


# ---------------------------------------------------------------------------
# Anomaly: weekend_posting
# ---------------------------------------------------------------------------


def test_weekend_posting_plants_saturdays():
    specs = [AnomalySpec(type="weekend_posting", count=10)]
    df = generate(200, seed=50, anomalies=specs)
    weekend_dates = [d for d in df["date"] if d.weekday() >= 5]
    assert len(weekend_dates) >= 10


def test_weekend_posting_no_extra_rows():
    specs = [AnomalySpec(type="weekend_posting", count=5)]
    df = generate(100, seed=51, anomalies=specs)
    assert len(df) == 100


# ---------------------------------------------------------------------------
# Anomaly: after_hours_posting
# ---------------------------------------------------------------------------


def test_after_hours_posting_time_22():
    specs = [AnomalySpec(type="after_hours_posting", count=10)]
    df = generate(100, seed=60, anomalies=specs)
    after_hours = [t for t in df["time"] if t == dt.time(22, 0, 0)]
    assert len(after_hours) == 10


def test_after_hours_posting_no_extra_rows():
    specs = [AnomalySpec(type="after_hours_posting", count=5)]
    df = generate(100, seed=61, anomalies=specs)
    assert len(df) == 100


# ---------------------------------------------------------------------------
# Anomaly: round_amount
# ---------------------------------------------------------------------------


def test_round_amount_multiples_of_1000():
    specs = [AnomalySpec(type="round_amount", count=20)]
    df = generate(200, seed=70, anomalies=specs)
    round_amounts = [a for a in df["amount"] if a % 1000 == 0]
    assert len(round_amounts) >= 20


def test_round_amount_custom_threshold():
    specs = [AnomalySpec(type="round_amount", count=10, params={"threshold": 500.0})]
    df = generate(200, seed=71, anomalies=specs)
    round_amounts = [a for a in df["amount"] if a % 500 == 0]
    assert len(round_amounts) >= 10


def test_round_amount_no_extra_rows():
    specs = [AnomalySpec(type="round_amount", count=5)]
    df = generate(100, seed=72, anomalies=specs)
    assert len(df) == 100


# ---------------------------------------------------------------------------
# Anomaly: ml_pattern
# ---------------------------------------------------------------------------


def test_ml_pattern_no_extra_rows():
    specs = [AnomalySpec(type="ml_pattern", count=5)]
    df = generate(100, seed=80, anomalies=specs)
    assert len(df) == 100


def test_ml_pattern_produces_weekends_and_after_hours():
    specs = [AnomalySpec(type="ml_pattern", count=10)]
    df = generate(200, seed=81, anomalies=specs)
    weekend_or_22 = df[
        (df["date"].apply(lambda d: d.weekday()) >= 5)
        | (df["time"] == dt.time(22, 0, 0))
    ]
    assert len(weekend_or_22) >= 10


# ---------------------------------------------------------------------------
# Multiple anomaly specs
# ---------------------------------------------------------------------------


def test_multiple_anomaly_specs():
    specs = [
        AnomalySpec(type="weekend_posting", count=5),
        AnomalySpec(type="after_hours_posting", count=5),
        AnomalySpec(type="duplicate_payment", count=3),
    ]
    df = generate(100, seed=90, anomalies=specs)
    assert len(df) == 103  # 3 duplicates appended


def test_multiple_specs_reproducible():
    specs = [
        AnomalySpec(type="statistical_outlier", count=5),
        AnomalySpec(type="round_amount", count=5),
    ]
    df1 = generate(100, seed=91, anomalies=specs)
    df2 = generate(100, seed=91, anomalies=specs)
    assert list(df1["amount"]) == list(df2["amount"])


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_invalid_count_raises():
    with pytest.raises(ValueError, match="count must be >= 1"):
        generate(0, seed=0)


def test_invalid_anomaly_type_raises():
    with pytest.raises(ValueError, match="Unknown anomaly type"):
        generate(50, seed=0, anomalies=[AnomalySpec(type="nonexistent", count=1)])  # type: ignore[arg-type]


def test_anomaly_count_exceeds_dataset_raises():
    with pytest.raises(ValueError):
        generate(10, seed=0, anomalies=[AnomalySpec(type="weekend_posting", count=20)])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_creates_csv(tmp_path: Path):
    out = tmp_path / "output.csv"
    rc = main(["--count", "50", "--seed", "0", "--output", str(out)])
    assert rc == 0
    assert out.exists()
    df = pd.read_csv(out)
    assert len(df) == 50
    for col in REQUIRED_COLUMNS:
        assert col in df.columns


def test_cli_anomalies_flag(tmp_path: Path):
    out = tmp_path / "output.csv"
    rc = main([
        "--count", "100",
        "--seed", "5",
        "--output", str(out),
        "--anomalies", "weekend_posting:5,round_amount:5",
    ])
    assert rc == 0
    df = pd.read_csv(out)
    assert len(df) == 100  # no extra rows for these anomaly types
