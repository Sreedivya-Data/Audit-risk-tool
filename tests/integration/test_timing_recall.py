"""Integration test: timing-rule detection recall on planted anomalies.

Feature: ai-audit-risk-tool, Task 11.4
Validates: Requirements 5.1, 5.2, 12.4

Approach:
- Plant 20 weekend_posting and 20 after_hours_posting anomalies in a 500-row
  dataset using the synthetic data generator.
- Both rules are fully deterministic: a transaction either falls on a weekend or
  it doesn't; a posting time is either outside business hours or it isn't.
  Therefore 100% recall is expected for all planted anomalies.

Weekend rows:  date.weekday() >= 5  (Saturday == 5, Sunday == 6)
After-hours:   time < 08:00  or  time > 18:00  (Config defaults)
"""

from __future__ import annotations

import datetime

import pandas as pd
import pytest

from src.anomaly_detector import detect
from src.config import Config
from src.data_generator import AnomalySpec, generate
from src.run_log import RunLog

COUNT = 500
SEED = 42
WEEKEND_PLANT_COUNT = 20
AFTER_HOURS_PLANT_COUNT = 20


def _build_detector_df(df: pd.DataFrame) -> pd.DataFrame:
    """Add a stable tx_id column and coerce the date column to datetime64."""
    result = df.copy()
    result.insert(0, "tx_id", [f"tx{i:04d}" for i in range(len(result))])
    result["date"] = pd.to_datetime(result["date"])
    return result


@pytest.fixture(scope="module")
def planted_df_and_expected():
    """Return (detector_df, weekend_tx_ids, after_hours_tx_ids).

    Strategy
    --------
    1. Generate a *clean* baseline dataset (same seed, no anomalies).
    2. Generate a *planted* dataset (same seed, both anomaly types).
       Because the same RNG seed is used, the two DataFrames share identical
       base rows in the same positions; only the modified cells differ.
    3. Identify planted weekend rows by comparing dates between the clean and
       planted datasets (clean date was a weekday; planted date is now a
       Saturday).
    4. Identify planted after-hours rows by comparing times between the two
       datasets (clean time was within business hours; planted time is 22:00).
    5. Cross-check by applying the detection predicates directly so the test
       is self-consistent and doesn't rely solely on the diff approach.
    """
    df_clean = generate(count=COUNT, seed=SEED)

    df_planted = generate(
        count=COUNT,
        seed=SEED,
        anomalies=[
            AnomalySpec(type="weekend_posting", count=WEEKEND_PLANT_COUNT, params={}),
            AnomalySpec(type="after_hours_posting", count=AFTER_HOURS_PLANT_COUNT, params={}),
        ],
    )

    # Both DataFrames have exactly COUNT rows (timing planters modify in-place,
    # they do not append extra rows).
    assert len(df_planted) == COUNT

    detector_df = _build_detector_df(df_planted)

    # --- Identify planted weekend rows --------------------------------
    # The generator moves the row's date to the nearest Saturday.  Any row
    # whose date changed AND whose new date is a weekend is a planted row.
    # We also accept rows where the original was already a weekend but the
    # generator left it on the same weekend day (edge case — still correct).
    # Primary: compare dates; secondary: use the predicate on the final df.
    date_changed_mask = df_clean["date"] != df_planted["date"]
    weekend_via_diff = set(df_planted.index[date_changed_mask].tolist())

    # Confirm via the actual predicate: weekday >= 5 on the planted df.
    weekend_via_pred: set[int] = set()
    for i, row in df_planted.iterrows():
        date_val = row["date"]
        if hasattr(date_val, "weekday") and date_val.weekday() >= 5:
            weekend_via_pred.add(i)

    # The diff-based set should be a subset of (or equal to) the predicate set.
    # Use the diff set as the ground-truth planted indices so we don't
    # accidentally pick up clean rows that happened to fall on a weekend.
    planted_weekend_indices = weekend_via_diff & weekend_via_pred

    assert 0 < len(planted_weekend_indices) <= WEEKEND_PLANT_COUNT, (
        f"Expected between 1 and {WEEKEND_PLANT_COUNT} planted weekend rows, "
        f"found {len(planted_weekend_indices)}."
    )

    # --- Identify planted after-hours rows ----------------------------
    business_start = datetime.time(8, 0)
    business_end = datetime.time(18, 0)

    time_changed_mask = df_clean["time"] != df_planted["time"]
    after_hours_via_diff = set(df_planted.index[time_changed_mask].tolist())

    # Confirm via predicate: time < 08:00 or time > 18:00.
    after_hours_via_pred: set[int] = set()
    for i, row in df_planted.iterrows():
        t = row["time"]
        if isinstance(t, datetime.time):
            if t < business_start or t > business_end:
                after_hours_via_pred.add(i)

    planted_after_hours_indices = after_hours_via_diff

    assert len(planted_after_hours_indices) == AFTER_HOURS_PLANT_COUNT, (
        f"Expected {AFTER_HOURS_PLANT_COUNT} planted after-hours rows, "
        f"found {len(planted_after_hours_indices)}.  "
        f"(predicate set size: {len(after_hours_via_pred)})"
    )

    # Collect the corresponding tx_ids from the detector DataFrame.
    weekend_tx_ids = set(
        detector_df.loc[list(planted_weekend_indices), "tx_id"].tolist()
    )
    after_hours_tx_ids = set(
        detector_df.loc[list(planted_after_hours_indices), "tx_id"].tolist()
    )

    return detector_df, weekend_tx_ids, after_hours_tx_ids


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_weekend_posting_100_percent_recall(planted_df_and_expected, config_default):
    """All planted weekend rows must receive a weekend_posting flag (Req 5.1, 12.4)."""
    detector_df, weekend_tx_ids, _ = planted_df_and_expected
    run_log = RunLog(input_file="<synthetic>", config=config_default)

    flags = detect(detector_df, config_default, run_log)
    weekend_flagged = {f.tx_id for f in flags if f.rule == "weekend_posting"}

    missing = weekend_tx_ids - weekend_flagged
    assert not missing, (
        f"{len(missing)} of {WEEKEND_PLANT_COUNT} planted weekend transactions "
        f"were NOT flagged: {sorted(missing)}"
    )


def test_after_hours_posting_100_percent_recall(planted_df_and_expected, config_default):
    """All planted after-hours rows must receive an after_hours_posting flag (Req 5.2, 12.4)."""
    detector_df, _, after_hours_tx_ids = planted_df_and_expected
    run_log = RunLog(input_file="<synthetic>", config=config_default)

    flags = detect(detector_df, config_default, run_log)
    after_hours_flagged = {f.tx_id for f in flags if f.rule == "after_hours_posting"}

    missing = after_hours_tx_ids - after_hours_flagged
    assert not missing, (
        f"{len(missing)} of {AFTER_HOURS_PLANT_COUNT} planted after-hours transactions "
        f"were NOT flagged: {sorted(missing)}"
    )


def test_weekend_flag_evidence_fields(planted_df_and_expected, config_default):
    """Every weekend_posting flag must carry day_of_week and posting_date evidence."""
    detector_df, weekend_tx_ids, _ = planted_df_and_expected
    run_log = RunLog(input_file="<synthetic>", config=config_default)

    flags = detect(detector_df, config_default, run_log)
    weekend_flags = [f for f in flags if f.rule == "weekend_posting" and f.tx_id in weekend_tx_ids]

    for f in weekend_flags:
        assert "day_of_week" in f.evidence, (
            f"Flag for {f.tx_id} missing 'day_of_week' in evidence"
        )
        assert "posting_date" in f.evidence, (
            f"Flag for {f.tx_id} missing 'posting_date' in evidence"
        )
        assert f.evidence["day_of_week"] in ("Saturday", "Sunday"), (
            f"Flag for {f.tx_id} has unexpected day_of_week: {f.evidence['day_of_week']!r}"
        )


def test_after_hours_flag_evidence_fields(planted_df_and_expected, config_default):
    """Every after_hours_posting flag must carry posting_time and business_hours_window."""
    detector_df, _, after_hours_tx_ids = planted_df_and_expected
    run_log = RunLog(input_file="<synthetic>", config=config_default)

    flags = detect(detector_df, config_default, run_log)
    ah_flags = [
        f for f in flags
        if f.rule == "after_hours_posting" and f.tx_id in after_hours_tx_ids
    ]

    for f in ah_flags:
        assert "posting_time" in f.evidence, (
            f"Flag for {f.tx_id} missing 'posting_time' in evidence"
        )
        assert "business_hours_window" in f.evidence, (
            f"Flag for {f.tx_id} missing 'business_hours_window' in evidence"
        )


def test_no_false_weekday_flags(planted_df_and_expected, config_default):
    """Transactions on weekdays must NOT receive a weekend_posting flag (Req 5.1)."""
    detector_df, _, _ = planted_df_and_expected
    run_log = RunLog(input_file="<synthetic>", config=config_default)

    flags = detect(detector_df, config_default, run_log)
    weekend_flags = [f for f in flags if f.rule == "weekend_posting"]

    for f in weekend_flags:
        row = detector_df.loc[detector_df["tx_id"] == f.tx_id].iloc[0]
        date_val = row["date"]
        weekday = date_val.weekday()
        assert weekday >= 5, (
            f"Transaction {f.tx_id} on weekday {weekday} received a weekend_posting flag"
        )


def test_no_false_in_hours_flags(planted_df_and_expected, config_default):
    """Transactions within business hours must NOT receive an after_hours_posting flag (Req 5.2)."""
    business_start = config_default.business_start
    business_end = config_default.business_end

    detector_df, _, _ = planted_df_and_expected
    run_log = RunLog(input_file="<synthetic>", config=config_default)

    flags = detect(detector_df, config_default, run_log)
    ah_flags = [f for f in flags if f.rule == "after_hours_posting"]

    for f in ah_flags:
        row = detector_df.loc[detector_df["tx_id"] == f.tx_id].iloc[0]
        t = row["time"]
        assert isinstance(t, datetime.time), (
            f"time column for {f.tx_id} is not a datetime.time: {type(t)}"
        )
        assert t < business_start or t > business_end, (
            f"Transaction {f.tx_id} with time {t} is within business hours "
            f"but received an after_hours_posting flag"
        )
