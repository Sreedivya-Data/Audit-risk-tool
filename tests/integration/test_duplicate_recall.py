"""Integration test: duplicate-payment detection recall on planted anomalies.

Feature: ai-audit-risk-tool, Task 11.3
Validates: Requirements 4.4, 12.4

Approach:
- Generate a base (clean) dataset and, with the same seed, a dataset with
  10 planted duplicate pairs. The ``duplicate_payment`` planter appends a
  copy of 10 chosen rows with a posting date 1-3 days after the original.
  Because every generated row carries a unique description, each appended
  copy can be matched back to its original by description.
- Run the rule-based anomaly detector over the planted dataset and assert
  that BOTH members of every planted pair receive a ``duplicate_payment``
  flag.
"""

from __future__ import annotations

import pytest

import pandas as pd

from src.anomaly_detector import detect
from src.config import Config
from src.data_generator import AnomalySpec, generate
from src.run_log import RunLog

COUNT = 500
SEED = 42
N_PAIRS = 10


@pytest.fixture
def planted_and_pairs():
    """Return (planted_df, pairs) where pairs is a list of (orig_idx, copy_idx)."""
    clean = generate(COUNT, SEED)
    planted = generate(
        COUNT, SEED, anomalies=[AnomalySpec(type="duplicate_payment", count=N_PAIRS)]
    )
    assert len(planted) == COUNT + N_PAIRS

    # The planter appends the copies at the end of the DataFrame, so the
    # last N_PAIRS rows are the copies; the originals live in the first
    # COUNT rows and are found by their unique description.
    desc_to_idx = {
        clean.at[i, "description"]: i for i in range(len(clean))
    }

    pairs: list[tuple[int, int]] = []
    for copy_idx in range(COUNT, len(planted)):
        copy = planted.loc[copy_idx]
        original_idx = desc_to_idx[copy["description"]]
        # Sanity: original and copy share everything except the date, and
        # the copy is posted 1-3 days later.
        assert copy["vendor"] == planted.at[original_idx, "vendor"]
        assert copy["amount"] == planted.at[original_idx, "amount"]
        gap = (copy["date"] - planted.at[original_idx, "date"]).days
        assert 1 <= gap <= 3
        pairs.append((original_idx, copy_idx))

    # Assign deterministic tx_ids equal to the row positions and coerce the
    # date column to datetime64, mirroring what Transaction_Loader produces.
    planted = planted.copy()
    planted.insert(0, "tx_id", [f"tx{i:04d}" for i in range(len(planted))])
    planted["date"] = pd.to_datetime(planted["date"])
    return planted, pairs


def test_planted_duplicate_pairs_both_flagged(planted_and_pairs, config_default):
    planted, pairs = planted_and_pairs
    run_log = RunLog(input_file="synthetic", config=config_default)

    flags = detect(planted, config_default, run_log)
    dup_flagged = {f.tx_id for f in flags if f.rule == "duplicate_payment"}

    for original_idx, copy_idx in pairs:
        orig_id = planted.at[original_idx, "tx_id"]
        copy_id = planted.at[copy_idx, "tx_id"]
        assert orig_id in dup_flagged, (
            f"Original row {orig_id} of planted pair "
            f"(copy {copy_id}) was not flagged as duplicate_payment"
        )
        assert copy_id in dup_flagged, (
            f"Copy row {copy_id} of planted pair "
            f"(original {orig_id}) was not flagged as duplicate_payment"
        )


def test_planted_pair_flags_carry_pair_evidence(planted_and_pairs, config_default):
    """Each planted-pair flag's evidence points at the other pair member (Req 4.4)."""
    planted, pairs = planted_and_pairs
    run_log = RunLog(input_file="synthetic", config=config_default)

    flags = [
        f for f in detect(planted, config_default, run_log)
        if f.rule == "duplicate_payment"
    ]

    pair_ids = set()
    for original_idx, copy_idx in pairs:
        pair_ids.add(planted.at[original_idx, "tx_id"])
        pair_ids.add(planted.at[copy_idx, "tx_id"])

    pair_flags = [f for f in flags if f.tx_id in pair_ids]
    # At minimum, every planted member has a flag.
    assert pair_ids <= {f.tx_id for f in pair_flags}

    for f in pair_flags:
        assert f.evidence.get("paired_tx_id") in pair_ids
        assert 0 <= f.evidence.get("date_gap_days", -1) <= 7
        assert f.evidence.get("group_size", 0) >= 2
