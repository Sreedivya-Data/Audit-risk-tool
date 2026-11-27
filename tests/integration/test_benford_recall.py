"""Integration test: Benford's law deviation recall across multiple seeds.

Plants 200 fabricated amounts whose first digit is 9 (a severe skew away
from Benford's expected distribution) in a 1,000-transaction base dataset
and verifies that ``anomaly_detector.detect`` raises the ``benford_deviation``
flag in at least 90 % of 10 seeded runs (≥ 9 of 10).

Validates: Requirements 3.4, 12.4
"""

from __future__ import annotations

import pytest

from src.anomaly_detector import BENFORD_DEVIATION_RULE, detect
from src.config import Config
from src.data_generator import AnomalySpec, generate
from src.run_log import RunLog

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_COUNT = 1_000           # well above benford_min_count=100
BENFORD_ANOMALY_COUNT = 200  # plant 200 nine-leading-digit amounts
NUM_SEEDS = 10               # run with seeds 0–9
MIN_DETECTION_RUNS = 9       # require ≥ 90 % detection rate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_df_with_tx_id(seed: int) -> "pd.DataFrame":
    """Generate a 1,000-row dataset with 200 Benford violations and add tx_id."""
    import hashlib
    import pandas as pd

    df = generate(
        count=BASE_COUNT,
        seed=seed,
        anomalies=[AnomalySpec(type="benford_violation", count=BENFORD_ANOMALY_COUNT)],
    )

    # anomaly_detector.detect() expects a tx_id column; assign stable hashes.
    ids = []
    for _, row in df.iterrows():
        parts = [str(v) for v in row.tolist()]
        raw = "|".join(parts)
        ids.append(hashlib.sha256(raw.encode()).hexdigest()[:12])
    df["tx_id"] = ids
    return df


def _run_produces_benford_flag(seed: int, config: Config) -> bool:
    """Return True if the detector emits a benford_deviation flag for this seed."""
    df = _build_df_with_tx_id(seed)
    run_log = RunLog(input_file="<synthetic>", config=config)
    flags = detect(df, config, run_log)
    return any(f.rule == BENFORD_DEVIATION_RULE for f in flags)


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

def test_benford_recall(config_default: Config) -> None:
    """benford_deviation must be triggered in ≥ 90 % of 10 seeded runs.

    For each seed in 0–9, a fresh 1,000-row synthetic dataset is produced
    with 200 amounts whose first digit is 9.  This heavy over-representation
    of digit 9 (Benford expects ~4.6 % but we plant 20 %) should reliably
    shift the chi-square p-value below 0.05.

    Validates: Requirements 3.4, 12.4
    """
    detection_count = sum(
        _run_produces_benford_flag(seed, config_default)
        for seed in range(NUM_SEEDS)
    )

    assert detection_count >= MIN_DETECTION_RUNS, (
        f"Benford recall too low: detected in only {detection_count}/{NUM_SEEDS} runs "
        f"(need ≥ {MIN_DETECTION_RUNS}).  "
        f"Check that 200 nine-leading-digit anomalies skew the chi-square p-value "
        f"below config.benford_significance={config_default.benford_significance}."
    )
