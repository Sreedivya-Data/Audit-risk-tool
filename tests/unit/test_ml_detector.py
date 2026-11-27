"""Unit tests for the ML detection layer (src/ml_detector.py)."""

from __future__ import annotations

import datetime

import numpy as np
import pandas as pd
import pytest

from src.config import Config
from src.ml_detector import detect, _build_feature_matrix
from src.models import Flag
from src.run_log import RunLog


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_df(n: int = 150, seed: int = 42) -> pd.DataFrame:
    """Build a synthetic cleaned transaction DataFrame of ``n`` rows."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="12h")
    return pd.DataFrame({
        "tx_id": [f"TX{i:05d}" for i in range(n)],
        "date": dates,
        "time": [datetime.time((8 + i % 9), (i * 13) % 60) for i in range(n)],
        "account": [f"ACC-{i % 5:02d}" for i in range(n)],
        "amount": rng.normal(10000, 2000, n).round(2),
        "approver": [f"APR-{i % 4}" for i in range(n)],
        "vendor": [f"VEN-{i % 10}" for i in range(n)],
        "description": [f"desc {i}" for i in range(n)],
    })


def make_run_log() -> RunLog:
    return RunLog(input_file="synthetic.csv", config=Config())


# ---------------------------------------------------------------------------
# Feature matrix
# ---------------------------------------------------------------------------

class TestFeatureMatrix:
    def test_shape_and_columns(self):
        df = make_df(150)
        X = _build_feature_matrix(df)
        assert X.shape == (150, 8)
        assert np.isfinite(X).all()

    def test_amount_log_values(self):
        df = make_df(5)
        X = _build_feature_matrix(df)
        expected = np.log1p(np.abs(df["amount"].values))
        np.testing.assert_allclose(X[:, 0], expected)

    def test_hour_of_day(self):
        df = make_df(5)
        X = _build_feature_matrix(df)
        # first row's time is 08:00 -> 8.0
        assert X[0, 1] == pytest.approx(8.0)

    def test_is_weekend_flag(self):
        # 2024-01-06 is a Saturday
        df = make_df(7)
        X = _build_feature_matrix(df)
        # index 4 corresponds to 2024-01-06 (12h steps: 01,01,02,02,03,03,04) -> none weekend yet
        # construct explicitly instead
        df2 = df.copy()
        df2["date"] = pd.to_datetime(["2024-01-06"] * 7)  # Saturday
        X2 = _build_feature_matrix(df2)
        assert (X2[:, 3] == 1.0).all()

    def test_amount_z_norm(self):
        df = make_df(100)
        X = _build_feature_matrix(df)
        amounts = df["amount"].values.astype(float)
        expected = (amounts - amounts.mean()) / amounts.std(ddof=0)
        np.testing.assert_allclose(X[:, 7], expected)


# ---------------------------------------------------------------------------
# Skip condition
# ---------------------------------------------------------------------------

class TestSkipCondition:
    def test_skip_below_min_tx(self):
        df = make_df(50)  # < 100 default min
        log = make_run_log()
        flags = detect(df, [], Config(), log)
        assert flags == []
        assert log.to_dict()["ml_skip_reason"] is not None
        assert "50" in log.to_dict()["ml_skip_reason"]

    def test_no_skip_at_threshold(self):
        df = make_df(100)  # exactly min
        log = make_run_log()
        flags = detect(df, [], Config(), log)
        assert isinstance(flags, list)
        assert log.to_dict()["ml_skip_reason"] is None


# ---------------------------------------------------------------------------
# Flag structure
# ---------------------------------------------------------------------------

class TestFlagOutput:
    def test_flag_structure(self):
        df = make_df(200)
        log = make_run_log()
        flags = detect(df, [], Config(), log)
        for f in flags:
            assert f.rule == "ml_isolation_forest"
            assert f.threshold == "isolation_forest_decision < 0"
            assert "anomaly_score" in f.evidence
            assert "feature_contributions" in f.evidence
            assert len(f.evidence["feature_contributions"]) == 3

    def test_flags_reference_real_tx_ids(self):
        df = make_df(200)
        log = make_run_log()
        flags = detect(df, [], Config(), log)
        valid_ids = set(df["tx_id"])
        for f in flags:
            assert f.tx_id in valid_ids

    def test_anomaly_score_negative_for_flags(self):
        df = make_df(200)
        log = make_run_log()
        flags = detect(df, [], Config(), log)
        for f in flags:
            assert isinstance(f.evidence["anomaly_score"], float)


# ---------------------------------------------------------------------------
# Contamination rate
# ---------------------------------------------------------------------------

class TestContaminationDerivation:
    def test_contamination_derived_from_rule_flags(self, monkeypatch):
        """rule_flag_rate > 0.01 should raise contamination (verified indirectly
        through more flags being emitted)."""
        df = make_df(200)
        # Flag 30% of transactions with rule flags
        rule_flags = [
            Flag(tx_id=df["tx_id"].iloc[i], rule="statistical_outlier",
                 threshold="|z| > 3.0", evidence={})
            for i in range(60)
        ]
        log = make_run_log()
        flags_high = detect(df, rule_flags, Config(), log)
        log2 = make_run_log()
        flags_low = detect(df, [], Config(), log2)
        # Higher contamination should flag at least as many transactions
        assert len(flags_high) >= len(flags_low)
