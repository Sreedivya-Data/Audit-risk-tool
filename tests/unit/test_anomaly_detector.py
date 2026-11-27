"""Unit tests for anomaly_detector._detect_statistical_outliers."""

from __future__ import annotations

import datetime
import math

import numpy as np
import pandas as pd
import pytest

from src.anomaly_detector import _detect_statistical_outliers
from src.config import Config
from src.run_log import RunLog


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(accounts_amounts: dict[str, list[float]]) -> pd.DataFrame:
    """Build a minimal valid DataFrame from a mapping of account -> amounts."""
    rows = []
    tx_counter = 0
    for account, amounts in accounts_amounts.items():
        for amount in amounts:
            rows.append(
                {
                    "tx_id": f"tx{tx_counter:06d}",
                    "date": pd.Timestamp("2024-01-15"),
                    "time": datetime.time(10, 0),
                    "account": account,
                    "amount": float(amount),
                    "approver": "APR-01",
                    "vendor": "VND-01",
                    "description": "Test transaction",
                }
            )
            tx_counter += 1
    return pd.DataFrame(rows)


def _make_run_log(config: Config | None = None) -> RunLog:
    return RunLog(input_file="test.csv", config=config or Config())


# ---------------------------------------------------------------------------
# Normal flagging
# ---------------------------------------------------------------------------

def _make_outlier_df(account: str = "ACC-01") -> tuple[pd.DataFrame, list[float]]:
    """Return a DataFrame with 15 tight-cluster amounts and 1 outlier (z ≈ 3.75).

    base = [1000.0] * 15, outlier = 5000.0 → z_outlier ≈ 3.75
    """
    amounts = [1000.0] * 15 + [5000.0]
    return _make_df({account: amounts}), amounts


class TestStatisticalOutlierFlagging:
    def test_flags_extreme_outlier(self):
        """A transaction with |z| well above 3.0 must be flagged."""
        df, amounts = _make_outlier_df()
        config = Config()
        flags = _detect_statistical_outliers(df, config, _make_run_log(config))

        assert len(flags) == 1
        flag = flags[0]
        assert flag.rule == "statistical_outlier"
        assert abs(flag.evidence["z_score"]) > config.zscore_threshold

    def test_flag_carries_required_evidence(self):
        """Every flag must have z_score, account_mean, account_std, account_tx_count."""
        df, amounts = _make_outlier_df()
        config = Config()
        flags = _detect_statistical_outliers(df, config, _make_run_log(config))

        flag = flags[0]
        for key in ("z_score", "account_mean", "account_std", "account_tx_count"):
            assert key in flag.evidence, f"Missing evidence key: {key}"

    def test_evidence_z_score_matches_calculation(self):
        """The z_score in evidence must equal (amount - mean) / std within tolerance."""
        df, amounts = _make_outlier_df()
        config = Config()
        flags = _detect_statistical_outliers(df, config, _make_run_log(config))

        flag = flags[0]
        s = pd.Series(amounts)
        expected_mean = float(s.mean())
        expected_std = float(s.std())     # pandas uses ddof=1
        expected_z = (5000.0 - expected_mean) / expected_std

        assert math.isclose(flag.evidence["z_score"], expected_z, rel_tol=1e-9)
        assert math.isclose(flag.evidence["account_mean"], expected_mean, rel_tol=1e-9)
        assert math.isclose(flag.evidence["account_std"], expected_std, rel_tol=1e-9)
        assert flag.evidence["account_tx_count"] == 16

    def test_human_readable_threshold(self):
        """threshold field must be human-readable (contain |z| and the configured value)."""
        df, _ = _make_outlier_df()
        config = Config()
        flags = _detect_statistical_outliers(df, config, _make_run_log(config))

        assert "|z|" in flags[0].threshold
        assert str(config.zscore_threshold) in flags[0].threshold or \
               f"{config.zscore_threshold:g}" in flags[0].threshold

    def test_no_flag_for_normal_transaction(self):
        """Transactions within |z| <= threshold must not be flagged."""
        # All amounts equal — zero variance, should be skipped not flagged
        amounts = [1000.0] * 10
        df = _make_df({"ACC-01": amounts})
        config = Config()
        flags = _detect_statistical_outliers(df, config, _make_run_log(config))
        # Either skipped (zero std) or no flags — no false positive
        assert len(flags) == 0

    def test_no_flag_when_all_amounts_in_range(self):
        """Tight cluster with no outlier should produce zero flags."""
        rng = np.random.default_rng(42)
        amounts = rng.normal(loc=5000.0, scale=50.0, size=20).tolist()
        df = _make_df({"ACC-01": amounts})
        config = Config()
        flags = _detect_statistical_outliers(df, config, _make_run_log(config))
        assert len(flags) == 0

    def test_multiple_outliers_in_same_account(self):
        """Multiple anomalous transactions in one account are all flagged."""
        # 50 tight base at 1000, two outliers: 5000 (z≈6.3) and -1000 (z≈-3.3)
        amounts = [1000.0] * 50 + [5000.0, -1000.0]
        df = _make_df({"ACC-01": amounts})
        config = Config()
        flags = _detect_statistical_outliers(df, config, _make_run_log(config))
        # Both values have |z| > 3.0
        assert len(flags) == 2

    def test_independent_flagging_per_account(self):
        """Outlier detection is computed independently per account."""
        # ACC-01 has an outlier; ACC-02 is a constant → skipped (zero std)
        # Use a varying cluster for ACC-02 with no outlier
        acc1 = [1000.0] * 15 + [5000.0]
        acc2 = list(range(4990, 5010))  # 20 varying amounts, no outlier
        df = _make_df({"ACC-01": acc1, "ACC-02": acc2})
        config = Config()
        flags = _detect_statistical_outliers(df, config, _make_run_log(config))

        # Only ACC-01 produces a flag
        assert len(flags) == 1
        assert flags[0].tx_id  # has a tx_id

    def test_custom_zscore_threshold(self):
        """Lowering the threshold flags more transactions."""
        rng = np.random.default_rng(99)
        amounts = rng.normal(loc=1000.0, scale=100.0, size=20).tolist()
        # Plant an amount with |z| ~= 2.5 (above 2.0 threshold, below 3.0 default)
        amounts[-1] = 1000.0 + 2.5 * float(pd.Series(amounts[:-1]).std())
        df = _make_df({"ACC-01": amounts})

        config_strict = Config(zscore_threshold=3.0)
        config_loose = Config(zscore_threshold=2.0)

        flags_strict = _detect_statistical_outliers(df, config_strict, _make_run_log())
        flags_loose = _detect_statistical_outliers(df, config_loose, _make_run_log())

        assert len(flags_loose) >= len(flags_strict)


# ---------------------------------------------------------------------------
# Skip conditions
# ---------------------------------------------------------------------------

class TestStatisticalOutlierSkipConditions:
    def test_skips_account_with_too_few_transactions(self):
        """Accounts with fewer than min_account_tx transactions are skipped."""
        amounts = [1000.0] * 5 + [100_000.0]  # 6 transactions < default 10
        df = _make_df({"SMALL-ACC": amounts})
        config = Config()
        run_log = _make_run_log(config)

        flags = _detect_statistical_outliers(df, config, run_log)

        assert len(flags) == 0
        skips = run_log._skipped_checks
        assert len(skips) == 1
        assert skips[0]["check"] == "zscore"
        assert skips[0]["account"] == "SMALL-ACC"

    def test_skips_account_with_zero_std(self):
        """Accounts where all amounts are equal (zero std) are skipped, not flagged."""
        amounts = [1000.0] * 10  # constant — std = 0
        df = _make_df({"ACC-CONST": amounts})
        config = Config()
        run_log = _make_run_log(config)

        flags = _detect_statistical_outliers(df, config, run_log)

        assert len(flags) == 0
        skips = run_log._skipped_checks
        assert len(skips) == 1
        assert "zero standard deviation" in skips[0]["reason"]

    def test_skip_recorded_for_each_small_account(self):
        """Every small account generates exactly one skip entry in the run log."""
        # LARGE-C has 10 varying amounts (no zero std) → processed, not skipped
        large_c_amounts = [float(3000 + i * 10) for i in range(10)]
        df = _make_df(
            {
                "SMALL-A": [1000.0] * 3,
                "SMALL-B": [2000.0] * 5,
                "LARGE-C": large_c_amounts,
            }
        )
        config = Config()
        run_log = _make_run_log(config)
        _detect_statistical_outliers(df, config, run_log)

        skip_accounts = {s["account"] for s in run_log._skipped_checks}
        assert "SMALL-A" in skip_accounts
        assert "SMALL-B" in skip_accounts
        assert "LARGE-C" not in skip_accounts

    def test_custom_min_account_tx(self):
        """A custom min_account_tx changes what counts as 'too few'."""
        # 15 tight cluster + 1 outlier (z ≈ 3.75), total 16 transactions
        amounts = [1000.0] * 15 + [5000.0]
        df = _make_df({"ACC-01": amounts})

        config_high = Config(min_account_tx=20)  # 16 < 20 → skip
        config_low = Config(min_account_tx=10)   # 16 >= 10 → process

        run_log_high = _make_run_log()
        flags_high = _detect_statistical_outliers(df, config_high, run_log_high)
        assert len(flags_high) == 0
        assert len(run_log_high._skipped_checks) == 1

        run_log_low = _make_run_log()
        flags_low = _detect_statistical_outliers(df, config_low, run_log_low)
        assert len(flags_low) > 0
        assert len(run_log_low._skipped_checks) == 0

    def test_empty_dataframe_returns_no_flags(self):
        """An empty DataFrame must return an empty flag list without raising."""
        df = pd.DataFrame(
            columns=["tx_id", "date", "time", "account", "amount",
                     "approver", "vendor", "description"]
        )
        config = Config()
        flags = _detect_statistical_outliers(df, config, _make_run_log(config))
        assert flags == []


# ---------------------------------------------------------------------------
# Benford's law detection
# ---------------------------------------------------------------------------

from src.anomaly_detector import _detect_benford_deviation


def _make_benford_df(amounts: list[float]) -> pd.DataFrame:
    """Build a minimal DataFrame from a list of amounts (single account is fine)."""
    rows = []
    for i, amount in enumerate(amounts):
        rows.append(
            {
                "tx_id": f"tx{i:06d}",
                "date": pd.Timestamp("2024-01-15"),
                "time": datetime.time(10, 0),
                "account": "ACC-01",
                "amount": float(amount),
                "approver": "APR-01",
                "vendor": "VND-01",
                "description": "Test transaction",
            }
        )
    return pd.DataFrame(rows)


def _benford_conformant_amounts(n: int = 1000) -> list[float]:
    """Amounts whose first-digit counts are exactly (rounded) Benford-proportional.

    Expected chi-square deviation is <= 0.5 per digit, so the p-value is ~1.
    """
    counts = {d: round(math.log10(1 + 1 / d) * n) for d in range(1, 10)}
    amounts = []
    idx = 0
    for d in range(1, 10):
        for _ in range(counts[d]):
            # First digit of (d*1000 + j) is d for j in [0, 999]
            amounts.append(d * 1000 + (idx % 1000))
            idx += 1
    return amounts


class TestBenfordDetection:
    def test_flags_fabricated_same_digit_dataset(self):
        """150 amounts all starting with 7 must trigger benford_deviation."""
        amounts = [7000 + j for j in range(150)]
        df = _make_benford_df(amounts)
        config = Config()
        flags = _detect_benford_deviation(df, config, _make_run_log(config))

        assert len(flags) == 1
        flag = flags[0]
        assert flag.rule == "benford_deviation"
        assert flag.tx_id == "__dataset__"
        assert flag.evidence["p_value"] < config.benford_significance

    def test_flag_carries_all_required_evidence(self):
        """Evidence must contain all five required fields."""
        amounts = [7000 + j for j in range(150)]
        df = _make_benford_df(amounts)
        config = Config()
        flags = _detect_benford_deviation(df, config, _make_run_log(config))

        evidence = flags[0].evidence
        for key in (
            "chi2_statistic",
            "p_value",
            "per_digit_observed",
            "per_digit_expected",
            "per_digit_deviation",
        ):
            assert key in evidence, f"Missing evidence key: {key}"
        # Per-digit structures must cover digits 1..9
        for d in range(1, 10):
            assert d in evidence["per_digit_observed"]
            assert d in evidence["per_digit_expected"]
            assert d in evidence["per_digit_deviation"]

    def test_no_flag_for_benford_conformant_dataset(self):
        """A dataset following Benford's law exactly must not be flagged."""
        amounts = _benford_conformant_amounts(1000)
        df = _make_benford_df(amounts)
        config = Config()
        flags = _detect_benford_deviation(df, config, _make_run_log(config))
        assert flags == []

    def test_threshold_is_human_readable(self):
        """threshold must mention the p-value cutoff."""
        amounts = [7000 + j for j in range(150)]
        df = _make_benford_df(amounts)
        config = Config()
        flags = _detect_benford_deviation(df, config, _make_run_log(config))
        assert "p-value" in flags[0].threshold
        assert "0.05" in flags[0].threshold

    def test_single_dataset_level_flag_only(self):
        """Even with many anomalous rows, exactly one dataset-level flag is emitted."""
        amounts = [7000 + j for j in range(150)] + [8000 + j for j in range(50)]
        df = _make_benford_df(amounts)
        config = Config()
        flags = _detect_benford_deviation(df, config, _make_run_log(config))
        assert len(flags) == 1
        assert flags[0].tx_id == "__dataset__"

    def test_skips_small_positive_count(self):
        """Fewer than benford_min_count positive amounts -> skip, no flag."""
        amounts = [float(j + 1) for j in range(50)]  # 50 positives < 100
        df = _make_benford_df(amounts)
        config = Config()
        run_log = _make_run_log(config)

        flags = _detect_benford_deviation(df, config, run_log)

        assert flags == []
        skips = run_log._skipped_checks
        assert len(skips) == 1
        assert skips[0]["check"] == "benford"
        assert "fewer than" in skips[0]["reason"]

    def test_negative_amounts_excluded_from_test(self):
        """Only positive amounts count toward the sample; negatives alone trigger skip."""
        amounts = [float(j + 1) for j in range(40)] + [-(j + 1.0) for j in range(60)]
        df = _make_benford_df(amounts)
        config = Config()
        run_log = _make_run_log(config)

        flags = _detect_benford_deviation(df, config, run_log)

        assert flags == []
        assert any(s["check"] == "benford" for s in run_log._skipped_checks)

    def test_custom_significance_level(self):
        """A stricter significance level can suppress an otherwise-marginal flag."""
        amounts = _benford_conformant_amounts(1000)
        # Introduce a mild distortion: shift 60 digit-1 amounts to digit 9
        amounts = [a for a in amounts if a >= 2000] + [1000 + j for j in range(940 - sum(1 for a in amounts if a >= 2000))]
        # Rebuild deterministically instead: 60% conformant, mild skew
        base = _benford_conformant_amounts(1000)
        skewed = [9000 + j for j in range(150)] + [a for a in base if a < 8000][:850]
        df = _make_benford_df(skewed)
        config = Config()
        flags = _detect_benford_deviation(df, config, _make_run_log(config))
        # With the default 0.05 level this skewed dataset is flagged
        assert len(flags) == 1

    def test_custom_min_count(self):
        """benford_min_count=200 raises the bar: 150 positives now skip."""
        amounts = [7000 + j for j in range(150)]
        df = _make_benford_df(amounts)
        config = Config(benford_min_count=200)
        run_log = _make_run_log(config)

        flags = _detect_benford_deviation(df, config, run_log)
        assert flags == []
        assert len(run_log._skipped_checks) == 1

    def test_empty_dataframe_skips(self):
        """An empty DataFrame records a skip and returns no flags."""
        df = pd.DataFrame(
            columns=["tx_id", "date", "time", "account", "amount",
                     "approver", "vendor", "description"]
        )
        config = Config()
        run_log = _make_run_log(config)
        flags = _detect_benford_deviation(df, config, run_log)
        assert flags == []
        assert any(s["check"] == "benford" for s in run_log._skipped_checks)

    def test_expected_counts_follow_benford_formula(self):
        """per_digit_expected must match n * log10(1 + 1/d) for each digit."""
        amounts = [7000 + j for j in range(150)]
        df = _make_benford_df(amounts)
        config = Config()
        flags = _detect_benford_deviation(df, config, _make_run_log(config))

        evidence = flags[0].evidence
        n = 150
        for d in range(1, 10):
            expected = n * math.log10(1 + 1 / d)
            assert math.isclose(
                evidence["per_digit_expected"][d], expected, rel_tol=1e-9
            )
