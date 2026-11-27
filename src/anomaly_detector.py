"""Rule-based anomaly detection for the AI Audit and Risk Analytics Tool.

This module hosts individual rule implementations. The public
``detect`` entry point (aggregating all rules) is defined in later tasks;
for now each rule is exposed as a private helper invoked by it.
"""

from __future__ import annotations

import datetime
import math

import numpy as np
import pandas as pd
from scipy import stats

from src.config import Config
from src.models import Flag
from src.run_log import RunLog

STATISTICAL_OUTLIER_RULE = "statistical_outlier"
BENFORD_DEVIATION_RULE = "benford_deviation"
BENFORD_DATASET_TX_ID = "__dataset__"


def _detect_statistical_outliers(
    df: pd.DataFrame,
    config: Config,
    run_log: RunLog,
) -> list[Flag]:
    """Flag transactions whose amount is unusual relative to their account history.

    For each account with at least ``config.min_account_tx`` transactions and a
    non-zero amount standard deviation, the z-score
    ``(amount - account_mean) / account_std`` is computed. Transactions with
    ``abs(z_score) > config.zscore_threshold`` receive a
    ``statistical_outlier`` flag carrying the z-score, account mean, account
    standard deviation, and account transaction count as evidence.

    Accounts with too few transactions, or with zero standard deviation,
    are skipped and the skip is recorded in the run log.
    """
    flags: list[Flag] = []
    threshold_label = f"|z| > {config.zscore_threshold:g}"

    if df.empty:
        return flags

    stats = df.groupby("account")["amount"].agg(["mean", "std", "count"])

    for account, row in stats.iterrows():
        count = int(row["count"])

        if count < config.min_account_tx:
            run_log.log_skip(
                check="zscore",
                account=account,
                reason=f"fewer than {config.min_account_tx} transactions ({count})",
            )
            continue

        std = row["std"]
        if std is None or math.isnan(std) or std == 0:
            run_log.log_skip(
                check="zscore",
                account=account,
                reason="zero standard deviation (constant amounts)",
            )
            continue

        mean = float(row["mean"])
        std_f = float(std)

        account_df = df[df["account"] == account]
        z_scores = (account_df["amount"] - mean) / std_f
        mask = z_scores.abs() > config.zscore_threshold

        for idx in account_df.index[mask]:
            flags.append(
                Flag(
                    tx_id=str(df.at[idx, "tx_id"]),
                    rule=STATISTICAL_OUTLIER_RULE,
                    threshold=threshold_label,
                    evidence={
                        "z_score": float(z_scores.loc[idx]),
                        "account_mean": mean,
                        "account_std": std_f,
                        "account_tx_count": count,
                    },
                )
            )

    return flags


DUPLICATE_PAYMENT_RULE = "duplicate_payment"


def _detect_duplicate_payments(
    df: pd.DataFrame,
    config: Config,
    run_log: RunLog,
) -> list[Flag]:
    """Flag pairs of transactions that share the same vendor and amount posted within
    ``config.duplicate_window_days`` calendar days of each other.

    Two flags are emitted per matched pair — one for each member — with evidence fields:
    - ``paired_tx_id``: the ``tx_id`` of the other transaction in the pair.
    - ``date_gap_days``: the difference in posting dates (in days, as a float).
    - ``group_size``: the total number of transactions in the ``(vendor, amount)``
      window group that this transaction belongs to.

    When a ``(vendor, amount)`` window group has more than 50 members, every member is
    still flagged and the ``group_size`` evidence field records the actual count.

    Accounts-level skips are not applicable here; this check always runs if the
    DataFrame is non-empty.
    """
    flags: list[Flag] = []
    threshold_label = (
        f"same vendor+amount within {config.duplicate_window_days} days"
    )

    if df.empty:
        return flags

    # Work on a lightweight copy of only the columns we need.
    work = df[["tx_id", "vendor", "amount", "date"]].copy()

    max_gap = pd.Timedelta(days=config.duplicate_window_days)

    # Iterate over (vendor, amount) groups.  Within each group we cross-join the
    # group with itself and retain only pairs where:
    #   • the two transactions are distinct (different tx_id)
    #   • the absolute posting-date gap is at most max_gap
    # Deduplication via seen_pairs ensures each unordered pair emits exactly two
    # flags (one per member), regardless of how many rows the cross-join produces.
    seen_pairs: set[frozenset[str]] = set()

    for (vendor, amount), group in work.groupby(["vendor", "amount"], sort=False):
        if len(group) < 2:
            # Need at least two transactions to form a pair.
            continue

        # Cross-join the group with itself.
        left = group.rename(columns={"tx_id": "tx_id_a", "date": "date_a"})
        right = group.rename(columns={"tx_id": "tx_id_b", "date": "date_b"})
        cross = left[["tx_id_a", "date_a"]].merge(
            right[["tx_id_b", "date_b"]], how="cross"
        )

        # Keep only pairs where a != b and date gap is within the window.
        cross = cross[cross["tx_id_a"] != cross["tx_id_b"]].copy()
        cross["gap"] = (cross["date_b"] - cross["date_a"]).abs()
        cross = cross[cross["gap"] <= max_gap]

        if cross.empty:
            continue

        # Collect all tx_ids involved in at least one qualifying pair for group_size.
        involved_tx_ids: set[str] = (
            set(cross["tx_id_a"]) | set(cross["tx_id_b"])
        )
        group_size = len(involved_tx_ids)

        for _, row in cross.iterrows():
            tx_a = str(row["tx_id_a"])
            tx_b = str(row["tx_id_b"])
            pair_key = frozenset([tx_a, tx_b])

            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            gap_days = float(row["gap"] / pd.Timedelta(days=1))

            # Flag the first member of the pair (tx_a).
            flags.append(
                Flag(
                    tx_id=tx_a,
                    rule=DUPLICATE_PAYMENT_RULE,
                    threshold=threshold_label,
                    evidence={
                        "paired_tx_id": tx_b,
                        "date_gap_days": gap_days,
                        "group_size": group_size,
                    },
                )
            )

            # Flag the second member of the pair (tx_b).
            flags.append(
                Flag(
                    tx_id=tx_b,
                    rule=DUPLICATE_PAYMENT_RULE,
                    threshold=threshold_label,
                    evidence={
                        "paired_tx_id": tx_a,
                        "date_gap_days": gap_days,
                        "group_size": group_size,
                    },
                )
            )

    return flags


def _detect_benford_deviation(
    df: pd.DataFrame,
    config: Config,
    run_log: RunLog,
) -> list[Flag]:
    """Test whether the first-digit distribution of positive amounts follows Benford's law.

    Computed at the dataset level (not per account) to ensure a sufficient sample
    size for the chi-square test.  If the number of positive amounts is below
    ``config.benford_min_count``, the check is skipped and recorded in the run log.

    When the chi-square p-value is below ``config.benford_significance`` (default 0.05),
    a **single** dataset-level flag is emitted with ``tx_id="__dataset__"`` and
    evidence fields:

    - ``chi2_statistic``: the chi-square test statistic.
    - ``p_value``: the p-value from the chi-square test.
    - ``per_digit_observed``: dict mapping digit (1–9) to observed count.
    - ``per_digit_expected``: dict mapping digit (1–9) to expected count.
    - ``per_digit_deviation``: dict mapping digit (1–9) to (observed - expected).
    """
    flags: list[Flag] = []
    threshold_label = f"chi2 p-value < {config.benford_significance:g}"

    # Keep only positive amounts (negatives / zero don't have a meaningful first digit
    # in the Benford sense and are conventionally excluded).
    if df.empty:
        run_log.log_skip(
            check="benford",
            reason="empty DataFrame — no positive amounts to test",
        )
        return flags

    positive_amounts = df.loc[df["amount"] > 0, "amount"]
    n_positive = len(positive_amounts)

    if n_positive < config.benford_min_count:
        run_log.log_skip(
            check="benford",
            reason=(
                f"fewer than {config.benford_min_count} positive amounts "
                f"({n_positive})"
            ),
        )
        return flags

    # --- Compute observed first-digit counts for digits 1–9 ---
    def _first_digit(amount: float) -> int:
        """Return the leading significant digit (1–9) of a positive amount."""
        # Use the string representation to avoid floating-point artefacts such as
        # scientific notation; strip a leading '-' just in case.
        s = f"{abs(amount):.10g}".lstrip("-").lstrip("0").lstrip(".")
        for ch in s:
            if ch.isdigit() and ch != "0":
                return int(ch)
        # Fallback: should not happen for positive amounts, but guard anyway.
        return 1

    first_digits = positive_amounts.apply(_first_digit)

    # Count occurrences for digits 1–9
    observed_counts = {d: int((first_digits == d).sum()) for d in range(1, 10)}

    # --- Benford expected frequencies: P(d) = log10(1 + 1/d) ---
    expected_props = {d: math.log10(1.0 + 1.0 / d) for d in range(1, 10)}
    # Scale to the actual sample size so scipy gets counts, not proportions.
    expected_counts = {d: expected_props[d] * n_positive for d in range(1, 10)}

    # --- Chi-square test ---
    obs_array = [observed_counts[d] for d in range(1, 10)]
    exp_array = [expected_counts[d] for d in range(1, 10)]

    chi2_stat, p_value = stats.chisquare(f_obs=obs_array, f_exp=exp_array)
    chi2_stat = float(chi2_stat)
    p_value = float(p_value)

    if p_value < config.benford_significance:
        per_digit_deviation = {
            d: observed_counts[d] - expected_counts[d] for d in range(1, 10)
        }
        flags.append(
            Flag(
                tx_id=BENFORD_DATASET_TX_ID,
                rule=BENFORD_DEVIATION_RULE,
                threshold=threshold_label,
                evidence={
                    "chi2_statistic": chi2_stat,
                    "p_value": p_value,
                    "per_digit_observed": observed_counts,
                    "per_digit_expected": expected_counts,
                    "per_digit_deviation": per_digit_deviation,
                },
            )
        )

    return flags


WEEKEND_POSTING_RULE = "weekend_posting"
AFTER_HOURS_POSTING_RULE = "after_hours_posting"

_DAY_NAMES = {
    0: "Monday",
    1: "Tuesday",
    2: "Wednesday",
    3: "Thursday",
    4: "Friday",
    5: "Saturday",
    6: "Sunday",
}


def _detect_weekend_postings(
    df: pd.DataFrame,
    config: Config,
    run_log: RunLog,
) -> list[Flag]:
    """Flag transactions posted on a Saturday or Sunday.

    For each row whose ``date.weekday()`` is 5 (Saturday) or 6 (Sunday) a
    ``weekend_posting`` flag is emitted with evidence:

    - ``day_of_week``: the full day name, e.g. ``"Saturday"``.
    - ``posting_date``: the date as an ISO-8601 string, e.g. ``"2024-03-02"``.

    Threshold label: ``"weekday >= 5 (Saturday or Sunday)"``.
    """
    flags: list[Flag] = []
    threshold_label = "weekday >= 5 (Saturday or Sunday)"

    if df.empty:
        return flags

    for idx, row in df.iterrows():
        date_val = row["date"]
        # date column may be datetime64 (Timestamp) or a plain date object.
        if hasattr(date_val, "weekday"):
            weekday = date_val.weekday()
        else:
            # Fallback: shouldn't normally happen with a cleaned DataFrame.
            continue

        if weekday >= 5:
            flags.append(
                Flag(
                    tx_id=str(row["tx_id"]),
                    rule=WEEKEND_POSTING_RULE,
                    threshold=threshold_label,
                    evidence={
                        "day_of_week": _DAY_NAMES.get(weekday, str(weekday)),
                        "posting_date": (
                            date_val.date().isoformat()
                            if hasattr(date_val, "date")
                            else str(date_val)
                        ),
                    },
                )
            )

    return flags


def _detect_after_hours_postings(
    df: pd.DataFrame,
    config: Config,
    run_log: RunLog,
) -> list[Flag]:
    """Flag transactions posted outside the configured business-hours window.

    A transaction is after-hours when its ``time`` value is strictly before
    ``config.business_start`` OR strictly after ``config.business_end``.

    Evidence fields per flag:

    - ``posting_time``: the time formatted as ``"HH:MM"``.
    - ``business_hours_window``: the configured window, e.g. ``"08:00-18:00"``.

    Threshold label: ``"outside HH:MM-HH:MM"`` using the configured window.
    """
    flags: list[Flag] = []

    start = config.business_start
    end = config.business_end
    window_str = f"{start.strftime('%H:%M')}-{end.strftime('%H:%M')}"
    threshold_label = f"outside {window_str}"

    if df.empty:
        return flags

    for idx, row in df.iterrows():
        time_val = row["time"]

        # time column holds Python datetime.time objects after cleaning.
        if not isinstance(time_val, datetime.time):
            # If NaT / None / unexpected type, skip gracefully.
            continue

        if time_val < start or time_val > end:
            flags.append(
                Flag(
                    tx_id=str(row["tx_id"]),
                    rule=AFTER_HOURS_POSTING_RULE,
                    threshold=threshold_label,
                    evidence={
                        "posting_time": time_val.strftime("%H:%M"),
                        "business_hours_window": window_str,
                    },
                )
            )

    return flags


ROUND_AMOUNT_RULE = "round_amount"


def _detect_round_amounts(
    df: pd.DataFrame,
    config: Config,
    run_log: RunLog,
) -> list[Flag]:
    """Flag transactions whose amount is an exact multiple of ``config.round_threshold``.

    Only positive amounts are considered — credits (negative) and zero are skipped.
    Floating-point comparison uses a tolerance of 1e-9 to guard against precision
    artefacts: ``abs(amount % threshold) < 1e-9``.

    Evidence fields per flag:

    - ``amount``: the transaction amount.
    - ``threshold``: the configured round-number threshold.

    Threshold label: ``"amount % <threshold> == 0"`` (e.g. ``"amount % 1000.0 == 0"``).
    """
    flags: list[Flag] = []
    threshold = config.round_threshold
    threshold_label = f"amount % {threshold:g} == 0"

    if df.empty:
        return flags

    for idx, row in df.iterrows():
        amount = float(row["amount"])

        # Only flag positive amounts.
        if amount <= 0:
            continue

        remainder = abs(amount % threshold)
        # Use tolerance to handle floating-point precision issues.
        if remainder < 1e-9 or abs(remainder - threshold) < 1e-9:
            flags.append(
                Flag(
                    tx_id=str(row["tx_id"]),
                    rule=ROUND_AMOUNT_RULE,
                    threshold=threshold_label,
                    evidence={
                        "amount": amount,
                        "threshold": threshold,
                    },
                )
            )

    return flags


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def detect(df: pd.DataFrame, config: Config, run_log: RunLog) -> list[Flag]:
    """Run all rule-based anomaly checks.

    Calls each individual rule detector in sequence and returns a flat list of
    all emitted :class:`~src.models.Flag` objects.  A single transaction may
    appear multiple times — once per triggered rule.

    Rules executed (in order):
    1. Statistical outlier (z-score per account)
    2. Benford's law deviation (dataset-level chi-square)
    3. Duplicate payment (same vendor + amount within window)
    4. Weekend posting
    5. After-hours posting
    6. Round-number amount

    Parameters
    ----------
    df:
        Cleaned transaction DataFrame produced by ``Transaction_Loader.load``.
    config:
        Pipeline configuration (thresholds, windows, etc.).
    run_log:
        Shared run log; each rule records its skips here.

    Returns
    -------
    list[Flag]
        Combined flat list of all flags across all rules.
    """
    all_flags: list[Flag] = []

    all_flags.extend(_detect_statistical_outliers(df, config, run_log))
    all_flags.extend(_detect_benford_deviation(df, config, run_log))
    all_flags.extend(_detect_duplicate_payments(df, config, run_log))
    all_flags.extend(_detect_weekend_postings(df, config, run_log))
    all_flags.extend(_detect_after_hours_postings(df, config, run_log))
    all_flags.extend(_detect_round_amounts(df, config, run_log))

    return all_flags
