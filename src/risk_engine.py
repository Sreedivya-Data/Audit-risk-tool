"""Risk scoring and rollup engine for the AI Audit and Risk Analytics Tool."""

from __future__ import annotations

import json
from collections import defaultdict

import pandas as pd

from src.config import Config
from src.models import Flag


def score(
    df: pd.DataFrame,
    flags: list[Flag],
    config: Config,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Assign risk scores to transactions and produce account / approver rollups.

    Parameters
    ----------
    df:
        Cleaned transaction DataFrame produced by Transaction_Loader.
    flags:
        All flags emitted by Anomaly_Detector and ML_Detector.  Flags with
        ``tx_id == "__dataset__"`` (e.g. benford_deviation dataset-level
        flags) are ignored for per-transaction scoring.
    config:
        Tool configuration, including ``config.rule_weights``.

    Returns
    -------
    tx_scores:
        A copy of *df* with two additional columns:
        - ``risk_score``      — integer 0-100
        - ``score_breakdown`` — JSON string of ``{rule: points}`` per flag
    account_rollup:
        One row per unique account.
    approver_rollup:
        One row per unique approver.
    """
    # ------------------------------------------------------------------
    # 1. Build per-transaction scoring
    # ------------------------------------------------------------------

    # Group flags by tx_id, excluding dataset-level sentinels.
    tx_flags: dict[str, list[Flag]] = defaultdict(list)
    for flag in flags:
        if flag.tx_id != "__dataset__":
            tx_flags[flag.tx_id].append(flag)

    risk_scores: list[int] = []
    score_breakdowns: list[str] = []

    for _, row in df.iterrows():
        tx_id = row["tx_id"]
        tx_flag_list = tx_flags.get(tx_id, [])

        if not tx_flag_list:
            risk_scores.append(0)
            score_breakdowns.append(json.dumps({}))
            continue

        # Sum weights for every flag instance; each flag contributes its
        # rule's base weight independently (a rule may fire more than once).
        raw_total = 0
        breakdown: dict[str, int] = {}
        for flag in tx_flag_list:
            weight = config.rule_weights.get(flag.rule, 0)
            raw_total += weight
            # breakdown accumulates per-flag contribution; if the same rule
            # fires multiple times each occurrence is recorded as an addend.
            breakdown[flag.rule] = breakdown.get(flag.rule, 0) + weight

        capped = min(raw_total, 100)
        risk_scores.append(capped)
        score_breakdowns.append(json.dumps(breakdown))

    tx_scores = df.copy()
    tx_scores["risk_score"] = risk_scores
    tx_scores["score_breakdown"] = score_breakdowns

    # ------------------------------------------------------------------
    # 2. Build rollups
    # ------------------------------------------------------------------

    account_rollup = _build_rollup(tx_scores, tx_flags, group_col="account")
    approver_rollup = _build_rollup(tx_scores, tx_flags, group_col="approver")

    return tx_scores, account_rollup, approver_rollup


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_rollup(
    tx_scores: pd.DataFrame,
    tx_flags: dict[str, list[Flag]],
    group_col: str,
) -> pd.DataFrame:
    """Build an account or approver rollup DataFrame.

    Parameters
    ----------
    tx_scores:
        Transaction DataFrame with ``risk_score`` column.
    tx_flags:
        Mapping of tx_id → list of Flag objects (dataset-level flags excluded).
    group_col:
        ``"account"`` or ``"approver"``.
    """
    rows = []

    for group_key, group_df in tx_scores.groupby(group_col, sort=True):
        tx_count = len(group_df)
        mean_risk = float(group_df["risk_score"].mean())
        max_risk = float(group_df["risk_score"].max())
        is_small_sample = tx_count < 5

        # Count flags per rule across the group's transactions
        flag_count_by_rule: dict[str, int] = defaultdict(int)
        for tx_id in group_df["tx_id"]:
            for flag in tx_flags.get(tx_id, []):
                flag_count_by_rule[flag.rule] += 1

        rows.append(
            {
                group_col: group_key,
                "tx_count": tx_count,
                "flag_count_by_rule": dict(flag_count_by_rule),
                "mean_risk_score": mean_risk,
                "max_risk_score": max_risk,
                "is_small_sample": is_small_sample,
            }
        )

    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=[
            group_col,
            "tx_count",
            "flag_count_by_rule",
            "mean_risk_score",
            "max_risk_score",
            "is_small_sample",
        ]
    )
