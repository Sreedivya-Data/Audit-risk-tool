"""ML-based anomaly detection using Isolation Forest."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder

from src.config import Config
from src.models import Flag
from src.run_log import RunLog

# Features used in the Isolation Forest model, in fixed order.
_FEATURE_NAMES = [
    "amount_log",
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "account_enc",
    "vendor_enc",
    "approver_enc",
    "amount_z_norm",
]


def _build_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    """Construct the numeric feature matrix from a cleaned transaction DataFrame.

    Returns an (n_samples, n_features) float64 array in the order of
    ``_FEATURE_NAMES``.
    """
    # amount_log
    amount_log = np.log1p(df["amount"].abs().values)

    # hour_of_day  (handle both datetime.time and NaT/NaN gracefully)
    def _time_to_hour(t) -> float:
        try:
            return t.hour + t.minute / 60.0
        except AttributeError:
            return 0.0  # fallback for unexpected nulls

    hour_of_day = np.array([_time_to_hour(t) for t in df["time"]], dtype=float)

    # day_of_week — df["date"] is datetime64[ns]
    day_of_week = pd.to_datetime(df["date"]).dt.day_of_week.values.astype(float)

    # is_weekend
    is_weekend = (day_of_week >= 5).astype(float)

    # Categorical label encoders — fit on the current dataset
    def _encode(series: pd.Series) -> np.ndarray:
        le = LabelEncoder()
        return le.fit_transform(series.astype(str)).astype(float)

    account_enc = _encode(df["account"])
    vendor_enc = _encode(df["vendor"])
    approver_enc = _encode(df["approver"])

    # amount_z_norm — global normalisation; guard against zero std
    amounts = df["amount"].values.astype(float)
    global_mean = amounts.mean()
    global_std = amounts.std(ddof=0)
    if global_std == 0.0:
        amount_z_norm = np.zeros(len(amounts))
    else:
        amount_z_norm = (amounts - global_mean) / global_std

    return np.column_stack([
        amount_log,
        hour_of_day,
        day_of_week,
        is_weekend,
        account_enc,
        vendor_enc,
        approver_enc,
        amount_z_norm,
    ])


def _top3_feature_contributions(
    row_features: np.ndarray,
    medians: np.ndarray,
) -> dict[str, float]:
    """Return the top-3 feature deltas (row value minus global median) by absolute value."""
    deltas = row_features - medians
    top_indices = np.argsort(np.abs(deltas))[::-1][:3]
    return {_FEATURE_NAMES[i]: float(deltas[i]) for i in top_indices}


def detect(
    df: pd.DataFrame,
    rule_flags: list[Flag],
    config: Config,
    run_log: RunLog,
) -> list[Flag]:
    """Train Isolation Forest and emit ML anomaly flags.

    Parameters
    ----------
    df:
        Cleaned transaction DataFrame produced by Transaction_Loader.
    rule_flags:
        Flags already emitted by the rule-based Anomaly_Detector.
    config:
        Pipeline configuration (thresholds, random state, etc.).
    run_log:
        Mutable run log; skip events are appended here.

    Returns
    -------
    list[Flag]
        One Flag per anomalous transaction with
        ``rule = "ml_isolation_forest"``.
    """
    # ------------------------------------------------------------------
    # Skip condition
    # ------------------------------------------------------------------
    if len(df) < config.min_ml_tx:
        reason = (
            f"Dataset has {len(df)} transactions, "
            f"which is below the minimum {config.min_ml_tx} required for ML detection."
        )
        run_log.log_ml_skip(reason)
        return []

    # ------------------------------------------------------------------
    # Feature matrix
    # ------------------------------------------------------------------
    X = _build_feature_matrix(df)

    # ------------------------------------------------------------------
    # Contamination rate
    # ------------------------------------------------------------------
    unique_flagged = len({f.tx_id for f in rule_flags if f.tx_id != "__dataset__"})
    rule_flag_rate = unique_flagged / len(df)
    contamination = 0.05
    # sklearn requires contamination <= 0.5
    contamination = min(contamination, 0.5)

    # ------------------------------------------------------------------
    # Train Isolation Forest
    # ------------------------------------------------------------------
    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=config.random_state,
        n_jobs=-1,
    )
    model.fit(X)

    # ------------------------------------------------------------------
    # Predict anomalies (-1 = anomaly, 1 = normal)
    # ------------------------------------------------------------------
    predictions = model.predict(X)       # shape (n,)
    anomaly_scores = model.decision_function(X)  # shape (n,); more negative = more anomalous

    # Per-feature medians across the whole dataset (for contribution deltas)
    feature_medians = np.median(X, axis=0)

    # ------------------------------------------------------------------
    # Build flags
    # ------------------------------------------------------------------
    flags: list[Flag] = []
    tx_ids = df["tx_id"].values if "tx_id" in df.columns else df.index.astype(str).values

    for i, pred in enumerate(predictions):
        if pred != -1:
            continue  # not an anomaly

        score = float(anomaly_scores[i])
        contributions = _top3_feature_contributions(X[i], feature_medians)

        flags.append(Flag(
            tx_id=str(tx_ids[i]),
            rule="ml_isolation_forest",
            threshold="isolation_forest_decision < 0",
            evidence={
                "anomaly_score": score,
                "feature_contributions": contributions,
            },
        ))

    return flags
