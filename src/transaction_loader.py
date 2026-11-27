"""Transaction_Loader: reads, validates, and cleans the transaction dataset.

The loader reads CSV/Excel input as raw strings (no silent type
coercion), validates the required schema, assigns a deterministic
``tx_id`` derived from the original row contents, coerces ``date`` /
``time`` / ``amount`` into typed columns, drops rows missing required
fields, removes exact duplicates, and reports everything it did in a
``ValidationReport``.
"""

from __future__ import annotations

import hashlib
import re
from datetime import time
from pathlib import Path

import pandas as pd

from src.config import Config
from src.exceptions import EmptyDatasetError, MissingColumnsError
from src.models import ValidationReport
from src.run_log import RunLog

#: Columns the input file must contain.
REQUIRED_COLUMNS = {"date", "time", "account", "amount", "approver", "vendor", "description"}

#: Supported input file extensions.
_SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls"}

_TIME_RE = re.compile(r"^(\d{1,2}):(\d{2})(?::(\d{2}))?$")


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _read_raw(path: Path) -> pd.DataFrame:
    """Read the input file with all columns as strings."""
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    suffix = path.suffix.lower()
    if suffix not in _SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file format: '{path.suffix}'. "
            f"Supported formats: {', '.join(sorted(_SUPPORTED_EXTENSIONS))}"
        )

    if suffix == ".csv":
        return pd.read_csv(path, dtype=str)
    return pd.read_excel(path, dtype=str)


def _validate_columns(df: pd.DataFrame) -> None:
    """Ensure every required column is present; raise MissingColumnsError otherwise."""
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise MissingColumnsError(missing)


def _assign_tx_ids(df: pd.DataFrame) -> pd.Series:
    """Compute a stable tx_id from the original row contents (pre-cleaning).

    The id is the first 12 hex chars of sha256 over the concatenation of
    all original column values, so it is deterministic across runs and
    independent of any later cleaning.
    """
    ids = []
    for _, row in df.iterrows():
        parts = ["" if pd.isna(v) else str(v) for v in row.tolist()]
        row_as_string = "|".join(parts)
        ids.append(hashlib.sha256(row_as_string.encode("utf-8")).hexdigest()[:12])
    return pd.Series(ids, index=df.index, name="tx_id", dtype=str)


def _is_clean_date(raw: str, parsed: pd.Timestamp) -> bool:
    """A raw date string is 'clean' when it is already canonical ISO yyyy-mm-dd."""
    return parsed == parsed.normalize() and raw == parsed.date().isoformat()


def _is_clean_amount(raw: str) -> bool:
    """A raw amount string is 'clean' when float() parses it unchanged."""
    try:
        float(raw)
    except (TypeError, ValueError):
        return False
    return True


def _parse_time_value(raw: object) -> time | None:
    """Parse 'HH:MM' or 'HH:MM:SS'; return None when unparseable."""
    if isinstance(raw, time):
        return raw
    if not isinstance(raw, str):
        return None
    match = _TIME_RE.match(raw.strip())
    if not match:
        return None
    hour, minute = int(match.group(1)), int(match.group(2))
    second = int(match.group(3)) if match.group(3) else 0
    try:
        return time(hour, minute, second)
    except ValueError:
        return None


def _canonical_time_str(value: time) -> str:
    if value.second == 0:
        return value.strftime("%H:%M")
    return value.strftime("%H:%M:%S")


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------

def load(path: str | Path, config: Config) -> tuple[pd.DataFrame, ValidationReport]:
    """Load, validate, and clean the transaction dataset.

    Returns a ``(df, report)`` tuple where ``df`` carries the typed
    columns ``tx_id, date, time, account, amount, approver, vendor,
    description`` and ``report`` describes every coercion, exclusion,
    and duplicate removal performed.
    """
    path = Path(path)
    raw_df = _read_raw(path)

    # Structural emptiness: header-only file.
    if raw_df.empty:
        raise EmptyDatasetError(
            "Dataset is empty after loading and cleaning. "
            "Row counts: loaded=0, coerced=0, excluded=0, duplicates_removed=0."
        )

    _validate_columns(raw_df)

    # Keep the untouched original columns for duplicate detection.
    original_columns = list(raw_df.columns)
    raw_values = raw_df.copy()

    # --- tx_id BEFORE any cleaning -----------------------------------
    tx_ids = _assign_tx_ids(raw_df)

    # --- Type coercion -----------------------------------------------
    coercion_detail: list[dict] = []
    rows_with_coercion: set[int] = set()

    # format='mixed' lets each row use its own natural format
    # (ISO, slash dates, ...) while errors='coerce' maps bad values to NaT.
    dates = pd.to_datetime(raw_df["date"], errors="coerce", format="mixed")
    amounts = pd.to_numeric(raw_df["amount"], errors="coerce")
    times = raw_df["time"].map(_parse_time_value)

    for idx in raw_df.index:
        raw_date = raw_df.at[idx, "date"]
        parsed_date = dates.at[idx]
        if pd.isna(parsed_date):
            coercion_detail.append({
                "row_index": idx,
                "field": "date",
                "raw_value": None if pd.isna(raw_date) else str(raw_date),
                "coerced_value": None,
            })
            rows_with_coercion.add(idx)
        elif pd.isna(raw_date) or str(raw_date).strip() != parsed_date.date().isoformat():
            coercion_detail.append({
                "row_index": idx,
                "field": "date",
                "raw_value": str(raw_date),
                "coerced_value": parsed_date.date().isoformat(),
            })
            rows_with_coercion.add(idx)

    for idx in raw_df.index:
        raw_amount = raw_df.at[idx, "amount"]
        coerced_amount = amounts.at[idx]
        if pd.isna(coerced_amount):
            coercion_detail.append({
                "row_index": idx,
                "field": "amount",
                "raw_value": None if pd.isna(raw_amount) else str(raw_amount),
                "coerced_value": None,
            })
            rows_with_coercion.add(idx)
        elif not _is_clean_amount(str(raw_amount).strip()):
            coercion_detail.append({
                "row_index": idx,
                "field": "amount",
                "raw_value": str(raw_amount),
                "coerced_value": float(coerced_amount),
            })
            rows_with_coercion.add(idx)

    for idx in raw_df.index:
        raw_time = raw_df.at[idx, "time"]
        parsed_time = times.at[idx]
        if parsed_time is None:
            coercion_detail.append({
                "row_index": idx,
                "field": "time",
                "raw_value": None if pd.isna(raw_time) else str(raw_time),
                "coerced_value": None,
            })
            rows_with_coercion.add(idx)
        elif pd.isna(raw_time) or str(raw_time).strip() != _canonical_time_str(parsed_time):
            coercion_detail.append({
                "row_index": idx,
                "field": "time",
                "raw_value": str(raw_time),
                "coerced_value": _canonical_time_str(parsed_time),
            })
            rows_with_coercion.add(idx)

    # --- Exclusion: null account / amount / date after coercion -------
    exclusion_detail: list[dict] = []

    def _account_is_null(idx) -> bool:
        value = raw_df.at[idx, "account"]
        return pd.isna(value) or str(value).strip() == ""

    excluded_mask = (
        raw_df.index.map(_account_is_null)
        | dates.isna()
        | amounts.isna()
    )

    for idx in raw_df.index:
        if pd.isna(dates.at[idx]):
            exclusion_detail.append({"row_index": idx, "reason": "date is null or unparseable"})
        elif pd.isna(amounts.at[idx]):
            exclusion_detail.append({"row_index": idx, "reason": "amount is null or non-numeric"})
        elif _account_is_null(idx):
            exclusion_detail.append({"row_index": idx, "reason": "account is null or empty"})

    kept_df = raw_df.loc[~excluded_mask].copy()
    kept_dates = dates.loc[~excluded_mask]
    kept_amounts = amounts.loc[~excluded_mask]
    kept_times = times.loc[~excluded_mask]
    kept_tx_ids = tx_ids.loc[~excluded_mask]

    # --- Duplicate removal on ORIGINAL columns ------------------------
    original_subset = raw_values.loc[kept_df.index]
    dup_mask = original_subset.duplicated(keep="first")
    duplicates_removed = int(dup_mask.sum())

    kept_df = kept_df.loc[~dup_mask].copy()
    kept_dates = kept_dates.loc[~dup_mask]
    kept_amounts = kept_amounts.loc[~dup_mask]
    kept_times = kept_times.loc[~dup_mask]
    kept_tx_ids = kept_tx_ids.loc[~dup_mask]

    rows_loaded = len(raw_df)
    rows_excluded = len(exclusion_detail)

    # --- Empty dataset after cleaning ---------------------------------
    if kept_df.empty:
        raise EmptyDatasetError(
            "Dataset is empty after loading and cleaning. "
            f"Row counts: loaded={rows_loaded}, "
            f"coerced={len(rows_with_coercion)}, "
            f"excluded={rows_excluded}, "
            f"duplicates_removed={duplicates_removed}."
        )

    # --- Assemble the cleaned, typed DataFrame ------------------------
    cleaned = pd.DataFrame(index=kept_df.index)
    cleaned["tx_id"] = kept_tx_ids.astype(str)
    cleaned["date"] = kept_dates.astype("datetime64[ns]")
    cleaned["time"] = kept_times
    cleaned["account"] = kept_df["account"].fillna("").astype(str).str.strip()
    cleaned["amount"] = kept_amounts.astype("float64")
    cleaned["approver"] = kept_df["approver"].fillna("").astype(str).str.strip()
    cleaned["vendor"] = kept_df["vendor"].fillna("").astype(str).str.strip()
    cleaned["description"] = kept_df["description"].fillna("").astype(str).str.strip()

    cleaned = cleaned.reset_index(drop=True)

    report = ValidationReport(
        rows_loaded=rows_loaded,
        rows_coerced=len(rows_with_coercion),
        rows_excluded=rows_excluded,
        duplicates_removed=duplicates_removed,
        coercion_detail=coercion_detail,
        exclusion_detail=exclusion_detail,
    )

    # --- Run log ------------------------------------------------------
    run_log = RunLog(input_file=str(path), config=config)
    run_log.log_row_counts(
        loaded=report.rows_loaded,
        coerced=report.rows_coerced,
        excluded=report.rows_excluded,
        duplicates_removed=report.duplicates_removed,
    )
    run_log.save()

    return cleaned, report
