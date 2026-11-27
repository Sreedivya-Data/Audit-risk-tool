"""Structured JSON run log writer and output-artifact writer used by all pipeline modules."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from src.config import Config, config_snapshot

if TYPE_CHECKING:
    from src.models import Flag


class RunLog:
    """Accumulates pipeline events and serialises them to output/run_<timestamp>.json."""

    def __init__(self, input_file: str, config: Config) -> None:
        self.run_id: str = str(uuid.uuid4())
        self.timestamp: str = datetime.now(timezone.utc).isoformat()
        self.input_file: str = input_file
        self._config_snapshot: dict[str, Any] = config_snapshot(config)

        # Mutable state accumulated during the run
        self._row_counts: dict[str, int] = {
            "loaded": 0,
            "coerced": 0,
            "excluded": 0,
            "duplicates_removed": 0,
        }
        self._rule_flag_counts: dict[str, int] = {}
        self._skipped_checks: list[dict[str, Any]] = []
        self._ml_skip_reason: str | None = None
        self._ai_prompt: str | None = None
        self._ai_response: str | None = None
        self._summary_text: str | None = None

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def log_row_counts(
        self,
        loaded: int,
        coerced: int,
        excluded: int,
        duplicates_removed: int,
    ) -> None:
        """Record the row-count statistics from Transaction_Loader."""
        self._row_counts = {
            "loaded": loaded,
            "coerced": coerced,
            "excluded": excluded,
            "duplicates_removed": duplicates_removed,
        }

    def log_rule_flag_counts(self, counts: dict[str, int]) -> None:
        """Record the per-rule flag counts after Anomaly/ML detection."""
        self._rule_flag_counts = dict(counts)

    def log_skip(self, check: str, reason: str, **kwargs: Any) -> None:
        """Append a skipped-check entry.

        ``check`` is the check name (e.g. "zscore"), ``reason`` is the
        human-readable reason it was skipped.  Extra keyword arguments are
        merged into the entry for context (e.g. account="ACC-01").
        """
        entry: dict[str, Any] = {"check": check, "reason": reason}
        entry.update(kwargs)
        self._skipped_checks.append(entry)

    def log_ml_skip(self, reason: str) -> None:
        """Record the reason the ML detection step was skipped."""
        self._ml_skip_reason = reason

    def log_ai(self, prompt: str, response: str) -> None:
        """Store the AI model prompt and response verbatim."""
        self._ai_prompt = prompt
        self._ai_response = response

    def log_summary(self, summary_text: str) -> None:
        """Store the final summary text."""
        self._summary_text = summary_text

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return the full run log as a JSON-safe dictionary."""
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "input_file": self.input_file,
            "config_snapshot": self._config_snapshot,
            "row_counts": self._row_counts,
            "rule_flag_counts": self._rule_flag_counts,
            "skipped_checks": self._skipped_checks,
            "ml_skip_reason": self._ml_skip_reason,
            "ai_prompt": self._ai_prompt,
            "ai_response": self._ai_response,
            "summary_text": self._summary_text,
        }

    def save(self, output_dir: str | Path = "output/") -> Path:
        """Serialise the run log to output/run_<timestamp>.json.

        Creates the output directory if it doesn't exist.
        Returns the path to the written file.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Use the ISO timestamp with colons replaced so it's a valid filename
        safe_ts = self.timestamp.replace(":", "-").replace("+", "").split(".")[0]
        filename = f"run_{safe_ts}.json"
        path = output_dir / filename

        with path.open("w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, ensure_ascii=False)

        return path

    def finalise(self, output_dir: str | Path = "output/") -> Path:
        """Write the final run log to output/run_<timestamp>.json.

        Identical to :meth:`save` but named to match the pipeline convention of
        calling ``finalise`` after all logging helpers have been invoked.  Returns
        the path to the written JSON file.
        """
        return self.save(output_dir)

    def write_flags_artifact(
        self,
        flags: list[Flag],
        tx_scores: pd.DataFrame | None,
        output_dir: str | Path = "output/",
    ) -> Path:
        """Serialise all flags to Parquet and JSON output artifacts.

        Writes two files:
        - ``output/flags_<run_id>.parquet`` — primary columnar artifact
        - ``output/flags_<run_id>.json``    — portable JSON fallback

        Columns: ``run_id``, ``tx_id``, ``rule``, ``threshold``, ``evidence``
        (JSON string), ``risk_score``, ``score_breakdown`` (JSON string).

        ``risk_score`` and ``score_breakdown`` are looked up from *tx_scores* by
        ``tx_id``.  For the Benford dataset-level flag (``tx_id="__dataset__"``)
        or any ``tx_id`` not found in *tx_scores*, ``risk_score`` is set to 0 and
        ``score_breakdown`` to ``"{}"``.

        Parameters
        ----------
        flags:
            Flat list of :class:`~src.models.Flag` objects produced by the
            detection pipeline.
        tx_scores:
            DataFrame returned by :func:`~src.risk_engine.score` containing at
            least the columns ``tx_id``, ``risk_score``, and ``score_breakdown``.
            May be ``None`` when risk scoring was skipped.
        output_dir:
            Directory in which to write the artifact files.  Created if absent.

        Returns
        -------
        Path
            Path to the written ``.parquet`` file.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Build a lookup table: tx_id → (risk_score, score_breakdown_json)
        # from the scored transactions DataFrame.
        score_lookup: dict[str, tuple[float, str]] = {}
        if tx_scores is not None and not tx_scores.empty:
            for _, row in tx_scores.iterrows():
                tid = str(row["tx_id"])
                rs = float(row.get("risk_score", 0.0))
                sb = row.get("score_breakdown", "{}")
                # score_breakdown may already be a JSON string or a dict
                if isinstance(sb, dict):
                    sb = json.dumps(sb, ensure_ascii=False)
                elif not isinstance(sb, str):
                    sb = "{}"
                score_lookup[tid] = (rs, sb)

        # Build rows for the artifact
        rows: list[dict[str, Any]] = []
        for flag in flags:
            tid = flag.tx_id
            if tid == "__dataset__" or tid not in score_lookup:
                risk_score = 0.0
                score_breakdown = "{}"
            else:
                risk_score, score_breakdown = score_lookup[tid]

            rows.append(
                {
                    "run_id": self.run_id,
                    "tx_id": tid,
                    "rule": flag.rule,
                    "threshold": flag.threshold,
                    "evidence": json.dumps(flag.evidence, ensure_ascii=False, default=str),
                    "risk_score": risk_score,
                    "score_breakdown": score_breakdown,
                }
            )

        artifact_df = pd.DataFrame(
            rows,
            columns=[
                "run_id",
                "tx_id",
                "rule",
                "threshold",
                "evidence",
                "risk_score",
                "score_breakdown",
            ],
        )

        parquet_path = output_dir / f"flags_{self.run_id}.parquet"
        json_path = output_dir / f"flags_{self.run_id}.json"

        artifact_df.to_parquet(parquet_path, index=False)

        # Write the JSON fallback as a list of records
        artifact_df.to_json(json_path, orient="records", indent=2, force_ascii=False)

        return parquet_path
