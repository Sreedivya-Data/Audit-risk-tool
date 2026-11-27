"""AI-written audit summary with deterministic fallback.

Public interface
----------------
write_summary(flags, tx_scores, account_rollup, approver_rollup, config, run_log) -> str
"""

from __future__ import annotations

import json
import os
from datetime import date
from typing import Any

import pandas as pd

from src.config import Config
from src.models import Flag
from src.run_log import RunLog

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_LIMITATIONS_STATEMENT = (
    "Limitations: This analysis covers rule-based checks (statistical outlier, Benford's law, "
    "duplicate payments, weekend/after-hours postings, round amounts) and an exploratory Isolation "
    "Forest model. These checks are heuristic indicators; they do not prove fraud or error. "
    "All flagged items require auditor judgement before conclusions are drawn."
)


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


def write_summary(
    flags: list[Flag],
    tx_scores: pd.DataFrame,
    account_rollup: pd.DataFrame,
    approver_rollup: pd.DataFrame,
    config: Config,
    run_log: RunLog,
) -> str:
    """Generate a plain-English audit summary (≤ config.summary_max_words words).

    Attempts to call the OpenAI API with the structured prompt.  If the API key
    is not set, or if any exception occurs, the deterministic fallback template
    is used instead.

    The prompt text and raw API response (or ``"FALLBACK"``) are written
    verbatim to *run_log*.  The final summary (including the limitations
    statement) is also stored in *run_log*.

    Parameters
    ----------
    flags:
        All flags emitted by Anomaly_Detector and ML_Detector.
    tx_scores:
        Transaction DataFrame with ``risk_score`` and ``score_breakdown`` columns,
        as returned by Risk_Engine.score().
    account_rollup:
        Account-level rollup DataFrame from Risk_Engine.score().
    approver_rollup:
        Approver-level rollup DataFrame from Risk_Engine.score().
    config:
        Tool configuration.
    run_log:
        Run log instance; receives prompt, response, and final summary text.

    Returns
    -------
    str
        Plain-English audit summary, always ending with the limitations statement.
    """
    top_risks = _build_top_risks(flags, tx_scores, config)
    prompt = _build_prompt(flags, tx_scores, top_risks)

    response_text: str
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()

    if not api_key:
        # No key configured — skip the API call and use the fallback.
        response_text = "FALLBACK"
        summary_body = _deterministic_summary(flags, tx_scores, top_risks)
    else:
        # Attempt the OpenAI API call; fall back on any exception.
        try:
            import openai  # noqa: PLC0415 — intentional lazy import

            client = openai.OpenAI(api_key=api_key)
            completion = client.chat.completions.create(
                model=config.openai_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a professional auditor writing a concise, plain-English "
                            "audit summary for a non-technical client. Limit your response to "
                            f"{config.summary_max_words} words."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=config.openai_max_tokens,
                timeout=config.openai_timeout,
            )
            response_text = completion.choices[0].message.content or ""
            summary_body = response_text
        except Exception:  # noqa: BLE001 — intentional broad catch per spec
            response_text = "FALLBACK"
            summary_body = _deterministic_summary(flags, tx_scores, top_risks)

    # Log prompt and response verbatim.
    run_log.log_ai(prompt=prompt, response=response_text)

    # Enforce word-count limit on the body before appending the limitations
    # statement (which must always be present in full).
    summary_body = _truncate_to_words(summary_body, config.summary_max_words)

    final_summary = f"{summary_body.rstrip()}\n\n{_LIMITATIONS_STATEMENT}"

    # Store in the run log.
    run_log.log_summary(final_summary)

    return final_summary


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def _build_top_risks(
    flags: list[Flag],
    tx_scores: pd.DataFrame,
    config: Config,
) -> list[dict[str, Any]]:
    """Return a list of up to config.top_risks_in_summary risk dicts, sorted
    by descending risk_score.

    Dataset-level flags (tx_id == "__dataset__") are listed first if they
    exist, then per-transaction risks.
    """
    max_top = config.top_risks_in_summary

    # Build a quick lookup of risk_score per tx_id.
    if not tx_scores.empty and "tx_id" in tx_scores.columns and "risk_score" in tx_scores.columns:
        score_lookup: dict[str, int] = dict(
            zip(tx_scores["tx_id"], tx_scores["risk_score"])
        )
    else:
        score_lookup = {}

    # Aggregate flags by (tx_id, rule) — keep the first flag's evidence as
    # representative, record occurrence count.
    agg: dict[tuple[str, str], dict[str, Any]] = {}
    for flag in flags:
        key = (flag.tx_id, flag.rule)
        if key not in agg:
            agg[key] = {
                "tx_id": flag.tx_id,
                "rule": flag.rule,
                "threshold": flag.threshold,
                "evidence": flag.evidence,
                "risk_score": score_lookup.get(flag.tx_id, 0),
            }
        else:
            # If the same (tx_id, rule) appears more than once just keep the
            # first occurrence; risk_score already accounts for the weight sum.
            pass

    # Sort dataset-level flags first (they carry Benford evidence), then by
    # descending risk score.
    ranked = sorted(
        agg.values(),
        key=lambda r: (r["tx_id"] != "__dataset__", -r["risk_score"]),
    )

    return ranked[:max_top]


def _build_prompt(
    flags: list[Flag],
    tx_scores: pd.DataFrame,
    top_risks: list[dict[str, Any]],
) -> str:
    """Build a structured prompt string for the OpenAI API call."""
    total_tx = len(tx_scores)
    # Count distinct flagged transactions (excluding dataset-level sentinels).
    flagged_tx_ids = {f.tx_id for f in flags if f.tx_id != "__dataset__"}
    flag_count = len(flagged_tx_ids)
    flag_pct = (flag_count / total_tx * 100) if total_tx > 0 else 0.0

    lines = [
        "You are preparing an audit summary for a non-technical client.",
        "",
        f"Dataset: {total_tx} transactions analysed, {flag_count} flagged ({flag_pct:.1f}%).",
        "",
        "Top risks (by risk score):",
    ]

    for i, risk in enumerate(top_risks, start=1):
        evidence_str = json.dumps(risk["evidence"], default=str)
        lines.append(
            f"{i}. Rule: {risk['rule']} | TX ID: {risk['tx_id']} | "
            f"Risk score: {risk['risk_score']} | Threshold: {risk['threshold']} | "
            f"Evidence: {evidence_str}"
        )

    lines += [
        "",
        "Write a concise, plain-English summary covering the top risks above, "
        "citing the specific evidence for each. Do NOT include a limitations statement "
        "(it will be appended separately). Limit to 400 words.",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Deterministic fallback template
# ---------------------------------------------------------------------------


def _deterministic_summary(
    flags: list[Flag],
    tx_scores: pd.DataFrame,
    top_risks: list[dict[str, Any]],
) -> str:
    """Produce the structured fallback summary from run data only.

    Every number in this output is sourced directly from the pipeline's
    computed statistics (Requirements 10.6).
    """
    today = date.today().isoformat()
    total_tx = len(tx_scores)
    flagged_tx_ids = {f.tx_id for f in flags if f.tx_id != "__dataset__"}
    flag_count = len(flagged_tx_ids)
    flag_pct = (flag_count / total_tx * 100) if total_tx > 0 else 0.0

    lines = [
        "Audit Summary (auto-generated)",
        f"Run date: {today}",
        f"Transactions analysed: {total_tx} | Flagged: {flag_count} ({flag_pct:.1f}%)",
        "",
        "Top risks:",
    ]

    if not top_risks:
        lines.append("  No risks detected.")
    else:
        for i, risk in enumerate(top_risks, start=1):
            evidence_summary = _format_evidence(risk["evidence"])
            lines.append(
                f"{i}. {risk['rule']}: risk score {risk['risk_score']}, "
                f"threshold {risk['threshold']}"
            )
            lines.append(f"   Evidence: {evidence_summary}")

    return "\n".join(lines)


def _format_evidence(evidence: dict[str, Any]) -> str:
    """Produce a compact, human-readable one-liner from an evidence dict."""
    if not evidence:
        return "(no evidence)"
    # Render as "key=value, key=value, …" keeping values concise.
    parts = []
    for k, v in evidence.items():
        # Truncate long nested structures to avoid runaway length.
        if isinstance(v, (dict, list)):
            v_str = json.dumps(v, default=str)
            if len(v_str) > 60:
                v_str = v_str[:57] + "..."
        else:
            v_str = str(v)
        parts.append(f"{k}={v_str}")
    return ", ".join(parts)


# ---------------------------------------------------------------------------
# Word-count utilities
# ---------------------------------------------------------------------------


def _word_count(text: str) -> int:
    """Return the number of whitespace-delimited tokens in *text*."""
    return len(text.split())


def _truncate_to_words(text: str, max_words: int) -> str:
    """Truncate *text* to at most *max_words* words, preserving whole words."""
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])
