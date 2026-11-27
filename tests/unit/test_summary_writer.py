"""Unit tests for summary_writer.py.

Covers:
- Deterministic fallback when OPENAI_API_KEY is not set
- Deterministic fallback when the OpenAI API raises an exception
- AI path with a mocked OpenAI client
- Limitations statement always present
- Word-count enforcement / truncation
- Numbers in summary are sourced from run data
- run_log entries (prompt, response, summary_text)
- Empty-flags edge case
- Dataset-level Benford flag handling

Requirements: 10.1–10.6
"""

from __future__ import annotations

import datetime
import json
import os
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.config import Config
from src.models import Flag
from src.run_log import RunLog
from src.summary_writer import (
    _LIMITATIONS_STATEMENT,
    _truncate_to_words,
    _word_count,
    write_summary,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def config_default() -> Config:
    return Config()


def _make_tx_scores(n: int = 3) -> pd.DataFrame:
    """Return a minimal tx_scores DataFrame with *n* rows."""
    # Build risk_score and score_breakdown lists dynamically for any n.
    risk_scores = [65, 10] + [0] * max(n - 2, 0)
    breakdowns = [
        json.dumps({"statistical_outlier": 30, "round_amount": 10, "ml_isolation_forest": 25}),
        json.dumps({"round_amount": 10}),
    ] + [json.dumps({})] * max(n - 2, 0)
    # Trim to exactly n entries.
    risk_scores = risk_scores[:n]
    breakdowns = breakdowns[:n]
    return pd.DataFrame(
        {
            "tx_id": [f"tx{i:04d}" for i in range(n)],
            "date": [datetime.date(2024, 1, 2)] * n,
            "account": [f"ACC-{i:02d}" for i in range(n)],
            "amount": [float((i + 1) * 1000) for i in range(n)],
            "approver": [f"APR-{i:02d}" for i in range(n)],
            "vendor": [f"VND-{i:02d}" for i in range(n)],
            "description": [f"desc {i}" for i in range(n)],
            "risk_score": risk_scores,
            "score_breakdown": breakdowns,
        }
    )


def _make_flags() -> list[Flag]:
    return [
        Flag(
            tx_id="tx0000",
            rule="statistical_outlier",
            threshold="|z| > 3.0",
            evidence={"z_score": 4.2, "account_mean": 10000.0, "account_std": 9500.0, "account_tx_count": 15},
        ),
        Flag(
            tx_id="tx0000",
            rule="round_amount",
            threshold="amount % 1000 == 0",
            evidence={"amount": 1000.0, "threshold": 1000.0},
        ),
        Flag(
            tx_id="tx0001",
            rule="round_amount",
            threshold="amount % 1000 == 0",
            evidence={"amount": 2000.0, "threshold": 1000.0},
        ),
    ]


def _make_rollup(col: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            col: ["ACC-00", "ACC-01"],
            "tx_count": [2, 1],
            "flag_count_by_rule": [{"statistical_outlier": 1}, {}],
            "mean_risk_score": [37.5, 0.0],
            "max_risk_score": [65.0, 0.0],
            "is_small_sample": [True, True],
        }
    )


def _run_summary(config: Config, flags=None, tx_scores=None, account_rollup=None, approver_rollup=None):
    """Helper: run write_summary and return (summary, run_log)."""
    if flags is None:
        flags = _make_flags()
    if tx_scores is None:
        tx_scores = _make_tx_scores()
    if account_rollup is None:
        account_rollup = _make_rollup("account")
    if approver_rollup is None:
        approver_rollup = _make_rollup("approver")

    run_log = RunLog("test.csv", config)
    summary = write_summary(flags, tx_scores, account_rollup, approver_rollup, config, run_log)
    return summary, run_log


# ---------------------------------------------------------------------------
# Fallback: no API key
# ---------------------------------------------------------------------------


class TestFallbackNoApiKey:
    """When OPENAI_API_KEY is not set, the deterministic template must be used."""

    def test_returns_string(self, config_default, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        summary, _ = _run_summary(config_default)
        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_limitations_always_present(self, config_default, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        summary, _ = _run_summary(config_default)
        assert _LIMITATIONS_STATEMENT in summary

    def test_run_log_response_is_fallback(self, config_default, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        _, run_log = _run_summary(config_default)
        assert run_log._ai_response == "FALLBACK"

    def test_run_log_prompt_logged(self, config_default, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        _, run_log = _run_summary(config_default)
        assert run_log._ai_prompt is not None
        assert len(run_log._ai_prompt) > 0

    def test_run_log_summary_text_matches_return(self, config_default, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        summary, run_log = _run_summary(config_default)
        assert run_log._summary_text == summary

    def test_word_count_within_limit(self, config_default, monkeypatch):
        """The summary body (before the limitations statement) must be ≤ max_words.
        The final string may exceed max_words because the limitations statement
        is always appended after truncation.
        """
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        summary, _ = _run_summary(config_default)
        # Strip the always-appended limitations statement to measure body only.
        body = summary.replace(_LIMITATIONS_STATEMENT, "").strip()
        assert _word_count(body) <= config_default.summary_max_words

    def test_fallback_contains_total_tx_count(self, config_default, monkeypatch):
        """Total transaction count must appear verbatim in the summary (Req 10.6)."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        tx_scores = _make_tx_scores(3)
        summary, _ = _run_summary(config_default, tx_scores=tx_scores)
        # tx_count == 3
        assert "3" in summary

    def test_fallback_contains_flag_count(self, config_default, monkeypatch):
        """Flagged-transaction count must appear in the summary (Req 10.6)."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        flags = _make_flags()
        summary, _ = _run_summary(config_default, flags=flags)
        # 2 distinct flagged tx_ids: tx0000, tx0001
        assert "2" in summary

    def test_fallback_contains_top_rule_name(self, config_default, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        summary, _ = _run_summary(config_default)
        assert "statistical_outlier" in summary

    def test_fallback_top_risk_score_present(self, config_default, monkeypatch):
        """The highest risk score (65) must appear somewhere in the summary (Req 10.6)."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        summary, _ = _run_summary(config_default)
        assert "65" in summary


# ---------------------------------------------------------------------------
# Fallback: API key present but call raises an exception
# ---------------------------------------------------------------------------


class TestFallbackOnException:
    """When the OpenAI call raises, fall back gracefully."""

    def test_fallback_on_timeout(self, config_default, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        mock_openai = MagicMock()
        mock_openai.OpenAI.return_value.chat.completions.create.side_effect = Exception("timeout")
        with patch.dict("sys.modules", {"openai": mock_openai}):
            summary, run_log = _run_summary(config_default)
        assert _LIMITATIONS_STATEMENT in summary
        assert run_log._ai_response == "FALLBACK"

    def test_fallback_on_auth_error(self, config_default, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-bad")
        mock_openai = MagicMock()
        mock_openai.OpenAI.return_value.chat.completions.create.side_effect = Exception("AuthenticationError")
        with patch.dict("sys.modules", {"openai": mock_openai}):
            summary, run_log = _run_summary(config_default)
        assert _LIMITATIONS_STATEMENT in summary
        assert run_log._ai_response == "FALLBACK"

    def test_summary_still_well_formed_on_exception(self, config_default, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        mock_openai = MagicMock()
        mock_openai.OpenAI.return_value.chat.completions.create.side_effect = RuntimeError("network")
        with patch.dict("sys.modules", {"openai": mock_openai}):
            summary, _ = _run_summary(config_default)
        assert len(summary) > 0
        assert _word_count(summary) <= config_default.summary_max_words


# ---------------------------------------------------------------------------
# AI path with mocked OpenAI client
# ---------------------------------------------------------------------------


class TestAIPath:
    """When the API key is present and the call succeeds, use the AI response."""

    def _mock_openai(self, response_text: str) -> MagicMock:
        mock_choice = MagicMock()
        mock_choice.message.content = response_text
        mock_completion = MagicMock()
        mock_completion.choices = [mock_choice]
        mock_client_instance = MagicMock()
        mock_client_instance.chat.completions.create.return_value = mock_completion
        mock_openai = MagicMock()
        mock_openai.OpenAI.return_value = mock_client_instance
        return mock_openai

    def test_uses_ai_response_body(self, config_default, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        ai_body = "The analysis identified three main risk areas."
        mock_openai = self._mock_openai(ai_body)
        with patch.dict("sys.modules", {"openai": mock_openai}):
            summary, run_log = _run_summary(config_default)
        assert ai_body in summary
        assert run_log._ai_response == ai_body

    def test_limitations_appended_to_ai_response(self, config_default, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        ai_body = "Risk summary from AI."
        mock_openai = self._mock_openai(ai_body)
        with patch.dict("sys.modules", {"openai": mock_openai}):
            summary, _ = _run_summary(config_default)
        assert _LIMITATIONS_STATEMENT in summary

    def test_prompt_logged_verbatim(self, config_default, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        ai_body = "Brief AI summary."
        mock_openai = self._mock_openai(ai_body)
        with patch.dict("sys.modules", {"openai": mock_openai}):
            summary, run_log = _run_summary(config_default)
        assert run_log._ai_prompt is not None
        # Prompt should reference rule names from the flags
        assert "statistical_outlier" in run_log._ai_prompt

    def test_response_logged_verbatim(self, config_default, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        ai_body = "Exact AI response text."
        mock_openai = self._mock_openai(ai_body)
        with patch.dict("sys.modules", {"openai": mock_openai}):
            _, run_log = _run_summary(config_default)
        assert run_log._ai_response == ai_body

    def test_word_count_within_limit_ai(self, config_default, monkeypatch):
        """Even if the AI sends more than 400 words, we enforce the limit on the body."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        # 500 words
        long_response = " ".join(["word"] * 500)
        mock_openai = self._mock_openai(long_response)
        with patch.dict("sys.modules", {"openai": mock_openai}):
            summary, _ = _run_summary(config_default)
        # Strip limitations before checking — it's always appended after truncation.
        body = summary.replace(_LIMITATIONS_STATEMENT, "").strip()
        assert _word_count(body) <= config_default.summary_max_words

    def test_api_called_with_correct_model(self, config_default, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        ai_body = "summary"
        mock_openai = self._mock_openai(ai_body)
        with patch.dict("sys.modules", {"openai": mock_openai}):
            _run_summary(config_default)
        call_kwargs = mock_openai.OpenAI.return_value.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == config_default.openai_model
        assert call_kwargs["max_tokens"] == config_default.openai_max_tokens
        assert call_kwargs["timeout"] == config_default.openai_timeout


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_no_flags(self, config_default, monkeypatch):
        """Zero flags — summary should still be valid and within word count."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        tx_scores = _make_tx_scores(5)
        tx_scores["risk_score"] = 0
        tx_scores["score_breakdown"] = json.dumps({})
        empty_rollup_acc = pd.DataFrame(
            columns=["account", "tx_count", "flag_count_by_rule", "mean_risk_score", "max_risk_score", "is_small_sample"]
        )
        empty_rollup_apr = pd.DataFrame(
            columns=["approver", "tx_count", "flag_count_by_rule", "mean_risk_score", "max_risk_score", "is_small_sample"]
        )
        summary, run_log = _run_summary(
            config_default,
            flags=[],
            tx_scores=tx_scores,
            account_rollup=empty_rollup_acc,
            approver_rollup=empty_rollup_apr,
        )
        assert _LIMITATIONS_STATEMENT in summary
        assert _word_count(summary) <= config_default.summary_max_words
        assert run_log._ai_response == "FALLBACK"

    def test_dataset_level_benford_flag(self, config_default, monkeypatch):
        """A dataset-level flag (tx_id='__dataset__') should not raise errors."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        flags = [
            Flag(
                tx_id="__dataset__",
                rule="benford_deviation",
                threshold="p < 0.05",
                evidence={"chi2_statistic": 25.3, "p_value": 0.001},
            )
        ]
        tx_scores = _make_tx_scores(3)
        tx_scores["risk_score"] = 0
        tx_scores["score_breakdown"] = json.dumps({})
        summary, _ = _run_summary(
            config_default,
            flags=flags,
            tx_scores=tx_scores,
            account_rollup=_make_rollup("account"),
            approver_rollup=_make_rollup("approver"),
        )
        assert _LIMITATIONS_STATEMENT in summary

    def test_custom_word_limit_enforced(self, monkeypatch):
        """Custom config.summary_max_words is respected for the body."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        config = Config(summary_max_words=50)
        summary, _ = _run_summary(config)
        body = summary.replace(_LIMITATIONS_STATEMENT, "").strip()
        assert _word_count(body) <= 50

    def test_summary_text_stored_in_run_log(self, config_default, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        summary, run_log = _run_summary(config_default)
        assert run_log._summary_text is not None
        assert run_log._summary_text == summary


# ---------------------------------------------------------------------------
# Word-count utility unit tests
# ---------------------------------------------------------------------------


class TestWordCountUtils:
    def test_word_count_empty(self):
        assert _word_count("") == 0

    def test_word_count_simple(self):
        assert _word_count("hello world foo") == 3

    def test_truncate_no_op_when_under_limit(self):
        text = "one two three"
        assert _truncate_to_words(text, 10) == text

    def test_truncate_at_exact_limit(self):
        text = "one two three"
        assert _truncate_to_words(text, 3) == "one two three"

    def test_truncate_over_limit(self):
        text = "one two three four five"
        result = _truncate_to_words(text, 3)
        assert result == "one two three"

    def test_truncate_preserves_whole_words(self):
        text = "alpha beta gamma delta"
        result = _truncate_to_words(text, 2)
        assert result == "alpha beta"
