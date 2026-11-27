"""Shared pytest fixtures for the AI Audit and Risk Analytics Tool test suite."""

from __future__ import annotations

import datetime

import pandas as pd
import pytest
from hypothesis import settings

from src.config import Config

# Apply global Hypothesis settings for all property-based tests
settings.register_profile("default", max_examples=100)
settings.load_profile("default")


@pytest.fixture
def config_default() -> Config:
    """Return a Config instance with all default values."""
    return Config()


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Return a small valid DataFrame (10 rows) matching the internal transaction schema."""
    return pd.DataFrame(
        {
            "tx_id": [f"abc{i:09d}" for i in range(10)],
            "date": pd.to_datetime(
                [
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-04",
                    "2024-01-05",
                    "2024-01-08",
                    "2024-01-09",
                    "2024-01-10",
                    "2024-01-11",
                    "2024-01-12",
                    "2024-01-15",
                ]
            ),
            "time": [
                datetime.time(9, 0),
                datetime.time(10, 30),
                datetime.time(11, 0),
                datetime.time(14, 0),
                datetime.time(8, 0),
                datetime.time(17, 59),
                datetime.time(13, 0),
                datetime.time(15, 0),
                datetime.time(9, 45),
                datetime.time(16, 30),
            ],
            "account": ["ACC-01"] * 5 + ["ACC-02"] * 5,
            "amount": [
                1000.0,
                1500.0,
                2000.0,
                1200.0,
                1800.0,
                3000.0,
                2500.0,
                2200.0,
                1900.0,
                2800.0,
            ],
            "approver": ["APR-01"] * 5 + ["APR-02"] * 5,
            "vendor": [f"VND-{i:02d}" for i in range(10)],
            "description": [f"Transaction {i}" for i in range(10)],
        }
    )
