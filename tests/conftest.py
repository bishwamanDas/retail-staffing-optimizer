"""Shared pytest fixtures for the RevInsight test-suite."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from revinsight.utils import load_and_validate_data, prepare_monthly_summary  # noqa: E402


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Return the project root directory."""
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def data_dir(project_root: Path) -> Path:
    """Return the bundled dataset directory."""
    return project_root / "data"


@pytest.fixture(scope="session")
def sample_csv(data_dir: Path) -> Path:
    """Return the path of the primary sample dataset."""
    path = data_dir / "RevI-Test.csv"
    if not path.exists():  # pragma: no cover - only in a stripped checkout
        pytest.skip(f"sample dataset not found at {path}")
    return path


@pytest.fixture(scope="session")
def large_csv(data_dir: Path) -> Path:
    """Return the path of the large sample dataset."""
    path = data_dir / "RevI-Test-Large.csv"
    if not path.exists():  # pragma: no cover
        pytest.skip(f"large dataset not found at {path}")
    return path


@pytest.fixture(scope="session")
def raw_df(sample_csv: Path) -> pd.DataFrame:
    """Return the validated raw sample dataframe."""
    return load_and_validate_data(sample_csv)


@pytest.fixture(scope="session")
def summary_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Return the monthly summary for the sample dataset."""
    return prepare_monthly_summary(raw_df)


@pytest.fixture(scope="session")
def sales_series(raw_df: pd.DataFrame) -> pd.Series:
    """Return the monthly sales series indexed by month number."""
    series = raw_df.set_index("Month")["Sales"].astype(float)
    series.index = series.index.astype(int)
    return series


@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    """Return a small, fully synthetic 36-month dataframe.

    The generator produces a deterministic seasonal signal with a stable
    headcount so the analytics have something well behaved to chew on.
    """
    rng = np.random.default_rng(7)
    months = np.arange(1, 37)
    rows = {"Month": months, "Sales": []}
    for sp in range(1, 5):
        rows[f"sale sp{sp}"] = []

    for month in months:
        seasonal = 40.0 + 12.0 * np.sin(2 * np.pi * (month - 1) / 12.0)
        total = max(seasonal + rng.normal(0, 2.0), 5.0)
        split = total / 4.0
        rows["Sales"].append(round(total, 4))
        for sp in range(1, 5):
            rows[f"sale sp{sp}"].append(round(split * rng.uniform(0.85, 1.15), 4))

    return pd.DataFrame(rows)


@pytest.fixture
def tiny_df() -> pd.DataFrame:
    """Return a minimal two-month dataframe for edge-case testing."""
    return pd.DataFrame(
        {
            "Month": [1, 2],
            "Sales": [100.0, 110.0],
            "sale sp1": [60.0, 55.0],
            "sale sp2": [40.0, 55.0],
        }
    )
