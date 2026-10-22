"""Tests for the seasonality, anomaly and workforce analytics."""

from __future__ import annotations

import pandas as pd
import pytest

from store_staffing.anomaly import detect_anomalies
from store_staffing.people import analyze_workforce
from store_staffing.seasonality import decompose_series, seasonal_indices, seasonality_strength
from store_staffing.utils import prepare_monthly_summary


class TestSeasonality:
    """Seasonal decomposition."""

    def test_decompose_series_returns_components(self, sales_series: pd.Series) -> None:
        result = decompose_series(sales_series, period=12)
        assert result["method"]
        assert len(result["observed"]) == len(sales_series)
        assert len(result["trend"]) == len(sales_series)
        assert len(result["seasonal"]) == len(sales_series)
        assert len(result["residual"]) == len(sales_series)

    def test_decompose_requires_enough_history(self, tiny_df: pd.DataFrame) -> None:
        series = tiny_df.set_index("Month")["Sales"]
        with pytest.raises(ValueError):
            decompose_series(series, period=12)

    def test_seasonal_indices_average_to_one(self, summary_df: pd.DataFrame) -> None:
        indices = seasonal_indices(summary_df, period=12)
        assert len(indices) == 12
        mean = sum(row["index"] for row in indices) / len(indices)
        assert mean == pytest.approx(1.0, abs=1e-6)

    def test_seasonal_indices_are_sorted_by_month(self, summary_df: pd.DataFrame) -> None:
        indices = seasonal_indices(summary_df, period=12)
        assert [row["month"] for row in indices] == list(range(1, 13))

    def test_seasonal_indices_on_short_data(self, tiny_df: pd.DataFrame) -> None:
        summary = prepare_monthly_summary(tiny_df)
        indices = seasonal_indices(summary, period=12)
        assert len(indices) == 12
        mean = sum(row["index"] for row in indices) / len(indices)
        assert mean == pytest.approx(1.0, abs=1e-6)

    def test_seasonality_strength_is_bounded(self, sales_series: pd.Series) -> None:
        result = seasonality_strength(decompose_series(sales_series, period=12))
        for key in ("seasonal", "trend", "id"):
            assert 0.0 <= result[key] <= 1.0


class TestAnomalies:
    """Anomaly detection."""

    def test_detect_returns_structure(self, summary_df: pd.DataFrame) -> None:
        result = detect_anomalies(summary_df)
        assert "anomalies" in result
        assert "n_anomalies" in result
        assert "structural_breaks" in result
        assert len(result["scores"]) == len(summary_df)

    def test_anomaly_records_have_severity(self, summary_df: pd.DataFrame) -> None:
        result = detect_anomalies(summary_df, threshold=2.0)
        for item in result["anomalies"]:
            assert item["severity"] in {"low", "medium", "high"}
            assert item["direction"] in {"spike", "dip"}

    def test_higher_threshold_finds_fewer(self, summary_df: pd.DataFrame) -> None:
        low = detect_anomalies(summary_df, threshold=1.5)["n_anomalies"]
        high = detect_anomalies(summary_df, threshold=4.0)["n_anomalies"]
        assert high <= low

    def test_handles_short_series(self, tiny_df: pd.DataFrame) -> None:
        summary = prepare_monthly_summary(tiny_df)
        result = detect_anomalies(summary)
        assert result["n_anomalies"] >= 0


class TestWorkforce:
    """Salesperson-level analytics."""

    def test_employees_are_reported(self, raw_df: pd.DataFrame) -> None:
        result = analyze_workforce(raw_df)
        assert len(result["employees"]) == 16
        for employee in result["employees"]:
            for key in (
                "id",
                "total_sales",
                "months_active",
                "first_month",
                "last_month",
                "still_active",
                "share_of_sales",
            ):
                assert key in employee

    def test_employees_sorted_by_total_sales(self, raw_df: pd.DataFrame) -> None:
        result = analyze_workforce(raw_df)
        totals = [e["total_sales"] for e in result["employees"]]
        assert totals == sorted(totals, reverse=True)

    def test_attrition_summary(self, raw_df: pd.DataFrame) -> None:
        result = analyze_workforce(raw_df)
        attrition = result["attrition"]
        assert attrition["total_leavers"] >= 0
        assert attrition["total_hires"] >= 0
        assert result["summary"]["current_headcount"] == 7

    def test_headcount_timeline_length(self, raw_df: pd.DataFrame) -> None:
        result = analyze_workforce(raw_df)
        assert len(result["headcount_timeline"]) == len(raw_df)
        assert result["headcount_timeline"][0]["employees"] >= 0

    def test_single_month_employee_is_handled(self) -> None:
        df = pd.DataFrame(
            {
                "Month": [1, 2, 3],
                "Sales": [100.0, 110.0, 120.0],
                "sale sp1": [100.0, 0.0, 0.0],
                "sale sp2": [0.0, 110.0, 120.0],
            }
        )
        result = analyze_workforce(df)
        assert result["attrition"]["total_leavers"] == 1
        assert result["summary"]["current_headcount"] == 1

    def test_empty_dataframe_is_handled(self) -> None:
        df = pd.DataFrame(columns=["Month", "Sales", "sale sp1"])
        result = analyze_workforce(df)
        assert result["employees"] == []
