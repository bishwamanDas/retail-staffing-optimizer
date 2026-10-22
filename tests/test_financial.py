"""Tests for the financial analytics layer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from store_staffing.financial_analysis import (
    calculate_break_even_analysis,
    calculate_contribution_margin,
    calculate_historical_margins,
    calculate_incremental_analysis,
    calculate_payback_period,
    calculate_roi,
    calculate_six_month_projection,
    calculate_summary_statistics,
    compute_gross_margin,
    estimate_vacancy_cost,
    generate_financial_report,
    run_sensitivity_analysis,
)


class TestMargins:
    """Margin computation and summary statistics."""

    def test_gross_margin_series(self, summary_df: pd.DataFrame) -> None:
        result = compute_gross_margin(summary_df["Sales"], summary_df["ActiveEmployees"])
        assert isinstance(result, pd.Series)
        assert result.iloc[0] == pytest.approx(
            0.45 * summary_df["Sales"].iloc[0] - 3.0 * summary_df["ActiveEmployees"].iloc[0]
        )

    def test_historical_margins_adds_column(self, summary_df: pd.DataFrame) -> None:
        frame = summary_df.drop(columns=["GrossMargin"])
        result = calculate_historical_margins(frame)
        assert "GrossMargin" in result.columns

    def test_historical_margins_requires_columns(self, summary_df: pd.DataFrame) -> None:
        with pytest.raises(KeyError):
            calculate_historical_margins(summary_df.drop(columns=["ActiveEmployees"]))

    def test_force_recomputes_margin(self, summary_df: pd.DataFrame) -> None:
        result = calculate_historical_margins(summary_df, salary_per_person=5.0, force=True)
        assert result["GrossMargin"].iloc[0] == pytest.approx(
            0.45 * summary_df["Sales"].iloc[0] - 5.0 * summary_df["ActiveEmployees"].iloc[0]
        )

    def test_summary_statistics_keys(self, summary_df: pd.DataFrame) -> None:
        stats = calculate_summary_statistics(summary_df)
        for key in (
            "avg_sales",
            "median_sales",
            "std_sales",
            "avg_margin",
            "total_margin",
            "min_margin",
            "max_margin",
            "avg_employees",
            "avg_sales_per_emp",
        ):
            assert key in stats
        assert stats["min_margin"] <= stats["max_margin"]
        assert stats["avg_sales"] > 0


class TestBreakEven:
    """Break-even analysis."""

    def test_break_even_values(self) -> None:
        result = calculate_break_even_analysis(3.0, 0.05)
        assert result["break_even_sales_per_employee"] == pytest.approx(6.6666667, rel=1e-6)
        assert result["effective_margin_rate"] == pytest.approx(0.45)

    def test_break_even_raises_when_margin_is_zero(self) -> None:
        with pytest.raises(ZeroDivisionError):
            calculate_break_even_analysis(3.0, 0.2, 0.8)


class TestContributionAndIncremental:
    """Headcount-level tables."""

    def test_contribution_margin_shape(self, summary_df: pd.DataFrame) -> None:
        result = calculate_contribution_margin(summary_df)
        assert "ActiveEmployees" in result.columns
        assert "AvgContributionPerEmployee" in result.columns
        assert len(result) == summary_df["ActiveEmployees"].nunique()

    def test_incremental_analysis_diff(self, summary_df: pd.DataFrame) -> None:
        result = calculate_incremental_analysis(summary_df)
        assert "IncrementalSales" in result.columns
        assert result["IncrementalSales"].iloc[0] is np.nan or pd.isna(
            result["IncrementalSales"].iloc[0]
        )
        assert result["IncrementalSales"].notna().sum() == len(result) - 1


class TestSensitivity:
    """Sensitivity grid."""

    def test_grid_contains_base_case(self) -> None:
        grid = run_sensitivity_analysis(100.0, 7, 3.0, 0.05)
        base = grid[(grid["salary"] == 3.0) & (np.isclose(grid["commission_rate"], 0.05))]
        assert len(base) == 1
        assert base["gross_margin"].iloc[0] == pytest.approx(0.45 * 100 - 21)

    def test_margin_vs_base_is_zero_for_base(self) -> None:
        grid = run_sensitivity_analysis(100.0, 7, 3.0, 0.05)
        base = grid[(grid["salary"] == 3.0) & (np.isclose(grid["commission_rate"], 0.05))]
        assert base["margin_vs_base"].iloc[0] == pytest.approx(0.0)

    def test_grid_uses_the_contract_field_names(self) -> None:
        """The grid keys are snake_case, matching the published contract."""
        grid = run_sensitivity_analysis(100.0, 7, 3.0, 0.05)
        assert set(grid.columns) == {
            "salary",
            "commission_rate",
            "effective_margin_rate",
            "gross_margin",
            "break_even_sales_per_emp",
            "margin_vs_base",
            "recommended_action",
        }


class TestProjection:
    """Six-month projection."""

    def test_projection_totals(self) -> None:
        result = calculate_six_month_projection([100.0] * 6, 7)
        assert result["total_sales"] == pytest.approx(600.0)
        assert result["total_salary_cost"] == pytest.approx(126.0)
        assert result["total_gross_margin"] == pytest.approx(6 * (45.0 - 21.0))
        assert result["break_even_month"] == 1

    def test_projection_requires_six_values(self) -> None:
        with pytest.raises(ValueError):
            calculate_six_month_projection([100.0] * 5, 7)


class TestPaybackAndRoi:
    """Payback period and ROI."""

    def test_payback_within_first_month(self) -> None:
        result = calculate_payback_period([20.0, 25.0, 30.0])
        assert result["ever_pays_back"] is True
        assert result["months_to_payback"] == pytest.approx(1.0)

    def test_payback_interpolates(self) -> None:
        # Month 1 contributes -0.60 L (5.34 L sales), then +2.00 L a month.
        # The cumulative turns positive 0.30 of the way through month 2.
        result = calculate_payback_period([5.34, 11.12, 11.12, 11.12])
        assert result["months_to_payback"] == pytest.approx(1.30, abs=0.02)

    def test_payback_never_when_unprofitable(self) -> None:
        result = calculate_payback_period([1.0, 1.0, 1.0])
        assert result["ever_pays_back"] is False
        assert result["months_to_payback"] is None

    def test_roi_calculation(self) -> None:
        result = calculate_roi(30.0, 18.0)
        assert result["net_benefit"] == pytest.approx(12.0)
        assert result["roi_pct"] == pytest.approx(66.67, rel=1e-3)

    def test_roi_handles_zero_cost(self) -> None:
        result = calculate_roi(30.0, 0.0)
        assert result["roi_pct"] == 0.0


class TestVacancyCost:
    """Cost of leaving a position unfilled."""

    def test_vacancy_cost_is_reported(self, summary_df: pd.DataFrame) -> None:
        result = estimate_vacancy_cost(summary_df, open_positions=1)
        assert result["open_positions"] == 1
        assert result["estimated_lost_sales_per_month"] > 0
        assert "interpretation" in result

    def test_two_positions_cost_more_than_one(self, summary_df: pd.DataFrame) -> None:
        one = estimate_vacancy_cost(summary_df, open_positions=1)
        two = estimate_vacancy_cost(summary_df, open_positions=2)
        assert two["estimated_margin_cost_per_month"] > one["estimated_margin_cost_per_month"]


class TestFinancialReport:
    """Master report."""

    def test_report_contains_all_sections(self, summary_df: pd.DataFrame) -> None:
        report = generate_financial_report(summary_df)
        for key in (
            "historical_margins",
            "summary_statistics",
            "break_even_analysis",
            "contribution_margin",
            "incremental_analysis",
            "sensitivity_analysis",
            "vacancy_cost",
        ):
            assert key in report
