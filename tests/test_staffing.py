"""Tests for the staffing optimisation engine."""

from __future__ import annotations

import pandas as pd
import pytest

from store_staffing.staffing_model import (
    NEW_HIRE_RAMP_FACTORS,
    analyze_productivity,
    backtest_staffing_rules,
    design_optimal_rule,
    detect_diminishing_returns,
    estimate_marginal_productivity,
    generate_staffing_report,
    optimize_headcount,
    probabilistic_scenario_analysis,
    recommend_staffing,
    scenario_analysis,
)

FORECAST = [110.0, 115.0, 108.0, 120.0, 125.0, 118.0]


class TestProductivity:
    """Productivity analytics."""

    def test_productivity_keys(self, raw_df: pd.DataFrame) -> None:
        result = analyze_productivity(raw_df)
        for key in (
            "avg_sales_per_emp",
            "median_sales_per_emp",
            "std_sales_per_emp",
            "best_employee_count",
            "diminishing_returns_analysis",
            "consistency_metrics",
        ):
            assert key in result
        assert result["avg_sales_per_emp"] > 0
        assert 0 <= result["consistency_metrics"]["coefficient_of_variation"] < 5

    def test_diminishing_returns(self, raw_df: pd.DataFrame) -> None:
        result = detect_diminishing_returns(raw_df)
        assert isinstance(result["diminishing_detected"], bool)
        assert result["optimal_count"] >= 1
        assert "interpretation" in result


class TestMarginalProductivity:
    """Controlled marginal estimate."""

    def test_regression_produces_estimate(self, raw_df: pd.DataFrame) -> None:
        result = estimate_marginal_productivity(raw_df)
        assert result["model"] in {"first_difference", "levels_ols"}
        assert result["marginal_sales_per_employee"] is not None
        assert result["r_squared"] > 0.5
        assert result["n_observations"] > 0

    def test_margin_contribution_is_consistent(self, raw_df: pd.DataFrame) -> None:
        result = estimate_marginal_productivity(raw_df, 3.0, 0.05, 0.50)
        expected = 0.45 * result["marginal_sales_per_employee"] - 3.0
        assert result["marginal_margin_contribution"] == pytest.approx(expected, rel=1e-3)

    def test_falls_back_without_variation(self, tiny_df: pd.DataFrame) -> None:
        result = estimate_marginal_productivity(tiny_df)
        assert result["model"] == "fallback"
        assert result["marginal_sales_per_employee"] is None


class TestScenarioAnalysis:
    """The four staffing policies."""

    def test_all_scenarios_present(self) -> None:
        scenarios = scenario_analysis(FORECAST, current_employees=7)
        assert set(scenarios) == {"A", "B", "C", "D"}

    def test_scenario_a_keeps_headcount(self) -> None:
        scenarios = scenario_analysis(FORECAST, current_employees=7)
        for month in scenarios["A"]["monthly_breakdown"]:
            assert month["employees"] == 7

    def test_scenario_b_hires_immediately(self) -> None:
        scenarios = scenario_analysis(FORECAST, current_employees=7)
        for month in scenarios["B"]["monthly_breakdown"]:
            assert month["employees"] == 8

    def test_scenario_c_waits_two_months(self) -> None:
        scenarios = scenario_analysis(FORECAST, current_employees=7)
        breakdown = scenarios["C"]["monthly_breakdown"]
        assert breakdown[0]["employees"] == 7
        assert breakdown[1]["employees"] == 7
        assert breakdown[2]["employees"] == 8

    def test_ramp_increases_sales_over_time(self) -> None:
        scenarios = scenario_analysis(FORECAST, current_employees=7)
        sales = [m["sales"] for m in scenarios["B"]["monthly_breakdown"]]
        assert sales[-1] > sales[0]

    def test_scenario_a_sales_equal_forecast(self) -> None:
        scenarios = scenario_analysis(FORECAST, current_employees=7)
        sales = [m["sales"] for m in scenarios["A"]["monthly_breakdown"]]
        assert sales == pytest.approx(FORECAST)

    def test_requires_enough_forecasts(self) -> None:
        with pytest.raises(ValueError):
            scenario_analysis([100.0, 110.0], current_employees=7, horizon=6)

    def test_rejects_invalid_headcount(self) -> None:
        with pytest.raises(ValueError):
            scenario_analysis(FORECAST, current_employees=0)

    def test_marginal_productivity_changes_outcome(self) -> None:
        low = scenario_analysis(FORECAST, 7, marginal_sales_per_employee=2.0)
        high = scenario_analysis(FORECAST, 7, marginal_sales_per_employee=25.0)
        assert high["B"]["total_gross_margin_6m"] > low["B"]["total_gross_margin_6m"]

    def test_probabilistic_scores_are_ordered(self) -> None:
        scenarios = scenario_analysis(FORECAST, 7)
        lower = [m * 0.9 for m in FORECAST]
        upper = [m * 1.1 for m in FORECAST]
        result = probabilistic_scenario_analysis(scenarios, lower, upper)
        assert set(result["scenarios"]) == {"A", "B", "C", "D"}
        for scenario in result["scenarios"].values():
            assert scenario["p10_margin"] <= scenario["p50_margin"]
            assert scenario["p50_margin"] <= scenario["p90_margin"]
            assert 0.0 <= scenario["prob_positive"] <= 1.0


class TestHeadcountOptimisation:
    """Headcount grid search."""

    def test_optimizer_returns_curve(self) -> None:
        result = optimize_headcount(FORECAST, current_employees=7)
        assert result["optimal_headcount"] >= 1
        assert len(result["curve"]) > 0
        assert "interpretation" in result

    def test_optimizer_picks_the_maximum_margin(self) -> None:
        result = optimize_headcount(FORECAST, current_employees=7)
        best = max(row["total_margin"] for row in result["curve"])
        chosen = next(
            row for row in result["curve"] if row["headcount"] == result["optimal_headcount"]
        )
        assert chosen["total_margin"] == pytest.approx(best)

    def test_optimizer_respects_marginal(self) -> None:
        low = optimize_headcount(FORECAST, 7, marginal_sales_per_employee=1.0)
        high = optimize_headcount(FORECAST, 7, marginal_sales_per_employee=40.0)
        assert high["optimal_headcount"] >= low["optimal_headcount"]

    def test_optimizer_handles_empty_forecast(self) -> None:
        result = optimize_headcount([], current_employees=7)
        assert result["curve"] == []


class TestRecommendation:
    """Recommendation synthesis."""

    def test_recommendation_structure(self, raw_df: pd.DataFrame) -> None:
        result = recommend_staffing(raw_df, FORECAST, current_employees=7)
        assert result["recommendation"]
        assert result["recommended_scenario"] in {"A", "B", "C", "D"}
        assert len(result["reasoning"]) >= 3
        assert result["confidence"] in {"LOW", "MEDIUM", "HIGH"}
        assert "margin_advantage_vs_do_nothing" in result["expected_impact"]

    def test_rule_design(self, raw_df: pd.DataFrame) -> None:
        result = design_optimal_rule(raw_df)
        assert result["optimal_employee_count"] >= 1
        assert "rule_description" in result
        assert result["rolling_avg_trigger"]["window_months"] == 3

    def test_rule_backtest_reports_savings(self, raw_df: pd.DataFrame) -> None:
        result = backtest_staffing_rules(raw_df)
        assert result["months_compared"] > 0
        assert 0 <= result["avoided_overstaffing_pct"] <= 100
        assert "interpretation" in result


class TestStaffingReport:
    """Master staffing report."""

    def test_report_contains_all_sections(self, raw_df: pd.DataFrame) -> None:
        report = generate_staffing_report(raw_df, FORECAST)
        for key in (
            "store_context",
            "current_situation",
            "productivity_analysis",
            "diminishing_returns",
            "scenario_comparison",
            "recommendation",
            "optimal_rule",
            "headcount_optimisation",
            "marginal_productivity",
            "rule_backtest",
        ):
            assert key in report

    def test_probabilistic_block_present_with_bounds(self, raw_df: pd.DataFrame) -> None:
        lower = [m * 0.9 for m in FORECAST]
        upper = [m * 1.1 for m in FORECAST]
        report = generate_staffing_report(
            raw_df, FORECAST, forecast_lower=lower, forecast_upper=upper
        )
        assert "probabilistic" in report
        assert report["probabilistic"]["best"] in {"A", "B", "C", "D"}


def test_ramp_factors_are_monotonic() -> None:
    assert NEW_HIRE_RAMP_FACTORS == sorted(NEW_HIRE_RAMP_FACTORS)
    assert NEW_HIRE_RAMP_FACTORS[0] < 1.0
    assert NEW_HIRE_RAMP_FACTORS[-1] == 1.0
