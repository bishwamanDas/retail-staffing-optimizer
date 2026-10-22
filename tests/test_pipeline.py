"""End-to-end pipeline tests and API-contract conformance."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from revinsight.config import AnalysisConfig
from revinsight.pipeline import (
    build_forecast_path,
    detect_cycle_length,
    run_analysis,
    simulate_scenarios,
)
from revinsight.utils import to_jsonable

REQUIRED_TOP_LEVEL = {
    "schema_version",
    "run_id",
    "created_at",
    "dataset",
    "assumptions",
    "executive",
    "kpis",
    "series",
    "forecast",
    "scenarios",
    "productivity",
    "sensitivity",
    "payback",
    "charts",
    "artifacts",
    "recommendation_text",
}


@pytest.fixture(scope="module")
def payload(sample_csv) -> dict:
    """Run the pipeline once for the whole module."""
    return run_analysis(
        filepath=str(sample_csv),
        output_base_dir=None,
        config=AnalysisConfig(),
        generate_artifacts=False,
    )


class TestPayloadContract:
    """The payload must match the published API contract."""

    def test_top_level_keys(self, payload: dict) -> None:
        assert REQUIRED_TOP_LEVEL.issubset(set(payload))

    def test_payload_is_json_serialisable(self, payload: dict) -> None:
        json.dumps(to_jsonable(payload), allow_nan=False)

    def test_dataset_block(self, payload: dict) -> None:
        dataset = payload["dataset"]
        assert dataset["months"] == 81
        assert dataset["salespersons"] == 16
        assert dataset["first_month"] == 1
        assert dataset["last_month"] == 81

    def test_assumptions_block(self, payload: dict) -> None:
        assumptions = payload["assumptions"]
        assert assumptions["salary"] == 3.0
        assert assumptions["commission"] == 0.05
        assert assumptions["forecast_months"] == 6

    def test_executive_block(self, payload: dict) -> None:
        executive = payload["executive"]
        assert executive["recommendation"]
        assert executive["recommended_scenario"] in {"A", "B", "C", "D"}
        assert executive["confidence"] in {"LOW", "MEDIUM", "HIGH"}
        assert executive["headline"]
        assert len(executive["reasoning"]) >= 3

    def test_kpis_are_populated(self, payload: dict) -> None:
        kpis = payload["kpis"]
        assert kpis["current_employees"] == 7
        assert kpis["avg_monthly_sales"] > 0
        assert kpis["break_even_sales_per_employee"] > 0
        assert kpis["best_forecast_model"]

    def test_series_lengths_match(self, payload: dict) -> None:
        series = payload["series"]
        months = len(series["months"])
        assert months == 81
        for key in ("sales", "employees", "gross_margin"):
            assert len(series[key]) == months
        assert len(series["seasonal_index"]) == 12

    def test_forecast_block(self, payload: dict) -> None:
        forecast = payload["forecast"]
        assert len(forecast["values"]) == 6
        assert len(forecast["lower_95"]) == 6
        assert len(forecast["upper_95"]) == 6
        assert len(forecast["months"]) == 6
        assert forecast["ensemble_members"]
        assert forecast["metrics"]
        assert forecast["methodology"]

    def test_prediction_interval_brackets_forecast(self, payload: dict) -> None:
        forecast = payload["forecast"]
        for value, lower, upper in zip(
            forecast["values"], forecast["lower_95"], forecast["upper_95"], strict=False
        ):
            assert lower <= value <= upper

    def test_scenarios_block(self, payload: dict) -> None:
        scenarios = payload["scenarios"]
        assert set(scenarios) == {"A", "B", "C", "D"}
        for scenario in scenarios.values():
            assert scenario["name"]
            assert len(scenario["monthly"]) == 6
            assert scenario["total_gross_margin"] is not None
            assert scenario["risk"]

    def test_productivity_block(self, payload: dict) -> None:
        productivity = payload["productivity"]
        assert productivity["by_headcount"]
        assert productivity["employees"]
        assert "attrition" in productivity
        assert productivity["marginal_productivity"]["marginal_sales_per_employee"] > 0

    def test_sensitivity_block(self, payload: dict) -> None:
        sensitivity = payload["sensitivity"]
        assert sensitivity["tornado"]
        assert sensitivity["monte_carlo"]["n_simulations"] > 0
        assert sensitivity["grid"]

    def test_payback_block(self, payload: dict) -> None:
        payback = payload["payback"]
        assert "months_to_payback" in payback
        assert "roi_pct" in payback

    def test_recommendation_text_is_populated(self, payload: dict) -> None:
        text = payload["recommendation_text"]
        assert len(text) > 1000
        assert "EXECUTIVE SUMMARY" in text
        assert "SCENARIO COMPARISON" in text

    def test_headline_matches_the_recommended_scenario(self, payload: dict) -> None:
        """The headline must never claim a hire when the answer is 'don't'."""
        headline = payload["executive"]["headline"]
        scenario = payload["executive"]["recommended_scenario"]
        if scenario == "A":
            assert "moving from" not in headline.lower()
            assert "heads adds" not in headline.lower()
        elif scenario == "B":
            assert "moving from" in headline.lower()

    def test_headline_never_contradicts_a_dont_hire_recommendation(self, sample_csv) -> None:
        """A high salary can flip the answer to 'don't hire'."""
        result = run_analysis(
            filepath=str(sample_csv),
            config=AnalysisConfig(salary_per_employee=9.0),
            generate_artifacts=False,
        )
        scenario = result["executive"]["recommended_scenario"]
        headline = result["executive"]["headline"]
        if scenario == "A":
            assert "heads adds" not in headline

    def test_business_assumptions_are_not_dataset_specific(self, payload: dict) -> None:
        text = payload["recommendation_text"].lower()
        assert "branded clothing" not in text
        assert "in the mall" not in text

    def test_sensitivity_grid_uses_snake_case(self, payload: dict) -> None:
        grid = payload["sensitivity"]["grid"]
        assert grid
        assert "salary" in grid[0]
        assert "gross_margin" in grid[0]
        assert "Salary" not in grid[0]

    def test_backtest_rows_are_scalar(self, payload: dict) -> None:
        for row in payload["forecast"]["backtest"]:
            assert isinstance(row["actual"], (int, float))
            assert isinstance(row["predicted"], (int, float))
            assert isinstance(row["actual_path"], list)

    def test_internal_block_is_not_leaked_in_json(self, payload: dict) -> None:
        printable = {k: v for k, v in payload.items() if not k.startswith("_")}
        assert "_internal" not in printable


class TestPipelineOptions:
    """Configuration changes must flow through the analysis."""

    def test_salary_changes_break_even(self, sample_csv) -> None:
        low = run_analysis(
            filepath=str(sample_csv),
            config=AnalysisConfig(salary_per_employee=2.0),
            generate_artifacts=False,
        )
        high = run_analysis(
            filepath=str(sample_csv),
            config=AnalysisConfig(salary_per_employee=5.0),
            generate_artifacts=False,
        )
        assert (
            low["kpis"]["break_even_sales_per_employee"]
            < high["kpis"]["break_even_sales_per_employee"]
        )

    def test_forecast_horizon_is_respected(self, sample_csv) -> None:
        result = run_analysis(
            filepath=str(sample_csv),
            config=AnalysisConfig(forecast_months=9),
            generate_artifacts=False,
        )
        assert len(result["forecast"]["values"]) == 9
        assert len(result["scenarios"]["A"]["monthly"]) == 9

    def test_start_month_shifts_seasonality(self, sample_csv) -> None:
        january = run_analysis(filepath=str(sample_csv), start_month=1, generate_artifacts=False)
        july = run_analysis(filepath=str(sample_csv), start_month=7, generate_artifacts=False)
        # The calendar-month labels always run January..December, but the
        # month-of-year assigned to each dataset month shifts.
        assert january["series"]["month_of_year"][0] == 1
        assert july["series"]["month_of_year"][0] == 7
        assert january["series"]["month_of_year"][12] == 1
        assert july["series"]["month_of_year"][12] == 7
        assert [row["name"] for row in january["series"]["seasonal_index"]] == [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ]

    def test_runs_on_large_dataset(self, large_csv) -> None:
        result = run_analysis(filepath=str(large_csv), generate_artifacts=False)
        assert result["dataset"]["months"] == 120
        assert result["kpis"]["current_employees"] > 0

    def test_dataframe_input(self, raw_df: pd.DataFrame) -> None:
        result = run_analysis(df=raw_df, generate_artifacts=False)
        assert result["dataset"]["name"] == "uploaded"
        assert result["dataset"]["months"] == 81

    def test_invalid_dataframe_raises(self) -> None:
        with pytest.raises(ValueError):
            run_analysis(df=pd.DataFrame({"a": [1]}), generate_artifacts=False)

    def test_missing_source_raises(self) -> None:
        with pytest.raises(ValueError):
            run_analysis(generate_artifacts=False)


class TestSimulator:
    """What-if simulation."""

    def test_simulation_returns_scenarios(self, payload: dict) -> None:
        result = simulate_scenarios(payload)
        assert set(result["scenarios"]) == {"A", "B", "C", "D"}
        assert result["best"] in {"A", "B", "C", "D"}
        assert result["break_even_sales_per_employee"] > 0

    def test_simulation_respects_decision_scope(self, payload: dict) -> None:
        result = simulate_scenarios(payload, decision="hold")
        assert result["best"] == "A"

    def test_sales_shock_changes_outcome(self, payload: dict) -> None:
        positive = simulate_scenarios(payload, sales_shock_pct=25.0)
        negative = simulate_scenarios(payload, sales_shock_pct=-25.0)
        assert (
            positive["scenarios"]["B"]["total_gross_margin"]
            > negative["scenarios"]["B"]["total_gross_margin"]
        )

    def test_higher_salary_reduces_margin(self, payload: dict) -> None:
        cheap = simulate_scenarios(payload, salary=2.0)
        dear = simulate_scenarios(payload, salary=6.0)
        assert (
            cheap["scenarios"]["B"]["total_gross_margin"]
            > dear["scenarios"]["B"]["total_gross_margin"]
        )

    def test_simulation_is_json_serialisable(self, payload: dict) -> None:
        json.dumps(to_jsonable(simulate_scenarios(payload)), allow_nan=False)

    def test_simulation_horizon_extends_beyond_the_forecast(self, payload: dict) -> None:
        """The simulator must honour horizons longer than the forecast.

        The model only forecasts six months, but the what-if sandbox
        offers up to 24; the extra months are filled with a seasonal-naive
        extension rather than being silently clamped.
        """
        forecast_len = len(payload["forecast"]["values"])
        assert forecast_len == 6
        result = simulate_scenarios(payload, horizon=12)
        assert len(result["scenarios"]["A"]["monthly"]) == 12
        assert "seasonal-naive" in result["note"]

    def test_simulation_horizon_is_capped_by_the_path(self, payload: dict) -> None:
        result = simulate_scenarios(payload, horizon=24)
        assert len(result["scenarios"]["A"]["monthly"]) == 24
        result = simulate_scenarios(payload, horizon=999)
        assert len(result["scenarios"]["A"]["monthly"]) <= 24

    def test_longer_horizon_changes_the_margin(self, payload: dict) -> None:
        short = simulate_scenarios(payload, horizon=6)
        long = simulate_scenarios(payload, horizon=18)
        assert (
            long["scenarios"]["A"]["total_gross_margin"]
            > short["scenarios"]["A"]["total_gross_margin"]
        )

    def test_long_horizon_has_probability_bands(self, payload: dict) -> None:
        result = simulate_scenarios(payload, horizon=12)
        scenario = result["scenarios"]["A"]
        assert scenario["p10_margin"] is not None
        assert scenario["p10_margin"] <= scenario["p50_margin"]
        assert scenario["p50_margin"] <= scenario["p90_margin"]


class TestForecastPath:
    """The seasonal-naive extension used by the simulator."""

    def test_path_length(self, sales_series: pd.Series) -> None:
        path = build_forecast_path(sales_series, [100.0, 110.0], total_months=12)
        assert len(path) == 12
        assert path[0] == 100.0
        assert path[1] == 110.0

    def test_extension_is_seasonal(self, sales_series: pd.Series) -> None:
        path = build_forecast_path(sales_series, [100.0], total_months=13)
        history = sales_series.values.astype(float)
        # The 13th entry repeats the observation one cycle earlier.
        assert path[12] == pytest.approx(history[-12])

    def test_path_handles_empty_forecast(self, sales_series: pd.Series) -> None:
        path = build_forecast_path(sales_series, [], total_months=6)
        assert len(path) == 6
        assert all(isinstance(v, float) for v in path)


class TestCycleDetection:
    """Automatic cycle-length detection."""

    def test_detects_cycle(self, sales_series: pd.Series) -> None:
        result = detect_cycle_length(sales_series)
        assert result["cycle_length"] in (3, 4, 6, 12)
        assert result["confidence"] in {"LOW", "MEDIUM", "HIGH"}

    def test_handles_short_series(self) -> None:
        result = detect_cycle_length(pd.Series([1.0, 2.0, 3.0]))
        assert result["cycle_length"] == 12
