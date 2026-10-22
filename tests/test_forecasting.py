"""Tests for the forecasting engine."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from revinsight.forecasting import (
    MODEL_ZOO,
    bootstrap_prediction_interval,
    evaluate_model,
    exponential_smoothing_forecast,
    forecast_with_employee_adjustment,
    linear_regression_forecast,
    model_forecast_available,
    moving_average_forecast,
    naive_forecast,
    plan_backtest,
    plot_forecast,
    seasonal_naive_forecast,
    select_best_model,
)


class TestForecastModels:
    """Individual model behaviour."""

    def test_naive_repeats_last_value(self, sales_series: pd.Series) -> None:
        forecast = naive_forecast(sales_series, periods=4)
        assert forecast.shape == (4,)
        assert np.allclose(forecast, float(sales_series.iloc[-1]))

    def test_seasonal_naive_uses_previous_cycle(self, sales_series: pd.Series) -> None:
        forecast = seasonal_naive_forecast(sales_series, periods=3)
        assert forecast.shape == (3,)
        assert forecast[0] == pytest.approx(float(sales_series.iloc[-12]))

    def test_seasonal_naive_requires_full_cycle(self, tiny_df: pd.DataFrame) -> None:
        series = tiny_df.set_index("Month")["Sales"]
        with pytest.raises(ValueError):
            seasonal_naive_forecast(series, periods=2)

    def test_moving_average_is_flat(self, sales_series: pd.Series) -> None:
        forecast = moving_average_forecast(sales_series, window=6, periods=3)
        expected = float(sales_series.iloc[-6:].mean())
        assert np.allclose(forecast, expected)

    def test_exponential_smoothing_returns_finite_values(self, sales_series: pd.Series) -> None:
        forecast = exponential_smoothing_forecast(sales_series, periods=6)
        assert forecast.shape == (6,)
        assert np.all(np.isfinite(forecast))

    def test_exponential_smoothing_falls_back_on_short_series(self, tiny_df: pd.DataFrame) -> None:
        series = tiny_df.set_index("Month")["Sales"]
        forecast = exponential_smoothing_forecast(series, periods=2)
        assert forecast.shape == (2,)
        assert np.all(np.isfinite(forecast))

    def test_linear_regression_trend_is_positive(self, sales_series: pd.Series) -> None:
        forecast = linear_regression_forecast(sales_series, periods=3)
        assert forecast.shape == (3,)
        assert np.all(np.isfinite(forecast))

    def test_employee_adjustment_shifts_forecast(self) -> None:
        base = np.array([100.0, 105.0, 110.0])
        adjusted = forecast_with_employee_adjustment(base, 10, 12, 2.5)
        assert np.allclose(adjusted, [105.0, 110.0, 115.0])

    def test_employee_adjustment_rejects_invalid_input(self) -> None:
        with pytest.raises(ValueError):
            forecast_with_employee_adjustment(np.array([1.0]), 0, 1, 1.0)
        with pytest.raises(ValueError):
            forecast_with_employee_adjustment(np.array([1.0]), 1, 2, -1.0)


class TestBacktestPlanning:
    """Walk-forward validation planning."""

    def test_plan_covers_validation_window(self) -> None:
        count, origins = plan_backtest(81, 6, max_origins=4)
        assert count == 4
        assert origins == [57, 63, 69, 75]
        assert origins[-1] + 6 == 81

    def test_plan_degrades_for_short_series(self) -> None:
        count, origins = plan_backtest(20, 6, max_origins=4)
        assert count >= 1
        assert all(0 < o < 20 for o in origins)

    def test_plan_handles_empty_series(self) -> None:
        count, origins = plan_backtest(0, 6)
        assert count == 0
        assert origins == []


class TestModelEvaluation:
    """Per-model scoring."""

    def test_evaluate_model_produces_metrics(self, sales_series: pd.Series) -> None:
        _, origins = plan_backtest(len(sales_series), 6, max_origins=4)
        result = evaluate_model(MODEL_ZOO[0], sales_series, origins, 6)
        assert result["model"] == "Naive"
        assert result["n_predictions"] > 0
        assert result["rmse"] > 0
        assert result["mape"] >= 0
        assert len(result["per_step_rmse"]) > 0

    def test_evaluate_model_handles_missing_history(self, sales_series: pd.Series) -> None:
        result = evaluate_model(MODEL_ZOO[3], sales_series, [2], 6)
        assert result["n_predictions"] == 0
        assert result["error"]


class TestModelSelection:
    """End-to-end model selection and ensembling."""

    def test_select_best_model_returns_full_payload(self, sales_series: pd.Series) -> None:
        result = select_best_model(sales_series, horizon=6, max_origins=4)
        assert result["model_name"]
        assert len(result["forecast_values"]) == 6
        assert len(result["lower_bound"]) == 6
        assert len(result["upper_bound"]) == 6
        assert result["ensemble_members"]
        assert result["methodology"]
        assert result["backtest"]

    def test_ensemble_beats_or_matches_best_member(self, sales_series: pd.Series) -> None:
        result = select_best_model(sales_series, horizon=6, max_origins=4)
        member_rmse = min(m["rmse"] for m in result["ensemble_members"])
        assert result["rmse"] <= member_rmse * 1.001

    def test_prediction_interval_brackets_the_forecast(self, sales_series: pd.Series) -> None:
        result = select_best_model(sales_series, horizon=6)
        forecast = result["forecast_values"]
        assert np.all(result["lower_bound"] <= forecast)
        assert np.all(result["upper_bound"] >= forecast)

    def test_weights_sum_to_one(self, sales_series: pd.Series) -> None:
        result = select_best_model(sales_series, horizon=6)
        assert sum(m["weight"] for m in result["ensemble_members"]) == pytest.approx(1.0)

    def test_short_series_falls_back(self, tiny_df: pd.DataFrame) -> None:
        series = tiny_df.set_index("Month")["Sales"]
        result = select_best_model(series, horizon=2)
        assert result["model_name"] == "Moving Average"
        assert len(result["forecast_values"]) == 2

    def test_model_availability_helper(self, sales_series: pd.Series) -> None:
        assert model_forecast_available(MODEL_ZOO[0], sales_series, 6) is True
        assert model_forecast_available(MODEL_ZOO[3], sales_series, 6) is True


class TestPredictionIntervals:
    """Bootstrap interval construction."""

    def test_interval_contains_forecast(self) -> None:
        forecast = np.array([100.0, 105.0, 110.0])
        lower, upper = bootstrap_prediction_interval(forecast, [1.0, -1.0, 2.0, -2.0, 0.5])
        assert np.all(lower <= forecast)
        assert np.all(upper >= forecast)

    def test_empty_residuals_falls_back(self) -> None:
        forecast = np.array([100.0, 105.0])
        lower, upper = bootstrap_prediction_interval(forecast, [])
        assert np.all(lower <= forecast)
        assert np.all(upper >= forecast)

    def test_wider_uncertainty_gives_wider_interval(self) -> None:
        forecast = np.array([100.0])
        narrow = bootstrap_prediction_interval(forecast, [0.1, -0.1])
        wide = bootstrap_prediction_interval(forecast, [10.0, -10.0])
        assert (wide[1] - wide[0]) > (narrow[1] - narrow[0])


class TestForecastPlot:
    """Chart generation."""

    def test_plot_forecast_writes_file(self, sales_series: pd.Series, tmp_path) -> None:
        forecast = moving_average_forecast(sales_series, periods=6)
        path = plot_forecast(sales_series, forecast, "Moving Average", str(tmp_path))
        assert path.endswith(".png")
        assert (tmp_path / "forecast_moving_average.png").exists()
