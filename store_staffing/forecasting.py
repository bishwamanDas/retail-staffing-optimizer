"""Production-grade sales forecasting engine.

Design goals
------------
* **Honest model selection.** Every candidate model is evaluated with
  *walk-forward* (rolling-origin) cross-validation on the actual
  deployment horizon, instead of a single train/test split.
* **Model zoo.** Naive, Seasonal-Naive, Moving-Average, Holt, Holt-Winters,
  Linear Regression and a lag-aware Random Forest regressor compete
  automatically.
* **Ensembling.** The top performers are blended with inverse-RMSE weights,
  which is consistently more robust than picking a single winner.
* **Uncertainty.** Prediction intervals are derived from an empirical
  residual bootstrap, so they widen realistically with the horizon.
* **Diagnostics.** MAE, RMSE, MAPE, sMAPE, MASE and R² are reported per
  model, alongside per-horizon-step error growth.

All monetary values are expressed in **Lakhs INR**.  The input
``sales_series`` is a :class:`pandas.Series` indexed by month number.
"""

from __future__ import annotations

# --------------------------------------------------------------------------- #
# Backend guard – must come before ANY other matplotlib import                #
# --------------------------------------------------------------------------- #
import matplotlib

matplotlib.use("Agg")

import logging
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.holtwinters import ExponentialSmoothing

logger = logging.getLogger(__name__)

__all__ = [
    "FORECAST_PERIODS",
    "VALIDATION_MONTHS",
    "DEFAULT_MA_WINDOW",
    "ForecastModel",
    "MODEL_ZOO",
    "plan_backtest",
    "evaluate_model",
    "bootstrap_prediction_interval",
    "model_forecast_available",
    "moving_average_forecast",
    "exponential_smoothing_forecast",
    "linear_regression_forecast",
    "select_best_model",
    "forecast_with_employee_adjustment",
    "plot_forecast",
    "generate_forecast_report",
]

# --------------------------------------------------------------------------- #
# Constants                                                                    #
# --------------------------------------------------------------------------- #
FORECAST_PERIODS: int = 6
VALIDATION_MONTHS: int = 12
DEFAULT_MA_WINDOW: int = 6
SEASONAL_PERIOD: int = 12
MIN_TRAIN_SIZE: int = 12
ENSEMBLE_SIZE: int = 3
BOOTSTRAP_SIMULATIONS: int = 2_000


# ===================================================================== #
#  Helper utilities                                                      #
# ===================================================================== #


def _clean_series(sales_series: pd.Series) -> pd.Series:
    """Return a copy of *sales_series* with NaN values forward-filled,
    then back-filled, so that downstream models receive a contiguous
    numeric series.
    """
    cleaned = pd.Series(sales_series).astype(float).copy()
    if cleaned.isna().any():
        nan_count = int(cleaned.isna().sum())
        logger.warning(
            "Sales series contains %d NaN value(s); applying forward- then back-fill.",
            nan_count,
        )
        cleaned = cleaned.ffill().bfill()
    return cleaned


def _compute_rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Compute Root Mean Squared Error between two arrays."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if actual.size == 0:
        return float("inf")
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def _compute_mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Compute Mean Absolute Error between two arrays."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if actual.size == 0:
        return float("inf")
    return float(np.mean(np.abs(actual - predicted)))


def _compute_mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Compute Mean Absolute Percentage Error, ignoring zero actuals."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    mask = np.abs(actual) > 1e-9
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100.0)


def _compute_smape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Compute symmetric Mean Absolute Percentage Error (scale free)."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    denom = (np.abs(actual) + np.abs(predicted)) / 2.0
    mask = denom > 1e-9
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs(actual[mask] - predicted[mask]) / denom[mask]) * 100.0)


def _compute_mase(actual: np.ndarray, predicted: np.ndarray, train: np.ndarray) -> float:
    """Compute the Mean Absolute Scaled Error.

    MASE scales the model error by the in-sample error of a one-step
    naive forecast, making it comparable across series of different
    volatility.

    Parameters
    ----------
    actual, predicted : np.ndarray
        Out-of-sample actuals and predictions.
    train : np.ndarray
        Training series used to compute the naive scaling factor.

    Returns
    -------
    float
        MASE value (``nan`` when the naive benchmark is degenerate).
    """
    train = np.asarray(train, dtype=float)
    if train.size < 2:
        return float("nan")
    naive_errors = np.abs(np.diff(train))
    scale = float(np.mean(naive_errors))
    if scale <= 1e-9:
        return float("nan")
    return _compute_mae(actual, predicted) / scale


def _compute_r2(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Compute the coefficient of determination (R²)."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if actual.size < 2:
        return float("nan")
    ss_res = float(np.sum((actual - predicted) ** 2))
    ss_tot = float(np.sum((actual - actual.mean()) ** 2))
    if ss_tot <= 1e-12:
        return float("nan")
    return 1.0 - (ss_res / ss_tot)


def _forecast_index(sales_series: pd.Series, periods: int = FORECAST_PERIODS) -> np.ndarray:
    """Generate the month-number index for the forecast horizon.

    Parameters
    ----------
    sales_series : pd.Series
        Historical sales (used only to find the last month number).
    periods : int, optional
        Number of future months to forecast.

    Returns
    -------
    np.ndarray
        Array of future month numbers, e.g. ``[82, 83, ..., 87]``.
    """
    last_month = int(sales_series.index.max())
    return np.arange(last_month + 1, last_month + 1 + periods)


def _month_of_year(index_values: np.ndarray) -> np.ndarray:
    """Map 1-based sequential month indices onto calendar months (1-12)."""
    return ((np.asarray(index_values, dtype=int) - 1) % 12) + 1


# ===================================================================== #
#  1. Individual forecast models                                          #
# ===================================================================== #


def naive_forecast(sales_series: pd.Series, periods: int = FORECAST_PERIODS) -> np.ndarray:
    """Repeat the last observed value for every future period.

    Parameters
    ----------
    sales_series : pd.Series
        Historical monthly sales.
    periods : int, optional
        Number of future months.

    Returns
    -------
    np.ndarray
        Constant forecast of length *periods*.
    """
    cleaned = _clean_series(sales_series)
    return np.full(periods, float(cleaned.iloc[-1]), dtype=float)


def seasonal_naive_forecast(
    sales_series: pd.Series,
    periods: int = FORECAST_PERIODS,
    season: int = SEASONAL_PERIOD,
) -> np.ndarray:
    """Repeat the value observed one seasonal cycle ago.

    Parameters
    ----------
    sales_series : pd.Series
        Historical monthly sales.
    periods : int, optional
        Number of future months.
    season : int, optional
        Length of the seasonal cycle (default 12 months).

    Returns
    -------
    np.ndarray
        Forecast of length *periods*.

    Raises
    ------
    ValueError
        If the series is shorter than one full seasonal cycle.
    """
    cleaned = _clean_series(sales_series)
    if len(cleaned) < season:
        raise ValueError(
            f"Seasonal-naive needs at least {season} observations; " f"got {len(cleaned)}."
        )
    values = cleaned.values.astype(float)
    forecast = np.empty(periods, dtype=float)
    for i in range(periods):
        forecast[i] = values[-season + (i % season)]
    return forecast


def moving_average_forecast(
    sales_series: pd.Series,
    window: int = DEFAULT_MA_WINDOW,
    periods: int = FORECAST_PERIODS,
) -> np.ndarray:
    """Forecast future sales using a **simple moving average**.

    The forecast value for every future period equals the mean of the last
    *window* observed values.  This is a naïve baseline — it produces a
    flat forecast.

    Parameters
    ----------
    sales_series : pd.Series
        Historical monthly sales (index = month number, values in Lakhs INR).
    window : int, optional
        Number of trailing months to average (default 6).
    periods : int, optional
        Number of future months to forecast (default 6).

    Returns
    -------
    np.ndarray
        Array of length *periods* with the forecasted values.
    """
    cleaned = _clean_series(sales_series)

    if len(cleaned) < window:
        logger.warning(
            "Series length (%d) < window (%d); using full series for MA.",
            len(cleaned),
            window,
        )
        window = len(cleaned)

    ma_value = float(cleaned.iloc[-window:].mean())
    return np.full(periods, ma_value, dtype=float)


def exponential_smoothing_forecast(
    sales_series: pd.Series,
    periods: int = FORECAST_PERIODS,
    seasonal: bool = True,
) -> np.ndarray:
    r"""Forecast future sales using **Holt-Winters Exponential Smoothing**.

    Implements additive Holt-Winters formulation with state equations:
        Level:    \ell_t = \alpha(y_t - s_{t-m}) + (1-\alpha)(\ell_{t-1} + b_{t-1})
        Trend:    b_t = \beta(\ell_t - \ell_{t-1}) + (1-\beta)b_{t-1}
        Seasonal: s_t = \gamma(y_t - \ell_{t-1} - b_{t-1}) + (1-\gamma)s_{t-m}
        Forecast: \hat{y}_{t+h|t} = \ell_t + h b_t + s_{t+h-m(k+1)}

    Attempts a full Holt-Winters model (trend + seasonality) first. If
    that fails (e.g. insufficient data for seasonal estimation), it falls
    back to a damped-trend model and finally to simple exponential
    smoothing (level only).

    Parameters
    ----------
    sales_series : pd.Series
        Historical monthly sales (index = month number, values in Lakhs INR).
    periods : int, optional
        Number of future months to forecast (default 6).
    seasonal : bool, optional
        Whether to attempt a seasonal component (default ``True``).

    Returns
    -------
    np.ndarray
        Array of length *periods* with the forecasted values.
    """
    cleaned = _clean_series(sales_series)
    values = cleaned.values.astype(float)

    attempts: list[tuple[str, dict[str, Any]]] = []
    if seasonal:
        attempts.append(
            (
                "Holt-Winters (additive)",
                {
                    "trend": "add",
                    "seasonal": "add",
                    "seasonal_periods": SEASONAL_PERIOD,
                },
            )
        )
    attempts.append(("Holt (damped trend)", {"trend": "add", "damped_trend": True}))
    attempts.append(("Simple Exponential Smoothing", {"trend": None, "seasonal": None}))

    for label, kwargs in attempts:
        try:
            if kwargs.get("seasonal_periods") and len(values) < 2 * kwargs["seasonal_periods"]:
                raise ValueError(
                    f"Need >= {2 * kwargs['seasonal_periods']} observations for a "
                    f"seasonal model; have {len(values)}."
                )
            model = ExponentialSmoothing(values, **kwargs)
            fitted = model.fit(optimized=True)
            forecast = np.asarray(fitted.forecast(periods), dtype=float)
            if not np.all(np.isfinite(forecast)):
                raise ValueError("model produced non-finite forecasts")
            logger.info("%s model fitted successfully.", label)
            return forecast
        except Exception as exc:  # noqa: BLE001 - deliberate graceful degradation
            logger.warning("%s failed (%s).", label, exc)

    logger.error("All exponential-smoothing variants failed; using MA fallback.")
    return moving_average_forecast(sales_series, periods=periods)


def linear_regression_forecast(
    sales_series: pd.Series,
    periods: int = FORECAST_PERIODS,
    seasonal: bool = True,
) -> np.ndarray:
    """Forecast future sales using **Ordinary Least Squares** regression.

    Fits ``Sales ~ b0 + b1 * Month`` and, when *seasonal* is enabled, adds
    month-of-year sine/cosine harmonics to absorb the retail calendar
    pattern.

    Parameters
    ----------
    sales_series : pd.Series
        Historical monthly sales (index = month number, values in Lakhs INR).
    periods : int, optional
        Number of future months to forecast (default 6).
    seasonal : bool, optional
        Include seasonal harmonics (default ``True``).

    Returns
    -------
    np.ndarray
        Array of length *periods* with the forecasted values.
    """
    cleaned = _clean_series(sales_series)

    months = cleaned.index.values.astype(float)
    values = cleaned.values.astype(float)

    def _features(idx: np.ndarray) -> np.ndarray:
        """Build the design matrix for month indices *idx*."""
        idx = np.asarray(idx, dtype=float)
        cols = [np.ones_like(idx), idx]
        if seasonal:
            moy = _month_of_year(idx)
            angle = 2.0 * np.pi * (moy - 1) / SEASONAL_PERIOD
            cols.extend([np.sin(angle), np.cos(angle)])
        return np.column_stack(cols)

    model = LinearRegression()
    model.fit(_features(months), values)

    future_months = _forecast_index(cleaned, periods).astype(float)
    forecast = model.predict(_features(future_months))

    logger.info(
        "Linear Regression forecast (slope=%.4f, intercept=%.4f).",
        model.coef_[1] if len(model.coef_) > 1 else float("nan"),
        model.intercept_,
    )
    return np.asarray(forecast, dtype=float)


def random_forest_forecast(
    sales_series: pd.Series,
    periods: int = FORECAST_PERIODS,
    n_estimators: int = 300,
    random_state: int = 42,
) -> np.ndarray:
    """Forecast future sales with a **lag-aware Random Forest** regressor.

    Features per month: the month index, annual sine/cosine harmonics and
    the previous 1, 2, 3 and 12 months of sales.  Multi-step forecasts are
    produced recursively (each prediction becomes the next lag).

    Parameters
    ----------
    sales_series : pd.Series
        Historical monthly sales.
    periods : int, optional
        Number of future months.
    n_estimators : int, optional
        Number of trees (default 300).
    random_state : int, optional
        Seed for reproducibility.

    Returns
    -------
    np.ndarray
        Forecast of length *periods*.

    Raises
    ------
    ValueError
        If there is not enough history to build the lag matrix.
    """
    cleaned = _clean_series(sales_series)
    values = cleaned.values.astype(float)
    months = cleaned.index.values.astype(float)
    lags = (1, 2, 3, 12)

    required = max(lags) + 2
    if len(values) < required:
        raise ValueError(
            f"Random Forest needs at least {required} observations; got {len(values)}."
        )

    def _row(value: float, idx: float, history: list[float]) -> list[float]:
        """Build a feature row from the lag *history*."""
        moy = ((int(idx) - 1) % 12) + 1
        angle = 2.0 * np.pi * (moy - 1) / SEASONAL_PERIOD
        feats = [float(idx), math_sin(angle), math_cos(angle)]
        for lag in lags:
            feats.append(history[-lag] if len(history) >= lag else float(np.mean(history)))
        feats.append(value)
        return feats

    def _matrix(idx: np.ndarray, hist: list[float]) -> np.ndarray:
        """Build the design matrix for a sequence of month indices."""
        return np.array([_row(hist[i], idx[i], hist[: i + 1]) for i in range(len(hist))])

    train_idx = months
    train_hist = list(values)
    X = _matrix(train_idx, train_hist)
    y = np.asarray(values, dtype=float)

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=None,
        min_samples_leaf=3,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X, y)

    history = list(values)
    future = _forecast_index(cleaned, periods)
    preds: list[float] = []
    for m in future:
        row = np.array(_row(history[-1], float(m), history))
        pred = float(model.predict(row.reshape(1, -1))[0])
        preds.append(pred)
        history.append(pred)

    logger.info("Random Forest forecast computed recursively over %d months.", periods)
    return np.asarray(preds, dtype=float)


# Small helpers so the Random Forest feature builder reads clearly.
def math_sin(x: float) -> float:
    """Return ``sin(x)``."""
    return float(np.sin(x))


def math_cos(x: float) -> float:
    """Return ``cos(x)``."""
    return float(np.cos(x))


# ===================================================================== #
#  2. Model registry                                                     #
# ===================================================================== #


@dataclass(frozen=True)
class ForecastModel:
    """Registry entry describing a candidate forecasting model.

    Attributes
    ----------
    name : str
        Display name used in reports and the web UI.
    func : Callable
        ``func(series, periods) -> np.ndarray`` implementation.
    min_history : int
        Minimum number of observations the model needs to run.
    description : str
        One-line explanation of the modelling assumption.
    requires_seasonality : bool
        Whether the model is fundamentally seasonal.
    """

    name: str
    func: Callable[[pd.Series, int], np.ndarray]
    min_history: int = MIN_TRAIN_SIZE
    description: str = ""
    requires_seasonality: bool = False

    def forecast(self, series: pd.Series, periods: int) -> np.ndarray:
        """Run the model on *series* for *periods* steps.

        Parameters
        ----------
        series : pd.Series
            Training series.
        periods : int
            Forecast horizon.

        Returns
        -------
        np.ndarray
            Forecast values.

        Raises
        ------
        ValueError
            If the series is shorter than ``min_history``.
        """
        if len(series) < self.min_history:
            raise ValueError(
                f"{self.name} needs >= {self.min_history} observations; " f"got {len(series)}."
            )
        return np.asarray(self.func(series, periods), dtype=float)


MODEL_ZOO: tuple[ForecastModel, ...] = (
    ForecastModel(
        name="Naive",
        func=lambda s, p: naive_forecast(s, p),
        min_history=2,
        description="Last observed value carried forward.",
    ),
    ForecastModel(
        name="Seasonal Naive",
        func=lambda s, p: seasonal_naive_forecast(s, p),
        min_history=SEASONAL_PERIOD + 1,
        description="Value from the same month one year earlier.",
        requires_seasonality=True,
    ),
    ForecastModel(
        name="Moving Average",
        func=lambda s, p: moving_average_forecast(s, window=DEFAULT_MA_WINDOW, periods=p),
        min_history=4,
        description="Mean of the trailing 6 months (flat baseline).",
    ),
    ForecastModel(
        name="Holt-Winters",
        func=lambda s, p: exponential_smoothing_forecast(s, p, seasonal=True),
        min_history=2 * SEASONAL_PERIOD,
        description="Additive trend plus additive annual seasonality.",
        requires_seasonality=True,
    ),
    ForecastModel(
        name="Holt (Damped Trend)",
        func=lambda s, p: exponential_smoothing_forecast(s, p, seasonal=False),
        min_history=6,
        description="Damped additive trend without seasonality.",
    ),
    ForecastModel(
        name="Linear Regression",
        func=lambda s, p: linear_regression_forecast(s, p, seasonal=True),
        min_history=8,
        description="OLS trend with annual sine/cosine harmonics.",
    ),
    ForecastModel(
        name="Random Forest",
        func=lambda s, p: random_forest_forecast(s, p),
        min_history=16,
        description="Gradient-free ensemble on trend, harmonics and lags.",
    ),
)


# ===================================================================== #
#  3. Walk-forward cross-validation                                      #
# ===================================================================== #


def plan_backtest(
    n_obs: int,
    horizon: int,
    max_origins: int = 4,
    min_train: int = MIN_TRAIN_SIZE,
) -> tuple[int, list[int]]:
    """Design the walk-forward validation plan.

    The plan consists of *origins* — cut points in the series — each of
    which trains a model on everything before it and forecasts *horizon*
    months ahead.  The origins are laid out so that the evaluated windows
    are contiguous and end at the last observation.

    Parameters
    ----------
    n_obs : int
        Number of available observations.
    horizon : int
        Forecast horizon to evaluate (months).
    max_origins : int, optional
        Maximum number of validation origins.
    min_train : int, optional
        Minimum training length required for the earliest origin.

    Returns
    -------
    tuple of (int, list of int)
        The number of origins used and their absolute index positions.

    Examples
    --------
    >>> plan_backtest(81, 6, max_origins=4)
    (4, [57, 63, 69, 75])
    """
    if n_obs <= 0:
        return 0, []
    horizon = max(1, int(horizon))
    origins = max(1, int(max_origins))

    while origins > 1 and n_obs - origins * horizon < min_train:
        origins -= 1
    if n_obs - origins * horizon < min_train:
        horizon = max(1, (n_obs - min_train) // origins)
        while origins > 1 and n_obs - origins * horizon < min_train:
            origins -= 1

    if n_obs - origins * horizon < 2:
        logger.warning("Not enough history for walk-forward validation; using a single origin.")
        origin = max(2, n_obs - horizon)
        return 1, [origin]

    positions = [n_obs - (origins - k) * horizon for k in range(origins)]
    return len(positions), positions


def evaluate_model(
    model: ForecastModel,
    series: pd.Series,
    origins: Sequence[int],
    horizon: int,
) -> dict[str, Any]:
    """Evaluate a single model with walk-forward validation.

    Parameters
    ----------
    model : ForecastModel
        Candidate model.
    series : pd.Series
        Full historical series.
    origins : sequence of int
        Absolute cut positions produced by :func:`plan_backtest`.
    horizon : int
        Number of months forecast at each origin.

    Returns
    -------
    dict
        Keys: ``model``, ``rmse``, ``mae``, ``mape``, ``smape``, ``mase``,
        ``r2``, ``n_predictions``, ``per_step_rmse``, ``residuals``,
        ``fold_forecasts``, ``error``.
    """
    values = series.values.astype(float)
    actuals: list[float] = []
    predictions: list[float] = []
    step_actual: list[list[float]] = []
    step_pred: list[list[float]] = []
    folds: list[dict[str, Any]] = []

    for origin in origins:
        if origin <= 2 or origin >= len(values):
            continue
        train = series.iloc[:origin]
        actual = values[origin : origin + horizon]
        if actual.size == 0:
            continue
        try:
            pred = model.forecast(train, actual.size)
        except Exception as exc:  # noqa: BLE001 - a failing model is skipped
            logger.warning("Model '%s' failed at origin %d: %s", model.name, origin, exc)
            return {
                "model": model.name,
                "error": str(exc),
                "rmse": float("inf"),
                "mae": float("inf"),
                "mape": float("nan"),
                "smape": float("nan"),
                "mase": float("nan"),
                "r2": float("nan"),
                "n_predictions": 0,
                "per_step_rmse": [],
                "residuals": [],
                "fold_forecasts": [],
            }

        actuals.extend(actual.tolist())
        predictions.extend(pred.tolist())
        step_actual.append(actual.tolist())
        step_pred.append(pred.tolist())
        folds.append(
            {
                "origin_month": int(series.index[origin]),
                "train_size": int(origin),
                "actual": [float(a) for a in actual],
                "predicted": [float(p) for p in pred],
            }
        )

    if not actuals:
        return {
            "model": model.name,
            "error": "no valid evaluation folds",
            "rmse": float("inf"),
            "mae": float("inf"),
            "mape": float("nan"),
            "smape": float("nan"),
            "mase": float("nan"),
            "r2": float("nan"),
            "n_predictions": 0,
            "per_step_rmse": [],
            "residuals": [],
            "fold_forecasts": [],
        }

    act = np.asarray(actuals, dtype=float)
    pred = np.asarray(predictions, dtype=float)
    residuals = act - pred

    per_step = []
    max_steps = max(len(s) for s in step_actual)
    for step in range(max_steps):
        a_vals = [s[step] for s in step_actual if len(s) > step]
        p_vals = [s[step] for s in step_pred if len(s) > step]
        if a_vals:
            per_step.append(
                {
                    "step": step + 1,
                    "rmse": round(_compute_rmse(a_vals, p_vals), 4),
                    "mae": round(_compute_mae(a_vals, p_vals), 4),
                    "n": len(a_vals),
                }
            )

    return {
        "model": model.name,
        "description": model.description,
        "rmse": round(_compute_rmse(act, pred), 4),
        "mae": round(_compute_mae(act, pred), 4),
        "mape": round(_compute_mape(act, pred), 4),
        "smape": round(_compute_smape(act, pred), 4),
        "mase": round(_compute_mase(act, pred, series.values.astype(float)), 4),
        "r2": round(_compute_r2(act, pred), 4),
        "n_predictions": int(act.size),
        "per_step_rmse": per_step,
        "residuals": [float(r) for r in residuals],
        "fold_forecasts": folds,
        "error": None,
    }


def bootstrap_prediction_interval(
    forecast: np.ndarray,
    residuals: Sequence[float],
    confidence: float = 0.95,
    n_simulations: int = BOOTSTRAP_SIMULATIONS,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Build prediction intervals from an empirical residual bootstrap.

    The observed out-of-sample residuals are resampled with replacement and
    added to the point forecast, which yields empirically-calibrated
    intervals that widen with the forecast horizon.

    Parameters
    ----------
    forecast : np.ndarray
        Point forecast.
    residuals : sequence of float
        Out-of-sample residuals (actual - predicted).
    confidence : float, optional
        Central coverage, e.g. ``0.95``.
    n_simulations : int, optional
        Number of bootstrap draws.
    random_state : int, optional
        Seed for reproducibility.

    Returns
    -------
    tuple of (np.ndarray, np.ndarray)
        Lower and upper bounds aligned with *forecast*.

    Examples
    --------
    >>> lower, upper = bootstrap_prediction_interval(
    ...     np.array([100.0, 105.0]), [1.0, -1.0, 2.0, -2.0]
    ... )
    >>> lower.shape == upper.shape
    True
    """
    forecast = np.asarray(forecast, dtype=float)
    res = np.asarray(list(residuals), dtype=float)

    if res.size == 0:
        spread = np.abs(forecast) * 0.10
        return forecast - spread, forecast + spread

    rng = np.random.default_rng(random_state)
    draws = rng.choice(res, size=(n_simulations, forecast.size), replace=True)
    paths = forecast[None, :] + draws

    alpha = (1.0 - confidence) / 2.0
    lower = np.percentile(paths, alpha * 100.0, axis=0)
    upper = np.percentile(paths, (1.0 - alpha) * 100.0, axis=0)

    # Guarantee the interval always contains the point forecast.
    lower = np.minimum(lower, forecast)
    upper = np.maximum(upper, forecast)
    return lower, upper


# ===================================================================== #
#  4. Best-model selection & ensembling                                   #
# ===================================================================== #


def model_forecast_available(
    model: ForecastModel,
    series: pd.Series,
    periods: int = FORECAST_PERIODS,
) -> bool:
    """Check whether *model* can produce a forecast for *series*.

    Parameters
    ----------
    model : ForecastModel
        Candidate model.
    series : pd.Series
        Historical series.
    periods : int, optional
        Forecast horizon.

    Returns
    -------
    bool
        ``True`` when the model has enough history and runs without error.
    """
    try:
        model.forecast(series, periods)
    except Exception:  # noqa: BLE001 - availability probe
        return False
    return True


def _score_blended_folds(
    evaluations: Sequence[dict[str, Any]],
    weights: Sequence[float],
) -> dict[str, Any]:
    """Score the *blended* ensemble on the shared walk-forward folds.

    Each evaluation carries the per-origin predictions it produced.  This
    helper re-blends those predictions with the ensemble weights and
    measures the error of the combination, which is the honest estimate of
    how the deployed ensemble would have performed.

    Parameters
    ----------
    evaluations : sequence of dict
        Evaluation results from :func:`evaluate_model`.
    weights : sequence of float
        Ensemble weights, aligned with *evaluations*.

    Returns
    -------
    dict
        Keys: ``rmse``, ``mae``, ``mape``, ``smape``, ``r2``,
        ``n_predictions``, ``residuals``, ``folds``.
    """
    weights_arr = np.asarray(list(weights), dtype=float)
    if weights_arr.sum() <= 0:
        weights_arr = np.ones(len(evaluations), dtype=float)
    weights_arr = weights_arr / weights_arr.sum()

    # Group folds by origin so every member is blended on the same split.
    by_origin: dict[int, dict[int, dict[str, Any]]] = {}
    for idx, evaluation in enumerate(evaluations):
        for fold in evaluation.get("fold_forecasts", []):
            by_origin.setdefault(int(fold["origin_month"]), {})[idx] = fold

    actuals: list[float] = []
    predictions: list[float] = []
    residuals: list[float] = []
    blended_folds: list[dict[str, Any]] = []

    for origin in sorted(by_origin):
        members = by_origin[origin]
        usable = [i for i in members if i < len(weights_arr)]
        if not usable:
            continue
        length = min(len(members[i]["predicted"]) for i in usable)
        if length == 0:
            continue
        blend = np.zeros(length, dtype=float)
        wsum = 0.0
        for i in usable:
            w = weights_arr[i]
            blend += w * np.asarray(members[i]["predicted"][:length], dtype=float)
            wsum += w
        if wsum > 0:
            blend = blend / wsum
        actual = np.asarray(members[usable[0]]["actual"][:length], dtype=float)

        actuals.extend(actual.tolist())
        predictions.extend(blend.tolist())
        residuals.extend((actual - blend).tolist())
        blended_folds.append(
            {
                "origin_month": origin,
                "actual": [float(a) for a in actual],
                "predicted": [float(p) for p in blend],
            }
        )

    if not actuals:
        return {
            "rmse": float("nan"),
            "mae": float("nan"),
            "mape": float("nan"),
            "smape": float("nan"),
            "r2": float("nan"),
            "n_predictions": 0,
            "residuals": [],
            "folds": [],
        }

    act = np.asarray(actuals, dtype=float)
    pred = np.asarray(predictions, dtype=float)
    return {
        "rmse": round(_compute_rmse(act, pred), 4),
        "mae": round(_compute_mae(act, pred), 4),
        "mape": round(_compute_mape(act, pred), 4),
        "smape": round(_compute_smape(act, pred), 4),
        "r2": round(_compute_r2(act, pred), 4),
        "n_predictions": int(act.size),
        "residuals": residuals,
        "folds": blended_folds,
    }


def select_best_model(
    sales_series: pd.Series,
    horizon: int = FORECAST_PERIODS,
    max_origins: int = 4,
    ensemble_size: int = ENSEMBLE_SIZE,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Select the best forecasting model with walk-forward validation.

    Procedure
    ---------
    1. Clean the series.
    2. Plan a set of walk-forward origins (see :func:`plan_backtest`).
    3. Score every model in :data:`MODEL_ZOO` on pooled out-of-sample
       errors across all origins.
    4. Blend the top ``ensemble_size`` models with inverse-RMSE weights.
    5. Re-fit the ensemble members on the full series and produce the
       final forecast plus bootstrap prediction intervals.

    Parameters
    ----------
    sales_series : pd.Series
        Historical monthly sales (index = month number, values in Lakhs INR).
    horizon : int, optional
        Forecast horizon in months.
    max_origins : int, optional
        Maximum number of walk-forward origins.
    ensemble_size : int, optional
        Number of models blended into the ensemble.
    confidence : float, optional
        Central coverage of the prediction interval.

    Returns
    -------
    dict
        Keys: ``model_name``, ``forecast_values``, ``rmse``, ``reasoning``,
        ``all_rmse``, ``ensemble_members``, ``lower_bound``, ``upper_bound``,
        ``model_metrics``, ``backtest``, ``residual_std``, ``methodology``.
    """
    cleaned = _clean_series(sales_series)
    n_obs = len(cleaned)

    if n_obs < MIN_TRAIN_SIZE:
        logger.warning(
            "Series too short (%d points) for walk-forward validation; "
            "defaulting to Moving Average.",
            n_obs,
        )
        forecast = moving_average_forecast(cleaned, periods=horizon)
        return {
            "model_name": "Moving Average",
            "forecast_values": forecast,
            "rmse": float("nan"),
            "reasoning": (
                "Series too short for walk-forward validation; defaulted to " "Moving Average."
            ),
            "all_rmse": {},
            "ensemble_members": [],
            "lower_bound": forecast * 0.9,
            "upper_bound": forecast * 1.1,
            "model_metrics": [],
            "backtest": [],
            "residual_std": float("nan"),
            "methodology": "Insufficient history for walk-forward validation.",
        }

    n_origins, origins = plan_backtest(n_obs, horizon, max_origins=max_origins)
    logger.info(
        "Walk-forward plan: %d origin(s) at %s (horizon=%d months).",
        n_origins,
        origins,
        horizon,
    )

    evaluations: list[dict[str, Any]] = []
    for model in MODEL_ZOO:
        result = evaluate_model(model, cleaned, origins, horizon)
        evaluations.append(result)
        logger.info(
            "Model '%s': RMSE=%.3f MAE=%.3f MAPE=%.2f%%",
            model.name,
            result["rmse"],
            result["mae"],
            result["mape"],
        )

    viable = [e for e in evaluations if e["n_predictions"] > 0 and np.isfinite(e["rmse"])]
    if not viable:
        logger.error("No model produced valid forecasts; using Moving Average.")
        forecast = moving_average_forecast(cleaned, periods=horizon)
        return {
            "model_name": "Moving Average",
            "forecast_values": forecast,
            "rmse": float("nan"),
            "reasoning": "All models failed; fell back to Moving Average.",
            "all_rmse": {},
            "ensemble_members": [],
            "lower_bound": forecast * 0.9,
            "upper_bound": forecast * 1.1,
            "model_metrics": [],
            "backtest": [],
            "residual_std": float("nan"),
            "methodology": "All candidate models failed.",
        }

    ranked = sorted(viable, key=lambda e: e["rmse"])
    ranked_all = sorted(evaluations, key=lambda e: (not np.isfinite(e["rmse"]), e["rmse"]))

    # --- build the ensemble from the top performers --------------------- #
    top = ranked[: max(1, ensemble_size)]
    inv_rmse = np.array([1.0 / max(e["rmse"], 1e-6) for e in top], dtype=float)
    raw_weights = inv_rmse / inv_rmse.sum()

    members: list[dict[str, Any]] = []
    member_forecasts: list[np.ndarray] = []
    kept_evaluations: list[dict[str, Any]] = []
    kept_weights: list[float] = []
    for weight, evaluation in zip(raw_weights, top, strict=False):
        model = next(m for m in MODEL_ZOO if m.name == evaluation["model"])
        try:
            fc = model.forecast(cleaned, horizon)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Refit of '%s' failed: %s", model.name, exc)
            continue
        member_forecasts.append(np.asarray(fc, dtype=float))
        kept_evaluations.append(evaluation)
        kept_weights.append(float(weight))
        members.append(
            {
                "model": model.name,
                "weight": round(float(weight), 4),
                "rmse": evaluation["rmse"],
                "mae": evaluation["mae"],
                "mape": evaluation["mape"],
            }
        )

    if not member_forecasts:
        member_forecasts = [np.asarray(moving_average_forecast(cleaned, periods=horizon))]
        members = [{"model": "Moving Average", "weight": 1.0, "rmse": float("nan")}]

    weights = np.asarray(kept_weights, dtype=float)
    if weights.sum() <= 0:
        weights = np.ones(len(members), dtype=float)
    weights = weights / weights.sum()
    for member, weight in zip(members, weights, strict=False):
        member["weight"] = round(float(weight), 4)

    member_matrix = np.vstack(member_forecasts)
    ensemble_forecast = np.average(member_matrix, axis=0, weights=weights)

    # --- residuals for the interval ------------------------------------- #
    ensemble_residuals: list[float] = []
    for evaluation in kept_evaluations:
        ensemble_residuals.extend(evaluation.get("residuals", []))
    if not ensemble_residuals:
        ensemble_residuals = list(np.asarray(ranked[0].get("residuals", [0.0]), dtype=float))

    # --- true out-of-sample score of the *blended* ensemble -------------- #
    ensemble_metrics = _score_blended_folds(kept_evaluations, weights)
    if ensemble_metrics.get("n_predictions"):
        ensemble_residuals = list(ensemble_metrics["residuals"])

    lower, upper = bootstrap_prediction_interval(
        ensemble_forecast, ensemble_residuals, confidence=confidence
    )

    best_name = members[0]["model"] if len(members) == 1 else f"Ensemble (Top {len(members)})"
    best_rmse = float(ensemble_metrics.get("rmse", np.mean([m["rmse"] for m in members])))

    # --- backtest table (per origin, ensemble prediction) ---------------- #
    # The contract exposes scalars (the value at the end of each origin's
    # window) plus the full path, so consumers do not have to guess which
    # element of the window they are looking at.
    backtest_rows: list[dict[str, Any]] = []
    for fold in ensemble_metrics.get("folds", []):
        actual_path = [round(float(a), 4) for a in fold.get("actual", [])]
        predicted_path = [round(float(p), 4) for p in fold.get("predicted", [])]
        backtest_rows.append(
            {
                "model": best_name,
                "origin_month": fold["origin_month"],
                "actual": actual_path[-1] if actual_path else None,
                "predicted": predicted_path[-1] if predicted_path else None,
                "error": (
                    round(actual_path[-1] - predicted_path[-1], 4)
                    if actual_path and predicted_path
                    else None
                ),
                "actual_path": actual_path,
                "predicted_path": predicted_path,
                "steps": len(actual_path),
            }
        )
    backtest_rows.sort(key=lambda r: r["origin_month"])

    reasoning_parts: list[str] = [
        f"Walk-forward validation ({n_origins} origin(s), {horizon}-month horizon):",
    ]
    for evaluation in ranked_all:
        if not np.isfinite(evaluation["rmse"]):
            reasoning_parts.append(f"  • {evaluation['model']}: unavailable")
            continue
        tag = " ← selected" if evaluation["model"] == top[0]["model"] else ""
        reasoning_parts.append(
            f"  • {evaluation['model']}: RMSE {evaluation['rmse']:.2f} L, "
            f"MAPE {evaluation['mape']:.1f} %{tag}"
        )
    if len(members) > 1:
        weight_desc = ", ".join(f"{m['model']} ({m['weight'] * 100:.0f}%)" for m in members)
        reasoning_parts.append(
            f"\nEnsemble of the top {len(members)} models with inverse-RMSE "
            f"weights: {weight_desc}."
        )

    method_desc = (
        f"Rolling-origin cross-validation ({n_origins} origins x {horizon}-month "
        f"horizon); final model(s) re-fitted on all {n_obs} observations; "
        f"prediction interval from a {BOOTSTRAP_SIMULATIONS:,}-draw residual bootstrap."
    )

    return {
        "model_name": best_name,
        "forecast_values": ensemble_forecast,
        "rmse": round(best_rmse, 4),
        "reasoning": "\n".join(reasoning_parts),
        "all_rmse": {e["model"]: e["rmse"] for e in ranked_all},
        "ensemble_members": members,
        "lower_bound": lower,
        "upper_bound": upper,
        "model_metrics": ranked_all,
        "backtest": backtest_rows,
        "residual_std": round(float(np.std(ensemble_residuals)), 4),
        "methodology": method_desc,
        "n_origins": n_origins,
        "ensemble_metrics": {
            "rmse": ensemble_metrics.get("rmse"),
            "mae": ensemble_metrics.get("mae"),
            "mape": ensemble_metrics.get("mape"),
            "smape": ensemble_metrics.get("smape"),
            "r2": ensemble_metrics.get("r2"),
            "n_predictions": ensemble_metrics.get("n_predictions", 0),
        },
        "origins": [int(cleaned.index[o]) if o < len(cleaned.index) else o for o in origins],
    }


# ===================================================================== #
#  5. Employee-Adjusted Forecast                                         #
# ===================================================================== #


def forecast_with_employee_adjustment(
    base_forecast: np.ndarray,
    current_employees: int,
    target_employees: int,
    productivity_per_emp: float,
) -> np.ndarray:
    """Adjust a base forecast proportionally based on a change in employee
    headcount.

    The adjustment assumes each employee contributes a constant average
    productivity (in Lakhs INR per month).  Adding or removing staff
    shifts the forecast linearly::

        ``adjusted = base + (target - current) x productivity_per_emp``

    Parameters
    ----------
    base_forecast : np.ndarray
        Unadjusted forecast values.
    current_employees : int
        Current number of employees.
    target_employees : int
        Planned number of employees going forward.
    productivity_per_emp : float
        Average monthly sales contribution per employee (Lakhs INR).

    Returns
    -------
    np.ndarray
        Employee-adjusted forecast of the same length as *base_forecast*.

    Raises
    ------
    ValueError
        If *current_employees* <= 0 or *productivity_per_emp* < 0.

    Examples
    --------
    >>> import numpy as np
    >>> forecast_with_employee_adjustment(np.array([100, 105]), 10, 12, 2.5)
    array([105., 110.])
    """
    if current_employees <= 0:
        raise ValueError(f"current_employees must be > 0, got {current_employees}.")
    if productivity_per_emp < 0:
        raise ValueError(f"productivity_per_emp must be >= 0, got {productivity_per_emp}.")

    delta_employees = target_employees - current_employees
    adjustment = delta_employees * productivity_per_emp
    adjusted = np.asarray(base_forecast, dtype=float) + adjustment

    logger.info(
        "Employee adjustment: %+d employees x %.2f Lakhs/emp = %+.2f Lakhs/month shift.",
        delta_employees,
        productivity_per_emp,
        adjustment,
    )
    return adjusted


# ===================================================================== #
#  6. Forecast Visualisation                                             #
# ===================================================================== #


def plot_forecast(
    sales_series: pd.Series,
    forecast_values: np.ndarray,
    model_name: str,
    output_dir: str,
    lower_bound: np.ndarray | None = None,
    upper_bound: np.ndarray | None = None,
    title_suffix: str = "",
) -> str:
    """Plot historical sales and the forecast with a prediction band.

    Parameters
    ----------
    sales_series : pd.Series
        Historical monthly sales.
    forecast_values : np.ndarray
        Forecasted values for the next *n* months.
    model_name : str
        Name of the model (used in the plot title and filename).
    output_dir : str
        Directory where the plot image will be saved.
    lower_bound : np.ndarray, optional
        Lower prediction bound.  Defaults to ±1.96σ of recent residuals.
    upper_bound : np.ndarray, optional
        Upper prediction bound.
    title_suffix : str, optional
        Extra text appended to the chart title.

    Returns
    -------
    str
        Absolute path to the saved plot image.
    """
    cleaned = _clean_series(sales_series)
    forecast_values = np.asarray(forecast_values, dtype=float)
    forecast_months = _forecast_index(cleaned, len(forecast_values))

    if lower_bound is None or upper_bound is None:
        lookback = min(12, len(cleaned))
        recent_std = float(cleaned.iloc[-lookback:].std())
        margin = 1.96 * (recent_std if np.isfinite(recent_std) else 0.0)
        lower = forecast_values - margin
        upper = forecast_values + margin
    else:
        lower = np.asarray(lower_bound, dtype=float)
        upper = np.asarray(upper_bound, dtype=float)

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(
        cleaned.index,
        cleaned.values,
        color="#3b82f6",
        linewidth=1.5,
        label="Historical Sales",
    )
    ax.fill_between(
        forecast_months,
        lower,
        upper,
        color="#f59e0b",
        alpha=0.18,
        label="95% Prediction Interval",
    )
    ax.plot(
        forecast_months,
        forecast_values,
        color="#f59e0b",
        linewidth=2.2,
        marker="o",
        markersize=5,
        label=f"{model_name} Forecast",
    )
    ax.plot(
        [cleaned.index[-1], forecast_months[0]],
        [cleaned.values[-1], forecast_values[0]],
        color="#f59e0b",
        linewidth=1.2,
        linestyle="--",
    )

    ax.set_title(
        f"Store Staffing Analyzer — Sales Forecast ({model_name}){title_suffix}",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("Month Number", fontsize=11)
    ax.set_ylabel("Sales (Lakhs INR)", fontsize=11)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    safe_name = model_name.replace(" ", "_").replace("(", "").replace(")", "").lower()
    filepath = os.path.join(output_dir, f"forecast_{safe_name}.png")
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("Forecast plot saved to %s", filepath)
    return os.path.abspath(filepath)


# ===================================================================== #
#  7. Master Report Generator                                            #
# ===================================================================== #


def generate_forecast_report(
    sales_series: pd.Series,
    output_dir: str,
    horizon: int = FORECAST_PERIODS,
    max_origins: int = 4,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Run all forecasting models, select the best one, generate plots and
    return a comprehensive results dictionary.

    Parameters
    ----------
    sales_series : pd.Series
        Historical monthly sales (index = month number, values in Lakhs INR).
    output_dir : str
        Directory where plots will be saved.
    horizon : int, optional
        Forecast horizon in months.
    max_origins : int, optional
        Maximum number of walk-forward validation origins.
    confidence : float, optional
        Central coverage of the prediction interval.

    Returns
    -------
    dict
        Top-level keys: ``historical_summary``, ``models``,
        ``best_model``, ``forecast_months``, ``plots``, ``backtest``.
    """
    cleaned = _clean_series(sales_series)

    first_value = float(cleaned.iloc[0])
    last_value = float(cleaned.iloc[-1])
    if first_value > 0 and last_value > 0:
        cagr = ((last_value / first_value) ** (1.0 / max(len(cleaned) - 1, 1)) - 1.0) * 100.0
    else:
        cagr = float("nan")

    historical_summary: dict[str, Any] = {
        "num_months": len(cleaned),
        "first_month": int(cleaned.index.min()),
        "last_month": int(cleaned.index.max()),
        "mean_sales": float(cleaned.mean()),
        "median_sales": float(cleaned.median()),
        "std_sales": float(cleaned.std()),
        "min_sales": float(cleaned.min()),
        "max_sales": float(cleaned.max()),
        "total_sales": float(cleaned.sum()),
        "cagr_monthly_pct": float(cagr),
    }

    best = select_best_model(
        cleaned,
        horizon=horizon,
        max_origins=max_origins,
        confidence=confidence,
    )

    forecast_months = _forecast_index(cleaned, horizon).tolist()
    model_results: dict[str, dict[str, Any]] = {}
    plot_paths: list[str] = []

    for evaluation in best["model_metrics"]:
        name = evaluation["model"]
        model_results[name] = {
            "forecast_values": [],
            "rmse": evaluation["rmse"],
            "mae": evaluation["mae"],
            "mape": evaluation["mape"],
            "mase": evaluation["mase"],
            "r2": evaluation["r2"],
            "description": evaluation.get("description", ""),
        }

    # Individual model plots (best effort — never fatal to the pipeline).
    for model in MODEL_ZOO:
        try:
            fc = model.forecast(cleaned, horizon)
            path = plot_forecast(cleaned, fc, model.name, output_dir)
            plot_paths.append(path)
            model_results.setdefault(model.name, {})["forecast_values"] = fc.tolist()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Plot/forecast for '%s' failed: %s", model.name, exc)

    try:
        best["model_name"].replace("(", "").replace(")", "").replace(" ", "_")
        best_path = plot_forecast(
            cleaned,
            best["forecast_values"],
            best["model_name"],
            output_dir,
            lower_bound=best["lower_bound"],
            upper_bound=best["upper_bound"],
            title_suffix=" — selected",
        )
        plot_paths.append(best_path)
    except Exception as exc:  # noqa: BLE001
        logger.error("Best-model plot failed: %s", exc)

    report: dict[str, Any] = {
        "historical_summary": historical_summary,
        "models": model_results,
        "model_candidates": len(MODEL_ZOO),
        "best_model": {
            "model_name": best["model_name"],
            "forecast_values": np.asarray(best["forecast_values"]).tolist(),
            "rmse": best["rmse"],
            "reasoning": best["reasoning"],
            "lower_bound": np.asarray(best["lower_bound"]).tolist(),
            "upper_bound": np.asarray(best["upper_bound"]).tolist(),
            "ensemble_members": best["ensemble_members"],
            "model_metrics": best.get("model_metrics", []),
            "residual_std": best["residual_std"],
            "methodology": best["methodology"],
            "ensemble_metrics": best.get("ensemble_metrics", {}),
            "n_origins": best.get("n_origins", 0),
        },
        "forecast_months": forecast_months,
        "backtest": best["backtest"],
        "plots": plot_paths,
    }

    logger.info(
        "Forecast report generated — model: %s (RMSE %.2f).",
        best["model_name"],
        best["rmse"] if np.isfinite(best["rmse"]) else 0.0,
    )
    return report


# ===================================================================== #
#  CLI convenience                                                       #
# ===================================================================== #

if __name__ == "__main__":
    import json

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    data_path = Path(__file__).resolve().parent.parent / "data" / "RevI-Test.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    df = pd.read_csv(data_path)
    df.columns = df.columns.str.strip()
    sales = df.set_index("Month")["Sales"]
    sales.index = sales.index.astype(int)
    sales = sales.astype(float)

    output = Path(__file__).resolve().parent.parent / "output" / "forecasting"
    report = generate_forecast_report(sales, str(output))

    print(json.dumps(report, indent=2, default=str))
