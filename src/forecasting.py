"""
forecasting.py
==============

Sales forecasting module for a branded-clothing store (RevInsight).

Provides three forecasting methods (Moving Average, Exponential Smoothing,
Linear Regression), automatic model selection via RMSE on a hold-out
validation set, employee-count-based forecast adjustment, visualisation
with confidence bands, and a master report generator.

All monetary values are expressed in **Lakhs INR**.
The input ``sales_series`` is a :class:`pandas.Series` indexed by month
number (1–81).  Forecasts are produced for months 82–87 (6 months ahead).
"""
from __future__ import annotations
# ---------------------------------------------------------------------------
# Backend guard – must come before ANY other matplotlib import
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Standard / third-party imports
# ---------------------------------------------------------------------------


import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FORECAST_PERIODS: int = 6
VALIDATION_MONTHS: int = 12
DEFAULT_MA_WINDOW: int = 6


# ===================================================================== #
#  Helper utilities                                                      #
# ===================================================================== #

def _clean_series(sales_series: pd.Series) -> pd.Series:
    """Return a copy of *sales_series* with NaN values forward-filled,
    then back-filled, so that downstream models receive a contiguous
    numeric series.

    Parameters
    ----------
    sales_series : pd.Series
        Raw sales data, possibly containing NaN values.

    Returns
    -------
    pd.Series
        Cleaned series with no NaN entries.
    """
    cleaned = sales_series.copy()
    if cleaned.isna().any():
        nan_count = cleaned.isna().sum()
        logger.warning(
            "Sales series contains %d NaN value(s); applying forward- "
            "then back-fill.",
            nan_count,
        )
        cleaned = cleaned.ffill().bfill()
    return cleaned


def _compute_rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Compute Root Mean Squared Error between two arrays.

    Parameters
    ----------
    actual : np.ndarray
        Ground-truth values.
    predicted : np.ndarray
        Model predictions (same length as *actual*).

    Returns
    -------
    float
        RMSE value.
    """
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def _forecast_index(sales_series: pd.Series, periods: int = FORECAST_PERIODS) -> np.ndarray:
    """Generate the month-number index for the forecast horizon.

    Parameters
    ----------
    sales_series : pd.Series
        Historical sales (used only to find the last month number).
    periods : int, optional
        Number of future months to forecast (default 6).

    Returns
    -------
    np.ndarray
        Array of future month numbers, e.g. ``[82, 83, …, 87]``.
    """
    last_month = int(sales_series.index.max())
    return np.arange(last_month + 1, last_month + 1 + periods)


# ===================================================================== #
#  1. Moving Average Forecast                                            #
# ===================================================================== #

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

    Raises
    ------
    ValueError
        If the series contains fewer data points than *window*.
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
    forecast = np.full(periods, ma_value)
    logger.info(
        "Moving Average (window=%d) forecast: %.2f Lakhs for each of %d months.",
        window,
        ma_value,
        periods,
    )
    return forecast


# ===================================================================== #
#  2. Exponential Smoothing Forecast                                     #
# ===================================================================== #

def exponential_smoothing_forecast(
    sales_series: pd.Series,
    periods: int = FORECAST_PERIODS,
) -> np.ndarray:
    """Forecast future sales using **Holt-Winters Exponential Smoothing**.

    Attempts a full Holt-Winters model (trend + seasonality) first.  If
    that fails (e.g. insufficient data for seasonal estimation), it falls
    back to simple exponential smoothing (level only).

    Parameters
    ----------
    sales_series : pd.Series
        Historical monthly sales (index = month number, values in Lakhs INR).
    periods : int, optional
        Number of future months to forecast (default 6).

    Returns
    -------
    np.ndarray
        Array of length *periods* with the forecasted values.
    """
    cleaned = _clean_series(sales_series)
    values = cleaned.values.astype(float)

    # --- Attempt 1: Holt-Winters (additive trend + additive seasonality) ---
    try:
        seasonal_period = 12  # monthly data → annual seasonality
        if len(values) < 2 * seasonal_period:
            raise ValueError(
                f"Need ≥ {2 * seasonal_period} observations for seasonal "
                f"model; have {len(values)}."
            )

        model = ExponentialSmoothing(
            values,
            trend="add",
            seasonal="add",
            seasonal_periods=seasonal_period,
        )
        fitted = model.fit(optimized=True)
        forecast = fitted.forecast(periods)
        logger.info("Holt-Winters (additive) model fitted successfully.")
        return np.array(forecast, dtype=float)

    except Exception as hw_err:  # noqa: BLE001
        logger.warning(
            "Holt-Winters failed (%s); falling back to simple exponential "
            "smoothing.",
            hw_err,
        )

    # --- Attempt 2: Simple Exponential Smoothing (level only) ---
    try:
        model = ExponentialSmoothing(values, trend=None, seasonal=None)
        fitted = model.fit(optimized=True)
        forecast = fitted.forecast(periods)
        logger.info("Simple Exponential Smoothing model fitted successfully.")
        return np.array(forecast, dtype=float)

    except Exception as ses_err:  # noqa: BLE001
        logger.error(
            "Simple Exponential Smoothing also failed (%s); returning MA "
            "fallback.",
            ses_err,
        )
        return moving_average_forecast(sales_series, periods=periods)


# ===================================================================== #
#  3. Linear Regression Forecast                                         #
# ===================================================================== #

def linear_regression_forecast(
    sales_series: pd.Series,
    periods: int = FORECAST_PERIODS,
) -> np.ndarray:
    """Forecast future sales using **Ordinary Least Squares (OLS) linear
    regression** on the month number.

    Fits ``Sales ~ β₀ + β₁ × Month`` and extrapolates forward.

    Parameters
    ----------
    sales_series : pd.Series
        Historical monthly sales (index = month number, values in Lakhs INR).
    periods : int, optional
        Number of future months to forecast (default 6).

    Returns
    -------
    np.ndarray
        Array of length *periods* with the forecasted values.
    """
    cleaned = _clean_series(sales_series)

    X_train = cleaned.index.values.reshape(-1, 1).astype(float)
    y_train = cleaned.values.astype(float)

    model = LinearRegression()
    model.fit(X_train, y_train)

    future_months = _forecast_index(sales_series, periods).reshape(-1, 1).astype(float)
    forecast = model.predict(future_months)

    logger.info(
        "Linear Regression forecast (slope=%.4f, intercept=%.4f).",
        model.coef_[0],
        model.intercept_,
    )
    return np.array(forecast, dtype=float)


# ===================================================================== #
#  4. Best-Model Selection                                               #
# ===================================================================== #

def select_best_model(
    sales_series: pd.Series,
) -> Dict[str, Any]:
    """Compare the three forecasting models on a hold-out validation set
    (last 12 months) and return the best one.

    Procedure
    ---------
    1. Split the series into *training* (all but last 12 months) and
       *validation* (last 12 months).
    2. Fit each model on the training set and forecast 12 periods.
    3. Compute RMSE against the validation set.
    4. Select the model with the lowest RMSE and re-train on the *full*
       series to produce the final 6-month forecast.

    Parameters
    ----------
    sales_series : pd.Series
        Historical monthly sales (index = month number, values in Lakhs INR).

    Returns
    -------
    dict
        Keys:

        * ``model_name``      – str, name of the winning model.
        * ``forecast_values``  – np.ndarray of 6 forecast values.
        * ``rmse``             – float, validation RMSE of the winning model.
        * ``reasoning``        – str, human-readable explanation.
        * ``all_rmse``         – dict mapping model name → RMSE.
    """
    cleaned = _clean_series(sales_series)

    # Determine usable validation size
    val_size = min(VALIDATION_MONTHS, len(cleaned) // 2)
    if val_size < 2:
        logger.warning(
            "Series too short (%d points) for meaningful validation; "
            "defaulting to Moving Average.",
            len(cleaned),
        )
        forecast = moving_average_forecast(sales_series)
        return {
            "model_name": "Moving Average",
            "forecast_values": forecast,
            "rmse": np.nan,
            "reasoning": (
                "Series too short for train/validation split; defaulted to "
                "Moving Average."
            ),
            "all_rmse": {},
        }

    train = cleaned.iloc[:-val_size]
    val_actual = cleaned.iloc[-val_size:].values.astype(float)

    # ---- Evaluate each model on the hold-out set ----
    models: Dict[str, Any] = {}

    # Moving Average
    try:
        ma_pred = moving_average_forecast(train, periods=val_size)
        models["Moving Average"] = _compute_rmse(val_actual, ma_pred)
    except Exception as exc:
        logger.warning("MA evaluation failed: %s", exc)
        models["Moving Average"] = np.inf

    # Exponential Smoothing
    try:
        es_pred = exponential_smoothing_forecast(train, periods=val_size)
        models["Exponential Smoothing"] = _compute_rmse(val_actual, es_pred)
    except Exception as exc:
        logger.warning("ES evaluation failed: %s", exc)
        models["Exponential Smoothing"] = np.inf

    # Linear Regression
    try:
        lr_pred = linear_regression_forecast(train, periods=val_size)
        models["Linear Regression"] = _compute_rmse(val_actual, lr_pred)
    except Exception as exc:
        logger.warning("LR evaluation failed: %s", exc)
        models["Linear Regression"] = np.inf

    # ---- Pick the winner ----
    best_name = min(models, key=models.get)  # type: ignore[arg-type]
    best_rmse = models[best_name]

    # ---- Re-train on the full series and forecast 6 months ----
    forecast_funcs = {
        "Moving Average": moving_average_forecast,
        "Exponential Smoothing": exponential_smoothing_forecast,
        "Linear Regression": linear_regression_forecast,
    }
    best_forecast = forecast_funcs[best_name](sales_series)

    reasoning_parts: List[str] = [
        f"Validation RMSE (last {val_size} months):",
    ]
    for name, rmse_val in sorted(models.items(), key=lambda kv: kv[1]):
        tag = " ← best" if name == best_name else ""
        reasoning_parts.append(f"  • {name}: {rmse_val:.2f} Lakhs{tag}")
    reasoning_parts.append(
        f"\n'{best_name}' achieved the lowest RMSE and was selected for the "
        f"final 6-month forecast."
    )
    reasoning = "\n".join(reasoning_parts)

    logger.info("Best model: %s (RMSE=%.2f)", best_name, best_rmse)

    return {
        "model_name": best_name,
        "forecast_values": best_forecast,
        "rmse": best_rmse,
        "reasoning": reasoning,
        "all_rmse": models,
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
    shifts the forecast linearly:

        ``adjusted = base + (target − current) × productivity_per_emp``

    Parameters
    ----------
    base_forecast : np.ndarray
        Unadjusted forecast values (e.g. from :func:`select_best_model`).
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
        If *current_employees* ≤ 0 or *productivity_per_emp* < 0.

    Examples
    --------
    >>> import numpy as np
    >>> base = np.array([100, 105, 110, 115, 120, 125])
    >>> forecast_with_employee_adjustment(base, 10, 12, 2.5)
    array([105. , 110. , 115. , 120. , 125. , 130. ])
    """
    if current_employees <= 0:
        raise ValueError(
            f"current_employees must be > 0, got {current_employees}."
        )
    if productivity_per_emp < 0:
        raise ValueError(
            f"productivity_per_emp must be ≥ 0, got {productivity_per_emp}."
        )

    delta_employees = target_employees - current_employees
    adjustment = delta_employees * productivity_per_emp

    adjusted = np.array(base_forecast, dtype=float) + adjustment

    logger.info(
        "Employee adjustment: %+d employees × %.2f Lakhs/emp = %+.2f "
        "Lakhs/month shift.",
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
) -> str:
    """Plot historical sales and the 6-month forecast with confidence
    bands, then save the figure to *output_dir*.

    The confidence band is computed as ±1.96 × (std of residuals from the
    last 12 historical months), approximating a 95 % prediction interval.

    Parameters
    ----------
    sales_series : pd.Series
        Historical monthly sales (index = month number, values in Lakhs INR).
    forecast_values : np.ndarray
        Forecasted values for the next *n* months.
    model_name : str
        Name of the model (used in the plot title and filename).
    output_dir : str
        Directory where the plot image will be saved.

    Returns
    -------
    str
        Absolute path to the saved plot image.
    """
    cleaned = _clean_series(sales_series)

    forecast_months = _forecast_index(cleaned, len(forecast_values))

    # --- Confidence band (±1.96σ of recent residuals) ---
    lookback = min(12, len(cleaned))
    recent_std = float(cleaned.iloc[-lookback:].std())
    margin = 1.96 * recent_std
    upper = forecast_values + margin
    lower = forecast_values - margin

    # --- Build the figure ---
    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(
        cleaned.index,
        cleaned.values,
        color="#2c3e50",
        linewidth=1.4,
        label="Historical Sales",
    )
    ax.plot(
        forecast_months,
        forecast_values,
        color="#e74c3c",
        linewidth=2.0,
        marker="o",
        markersize=5,
        label=f"{model_name} Forecast",
    )
    ax.fill_between(
        forecast_months,
        lower,
        upper,
        color="#e74c3c",
        alpha=0.15,
        label="95 % Confidence Band",
    )

    # Connect last historical point to first forecast point visually
    ax.plot(
        [cleaned.index[-1], forecast_months[0]],
        [cleaned.values[-1], forecast_values[0]],
        color="#e74c3c",
        linewidth=1.2,
        linestyle="--",
    )

    ax.set_title(
        f"RevInsight — 6-Month Sales Forecast ({model_name})",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("Month Number", fontsize=11)
    ax.set_ylabel("Sales (Lakhs INR)", fontsize=11)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()

    # --- Save ---
    os.makedirs(output_dir, exist_ok=True)
    safe_name = model_name.replace(" ", "_").lower()
    filepath = os.path.join(output_dir, f"forecast_{safe_name}.png")
    fig.savefig(filepath, dpi=150)
    plt.close(fig)

    logger.info("Forecast plot saved to %s", filepath)
    return os.path.abspath(filepath)


# ===================================================================== #
#  7. Master Report Generator                                            #
# ===================================================================== #

def generate_forecast_report(
    sales_series: pd.Series,
    output_dir: str,
) -> Dict[str, Any]:
    """Run **all** forecasting models, select the best one, generate
    plots, and return a comprehensive results dictionary.

    This is the main entry-point for end-to-end forecasting.

    Parameters
    ----------
    sales_series : pd.Series
        Historical monthly sales (index = month number, values in Lakhs INR).
    output_dir : str
        Directory where plots and any artefacts will be saved.

    Returns
    -------
    dict
        Comprehensive report with the following top-level keys:

        * ``historical_summary``  – dict of descriptive statistics.
        * ``models``              – dict of per-model results (forecast
          values + RMSE).
        * ``best_model``          – output of :func:`select_best_model`.
        * ``forecast_months``     – list of forecasted month numbers.
        * ``plots``               – list of file paths to saved plots.
    """
    cleaned = _clean_series(sales_series)

    # ------------------------------------------------------------------ #
    #  Historical summary                                                 #
    # ------------------------------------------------------------------ #
    historical_summary: Dict[str, Any] = {
        "num_months": len(cleaned),
        "first_month": int(cleaned.index.min()),
        "last_month": int(cleaned.index.max()),
        "mean_sales": float(cleaned.mean()),
        "median_sales": float(cleaned.median()),
        "std_sales": float(cleaned.std()),
        "min_sales": float(cleaned.min()),
        "max_sales": float(cleaned.max()),
        "total_sales": float(cleaned.sum()),
    }

    # ------------------------------------------------------------------ #
    #  Individual model forecasts                                         #
    # ------------------------------------------------------------------ #
    forecast_months = _forecast_index(cleaned).tolist()

    model_results: Dict[str, Dict[str, Any]] = {}
    plot_paths: List[str] = []

    # Moving Average
    try:
        ma_fc = moving_average_forecast(sales_series)
        model_results["Moving Average"] = {
            "forecast_values": ma_fc.tolist(),
        }
        path = plot_forecast(sales_series, ma_fc, "Moving Average", output_dir)
        plot_paths.append(path)
    except Exception as exc:
        logger.error("Moving Average forecast failed: %s", exc)
        model_results["Moving Average"] = {"error": str(exc)}

    # Exponential Smoothing
    try:
        es_fc = exponential_smoothing_forecast(sales_series)
        model_results["Exponential Smoothing"] = {
            "forecast_values": es_fc.tolist(),
        }
        path = plot_forecast(
            sales_series, es_fc, "Exponential Smoothing", output_dir
        )
        plot_paths.append(path)
    except Exception as exc:
        logger.error("Exponential Smoothing forecast failed: %s", exc)
        model_results["Exponential Smoothing"] = {"error": str(exc)}

    # Linear Regression
    try:
        lr_fc = linear_regression_forecast(sales_series)
        model_results["Linear Regression"] = {
            "forecast_values": lr_fc.tolist(),
        }
        path = plot_forecast(
            sales_series, lr_fc, "Linear Regression", output_dir
        )
        plot_paths.append(path)
    except Exception as exc:
        logger.error("Linear Regression forecast failed: %s", exc)
        model_results["Linear Regression"] = {"error": str(exc)}

    # ------------------------------------------------------------------ #
    #  Best-model selection                                               #
    # ------------------------------------------------------------------ #
    best = select_best_model(sales_series)

    # Add RMSE to individual model results
    for name, rmse_val in best.get("all_rmse", {}).items():
        if name in model_results and "error" not in model_results[name]:
            model_results[name]["rmse"] = rmse_val

    # Plot the best model separately (labelled "Best")
    try:
        best_path = plot_forecast(
            sales_series,
            best["forecast_values"],
            f"Best — {best['model_name']}",
            output_dir,
        )
        plot_paths.append(best_path)
    except Exception as exc:
        logger.error("Best-model plot failed: %s", exc)

    # ------------------------------------------------------------------ #
    #  Assemble report                                                    #
    # ------------------------------------------------------------------ #
    report: Dict[str, Any] = {
        "historical_summary": historical_summary,
        "models": model_results,
        "best_model": {
            "model_name": best["model_name"],
            "forecast_values": best["forecast_values"].tolist(),
            "rmse": best["rmse"],
            "reasoning": best["reasoning"],
        },
        "forecast_months": forecast_months,
        "plots": plot_paths,
    }

    logger.info(
        "Forecast report generated — best model: %s (RMSE %.2f).",
        best["model_name"],
        best["rmse"] if not np.isnan(best["rmse"]) else 0.0,
    )

    return report


# ===================================================================== #
#  CLI convenience                                                       #
# ===================================================================== #

if __name__ == "__main__":
    import json

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # --- Load sample data ---
    data_path = Path(__file__).resolve().parent.parent / "data" / "RevI-Test.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    df = pd.read_csv(data_path)
    # Normalise column names (strip whitespace)
    df.columns = df.columns.str.strip()
    sales = df.set_index("Month")["Sales"]
    sales.index = sales.index.astype(int)
    sales = sales.astype(float)

    output = Path(__file__).resolve().parent.parent / "output" / "forecasting"
    report = generate_forecast_report(sales, str(output))

    # Pretty-print the report (skip numpy arrays)
    print(json.dumps(report, indent=2, default=str))
