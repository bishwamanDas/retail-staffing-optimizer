"""Seasonal decomposition and calendar analysis for Store Staffing Analyzer.

This module isolates the *calendar* dimension of the Store Staffing Analyzer
analytics engine.  It decomposes a monthly sales series into
trend / seasonal / residual components, derives normalised seasonal
indices per calendar month and quantifies how much of the observed
variation is driven by the time of year.

Method
------
:func:`decompose_series` tries three estimators in order:

1. **STL** (Seasonal-Trend decomposition using LOESS) with
   ``robust=True``.  This is the preferred estimator because it
   down-weights outliers and tolerates a slowly changing seasonal
   shape.
2. **Classical additive** decomposition
   (:func:`statsmodels.tsa.seasonal.seasonal_decompose`).
3. A **mean-centred seasonal-index** estimator built on a centred
   moving average — the last-resort fallback when the two
   statsmodels estimators cannot be fitted.

Seasonal estimation needs at least two full seasonal cycles, i.e.
``2 * period`` observations (24 months for the default annual
period).  Shorter series raise :class:`ValueError`.

All monetary values are denominated in **Lakhs INR**.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL, seasonal_decompose

from .utils import infer_month_of_year, month_name, safe_div

logger = logging.getLogger(__name__)

__all__ = [
    "MIN_SEASONAL_OBSERVATIONS",
    "decompose_series",
    "seasonal_indices",
    "seasonality_strength",
]

MIN_SEASONAL_OBSERVATIONS: int = 24
"""Minimum observations required for seasonal estimation (2 x 12 months)."""

_ROUND_DP: int = 6
"""Decimal places used when rounding seasonal indices."""


# --------------------------------------------------------------------------- #
#  Internal helpers                                                            #
# --------------------------------------------------------------------------- #


def _clean_sales_series(sales: pd.Series) -> pd.Series:
    """Return *sales* as a float Series without NaN or infinite values.

    Parameters
    ----------
    sales : pd.Series
        Raw monthly sales values.

    Returns
    -------
    pd.Series
        Cleaned float Series, order preserved.
    """
    if sales is None:
        raise ValueError("sales must be a pandas Series, got None.")

    cleaned = pd.Series(sales, dtype="float64")
    cleaned = cleaned.replace([np.inf, -np.inf], np.nan).dropna()
    return cleaned.astype("float64")


def _centred_moving_average(values: np.ndarray, period: int) -> np.ndarray:
    """Compute a centred moving average of width *period*.

    For an even *period* the classic ``2 x period`` weighting is used
    (half weight on the two outermost observations), which is the
    standard way of centring an even window.

    Parameters
    ----------
    values : np.ndarray
        Observations (1-D, float).
    period : int
        Window width in observations.

    Returns
    -------
    np.ndarray
        Array of the same length as *values*; entries that cannot be
        computed (the leading/trailing edges) are ``NaN``.
    """
    n = len(values)
    result = np.full(n, np.nan, dtype="float64")
    if n == 0 or period < 2:
        return result

    half = period // 2
    if period % 2 == 1:
        for i in range(half, n - half):
            result[i] = float(np.mean(values[i - half : i + half + 1]))
    else:
        for i in range(half, n - half):
            window = values[i - half : i + half + 1]
            total = float(np.sum(window)) - 0.5 * float(window[0]) - 0.5 * float(window[-1])
            result[i] = total / period
    return result


def _variance(values: np.ndarray) -> float:
    """Return the sample variance of *values* (0.0 when undefined).

    Parameters
    ----------
    values : np.ndarray
        Observations (1-D, float).

    Returns
    -------
    float
        Sample variance (``ddof=1``) or ``0.0`` for fewer than two
        observations.
    """
    array = np.asarray(values, dtype="float64")
    array = array[np.isfinite(array)]
    if array.size < 2:
        return 0.0
    return float(np.var(array, ddof=1))


def _clip01(value: float) -> float:
    """Clip *value* into the closed interval [0, 1].

    Parameters
    ----------
    value : float
        Raw score.

    Returns
    -------
    float
        Clipped score.
    """
    if not np.isfinite(value):
        return 0.0
    return float(min(1.0, max(0.0, value)))


def _resolve_sales_column(df: pd.DataFrame) -> str:
    """Return the name of the total-sales column in *df*.

    Parameters
    ----------
    df : pd.DataFrame
        Summary DataFrame.

    Returns
    -------
    str
        Exact column name.

    Raises
    ------
    ValueError
        If no ``Sales`` column can be found.
    """
    for col in df.columns:
        if str(col).strip().lower() == "sales":
            return col
    raise ValueError("No 'Sales' column found in the supplied DataFrame.")


def _resolve_month_of_year(df: pd.DataFrame) -> np.ndarray:
    """Return the calendar month (1-12) for every row of *df*.

    Uses the ``MonthOfYear`` column when present, otherwise derives it
    from the ``Month`` column with
    :func:`store_staffing.utils.infer_month_of_year`, otherwise assumes a
    January start.

    Parameters
    ----------
    df : pd.DataFrame
        Summary DataFrame.

    Returns
    -------
    np.ndarray
        Integer array of calendar month numbers, one per row.
    """
    if "MonthOfYear" in df.columns:
        return df["MonthOfYear"].astype(int).to_numpy()

    if "Month" in df.columns:
        return np.array(
            [infer_month_of_year(int(m)) for m in df["Month"].astype(int)],
            dtype=int,
        )

    return np.array(
        [(i % 12) + 1 for i in range(len(df))],
        dtype=int,
    )


def _method_order(method: str) -> list[str]:
    """Return the ordered list of decomposition estimators to try.

    Parameters
    ----------
    method : str
        Preferred estimator: ``"stl"``, ``"classical"`` or
        ``"seasonal_index"`` (aliases: ``"index"``, ``"mean"``).

    Returns
    -------
    list of str
        Estimator names, most preferred first.
    """
    key = str(method or "stl").strip().lower()
    if key in {"classical", "classical_additive", "additive"}:
        return ["classical", "stl", "seasonal_index"]
    if key in {"seasonal_index", "index", "mean", "mean_index"}:
        return ["seasonal_index", "stl", "classical"]
    if key != "stl":
        logger.warning("Unknown decomposition method %r; defaulting to 'stl'.", method)
    return ["stl", "classical", "seasonal_index"]


def _decompose_stl(values: np.ndarray, period: int) -> dict[str, Any]:
    """Decompose *values* with a robust STL fit.

    Parameters
    ----------
    values : np.ndarray
        Observations (1-D, float).
    period : int
        Seasonal period (12 for monthly data).

    Returns
    -------
    dict
        Dictionary with ``trend``, ``seasonal``, ``residual`` and
        ``observed`` lists plus the estimator name.
    """
    fit = STL(values, period=period, robust=True).fit()
    return {
        "trend": np.asarray(fit.trend, dtype="float64"),
        "seasonal": np.asarray(fit.seasonal, dtype="float64"),
        "residual": np.asarray(fit.resid, dtype="float64"),
        "observed": np.asarray(values, dtype="float64"),
        "method": "stl",
    }


def _decompose_classical(values: np.ndarray, period: int) -> dict[str, Any]:
    """Decompose *values* with a classical additive decomposition.

    Parameters
    ----------
    values : np.ndarray
        Observations (1-D, float).
    period : int
        Seasonal period (12 for monthly data).

    Returns
    -------
    dict
        Dictionary with ``trend``, ``seasonal``, ``residual`` and
        ``observed`` lists plus the estimator name.
    """
    fit = seasonal_decompose(
        values,
        model="additive",
        period=period,
        extrapolate_trend="freq",
    )
    return {
        "trend": np.asarray(fit.trend, dtype="float64"),
        "seasonal": np.asarray(fit.seasonal, dtype="float64"),
        "residual": np.asarray(fit.resid, dtype="float64"),
        "observed": np.asarray(values, dtype="float64"),
        "method": "classical",
    }


def _decompose_seasonal_index(values: np.ndarray, period: int) -> dict[str, Any]:
    """Decompose *values* with a mean-centred seasonal-index estimator.

    The trend is a centred moving average (extended to the edges with
    the nearest available value), the seasonal component is the mean
    deviation-from-trend of each seasonal position, re-centred so that
    it averages to zero over a full cycle.

    Parameters
    ----------
    values : np.ndarray
        Observations (1-D, float).
    period : int
        Seasonal period (12 for monthly data).

    Returns
    -------
    dict
        Dictionary with ``trend``, ``seasonal``, ``residual`` and
        ``observed`` lists plus the estimator name.
    """
    trend = _centred_moving_average(values, period)
    if np.all(~np.isfinite(trend)):
        # Not enough observations for a centred average: use a flat trend.
        trend = np.full(len(values), float(np.mean(values)), dtype="float64")
    else:
        trend = pd.Series(trend).ffill().bfill().to_numpy(dtype="float64")

    detrended = values - trend
    positions = np.arange(len(values)) % period
    seasonal = np.zeros(len(values), dtype="float64")
    for position in range(period):
        mask = positions == position
        if mask.any():
            seasonal[mask] = float(np.mean(detrended[mask]))
    seasonal = seasonal - float(np.mean(seasonal))

    residual = values - trend - seasonal
    return {
        "trend": trend,
        "seasonal": seasonal,
        "residual": residual,
        "observed": np.asarray(values, dtype="float64"),
        "method": "seasonal_index",
    }


# --------------------------------------------------------------------------- #
#  1. Seasonal decomposition                                                    #
# --------------------------------------------------------------------------- #


def decompose_series(
    sales: pd.Series,
    period: int = 12,
    method: str = "stl",
) -> dict[str, Any]:
    """Decompose a monthly sales series into trend, seasonal and residual.

    The estimator is chosen with *method*; if the preferred estimator
    raises, the remaining estimators are attempted in turn so that the
    function degrades gracefully instead of failing.

    Parameters
    ----------
    sales : pd.Series
        Monthly sales values in Lakhs INR.  ``NaN`` entries are dropped
        and the returned arrays correspond to the remaining observations
        in their original order.
    period : int, optional
        Length of one seasonal cycle in observations (default ``12``,
        i.e. monthly data with annual seasonality).
    method : str, optional
        Preferred estimator — ``"stl"`` (default), ``"classical"`` or
        ``"seasonal_index"``.

    Returns
    -------
    dict
        Dictionary with the following keys:

        * ``trend``     – list[float], the smooth trend component.
        * ``seasonal``  – list[float], the seasonal component.
        * ``residual``  – list[float], the unexplained remainder.
        * ``observed``  – list[float], the input observations.
        * ``method``    – str, the estimator actually used
          (``"stl"``, ``"classical"`` or ``"seasonal_index"``).
        * ``period``    – int, the seasonal period used.

    Raises
    ------
    ValueError
        If *sales* is empty, if *period* is smaller than 2, or if fewer
        than ``2 * period`` observations are available (24 months for
        the default annual period).

    Examples
    --------
    >>> import pandas as pd
    >>> sales = pd.Series([100.0 + (i % 12) * 4 for i in range(36)])
    >>> result = decompose_series(sales)
    >>> result["method"]
    'stl'
    >>> len(result["trend"])
    36
    >>> result["period"]
    12
    """
    if period < 2:
        raise ValueError(f"period must be >= 2, got {period}.")

    series = _clean_sales_series(sales)
    minimum = max(MIN_SEASONAL_OBSERVATIONS, 2 * period)
    if len(series) < minimum:
        raise ValueError(
            f"Seasonal decomposition requires at least {minimum} observations "
            f"(2 x period={period}); got {len(series)}."
        )

    values = series.to_numpy(dtype="float64")
    estimators = {
        "stl": _decompose_stl,
        "classical": _decompose_classical,
        "seasonal_index": _decompose_seasonal_index,
    }
    for estimator in _method_order(method):
        try:
            result = estimators[estimator](values, period)
        except Exception as exc:  # noqa: BLE001 - graceful degradation
            logger.warning(
                "Decomposition method %r failed (%s); trying the next one.",
                estimator,
                exc,
            )
            continue

        logger.info(
            "Decomposed %d observations with %r (period=%d).",
            len(values),
            result["method"],
            period,
        )
        return {
            "trend": [float(v) for v in result["trend"]],
            "seasonal": [float(v) for v in result["seasonal"]],
            "residual": [float(v) for v in result["residual"]],
            "observed": [float(v) for v in result["observed"]],
            "method": str(result["method"]),
            "period": int(period),
        }

    raise RuntimeError("All seasonal decomposition estimators failed for the supplied series.")


# --------------------------------------------------------------------------- #
#  2. Seasonal indices                                                         #
# --------------------------------------------------------------------------- #


def seasonal_indices(
    summary_df: pd.DataFrame,
    period: int = 12,
) -> list[dict[str, Any]]:
    """Compute normalised seasonal indices per calendar month.

    For every observation the ratio ``actual / centred moving average``
    is formed.  The mean ratio of each calendar month becomes that
    month's raw index; the twelve indices are then normalised so that
    they average to **exactly 1.0** (an index of ``1.15`` therefore
    means the month typically runs 15 % above the annual norm).

    Parameters
    ----------
    summary_df : pd.DataFrame
        Monthly summary as produced by
        :func:`store_staffing.utils.prepare_monthly_summary`.  Must contain
        a ``Sales`` column; ``MonthOfYear`` is used when available and
        derived from ``Month`` otherwise.
    period : int, optional
        Length of one seasonal cycle in observations (default ``12``).

    Returns
    -------
    list of dict
        Twelve dictionaries sorted by calendar month, each with the
        keys ``month`` (int, 1-12), ``name`` (str, e.g. ``"Jan"``),
        ``index`` (float), ``strength`` (float, 0-1), ``avg_sales``
        (float, Lakhs INR) and ``months_observed`` (int).  The twelve
        ``index`` values average to exactly 1.0.

    Examples
    --------
    >>> import pandas as pd
    >>> summary = pd.DataFrame({
    ...     "Month": list(range(1, 37)),
    ...     "MonthOfYear": [((m - 1) % 12) + 1 for m in range(1, 37)],
    ...     "Sales": [100.0 + ((m - 1) % 12) * 5 for m in range(1, 37)],
    ... })
    >>> indices = seasonal_indices(summary)
    >>> len(indices)
    12
    >>> round(sum(i["index"] for i in indices) / 12, 9)
    1.0
    >>> indices[0]["name"]
    'Jan'
    """
    if summary_df is None or len(summary_df) == 0:
        logger.warning("Empty summary supplied to seasonal_indices(); returning [].")
        return []

    sales_col = _resolve_sales_column(summary_df)
    values = (
        pd.to_numeric(summary_df[sales_col], errors="coerce").fillna(0.0).to_numpy(dtype="float64")
    )
    month_of_year = _resolve_month_of_year(summary_df)

    cma = _centred_moving_average(values, period)
    valid = np.isfinite(cma) & (cma > 0) & np.isfinite(values)

    raw_index: dict[int, float | None] = {}
    for month in range(1, 13):
        group = (month_of_year == month) & valid
        if group.any():
            mean_ratio = float(np.mean(values[group] / cma[group]))
            raw_index[month] = mean_ratio if mean_ratio > 0 else None
        else:
            raw_index[month] = None

    observed_indices = [v for v in raw_index.values() if v is not None]
    if observed_indices:
        mean_raw = float(np.mean(observed_indices))
        normalised = {
            month: (value / mean_raw if value is not None else 1.0)
            for month, value in raw_index.items()
        }
    else:
        logger.warning("No usable centred moving average; using flat indices.")
        normalised = dict.fromkeys(range(1, 13), 1.0)

    results: list[dict[str, Any]] = []
    for month in range(1, 13):
        group = (month_of_year == month) & valid
        count = int(group.sum())
        index_value = float(normalised[month])

        if count >= 2:
            idx_raw = float(raw_index[month]) if raw_index[month] is not None else 1.0
            residual = values[group] - idx_raw * cma[group]
            deseasonalised = values[group] / idx_raw
            var_resid = _variance(residual)
            var_deseasonalised = _variance(deseasonalised)
            strength = (
                1.0 - safe_div(var_resid, var_deseasonalised) if var_deseasonalised > 0 else 0.0
            )
        else:
            strength = 0.0

        month_rows = month_of_year == month
        avg_sales = float(np.mean(values[month_rows])) if month_rows.any() else 0.0
        months_observed = int(month_rows.sum())

        results.append(
            {
                "month": int(month),
                "name": month_name(month),
                "index": index_value,
                "strength": _clip01(strength),
                "avg_sales": avg_sales,
                "months_observed": months_observed,
            }
        )

    _normalise_to_unit_mean(results)
    return results


def _normalise_to_unit_mean(results: list[dict[str, Any]]) -> None:
    """Force the ``index`` values of *results* to average to exactly 1.0.

    The indices are rounded to six decimals and the rounding residue is
    folded back into the largest index so that the twelve values sum to
    12.0.

    Parameters
    ----------
    results : list of dict
        Seasonal-index dictionaries, modified in place.
    """
    if not results:
        return

    indices = [round(float(r["index"]), _ROUND_DP) for r in results]
    for _ in range(3):
        residual = 12.0 - float(np.sum(indices))
        if abs(residual) < 1e-12:
            break
        largest = int(np.argmax(indices))
        indices[largest] = round(indices[largest] + residual, _ROUND_DP)

    for entry, value in zip(results, indices, strict=False):
        entry["index"] = float(value)


# --------------------------------------------------------------------------- #
#  3. Seasonality strength                                                      #
# --------------------------------------------------------------------------- #


def seasonality_strength(decomposition: dict[str, Any]) -> dict[str, float]:
    """Quantify the seasonal, trend and idiosyncratic content of a series.

    The scores follow the standard Hyndman definitions:

    * ``seasonal`` = ``1 - Var(residual) / Var(trend + seasonal)``
    * ``trend``    = ``1 - Var(seasonal + residual) / Var(trend)``
    * ``id``       = ``Var(residual) / Var(observed)`` — the share of
      the variation that neither the trend nor the season explains.

    Every score lies in ``[0, 1]``.

    Parameters
    ----------
    decomposition : dict
        Output of :func:`decompose_series`.

    Returns
    -------
    dict
        Mapping with the keys ``"seasonal"``, ``"trend"`` and ``"id"``
        (floats in ``[0, 1]``).

    Raises
    ------
    ValueError
        If *decomposition* does not contain the ``trend``, ``seasonal``
        and ``residual`` components.

    Examples
    --------
    >>> import pandas as pd
    >>> sales = pd.Series([100.0 + (i % 12) * 4 for i in range(36)])
    >>> strength = seasonality_strength(decompose_series(sales))
    >>> sorted(strength)
    ['id', 'seasonal', 'trend']
    >>> 0.0 <= strength["seasonal"] <= 1.0
    True
    """
    if not isinstance(decomposition, dict):
        raise ValueError("decomposition must be a dictionary.")

    missing = [key for key in ("trend", "seasonal", "residual") if key not in decomposition]
    if missing:
        raise ValueError(f"decomposition is missing the following component(s): {missing}.")

    trend = np.asarray(decomposition["trend"], dtype="float64")
    seasonal = np.asarray(decomposition["seasonal"], dtype="float64")
    residual = np.asarray(decomposition["residual"], dtype="float64")
    observed = np.asarray(
        decomposition.get("observed", trend + seasonal + residual),
        dtype="float64",
    )

    var_residual = _variance(residual)
    var_seasonal_trend = _variance(seasonal + trend)
    var_seasonal_residual = _variance(seasonal + residual)
    var_trend = _variance(trend)
    var_observed = _variance(observed)

    seasonal_score = (
        1.0 - safe_div(var_residual, var_seasonal_trend) if var_seasonal_trend > 0 else 0.0
    )
    trend_score = 1.0 - safe_div(var_seasonal_residual, var_trend) if var_trend > 0 else 0.0
    id_score = safe_div(var_residual, var_observed) if var_observed > 0 else 0.0

    result = {
        "seasonal": _clip01(seasonal_score),
        "trend": _clip01(trend_score),
        "id": _clip01(id_score),
    }
    logger.info(
        "Seasonality strength — seasonal=%.3f, trend=%.3f, id=%.3f.",
        result["seasonal"],
        result["trend"],
        result["id"],
    )
    return result
