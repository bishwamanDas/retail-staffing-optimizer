"""Robust anomaly detection and structural-break analysis for RevInsight.

The module works on the *deseasonalised* view of the store's sales so
that a genuine December spike is not mistaken for an anomaly.  When a
decomposition produced by :func:`revinsight.seasonality.decompose_series`
is available, the residuals are scored; otherwise the raw monthly sales
are scored directly.

Scoring
-------
Every observation is standardised with a **robust z-score**::

    z = (x - median(x)) / (1.4826 * MAD(x))

where ``MAD`` is the median absolute deviation.  The ``1.4826`` factor
makes the estimator consistent with the standard deviation for
Gaussian data.  When ``MAD`` is zero (a very flat series) the sample
standard deviation is used instead.

All monetary values are denominated in **Lakhs INR**.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from .utils import infer_month_of_year, month_name

logger = logging.getLogger(__name__)

__all__ = [
    "MAD_SCALE",
    "detect_anomalies",
]

MAD_SCALE: float = 1.4826
"""Consistency factor converting a MAD into a Gaussian standard deviation."""

_ROUND_DP: int = 4
"""Decimal places used when rounding monetary values."""

_TRAILING_WINDOW: int = 3
"""Window (months) of the trailing average used for structural breaks."""

_BREAK_SIGMA: float = 2.0
"""Number of standard deviations a shift must exceed to be a break."""


# --------------------------------------------------------------------------- #
#  Internal helpers                                                            #
# --------------------------------------------------------------------------- #


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
    return np.array([(i % 12) + 1 for i in range(len(df))], dtype=int)


def _resolve_month_labels(df: pd.DataFrame) -> np.ndarray:
    """Return the sequential month label (1-based) for every row of *df*.

    Parameters
    ----------
    df : pd.DataFrame
        Summary DataFrame.

    Returns
    -------
    np.ndarray
        Integer array of month labels, one per row.
    """
    if "Month" in df.columns:
        return df["Month"].astype(int).to_numpy()
    return np.arange(1, len(df) + 1, dtype=int)


def _robust_scale(values: np.ndarray) -> tuple[float, float]:
    """Return the robust centre and scale of *values*.

    The scale is ``1.4826 * MAD``; when the MAD is zero the sample
    standard deviation is used, and when that is zero too the scale
    falls back to ``1.0`` (which yields a z-score of zero everywhere and
    therefore no anomalies).

    Parameters
    ----------
    values : np.ndarray
        Observations (1-D, float).

    Returns
    -------
    tuple of float
        ``(median, scale)``.
    """
    array = np.asarray(values, dtype="float64")
    array = array[np.isfinite(array)]
    if array.size == 0:
        return 0.0, 1.0

    median = float(np.median(array))
    mad = float(np.median(np.abs(array - median)))
    scale = MAD_SCALE * mad
    if scale == 0.0:
        scale = float(np.std(array, ddof=1)) if array.size > 1 else 0.0
    if scale == 0.0 or not np.isfinite(scale):
        scale = 1.0
    return median, scale


def _severity(score: float) -> str:
    """Map an absolute robust z-score onto a severity label.

    Parameters
    ----------
    score : float
        Signed robust z-score.

    Returns
    -------
    str
        ``"low"``, ``"medium"`` or ``"high"``.
    """
    magnitude = abs(float(score))
    if magnitude < 4.0:
        return "low"
    if magnitude < 6.0:
        return "medium"
    return "high"


def _seasonal_index(
    decomposition: dict[str, Any] | None,
    month_of_year: np.ndarray,
    n: int,
) -> np.ndarray:
    """Return the multiplicative seasonal index of every observation.

    The index is derived from the decomposition itself: for each
    calendar month the mean of ``seasonal / trend`` is computed and
    turned into a multiplicative factor ``1 + mean(seasonal / trend)``.
    A factor of ``1.0`` means "no seasonal effect".

    Parameters
    ----------
    decomposition : dict or None
        Output of :func:`revinsight.seasonality.decompose_series`.
    month_of_year : np.ndarray
        Calendar month (1-12) of every observation.
    n : int
        Number of observations.

    Returns
    -------
    np.ndarray
        Array of *n* seasonal factors, each ``> 0``.
    """
    index = np.ones(n, dtype="float64")
    if not decomposition:
        return index

    trend = np.asarray(decomposition.get("trend", []), dtype="float64")
    seasonal = np.asarray(decomposition.get("seasonal", []), dtype="float64")
    if trend.size != n or seasonal.size != n:
        return index

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(np.abs(trend) > 1e-9, seasonal / trend, np.nan)

    for month in range(1, 13):
        selection = month_of_year == month
        if not selection.any():
            continue
        value = float(np.nanmean(ratio[selection]))
        if np.isfinite(value) and (1.0 + value) > 0:
            index[selection] = 1.0 + value
    return index


def _structural_breaks(
    level: np.ndarray,
    month_labels: np.ndarray,
) -> list[dict[str, Any]]:
    """Detect level shifts in *level* using a trailing moving average.

    The three-month trailing average of the deseasonalised level is
    compared with the same average of the prior window.  A break is
    flagged in month *t* when the trailing average moves by more than
    two standard deviations of the historical month-over-month changes
    of that trailing average.

    The scale is measured over the whole series rather than over the
    three observations of the prior window because a three-point window
    only provides two degrees of freedom, which flags a large share of
    months on a noisy series.

    Parameters
    ----------
    level : np.ndarray
        Deseasonalised level series (1-D, float).
    month_labels : np.ndarray
        Month label of every observation.

    Returns
    -------
    list of dict
        Each entry has the keys ``month`` (int), ``direction``
        (``"up"`` / ``"down"``) and ``change_pct`` (float).
    """
    breaks: list[dict[str, Any]] = []
    series = pd.Series(np.asarray(level, dtype="float64"))
    ma = series.rolling(window=_TRAILING_WINDOW, min_periods=_TRAILING_WINDOW).mean()
    changes = ma.diff()

    scale = float(changes.std(ddof=1)) if len(changes) > 1 else 0.0
    if not np.isfinite(scale) or scale == 0.0:
        scale = float(series.std(ddof=1)) if len(series) > 1 else 0.0
    if not np.isfinite(scale) or scale == 0.0:
        logger.warning("Flat series; no structural break can be detected.")
        return breaks

    for i in range(1, len(ma)):
        change = changes.iloc[i]
        previous = ma.iloc[i - 1]
        if pd.isna(change) or pd.isna(previous) or previous == 0:
            continue
        if abs(float(change)) <= _BREAK_SIGMA * scale:
            continue

        change_pct = float(change) / abs(float(previous)) * 100.0
        breaks.append(
            {
                "month": int(month_labels[i]),
                "direction": "up" if change > 0 else "down",
                "change_pct": round(float(change_pct), _ROUND_DP),
            }
        )

    if breaks:
        logger.info("Detected %d structural break(s).", len(breaks))
    return breaks


# --------------------------------------------------------------------------- #
#  1. Anomaly detection                                                        #
# --------------------------------------------------------------------------- #


def detect_anomalies(
    summary_df: pd.DataFrame,
    decomposition: dict[str, Any] | None = None,
    threshold: float = 3.0,
) -> dict[str, Any]:
    """Detect unusual months using a robust z-score.

    When *decomposition* is supplied (the output of
    :func:`revinsight.seasonality.decompose_series`) the **residual**
    component is scored, so recurring seasonal effects are removed
    before the outliers are looked for.  Without a decomposition the
    raw monthly sales are scored instead.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Monthly summary as produced by
        :func:`revinsight.utils.prepare_monthly_summary`.  Must contain
        a ``Sales`` column.
    decomposition : dict, optional
        Seasonal decomposition whose ``residual`` component should be
        scored.  It is ignored when its length does not match the
        number of rows in *summary_df*.
    threshold : float, optional
        Absolute robust z-score above which a month is flagged
        (default ``3.0``).

    Returns
    -------
    dict
        Dictionary with the following keys:

        * ``anomalies``        – list[dict] of flagged months with the
          keys ``month`` (int, dataset month label), ``name`` (str,
          calendar month name), ``sales`` (float, Lakhs INR),
          ``expected`` (float, the fitted value), ``score`` (float, the
          signed robust z-score), ``direction`` (``"spike"`` /
          ``"dip"``) and ``severity`` (``"low"`` / ``"medium"`` /
          ``"high"``).
        * ``n_anomalies``      – int, number of flagged months.
        * ``threshold``        – float, the threshold that was applied.
        * ``scores``           – list[float], one robust z-score per
          month, in chronological order.
        * ``structural_breaks`` – list[dict] of level shifts with the
          keys ``month`` (int), ``direction`` (``"up"`` / ``"down"``)
          and ``change_pct`` (float).  They are detected on the
          deseasonalised level: the three-month trailing average has to
          move by more than two standard deviations of its own history.

    Raises
    ------
    ValueError
        If *summary_df* is empty or has no ``Sales`` column.

    Examples
    --------
    >>> import pandas as pd
    >>> summary = pd.DataFrame({
    ...     "Month": list(range(1, 13)),
    ...     "MonthOfYear": list(range(1, 13)),
    ...     "Sales": [100.0] * 12,
    ... })
    >>> result = detect_anomalies(summary)
    >>> result["n_anomalies"]
    0
    >>> len(result["scores"])
    12
    """
    if summary_df is None or len(summary_df) == 0:
        raise ValueError("summary_df must contain at least one row.")

    sales_col = _resolve_sales_column(summary_df)
    sales = (
        pd.to_numeric(summary_df[sales_col], errors="coerce").fillna(0.0).to_numpy(dtype="float64")
    )
    month_labels = _resolve_month_labels(summary_df)
    month_of_year = _resolve_month_of_year(summary_df)
    n = len(sales)

    signal = sales
    expected = np.full(n, np.nan, dtype="float64")
    scored_on = "sales"

    if decomposition:
        residual = np.asarray(decomposition.get("residual", []), dtype="float64")
        if residual.size == n:
            signal = residual
            scored_on = "residual"
            trend = np.asarray(decomposition.get("trend", []), dtype="float64")
            seasonal = np.asarray(decomposition.get("seasonal", []), dtype="float64")
            if trend.size == n and seasonal.size == n:
                expected = trend + seasonal
            else:
                expected = sales - signal
        else:
            logger.warning(
                "Decomposition length (%d) does not match the summary (%d); "
                "scoring raw sales instead.",
                residual.size,
                n,
            )

    if not np.all(np.isfinite(signal)):
        signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)

    median, scale = _robust_scale(signal)
    scores = (signal - median) / scale if scale else np.zeros(n, dtype="float64")
    scores = np.asarray(scores, dtype="float64")

    if not np.any(np.isfinite(expected)):
        expected = np.full(n, median, dtype="float64")
    expected = np.nan_to_num(expected, nan=0.0, posinf=0.0, neginf=0.0)

    anomalies: list[dict[str, Any]] = []
    for i in range(n):
        score = float(scores[i])
        if not np.isfinite(score) or abs(score) <= threshold:
            continue
        calendar_month = int(month_of_year[i])
        anomalies.append(
            {
                "month": int(month_labels[i]),
                "name": month_name(calendar_month),
                "sales": round(float(sales[i]), _ROUND_DP),
                "expected": round(float(expected[i]), _ROUND_DP),
                "score": round(score, _ROUND_DP),
                "direction": "spike" if score > 0 else "dip",
                "severity": _severity(score),
            }
        )

    level = sales / _seasonal_index(decomposition, month_of_year, n)
    breaks = _structural_breaks(level, month_labels)

    logger.info(
        "Anomaly detection (%s): %d month(s) beyond |z| > %.2f, %d structural " "break(s).",
        scored_on,
        len(anomalies),
        threshold,
        len(breaks),
    )

    return {
        "anomalies": anomalies,
        "n_anomalies": len(anomalies),
        "threshold": float(threshold),
        "scores": [round(float(s), _ROUND_DP) for s in scores],
        "structural_breaks": breaks,
    }
