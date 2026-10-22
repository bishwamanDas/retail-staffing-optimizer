"""Salesperson cohort, tenure and attrition analysis for RevInsight.

The module reconstructs the store's staffing history from the raw
monthly sales table.  A salesperson column is treated as a single
person: the months in which their contribution is strictly positive
are the months they were on the roster, which makes it possible to
derive joining dates, leaving dates, tenure, ramp-up speed and
attrition behaviour.

Conventions
-----------
* **Active month** — a month in which the salesperson's individual
  sales are strictly positive.  A zero means the person was not on
  the roster (not yet hired, on leave, or already gone), matching
  :func:`revinsight.utils.count_active_employees`.
* **Leaver** — someone who was active at some point and whose final
  active month falls *strictly* before the dataset's last month.
* **Still active** — a leaver's counterpart: their final active month
  *is* the dataset's last month.
* **ramp_months** — the number of months from a person's first active
  month until they first reach 80 % of their own peak monthly sales.
  A value of ``0`` means they never reached that level (they were
  never active at all).
* **avg_monthly_sales / avg_sales_when_active** — both are the mean
  sales across the months the person was active, i.e.
  ``total_sales / months_active``.  They are reported under two names
  because the API contract uses both.

All monetary values are denominated in **Lakhs INR**.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from .utils import get_salesperson_columns, infer_month_of_year, safe_div

logger = logging.getLogger(__name__)

__all__ = [
    "RAMP_THRESHOLD",
    "RETENTION_MONTHS",
    "analyze_workforce",
]

RAMP_THRESHOLD: float = 0.80
"""Share of a salesperson's own peak that defines full ramp-up."""

RETENTION_MONTHS: int = 12
"""Tenure (months) a hire must reach to count as retained."""

_ROUND_DP: int = 4
"""Decimal places used when rounding monetary values."""


# --------------------------------------------------------------------------- #
#  Internal helpers                                                            #
# --------------------------------------------------------------------------- #


def _month_labels(df: pd.DataFrame) -> np.ndarray:
    """Return the sequential month label (1-based) of every row of *df*.

    Parameters
    ----------
    df : pd.DataFrame
        Raw sales DataFrame.

    Returns
    -------
    np.ndarray
        Integer array of month labels, one per row.
    """
    if "Month" in df.columns:
        return df["Month"].astype(int).to_numpy()
    return np.arange(1, len(df) + 1, dtype=int)


def _store_total_sales(df: pd.DataFrame, sp_cols: list[str]) -> float:
    """Return the total store sales used as the share-of-sales denominator.

    Parameters
    ----------
    df : pd.DataFrame
        Raw sales DataFrame.
    sp_cols : list of str
        Salesperson column names.

    Returns
    -------
    float
        Total sales in Lakhs INR.
    """
    for col in df.columns:
        if str(col).strip().lower() == "sales":
            return float(pd.to_numeric(df[col], errors="coerce").fillna(0.0).sum())
    if sp_cols:
        return float(
            df[sp_cols]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype="float64")
            .sum()
        )
    return 0.0


def _calendar_labels(month_labels: np.ndarray, start_month: int) -> list[str]:
    """Return a human-readable calendar label for every dataset month.

    The dataset has no calendar dates, so the year is expressed relative
    to the start of the dataset (``"Y1"`` for its first calendar year).

    Parameters
    ----------
    month_labels : np.ndarray
        Sequential month labels.
    start_month : int
        Calendar month (1-12) that the first dataset month represents.

    Returns
    -------
    list of str
        Labels such as ``"Y1 Dec"``.
    """
    names = [
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
    labels: list[str] = []
    for label in month_labels:
        calendar_month = infer_month_of_year(int(label), start_month)
        year = (int(label) - 1 + start_month - 1) // 12 + 1
        labels.append(f"Y{year} {names[calendar_month - 1]}")
    return labels


def _ramp_months(values: np.ndarray, active_idx: np.ndarray) -> int:
    """Return the number of months needed to reach 80 % of the peak.

    Parameters
    ----------
    values : np.ndarray
        The salesperson's monthly sales values.
    active_idx : np.ndarray
        Positions (0-based) of the months in which they were active.

    Returns
    -------
    int
        Number of months from the first active month until the 80 %
        threshold is first met; ``0`` when it is never met.
    """
    if active_idx.size == 0:
        return 0

    peak = float(np.max(values[active_idx]))
    if peak <= 0:
        return 0

    threshold = RAMP_THRESHOLD * peak
    for offset, position in enumerate(active_idx):
        if values[position] >= threshold - 1e-9:
            return int(offset + 1)
    return 0


def _empty_result() -> dict[str, Any]:
    """Return a well-formed, empty workforce result.

    Used when the supplied DataFrame contains no usable rows so that
    downstream consumers never receive ``None``.

    Returns
    -------
    dict
        Empty workforce analysis structure.
    """
    return {
        "employees": [],
        "attrition": {
            "total_leavers": 0,
            "total_hires": 0,
            "avg_tenure_months": 0.0,
            "recent_12m_leavers": 0,
            "retention_rate": 0.0,
            "timeline": [],
        },
        "headcount_timeline": [],
        "summary": {
            "avg_headcount": 0.0,
            "max_headcount": 0,
            "min_headcount": 0,
            "current_headcount": 0,
            "avg_tenure_active": 0.0,
        },
    }


# --------------------------------------------------------------------------- #
#  1. Workforce analysis                                                        #
# --------------------------------------------------------------------------- #


def analyze_workforce(
    df: pd.DataFrame,
    start_month: int = 1,
) -> dict[str, Any]:
    """Analyse the store's salesforce: productivity, tenure and attrition.

    Every ``sale spN`` column is treated as one person.  The analysis
    reconstructs when each person joined, when they last contributed,
    how quickly they ramped up, who left and how the headcount evolved
    month by month.

    Parameters
    ----------
    df : pd.DataFrame
        Raw sales DataFrame as returned by
        :func:`revinsight.utils.load_and_validate_data` — it must
        contain the ``Month`` column and the ``sale spN`` columns.
    start_month : int, optional
        Calendar month (1-12) that dataset month 1 corresponds to
        (default ``1``).  Used to anchor the calendar interpretation of
        the dataset months.

    Returns
    -------
    dict
        Dictionary with the following keys:

        * ``employees``         – list[dict], one entry per salesperson
          (sorted by ``total_sales`` descending) with the keys ``id``,
          ``total_sales``, ``avg_monthly_sales``, ``months_active``,
          ``first_month``, ``last_month``, ``still_active``,
          ``avg_sales_when_active``, ``peak_month_sales``,
          ``share_of_sales``, ``tenure_months`` and ``ramp_months``.
          A salesperson column that was never active is still reported
          (with zeros and ``first_month``/``last_month`` set to ``0``)
          but never counts as a hire.
        * ``attrition``         – dict with ``total_leavers``,
          ``total_hires``, ``avg_tenure_months``,
          ``recent_12m_leavers``, ``retention_rate`` and ``timeline``
          (the months in which somebody joined or left).
        * ``headcount_timeline`` – list[dict], one entry per dataset
          month with ``month``, ``employees``, ``joined`` and ``left``.
        * ``summary``           – dict with ``avg_headcount``,
          ``max_headcount``, ``min_headcount``, ``current_headcount``
          and ``avg_tenure_active``.

    Raises
    ------
    ValueError
        If *start_month* is outside the range 1-12.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     "Month": [1, 2, 3],
    ...     "Sales": [10.0, 20.0, 15.0],
    ...     "sale sp1": [10.0, 12.0, 15.0],
    ...     "sale sp2": [0.0, 8.0, 0.0],
    ... })
    >>> result = analyze_workforce(df)
    >>> result["summary"]["current_headcount"]
    1
    >>> result["attrition"]["total_leavers"]
    1
    >>> result["employees"][0]["id"]
    'sale sp1'
    """
    if not 1 <= int(start_month) <= 12:
        raise ValueError(f"start_month must be between 1 and 12, got {start_month}.")

    if df is None or len(df) == 0:
        logger.warning("Empty DataFrame supplied to analyze_workforce().")
        return _empty_result()

    sp_cols = get_salesperson_columns(df)
    if not sp_cols:
        logger.warning("No salesperson columns found; returning an empty result.")
        return _empty_result()

    month_labels = _month_labels(df)
    n = len(df)
    last_month = int(month_labels[-1])

    sales_matrix = df[sp_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype("float64")
    active_matrix = sales_matrix > 0
    store_total = _store_total_sales(df, sp_cols)

    # ------------------------------------------------------------------ #
    #  1. Per-salesperson metrics                                         #
    # ------------------------------------------------------------------ #
    employees: list[dict[str, Any]] = []
    joined_by_month: dict[int, list[str]] = {}
    left_by_month: dict[int, list[str]] = {}
    leavers: list[dict[str, Any]] = []
    leaver_positions: list[int] = []
    still_active: list[dict[str, Any]] = []
    total_hires = 0

    for col in sp_cols:
        values = sales_matrix[col].to_numpy(dtype="float64")
        active_idx = np.flatnonzero(active_matrix[col].to_numpy(dtype=bool))
        total_sales = float(np.sum(values))

        if active_idx.size == 0:
            employees.append(
                {
                    "id": str(col),
                    "total_sales": 0.0,
                    "avg_monthly_sales": 0.0,
                    "months_active": 0,
                    "first_month": 0,
                    "last_month": 0,
                    "still_active": False,
                    "avg_sales_when_active": 0.0,
                    "peak_month_sales": 0.0,
                    "share_of_sales": 0.0,
                    "tenure_months": 0,
                    "ramp_months": 0,
                }
            )
            continue

        total_hires += 1
        first_position = int(active_idx[0])
        last_position = int(active_idx[-1])
        first_month = int(month_labels[first_position])
        last_month_sp = int(month_labels[last_position])
        months_active = int(active_idx.size)
        peak = float(np.max(values[active_idx]))
        avg_active = total_sales / months_active if months_active else 0.0
        is_active = last_month_sp >= last_month
        tenure = last_month_sp - first_month + 1

        record = {
            "id": str(col),
            "total_sales": round(total_sales, _ROUND_DP),
            "avg_monthly_sales": round(avg_active, _ROUND_DP),
            "months_active": months_active,
            "first_month": first_month,
            "last_month": last_month_sp,
            "still_active": bool(is_active),
            "avg_sales_when_active": round(avg_active, _ROUND_DP),
            "peak_month_sales": round(peak, _ROUND_DP),
            "share_of_sales": round(safe_div(total_sales, store_total), _ROUND_DP),
            "tenure_months": int(tenure),
            "ramp_months": int(_ramp_months(values, active_idx)),
        }
        employees.append(record)

        joined_by_month.setdefault(first_month, []).append(str(col))
        if is_active:
            still_active.append(record)
        else:
            leavers.append(record)
            leaver_positions.append(last_position)
            left_by_month.setdefault(last_month_sp, []).append(str(col))

    employees.sort(key=lambda e: (-e["total_sales"], e["id"]))

    # ------------------------------------------------------------------ #
    #  2. Headcount timeline                                              #
    # ------------------------------------------------------------------ #
    headcount_per_month = active_matrix.sum(axis=1).astype(int).tolist()
    headcount_timeline: list[dict[str, Any]] = []
    for i in range(n):
        month = int(month_labels[i])
        headcount_timeline.append(
            {
                "month": month,
                "employees": int(headcount_per_month[i]),
                "joined": list(joined_by_month.get(month, [])),
                "left": list(left_by_month.get(month, [])),
            }
        )

    # ------------------------------------------------------------------ #
    #  3. Attrition                                                        #
    # ------------------------------------------------------------------ #
    calendar_labels = _calendar_labels(month_labels, start_month)
    # "Recent" means: their departure happened within the final 12
    # calendar months covered by the dataset.
    recent_start = max(0, n - 12)
    recent_leavers = [
        leaver
        for leaver, position in zip(leavers, leaver_positions, strict=False)
        if position >= recent_start
    ]

    retained = [leaver for leaver in leavers if int(leaver["tenure_months"]) >= RETENTION_MONTHS]
    retention_rate = safe_div(len(retained), len(leavers), default=0.0)
    avg_tenure = float(np.mean([leaver["tenure_months"] for leaver in leavers])) if leavers else 0.0

    attrition_timeline: list[dict[str, Any]] = []
    for i in range(n):
        month = int(month_labels[i])
        joined = joined_by_month.get(month, [])
        left = left_by_month.get(month, [])
        if joined or left:
            attrition_timeline.append(
                {
                    "month": month,
                    "joined": list(joined),
                    "left": list(left),
                }
            )

    # ------------------------------------------------------------------ #
    #  4. Summary                                                          #
    # ------------------------------------------------------------------ #
    avg_headcount = float(np.mean(headcount_per_month)) if headcount_per_month else 0.0
    avg_tenure_active = (
        float(np.mean([e["tenure_months"] for e in still_active])) if still_active else 0.0
    )

    headcounts = list(headcount_per_month)
    max_headcount = int(max(headcounts)) if headcounts else 0
    min_headcount = int(min(headcounts)) if headcounts else 0
    current_headcount = int(headcounts[-1]) if headcounts else 0

    logger.info(
        "Workforce analysis: %d salesperson(s), %d hire(s), %d leaver(s); "
        "calendar window %s -> %s.",
        len(employees),
        total_hires,
        len(leavers),
        calendar_labels[0] if calendar_labels else "n/a",
        calendar_labels[-1] if calendar_labels else "n/a",
    )

    return {
        "employees": employees,
        "attrition": {
            "total_leavers": int(len(leavers)),
            "total_hires": int(total_hires),
            "avg_tenure_months": round(avg_tenure, _ROUND_DP),
            "recent_12m_leavers": int(len(recent_leavers)),
            "retention_rate": round(float(retention_rate), _ROUND_DP),
            "timeline": attrition_timeline,
        },
        "headcount_timeline": headcount_timeline,
        "summary": {
            "avg_headcount": round(avg_headcount, _ROUND_DP),
            "max_headcount": max_headcount,
            "min_headcount": min_headcount,
            "current_headcount": current_headcount,
            "avg_tenure_active": round(avg_tenure_active, _ROUND_DP),
        },
    }
