"""Financial analysis module for branded-clothing store staffing decisions.

Provides functions to compute gross margins, break-even points,
contribution margins, incremental employee impact, sensitivity
scenarios, vacancy cost, payback period and ROI, six-month projections,
and a comprehensive financial report.

Gross Margin formula
--------------------
    GrossMargin = (1 - COGS_rate - commission_rate) x Sales - salary x Employees

    where salary, commission_rate are configurable via CLI arguments
    and COGS_rate defaults to 0.50 (50%).

All monetary values are in **Lakhs INR** unless stated otherwise.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

COGS_RATE: float = 0.50
"""Default cost-of-goods-sold rate applied to sales."""

__all__ = [
    "COGS_RATE",
    "compute_gross_margin",
    "calculate_historical_margins",
    "calculate_summary_statistics",
    "calculate_break_even_analysis",
    "calculate_contribution_margin",
    "calculate_incremental_analysis",
    "run_sensitivity_analysis",
    "calculate_six_month_projection",
    "estimate_vacancy_cost",
    "calculate_payback_period",
    "calculate_roi",
    "generate_financial_report",
]


# --------------------------------------------------------------------------- #
# Helper - margin calculation                                                  #
# --------------------------------------------------------------------------- #
def compute_gross_margin(
    sales: float | pd.Series,
    employees: int | pd.Series,
    salary_per_person: float = 3.0,
    commission_rate: float = 0.05,
    cogs_rate: float = COGS_RATE,
) -> float | pd.Series:
    """Return gross margin using the standard formula.

    Parameters
    ----------
    sales : float or pd.Series
        Monthly sales in Lakhs INR.
    employees : int or pd.Series
        Number of active employees for that month.
    salary_per_person : float, optional
        Monthly salary per employee.
    commission_rate : float, optional
        Commission rate on sales.
    cogs_rate : float, optional
        Cost of goods sold rate.

    Returns
    -------
    float or pd.Series
        Gross margin in Lakhs INR.
    """
    return (1.0 - cogs_rate - commission_rate) * sales - salary_per_person * employees


# Backwards-compatible private alias used elsewhere in the package.
_compute_gross_margin = compute_gross_margin


# --------------------------------------------------------------------------- #
# 1. Historical margins                                                        #
# --------------------------------------------------------------------------- #
def calculate_historical_margins(
    summary_df: pd.DataFrame,
    salary_per_person: float = 3.0,
    commission_rate: float = 0.05,
    cogs_rate: float = COGS_RATE,
    force: bool = False,
) -> pd.DataFrame:
    """Ensure a ``GrossMargin`` column exists on *summary_df*.

    The margin is computed as::

        GrossMargin = (1 - COGS - commission) x Sales - salary x ActiveEmployees

    Parameters
    ----------
    summary_df : pd.DataFrame
        Must contain ``Sales`` and ``ActiveEmployees`` columns.
    salary_per_person : float, optional
        Monthly salary cost per employee.
    commission_rate : float, optional
        Commission rate.
    cogs_rate : float, optional
        Cost of goods sold rate.
    force : bool, optional
        Recompute even if the column already exists.  Useful when the
        caller changed the salary or commission assumptions after the
        summary was built.

    Returns
    -------
    pd.DataFrame
        A copy of *summary_df* with the ``GrossMargin`` column present.

    Raises
    ------
    KeyError
        If required columns are missing from *summary_df*.
    """
    required_cols = {"Sales", "ActiveEmployees"}
    missing = required_cols - set(summary_df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    df = summary_df.copy()
    if force or "GrossMargin" not in df.columns:
        df["GrossMargin"] = compute_gross_margin(
            df["Sales"], df["ActiveEmployees"], salary_per_person, commission_rate, cogs_rate
        )
    return df


# --------------------------------------------------------------------------- #
# 2. Summary statistics                                                        #
# --------------------------------------------------------------------------- #
def calculate_summary_statistics(
    summary_df: pd.DataFrame,
    salary_per_person: float = 3.0,
    commission_rate: float = 0.05,
    cogs_rate: float = COGS_RATE,
) -> dict[str, float]:
    """Compute key summary statistics from the monthly data.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Must contain ``Sales`` and ``ActiveEmployees`` columns.

    Returns
    -------
    dict
        Keys: ``avg_sales``, ``median_sales``, ``std_sales``, ``avg_margin``,
        ``median_margin``, ``std_margin``, ``avg_employees``,
        ``avg_sales_per_emp``, ``total_margin``, ``min_margin``,
        ``max_margin``, ``margin_volatility``, ``total_sales``,
        ``total_staff_cost``, ``total_commission``.
    """
    df = calculate_historical_margins(summary_df, salary_per_person, commission_rate, cogs_rate)

    sales: pd.Series = df["Sales"]
    margin: pd.Series = df["GrossMargin"]
    employees: pd.Series = df["ActiveEmployees"]

    avg_employees = employees.mean()
    eff_rate = 1.0 - cogs_rate - commission_rate

    return {
        "avg_sales": float(sales.mean()),
        "median_sales": float(sales.median()),
        "std_sales": float(sales.std()),
        "avg_margin": float(margin.mean()),
        "median_margin": float(margin.median()),
        "std_margin": float(margin.std()),
        "avg_employees": float(avg_employees),
        "avg_sales_per_emp": float(sales.mean() / avg_employees) if avg_employees else 0.0,
        "total_margin": float(margin.sum()),
        "min_margin": float(margin.min()),
        "max_margin": float(margin.max()),
        "margin_volatility": float(margin.std()),
        "total_sales": float(sales.sum()),
        "total_staff_cost": float((employees * salary_per_person).sum()),
        "total_commission": float(sales.sum() * commission_rate),
        "effective_margin_rate": float(eff_rate),
    }


# --------------------------------------------------------------------------- #
# 3. Break-even analysis                                                       #
# --------------------------------------------------------------------------- #
def calculate_break_even_analysis(
    salary_per_person: float = 3.0,
    commission_rate: float = 0.05,
    cogs_rate: float = COGS_RATE,
) -> dict[str, float]:
    """Return break-even metrics for one employee.

    Break-even sales per employee::

        salary / effective_margin_rate = 3 / 0.45 ~ 6.667 Lakhs

    Returns
    -------
    dict
        Keys: ``break_even_sales_per_employee``, ``monthly_salary_cost``,
        ``commission_rate``, ``effective_margin_rate``.
    """
    effective_margin_rate = 1.0 - cogs_rate - commission_rate
    if effective_margin_rate <= 0:
        raise ZeroDivisionError(
            "effective margin rate must be positive to compute break-even sales"
        )
    return {
        "break_even_sales_per_employee": salary_per_person / effective_margin_rate,
        "monthly_salary_cost": salary_per_person,
        "commission_rate": commission_rate,
        "effective_margin_rate": effective_margin_rate,
    }


# --------------------------------------------------------------------------- #
# 4. Contribution margin                                                       #
# --------------------------------------------------------------------------- #
def calculate_contribution_margin(
    summary_df: pd.DataFrame,
    salary_per_person: float = 3.0,
    commission_rate: float = 0.05,
    cogs_rate: float = COGS_RATE,
) -> pd.DataFrame:
    """Compute average contribution margin **per employee** at each staffing level.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Must contain ``Sales`` and ``ActiveEmployees`` columns.

    Returns
    -------
    pd.DataFrame
        One row per headcount with ``ActiveEmployees``, ``AvgSales``,
        ``AvgGrossMargin``, ``AvgContributionPerEmployee`` and
        ``MonthsObserved``.
    """
    df = calculate_historical_margins(summary_df, salary_per_person, commission_rate, cogs_rate)

    grouped = df.groupby("ActiveEmployees").agg(
        AvgSales=("Sales", "mean"),
        AvgGrossMargin=("GrossMargin", "mean"),
        MonthsObserved=("GrossMargin", "count"),
    )

    grouped["AvgContributionPerEmployee"] = grouped["AvgGrossMargin"] / grouped.index

    return grouped.reset_index()


# --------------------------------------------------------------------------- #
# 5. Incremental analysis                                                     #
# --------------------------------------------------------------------------- #
def calculate_incremental_analysis(
    summary_df: pd.DataFrame,
    salary_per_person: float = 3.0,
    commission_rate: float = 0.05,
    cogs_rate: float = COGS_RATE,
) -> pd.DataFrame:
    """Analyse the incremental impact of adding one more employee.

    Steps
    -----
    1. Group by ``ActiveEmployees`` and compute averages.
    2. Sort by employee count.
    3. Calculate the *incremental* change in sales and margin when
       moving from one staffing level to the next.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Must contain ``Sales`` and ``ActiveEmployees`` columns.

    Returns
    -------
    pd.DataFrame
        Columns: ``ActiveEmployees``, ``AvgSales``, ``AvgGrossMargin``,
        ``AvgSalesPerEmployee``, ``IncrementalSales``,
        ``IncrementalMargin``.
    """
    df = calculate_historical_margins(summary_df, salary_per_person, commission_rate, cogs_rate)

    grouped = (
        df.groupby("ActiveEmployees")
        .agg(
            AvgSales=("Sales", "mean"),
            AvgGrossMargin=("GrossMargin", "mean"),
        )
        .sort_index()
    )

    grouped["AvgSalesPerEmployee"] = grouped["AvgSales"] / grouped.index
    grouped["IncrementalSales"] = grouped["AvgSales"].diff()
    grouped["IncrementalMargin"] = grouped["AvgGrossMargin"].diff()
    grouped["MarginalSalesPerAddedEmployee"] = grouped["IncrementalSales"].diff()

    return grouped.reset_index()


# --------------------------------------------------------------------------- #
# 6. Sensitivity analysis                                                     #
# --------------------------------------------------------------------------- #
def run_sensitivity_analysis(
    base_sales: float,
    base_employees: int,
    base_salary: float = 3.0,
    base_commission: float = 0.05,
    salary_range: tuple[float, float, float] = (2.0, 4.0, 0.5),
    commission_range: tuple[float, float, float] = (0.03, 0.07, 0.01),
    cogs_rate: float = COGS_RATE,
) -> pd.DataFrame:
    """Test gross-margin sensitivity to salary and commission assumptions.

    For every combination of salary and commission rate within the
    specified ranges the gross margin is recalculated as::

        margin = (1 - COGS - commission) x sales - salary x employees

    Parameters
    ----------
    base_sales : float
        Reference monthly sales figure (Lakhs INR).
    base_employees : int
        Reference employee count.
    base_salary : float, optional
        Baseline salary assumption.
    base_commission : float, optional
        Baseline commission assumption.
    salary_range : tuple of (start, stop, step)
        Salary costs to test (Lakhs/person/month).  *stop* is inclusive.
    commission_range : tuple of (start, stop, step)
        Commission rates to test (fraction).  *stop* is inclusive.
    cogs_rate : float, optional
        Cost of goods sold rate.

    Returns
    -------
    pd.DataFrame
        One row per scenario with ``Salary``, ``CommissionRate``,
        ``EffectiveMarginRate``, ``GrossMargin``,
        ``BreakEvenSalesPerEmp``, ``MarginVsBase`` and
        ``RecommendedAction``.
    """
    salaries = np.arange(
        salary_range[0],
        salary_range[1] + salary_range[2] / 2,  # inclusive stop
        salary_range[2],
    )
    commissions = np.arange(
        commission_range[0],
        commission_range[1] + commission_range[2] / 2,
        commission_range[2],
    )

    base_margin = compute_gross_margin(
        base_sales, base_employees, base_salary, base_commission, cogs_rate
    )

    rows: list[dict[str, float]] = []
    for salary in salaries:
        for comm in commissions:
            eff_rate = 1.0 - cogs_rate - comm
            margin = eff_rate * base_sales - salary * base_employees
            be_sales = salary / eff_rate if eff_rate > 0 else float("inf")
            rows.append(
                {
                    "salary": round(float(salary), 2),
                    "commission_rate": round(float(comm), 4),
                    "effective_margin_rate": round(float(eff_rate), 4),
                    "gross_margin": round(float(margin), 4),
                    "break_even_sales_per_emp": round(float(be_sales), 4),
                    "margin_vs_base": round(float(margin - base_margin), 4),
                    "recommended_action": (
                        "Healthy margin" if margin > 0 else "Margin negative — restructure"
                    ),
                }
            )

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 7. Six-month projection                                                     #
# --------------------------------------------------------------------------- #
def calculate_six_month_projection(
    forecast_sales: list[float],
    num_employees: int,
    salary_per_person: float = 3.0,
    commission_rate: float = 0.05,
    cogs_rate: float = COGS_RATE,
) -> dict[str, Any]:
    """Project total gross margin over a six-month horizon.

    Parameters
    ----------
    forecast_sales : list of float
        Exactly six monthly sales forecasts (Lakhs INR).
    num_employees : int
        Fixed employee headcount for the projection period.
    salary_per_person : float, optional
        Monthly salary cost per employee (default 3 Lakhs).
    commission_rate : float, optional
        Commission rate.
    cogs_rate : float, optional
        Cost of goods sold rate.

    Returns
    -------
    dict
        Keys: ``forecast_sales``, ``num_employees``, ``salary_per_person``,
        ``monthly_margins``, ``total_sales``, ``total_salary_cost``,
        ``total_gross_margin``, ``avg_monthly_margin``, ``break_even_month``.

    Raises
    ------
    ValueError
        If *forecast_sales* does not contain exactly six values.
    """
    if len(forecast_sales) != 6:
        raise ValueError(f"Expected exactly 6 monthly forecasts, got {len(forecast_sales)}.")

    eff_rate = 1.0 - cogs_rate - commission_rate
    monthly_margins = [eff_rate * s - salary_per_person * num_employees for s in forecast_sales]

    total_sales = sum(forecast_sales)
    total_salary = salary_per_person * num_employees * 6
    total_margin = sum(monthly_margins)
    avg_margin = total_margin / 6

    cumulative = 0.0
    break_even_month: int | None = None
    for idx, m in enumerate(monthly_margins, start=1):
        cumulative += m
        if cumulative > 0 and break_even_month is None:
            break_even_month = idx

    return {
        "forecast_sales": list(forecast_sales),
        "num_employees": num_employees,
        "salary_per_person": salary_per_person,
        "monthly_margins": monthly_margins,
        "total_sales": round(total_sales, 4),
        "total_salary_cost": round(total_salary, 4),
        "total_gross_margin": round(total_margin, 4),
        "avg_monthly_margin": round(avg_margin, 4),
        "break_even_month": break_even_month,
    }


# --------------------------------------------------------------------------- #
# 8. Vacancy cost                                                              #
# --------------------------------------------------------------------------- #
def estimate_vacancy_cost(
    summary_df: pd.DataFrame,
    open_positions: int = 1,
    salary_per_person: float = 3.0,
    commission_rate: float = 0.05,
    cogs_rate: float = COGS_RATE,
) -> dict[str, Any]:
    """Estimate the financial cost of leaving a position unfilled.

    The estimate is built from the *marginal* revenue an additional
    employee contributes at the observed headcount levels, damped by a
    service-capacity factor to reflect that one unfilled seat rarely
    removes all of that revenue.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Must contain ``Sales`` and ``ActiveEmployees`` columns.
    open_positions : int, optional
        Number of unfilled positions (default 1).
    salary_per_person : float, optional
        Monthly salary per employee.
    commission_rate : float, optional
        Commission rate.
    cogs_rate : float, optional
        Cost of goods sold rate.

    Returns
    -------
    dict
        Keys: ``open_positions``, ``marginal_sales_per_employee``,
        ``service_capacity_factor``, ``estimated_lost_sales_per_month``,
        ``estimated_margin_cost_per_month``, ``annualised_cost``,
        ``interpretation``.

    Notes
    -----
    When the headcount curve is not estimable (for example only one
    headcount level was ever observed), the function falls back to the
    average sales per employee and flags the reduced confidence.
    """
    df = calculate_historical_margins(summary_df, salary_per_person, commission_rate, cogs_rate)

    grouped = (
        df.groupby("ActiveEmployees")
        .agg(avg_sales=("Sales", "mean"), months=("Sales", "count"))
        .sort_index()
    )

    marginal_sales: float | None = None
    confidence = "HIGH"
    if len(grouped) >= 2:
        deltas = grouped["avg_sales"].diff()
        # Marginal revenue of the *last* observed step (closest to the
        # current headcount) is the most relevant for a vacancy decision.
        last_delta = deltas.iloc[-1]
        if pd.notna(last_delta) and last_delta > 0:
            marginal_sales = float(last_delta)
        else:
            valid = deltas.dropna()
            if len(valid):
                marginal_sales = float(valid.mean())
    else:
        confidence = "LOW"

    if marginal_sales is None or marginal_sales <= 0:
        avg_emp = float(df["ActiveEmployees"].replace(0, np.nan).mean())
        marginal_sales = float(df["Sales"].mean() / avg_emp) if avg_emp else 0.0
        confidence = "LOW"

    service_capacity_factor = 0.35
    lost_sales_per_month = marginal_sales * open_positions * service_capacity_factor
    margin_rate = 1.0 - cogs_rate - commission_rate
    margin_cost_per_month = lost_sales_per_month * margin_rate
    salary_saving = salary_per_person * open_positions

    net_monthly = margin_cost_per_month - salary_saving

    return {
        "open_positions": int(open_positions),
        "marginal_sales_per_employee": round(marginal_sales, 4),
        "service_capacity_factor": service_capacity_factor,
        "estimated_lost_sales_per_month": round(lost_sales_per_month, 4),
        "estimated_margin_cost_per_month": round(margin_cost_per_month, 4),
        "monthly_salary_saving": round(salary_saving, 4),
        "net_monthly_impact": round(net_monthly, 4),
        "annualised_cost": round(margin_cost_per_month * 12, 4),
        "confidence": confidence,
        "interpretation": (
            f"Leaving {open_positions} position(s) open is estimated to cost "
            f"Rs.{margin_cost_per_month:.2f} L of gross margin per month "
            f"(Rs.{margin_cost_per_month * 12:.2f} L annually), against a "
            f"salary saving of Rs.{salary_saving:.2f} L per month. Net monthly "
            f"impact of hiring: Rs.{net_monthly:.2f} L."
        ),
    }


# --------------------------------------------------------------------------- #
# 9. Payback period & ROI                                                     #
# --------------------------------------------------------------------------- #
def calculate_payback_period(
    incremental_sales_by_month: Sequence[float],
    salary_per_person: float = 3.0,
    commission_rate: float = 0.05,
    cogs_rate: float = COGS_RATE,
) -> dict[str, Any]:
    """Compute how long a new hire takes to pay back their cost.

    Each month's incremental gross margin is compared with the monthly
    salary.  The payback period is the (possibly fractional) month at
    which the cumulative contribution first covers the salary cost.

    Parameters
    ----------
    incremental_sales_by_month : sequence of float
        Extra revenue the new hire is expected to add each month.
    salary_per_person : float, optional
        Monthly salary of the new hire.
    commission_rate : float, optional
        Commission rate.
    cogs_rate : float, optional
        Cost of goods sold rate.

    Returns
    -------
    dict
        Keys: ``monthly_contribution``, ``cumulative_contribution``,
        ``months_to_payback``, ``total_contribution``,
        ``investment_per_month``, ``roi_pct``, ``ever_pays_back``.

    Examples
    --------
    >>> result = calculate_payback_period([20, 25, 30])
    >>> result["ever_pays_back"]
    True
    """
    margin_rate = 1.0 - cogs_rate - commission_rate
    contributions = [float(s) * margin_rate - salary_per_person for s in incremental_sales_by_month]

    cumulative: list[float] = []
    running = 0.0
    for c in contributions:
        running += c
        cumulative.append(round(running, 4))

    payback: float | None = None
    for idx, cum in enumerate(cumulative):
        if cum >= 0:
            prev = cumulative[idx - 1] if idx > 0 else 0.0
            step = contributions[idx]
            if idx == 0 or step <= 0:
                # The very first month already covers the investment.
                payback = float(idx + 1)
            else:
                # Fraction of the (idx+1)-th month needed to break even.
                payback = round(idx + (-prev / step), 2)
            break

    total_contribution = sum(contributions)
    investment = salary_per_person * len(incremental_sales_by_month)
    roi = (total_contribution / investment * 100.0) if investment else 0.0

    return {
        "monthly_contribution": [round(c, 4) for c in contributions],
        "cumulative_contribution": cumulative,
        "months_to_payback": payback,
        "total_contribution": round(total_contribution, 4),
        "investment_per_month": salary_per_person,
        "roi_pct": round(roi, 2),
        "ever_pays_back": bool(payback is not None),
    }


def calculate_roi(
    incremental_sales_total: float,
    total_cost: float,
) -> dict[str, float]:
    """Compute a simple return-on-investment for a staffing decision.

    Parameters
    ----------
    incremental_sales_total : float
        Total additional gross margin generated by the decision.
    total_cost : float
        Total incremental cost of the decision (salary + any one-off).

    Returns
    -------
    dict
        Keys: ``incremental_margin``, ``total_cost``, ``roi_pct``,
        ``net_benefit``.
    """
    net = incremental_sales_total - total_cost
    roi = (net / total_cost * 100.0) if total_cost else 0.0
    return {
        "incremental_margin": round(float(incremental_sales_total), 4),
        "total_cost": round(float(total_cost), 4),
        "roi_pct": round(float(roi), 2),
        "net_benefit": round(float(net), 4),
    }


# --------------------------------------------------------------------------- #
# 10. Master report generator                                                 #
# --------------------------------------------------------------------------- #
def generate_financial_report(
    summary_df: pd.DataFrame,
    salary_per_person: float = 3.0,
    commission_rate: float = 0.05,
    cogs_rate: float = COGS_RATE,
    open_positions: int = 1,
) -> dict[str, Any]:
    """Run **all** financial analyses and bundle results into one dictionary.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Must contain ``Sales`` and ``ActiveEmployees`` columns.
    salary_per_person : float, optional
        Monthly salary per employee.
    commission_rate : float, optional
        Commission rate.
    cogs_rate : float, optional
        Cost of goods sold rate.
    open_positions : int, optional
        Number of currently unfilled positions (default 1).

    Returns
    -------
    dict
        Top-level keys: ``historical_margins``, ``summary_statistics``,
        ``break_even_analysis``, ``contribution_margin``,
        ``incremental_analysis``, ``sensitivity_analysis``,
        ``vacancy_cost``.
    """
    df = calculate_historical_margins(summary_df, salary_per_person, commission_rate, cogs_rate)

    stats = calculate_summary_statistics(df, salary_per_person, commission_rate, cogs_rate)

    report: dict[str, Any] = {
        "historical_margins": df,
        "summary_statistics": stats,
        "break_even_analysis": calculate_break_even_analysis(
            salary_per_person, commission_rate, cogs_rate
        ),
        "contribution_margin": calculate_contribution_margin(
            df, salary_per_person, commission_rate, cogs_rate
        ),
        "incremental_analysis": calculate_incremental_analysis(
            df, salary_per_person, commission_rate, cogs_rate
        ),
        "sensitivity_analysis": run_sensitivity_analysis(
            base_sales=stats["avg_sales"],
            base_employees=round(stats["avg_employees"]),
            base_salary=salary_per_person,
            base_commission=commission_rate,
            cogs_rate=cogs_rate,
        ),
        "vacancy_cost": estimate_vacancy_cost(
            df, open_positions, salary_per_person, commission_rate, cogs_rate
        ),
    }

    return report
