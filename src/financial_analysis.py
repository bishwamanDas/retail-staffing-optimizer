"""Financial analysis module for branded-clothing store staffing decisions.

Provides functions to compute gross margins, break-even points,
contribution margins, incremental employee impact, sensitivity
scenarios, six-month projections, and a comprehensive financial report.

Gross Margin formula
--------------------
    GrossMargin = (1 - COGS_rate - commission_rate) × Sales − salary × Employees

    where salary, commission_rate are configurable via CLI arguments
    and COGS_rate defaults to 0.50 (50%).

All monetary values are in **Lakhs INR** unless stated otherwise.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


# Helper - margin calculation
def _compute_gross_margin(sales: Union[float, pd.Series],
                          employees: Union[int, pd.Series],
                          salary_per_person: float = 3.0,
                          commission_rate: float = 0.05) -> Union[float, pd.Series]:
    """Return gross margin using the standard formula.

    Parameters
    ----------
    sales : float or pd.Series
        Monthly sales in Lakhs INR.
    employees : int or pd.Series
        Number of active employees for that month.

    Returns
    -------
    float or pd.Series
        Gross margin in Lakhs INR.
    """
    return (1.0 - 0.50 - commission_rate) * sales - salary_per_person * employees


# 1. calculate_historical_margins
def calculate_historical_margins(summary_df: pd.DataFrame, salary_per_person: float = 3.0, commission_rate: float = 0.05) -> pd.DataFrame:
    """Add a ``GrossMargin`` column to *summary_df* if it is not already present.

    The margin is computed as::

        GrossMargin = 0.45 × Sales − 3 × ActiveEmployees

    Parameters
    ----------
    summary_df : pd.DataFrame
        Must contain at least ``Sales`` and ``ActiveEmployees`` columns.

    Returns
    -------
    pd.DataFrame
        The same DataFrame with the ``GrossMargin`` column added (or
        unchanged if it already existed).

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
    if "GrossMargin" not in df.columns:
        df["GrossMargin"] = _compute_gross_margin(df["Sales"], df["ActiveEmployees"], salary_per_person, commission_rate)
    return df


# 2. calculate_summary_statistics
def calculate_summary_statistics(summary_df: pd.DataFrame, salary_per_person: float = 3.0, commission_rate: float = 0.05) -> Dict[str, float]:
    """Compute key summary statistics from the monthly data.

    If ``GrossMargin`` is not yet a column, it is computed on-the-fly
    without mutating the input DataFrame.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Must contain ``Sales`` and ``ActiveEmployees`` columns.

    Returns
    -------
    dict
        Keys:

        * ``avg_sales``          – mean monthly sales
        * ``median_sales``       – median monthly sales
        * ``std_sales``          – standard deviation of monthly sales
        * ``avg_margin``         – mean monthly gross margin
        * ``median_margin``      – median monthly gross margin
        * ``std_margin``         – standard deviation of gross margins
        * ``avg_employees``      – mean monthly active employee count
        * ``avg_sales_per_emp``  – average sales per employee
        * ``total_margin``       – cumulative gross margin over all months
        * ``min_margin``         – worst monthly gross margin
        * ``max_margin``         – best monthly gross margin
    """
    df = calculate_historical_margins(summary_df, salary_per_person, commission_rate)

    sales: pd.Series = df["Sales"]
    margin: pd.Series = df["GrossMargin"]
    employees: pd.Series = df["ActiveEmployees"]

    avg_employees = employees.mean()

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
    }


# 3. calculate_break_even_analysis
def calculate_break_even_analysis(salary_per_person: float = 3.0, commission_rate: float = 0.05) -> Dict[str, float]:
    """Return break-even metrics for one employee.

    Break-even sales per employee::

        salary / effective_margin_rate = 3 / 0.45 ≈ 6.667 Lakhs

    Returns
    -------
    dict
        Keys:

        * ``break_even_sales_per_employee`` – sales needed to cover one
          employee's salary (≈ 6.667 Lakhs)
        * ``monthly_salary_cost`` – fixed salary per person (3 Lakhs)
        * ``commission_rate`` – sales commission fraction (0.05)
        * ``effective_margin_rate`` – net margin rate after COGS and
          commission (0.45)
    """
    effective_margin_rate = 1.0 - 0.50 - commission_rate
    return {
        "break_even_sales_per_employee": salary_per_person / effective_margin_rate,
        "monthly_salary_cost": salary_per_person,
        "commission_rate": commission_rate,
        "effective_margin_rate": effective_margin_rate,
    }


# 4. calculate_contribution_margin
def calculate_contribution_margin(summary_df: pd.DataFrame, salary_per_person: float = 3.0, commission_rate: float = 0.05) -> pd.DataFrame:
    """Compute average contribution margin **per employee** at each staffing level.

    Groups the data by ``ActiveEmployees`` and calculates:

    * mean sales
    * mean gross margin
    * mean contribution margin per employee (= gross margin / employees)
    * number of months observed at that level

    Parameters
    ----------
    summary_df : pd.DataFrame
        Must contain ``Sales`` and ``ActiveEmployees`` columns.

    Returns
    -------
    pd.DataFrame
        Indexed by ``ActiveEmployees`` with columns
        ``AvgSales``, ``AvgGrossMargin``, ``AvgContributionPerEmployee``,
        and ``MonthsObserved``.
    """
    df = calculate_historical_margins(summary_df, salary_per_person, commission_rate)

    grouped = df.groupby("ActiveEmployees").agg(
        AvgSales=("Sales", "mean"),
        AvgGrossMargin=("GrossMargin", "mean"),
        MonthsObserved=("GrossMargin", "count"),
    )

    grouped["AvgContributionPerEmployee"] = (
        grouped["AvgGrossMargin"] / grouped.index
    )

    return grouped.reset_index()


# 5. calculate_incremental_analysis
def calculate_incremental_analysis(summary_df: pd.DataFrame, salary_per_person: float = 3.0, commission_rate: float = 0.05) -> pd.DataFrame:
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
        Columns:

        * ``ActiveEmployees``
        * ``AvgSales``
        * ``AvgGrossMargin``
        * ``AvgSalesPerEmployee``
        * ``IncrementalSales``   – change in avg sales vs. previous level
        * ``IncrementalMargin``  – change in avg margin vs. previous level
    """
    df = calculate_historical_margins(summary_df, salary_per_person, commission_rate)

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

    return grouped.reset_index()


# 6. run_sensitivity_analysis
def run_sensitivity_analysis(
    base_sales: float,
    base_employees: int,
    base_salary: float = 3.0,
    base_commission: float = 0.05,
    salary_range: Tuple[float, float, float] = (2.0, 4.0, 0.5),
    commission_range: Tuple[float, float, float] = (0.03, 0.07, 0.01),
) -> pd.DataFrame:
    """Test gross-margin sensitivity to salary and commission assumptions.

    For every combination of salary and commission rate within the
    specified ranges, the gross margin is recalculated as::

        margin = (1 − 0.50 − commission) × sales − salary × employees

    Parameters
    ----------
    base_sales : float
        Reference monthly sales figure (Lakhs INR).
    base_employees : int
        Reference employee count.
    salary_range : tuple of (start, stop, step)
        Salary costs to test (Lakhs/person/month).  *stop* is inclusive.
    commission_range : tuple of (start, stop, step)
        Commission rates to test (fraction).  *stop* is inclusive.

    Returns
    -------
    pd.DataFrame
        One row per scenario with columns:

        * ``Salary``          – salary assumption (Lakhs)
        * ``CommissionRate``   – commission assumption
        * ``EffectiveMarginRate`` – resulting margin rate
        * ``GrossMargin``     – resulting gross margin (Lakhs)
        * ``BreakEvenSalesPerEmp`` – break-even sales per employee
        * ``MarginVsBase``    – difference from the base-case margin
    """
    cogs_rate = 0.50  # fixed cost-of-goods-sold fraction

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

    base_margin = _compute_gross_margin(base_sales, base_employees, base_salary, base_commission)

    rows: List[Dict[str, float]] = []
    for salary in salaries:
        for comm in commissions:
            eff_rate = 1.0 - cogs_rate - comm
            margin = eff_rate * base_sales - salary * base_employees
            be_sales = salary / eff_rate if eff_rate > 0 else np.inf
            rows.append(
                {
                    "Salary": round(float(salary), 2),
                    "CommissionRate": round(float(comm), 4),
                    "EffectiveMarginRate": round(float(eff_rate), 4),
                    "GrossMargin": round(float(margin), 4),
                    "BreakEvenSalesPerEmp": round(float(be_sales), 4),
                    "MarginVsBase": round(float(margin - base_margin), 4),
                }
            )

    return pd.DataFrame(rows)


# 7. calculate_six_month_projection
def calculate_six_month_projection(
    forecast_sales: List[float],
    num_employees: int,
    salary_per_person: float = 3.0,
    commission_rate: float = 0.05,
) -> Dict[str, Any]:
    """Project total gross margin over a six-month horizon.

    Parameters
    ----------
    forecast_sales : list of float
        Exactly six monthly sales forecasts (Lakhs INR).
    num_employees : int
        Fixed employee headcount for the projection period.
    salary_per_person : float, optional
        Monthly salary cost per employee (default 3 Lakhs).

    Returns
    -------
    dict
        Keys:

        * ``forecast_sales``       – input sales list
        * ``num_employees``        – headcount used
        * ``salary_per_person``    – salary assumption
        * ``monthly_margins``      – list of per-month gross margins
        * ``total_sales``          – sum of forecasted sales
        * ``total_salary_cost``    – total salary outlay over 6 months
        * ``total_gross_margin``   – cumulative gross margin
        * ``avg_monthly_margin``   – mean monthly margin
        * ``break_even_month``     – 1-based index of first profitable
          month, or ``None`` if none are profitable

    Raises
    ------
    ValueError
        If *forecast_sales* does not contain exactly six values.
    """
    if len(forecast_sales) != 6:
        raise ValueError(
            f"Expected exactly 6 monthly forecasts, got {len(forecast_sales)}."
        )

    eff_rate = 1.0 - 0.50 - commission_rate
    monthly_margins = [
        eff_rate * s - salary_per_person * num_employees
        for s in forecast_sales
    ]

    total_sales = sum(forecast_sales)
    total_salary = salary_per_person * num_employees * 6
    total_margin = sum(monthly_margins)
    avg_margin = total_margin / 6

    # First month where cumulative margin turns positive
    cumulative = 0.0
    break_even_month: Optional[int] = None
    for idx, m in enumerate(monthly_margins, start=1):
        cumulative += m
        if cumulative > 0 and break_even_month is None:
            break_even_month = idx

    return {
        "forecast_sales": forecast_sales,
        "num_employees": num_employees,
        "salary_per_person": salary_per_person,
        "monthly_margins": monthly_margins,
        "total_sales": round(total_sales, 4),
        "total_salary_cost": round(total_salary, 4),
        "total_gross_margin": round(total_margin, 4),
        "avg_monthly_margin": round(avg_margin, 4),
        "break_even_month": break_even_month,
    }


# 8. generate_financial_report
def generate_financial_report(summary_df: pd.DataFrame, salary_per_person: float = 3.0, commission_rate: float = 0.05) -> Dict[str, Any]:
    """Run **all** financial analyses and bundle results into one dictionary.

    This is the master entry-point for the module.  It calls every
    analysis function defined above and returns a single nested
    dictionary suitable for serialisation or downstream reporting.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Must contain ``Sales`` and ``ActiveEmployees`` columns.

    Returns
    -------
    dict
        Top-level keys:

        * ``historical_margins``    – DataFrame with ``GrossMargin`` added
        * ``summary_statistics``    – output of :func:`calculate_summary_statistics`
        * ``break_even_analysis``   – output of :func:`calculate_break_even_analysis`
        * ``contribution_margin``   – output of :func:`calculate_contribution_margin`
        * ``incremental_analysis``  – output of :func:`calculate_incremental_analysis`
        * ``sensitivity_analysis``  – output of :func:`run_sensitivity_analysis`
          (uses historical averages as the base case)
    """
    # Ensure margins exist
    df = calculate_historical_margins(summary_df, salary_per_person, commission_rate)

    # Summary stats first - we need them for the sensitivity base case
    stats = calculate_summary_statistics(df, salary_per_person, commission_rate)

    report: Dict[str, Any] = {
        "historical_margins": df,
        "summary_statistics": stats,
        "break_even_analysis": calculate_break_even_analysis(salary_per_person, commission_rate),
        "contribution_margin": calculate_contribution_margin(df, salary_per_person, commission_rate),
        "incremental_analysis": calculate_incremental_analysis(df, salary_per_person, commission_rate),
        "sensitivity_analysis": run_sensitivity_analysis(
            base_sales=stats["avg_sales"],
            base_employees=round(stats["avg_employees"]),
            base_salary=salary_per_person,
            base_commission=commission_rate,
        ),
    }

    return report
