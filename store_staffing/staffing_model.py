"""Staffing analysis, optimisation and scenario modelling.

Business context
-----------------
* **Location**: Mall in Bangalore, selling branded clothing.
* **Team structure**: Dynamically determined from the dataset.
* **Salary / commission**: Configurable (defaults 3 Lakhs per person per
  month and 5 % of sales).
* **Gross margin**: ``(1 - COGS - commission) * Sales - salary * Employees``.

Beyond the original scenario comparison this module adds:

* a **grid-search headcount optimiser** that chooses the target headcount
  that maximises expected margin over the forecast horizon,
* **probabilistic scenario scoring** (P10/P50/P90 and the probability that
  a scenario beats the do-nothing baseline),
* an **honest backtest** of the adaptive staffing rule against the
  "always replace" policy, so the claimed savings are computed rather
  than asserted.

All monetary values are in **Lakhs INR** unless stated otherwise.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "NEW_HIRE_RAMP_FACTORS",
    "FORECAST_HORIZON",
    "analyze_productivity",
    "detect_diminishing_returns",
    "scenario_analysis",
    "probabilistic_scenario_analysis",
    "optimize_headcount",
    "recommend_staffing",
    "design_optimal_rule",
    "backtest_staffing_rules",
    "generate_staffing_report",
]

# --------------------------------------------------------------------------- #
# Constants                                                                    #
# --------------------------------------------------------------------------- #
NEW_HIRE_RAMP_FACTORS: list[float] = [
    0.70,  # Month 1
    0.75,  # Month 2
    0.85,  # Month 3
    0.90,  # Month 4
    0.95,  # Month 5
    1.00,  # Month 6+
]
"""Productivity ramp-up multipliers for a newly hired employee."""

FORECAST_HORIZON: int = 6
"""Default forecast window (months)."""

COGS_RATE: float = 0.50
"""Cost-of-goods-sold rate applied to sales."""

Z_95: float = 1.959963985
"""Two-sided 95 % normal quantile used to derive sigma from an interval."""


# --------------------------------------------------------------------------- #
# Helper utilities                                                             #
# --------------------------------------------------------------------------- #
def _count_active_employees(row: pd.Series) -> int:
    """Count salesperson columns with positive sales in a single row.

    Parameters
    ----------
    row : pd.Series
        A row containing ``sale sp*`` columns.

    Returns
    -------
    int
        Number of salespersons with sales > 0 this month.
    """
    sp_cols = [c for c in row.index if str(c).strip().lower().startswith("sale sp")]
    if not sp_cols:
        return 0
    return int((row[sp_cols] > 0).sum())


def _ensure_active_employees(df: pd.DataFrame) -> pd.DataFrame:
    """Add an ``ActiveEmployees`` column if it does not already exist."""
    if "ActiveEmployees" not in df.columns:
        df = df.copy()
        df["ActiveEmployees"] = df.apply(_count_active_employees, axis=1)
    return df


def _sales_column(df: pd.DataFrame) -> str:
    """Resolve the name of the total-sales column (handles whitespace).

    Raises
    ------
    KeyError
        If no sales column is found.
    """
    for col in df.columns:
        if str(col).strip().lower() == "sales":
            return col
    raise KeyError("No 'Sales' column found in the DataFrame.")


def _month_column(df: pd.DataFrame) -> str:
    """Resolve the name of the month-index column (handles whitespace).

    Raises
    ------
    KeyError
        If no month column is found.
    """
    for col in df.columns:
        if str(col).strip().lower() == "month":
            return col
    raise KeyError("No 'Month' column found in the DataFrame.")


def _gross_margin(
    sales: float,
    employees: int,
    salary_per_employee: float = 3.0,
    commission_rate: float = 0.05,
    cogs_rate: float = COGS_RATE,
) -> float:
    """Compute gross margin for a single month.

    ``GrossMargin = (1 - COGS - commission) x Sales - salary x Employees``
    """
    return (1.0 - cogs_rate - commission_rate) * sales - salary_per_employee * employees


def _total_compensation(
    sales: float,
    employees: int,
    salary_per_employee: float = 3.0,
    commission_rate: float = 0.05,
) -> float:
    """Total monthly staff cost (salary + commission)."""
    return salary_per_employee * employees + commission_rate * sales


def _ramp_factor(index: int, ramp: Sequence[float]) -> float:
    """Return the productivity ramp factor for month *index* (0-based)."""
    if not ramp:
        return 1.0
    if index < 0:
        index = 0
    if index >= len(ramp):
        return float(ramp[-1])
    return float(ramp[index])


# --------------------------------------------------------------------------- #
# 1b. Marginal productivity (controlled regression)                            #
# --------------------------------------------------------------------------- #
def _headcount_bounds(df: pd.DataFrame) -> tuple[int, int]:
    """Return the plausible headcount range to search.

    The marginal effect of one extra employee is only identified over the
    headcount levels the store has actually operated, so the optimiser is
    bounded by the observed range extended by a single additional hire.

    Parameters
    ----------
    df : pd.DataFrame
        Monthly data with an ``ActiveEmployees`` column (or ``sale sp*``
        columns).

    Returns
    -------
    tuple of int
        ``(low, high)`` inclusive bounds.
    """
    data = _ensure_active_employees(df)
    levels = data.loc[data["ActiveEmployees"] > 0, "ActiveEmployees"]
    if levels.empty:
        return (1, 2)
    return (max(1, int(levels.min())), int(levels.max()) + 1)


def estimate_marginal_productivity(
    df: pd.DataFrame,
    salary_per_employee: float = 3.0,
    commission_rate: float = 0.05,
    cogs_rate: float = COGS_RATE,
) -> dict[str, Any]:
    """Estimate the sales contributed by *one additional employee*.

    A naive comparison of average sales at each observed headcount is
    badly confounded: the months that happened to run with eight staff are
    not a random sample of months, and the store grew over the sample
    period, so raw averages attribute growth to headcount.

    Two controlled specifications are tried, in order:

    1. **First differences** (preferred)::

           dSales ~ trend + month-of-year effects + dHeadcount

       Differencing removes the secular trend entirely, so the
       headcount coefficient captures only the change in sales that
       coincides with a change in staffing.
    2. **Levels with controls** (fallback)::

           Sales ~ trend + month-of-year fixed effects + Headcount

    The implied marginal gross-margin contribution is
    ``margin_rate x coefficient - salary``.

    Parameters
    ----------
    df : pd.DataFrame
        Monthly data containing ``Month``, ``Sales`` and either
        ``ActiveEmployees`` or the ``sale sp*`` columns.
    salary_per_employee : float, optional
        Monthly salary per employee.
    commission_rate : float, optional
        Commission rate.
    cogs_rate : float, optional
        Cost of goods sold rate.

    Returns
    -------
    dict
        Keys: ``marginal_sales_per_employee``,
        ``marginal_margin_contribution``, ``p_value``, ``r_squared``,
        ``n_observations``, ``model`` (``"first_difference"``,
        ``"levels_ols"`` or ``"fallback"``), ``observed_marginal_step``,
        ``note``.
    """
    data = _ensure_active_employees(df.copy())
    sales_col = _sales_column(data)
    month_col = _month_column(data)

    data = data[[month_col, sales_col, "ActiveEmployees"]].copy()
    data = data.rename(
        columns={month_col: "Month", sales_col: "Sales", "ActiveEmployees": "Headcount"}
    )
    data = data.dropna(subset=["Month", "Sales", "Headcount"])
    data = data[data["Headcount"] > 0].sort_values("Month").reset_index(drop=True)
    if len(data) < 12 or data["Headcount"].nunique() < 2:
        logger.info("Not enough variation for a controlled marginal estimate.")
        return {
            "marginal_sales_per_employee": None,
            "marginal_margin_contribution": None,
            "p_value": None,
            "r_squared": None,
            "n_observations": int(len(data)),
            "model": "fallback",
            "observed_marginal_step": None,
            "note": "Insufficient headcount variation to estimate a controlled marginal effect.",
        }

    data["trend"] = data["Month"].astype(float)
    data["MonthOfYear"] = (((data["Month"].astype(int) - 1) % 12) + 1).astype(str)

    margin_rate = 1.0 - cogs_rate - commission_rate

    # --- observed (uncontrolled) marginal step, used as a reference ----- #
    means = data.groupby("Headcount")["Sales"].mean().sort_index()
    observed_step: float | None = None
    if len(means) >= 2:
        levels = list(means.index)
        last, previous = levels[-1], levels[-2]
        observed_step = round(float((means[last] - means[previous]) / max(last - previous, 1)), 4)

    # --- specification 1: first differences ----------------------------- #
    diff = data.copy()
    diff["dSales"] = diff["Sales"].diff()
    diff["dHeadcount"] = diff["Headcount"].diff()
    diff = diff.dropna(subset=["dSales", "dHeadcount"])

    coefficient: float | None = None
    p_value: float | None = None
    r_squared: float | None = None
    model_name = "fallback"
    n_obs = int(len(diff))

    if len(diff) >= 20 and diff["dHeadcount"].abs().sum() > 0:
        try:
            import statsmodels.formula.api as smf

            fit = smf.ols("dSales ~ trend + C(MonthOfYear) + dHeadcount", data=diff).fit()
            candidate = float(fit.params.get("dHeadcount", float("nan")))
            if np.isfinite(candidate):
                coefficient = candidate
                p_value = float(fit.pvalues.get("dHeadcount", float("nan")))
                r_squared = float(fit.rsquared)
                model_name = "first_difference"
        except Exception as exc:  # noqa: BLE001
            logger.warning("First-difference regression failed: %s", exc)

    # --- specification 2: levels with controls -------------------------- #
    if coefficient is None:
        try:
            import statsmodels.formula.api as smf

            fit = smf.ols("Sales ~ trend + C(MonthOfYear) + Headcount", data=data).fit()
            candidate = float(fit.params.get("Headcount", float("nan")))
            if np.isfinite(candidate):
                coefficient = candidate
                p_value = float(fit.pvalues.get("Headcount", float("nan")))
                r_squared = float(fit.rsquared)
                model_name = "levels_ols"
                n_obs = int(len(data))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Levels regression failed: %s", exc)

    if coefficient is None:
        return {
            "marginal_sales_per_employee": None,
            "marginal_margin_contribution": None,
            "p_value": None,
            "r_squared": None,
            "n_observations": int(len(data)),
            "model": "fallback",
            "observed_marginal_step": observed_step,
            "note": "Regression failed; falling back to average productivity.",
        }

    marginal_margin = margin_rate * coefficient - salary_per_employee

    notes = {
        "first_difference": (
            "First-difference OLS: month-on-month changes in sales are "
            "regressed on month-on-month changes in headcount with a trend "
            "and month-of-year controls, so the secular growth of the store "
            "is differenced out before the staffing effect is measured."
        ),
        "levels_ols": (
            "Ordinary least squares controlling for a linear trend and "
            "month-of-year fixed effects; the headcount coefficient is the "
            "controlled marginal sales per additional employee."
        ),
        "fallback": "Regression failed; falling back to average productivity.",
    }

    return {
        "marginal_sales_per_employee": round(coefficient, 4),
        "marginal_margin_contribution": round(marginal_margin, 4),
        "p_value": round(p_value, 6) if p_value is not None and np.isfinite(p_value) else None,
        "r_squared": round(r_squared, 4) if r_squared is not None else None,
        "n_observations": n_obs,
        "model": model_name,
        "observed_marginal_step": observed_step,
        "note": notes.get(model_name, notes["fallback"]),
    }


# --------------------------------------------------------------------------- #
# 1. Productivity analysis                                                     #
# --------------------------------------------------------------------------- #
def analyze_productivity(summary_df: pd.DataFrame) -> dict[str, Any]:
    """Analyse per-employee sales productivity across all months.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Monthly summary data with ``Sales`` and ``sale sp*`` columns.

    Returns
    -------
    dict
        Keys: ``avg_sales_per_emp``, ``median_sales_per_emp``,
        ``std_sales_per_emp``, ``best_employee_count``,
        ``diminishing_returns_analysis``, ``consistency_metrics``,
        ``total_sales``, ``total_employee_months``.
    """
    df = _ensure_active_employees(summary_df)
    sales_col = _sales_column(df)

    df = df.copy()
    df["SalesPerEmp"] = df[sales_col] / df["ActiveEmployees"].replace(0, np.nan)

    sales_per_emp = df["SalesPerEmp"]

    dim_table = (
        df.groupby("ActiveEmployees")
        .agg(
            avg_sales_per_emp=("SalesPerEmp", "mean"),
            avg_total_sales=(sales_col, "mean"),
            month_count=("SalesPerEmp", "count"),
        )
        .reset_index()
        .sort_values("ActiveEmployees")
    )
    dim_table = dim_table[dim_table["ActiveEmployees"] > 0]

    if dim_table.empty:
        best_count = 0
    else:
        best_idx = dim_table["avg_sales_per_emp"].idxmax()
        best_count = int(dim_table.loc[best_idx, "ActiveEmployees"])

    mean_spe = float(sales_per_emp.mean()) if sales_per_emp.notna().any() else 0.0
    cv = float(sales_per_emp.std() / mean_spe) if mean_spe else 0.0

    return {
        "avg_sales_per_emp": round(mean_spe, 4),
        "median_sales_per_emp": round(float(sales_per_emp.median()), 4),
        "std_sales_per_emp": round(float(sales_per_emp.std()), 4),
        "best_employee_count": best_count,
        "diminishing_returns_analysis": dim_table,
        "consistency_metrics": {
            "coefficient_of_variation": round(cv, 4),
            "min_sales_per_emp": round(float(sales_per_emp.min()), 4),
            "max_sales_per_emp": round(float(sales_per_emp.max()), 4),
            "range_sales_per_emp": round(float(sales_per_emp.max() - sales_per_emp.min()), 4),
        },
        "total_sales": round(float(df[sales_col].sum()), 4),
        "total_employee_months": int(df["ActiveEmployees"].sum()),
    }


# --------------------------------------------------------------------------- #
# 2. Diminishing returns detection                                             #
# --------------------------------------------------------------------------- #
def detect_diminishing_returns(
    summary_df: pd.DataFrame,
    salary_per_employee: float = 3.0,
    commission_rate: float = 0.05,
    cogs_rate: float = COGS_RATE,
) -> dict[str, Any]:
    """Detect whether adding staff shows diminishing marginal returns.

    Groups data by ``ActiveEmployees``, computes average sales per
    employee, and tests for monotonically decreasing marginal
    productivity as headcount increases.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Monthly summary data with ``Sales`` and ``sale sp*`` columns.
    salary_per_employee : float, optional
        Monthly salary per employee.
    commission_rate : float, optional
        Commission rate.
    cogs_rate : float, optional
        Cost of goods sold rate.

    Returns
    -------
    dict
        Keys: ``grouped_data``, ``diminishing_detected``,
        ``marginal_changes``, ``interpretation``, ``optimal_count``,
        ``optimal_margin_count``.
    """
    df = _ensure_active_employees(summary_df)
    sales_col = _sales_column(df)

    df = df.copy()
    df["SalesPerEmp"] = df[sales_col] / df["ActiveEmployees"].replace(0, np.nan)
    df["GrossMargin"] = df.apply(
        lambda r: _gross_margin(
            r[sales_col], r["ActiveEmployees"], salary_per_employee, commission_rate, cogs_rate
        ),
        axis=1,
    )
    df["MarginPerEmp"] = df["GrossMargin"] / df["ActiveEmployees"].replace(0, np.nan)

    grouped = (
        df.groupby("ActiveEmployees")
        .agg(
            avg_sales_per_emp=("SalesPerEmp", "mean"),
            avg_total_sales=(sales_col, "mean"),
            avg_gross_margin=("GrossMargin", "mean"),
            avg_margin_per_emp=("MarginPerEmp", "mean"),
            month_count=("SalesPerEmp", "count"),
        )
        .reset_index()
        .sort_values("ActiveEmployees")
    )
    grouped = grouped[grouped["ActiveEmployees"] > 0]

    marginal_changes: list[dict[str, Any]] = []
    diminishing_detected = False
    rows = grouped.to_dict("records")

    for i in range(1, len(rows)):
        prev, curr = rows[i - 1], rows[i]
        delta_emp = int(curr["ActiveEmployees"]) - int(prev["ActiveEmployees"])
        delta_sales = float(curr["avg_total_sales"]) - float(prev["avg_total_sales"])
        marginal_sales = delta_sales / delta_emp if delta_emp else 0.0
        change_per_emp = float(curr["avg_sales_per_emp"]) - float(prev["avg_sales_per_emp"])

        if change_per_emp < 0:
            diminishing_detected = True

        marginal_changes.append(
            {
                "from_employees": int(prev["ActiveEmployees"]),
                "to_employees": int(curr["ActiveEmployees"]),
                "marginal_sales_per_added": round(marginal_sales, 4),
                "change_in_sales_per_emp": round(change_per_emp, 4),
                "diminishing": change_per_emp < 0,
            }
        )

    optimal_count = 0
    optimal_margin_count = 0
    if not grouped.empty:
        optimal_count = int(grouped.loc[grouped["avg_margin_per_emp"].idxmax(), "ActiveEmployees"])
        optimal_margin_count = int(
            grouped.loc[grouped["avg_gross_margin"].idxmax(), "ActiveEmployees"]
        )

    if diminishing_detected:
        interpretation = (
            "Diminishing returns DETECTED. At some headcount transitions, "
            "adding an employee reduces average sales per employee. "
            f"Margin-per-employee is maximised at {optimal_count} employees."
        )
    else:
        interpretation = (
            "No clear diminishing returns detected across observed headcount "
            "levels. Each additional employee still contributes positively to "
            "per-capita productivity within the observed range."
        )

    return {
        "grouped_data": grouped,
        "diminishing_detected": diminishing_detected,
        "marginal_changes": marginal_changes,
        "interpretation": interpretation,
        "optimal_count": optimal_count,
        "optimal_margin_count": optimal_margin_count,
    }


# --------------------------------------------------------------------------- #
# 3. Scenario analysis                                                         #
# --------------------------------------------------------------------------- #
def scenario_analysis(
    forecast_sales: list[float],
    current_employees: int = 6,
    salary_per_employee: float = 3.0,
    commission_rate: float = 0.05,
    cogs_rate: float = COGS_RATE,
    ramp: Sequence[float] | None = None,
    horizon: int = FORECAST_HORIZON,
    marginal_sales_per_employee: float | None = None,
) -> dict[str, dict[str, Any]]:
    """Compare four staffing scenarios over the forecast horizon.

    Scenarios
    ---------
    A — **Don't hire**: stay at *current_employees*.
    B — **Hire immediately**: move to *current_employees + 1* from month 1
        applying the new-hire ramp-up.
    C — **Hire after 2 months**: stay at *current_employees* for months
        1-2, then add one hire for months 3-6 with ramp-up starting then.
    D — **Conditional hire**: hire only once forecast sales exceed the
        threshold at which an extra head is covered by its own margin.

    Parameters
    ----------
    forecast_sales : list[float]
        Predicted monthly sales for the forecast horizon (Lakhs).
    current_employees : int, optional
        Current headcount after the resignation.
    salary_per_employee : float, optional
        Monthly salary per employee.
    commission_rate : float, optional
        Commission rate.
    cogs_rate : float, optional
        Cost of goods sold rate.
    ramp : sequence of float, optional
        New-hire productivity ramp factors (defaults to
        :data:`NEW_HIRE_RAMP_FACTORS`).
    horizon : int, optional
        Number of months to model (default 6).
    marginal_sales_per_employee : float, optional
        Sales contributed by one additional employee.  When omitted the
        average sales per employee is used.

    Returns
    -------
    dict
        Mapping of scenario label (``"A"`` … ``"D"``) to a dict with
        ``name``, ``monthly_breakdown``, ``total_sales_6m``,
        ``total_gross_margin_6m``, ``total_staff_cost_6m``,
        ``avg_monthly_margin``, ``risk_assessment``, ``advantages`` and
        ``disadvantages``.

    Raises
    ------
    ValueError
        If *forecast_sales* has fewer than *horizon* values.
    """
    if len(forecast_sales) < horizon:
        raise ValueError(
            f"forecast_sales must have at least {horizon} values; got {len(forecast_sales)}."
        )
    if current_employees <= 0:
        raise ValueError(f"current_employees must be > 0; got {current_employees}.")

    ramp_factors = list(ramp) if ramp else list(NEW_HIRE_RAMP_FACTORS)
    base_sales = [float(s) for s in forecast_sales[:horizon]]
    target_employees = current_employees + 1
    margin_rate = 1.0 - cogs_rate - commission_rate

    avg_prod_per_emp = float(np.mean(base_sales)) / current_employees
    if marginal_sales_per_employee is not None and marginal_sales_per_employee > 0:
        hire_productivity = float(marginal_sales_per_employee)
    else:
        hire_productivity = avg_prod_per_emp

    breakeven_delta_sales = salary_per_employee / margin_rate if margin_rate > 0 else float("inf")

    def _build_scenario(
        name: str,
        hire_schedule: list[int],
        ramp_offsets: list[int | None] | None = None,
    ) -> dict[str, Any]:
        """Build a single scenario given a per-month employee schedule.

        Parameters
        ----------
        name : str
            Human-readable scenario name.
        hire_schedule : list[int]
            Employee count for each month.
        ramp_offsets : list[int | None] | None
            For each month, the ramp-up month index (0-based) for the new
            hire.  ``None`` means no new hire that month.
        """
        monthly: list[dict[str, Any]] = []
        total_sales = 0.0
        total_margin = 0.0
        total_cost = 0.0

        for m in range(horizon):
            emps = hire_schedule[m]
            sales_m = base_sales[m]

            if emps > current_employees and ramp_offsets is not None:
                ramp_idx = ramp_offsets[m]
                sales_m += hire_productivity * _ramp_factor(
                    ramp_idx if ramp_idx is not None else 0, ramp_factors
                )

            margin_m = _gross_margin(sales_m, emps, salary_per_employee, commission_rate, cogs_rate)
            cost_m = _total_compensation(sales_m, emps, salary_per_employee, commission_rate)

            monthly.append(
                {
                    "month": m + 1,
                    "employees": int(emps),
                    "sales": round(sales_m, 4),
                    "gross_margin": round(margin_m, 4),
                    "staff_cost": round(cost_m, 4),
                    "net_contribution": round(margin_m, 4),
                }
            )

            total_sales += sales_m
            total_margin += margin_m
            total_cost += cost_m

        return {
            "name": name,
            "monthly_breakdown": monthly,
            "total_sales_6m": round(total_sales, 4),
            "total_gross_margin_6m": round(total_margin, 4),
            "total_staff_cost_6m": round(total_cost, 4),
            "avg_monthly_margin": round(total_margin / horizon, 4),
        }

    # ---- Scenario A: Don't Hire ---------------------------------------- #
    sched_a = [current_employees] * horizon
    scenario_a = _build_scenario(f"Don't Hire (Stay at {current_employees})", sched_a)
    scenario_a["risk_assessment"] = (
        "LOW financial risk but HIGH operational risk. Existing staff may "
        "burn out; customer service quality could decline during peak months."
    )
    scenario_a["advantages"] = [
        "Lowest fixed staff cost.",
        "No onboarding overhead.",
        "Immediate margin improvement if sales stay moderate.",
    ]
    scenario_a["disadvantages"] = [
        "Increased workload on remaining staff -> burnout risk.",
        "Potential lost sales during peak periods.",
        "Reduced customer service quality.",
        "Harder to hire later (talent pipeline dries up).",
    ]

    # ---- Scenario B: Hire Immediately ----------------------------------- #
    sched_b = [target_employees] * horizon
    ramp_b: list[int | None] = list(range(horizon))
    scenario_b = _build_scenario(
        f"Hire Immediately ({target_employees} from month 1)", sched_b, ramp_b
    )
    scenario_b["risk_assessment"] = (
        "MODERATE risk. Higher fixed cost from day one, but the new hire "
        f"ramps up productivity over {len(ramp_factors)} months. "
        "If sales dip, margins shrink."
    )
    scenario_b["advantages"] = [
        "Full team restored -> balanced workload.",
        f"New hire reaches full productivity by month {len(ramp_factors)}.",
        "Better customer experience -> potential sales uplift.",
        "Lower attrition risk for existing staff.",
    ]
    scenario_b["disadvantages"] = [
        "Higher salary cost from month 1.",
        f"New hire at {ramp_factors[0] * 100:.0f} % productivity initially.",
        "Recruitment & onboarding effort required now.",
    ]

    # ---- Scenario C: Hire After 2 Months -------------------------------- #
    wait_months = min(2, max(0, horizon - 1))
    sched_c = [current_employees] * wait_months + [target_employees] * (horizon - wait_months)
    ramp_c: list[int | None] = [None] * wait_months + list(range(horizon - wait_months))
    scenario_c = _build_scenario(f"Hire After {wait_months} Months", sched_c, ramp_c)
    scenario_c["risk_assessment"] = (
        f"LOW-MODERATE risk. Saves {wait_months} months of extra salary while still "
        "bringing a new hire on board before peak season pressure builds."
    )
    scenario_c["advantages"] = [
        f"Saves Rs.{salary_per_employee * wait_months:.1f} Lakhs in salary over the first "
        f"{wait_months} months.",
        "Time to find a quality candidate.",
        "Balances cost-saving with staffing needs.",
    ]
    scenario_c["disadvantages"] = [
        f"Existing staff still short-handed for {wait_months} months.",
        "New hire ramp-up starts later -> peak months may be under-staffed.",
        "Delayed productivity contribution.",
    ]

    # ---- Scenario D: Conditional Hire ----------------------------------- #
    threshold = current_employees * avg_prod_per_emp + breakeven_delta_sales
    sched_d: list[int] = []
    ramp_d: list[int | None] = []
    hired = False
    hire_month_idx: int | None = None

    for m in range(horizon):
        if not hired and base_sales[m] >= threshold:
            hired = True
            hire_month_idx = m
        if hired and hire_month_idx is not None:
            sched_d.append(target_employees)
            ramp_d.append(m - hire_month_idx)
        else:
            sched_d.append(current_employees)
            ramp_d.append(None)

    scenario_d = _build_scenario(
        f"Conditional Hire (trigger >= {threshold:.2f} L sales)", sched_d, ramp_d
    )
    scenario_d["threshold"] = round(threshold, 4)
    scenario_d["triggered"] = bool(hired)
    scenario_d["trigger_month"] = (hire_month_idx + 1) if hire_month_idx is not None else None
    scenario_d["risk_assessment"] = (
        "LOW risk. Data-driven decision minimises cost if sales are weak, "
        "but may result in delayed hiring when the market picks up."
    )
    scenario_d["advantages"] = [
        "Hiring only when sales justify the cost.",
        "Minimises downside in low-sales months.",
        "Aligns staffing with revenue reality.",
    ]
    scenario_d["disadvantages"] = [
        "Reactive rather than proactive.",
        "May miss early indicators of a sales upswing.",
        "Administrative complexity of a conditional rule.",
        f"Threshold ({threshold:.2f} L) may need periodic recalibration.",
    ]

    return {
        "A": scenario_a,
        "B": scenario_b,
        "C": scenario_c,
        "D": scenario_d,
    }


# --------------------------------------------------------------------------- #
# 4. Probabilistic scenario scoring                                            #
# --------------------------------------------------------------------------- #
def _normal_cdf(x: float) -> float:
    """Return the standard normal CDF at *x*."""
    from math import erf, sqrt

    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def probabilistic_scenario_analysis(
    scenarios: dict[str, dict[str, Any]],
    forecast_lower: Sequence[float],
    forecast_upper: Sequence[float],
    commission_rate: float = 0.05,
    cogs_rate: float = COGS_RATE,
) -> dict[str, Any]:
    """Attach probabilistic statistics to each scenario.

    The forecast interval is converted into a per-month sigma and the
    resulting margin variance is propagated through each scenario, giving
    P10/P50/P90 total-margin bands and the probability that each scenario
    beats the do-nothing baseline.

    Parameters
    ----------
    scenarios : dict
        Output of :func:`scenario_analysis`.
    forecast_lower, forecast_upper : sequence of float
        Lower and upper prediction bounds aligned with the forecast horizon.
    commission_rate : float, optional
        Commission rate.
    cogs_rate : float, optional
        Cost of goods sold rate.

    Returns
    -------
    dict
        Keys: ``scenarios`` (the input enriched in place and returned),
        ``baseline``, ``best``, ``spread``.
    """
    margin_rate = 1.0 - cogs_rate - commission_rate
    lower = np.asarray(list(forecast_lower), dtype=float)
    upper = np.asarray(list(forecast_upper), dtype=float)
    sigma = np.maximum((upper - lower) / (2.0 * Z_95), 0.0)

    enriched: dict[str, dict[str, Any]] = {}
    for key, scenario in scenarios.items():
        monthly = scenario.get("monthly_breakdown", [])
        # Each additional employee beyond the baseline contributes the same
        # absolute sales uncertainty, so only the headcount delta scales it.
        headcounts = [m.get("employees", 0) for m in monthly]
        base_headcount = headcounts[0] if headcounts else 0
        variances = [
            (margin_rate * sigma[i] * (h / max(base_headcount, 1))) ** 2
            for i, h in enumerate(headcounts)
            if i < len(sigma)
        ]
        std = float(np.sqrt(sum(variances))) if variances else 0.0
        mean = float(scenario.get("total_gross_margin_6m", 0.0))

        scenario["margin_std"] = round(std, 4)
        scenario["p10_margin"] = round(mean - 1.2816 * std, 4)
        scenario["p50_margin"] = round(mean, 4)
        scenario["p90_margin"] = round(mean + 1.2816 * std, 4)
        scenario["prob_positive"] = round(
            _normal_cdf(mean / std) if std > 0 else (1.0 if mean > 0 else 0.0), 4
        )
        enriched[key] = scenario

    baseline_key = "A"
    baseline = enriched.get(baseline_key, {})
    base_mean = float(baseline.get("total_gross_margin_6m", 0.0))
    base_std = float(baseline.get("margin_std", 0.0))

    for scenario in enriched.values():
        mean = float(scenario.get("total_gross_margin_6m", 0.0))
        std = float(scenario.get("margin_std", 0.0))
        diff_std = float(np.sqrt(std**2 + base_std**2))
        if diff_std > 0:
            prob_better = _normal_cdf((mean - base_mean) / diff_std)
        else:
            prob_better = 1.0 if mean > base_mean else (0.5 if mean == base_mean else 0.0)
        scenario["prob_better_than_baseline"] = round(prob_better, 4)
        scenario["delta_vs_baseline"] = round(mean - base_mean, 4)

    ranked = sorted(enriched.items(), key=lambda kv: kv[1]["total_gross_margin_6m"], reverse=True)
    best_key = ranked[0][0] if ranked else baseline_key
    spread = (
        float(ranked[0][1]["total_gross_margin_6m"] - ranked[-1][1]["total_gross_margin_6m"])
        if len(ranked) > 1
        else 0.0
    )

    return {
        "scenarios": enriched,
        "baseline": baseline_key,
        "best": best_key,
        "spread": round(spread, 4),
    }


# --------------------------------------------------------------------------- #
# 5. Headcount optimiser                                                       #
# --------------------------------------------------------------------------- #
def optimize_headcount(
    forecast_sales: Sequence[float],
    current_employees: int,
    salary_per_employee: float = 3.0,
    commission_rate: float = 0.05,
    cogs_rate: float = COGS_RATE,
    ramp: Sequence[float] | None = None,
    min_headcount: int = 1,
    max_headcount: int = 30,
    horizon: int = FORECAST_HORIZON,
    marginal_sales_per_employee: float | None = None,
    productivity_curve: dict[int, float] | None = None,
    headcount_bounds: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Grid-search the headcount that maximises expected margin.

    Sales capacity is modelled with the **controlled marginal effect** of
    one additional employee (see
    :func:`estimate_marginal_productivity`).  For a candidate headcount
    *t* the expected sales are::

        sales(t, m) = base_sales(m) + marginal x (t - current) x ramp(m)

    where ``ramp(m)`` is the average productivity ramp of the newly hired
    staff.  Cutting the team applies the effect immediately, because
    existing capacity cannot be "ramped down".

    Parameters
    ----------
    forecast_sales : sequence of float
        Forecasted monthly sales at the *current* headcount.
    current_employees : int
        Current headcount.
    salary_per_employee : float, optional
        Monthly salary per employee.
    commission_rate : float, optional
        Commission rate.
    cogs_rate : float, optional
        Cost of goods sold rate.
    ramp : sequence of float, optional
        New-hire productivity ramp factors.
    min_headcount : int, optional
        Lower bound of the search grid.
    max_headcount : int, optional
        Upper bound of the search grid.
    horizon : int, optional
        Number of months to model.
    marginal_sales_per_employee : float, optional
        Controlled marginal sales per additional employee.  When omitted
        the average sales per employee is used.
    productivity_curve : dict, optional
        Observed headcount -> average sales per employee.  Used only as a
        fallback when no marginal estimate is available.
    headcount_bounds : tuple of int, optional
        ``(low, high)`` limits for the search grid.  The marginal effect
        is only identified inside the headcount range the store has
        actually operated, so the search is bounded by the observed
        range (plus one extra hire) by default.

    Returns
    -------
    dict
        Keys: ``optimal_headcount``, ``optimal_total_margin``,
        ``curve`` (one row per candidate), ``avg_sales_per_employee``,
        ``marginal_sales_per_employee``, ``interpretation``.
    """
    ramp_factors = list(ramp) if ramp else list(NEW_HIRE_RAMP_FACTORS)
    base = [float(s) for s in forecast_sales[:horizon]]
    if not base:
        return {
            "optimal_headcount": current_employees,
            "optimal_total_margin": 0.0,
            "curve": [],
            "interpretation": "No forecast available.",
        }

    1.0 - cogs_rate - commission_rate
    curve_map = dict(productivity_curve or {})
    avg_prod = float(np.mean(base)) / max(current_employees, 1)

    if marginal_sales_per_employee is not None and marginal_sales_per_employee > 0:
        marginal = float(marginal_sales_per_employee)
        marginal_source = "controlled regression estimate"
    elif curve_map:
        # Fall back to the last observed marginal step of the curve.
        levels = sorted(curve_map)
        if len(levels) >= 2:
            last, previous = levels[-1], levels[-2]
            marginal = max(
                0.0,
                (levels[-1] * curve_map[levels[-1]] - previous * curve_map[previous])
                / max(last - previous, 1),
            )
        else:
            marginal = avg_prod
        marginal_source = "observed productivity curve"
    else:
        marginal = avg_prod
        marginal_source = "average sales per employee"

    def _ramp_progress(month_index: int, hires: int) -> float:
        """Return the average ramp factor of *hires* new staff at a month."""
        if hires <= 0:
            return 0.0
        factors = [
            _ramp_factor(month_index - h, ramp_factors)
            for h in range(hires)
            if month_index - h >= 0
        ]
        return float(np.mean(factors)) if factors else 0.0

    curve: list[dict[str, Any]] = []
    best_headcount = current_employees
    best_margin = -float("inf")

    lo = max(1, int(min_headcount))
    hi = max(lo, int(max_headcount))
    if headcount_bounds is not None:
        lo = max(lo, int(headcount_bounds[0]))
        hi = max(lo, int(headcount_bounds[1]))
    for target in range(lo, hi + 1):
        total_margin = 0.0
        for m in range(horizon):
            if target > current_employees:
                hires = target - current_employees
                sales_m = base[m] + marginal * hires * _ramp_progress(m, hires)
            else:
                # Capacity reductions apply immediately.
                sales_m = base[m] + marginal * (target - current_employees)
            total_margin += _gross_margin(
                sales_m, target, salary_per_employee, commission_rate, cogs_rate
            )

        curve.append(
            {
                "headcount": target,
                "total_margin": round(total_margin, 4),
                "avg_monthly_margin": round(total_margin / horizon, 4),
                "margin_per_employee": round(total_margin / horizon / max(target, 1), 4),
            }
        )
        if total_margin > best_margin:
            best_margin = total_margin
            best_headcount = target

    return {
        "optimal_headcount": int(best_headcount),
        "optimal_total_margin": round(float(best_margin), 4),
        "curve": curve,
        "avg_sales_per_employee": round(avg_prod, 4),
        "marginal_sales_per_employee": round(marginal, 4),
        "marginal_source": marginal_source,
        "interpretation": (
            f"Maximising expected gross margin over the {horizon}-month horizon "
            f"gives an optimal headcount of {best_headcount} (expected margin "
            f"Rs.{best_margin:.2f} L), using a marginal contribution of "
            f"Rs.{marginal:.2f} L per employee ({marginal_source})."
        ),
    }


# --------------------------------------------------------------------------- #
# 6. Staffing recommendation                                                   #
# --------------------------------------------------------------------------- #
def recommend_staffing(
    summary_df: pd.DataFrame,
    forecast_sales: list[float],
    current_employees: int = 6,
    salary_per_employee: float = 3.0,
    commission_rate: float = 0.05,
    cogs_rate: float = COGS_RATE,
    scenarios: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Synthesise all analyses into a single staffing recommendation.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Historical monthly summary data.
    forecast_sales : list[float]
        Forecasted monthly sales for the horizon (Lakhs).
    current_employees : int, optional
        Current headcount after the resignation.
    salary_per_employee : float, optional
        Monthly salary per employee.
    commission_rate : float, optional
        Commission rate.
    cogs_rate : float, optional
        Cost of goods sold rate.
    scenarios : dict, optional
        Pre-computed scenarios (avoids recomputation).

    Returns
    -------
    dict
        Keys: ``recommendation``, ``recommended_scenario``, ``reasoning``,
        ``expected_impact``, ``confidence``, ``alternative``,
        ``confidence_drivers``.
    """
    df = _ensure_active_employees(summary_df)
    sales_col = _sales_column(df)
    month_col = _month_column(df)

    productivity = analyze_productivity(df)
    diminishing = detect_diminishing_returns(df, salary_per_employee, commission_rate, cogs_rate)
    if scenarios is None:
        scenarios = scenario_analysis(
            forecast_sales,
            current_employees=current_employees,
            salary_per_employee=salary_per_employee,
            commission_rate=commission_rate,
            cogs_rate=cogs_rate,
        )

    ranked = sorted(
        scenarios.items(),
        key=lambda kv: kv[1]["total_gross_margin_6m"],
        reverse=True,
    )
    best_key, best = ranked[0]
    runner_key, runner = ranked[1]

    reasoning: list[str] = []
    reasoning.append(
        f"Historical avg sales/employee: Rs.{productivity['avg_sales_per_emp']:.2f} L "
        f"(CV = {productivity['consistency_metrics']['coefficient_of_variation']:.2f})."
    )

    if diminishing["diminishing_detected"]:
        reasoning.append(
            "Diminishing returns detected; optimal headcount by margin/employee "
            f"is {diminishing['optimal_count']}."
        )
    else:
        reasoning.append(
            "No clear diminishing returns - each employee still contributes "
            "positively within the observed range."
        )

    margin_diff = best["total_gross_margin_6m"] - runner["total_gross_margin_6m"]
    baseline_margin = scenarios.get("A", {}).get("total_gross_margin_6m", 0.0)
    baseline_diff = best["total_gross_margin_6m"] - baseline_margin
    reasoning.append(
        f"Scenario {best_key} ({best['name']}) yields the highest 6-month gross "
        f"margin at Rs.{best['total_gross_margin_6m']:.2f} L, which is "
        f"Rs.{margin_diff:.2f} L more than runner-up Scenario {runner_key} and "
        f"Rs.{baseline_diff:.2f} L more than not hiring at all."
    )

    last_row = df.iloc[-1]
    current_emp = int(last_row["ActiveEmployees"])
    last_sales = float(last_row[sales_col])
    reasoning.append(
        f"Most recent month (Month {int(last_row[month_col])}): {current_emp} active "
        f"employees, Rs.{last_sales:.2f} L sales."
    )

    cv = productivity["consistency_metrics"]["coefficient_of_variation"]
    if cv < 0.15:
        confidence = "HIGH"
    elif cv < 0.30:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    confidence_drivers = {
        "sales_consistency_cv": round(cv, 4),
        "historical_months": int(len(df)),
        "headcount_observations": int(df["ActiveEmployees"].nunique()),
        "margin_gap_to_runner_up": round(margin_diff, 4),
        "note": (
            "Confidence is driven by sales consistency (coefficient of "
            "variation), the depth of history and how clearly the winning "
            "scenario separates from the runner-up."
        ),
    }

    return {
        "recommendation": best["name"],
        "recommended_scenario": best_key,
        "reasoning": reasoning,
        "expected_impact": {
            "total_sales_6m": best["total_sales_6m"],
            "total_gross_margin_6m": best["total_gross_margin_6m"],
            "total_staff_cost_6m": best["total_staff_cost_6m"],
            "avg_monthly_margin": best["avg_monthly_margin"],
            "margin_advantage_over_runner_up": round(margin_diff, 4),
            "margin_advantage_vs_do_nothing": round(baseline_diff, 4),
        },
        "confidence": confidence,
        "confidence_drivers": confidence_drivers,
        "alternative": f"Scenario {runner_key}: {runner['name']}",
    }


# --------------------------------------------------------------------------- #
# 7. Optimal staffing rule design                                             #
# --------------------------------------------------------------------------- #
def design_optimal_rule(
    summary_df: pd.DataFrame,
    salary_per_employee: float = 3.0,
    commission_rate: float = 0.05,
    cogs_rate: float = COGS_RATE,
) -> dict[str, Any]:
    """Design a data-driven staffing rule superior to 'always replace'.

    Uses historical data to determine:

    1. **Optimal headcount** that maximises gross margin.
    2. **Rolling-average trigger**: hire/fire based on a 3-month rolling
       average of sales per employee.
    3. **Threshold-based rule**: hire when sales per employee exceed an
       upper bound, hold when in a neutral band, and do not replace when
       below a lower bound.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Historical monthly summary data.
    salary_per_employee : float, optional
        Monthly salary per employee.
    commission_rate : float, optional
        Commission rate.
    cogs_rate : float, optional
        Cost of goods sold rate.

    Returns
    -------
    dict
        Keys: ``optimal_employee_count``, ``rolling_avg_trigger``,
        ``threshold_rule``, ``rule_description``,
        ``comparison_with_always_replace``.
    """
    df = _ensure_active_employees(summary_df)
    sales_col = _sales_column(df)

    df = df.copy()
    df["SalesPerEmp"] = df[sales_col] / df["ActiveEmployees"].replace(0, np.nan)
    df["GrossMargin"] = df.apply(
        lambda r: _gross_margin(
            r[sales_col], r["ActiveEmployees"], salary_per_employee, commission_rate, cogs_rate
        ),
        axis=1,
    )

    margin_by_count = (
        df.groupby("ActiveEmployees")
        .agg(avg_margin=("GrossMargin", "mean"))
        .reset_index()
        .sort_values("avg_margin", ascending=False)
    )
    optimal_count = (
        int(margin_by_count.iloc[0]["ActiveEmployees"]) if not margin_by_count.empty else 0
    )

    window = 3
    df["RollingSalesPerEmp"] = df["SalesPerEmp"].rolling(window=window, min_periods=1).mean()
    mean_spe = float(df["SalesPerEmp"].mean())
    std_spe = float(df["SalesPerEmp"].std())

    upper_trigger = round(mean_spe + 0.5 * std_spe, 4)
    lower_trigger = round(mean_spe - 0.5 * std_spe, 4)

    rolling_avg_trigger = {
        "window_months": window,
        "hire_above": upper_trigger,
        "hold_band": (lower_trigger, upper_trigger),
        "do_not_replace_below": lower_trigger,
        "current_rolling_avg": round(float(df["RollingSalesPerEmp"].iloc[-1]), 4),
    }

    breakeven_sales_per_emp = salary_per_employee / (1.0 - cogs_rate - commission_rate)
    p25 = float(df["SalesPerEmp"].quantile(0.25))
    p75 = float(df["SalesPerEmp"].quantile(0.75))

    threshold_rule = {
        "hire_threshold": round(p75, 4),
        "hold_range": (round(p25, 4), round(p75, 4)),
        "no_replace_threshold": round(p25, 4),
        "breakeven_sales_per_emp": round(breakeven_sales_per_emp, 4),
        "actions": {
            f"sales_per_emp >= {p75:.2f}": "HIRE an additional employee.",
            f"{p25:.2f} <= sales_per_emp < {p75:.2f}": "HOLD current headcount.",
            f"sales_per_emp < {p25:.2f}": "DO NOT replace departing employee.",
        },
    }

    rule_description = (
        f"Adaptive Staffing Rule: Maintain a 3-month rolling average of "
        f"sales per employee. If the rolling average exceeds "
        f"Rs.{upper_trigger:.2f} L, hire one additional employee. If it "
        f"falls between Rs.{lower_trigger:.2f} L and Rs.{upper_trigger:.2f} L, "
        f"hold current headcount and replace departures. If it drops below "
        f"Rs.{lower_trigger:.2f} L, do NOT replace departing staff. "
        f"Optimal target headcount based on historical margins: "
        f"{optimal_count} employees."
    )

    comparison = {
        "always_replace_flaw": (
            "The 'always replace' rule ignores current sales levels.  "
            "During low-sales months the extra salary erodes margin."
        ),
        "adaptive_rule_advantage": (
            "The adaptive rule ties staffing decisions to actual revenue "
            "performance, avoiding unnecessary costs in lean months and "
            "scaling up when the market supports it."
        ),
        "savings_evidence": (
            "Run `backtest_staffing_rules` for the measured backtested "
            "saving rather than an assumed percentage."
        ),
    }

    return {
        "optimal_employee_count": optimal_count,
        "rolling_avg_trigger": rolling_avg_trigger,
        "threshold_rule": threshold_rule,
        "rule_description": rule_description,
        "comparison_with_always_replace": comparison,
    }


# --------------------------------------------------------------------------- #
# 8. Rule backtesting                                                          #
# --------------------------------------------------------------------------- #
def backtest_staffing_rules(
    summary_df: pd.DataFrame,
    salary_per_employee: float = 3.0,
    commission_rate: float = 0.05,
    cogs_rate: float = COGS_RATE,
    target_headcount: int | None = None,
) -> dict[str, Any]:
    """Backtest the adaptive staffing rule against 'always replace'.

    Both policies are replayed over the historical record and their
    cumulative gross margins are compared.  The exercise assumes sales are
    exogenous (i.e. it prices *staffing decisions* against realised
    demand rather than modelling the demand response).

    Parameters
    ----------
    summary_df : pd.DataFrame
        Historical monthly summary data.
    salary_per_employee : float, optional
        Monthly salary per employee.
    commission_rate : float, optional
        Commission rate.
    cogs_rate : float, optional
        Cost of goods sold rate.
    target_headcount : int, optional
        Headcount maintained by the 'always replace' policy.  Defaults to
        the headcount that maximised historical margin.

    Returns
    -------
    dict
        Keys: ``always_replace``, ``adaptive``, ``improvement``,
        ``avoided_overstaffing_pct``, ``months_compared``,
        ``interpretation``.
    """
    df = _ensure_active_employees(summary_df)
    sales_col = _sales_column(df)

    df = df.copy()
    df["SalesPerEmp"] = df[sales_col] / df["ActiveEmployees"].replace(0, np.nan)
    df["GrossMargin"] = df.apply(
        lambda r: _gross_margin(
            r[sales_col], r["ActiveEmployees"], salary_per_employee, commission_rate, cogs_rate
        ),
        axis=1,
    )

    if target_headcount is None:
        by_count = df.groupby("ActiveEmployees")["GrossMargin"].mean().sort_values(ascending=False)
        target_headcount = int(by_count.index[0]) if len(by_count) else 1

    window = 3
    rolling = df["SalesPerEmp"].rolling(window=window, min_periods=1).mean()
    mean_spe = float(df["SalesPerEmp"].mean())
    std_spe = float(df["SalesPerEmp"].std())
    upper = mean_spe + 0.5 * std_spe
    lower = mean_spe - 0.5 * std_spe

    always_margin = 0.0
    adaptive_margin = 0.0
    avoided_months = 0
    compared = 0
    adaptive_headcount = target_headcount

    for idx in range(len(df)):
        sales = float(df.iloc[idx][sales_col])
        rolling_now = rolling.iloc[idx]
        if pd.isna(rolling_now):
            continue

        # --- adaptive policy ------------------------------------------- #
        if rolling_now > upper:
            adaptive_headcount = target_headcount + 1
        elif rolling_now < lower:
            adaptive_headcount = max(1, target_headcount - 1)
        else:
            adaptive_headcount = target_headcount

        adaptive_margin += _gross_margin(
            sales, adaptive_headcount, salary_per_employee, commission_rate, cogs_rate
        )
        always_margin += _gross_margin(
            sales, target_headcount, salary_per_employee, commission_rate, cogs_rate
        )

        if adaptive_headcount < target_headcount:
            avoided_months += 1
        compared += 1

    improvement = adaptive_margin - always_margin
    avoided_pct = (avoided_months / compared * 100.0) if compared else 0.0

    return {
        "always_replace": {
            "target_headcount": int(target_headcount),
            "total_margin": round(always_margin, 4),
            "avg_monthly_margin": round(always_margin / compared, 4) if compared else 0.0,
        },
        "adaptive": {
            "total_margin": round(adaptive_margin, 4),
            "avg_monthly_margin": round(adaptive_margin / compared, 4) if compared else 0.0,
        },
        "improvement": round(improvement, 4),
        "improvement_pct": round(
            (improvement / always_margin * 100.0) if always_margin else 0.0, 2
        ),
        "avoided_overstaffing_pct": round(avoided_pct, 2),
        "avoided_overstaffing_months": int(avoided_months),
        "months_compared": int(compared),
        "interpretation": (
            f"Replaying both policies over {compared} months, the adaptive rule "
            f"produced Rs.{improvement:.2f} L of gross margin "
            f"({round((improvement / always_margin * 100.0) if always_margin else 0.0, 1)}%) "
            f"more than always replacing, and avoided overstaffing in "
            f"{avoided_months} month(s) ({avoided_pct:.0f} % of the period)."
        ),
    }


# --------------------------------------------------------------------------- #
# 9. Master report generator                                                   #
# --------------------------------------------------------------------------- #
def generate_staffing_report(
    summary_df: pd.DataFrame,
    forecast_sales: list[float],
    salary_per_employee: float = 3.0,
    commission_rate: float = 0.05,
    cogs_rate: float = COGS_RATE,
    forecast_lower: Sequence[float] | None = None,
    forecast_upper: Sequence[float] | None = None,
    ramp: Sequence[float] | None = None,
    horizon: int = FORECAST_HORIZON,
) -> dict[str, Any]:
    """Generate a comprehensive staffing analysis report.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Historical monthly summary data with ``Sales`` and ``sale sp*`` columns.
    forecast_sales : list[float]
        Forecasted monthly sales for the horizon (Lakhs).
    salary_per_employee : float, optional
        Monthly salary per employee.
    commission_rate : float, optional
        Commission rate.
    cogs_rate : float, optional
        Cost of goods sold rate.
    forecast_lower, forecast_upper : sequence of float, optional
        Prediction bounds used for probabilistic scenario scoring.
    ramp : sequence of float, optional
        New-hire productivity ramp factors.
    horizon : int, optional
        Number of months to model (default 6).

    Returns
    -------
    dict
        Top-level keys: ``store_context``, ``current_situation``,
        ``productivity_analysis``, ``diminishing_returns``,
        ``scenario_comparison``, ``recommendation``, ``optimal_rule``,
        ``headcount_optimisation``, ``marginal_productivity``,
        ``rule_backtest``.
    """
    horizon = max(1, min(int(horizon), len(forecast_sales) or int(horizon)))

    df = _ensure_active_employees(summary_df)
    sales_col = _sales_column(df)
    month_col = _month_column(df)

    last_row = df.iloc[-1]
    current_employees = int(last_row["ActiveEmployees"])
    last_month = int(last_row[month_col])
    last_sales = float(last_row[sales_col])

    marginal = estimate_marginal_productivity(df, salary_per_employee, commission_rate, cogs_rate)
    marginal_sales = marginal.get("marginal_sales_per_employee")

    scenarios = scenario_analysis(
        forecast_sales,
        current_employees=current_employees,
        salary_per_employee=salary_per_employee,
        commission_rate=commission_rate,
        cogs_rate=cogs_rate,
        ramp=ramp,
        horizon=horizon,
        marginal_sales_per_employee=marginal_sales,
    )

    probabilistic: dict[str, Any] = {}
    if forecast_lower is not None and forecast_upper is not None:
        probabilistic = probabilistic_scenario_analysis(
            scenarios,
            forecast_lower,
            forecast_upper,
            commission_rate=commission_rate,
            cogs_rate=cogs_rate,
        )

    optimisation = optimize_headcount(
        forecast_sales,
        current_employees=current_employees,
        salary_per_employee=salary_per_employee,
        commission_rate=commission_rate,
        cogs_rate=cogs_rate,
        ramp=ramp,
        horizon=horizon,
        marginal_sales_per_employee=marginal_sales,
        headcount_bounds=_headcount_bounds(df),
    )

    report: dict[str, Any] = {
        "store_context": {
            "location": "Mall in Bangalore",
            "product": "Branded clothing",
            "operating_months": last_month,
            "target_staff_count": current_employees + 1,
            "salary_per_employee_lakhs": salary_per_employee,
            "commission_rate": commission_rate,
            "margin_formula": (
                f"{1.0 - cogs_rate - commission_rate:.2f} * Sales - "
                f"{salary_per_employee} * Employees"
            ),
        },
        "current_situation": {
            "month": last_month,
            "active_employees": current_employees,
            "monthly_sales": round(last_sales, 4),
            "sales_per_employee": round(last_sales / max(current_employees, 1), 4),
            "gross_margin": round(
                _gross_margin(
                    last_sales, current_employees, salary_per_employee, commission_rate, cogs_rate
                ),
                4,
            ),
            "note": f"Currently {current_employees} active employees.",
        },
        "productivity_analysis": analyze_productivity(df),
        "diminishing_returns": detect_diminishing_returns(
            df, salary_per_employee, commission_rate, cogs_rate
        ),
        "scenario_comparison": scenarios,
        "recommendation": recommend_staffing(
            df,
            forecast_sales,
            current_employees=current_employees,
            salary_per_employee=salary_per_employee,
            commission_rate=commission_rate,
            cogs_rate=cogs_rate,
            scenarios=scenarios,
        ),
        "optimal_rule": design_optimal_rule(df, salary_per_employee, commission_rate, cogs_rate),
        "headcount_optimisation": optimisation,
        "marginal_productivity": marginal,
        "rule_backtest": backtest_staffing_rules(
            df, salary_per_employee, commission_rate, cogs_rate
        ),
    }

    if probabilistic:
        report["probabilistic"] = probabilistic

    return report
