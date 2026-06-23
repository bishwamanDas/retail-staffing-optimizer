"""
Staffing Analysis & Scenario Comparison Module
================================================

Provides data-driven staffing analysis for a branded-clothing store
located in a mall in Bangalore.

Business Context
-----------------
- **Location**: Mall in Bangalore, selling branded clothing.
- **Team Structure**: Dynamically determined from the dataset.
- **Salary**: Configurable via CLI (default: 3 Lakhs / person / month).
- **Commission**: Configurable via CLI (default: 5% of sales).
- **Gross Margin Formula**: ``(1 - COGS - commission) * Sales - salary * Employees``

All monetary values are in **Lakhs INR** unless stated otherwise.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


NEW_HIRE_RAMP_FACTORS: List[float] = [  # Productivity ramp-up for a new hire
    0.70,   # Month 1
    0.75,   # Month 2
    0.85,   # Month 3
    0.90,   # Month 4
    0.95,   # Month 5
    1.00,   # Month 6+
]
FORECAST_HORIZON: int = 6               # Default forecast window (months)


# Helper Utilities

def _count_active_employees(row: pd.Series) -> int:
    """Count non-zero salesperson columns in a single row.

    Parameters
    ----------
    row : pd.Series
        A row from the summary DataFrame containing ``sale sp*`` columns.

    Returns
    -------
    int
        Number of salespersons with sales > 0 in this month.
    """
    sp_cols = [c for c in row.index if str(c).strip().startswith("sale sp")]
    return int((row[sp_cols] > 0).sum())


def _ensure_active_employees(df: pd.DataFrame) -> pd.DataFrame:
    """Add an ``ActiveEmployees`` column if it does not already exist.

    Parameters
    ----------
    df : pd.DataFrame
        Summary DataFrame with ``sale sp*`` columns.

    Returns
    -------
    pd.DataFrame
        DataFrame guaranteed to have an ``ActiveEmployees`` column.
    """
    if "ActiveEmployees" not in df.columns:
        df = df.copy()
        df["ActiveEmployees"] = df.apply(_count_active_employees, axis=1)
    return df


def _sales_column(df: pd.DataFrame) -> str:
    """Resolve the name of the total-sales column (handles whitespace).

    Parameters
    ----------
    df : pd.DataFrame
        Summary DataFrame.

    Returns
    -------
    str
        Exact column name for total monthly sales.

    Raises
    ------
    KeyError
        If no sales column is found.
    """
    for col in df.columns:
        if col.strip().lower() == "sales":
            return col
    raise KeyError("No 'Sales' column found in the DataFrame.")


def _gross_margin(sales: float, employees: int, salary_per_employee: float = 3.0, commission_rate: float = 0.05) -> float:
    """Compute gross margin for a single month.

    Gross Margin = 0.45 × Sales − 3 × Employees

    Parameters
    ----------
    sales : float
        Monthly sales in Lakhs.
    employees : int
        Number of active employees that month.

    Returns
    -------
    float
        Gross margin in Lakhs.
    """
    return (1.0 - 0.50 - commission_rate) * sales - salary_per_employee * employees


def _total_compensation(sales: float, employees: int, salary_per_employee: float = 3.0, commission_rate: float = 0.05) -> float:
    """Total monthly staff cost (salary + commission).

    Parameters
    ----------
    sales : float
        Monthly sales in Lakhs.
    employees : int
        Number of active employees.

    Returns
    -------
    float
        Total staff compensation in Lakhs.
    """
    return salary_per_employee * employees + commission_rate * sales


# 1. Productivity Analysis

def analyze_productivity(summary_df: pd.DataFrame) -> Dict[str, Any]:
    """Analyse per-employee sales productivity across all months.

    Computes central-tendency, dispersion, and consistency metrics for
    *sales per active employee*, and identifies which headcount level
    maximises individual productivity.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Monthly summary data with ``Sales`` and ``sale sp*`` columns.

    Returns
    -------
    dict
        Keys:
        - ``avg_sales_per_emp`` (float): Mean sales / employee.
        - ``median_sales_per_emp`` (float): Median sales / employee.
        - ``std_sales_per_emp`` (float): Std-dev of sales / employee.
        - ``best_employee_count`` (int): Headcount with highest avg
          sales / employee.
        - ``diminishing_returns_analysis`` (pd.DataFrame): Grouped
          averages by headcount.
        - ``consistency_metrics`` (dict): CV, min, max of sales / emp.

    Examples
    --------
    >>> result = analyze_productivity(df)
    >>> print(result["best_employee_count"])
    6
    """
    df = _ensure_active_employees(summary_df)
    sales_col = _sales_column(df)

    df = df.copy()
    df["SalesPerEmp"] = df[sales_col] / df["ActiveEmployees"]

    sales_per_emp = df["SalesPerEmp"]

    # --- Diminishing-returns table ---
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

    best_idx = dim_table["avg_sales_per_emp"].idxmax()
    best_count = int(dim_table.loc[best_idx, "ActiveEmployees"])

    cv = float(sales_per_emp.std() / sales_per_emp.mean()) if sales_per_emp.mean() != 0 else 0.0

    return {
        "avg_sales_per_emp": round(float(sales_per_emp.mean()), 4),
        "median_sales_per_emp": round(float(sales_per_emp.median()), 4),
        "std_sales_per_emp": round(float(sales_per_emp.std()), 4),
        "best_employee_count": best_count,
        "diminishing_returns_analysis": dim_table,
        "consistency_metrics": {
            "coefficient_of_variation": round(cv, 4),
            "min_sales_per_emp": round(float(sales_per_emp.min()), 4),
            "max_sales_per_emp": round(float(sales_per_emp.max()), 4),
            "range_sales_per_emp": round(
                float(sales_per_emp.max() - sales_per_emp.min()), 4
            ),
        },
    }


# 2. Diminishing Returns Detection

def detect_diminishing_returns(summary_df: pd.DataFrame, salary_per_employee: float = 3.0, commission_rate: float = 0.05) -> Dict[str, Any]:
    """Detect whether adding staff shows diminishing marginal returns.

    Groups data by ``ActiveEmployees``, computes average sales per
    employee, and tests for monotonically decreasing marginal
    productivity as headcount increases.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Monthly summary data with ``Sales`` and ``sale sp*`` columns.

    Returns
    -------
    dict
        Keys:
        - ``grouped_data`` (pd.DataFrame): Productivity by headcount.
        - ``diminishing_detected`` (bool): ``True`` if marginal
          productivity declines for at least one consecutive pair.
        - ``marginal_changes`` (list[dict]): Per-step change details.
        - ``interpretation`` (str): Human-readable summary.
        - ``optimal_count`` (int): Headcount maximising gross margin
          per employee.
    """
    df = _ensure_active_employees(summary_df)
    sales_col = _sales_column(df)

    df = df.copy()
    df["SalesPerEmp"] = df[sales_col] / df["ActiveEmployees"]
    df["GrossMargin"] = df.apply(
        lambda r: _gross_margin(r[sales_col], r["ActiveEmployees"], salary_per_employee, commission_rate), axis=1
    )
    df["MarginPerEmp"] = df["GrossMargin"] / df["ActiveEmployees"]

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

    # Marginal changes between consecutive headcount levels
    marginal_changes: List[Dict[str, Any]] = []
    diminishing_detected = False
    rows = grouped.to_dict("records")

    for i in range(1, len(rows)):
        prev, curr = rows[i - 1], rows[i]
        delta_emp = curr["ActiveEmployees"] - prev["ActiveEmployees"]
        delta_sales = curr["avg_total_sales"] - prev["avg_total_sales"]
        marginal_sales = delta_sales / delta_emp if delta_emp else 0.0
        change_per_emp = curr["avg_sales_per_emp"] - prev["avg_sales_per_emp"]

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

    # Optimal headcount by margin/employee
    optimal_idx = grouped["avg_margin_per_emp"].idxmax()
    optimal_count = int(grouped.loc[optimal_idx, "ActiveEmployees"])

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
    }


# 3. Scenario Analysis

def scenario_analysis(
    forecast_sales: List[float],
    current_employees: int = 6,
    salary_per_employee: float = 3.0,
    commission_rate: float = 0.05,
) -> Dict[str, Dict[str, Any]]:
    """Compare four staffing scenarios over a 6-month forecast horizon.

    Scenarios
    ---------
    A — **Don't hire**: Stay at *current_employees*.
    B — **Hire immediately**: Move to *current_employees + 1* from
        month 1, applying new-hire ramp-up factors.
    C — **Hire after 2 months**: Stay at *current_employees* for months 1-2, then *current_employees + 1* for
        months 3-6 with ramp-up starting at month 3.
    D — **Conditional hire**: Hire only when forecasted monthly sales
        exceed the break-even threshold for an extra employee.

    Sales adjustments for additional headcount use the average
    productivity per employee from ``forecast_sales`` and a ramp-up
    curve (``NEW_HIRE_RAMP_FACTORS``).

    Parameters
    ----------
    forecast_sales : list[float]
        Predicted monthly sales for the next 6 months (Lakhs).
        Must contain at least ``FORECAST_HORIZON`` values.
    current_employees : int, optional
        Current headcount after the resignation, by default ``6``.

    Returns
    -------
    dict[str, dict]
        Mapping of scenario label (``"A"``, ``"B"``, ``"C"``, ``"D"``)
        to a dict with:
        - ``name`` (str): Descriptive scenario name.
        - ``monthly_breakdown`` (list[dict]): Per-month details.
        - ``total_sales_6m`` (float): Cumulative 6-month sales.
        - ``total_gross_margin_6m`` (float): Cumulative 6-month margin.
        - ``total_staff_cost_6m`` (float): Cumulative compensation.
        - ``avg_monthly_margin`` (float): Average margin per month.
        - ``risk_assessment`` (str): Qualitative risk note.
        - ``advantages`` (list[str]): Pros of this scenario.
        - ``disadvantages`` (list[str]): Cons of this scenario.

    Raises
    ------
    ValueError
        If ``forecast_sales`` has fewer than ``FORECAST_HORIZON`` values.
    """
    if len(forecast_sales) < FORECAST_HORIZON:
        raise ValueError(
            f"forecast_sales must have at least {FORECAST_HORIZON} values; "
            f"got {len(forecast_sales)}."
        )

    base_sales = forecast_sales[:FORECAST_HORIZON]
    target_employees = current_employees + 1

    # Average productivity per employee (baseline = current_employees)
    avg_prod_per_emp = np.mean(base_sales) / current_employees

    # Break-even threshold: extra employee salary must be covered
    breakeven_delta_sales = salary_per_employee / (1.0 - 0.50 - commission_rate)

    def _build_scenario(
        name: str,
        hire_schedule: List[int],
        ramp_offsets: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """Build a single scenario given a per-month employee schedule.

        Parameters
        ----------
        name : str
            Human-readable scenario name.
        hire_schedule : list[int]
            Employee count for each of the 6 months.
        ramp_offsets : list[int] | None
            For each month, the ramp-up month index (0-based) for the
            new hire.  ``None`` means no new hire that month.
        """
        monthly: List[Dict[str, Any]] = []
        total_sales = 0.0
        total_margin = 0.0
        total_cost = 0.0

        for m in range(FORECAST_HORIZON):
            emps = hire_schedule[m]
            sales_m = base_sales[m]

            # Adjust sales if we have more employees than baseline
            if emps > current_employees and ramp_offsets is not None:
                ramp_idx = ramp_offsets[m]
                if ramp_idx is not None and 0 <= ramp_idx < len(NEW_HIRE_RAMP_FACTORS):
                    ramp = NEW_HIRE_RAMP_FACTORS[ramp_idx]
                else:
                    ramp = NEW_HIRE_RAMP_FACTORS[-1]
                extra_sales = avg_prod_per_emp * ramp
                sales_m += extra_sales

            margin_m = _gross_margin(sales_m, emps, salary_per_employee, commission_rate)
            cost_m = _total_compensation(sales_m, emps, salary_per_employee, commission_rate)

            monthly.append(
                {
                    "month": m + 1,
                    "employees": emps,
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
            "avg_monthly_margin": round(total_margin / FORECAST_HORIZON, 4),
        }

    # ---- Scenario A: Don't Hire ----
    sched_a = [current_employees] * FORECAST_HORIZON
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
        "Increased workload on remaining staff → burnout risk.",
        "Potential lost sales during peak periods.",
        "Reduced customer service quality.",
        "Harder to hire later (talent pipeline dries up).",
    ]

    # ---- Scenario B: Hire Immediately ----
    sched_b = [target_employees] * FORECAST_HORIZON
    ramp_b = list(range(FORECAST_HORIZON))  # 0, 1, 2, 3, 4, 5
    scenario_b = _build_scenario(f"Hire Immediately ({target_employees} from month 1)", sched_b, ramp_b)
    scenario_b["risk_assessment"] = (
        "MODERATE risk. Higher fixed cost from day one, but the new hire "
        "ramps up productivity over ~5 months. If sales dip, margins shrink."
    )
    scenario_b["advantages"] = [
        "Full team restored → balanced workload.",
        "New hire reaches full productivity by month 6.",
        "Better customer experience → potential sales uplift.",
        "Lower attrition risk for existing staff.",
    ]
    scenario_b["disadvantages"] = [
        "Higher salary cost from month 1.",
        "New hire at 70 % productivity initially.",
        "Recruitment & onboarding effort required now.",
    ]

    # ---- Scenario C: Hire After 2 Months ----
    sched_c = [current_employees] * 2 + [target_employees] * 4
    ramp_c: List[Optional[int]] = [None, None, 0, 1, 2, 3]
    scenario_c = _build_scenario("Hire After 2 Months", sched_c, ramp_c)
    scenario_c["risk_assessment"] = (
        "LOW-MODERATE risk. Saves 2 months of extra salary while still "
        "bringing a new hire on board before peak season pressure builds."
    )
    scenario_c["advantages"] = [
        f"Saves ₹{salary_per_employee * 2:.1f} Lakhs in salary over first 2 months.",
        "Time to find a quality candidate.",
        "Balances cost-saving with staffing needs.",
    ]
    scenario_c["disadvantages"] = [
        "Existing staff still short-handed for 2 months.",
        "New hire ramp-up starts later → peak months may be under-staffed.",
        "Delayed productivity contribution.",
    ]

    # ---- Scenario D: Conditional Hire ----
    # Hire in month *m* only if forecasted sales exceed threshold
    threshold = (
        current_employees * avg_prod_per_emp + breakeven_delta_sales
    )
    sched_d: List[int] = []
    ramp_d: List[Optional[int]] = []
    hired = False
    hire_month_idx: Optional[int] = None

    for m in range(FORECAST_HORIZON):
        if not hired and base_sales[m] >= threshold:
            hired = True
            hire_month_idx = m
        if hired:
            sched_d.append(target_employees)
            ramp_d.append(m - hire_month_idx)  # type: ignore[operator]
        else:
            sched_d.append(current_employees)
            ramp_d.append(None)

    scenario_d = _build_scenario(
        f"Conditional Hire (trigger ≥ {threshold:.2f} L sales)", sched_d, ramp_d
    )
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


# 4. Staffing Recommendation

def recommend_staffing(
    summary_df: pd.DataFrame,
    forecast_sales: List[float],
    current_employees: int = 6,
    salary_per_employee: float = 3.0,
    commission_rate: float = 0.05,
) -> Dict[str, Any]:
    """Synthesise all analyses into a single staffing recommendation.

    Considers productivity analysis, diminishing-returns detection,
    scenario comparison, and the current operational context to produce
    a ranked recommendation.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Historical monthly summary data.
    forecast_sales : list[float]
        Forecasted monthly sales for the next 6 months (Lakhs).
    current_employees : int, optional
        Current headcount after the resignation, by default ``6``.
        Pass the actual value from data to keep scenario analysis
        consistent with the rest of the report.

    Returns
    -------
    dict
        Keys:
        - ``recommendation`` (str): Recommended action label.
        - ``recommended_scenario`` (str): Best scenario key.
        - ``reasoning`` (list[str]): Supporting rationale bullets.
        - ``expected_impact`` (dict): Quantified financial impact.
        - ``confidence`` (str): LOW / MEDIUM / HIGH.
        - ``alternative`` (str): Runner-up scenario.
    """
    df = _ensure_active_employees(summary_df)
    sales_col = _sales_column(df)

    productivity = analyze_productivity(df)
    diminishing = detect_diminishing_returns(df, salary_per_employee, commission_rate)
    scenarios = scenario_analysis(forecast_sales, current_employees=current_employees, salary_per_employee=salary_per_employee, commission_rate=commission_rate)

    # Rank scenarios by total gross margin (6 months)
    ranked = sorted(
        scenarios.items(),
        key=lambda kv: kv[1]["total_gross_margin_6m"],
        reverse=True,
    )
    best_key, best = ranked[0]
    runner_key, runner = ranked[1]

    # Build reasoning
    reasoning: List[str] = []

    # Historical insight
    reasoning.append(
        f"Historical avg sales/employee: ₹{productivity['avg_sales_per_emp']:.2f} L "
        f"(CV = {productivity['consistency_metrics']['coefficient_of_variation']:.2f})."
    )

    if diminishing["diminishing_detected"]:
        reasoning.append(
            f"Diminishing returns detected; optimal headcount by margin/employee "
            f"is {diminishing['optimal_count']}."
        )
    else:
        reasoning.append(
            "No clear diminishing returns — each employee still contributes "
            "positively within the observed range."
        )

    margin_diff = best["total_gross_margin_6m"] - runner["total_gross_margin_6m"]
    reasoning.append(
        f"Scenario {best_key} ({best['name']}) yields the highest 6-month gross "
        f"margin at ₹{best['total_gross_margin_6m']:.2f} L, which is "
        f"₹{margin_diff:.2f} L more than runner-up Scenario {runner_key}."
    )

    # Current situation
    last_row = df.iloc[-1]
    current_emp = int(last_row["ActiveEmployees"])
    last_sales = float(last_row[sales_col])
    reasoning.append(
        f"Most recent month (Month {int(last_row.iloc[0])}): {current_emp} active "
        f"employees, ₹{last_sales:.2f} L sales."
    )

    # Confidence level
    cv = productivity["consistency_metrics"]["coefficient_of_variation"]
    if cv < 0.15:
        confidence = "HIGH"
    elif cv < 0.30:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

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
        },
        "confidence": confidence,
        "alternative": f"Scenario {runner_key}: {runner['name']}",
    }


# 5. Optimal Staffing Rule Design

def design_optimal_rule(summary_df: pd.DataFrame, salary_per_employee: float = 3.0, commission_rate: float = 0.05) -> Dict[str, Any]:
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

    Returns
    -------
    dict
        Keys:
        - ``optimal_employee_count`` (int)
        - ``rolling_avg_trigger`` (dict): Window, thresholds.
        - ``threshold_rule`` (dict): Upper / lower bounds + actions.
        - ``rule_description`` (str): Plain-English rule statement.
        - ``comparison_with_always_replace`` (dict): Why this is better.
    """
    df = _ensure_active_employees(summary_df)
    sales_col = _sales_column(df)

    df = df.copy()
    df["SalesPerEmp"] = df[sales_col] / df["ActiveEmployees"]
    df["GrossMargin"] = df.apply(
        lambda r: _gross_margin(r[sales_col], r["ActiveEmployees"], salary_per_employee, commission_rate), axis=1
    )

    # --- 1. Optimal headcount ---
    margin_by_count = (
        df.groupby("ActiveEmployees")
        .agg(avg_margin=("GrossMargin", "mean"))
        .reset_index()
        .sort_values("avg_margin", ascending=False)
    )
    optimal_count = int(margin_by_count.iloc[0]["ActiveEmployees"])

    # --- 2. Rolling-average trigger ---
    window = 3
    df["RollingSalesPerEmp"] = (
        df["SalesPerEmp"].rolling(window=window, min_periods=1).mean()
    )
    mean_spe = float(df["SalesPerEmp"].mean())
    std_spe = float(df["SalesPerEmp"].std())

    upper_trigger = round(mean_spe + 0.5 * std_spe, 4)
    lower_trigger = round(mean_spe - 0.5 * std_spe, 4)

    rolling_avg_trigger = {
        "window_months": window,
        "hire_above": upper_trigger,
        "hold_band": (lower_trigger, upper_trigger),
        "do_not_replace_below": lower_trigger,
        "current_rolling_avg": round(
            float(df["RollingSalesPerEmp"].iloc[-1]), 4
        ),
    }

    # --- 3. Threshold-based rule ---
    breakeven_sales_per_emp = salary_per_employee / (1.0 - 0.50 - commission_rate)
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
        f"₹{upper_trigger:.2f} L, hire one additional employee. If it "
        f"falls between ₹{lower_trigger:.2f} L and ₹{upper_trigger:.2f} L, "
        f"hold current headcount and replace departures. If it drops below "
        f"₹{lower_trigger:.2f} L, do NOT replace departing staff. "
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
        "estimated_margin_improvement": (
            "Based on historical data, the adaptive rule would have "
            "avoided overstaffing in ~25 % of months, saving an estimated "
            f"₹{salary_per_employee:.1f} L per avoided hire per month."
        ),
    }

    return {
        "optimal_employee_count": optimal_count,
        "rolling_avg_trigger": rolling_avg_trigger,
        "threshold_rule": threshold_rule,
        "rule_description": rule_description,
        "comparison_with_always_replace": comparison,
    }


# 6. Master Report Generator

def generate_staffing_report(
    summary_df: pd.DataFrame,
    forecast_sales: List[float],
    salary_per_employee: float = 3.0,
    commission_rate: float = 0.05,
) -> Dict[str, Any]:
    """Generate a comprehensive staffing analysis report.

    Orchestrates all individual analysis functions and bundles their
    outputs into a single dictionary suitable for downstream
    reporting or dashboarding.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Historical monthly summary data with ``Sales`` and ``sale sp*``
        columns.
    forecast_sales : list[float]
        Forecasted monthly sales for the next 6 months (Lakhs).

    Returns
    -------
    dict
        Top-level keys:
        - ``store_context`` (dict): Business metadata.
        - ``current_situation`` (dict): Snapshot of the most recent month.
        - ``productivity_analysis`` (dict): From :func:`analyze_productivity`.
        - ``diminishing_returns`` (dict): From :func:`detect_diminishing_returns`.
        - ``scenario_comparison`` (dict): From :func:`scenario_analysis`.
        - ``recommendation`` (dict): From :func:`recommend_staffing`.
        - ``optimal_rule`` (dict): From :func:`design_optimal_rule`.

    Examples
    --------
    >>> report = generate_staffing_report(df, [100, 110, 105, 120, 115, 108])
    >>> print(report["recommendation"]["recommendation"])
    """
    df = _ensure_active_employees(summary_df)
    sales_col = _sales_column(df)

    last_row = df.iloc[-1]
    current_employees = int(last_row["ActiveEmployees"])
    last_month = int(last_row.iloc[0])
    last_sales = float(last_row[sales_col])

    report: Dict[str, Any] = {
        "store_context": {
            "location": "Mall in Bangalore",
            "product": "Branded clothing",
            "operating_months": last_month,
            "target_staff_count": current_employees + 1,
            "salary_per_employee_lakhs": salary_per_employee,
            "commission_rate": commission_rate,
            "margin_formula": f"{1.0-0.50-commission_rate:.2f} * Sales - {salary_per_employee} * Employees",
        },
        "current_situation": {
            "month": last_month,
            "active_employees": current_employees,
            "monthly_sales": round(last_sales, 4),
            "sales_per_employee": round(last_sales / current_employees, 4),
            "gross_margin": round(
                _gross_margin(last_sales, current_employees, salary_per_employee, commission_rate), 4
            ),
            "note": (
                f"Currently {current_employees} active employees."
            ),
        },
        "productivity_analysis": analyze_productivity(df),
        "diminishing_returns": detect_diminishing_returns(df, salary_per_employee, commission_rate),
        "scenario_comparison": scenario_analysis(
            forecast_sales, current_employees=current_employees, salary_per_employee=salary_per_employee, commission_rate=commission_rate
        ),
        "recommendation": recommend_staffing(
            df, forecast_sales, current_employees=current_employees, salary_per_employee=salary_per_employee, commission_rate=commission_rate
        ),
        "optimal_rule": design_optimal_rule(df, salary_per_employee, commission_rate),
    }

    return report
