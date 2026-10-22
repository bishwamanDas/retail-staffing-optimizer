"""Professional narrative generation for the staffing recommendation.

Turns the structured analysis payload into a board-ready plain-text
report.  Every figure is read from the analysis results — nothing is
hard-coded — so the narrative stays truthful when the assumptions change.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["build_recommendation_text", "build_headline"]

WIDTH: int = 78
"""Character width of the generated text report."""


def _rule(char: str = "-") -> str:
    """Return a horizontal rule of the report width."""
    return char * WIDTH


def _fmt(value: Any, prefix: str = "Rs.", suffix: str = " L", digits: int = 2) -> str:
    """Format a monetary value, tolerating ``None`` and non-numeric input."""
    if isinstance(value, (int, float)):
        return f"{prefix}{value:,.{digits}f}{suffix}"
    return str(value)


def build_headline(payload: dict[str, Any]) -> str:
    """Build a one-sentence headline summarising the recommendation.

    The wording follows the *recommended* scenario, so a "don't hire"
    outcome never claims that a hire will add margin.

    Parameters
    ----------
    payload : dict
        The analysis payload produced by the pipeline.

    Returns
    -------
    str
        A short, quotable headline.
    """
    executive = payload.get("executive", {}) or {}
    kpis = payload.get("kpis", {}) or {}
    impact = executive.get("expected_impact", {}) or {}
    current = kpis.get("current_employees", "?")
    action = executive.get("recommendation", "N/A")
    scenario = str(executive.get("recommended_scenario", "") or "")
    horizon = payload.get("assumptions", {}).get("forecast_months") or 6

    if scenario == "B":
        gain = impact.get("margin_advantage_vs_do_nothing")
        if isinstance(gain, (int, float)):
            return (
                f"{action} - moving from {current} to {current + 1} heads adds "
                f"{_fmt(gain)} of gross margin over the next {horizon} months versus "
                f"leaving the vacancy unfilled."
            )
    elif scenario == "A":
        total = impact.get("total_gross_margin_6m")
        lead = impact.get("margin_advantage_over_runner_up")
        if isinstance(total, (int, float)):
            extra = (
                f", {_fmt(lead)} better than the runner-up policy"
                if isinstance(lead, (int, float)) and lead > 0
                else ""
            )
            return (
                f"{action} - expected gross margin is {_fmt(total)} over the next "
                f"{horizon} months{extra}."
            )

    fallback = impact.get("margin_advantage_vs_do_nothing")
    if isinstance(fallback, (int, float)):
        return (
            f"{action} - {_fmt(fallback)} of gross margin difference over the next "
            f"{horizon} months versus leaving the vacancy unfilled."
        )
    return f"{action} - currently {current} active employees."


def build_recommendation_text(
    payload: dict[str, Any],
    salary: float = 3.0,
    commission: float = 0.05,
) -> str:
    """Render the full multi-section recommendation report.

    Parameters
    ----------
    payload : dict
        The analysis payload produced by the pipeline.
    salary : float, optional
        Monthly salary per employee (Lakhs INR) used in the assumptions.
    commission : float, optional
        Commission rate used in the assumptions.

    Returns
    -------
    str
        Formatted recommendation text.
    """
    lines: list[str] = []
    executive = payload.get("executive", {})
    kpis = payload.get("kpis", {})
    dataset = payload.get("dataset", {})
    forecast = payload.get("forecast", {})
    scenarios = payload.get("scenarios", {})
    productivity = payload.get("productivity", {})
    sensitivity = payload.get("sensitivity", {})
    payback = payload.get("payback", {})
    horizon = payload.get("assumptions", {}).get("forecast_months") or 6
    zoo_size = payload.get("forecast", {}).get("model_candidates") or len(
        forecast.get("metrics", [])
    )

    # ------------------------------------------------------------------ #
    # Header                                                              #
    # ------------------------------------------------------------------ #
    lines.append(_rule("="))
    lines.append(f"  {str(dataset.get('name', 'STORE')).upper()} - STAFFING RECOMMENDATION")
    lines.append(_rule("="))
    lines.append(f"  Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(
        f"  Dataset   : {dataset.get('name', 'N/A')} "
        f"({dataset.get('months', 0)} months, {dataset.get('salespersons', 0)} salespersons)"
    )
    lines.append("")
    lines.append(f"  {build_headline(payload)}")
    lines.append("")

    # ------------------------------------------------------------------ #
    # 1. Executive summary                                                #
    # ------------------------------------------------------------------ #
    lines.append(_rule())
    lines.append("  1. EXECUTIVE SUMMARY")
    lines.append(_rule())
    lines.append(f"  Current staff count       : {kpis.get('current_employees', 'N/A')}")
    lines.append(f"  Recommended action        : {executive.get('recommendation', 'N/A')}")
    lines.append(f"  Confidence level          : {executive.get('confidence', 'N/A')}")
    lines.append(f"  Runner-up option          : {executive.get('runner_up', 'N/A')}")
    lines.append(f"  Avg monthly sales (Lakhs) : {_fmt(kpis.get('avg_monthly_sales'))}")
    lines.append(f"  Avg monthly margin (Lakhs): {_fmt(kpis.get('avg_monthly_margin'))}")
    lines.append(f"  Optimal headcount         : {kpis.get('optimal_headcount', 'N/A')}")
    lines.append(
        f"  Controlled marginal sales : {_fmt(kpis.get('marginal_sales_per_employee'))}"
        " per additional employee"
    )
    lines.append("")

    # ------------------------------------------------------------------ #
    # 2. Financial highlights                                             #
    # ------------------------------------------------------------------ #
    lines.append(_rule())
    lines.append("  2. FINANCIAL HIGHLIGHTS")
    lines.append(_rule())
    lines.append(f"  Best monthly margin       : {_fmt(kpis.get('best_margin'))}")
    lines.append(f"  Worst monthly margin      : {_fmt(kpis.get('worst_margin'))}")
    lines.append(f"  Sales variability (std)   : {_fmt(kpis.get('sales_volatility'))}")
    lines.append(f"  Break-even sales/employee : {_fmt(kpis.get('break_even_sales_per_employee'))}")
    lines.append(f"  Avg sales per employee    : {_fmt(kpis.get('avg_sales_per_employee'))}")
    lines.append("")

    # ------------------------------------------------------------------ #
    # 3. Forecast outlook                                                 #
    # ------------------------------------------------------------------ #
    lines.append(_rule())
    lines.append("  3. FORECAST OUTLOOK")
    lines.append(_rule())
    lines.append(f"  Selected model            : {forecast.get('model', 'N/A')}")
    validation_rmse = kpis.get("validation_rmse")
    if validation_rmse is None:
        validation_rmse = forecast.get("validation_rmse")
    validation_mape = kpis.get("validation_mape")
    if validation_mape is None:
        validation_mape = forecast.get("validation_mape")
    lines.append(f"  Validation RMSE           : {_fmt(validation_rmse)}")
    if validation_mape is not None:
        lines.append(f"  Validation MAPE           : {validation_mape:.2f} %")
    lines.append(f"  Forecast horizon          : {len(forecast.get('months', []))} months")
    values = forecast.get("values") or []
    if values:
        lines.append(f"  Forecast avg sales (Lakhs): {_fmt(sum(values) / len(values))}")
    lower = forecast.get("lower_95") or []
    upper = forecast.get("upper_95") or []
    if lower and upper:
        lines.append(f"  95% interval (month 1)    : {_fmt(lower[0])} to {_fmt(upper[0])}")
        lines.append(
            f"  95% interval (month {len(lower)})    : {_fmt(lower[-1])} to {_fmt(upper[-1])}"
        )
    members = forecast.get("ensemble_members") or []
    if members:
        lines.append("")
        lines.append("  Ensemble composition:")
        for member in members:
            lines.append(
                f"    - {member.get('model', '?')}: "
                f"{float(member.get('weight', 0)) * 100:.0f}% weight "
                f"(RMSE {_fmt(member.get('rmse'))})"
            )
    if forecast.get("methodology"):
        lines.append("")
        lines.append(f"  Method: {forecast['methodology']}")
    lines.append("")

    # ------------------------------------------------------------------ #
    # 4. Staffing recommendation                                          #
    # ------------------------------------------------------------------ #
    lines.append(_rule())
    lines.append("  4. STAFFING RECOMMENDATION")
    lines.append(_rule())
    lines.append(f"  Recommended action        : {executive.get('recommendation', 'N/A')}")
    lines.append(f"  Confidence level          : {executive.get('confidence', 'N/A')}")
    lines.append(f"  Runner-up option          : {executive.get('runner_up', 'N/A')}")

    reasoning = executive.get("reasoning") or []
    if reasoning:
        lines.append("")
        lines.append("  Reasoning:")
        for bullet in reasoning:
            lines.append(f"    - {bullet}")

    expected = executive.get("expected_impact") or {}
    if expected:
        lines.append("")
        lines.append(f"  Expected {horizon}-Month Impact:")
        for key, value in expected.items():
            label = key.replace("_", " ").title()
            if isinstance(value, (int, float)):
                lines.append(f"    {label:38s}: {_fmt(value)}")
            else:
                lines.append(f"    {label:38s}: {value}")
    lines.append("")

    # ------------------------------------------------------------------ #
    # 5. Scenario comparison                                              #
    # ------------------------------------------------------------------ #
    lines.append(_rule())
    lines.append("  5. SCENARIO COMPARISON")
    lines.append(_rule())
    for key in sorted(scenarios.keys()):
        scenario = scenarios[key]
        lines.append(f"  Scenario {key} - {scenario.get('name', key)}")
        lines.append(
            f"    {horizon}-Month Gross Margin  : {_fmt(scenario.get('total_gross_margin'))}"
        )
        if scenario.get("delta_vs_best") is not None:
            lines.append(f"    Difference vs best     : {_fmt(scenario.get('delta_vs_best'))}")
        if scenario.get("prob_better_than_baseline") is not None:
            lines.append(
                f"    P(beats do-nothing)    : "
                f"{float(scenario.get('prob_better_than_baseline')) * 100:.0f} %"
            )
        risk = scenario.get("risk", "")
        if risk:
            lines.append(f"    Risk                    : {risk}")
    lines.append("")

    # ------------------------------------------------------------------ #
    # 6. Recommended business rule                                        #
    # ------------------------------------------------------------------ #
    lines.append(_rule())
    lines.append("  6. RECOMMENDED BUSINESS RULE")
    lines.append(_rule())
    optimal_rule = payload.get("optimal_rule", {})
    lines.append(f"  {optimal_rule.get('rule_description', 'N/A')}")
    backtest = payload.get("rule_backtest", {})
    if backtest:
        lines.append("")
        lines.append("  Backtested against 'always replace':")
        lines.append(f"    - {backtest.get('interpretation', 'N/A')}")
    lines.append("")

    # ------------------------------------------------------------------ #
    # 7. Risk & sensitivity                                               #
    # ------------------------------------------------------------------ #
    lines.append(_rule())
    lines.append("  7. RISK & SENSITIVITY")
    lines.append(_rule())
    tornado = sensitivity.get("tornado") or []
    if isinstance(tornado, list):
        ranking = [str(item.get("variable", "?")) for item in tornado]
    else:
        ranking = list(tornado.get("ranking", []))
    if ranking:
        lines.append("  Drivers of the margin outcome (most material first):")
        for index, name in enumerate(ranking, start=1):
            lines.append(f"    {index}. {name}")
    monte_carlo = sensitivity.get("monte_carlo") or {}
    if monte_carlo:
        lines.append("")
        lines.append("  Monte-Carlo margin distribution:")
        lines.append(
            f"    P10 / P50 / P90          : {_fmt(monte_carlo.get('p10'))} / "
            f"{_fmt(monte_carlo.get('p50'))} / {_fmt(monte_carlo.get('p90'))}"
        )
        probability = monte_carlo.get("prob_positive")
        if probability is not None:
            lines.append(f"    P(margin > 0)           : {float(probability) * 100:.0f} %")
    payback_months = payback.get("months_to_payback")
    if payback_months is not None:
        lines.append("")
        lines.append(
            f"  Payback on the recommended hire: {payback_months} month(s); "
            f"ROI {payback.get('roi_pct', 0):.1f} %."
        )
    lines.append("")

    # ------------------------------------------------------------------ #
    # 8. Workforce insights                                               #
    # ------------------------------------------------------------------ #
    lines.append(_rule())
    lines.append("  8. WORKFORCE INSIGHTS")
    lines.append(_rule())
    attrition = productivity.get("attrition") or {}
    if attrition:
        lines.append(f"  Total leavers (historical): {attrition.get('total_leavers', 0)}")
        lines.append(
            f"  Average tenure of leavers : {attrition.get('avg_tenure_months', 0):.1f} months"
        )
        lines.append(f"  Leavers in last 12 months : {attrition.get('recent_12m_leavers', 0)}")
    if kpis.get("anomalies_detected") is not None:
        lines.append(f"  Anomalous months detected : {kpis.get('anomalies_detected')}")
    lines.append("")

    # ------------------------------------------------------------------ #
    # 9. Business assumptions                                             #
    # ------------------------------------------------------------------ #
    lines.append(_rule())
    lines.append("  9. BUSINESS ASSUMPTIONS")
    lines.append(_rule())
    assumptions = payload.get("assumptions", {}) or {}
    cogs_rate = assumptions.get("cogs_rate")
    ramp = assumptions.get("new_hire_ramp") or []
    ramp_text = "the modelled ramp-up curve"
    if isinstance(ramp, (list, tuple)) and len(ramp) >= 2:
        try:
            ramp_text = (
                f"{float(ramp[0]) * 100:.0f}% -> {float(ramp[-1]) * 100:.0f}% "
                f"over {len(ramp)} months"
            )
        except (TypeError, ValueError):
            ramp_text = "the modelled ramp-up curve"
    cogs_text = (
        f"{float(cogs_rate) * 100:.0f}% of sales"
        if isinstance(cogs_rate, (int, float))
        else "the configured rate of sales"
    )

    for assumption in [
        "Customer traffic remains stable.",
        "No significant hiring/recruitment costs are incurred.",
        f"No training-period productivity loss beyond the modelled ramp-up " f"({ramp_text}).",
        f"Monthly salary stays fixed at Rs. {salary} Lakhs per employee.",
        f"Commission remains fixed at {commission * 100:.1f}% of total sales.",
        f"Cost of goods sold stays at {cogs_text}.",
        "Economic conditions and consumer demand remain similar.",
        "No major competitor entry or exit during the projection period.",
        "Store operating hours and footfall remain unchanged.",
        "The pricing and assortment strategy remains consistent.",
    ]:
        lines.append(f"    - {assumption}")
    lines.append("")

    # ------------------------------------------------------------------ #
    # 10. Methodology notes                                               #
    # ------------------------------------------------------------------ #
    lines.append(_rule())
    lines.append("  10. METHODOLOGY NOTES")
    lines.append(_rule())
    lines.append("  - All monetary values are in Lakhs INR.")
    lines.append("  - Salesperson productivity is measured as individual revenue")
    lines.append("    contribution relative to total monthly sales.")
    lines.append("  - Forecasts use rolling-origin (walk-forward) cross-validation")
    lines.append(f"    across a {zoo_size}-model zoo; the top performers are blended with")
    lines.append("    inverse-RMSE weights.")
    lines.append("  - Prediction intervals come from an empirical residual bootstrap,")
    lines.append("    so they widen with the forecast horizon.")
    lines.append("  - Staffing recommendations consider seasonality, forecast")
    lines.append("    demand, diminishing returns and per-employee efficiency.")
    lines.append("  - Scenario analysis includes a new-hire ramp-up curve")
    lines.append(f"    ({ramp_text}) and a probabilistic margin band.")
    lines.append(_rule("="))

    return "\n".join(lines)
