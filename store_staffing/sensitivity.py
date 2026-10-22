"""Risk and sensitivity analytics for the staffing decision.

Two complementary tools live here:

* :func:`tornado_analysis` — a one-at-a-time sensitivity sweep that shows
  which assumption actually drives the margin outcome.
* :func:`monte_carlo_margin` — a stochastic simulation that propagates
  demand, salary and commission uncertainty into a margin distribution and
  reports how often each staffing scenario wins.

Together they answer the question a store owner actually cares about:
*how confident can we be in the recommendation?*

All monetary values are in **Lakhs INR**.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "tornado_analysis",
    "monte_carlo_margin",
    "decision_stability",
]


def _margin(sales: float, employees: int, salary: float, commission: float, cogs: float) -> float:
    """Return the gross margin for the given inputs."""
    return (1.0 - cogs - commission) * sales - salary * employees


def tornado_analysis(
    base_sales: float,
    base_employees: int,
    base_salary: float = 3.0,
    base_commission: float = 0.05,
    base_cogs: float = 0.50,
    salary_range: Sequence[float] = (2.0, 4.5),
    commission_range: Sequence[float] = (0.02, 0.09),
    cogs_range: Sequence[float] = (0.42, 0.58),
    sales_range: Sequence[float] = (0.8, 1.2),
) -> dict[str, Any]:
    """Compute a tornado (one-at-a-time) sensitivity sweep.

    Each variable is swung to its low and high bound while the others are
    held at their base values.  The resulting swing in gross margin ranks
    the assumptions by materiality.

    Parameters
    ----------
    base_sales : float
        Reference monthly sales (Lakhs INR).
    base_employees : int
        Reference headcount.
    base_salary : float, optional
        Baseline monthly salary per employee.
    base_commission : float, optional
        Baseline commission rate.
    base_cogs : float, optional
        Baseline cost-of-goods rate.
    salary_range : sequence of float, optional
        ``(low, high)`` monthly salary bounds.
    commission_range : sequence of float, optional
        ``(low, high)`` commission-rate bounds.
    cogs_range : sequence of float, optional
        ``(low, high)`` cost-of-goods bounds.
    sales_range : sequence of float, optional
        ``(low_mult, high_mult)`` multipliers applied to base sales.

    Returns
    -------
    dict
        Keys: ``base_margin``, ``variables`` (sorted by impact),
        ``ranking`` (list of variable names, most material first).
    """
    base_margin = _margin(base_sales, base_employees, base_salary, base_commission, base_cogs)

    specifications = [
        (
            "Monthly salary",
            (base_sales, base_employees, float(salary_range[0]), base_commission, base_cogs),
            (base_sales, base_employees, float(salary_range[1]), base_commission, base_cogs),
            salary_range[0],
            salary_range[1],
        ),
        (
            "Commission rate",
            (base_sales, base_employees, base_salary, float(commission_range[0]), base_cogs),
            (base_sales, base_employees, base_salary, float(commission_range[1]), base_cogs),
            commission_range[0],
            commission_range[1],
        ),
        (
            "COGS rate",
            (base_sales, base_employees, base_salary, base_commission, float(cogs_range[0])),
            (base_sales, base_employees, base_salary, base_commission, float(cogs_range[1])),
            cogs_range[0],
            cogs_range[1],
        ),
        (
            "Sales volume",
            (
                base_sales * float(sales_range[0]),
                base_employees,
                base_salary,
                base_commission,
                base_cogs,
            ),
            (
                base_sales * float(sales_range[1]),
                base_employees,
                base_salary,
                base_commission,
                base_cogs,
            ),
            f"x{sales_range[0]}",
            f"x{sales_range[1]}",
        ),
    ]

    variables: list[dict[str, Any]] = []
    for name, low_spec, high_spec, low_value, high_value in specifications:
        low_margin = _margin(*low_spec)
        high_margin = _margin(*high_spec)
        variables.append(
            {
                "variable": name,
                "low": low_value,
                "high": high_value,
                "low_margin": round(low_margin, 4),
                "high_margin": round(high_margin, 4),
                "swing": round(abs(high_margin - low_margin), 4),
                "delta_vs_base_low": round(low_margin - base_margin, 4),
                "delta_vs_base_high": round(high_margin - base_margin, 4),
            }
        )

    variables.sort(key=lambda v: v["swing"], reverse=True)

    return {
        "base_margin": round(base_margin, 4),
        "variables": variables,
        "ranking": [v["variable"] for v in variables],
    }


def monte_carlo_margin(
    forecast_sales: Sequence[float],
    forecast_lower: Sequence[float] | None = None,
    forecast_upper: Sequence[float] | None = None,
    employees_by_scenario: dict[str, Sequence[int]] | None = None,
    salary: float = 3.0,
    commission: float = 0.05,
    cogs: float = 0.50,
    n_simulations: int = 5_000,
    salary_volatility: float = 0.05,
    commission_volatility: float = 0.005,
    random_seed: int = 42,
) -> dict[str, Any]:
    """Simulate the margin distribution under demand and cost uncertainty.

    Monthly sales are sampled from a normal distribution calibrated to the
    forecast interval; salary and commission are perturbed by their own
    small volatilities.  Each scenario's headcount schedule is then priced
    over the simulated demand paths.

    Parameters
    ----------
    forecast_sales : sequence of float
        Point forecast of monthly sales.
    forecast_lower, forecast_upper : sequence of float, optional
        Prediction bounds used to calibrate the demand distribution.  When
        omitted a 10 % coefficient of variation is assumed.
    employees_by_scenario : dict, optional
        Mapping of scenario key -> per-month headcount schedule.  When
        omitted only the base headcount is priced.
    salary : float, optional
        Monthly salary per employee.
    commission : float, optional
        Commission rate.
    cogs : float, optional
        Cost-of-goods rate.
    n_simulations : int, optional
        Number of Monte-Carlo draws.
    salary_volatility : float, optional
        Relative standard deviation of the salary assumption.
    commission_volatility : float, optional
        Absolute standard deviation of the commission rate.
    random_seed : int, optional
        Seed for reproducibility.

    Returns
    -------
    dict
        Keys: ``n_simulations``, ``scenarios`` (per-scenario statistics),
        ``baseline_margin_distribution``.
    """
    rng = np.random.default_rng(random_seed)

    point = np.asarray(list(forecast_sales), dtype=float)
    horizon = point.size
    if horizon == 0:
        return {"n_simulations": 0, "scenarios": {}, "baseline_margin_distribution": {}}

    if forecast_lower is not None and forecast_upper is not None:
        lower = np.asarray(list(forecast_lower), dtype=float)
        upper = np.asarray(list(forecast_upper), dtype=float)
        sigma = np.maximum((upper - lower) / (2.0 * 1.959963985), 1e-6)
    else:
        sigma = np.abs(point) * 0.10

    # --- sample the uncertainty drivers ---------------------------------- #
    sales_draws = rng.normal(point, sigma, size=(n_simulations, horizon))
    sales_draws = np.maximum(sales_draws, 0.0)
    salary_draws = rng.normal(salary, max(salary * salary_volatility, 1e-9), size=n_simulations)
    commission_draws = rng.normal(commission, max(commission_volatility, 1e-9), size=n_simulations)
    commission_draws = np.clip(commission_draws, 0.0, 0.5)

    def _simulate(headcounts: Sequence[int]) -> np.ndarray:
        """Return the total margin for each simulation."""
        headcounts_arr = np.asarray(list(headcounts)[:horizon], dtype=float)
        if headcounts_arr.size < horizon:
            headcounts_arr = np.pad(headcounts_arr, (0, horizon - headcounts_arr.size), mode="edge")
        rate = (1.0 - cogs - commission_draws)[:, None]
        margin = rate * sales_draws - salary_draws[:, None] * headcounts_arr[None, :]
        return margin.sum(axis=1)

    results: dict[str, Any] = {}
    if employees_by_scenario:
        for key, headcounts in employees_by_scenario.items():
            totals = _simulate(headcounts)
            results[key] = {
                "mean": round(float(np.mean(totals)), 4),
                "std": round(float(np.std(totals)), 4),
                "p05": round(float(np.percentile(totals, 5)), 4),
                "p10": round(float(np.percentile(totals, 10)), 4),
                "p25": round(float(np.percentile(totals, 25)), 4),
                "p50": round(float(np.percentile(totals, 50)), 4),
                "p75": round(float(np.percentile(totals, 75)), 4),
                "p90": round(float(np.percentile(totals, 90)), 4),
                "p95": round(float(np.percentile(totals, 95)), 4),
                "prob_positive": round(float(np.mean(totals > 0)), 4),
                "worst_case": round(float(np.min(totals)), 4),
                "best_case": round(float(np.max(totals)), 4),
            }
    else:
        totals = _simulate([1] * horizon)
        results["base"] = {
            "mean": round(float(np.mean(totals)), 4),
            "std": round(float(np.std(totals)), 4),
            "p10": round(float(np.percentile(totals, 10)), 4),
            "p50": round(float(np.percentile(totals, 50)), 4),
            "p90": round(float(np.percentile(totals, 90)), 4),
            "prob_positive": round(float(np.mean(totals > 0)), 4),
        }

    return {
        "n_simulations": int(n_simulations),
        "scenarios": results,
        "baseline_margin_distribution": results.get("A", {}),
    }


def decision_stability(
    monte_carlo_result: dict[str, Any],
    scenario_labels: dict[str, str],
) -> dict[str, Any]:
    """Summarise how decisive the Monte-Carlo outcome is.

    Parameters
    ----------
    monte_carlo_result : dict
        Output of :func:`monte_carlo_margin`.
    scenario_labels : dict
        Mapping of scenario key -> human-readable label.

    Returns
    -------
    dict
        Keys: ``leader``, ``leader_label``, ``margin_over_runner_up``,
        ``decisive`` (bool), ``note``.
    """
    scenarios = monte_carlo_result.get("scenarios", {})
    if not scenarios:
        return {
            "leader": None,
            "leader_label": "n/a",
            "margin_over_runner_up": 0.0,
            "decisive": False,
            "note": "No simulation results available.",
        }

    ranked = sorted(scenarios.items(), key=lambda kv: kv[1].get("mean", 0.0), reverse=True)
    leader_key, leader = ranked[0]
    runner = ranked[1][1] if len(ranked) > 1 else {"mean": 0.0}
    gap = float(leader.get("mean", 0.0)) - float(runner.get("mean", 0.0))
    spread = float(leader.get("std", 0.0))

    decisive = bool(gap > 0 and gap >= 0.5 * spread)

    return {
        "leader": leader_key,
        "leader_label": scenario_labels.get(leader_key, leader_key),
        "margin_over_runner_up": round(gap, 4),
        "decisive": decisive,
        "note": (
            "The winning scenario is separated from the runner-up by more than "
            "half a standard deviation, so the decision is robust to the "
            "simulated demand and cost uncertainty."
            if decisive
            else "The winning scenario is within half a standard deviation of "
            "the runner-up; treat the recommendation as directional rather "
            "than decisive."
        ),
    }
