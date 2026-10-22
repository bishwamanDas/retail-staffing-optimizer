"""End-to-end orchestration of the Store Staffing Analyzer analytics pipeline.

:func:`run_analysis` chains every analytical stage and returns a single
JSON-serialisable payload that matches the public API contract
(``docs/api_contract.md``).  The same payload powers the CLI, the REST
API and the PDF/HTML reports, which guarantees that every surface shows
exactly the same numbers.
"""

from __future__ import annotations

import logging
import math
import shutil
import tempfile
import time
import uuid
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .anomaly import detect_anomalies
from .config import DEFAULT_CONFIG, AnalysisConfig
from .financial_analysis import (
    calculate_payback_period,
    calculate_roi,
    generate_financial_report,
)
from .forecasting import generate_forecast_report
from .people import analyze_workforce
from .recommendation import build_headline, build_recommendation_text
from .seasonality import decompose_series, seasonal_indices, seasonality_strength
from .sensitivity import decision_stability, monte_carlo_margin, tornado_analysis
from .staffing_model import (
    generate_staffing_report,
    optimize_headcount,
    scenario_analysis,
)
from .utils import (
    load_and_validate_data,
    prepare_monthly_summary,
    setup_output_dirs,
)

logger = logging.getLogger(__name__)

__all__ = [
    "SCHEMA_VERSION",
    "run_analysis",
    "simulate_scenarios",
    "detect_cycle_length",
]

SCHEMA_VERSION: str = "2.0"
"""Version of the analysis payload schema."""


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #
def detect_cycle_length(
    sales: pd.Series,
    candidates: Sequence[int] = (3, 4, 6, 12),
) -> dict[str, Any]:
    """Detect the dominant seasonal cycle of a monthly series.

    The series is detrended with a centred moving average and the
    autocorrelation function is evaluated at each candidate lag.  The lag
    with the strongest correlation wins.

    Parameters
    ----------
    sales : pd.Series
        Monthly sales series.
    candidates : sequence of int, optional
        Candidate cycle lengths to test (default ``(3, 4, 6, 12)``).

    Returns
    -------
    dict
        Keys: ``cycle_length``, ``correlations`` (lag -> ACF),
        ``confidence`` (``"HIGH"``/``"MEDIUM"``/``"LOW"``).

    Examples
    --------
    >>> detect_cycle_length(pd.Series([1, 2, 3, 4, 5, 6]))["cycle_length"] in (3, 4, 6, 12)
    True
    """
    values = pd.Series(sales).astype(float)
    if len(values) < 8:
        return {"cycle_length": 12, "correlations": {}, "confidence": "LOW"}

    # --- detrend ------------------------------------------------------- #
    trend = values.rolling(window=5, center=True, min_periods=1).mean()
    detrended = (values - trend).dropna()
    if len(detrended) < 6:
        return {"cycle_length": 12, "correlations": {}, "confidence": "LOW"}

    array = detrended.to_numpy()
    denominator = float(np.sum((array - array.mean()) ** 2))
    if denominator <= 1e-12:
        return {"cycle_length": 12, "correlations": {}, "confidence": "LOW"}

    correlations: dict[int, float] = {}
    for lag in candidates:
        if lag >= len(array):
            continue
        head = array[:-lag]
        tail = array[lag:]
        head_c = head - head.mean()
        tail_c = tail - tail.mean()
        denom = float(np.sqrt(np.sum(head_c**2) * np.sum(tail_c**2)))
        correlations[int(lag)] = float(np.sum(head_c * tail_c) / denom) if denom > 1e-12 else 0.0

    if not correlations:
        return {"cycle_length": 12, "correlations": {}, "confidence": "LOW"}

    max_abs = max(abs(v) for v in correlations.values())
    # A shorter cycle is the *fundamental* period: a 6-month cycle also
    # correlates at lag 12, but lag 6 is the primitive pattern.  Prefer the
    # shortest lag whose correlation is close to the strongest one.
    threshold = max(0.5, 0.6 * max_abs)
    fundamental = [
        lag for lag, value in sorted(correlations.items()) if value > 0 and value >= threshold
    ]
    best_lag = (
        fundamental[0] if fundamental else max(correlations, key=lambda k: abs(correlations[k]))
    )
    best_value = abs(correlations[best_lag])
    confidence = "HIGH" if best_value >= 0.5 else ("MEDIUM" if best_value >= 0.3 else "LOW")

    return {
        "cycle_length": int(best_lag),
        "correlations": {k: round(v, 4) for k, v in correlations.items()},
        "confidence": confidence,
    }


def build_forecast_path(
    sales_series: pd.Series,
    forecast_values: Sequence[float],
    total_months: int = 24,
) -> list[float]:
    """Extend the model forecast to a longer path for the what-if simulator.

    The pipeline only forecasts the configured horizon (six months by
    default), but the simulator lets the user explore a 3-24 month
    window.  Months beyond the forecast are filled with a **seasonal-naive
    extension**: the observed pattern from the previous year is repeated.

    Parameters
    ----------
    sales_series : pd.Series
        Historical monthly sales, indexed by month number.
    forecast_values : sequence of float
        The model forecast for the first months.
    total_months : int, optional
        Length of the path to build (default 24).

    Returns
    -------
    list of float
        A path of at most ``total_months`` values.
    """
    forecast = [float(v) for v in forecast_values]
    path = list(forecast)
    if not path:
        path = [float(sales_series.iloc[-1]) if len(sales_series) else 0.0]

    history = sales_series.values.astype(float)
    season = 12
    for index in range(len(path), int(total_months)):
        # Seasonal-naive: repeat the value observed one cycle earlier.
        offset = len(history) - season + (index % season)
        if 0 <= offset < len(history):
            path.append(float(history[offset]))
        else:
            path.append(float(path[-1]))

    return path[: int(total_months)]


def _empty_series_payload(summary_df: pd.DataFrame) -> dict[str, Any]:
    """Return the ``series`` section for a given monthly summary."""
    return {
        "months": [int(m) for m in summary_df.get("Month", [])],
        "month_of_year": [int(m) for m in summary_df.get("MonthOfYear", [])],
        "sales": [round(float(v), 4) for v in summary_df.get("Sales", [])],
        "employees": [int(v) for v in summary_df.get("ActiveEmployees", [])],
        "sales_per_employee": [
            None if pd.isna(v) else round(float(v), 4)
            for v in summary_df.get("SalesPerEmployee", [])
        ],
        "gross_margin": [round(float(v), 4) for v in summary_df.get("GrossMargin", [])],
        "rolling_sales": [
            None if pd.isna(v) else round(float(v), 4) for v in summary_df.get("Rolling3Sales", [])
        ],
        "rolling_sales_per_employee": [
            None if pd.isna(v) else round(float(v), 4)
            for v in summary_df.get("Rolling3SalesPerEmp", [])
        ],
        "trend": [],
        "seasonal": [],
        "residual": [],
        "seasonal_index": [],
        "anomalies": [],
        "headcount_timeline": [],
    }


# --------------------------------------------------------------------------- #
# Main entry point                                                             #
# --------------------------------------------------------------------------- #
def render_artifacts(
    payload: dict[str, Any],
    output_base_dir: str,
    reports_base_dir: str | None = None,
) -> dict[str, Any]:
    """Render the charts, PDF report, HTML report and CSV for a run.

    The dashboard renders its own charts from the payload, so this step
    is only needed for the downloadable artefacts.  It is therefore
    deliberately separable from :func:`run_analysis`, which lets the API
    return the numbers immediately and generate the report on demand.

    Parameters
    ----------
    payload : dict
        A payload produced by :func:`run_analysis`.  The ``_internal``
        block must carry the dataframes and chart inputs.
    output_base_dir : str
        Directory for the charts and the CSV export.
    reports_base_dir : str, optional
        Directory for the PDF/HTML reports.  Defaults to a ``reports``
        folder inside *output_base_dir*.

    Returns
    -------
    dict
        The artefact paths that were produced.
    """
    internal = payload.get("_internal", {})
    summary_df = internal.get("summary_df")
    raw_df = internal.get("raw_df")
    config: AnalysisConfig = internal.get("config", DEFAULT_CONFIG)
    chart_inputs = internal.get("chart_inputs", {}) or {}
    run_id = payload.get("run_id", "run")

    if summary_df is None or raw_df is None:
        raise ValueError("The payload does not carry the data needed to render artefacts.")

    dirs = setup_output_dirs(output_base_dir)
    output_dir = str(dirs["output"])
    charts_target = str(dirs["charts"])
    reports_dir = Path(reports_base_dir) if reports_base_dir else (Path(output_dir) / "reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    scenarios = payload.get("scenarios", {}) or {}
    series_payload = payload.get("series", {}) or {}
    optimisation = chart_inputs.get("optimisation", {}) or {}
    tornado = chart_inputs.get("tornado", {}) or {}
    fin_stats = chart_inputs.get("financial_stats", {}) or {}
    decomposition = chart_inputs.get("decomposition", {}) or {}
    workforce = chart_inputs.get("workforce", {}) or {}

    from .visualization import (
        generate_all_charts,
        plot_anomalies,
        plot_decomposition,
        plot_headcount_optimization,
        plot_scenario_cumulative,
        plot_tornado,
        plot_workforce_tenure,
    )

    charts_target = str(dirs["charts"])
    scenario_margins = {
        f"{key}: {value['name']}": value["total_gross_margin"] for key, value in scenarios.items()
    }
    try:
        chart_paths = generate_all_charts(
            summary_df,
            raw_df,
            charts_target,
            scenarios_dict=scenario_margins or None,
            extra_charts={
                "decomposition": lambda d: plot_decomposition(decomposition, d),
                "anomalies": lambda d: plot_anomalies(summary_df, series_payload["anomalies"], d),
                "workforce_tenure": lambda d: plot_workforce_tenure(workforce, d),
                "headcount_optimization": lambda d: plot_headcount_optimization(
                    optimisation.get("curve", []), d, horizon=config.forecast_months
                ),
                "scenario_cumulative": lambda d: plot_scenario_cumulative(scenarios, d),
                "tornado": lambda d: plot_tornado(tornado, d),
            },
        )
        payload["charts"] = {Path(path).stem: Path(path).stem for path in chart_paths.values()}
    except Exception as exc:  # noqa: BLE001
        logger.error("Chart generation failed: %s", exc, exc_info=True)
        payload["charts"] = {}

    if output_dir:
        try:
            from .report_generator import generate_html_report, generate_pdf_report
        except Exception as exc:  # noqa: BLE001 - reports are optional
            logger.warning("Report generator unavailable: %s", exc)
            generate_html_report = None  # type: ignore[assignment]
            generate_pdf_report = None  # type: ignore[assignment]

        reports_dir = reports_dir or (Path(output_dir) / "reports")
        reports_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = reports_dir / "Recommendation_Report.pdf"
        html_path = reports_dir / "Recommendation_Report.html"
        csv_path = Path(output_dir) / "analysis_results.csv"

        if generate_pdf_report is not None:
            try:
                generate_pdf_report(
                    all_results=payload,
                    summary_df=summary_df,
                    charts_dir=charts_target,
                    output_path=str(pdf_path),
                    salary=config.salary_per_employee,
                    commission=config.commission_rate,
                )
            except Exception as exc:  # noqa: BLE001
                logger.error("PDF generation failed: %s", exc, exc_info=True)

        if generate_html_report is not None:
            try:
                generate_html_report(
                    payload,
                    output_path=str(html_path),
                    charts_dir=charts_target,
                )
            except Exception as exc:  # noqa: BLE001
                logger.error("HTML report generation failed: %s", exc, exc_info=True)

        try:
            results_df = summary_df.copy()
            for key in (
                "avg_sales",
                "avg_margin",
                "total_margin",
                "std_sales",
                "avg_employees",
            ):
                if key in fin_stats:
                    results_df[key] = fin_stats[key]
            results_df.to_csv(csv_path, index=False)
        except Exception as exc:  # noqa: BLE001
            logger.error("CSV export failed: %s", exc, exc_info=True)

        payload["artifacts"] = {
            "pdf_url": f"/api/v1/analyses/{run_id}/report.pdf",
            "html_url": f"/api/v1/analyses/{run_id}/report.html",
            "csv_url": f"/api/v1/analyses/{run_id}/results.csv",
            "charts_url": f"/api/v1/analyses/{run_id}/charts",
        }
        payload["_internal"]["artifact_paths"] = {
            "pdf": str(pdf_path),
            "html": str(html_path),
            "csv": str(csv_path),
            "charts": str(Path(charts_target)),
            "output": str(Path(output_dir)),
        }

    return payload["_internal"].get("artifact_paths", {})


def run_analysis(
    filepath: str | None = None,
    df: pd.DataFrame | None = None,
    output_base_dir: str | None = None,
    config: AnalysisConfig | None = None,
    start_month: int = 1,
    generate_artifacts: bool = True,
    run_id: str | None = None,
    reports_base_dir: str | None = None,
) -> dict[str, Any]:
    """Run the complete analysis pipeline and return the API payload.

    Parameters
    ----------
    filepath : str, optional
        Path to the input CSV.  Mutually exclusive with *df*.
    df : pd.DataFrame, optional
        Pre-loaded dataframe (used for uploads).
    output_base_dir : str, optional
        Directory for generated artefacts.  When omitted no files are
        written except the charts needed by the report.
    config : AnalysisConfig, optional
        Business assumptions (defaults to
        :data:`store_staffing.config.DEFAULT_CONFIG`).
    start_month : int, optional
        Calendar month that dataset month 1 corresponds to.
    generate_artifacts : bool, optional
        Render charts and write the PDF/HTML/CSV artefacts.
    run_id : str, optional
        Identifier for this run (a UUID4 is generated when omitted).
    reports_base_dir : str, optional
        Directory for the PDF/HTML reports.  Defaults to a ``reports``
        folder inside *output_base_dir*.

    Returns
    -------
    dict
        The analysis payload described in ``docs/api_contract.md``.

    Raises
    ------
    ValueError
        If neither *filepath* nor *df* is provided, or the data is invalid.
    """
    started = time.time()
    config = config or DEFAULT_CONFIG
    config.validate()
    run_id = run_id or uuid.uuid4().hex[:12]

    # ---- (a) load ------------------------------------------------------ #
    if df is None:
        if not filepath:
            raise ValueError("Either 'filepath' or 'df' must be provided.")
        raw_df = load_and_validate_data(filepath)
        dataset_name = Path(filepath).stem
    else:
        raw_df = df.copy()
        raw_df.columns = [str(c).strip() for c in raw_df.columns]
        if "Month" not in raw_df.columns or "Sales" not in raw_df.columns:
            raise ValueError("The uploaded dataset must contain 'Month' and 'Sales' columns.")
        raw_df["Month"] = pd.to_numeric(raw_df["Month"], errors="coerce")
        raw_df["Sales"] = pd.to_numeric(raw_df["Sales"], errors="coerce")
        raw_df = raw_df.dropna(subset=["Month", "Sales"]).reset_index(drop=True)
        dataset_name = "uploaded"
    logger.info("Loaded dataset '%s' with shape %s", dataset_name, raw_df.shape)

    # ---- (b) monthly summary ------------------------------------------- #
    summary_df = prepare_monthly_summary(
        raw_df,
        salary=config.salary_per_employee,
        commission=config.commission_rate,
        cogs_rate=config.cogs_rate,
        start_month=start_month,
    )

    sales_series = raw_df.set_index("Month")["Sales"].astype(float)
    sales_series.index = sales_series.index.astype(int)

    # ---- (c) output directories ----------------------------------------- #
    charts_dir = None
    output_dir = None
    reports_dir: Path | None = None
    temp_charts_dir: str | None = None
    if output_base_dir:
        dirs = setup_output_dirs(output_base_dir)
        output_dir = str(dirs["output"])
        charts_dir = str(dirs["charts"])
        reports_dir = dirs.get("reports") or (Path(output_dir) / "reports")
        if reports_base_dir:
            reports_dir = Path(reports_base_dir)
            reports_dir.mkdir(parents=True, exist_ok=True)
    else:
        # No artefact directory was requested: keep the forecast charts in a
        # temporary folder instead of polluting the working directory.
        temp_charts_dir = tempfile.mkdtemp(prefix="store_staffing-charts-")
        charts_dir = temp_charts_dir

    # ---- (d) seasonality ------------------------------------------------ #
    series_payload = _empty_series_payload(summary_df)
    decomposition: dict[str, Any] = {}
    seasonal_rows: list[dict[str, Any]] = []
    try:
        decomposition = decompose_series(sales_series, period=12)
        strength = seasonality_strength(decomposition)
        seasonal_rows = seasonal_indices(summary_df, period=12)
        series_payload["trend"] = [round(float(v), 4) for v in decomposition.get("trend", [])]
        series_payload["seasonal"] = [round(float(v), 4) for v in decomposition.get("seasonal", [])]
        series_payload["residual"] = [round(float(v), 4) for v in decomposition.get("residual", [])]
    except Exception as exc:  # noqa: BLE001 - seasonality is optional context
        logger.warning("Seasonal decomposition unavailable: %s", exc)
        strength = {"seasonal": 0.0, "trend": 0.0, "id": 1.0}

    series_payload["seasonal_index"] = [
        {
            "month": int(row.get("month", 0)),
            "name": row.get("name", ""),
            "index": round(float(row.get("index", 1.0)), 4),
            "strength": round(float(row.get("strength", 0.0)), 4),
            "avg_sales": round(float(row.get("avg_sales", 0.0)), 4),
            "months_observed": int(row.get("months_observed", 0)),
        }
        for row in seasonal_rows
    ]

    # ---- (e) anomaly detection ------------------------------------------ #
    cycle = detect_cycle_length(sales_series)
    anomaly_decomposition = decomposition
    try:
        if cycle.get("cycle_length") and cycle["cycle_length"] != 12 and len(sales_series) >= 48:
            anomaly_decomposition = decompose_series(
                sales_series, period=int(cycle["cycle_length"])
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Cycle-specific decomposition unavailable: %s", exc)

    try:
        anomaly_result = detect_anomalies(summary_df, decomposition=anomaly_decomposition)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Anomaly detection failed: %s", exc)
        anomaly_result = {"anomalies": [], "n_anomalies": 0, "structural_breaks": [], "scores": []}

    series_payload["anomalies"] = [
        {
            "month": int(item.get("month", 0)),
            "name": item.get("name", ""),
            "sales": round(float(item.get("sales", 0.0)), 4),
            "expected": round(float(item.get("expected", 0.0)), 4),
            "score": round(float(item.get("score", 0.0)), 4),
            "direction": item.get("direction", ""),
            "severity": item.get("severity", "low"),
        }
        for item in anomaly_result.get("anomalies", [])
    ]

    # ---- (f) workforce --------------------------------------------------- #
    try:
        workforce = analyze_workforce(raw_df, start_month=start_month)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Workforce analysis failed: %s", exc)
        workforce = {"employees": [], "attrition": {}, "headcount_timeline": [], "summary": {}}

    series_payload["headcount_timeline"] = workforce.get("headcount_timeline", [])

    # ---- (g) financial analysis ----------------------------------------- #
    financial = generate_financial_report(
        summary_df,
        salary_per_person=config.salary_per_employee,
        commission_rate=config.commission_rate,
        cogs_rate=config.cogs_rate,
    )
    fin_stats = financial.get("summary_statistics", {})

    # ---- (h) forecasting ------------------------------------------------- #
    forecast_report = generate_forecast_report(
        sales_series,
        charts_dir or str(Path(output_base_dir or ".") / "charts"),
        horizon=config.forecast_months,
        max_origins=config.backtest_origins,
        confidence=config.confidence_level,
    )
    best_model = forecast_report.get("best_model", {})
    forecast_values = list(best_model.get("forecast_values", []))
    forecast_lower = list(best_model.get("lower_bound", []))
    forecast_upper = list(best_model.get("upper_bound", []))

    if not forecast_values:
        logger.warning("Forecasting produced no values; falling back to the trailing mean.")
        fallback = float(summary_df["Sales"].tail(6).mean())
        forecast_values = [fallback] * config.forecast_months
        forecast_lower = [fallback * 0.9] * config.forecast_months
        forecast_upper = [fallback * 1.1] * config.forecast_months

    # ---- (i) staffing ----------------------------------------------------- #
    # A longer path lets the what-if simulator explore up to 24 months.
    forecast_path = build_forecast_path(sales_series, forecast_values, total_months=24)

    staffing = generate_staffing_report(
        raw_df,
        forecast_values,
        salary_per_employee=config.salary_per_employee,
        commission_rate=config.commission_rate,
        cogs_rate=config.cogs_rate,
        forecast_lower=forecast_lower,
        forecast_upper=forecast_upper,
        ramp=list(config.new_hire_ramp),
        horizon=config.forecast_months,
    )
    scenarios_raw = staffing.get("scenario_comparison", {})

    # Convert the internal 6-month field names into the public contract.
    scenarios: dict[str, Any] = {}
    for key, scenario in scenarios_raw.items():
        monthly = scenario.get("monthly_breakdown", [])
        scenarios[key] = {
            "key": key,
            "name": scenario.get("name", key),
            "total_sales": scenario.get("total_sales_6m", 0.0),
            "total_gross_margin": scenario.get("total_gross_margin_6m", 0.0),
            "total_staff_cost": scenario.get("total_staff_cost_6m", 0.0),
            "avg_monthly_margin": scenario.get("avg_monthly_margin", 0.0),
            "monthly": [
                {
                    "month": int(m.get("month", 0)),
                    "employees": int(m.get("employees", 0)),
                    "sales": round(float(m.get("sales", 0.0)), 4),
                    "gross_margin": round(float(m.get("gross_margin", 0.0)), 4),
                    "staff_cost": round(float(m.get("staff_cost", 0.0)), 4),
                }
                for m in monthly
            ],
            "risk": scenario.get("risk_assessment", ""),
            "advantages": scenario.get("advantages", []),
            "disadvantages": scenario.get("disadvantages", []),
            "margin_std": scenario.get("margin_std"),
            "p10_margin": scenario.get("p10_margin"),
            "p50_margin": scenario.get("p50_margin"),
            "p90_margin": scenario.get("p90_margin"),
            "prob_positive": scenario.get("prob_positive"),
            "prob_better_than_baseline": scenario.get("prob_better_than_baseline"),
            "delta_vs_baseline": scenario.get("delta_vs_baseline"),
            "threshold": scenario.get("threshold"),
            "triggered": scenario.get("triggered"),
            "trigger_month": scenario.get("trigger_month"),
        }

    if scenarios:
        best_margin = max(s["total_gross_margin"] for s in scenarios.values())
        for scenario in scenarios.values():
            scenario["delta_vs_best"] = round(scenario["total_gross_margin"] - best_margin, 4)

    # ---- (j) headcount optimisation --------------------------------------- #
    current_employees = (
        int(staffing.get("current_situation", {}).get("active_employees", 0) or 0) or 1
    )

    # Empirical productivity curve: average sales per employee at each
    # observed headcount level.  The optimiser uses it as a fallback when
    # no controlled marginal estimate is available.
    productivity_curve: dict[int, float] = {}
    for row in (
        staffing.get("diminishing_returns", {})
        .get("grouped_data", pd.DataFrame())
        .to_dict("records")
    ):
        headcount = int(row.get("ActiveEmployees", 0) or 0)
        value = row.get("avg_sales_per_emp")
        if headcount > 0 and value is not None and np.isfinite(value):
            productivity_curve[headcount] = float(value)

    marginal_block = staffing.get("marginal_productivity", {})
    marginal_sales = marginal_block.get("marginal_sales_per_employee")

    observed_levels = [
        int(row.get("ActiveEmployees", 0) or 0)
        for row in staffing.get("diminishing_returns", {})
        .get("grouped_data", pd.DataFrame())
        .to_dict("records")
    ]
    observed_levels = [level for level in observed_levels if level > 0]
    headcount_bounds = (min(observed_levels), max(observed_levels) + 1) if observed_levels else None

    optimisation = optimize_headcount(
        forecast_values,
        current_employees=current_employees,
        salary_per_employee=config.salary_per_employee,
        commission_rate=config.commission_rate,
        cogs_rate=config.cogs_rate,
        ramp=list(config.new_hire_ramp),
        horizon=config.forecast_months,
        marginal_sales_per_employee=marginal_sales,
        productivity_curve=productivity_curve,
        headcount_bounds=headcount_bounds,
    )

    # ---- (k) sensitivity & Monte Carlo ------------------------------------ #
    tornado = tornado_analysis(
        base_sales=float(fin_stats.get("avg_sales", 0.0)),
        base_employees=current_employees,
        base_salary=config.salary_per_employee,
        base_commission=config.commission_rate,
        base_cogs=config.cogs_rate,
    )

    employees_by_scenario = {
        key: [m["employees"] for m in scenario["monthly"]] for key, scenario in scenarios.items()
    }
    monte_carlo = monte_carlo_margin(
        forecast_values,
        forecast_lower=forecast_lower,
        forecast_upper=forecast_upper,
        employees_by_scenario=employees_by_scenario,
        salary=config.salary_per_employee,
        commission=config.commission_rate,
        cogs=config.cogs_rate,
        n_simulations=config.n_simulations,
        random_seed=config.random_seed,
    )
    stability = decision_stability(monte_carlo, {k: v["name"] for k, v in scenarios.items()})

    # The headline Monte-Carlo numbers describe the recommended scenario.
    recommended_key = staffing.get("recommendation", {}).get("recommended_scenario", "A")
    recommended_scenario = scenarios.get(recommended_key, {})
    mc_block = dict(monte_carlo.get("scenarios", {}).get(recommended_key, {}))
    mc_block.update(
        {
            "n_simulations": monte_carlo.get("n_simulations", 0),
            "decision_stability": stability,
            "per_scenario": monte_carlo.get("scenarios", {}),
            "prob_better_than_hold": recommended_scenario.get("prob_better_than_baseline"),
            "delta_vs_do_nothing": recommended_scenario.get("delta_vs_baseline"),
        }
    )

    # ---- (l) payback & ROI ------------------------------------------------- #
    best_scenario = scenarios.get(recommended_key, {})
    best_monthly = best_scenario.get("monthly", [])
    baseline_monthly = scenarios.get("A", {}).get("monthly", [])
    incremental_sales: list[float] = []
    for idx, month in enumerate(best_monthly):
        base_sales_m = (
            baseline_monthly[idx]["sales"] if idx < len(baseline_monthly) else month["sales"]
        )
        incremental_sales.append(max(0.0, float(month["sales"]) - float(base_sales_m)))
    payback = calculate_payback_period(
        incremental_sales,
        salary_per_person=config.salary_per_employee,
        commission_rate=config.commission_rate,
        cogs_rate=config.cogs_rate,
    )
    roi = calculate_roi(
        incremental_sales_total=sum(
            (1.0 - config.cogs_rate - config.commission_rate) * s for s in incremental_sales
        ),
        total_cost=config.salary_per_employee * len(incremental_sales),
    )
    payback.update(
        {
            "investment": config.salary_per_employee,
            "roi_pct": roi.get("roi_pct", payback.get("roi_pct", 0.0)),
            "net_benefit": roi.get("net_benefit", 0.0),
            "cumulative_margin_delta": payback.get("cumulative_contribution", []),
        }
    )

    # ---- (m) assemble the payload ------------------------------------------ #
    recommendation_block = staffing.get("recommendation", {})
    executive = {
        "recommendation": recommendation_block.get("recommendation", "N/A"),
        "recommended_scenario": recommended_key,
        "confidence": recommendation_block.get("confidence", "N/A"),
        "runner_up": recommendation_block.get("alternative", "N/A"),
        "reasoning": recommendation_block.get("reasoning", []),
        "expected_impact": recommendation_block.get("expected_impact", {}),
        "confidence_drivers": recommendation_block.get("confidence_drivers", {}),
    }

    kpis = {
        "current_employees": current_employees,
        "optimal_headcount": optimisation.get("optimal_headcount", current_employees),
        "avg_monthly_sales": fin_stats.get("avg_sales"),
        "avg_monthly_margin": fin_stats.get("avg_margin"),
        "total_margin": fin_stats.get("total_margin"),
        "best_margin": fin_stats.get("max_margin"),
        "worst_margin": fin_stats.get("min_margin"),
        "sales_volatility": fin_stats.get("std_sales"),
        "break_even_sales_per_employee": financial.get("break_even_analysis", {}).get(
            "break_even_sales_per_employee"
        ),
        "avg_sales_per_employee": fin_stats.get("avg_sales_per_emp"),
        "productivity_cv": staffing.get("productivity_analysis", {})
        .get("consistency_metrics", {})
        .get("coefficient_of_variation"),
        "forecast_avg_sales": (
            round(sum(forecast_values) / len(forecast_values), 4) if forecast_values else None
        ),
        "forecast_avg_margin": best_scenario.get("avg_monthly_margin"),
        "best_forecast_model": best_model.get("model_name"),
        "validation_rmse": best_model.get("rmse"),
        "validation_mape": best_model.get("ensemble_metrics", {}).get("mape"),
        "validation_mae": best_model.get("ensemble_metrics", {}).get("mae"),
        "validation_smape": best_model.get("ensemble_metrics", {}).get("smape"),
        "validation_r2": best_model.get("ensemble_metrics", {}).get("r2"),
        "diminishing_returns": staffing.get("diminishing_returns", {}).get("diminishing_detected"),
        "staff_cost_6m": best_scenario.get("total_staff_cost"),
        "payback_months": payback.get("months_to_payback"),
        "roi_pct": payback.get("roi_pct"),
        "anomalies_detected": int(anomaly_result.get("n_anomalies", 0)),
        "data_completeness": round(float(1.0 - raw_df.isna().sum().sum() / max(raw_df.size, 1)), 4),
        "confidence_level": recommendation_block.get("confidence", "N/A"),
        "seasonality_strength": strength.get("seasonal", 0.0),
        "detected_cycle_months": cycle.get("cycle_length"),
        "total_leavers": workforce.get("attrition", {}).get("total_leavers", 0),
        "marginal_sales_per_employee": marginal_sales,
        "marginal_margin_contribution": marginal_block.get("marginal_margin_contribution"),
        "marginal_model_r2": marginal_block.get("r_squared"),
    }

    forecast_payload = {
        "model": best_model.get("model_name"),
        "model_candidates": forecast_report.get("model_candidates"),
        "months": forecast_report.get("forecast_months", []),
        "values": [round(float(v), 4) for v in forecast_values],
        "lower_95": [round(float(v), 4) for v in forecast_lower],
        "upper_95": [round(float(v), 4) for v in forecast_upper],
        "ensemble_members": best_model.get("ensemble_members", []),
        "metrics": [
            {
                "model": m.get("model"),
                "rmse": m.get("rmse"),
                "mae": m.get("mae"),
                "mape": m.get("mape"),
                "mase": m.get("mase"),
                "r2": m.get("r2"),
                "smape": m.get("smape"),
                "rank": idx + 1,
            }
            for idx, m in enumerate(best_model.get("model_metrics", []))
        ],
        "backtest": forecast_report.get("backtest", []),
        "residual_std": best_model.get("residual_std"),
        "methodology": best_model.get("methodology"),
        "historical_summary": forecast_report.get("historical_summary", {}),
    }

    productivity_payload = {
        "by_headcount": [
            {
                "employees": int(row.get("ActiveEmployees", 0)),
                "avg_sales": round(float(row.get("avg_total_sales", 0.0)), 4),
                "avg_sales_per_employee": round(float(row.get("avg_sales_per_emp", 0.0)), 4),
                "avg_gross_margin": round(float(row.get("avg_gross_margin", 0.0)), 4),
                "months_observed": int(row.get("month_count", 0)),
            }
            for row in staffing.get("diminishing_returns", {})
            .get("grouped_data", pd.DataFrame())
            .to_dict("records")
        ],
        "diminishing_detected": staffing.get("diminishing_returns", {}).get("diminishing_detected"),
        "marginal_changes": staffing.get("diminishing_returns", {}).get("marginal_changes", []),
        "employees": workforce.get("employees", []),
        "attrition": workforce.get("attrition", {}),
        "headcount_curve": optimisation.get("curve", []),
        "optimal_headcount": optimisation.get("optimal_headcount"),
        "headcount_optimisation": {
            "optimal_headcount": optimisation.get("optimal_headcount"),
            "optimal_total_margin": optimisation.get("optimal_total_margin"),
            "marginal_sales_per_employee": optimisation.get("marginal_sales_per_employee"),
            "marginal_source": optimisation.get("marginal_source"),
            "interpretation": optimisation.get("interpretation"),
            "search_bounds": list(headcount_bounds) if headcount_bounds else None,
            "note": (
                "The optimiser searches the headcount range the store has "
                "actually operated (plus one hire) and treats the marginal "
                "contribution as constant, so a persistently positive "
                "marginal pushes the answer to the top of that range. The "
                "scenario comparison answers the practical question -- hire "
                "one, wait, or hold -- under an explicit ramp-up curve."
            ),
        },
        "marginal_productivity": marginal_block,
    }

    sensitivity_payload = {
        "tornado": tornado.get("variables", []),
        "tornado_base_margin": tornado.get("base_margin"),
        "monte_carlo": mc_block,
        "grid": financial.get("sensitivity_analysis", pd.DataFrame()).to_dict("records"),
    }

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "name": dataset_name,
            "filename": Path(filepath).name if filepath else f"{dataset_name}.csv",
            "rows": int(raw_df.shape[0]),
            "columns": int(raw_df.shape[1]),
            "months": int(len(summary_df)),
            "salespersons": int(
                len([c for c in raw_df.columns if str(c).lower().startswith("sale sp")])
            ),
            "total_sales": round(float(summary_df["Sales"].sum()), 4),
            "first_month": int(summary_df["Month"].iloc[0]),
            "last_month": int(summary_df["Month"].iloc[-1]),
            "completeness": kpis["data_completeness"],
        },
        "assumptions": {
            "salary": config.salary_per_employee,
            "commission": config.commission_rate,
            "cogs_rate": config.cogs_rate,
            "forecast_months": config.forecast_months,
            "confidence_level": config.confidence_level,
            "start_month": start_month,
            "effective_margin_rate": round(config.effective_margin_rate, 4),
            "new_hire_ramp": list(config.new_hire_ramp),
        },
        "executive": executive,
        "kpis": kpis,
        "series": series_payload,
        "forecast": forecast_payload,
        "scenarios": scenarios,
        "productivity": productivity_payload,
        "sensitivity": sensitivity_payload,
        "payback": payback,
        "optimal_rule": staffing.get("optimal_rule", {}),
        "rule_backtest": staffing.get("rule_backtest", {}),
        "vacancy_cost": financial.get("vacancy_cost", {}),
        "charts": {},
        "artifacts": {},
        "recommendation_text": "",
        "_internal": {
            "raw_df": raw_df,
            "summary_df": summary_df,
            "config": config,
            "start_month": start_month,
            "forecast_values": forecast_values,
            "forecast_lower": forecast_lower,
            "forecast_upper": forecast_upper,
            "forecast_path": forecast_path,
            "forecast_path_length": len(forecast_path),
            "marginal_sales_per_employee": marginal_sales,
            "chart_inputs": {
                "decomposition": decomposition,
                "workforce": workforce,
                "optimisation": optimisation,
                "tornado": tornado,
                "financial_stats": fin_stats,
            },
            "elapsed_seconds": None,
        },
    }

    payload["executive"]["headline"] = build_headline(payload)

    # ---- (n) charts & artefacts -------------------------------------------- #
    if generate_artifacts and output_base_dir:
        try:
            render_artifacts(payload, output_base_dir, str(reports_dir) if reports_dir else None)
        except Exception as exc:  # noqa: BLE001 - artefacts must never break a run
            logger.error("Artefact generation failed: %s", exc, exc_info=True)

    # ---- (o) recommendation text ------------------------------------------- #
    payload["recommendation_text"] = build_recommendation_text(
        payload,
        salary=config.salary_per_employee,
        commission=config.commission_rate,
    )
    if output_dir:
        try:
            text_path = Path(output_dir) / "recommendation.txt"
            text_path.write_text(payload["recommendation_text"], encoding="utf-8")
            payload["_internal"].setdefault("artifact_paths", {})["recommendation"] = str(text_path)
        except Exception as exc:  # noqa: BLE001
            logger.error("Writing recommendation.txt failed: %s", exc)

    payload["_internal"]["elapsed_seconds"] = round(time.time() - started, 2)

    if temp_charts_dir:
        shutil.rmtree(temp_charts_dir, ignore_errors=True)

    logger.info("Analysis complete in %.1f s (run_id=%s)", time.time() - started, run_id)
    return payload


# --------------------------------------------------------------------------- #
# What-if simulator                                                            #
# --------------------------------------------------------------------------- #
def simulate_scenarios(
    payload: dict[str, Any],
    salary: float | None = None,
    commission: float | None = None,
    horizon: int | None = None,
    decision: str = "auto",
    sales_shock_pct: float = 0.0,
) -> dict[str, Any]:
    """Re-price the staffing scenarios under changed assumptions.

    Parameters
    ----------
    payload : dict
        A previously computed analysis payload.
    salary : float, optional
        Overridden monthly salary per employee.
    commission : float, optional
        Overridden commission rate.
    horizon : int, optional
        Overridden forecast horizon (clipped to the available forecast).
    decision : str, optional
        ``"auto"`` (rank by margin), ``"hold"``, ``"hire"`` or
        ``"conditional"`` — restricts the comparison to one policy.
    sales_shock_pct : float, optional
        Percentage shift applied to every forecast month (e.g. ``-10``).
    decision : str, optional
        Restrict the comparison to one policy.

    Returns
    -------
    dict
        Keys: ``run_id``, ``assumptions``, ``scenarios``, ``best``,
        ``break_even_sales_per_employee``, ``payback_months``, ``note``.
    """
    internal = payload.get("_internal", {})
    config: AnalysisConfig = internal.get("config", DEFAULT_CONFIG)
    internal.get("summary_df")
    forecast_values: list[float] = list(internal.get("forecast_values", []))
    # The simulator may look further ahead than the model forecast; the
    # extended path fills those months with a seasonal-naive profile.
    forecast_path: list[float] = list(internal.get("forecast_path") or forecast_values)

    salary = config.salary_per_employee if salary is None else float(salary)
    commission = config.commission_rate if commission is None else float(commission)
    horizon = config.forecast_months if horizon is None else int(horizon)
    horizon = max(1, min(horizon, len(forecast_path) or horizon))

    shocked = [float(v) * (1.0 + float(sales_shock_pct) / 100.0) for v in forecast_path[:horizon]]
    if not shocked:
        shocked = [0.0] * horizon

    current_employees = int(payload.get("kpis", {}).get("current_employees", 1) or 1)

    scenarios_raw = scenario_analysis(
        shocked,
        current_employees=current_employees,
        salary_per_employee=salary,
        commission_rate=commission,
        cogs_rate=config.cogs_rate,
        ramp=list(config.new_hire_ramp),
        horizon=horizon,
        marginal_sales_per_employee=internal.get("marginal_sales_per_employee"),
    )

    if decision == "hold":
        allowed = {"A"}
    elif decision == "hire":
        allowed = {"B"}
    elif decision == "conditional":
        allowed = {"D"}
    else:
        allowed = set(scenarios_raw.keys())

    best_margin = max(
        (s["total_gross_margin_6m"] for k, s in scenarios_raw.items() if k in allowed),
        default=0.0,
    )

    scenarios: dict[str, Any] = {}
    for key, scenario in scenarios_raw.items():
        monthly = scenario.get("monthly_breakdown", [])
        scenarios[key] = {
            "key": key,
            "name": scenario.get("name", key),
            "total_sales": scenario.get("total_sales_6m", 0.0),
            "total_gross_margin": scenario.get("total_gross_margin_6m", 0.0),
            "total_staff_cost": scenario.get("total_staff_cost_6m", 0.0),
            "avg_monthly_margin": scenario.get("avg_monthly_margin", 0.0),
            "delta_vs_best": round(scenario.get("total_gross_margin_6m", 0.0) - best_margin, 4),
            "monthly": [
                {
                    "month": int(m.get("month", 0)),
                    "employees": int(m.get("employees", 0)),
                    "sales": round(float(m.get("sales", 0.0)), 4),
                    "gross_margin": round(float(m.get("gross_margin", 0.0)), 4),
                    "staff_cost": round(float(m.get("staff_cost", 0.0)), 4),
                }
                for m in monthly
            ],
            "risk": scenario.get("risk_assessment", ""),
            "advantages": scenario.get("advantages", []),
            "disadvantages": scenario.get("disadvantages", []),
            "in_scope": key in allowed,
        }

    ranked = sorted(
        (s for s in scenarios.values() if s["in_scope"]),
        key=lambda s: s["total_gross_margin"],
        reverse=True,
    )
    best_key = ranked[0]["key"] if ranked else "A"

    # Attach probabilistic margin bands to each scenario.
    lower_bounds = list(internal.get("forecast_lower", []))
    upper_bounds = list(internal.get("forecast_upper", []))
    if lower_bounds and upper_bounds:
        # Extend the interval beyond the forecast with a square-root
        # widening, which is the standard random-walk approximation for
        # forecast uncertainty.
        last_width = (upper_bounds[-1] - lower_bounds[-1]) / 2.0
        step_width = math.sqrt(abs(last_width) or 1.0)
        for index in range(len(lower_bounds), horizon):
            growth = step_width * math.sqrt(index - len(lower_bounds) + 1)
            lower_bounds.append(lower_bounds[-1] - growth)
            upper_bounds.append(upper_bounds[-1] + growth)
        lower_bounds = lower_bounds[:horizon]
        upper_bounds = upper_bounds[:horizon]

    if len(lower_bounds) == horizon and len(upper_bounds) == horizon:
        try:
            from .staffing_model import probabilistic_scenario_analysis

            raw_scenarios = {
                key: {
                    "total_gross_margin_6m": value["total_gross_margin"],
                    "monthly_breakdown": [
                        {
                            "month": m["month"],
                            "employees": m["employees"],
                            "gross_margin": m["gross_margin"],
                        }
                        for m in value["monthly"]
                    ],
                }
                for key, value in scenarios.items()
            }
            enriched = probabilistic_scenario_analysis(
                raw_scenarios,
                lower_bounds,
                upper_bounds,
                commission_rate=commission,
                cogs_rate=config.cogs_rate,
            )["scenarios"]
            for key, block in enriched.items():
                if key in scenarios:
                    scenarios[key].update(
                        {
                            "p10_margin": block.get("p10_margin"),
                            "p50_margin": block.get("p50_margin"),
                            "p90_margin": block.get("p90_margin"),
                            "prob_positive": block.get("prob_positive"),
                            "margin_std": block.get("margin_std"),
                            "prob_better_than_baseline": block.get("prob_better_than_baseline"),
                        }
                    )
        except Exception as exc:  # noqa: BLE001 - bands are optional
            logger.warning("Probabilistic scoring failed in the simulator: %s", exc)

    best_monthly = scenarios[best_key]["monthly"]
    base_monthly = scenarios["A"]["monthly"]
    incremental = [
        max(0.0, best_monthly[i]["sales"] - base_monthly[i]["sales"])
        for i in range(min(len(best_monthly), len(base_monthly)))
    ]
    payback = calculate_payback_period(
        incremental,
        salary_per_person=salary,
        commission_rate=commission,
        cogs_rate=config.cogs_rate,
    )

    margin_rate = 1.0 - config.cogs_rate - commission
    break_even = salary / margin_rate if margin_rate > 0 else float("inf")

    in_scope = [s for s in scenarios.values() if s["in_scope"]]
    count = len(in_scope)
    policy_word = "policy" if count == 1 else "policies"

    if decision == "auto":
        scope_text = f"Comparing all {count} staffing {policy_word}"
    else:
        names = ", ".join(str(s["name"]) for s in in_scope)
        scope_text = f"Comparing {count} staffing {policy_word} ({names})"

    shock_text = f", with a {sales_shock_pct:+.0f}% demand shock applied" if sales_shock_pct else ""
    modelled = len(forecast_values)
    extension_text = ""
    if horizon > modelled:
        extension_text = (
            f" Months {modelled + 1}-{horizon} extend beyond the model forecast "
            f"using a seasonal-naive profile."
        )
    note = (
        f"{scope_text} on expected gross margin over a {horizon}-month horizon"
        f"{shock_text}.{extension_text}"
    )

    return {
        "run_id": payload.get("run_id"),
        "assumptions": {
            "salary": salary,
            "commission": commission,
            "horizon": horizon,
            "decision": decision,
            "sales_shock_pct": sales_shock_pct,
            "current_employees": current_employees,
        },
        "scenarios": scenarios,
        "best": best_key,
        "break_even_sales_per_employee": round(break_even, 4),
        "payback_months": payback.get("months_to_payback"),
        "note": note,
    }
