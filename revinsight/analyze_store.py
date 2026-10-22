"""Backwards-compatible facade over the RevInsight analysis pipeline.

.. deprecated::
    Use :func:`revinsight.pipeline.run_analysis` directly.  This module is
    kept so that existing notebooks and scripts that imported
    ``analyze_store`` continue to work.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from .config import AnalysisConfig
from .pipeline import run_analysis
from .recommendation import build_recommendation_text
from .utils import (
    calculate_break_even_sales,
    calculate_gross_margin,
    get_salesperson_columns,
    load_and_validate_data,
    prepare_monthly_summary,
    setup_output_dirs,
)

logger = logging.getLogger(__name__)

__all__ = [
    "run_full_analysis",
    "run_data_understanding",
    "run_eda",
    "generate_recommendation_text",
    "save_analysis_results",
    "load_and_validate_data",
    "prepare_monthly_summary",
    "setup_output_dirs",
    "get_salesperson_columns",
    "calculate_gross_margin",
    "calculate_break_even_sales",
]


def run_full_analysis(
    filepath: str,
    output_base_dir: str,
    salary: float = 3.0,
    commission: float = 0.05,
    forecast_months: int = 6,
    start_month: int = 1,
    generate_artifacts: bool = True,
) -> dict[str, Any]:
    """Execute the complete RevInsight analysis pipeline.

    Parameters
    ----------
    filepath : str
        Path to the raw CSV data file.
    output_base_dir : str
        Root directory for all generated outputs.
    salary : float, optional
        Monthly salary per employee in Lakhs INR.
    commission : float, optional
        Commission rate as a fraction of sales.
    forecast_months : int, optional
        Length of the forecast horizon.
    start_month : int, optional
        Calendar month corresponding to dataset month 1.
    generate_artifacts : bool, optional
        Render charts and write the PDF/HTML/CSV artefacts.

    Returns
    -------
    dict
        The full analysis payload.
    """
    config = AnalysisConfig(
        salary_per_employee=salary,
        commission_rate=commission,
        forecast_months=forecast_months,
    )
    config.validate()
    return run_analysis(
        filepath=filepath,
        output_base_dir=output_base_dir,
        config=config,
        start_month=start_month,
        generate_artifacts=generate_artifacts,
    )


def run_data_understanding(df: pd.DataFrame, summary_df: pd.DataFrame) -> dict[str, Any]:
    """Summarise the shape, dtypes and quality of the loaded dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Raw validated dataset.
    summary_df : pd.DataFrame
        Monthly aggregated summary.

    Returns
    -------
    dict
        Descriptive statistics and data-quality notes.
    """
    salesperson_cols = get_salesperson_columns(df)
    num_months = int(df["Month"].nunique()) if "Month" in df.columns else len(df)
    total_missing = int(df.isna().sum().sum())

    zero_counts = (df[salesperson_cols] == 0).sum()
    high_zero_cols = zero_counts[zero_counts > (num_months * 0.5)]

    notes = []
    if total_missing > 0:
        notes.append(
            f"Dataset contains {total_missing} missing/null values across "
            f"{sum(1 for v in df.isna().sum().to_dict().values() if v > 0)} column(s)."
        )
    else:
        notes.append("No missing values detected -- data is complete.")
    if not high_zero_cols.empty:
        notes.append(
            f"{len(high_zero_cols)} salesperson column(s) have >50 % zero entries, "
            "indicating staggered hiring or attrition."
        )
    if num_months >= 24:
        notes.append(f"Dataset spans {num_months} months -- sufficient for seasonal analysis.")
    else:
        notes.append(f"Dataset spans only {num_months} months -- limited seasonal insight.")

    return {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "num_months": num_months,
        "num_employees": len(salesperson_cols),
        "missing_values": df.isna().sum().to_dict(),
        "basic_stats": df.describe(),
        "data_quality_notes": notes,
    }


def run_eda(
    df: pd.DataFrame,
    summary_df: pd.DataFrame,
    output_dir: str,
    scenarios_dict: dict[str, float] | None = None,
) -> dict[str, str]:
    """Generate every publication-quality chart into ``output_dir/charts``.

    Parameters
    ----------
    df : pd.DataFrame
        Raw validated dataset.
    summary_df : pd.DataFrame
        Monthly aggregated summary.
    output_dir : str
        Base output directory.
    scenarios_dict : dict, optional
        Scenario names mapped to total gross margin.

    Returns
    -------
    dict
        Mapping of chart name to file path.
    """
    from .visualization import generate_all_charts

    charts_dir = str(Path(output_dir) / "charts")
    return generate_all_charts(summary_df, df, charts_dir, scenarios_dict=scenarios_dict)


def generate_recommendation_text(
    staffing_results: dict[str, Any],
    financial_results: dict[str, Any] | None = None,
    forecast_results: dict[str, Any] | None = None,
    salary: float = 3.0,
    commission: float = 0.05,
) -> str:
    """Build the recommendation narrative.

    Accepts either the new payload shape (a single ``payload`` dict) or the
    legacy triple of sub-result dictionaries.
    """
    if "executive" in staffing_results:
        return build_recommendation_text(staffing_results, salary=salary, commission=commission)

    payload: dict[str, Any] = {
        "executive": staffing_results.get("recommendation", {}),
        "kpis": {
            "current_employees": staffing_results.get("current_situation", {}).get(
                "active_employees"
            ),
            "avg_monthly_sales": (financial_results or {})
            .get("summary_statistics", {})
            .get("avg_sales"),
            "avg_monthly_margin": (financial_results or {})
            .get("summary_statistics", {})
            .get("avg_margin"),
            "best_margin": (financial_results or {})
            .get("summary_statistics", {})
            .get("max_margin"),
            "worst_margin": (financial_results or {})
            .get("summary_statistics", {})
            .get("min_margin"),
            "sales_volatility": (financial_results or {})
            .get("summary_statistics", {})
            .get("std_sales"),
            "break_even_sales_per_employee": (financial_results or {})
            .get("break_even_analysis", {})
            .get("break_even_sales_per_employee"),
        },
        "forecast": (forecast_results or {}).get("best_model", {}),
        "scenarios": staffing_results.get("scenario_comparison", {}),
        "optimal_rule": staffing_results.get("optimal_rule", {}),
    }
    return build_recommendation_text(payload, salary=salary, commission=commission)


def save_analysis_results(
    summary_df: pd.DataFrame,
    financial_results: dict[str, Any],
    output_dir: str,
) -> str:
    """Persist the monthly analysis results to ``analysis_results.csv``.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Monthly aggregated summary.
    financial_results : dict
        Output of the financial analysis.
    output_dir : str
        Directory in which the CSV is created.

    Returns
    -------
    str
        Path to the written CSV file.
    """
    results_df = summary_df.copy()
    fin_stats = financial_results.get("summary_statistics", {}) if financial_results else {}
    if isinstance(fin_stats, dict):
        for key in (
            "avg_sales",
            "median_sales",
            "std_sales",
            "avg_margin",
            "total_margin",
            "min_margin",
            "max_margin",
            "avg_employees",
            "avg_sales_per_emp",
        ):
            if key in fin_stats:
                results_df[key] = fin_stats[key]

    csv_path = str(Path(output_dir) / "analysis_results.csv")
    results_df.to_csv(csv_path, index=False)
    logger.info("Analysis results saved -> %s", csv_path)
    return csv_path
