import matplotlib
matplotlib.use('Agg')

import logging
import os
from datetime import datetime
from pathlib import Path

import pandas as pd

from utils import (
    load_and_validate_data,
    prepare_monthly_summary,
    setup_output_dirs,
    get_salesperson_columns,
)
from visualization import generate_all_charts
from financial_analysis import generate_financial_report
from forecasting import generate_forecast_report
from staffing_model import generate_staffing_report
from report_generator import generate_pdf_report

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


# 1. Data Understanding

def run_data_understanding(df: pd.DataFrame, summary_df: pd.DataFrame) -> dict:
    """Print and return a comprehensive data-understanding summary.

    Parameters
    ----------
    df : pd.DataFrame
        Raw (validated) dataset loaded from CSV.
    summary_df : pd.DataFrame
        Monthly aggregated summary produced by ``prepare_monthly_summary``.

    Returns
    -------
    dict
        Dictionary with keys:
        - ``shape``            : tuple (rows, cols)
        - ``columns``          : list of column names
        - ``dtypes``           : dict mapping column -> dtype string
        - ``num_months``       : int, number of months in the dataset
        - ``num_employees``    : int, number of unique salesperson columns
        - ``missing_values``   : dict mapping column -> count of NaN values
        - ``basic_stats``      : pd.DataFrame from ``df.describe()``
        - ``data_quality_notes``: list of human-readable quality observations
    """
    logger.info("Running data understanding ...")

    salesperson_cols = get_salesperson_columns(df)

    # --- Shape & columns ---
    shape = df.shape
    columns = list(df.columns)
    dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}

    # --- Counts ---
    num_months = int(df["Month"].nunique()) if "Month" in df.columns else shape[0]
    num_employees = len(salesperson_cols)

    # --- Missing values ---
    missing_values = df.isnull().sum().to_dict()
    total_missing = sum(missing_values.values())

    # --- Basic statistics ---
    basic_stats = df.describe()

    # --- Data quality notes ---
    data_quality_notes: list[str] = []

    if total_missing > 0:
        data_quality_notes.append(
            f"Dataset contains {total_missing} missing/null values across "
            f"{sum(1 for v in missing_values.values() if v > 0)} column(s)."
        )
    else:
        data_quality_notes.append("No missing values detected -- data is complete.")

    # Check for zero-heavy salesperson columns (inactive months)
    zero_counts = (df[salesperson_cols] == 0).sum()
    high_zero_cols = zero_counts[zero_counts > (num_months * 0.5)]
    if not high_zero_cols.empty:
        data_quality_notes.append(
            f"{len(high_zero_cols)} salesperson column(s) have >50 % zero entries, "
            "indicating staggered hiring or attrition."
        )

    if num_months >= 12:
        data_quality_notes.append(
            f"Dataset spans {num_months} months -- sufficient for seasonal analysis."
        )
    else:
        data_quality_notes.append(
            f"Dataset spans only {num_months} months -- limited seasonal insight."
        )

    # ---- Console output ----
    print("\n" + "=" * 70)
    print("  DATA UNDERSTANDING")
    print("=" * 70)
    print(f"  Shape              : {shape[0]} rows x {shape[1]} columns")
    print(f"  Months             : {num_months}")
    print(f"  Salesperson columns: {num_employees}")
    print(f"  Total missing vals : {total_missing}")
    print("-" * 70)
    print("  Columns & dtypes:")
    for col in columns:
        print(f"    {col:<25s}  {dtypes[col]}")
    print("-" * 70)
    print("  Quality notes:")
    for note in data_quality_notes:
        print(f"    - {note}")
    print("-" * 70)
    print("  Basic statistics (describe):")
    print(basic_stats.to_string())
    print("=" * 70 + "\n")

    result = {
        "shape": shape,
        "columns": columns,
        "dtypes": dtypes,
        "num_months": num_months,
        "num_employees": num_employees,
        "missing_values": missing_values,
        "basic_stats": basic_stats,
        "data_quality_notes": data_quality_notes,
    }
    logger.info("Data understanding complete.")
    return result


# 2. Exploratory Data Analysis

def run_eda(
    df: pd.DataFrame,
    summary_df: pd.DataFrame,
    output_dir: str,
) -> dict:
    """Run full EDA by delegating chart generation to the visualization module.

    Parameters
    ----------
    df : pd.DataFrame
        Raw (validated) dataset.
    summary_df : pd.DataFrame
        Monthly aggregated summary.
    output_dir : str
        Base output directory; charts are saved inside ``<output_dir>/charts/``.

    Returns
    -------
    dict
        Summary dictionary returned by ``generate_all_charts`` containing
        paths to generated chart files and any computed EDA metrics.
    """
    logger.info("Starting exploratory data analysis ...")
    print("\n" + "=" * 70)
    print("  EXPLORATORY DATA ANALYSIS")
    print("=" * 70)

    try:
        charts_dir = os.path.join(output_dir, "charts")
        os.makedirs(charts_dir, exist_ok=True)

        # Note: generate_all_charts expects (summary_df, raw_df, output_dir)
        eda_summary = generate_all_charts(summary_df, df, charts_dir)

        print(f"  Charts saved to: {charts_dir}")
        logger.info("EDA charts generated successfully.")
    except Exception as exc:
        logger.error("EDA chart generation failed: %s", exc, exc_info=True)
        print(f"  [ERROR] Chart generation failed: {exc}")
        eda_summary = {"error": str(exc)}

    print("=" * 70 + "\n")
    return eda_summary


# 3. Recommendation Text

def generate_recommendation_text(
    staffing_results: dict,
    financial_results: dict,
    forecast_results: dict,
    salary: float = 3.0,
    commission: float = 0.05,
) -> str:
    """Create a professional, multi-section recommendation narrative.

    Safely extracts data from the nested result dictionaries produced by
    each analysis module.  Falls back to 'N/A' if a key is missing.

    Parameters
    ----------
    staffing_results : dict
        Output of ``generate_staffing_report``.
    financial_results : dict
        Output of ``generate_financial_report``.
    forecast_results : dict
        Output of ``generate_forecast_report``.

    Returns
    -------
    str
        Formatted recommendation text suitable for saving or printing.
    """
    lines: list[str] = []

    lines.append("=" * 72)
    lines.append("  REVINSIGHT -- BRANDED-CLOTHING STORE STAFFING RECOMMENDATION")
    lines.append("=" * 72)
    lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # --- Executive summary ---
    lines.append("-" * 72)
    lines.append("  1. EXECUTIVE SUMMARY")
    lines.append("-" * 72)

    # Extract from nested staffing results
    current_sit = staffing_results.get("current_situation", {})
    rec_info = staffing_results.get("recommendation", {})
    current_staff = current_sit.get("active_employees", "N/A")
    recommended_action = rec_info.get("recommendation", "N/A")

    # Extract from financial results
    fin_stats = financial_results.get("summary_statistics", {})
    total_margin = fin_stats.get("total_margin", "N/A")
    avg_sales = fin_stats.get("avg_sales", "N/A")
    avg_margin = fin_stats.get("avg_margin", "N/A")

    lines.append(f"  Current staff count       : {current_staff}")
    lines.append(f"  Recommended action        : {recommended_action}")
    if isinstance(avg_sales, float):
        lines.append(f"  Avg monthly sales (Lakhs) : Rs.{avg_sales:.2f}")
    else:
        lines.append(f"  Avg monthly sales (Lakhs) : {avg_sales}")
    if isinstance(avg_margin, float):
        lines.append(f"  Avg monthly margin (Lakhs): Rs.{avg_margin:.2f}")
    else:
        lines.append(f"  Avg monthly margin (Lakhs): {avg_margin}")
    lines.append("")

    # --- Financial highlights ---
    lines.append("-" * 72)
    lines.append("  2. FINANCIAL HIGHLIGHTS")
    lines.append("-" * 72)

    if isinstance(fin_stats.get("max_margin"), float):
        lines.append(f"  Best monthly margin       : Rs.{fin_stats['max_margin']:.2f} Lakhs")
    if isinstance(fin_stats.get("min_margin"), float):
        lines.append(f"  Worst monthly margin      : Rs.{fin_stats['min_margin']:.2f} Lakhs")
    if isinstance(fin_stats.get("std_sales"), float):
        lines.append(f"  Sales variability (std)   : Rs.{fin_stats['std_sales']:.2f} Lakhs")

    be_analysis = financial_results.get("break_even_analysis", {})
    be_sales = be_analysis.get("break_even_sales_per_employee", "N/A")
    if isinstance(be_sales, float):
        lines.append(f"  Break-even sales/employee : Rs.{be_sales:.2f} Lakhs")
    lines.append("")

    # --- Forecast outlook ---
    lines.append("-" * 72)
    lines.append("  3. FORECAST OUTLOOK")
    lines.append("-" * 72)

    if "error" not in forecast_results:
        best_model = forecast_results.get("best_model", {})
        model_name = best_model.get("model_name", "N/A")
        forecast_vals = best_model.get("forecast_values", [])
        model_rmse = best_model.get("rmse", "N/A")

        lines.append(f"  Selected model            : {model_name}")
        if isinstance(model_rmse, (int, float)):
            lines.append(f"  Validation RMSE           : Rs.{model_rmse:.2f} Lakhs")
        lines.append(f"  Forecast horizon          : 6 months")
        if forecast_vals:
            avg_fc = sum(forecast_vals) / len(forecast_vals)
            lines.append(f"  Forecast avg sales (Lakhs): Rs.{avg_fc:.2f}")

        reasoning = best_model.get("reasoning", "")
        if reasoning:
            lines.append("")
            lines.append("  Model selection reasoning:")
            for rline in reasoning.strip().split("\n"):
                lines.append(f"    {rline}")
    else:
        lines.append("  Forecasting was not available.")
    lines.append("")

    # --- Staffing recommendation ---
    lines.append("-" * 72)
    lines.append("  4. STAFFING RECOMMENDATION")
    lines.append("-" * 72)

    lines.append(f"  Recommended action        : {recommended_action}")
    confidence = rec_info.get("confidence", "N/A")
    lines.append(f"  Confidence level          : {confidence}")
    alternative = rec_info.get("alternative", "N/A")
    lines.append(f"  Runner-up option          : {alternative}")

    reasoning_bullets = rec_info.get("reasoning", [])
    if reasoning_bullets:
        lines.append("")
        lines.append("  Reasoning:")
        for bullet in reasoning_bullets:
            lines.append(f"    - {bullet}")

    expected = rec_info.get("expected_impact", {})
    if expected:
        lines.append("")
        lines.append("  Expected 6-Month Impact:")
        for k, v in expected.items():
            label = k.replace("_", " ").title()
            if isinstance(v, float):
                lines.append(f"    {label:35s}: Rs.{v:.2f} Lakhs")
            else:
                lines.append(f"    {label:35s}: {v}")
    lines.append("")

    # --- Scenario comparison ---
    lines.append("-" * 72)
    lines.append("  5. SCENARIO COMPARISON")
    lines.append("-" * 72)

    scenarios = staffing_results.get("scenario_comparison", {})
    for key in sorted(scenarios.keys()):
        sc = scenarios[key]
        name = sc.get("name", key)
        margin_6m = sc.get("total_gross_margin_6m", "N/A")
        risk = sc.get("risk_assessment", "")
        if isinstance(margin_6m, float):
            lines.append(f"  Scenario {key} — {name}")
            lines.append(f"    6-Month Gross Margin    : Rs.{margin_6m:.2f} Lakhs")
            lines.append(f"    Risk                    : {risk}")
        else:
            lines.append(f"  Scenario {key} -- {name}: {margin_6m}")
    lines.append("")

    # --- Optimal rule ---
    lines.append("-" * 72)
    lines.append("  6. RECOMMENDED BUSINESS RULE")
    lines.append("-" * 72)

    optimal_rule = staffing_results.get("optimal_rule", {})
    rule_desc = optimal_rule.get("rule_description", "N/A")
    lines.append(f"  {rule_desc}")
    lines.append("")

    comparison = optimal_rule.get("comparison_with_always_replace", {})
    if comparison:
        lines.append("  Why this is better than 'always replace':")
        for k, v in comparison.items():
            lines.append(f"    - {v}")
    lines.append("")

    # --- Business assumptions ---
    lines.append("-" * 72)
    lines.append("  7. BUSINESS ASSUMPTIONS")
    lines.append("-" * 72)
    assumptions = [
        "Customer traffic remains stable.",
        "No significant hiring/recruitment costs are incurred.",
        "No training period productivity loss beyond the modelled ramp-up.",
        f"Monthly salary stays fixed at Rs. {salary} Lakhs per employee.",
        f"Commission remains fixed at {commission * 100:.1f}% of total sales.",
        "Economic conditions and consumer demand remain similar.",
        "No major competitor entry or exit in the mall.",
        "Store operating hours and mall footfall remain unchanged.",
        "Branded clothing pricing strategy remains consistent.",
    ]
    for assumption in assumptions:
        lines.append(f"    - {assumption}")
    lines.append("")

    # --- Closing ---
    lines.append("-" * 72)
    lines.append("  8. METHODOLOGY NOTES")
    lines.append("-" * 72)
    lines.append("  - All monetary values are in Lakhs INR.")
    lines.append("  - Salesperson productivity is measured as individual revenue")
    lines.append("    contribution relative to total monthly sales.")
    lines.append("  - Forecasts are generated using time-series models trained on")
    lines.append("    historical monthly sales data (Moving Average, Exponential")
    lines.append("    Smoothing, Linear Regression) with automatic model selection")
    lines.append("    based on validation RMSE.")
    lines.append("  - Staffing recommendations consider seasonality, forecast")
    lines.append("    demand, diminishing returns analysis, and per-employee")
    lines.append("    efficiency metrics.")
    lines.append("  - Scenario analysis includes new-hire ramp-up productivity")
    lines.append("    curve (70% -> 100% over 6 months).")
    lines.append("=" * 72)

    return "\n".join(lines)


# 4. Save Analysis Results

def save_analysis_results(
    summary_df: pd.DataFrame,
    financial_results: dict,
    output_dir: str,
) -> str:
    """Persist key analysis results to a CSV file.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Monthly aggregated summary.
    financial_results : dict
        Output of ``generate_financial_report``.
    output_dir : str
        Directory in which ``analysis_results.csv`` is created.

    Returns
    -------
    str
        Absolute path to the saved CSV file.
    """
    logger.info("Saving analysis results to CSV ...")

    results_df = summary_df.copy()

    # Merge scalar financial metrics from the nested summary_statistics dict
    fin_stats = financial_results.get("summary_statistics", {})
    if isinstance(fin_stats, dict):
        for key in [
            "avg_sales",
            "median_sales",
            "std_sales",
            "avg_margin",
            "total_margin",
            "min_margin",
            "max_margin",
            "avg_employees",
            "avg_sales_per_emp",
        ]:
            if key in fin_stats:
                results_df[key] = fin_stats[key]

    csv_path = os.path.join(output_dir, "analysis_results.csv")
    results_df.to_csv(csv_path, index=False)

    logger.info("Analysis results saved -> %s", csv_path)
    print(f"  Analysis results CSV saved -> {csv_path}")
    return csv_path


# 5. Master Orchestrator

def run_full_analysis(
    filepath: str,
    output_base_dir: str,
    salary: float = 3.0,
    commission: float = 0.05,
) -> dict:
    """Execute the complete RevInsight store analysis pipeline.

    Pipeline steps
    --------------
    a. Load and validate data
    b. Prepare monthly summary
    c. Setup output directories
    d. Run data understanding
    e. Run EDA and generate charts
    f. Run financial analysis
    g. Run forecasting
    h. Run staffing analysis
    i. Generate recommendation text
    j. Save recommendation to ``output/recommendation.txt``
    k. Save analysis results to ``output/analysis_results.csv``
    l. Generate PDF report
    m. Return all results

    Parameters
    ----------
    filepath : str
        Path to the raw CSV data file.
    output_base_dir : str
        Root directory for all generated outputs (charts, reports, CSVs).

    Returns
    -------
    dict
        Consolidated dictionary containing all sub-results keyed by stage
        name (e.g. ``data_understanding``, ``eda``, ``financial``,
        ``forecast``, ``staffing``, ``recommendation_text``, ``output_paths``).
    """
    pipeline_start = datetime.now()
    logger.info("=" * 60)
    logger.info("RevInsight Full Analysis Pipeline -- START")
    logger.info("=" * 60)

    all_results: dict = {}

    # ---- (a) Load and validate data --------------------------------
    try:
        logger.info("Step [a] Loading and validating data ...")
        print("\n>> Loading data ...")
        df = load_and_validate_data(filepath)
        logger.info("Data loaded: %d rows x %d cols", *df.shape)
        print(f"  [OK] Loaded {df.shape[0]} rows x {df.shape[1]} columns")
    except Exception as exc:
        logger.critical("Data loading failed -- aborting pipeline: %s", exc, exc_info=True)
        print(f"  [FAIL] Data loading FAILED: {exc}")
        return {"error": f"Data loading failed: {exc}"}

    # ---- (b) Prepare monthly summary --------------------------------
    try:
        logger.info("Step [b] Preparing monthly summary ...")
        print(">> Preparing monthly summary ...")
        summary_df = prepare_monthly_summary(df, salary=salary, commission=commission)
        logger.info("Monthly summary ready: %d rows", len(summary_df))
        print(f"  [OK] Monthly summary: {len(summary_df)} months")
    except Exception as exc:
        logger.error("Monthly summary preparation failed: %s", exc, exc_info=True)
        print(f"  [FAIL] Monthly summary FAILED: {exc}")
        return {"error": f"Monthly summary failed: {exc}"}

    # ---- (c) Setup output directories --------------------------------
    try:
        logger.info("Step [c] Setting up output directories ...")
        print(">> Setting up output directories ...")
        dirs_dict = setup_output_dirs(output_base_dir)
        # setup_output_dirs returns a dict; use the output_base_dir as the
        # string path for downstream functions that expect a string.
        output_dir = str(output_base_dir)
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Output directory: %s", output_dir)
        print(f"  [OK] Output dir: {output_dir}")
    except Exception as exc:
        logger.error("Output directory setup failed: %s", exc, exc_info=True)
        print(f"  [FAIL] Output dir setup FAILED: {exc}")
        # Fallback: use output_base_dir directly
        output_dir = str(output_base_dir)
        os.makedirs(output_dir, exist_ok=True)

    # Determine project root and reports directory
    project_root = Path(output_dir).parent
    reports_dir = project_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    charts_dir = os.path.join(output_dir, "charts")
    os.makedirs(charts_dir, exist_ok=True)

    # ---- (d) Data understanding -------------------------------------
    try:
        logger.info("Step [d] Running data understanding ...")
        print(">> Running data understanding ...")
        data_understanding = run_data_understanding(df, summary_df)
        all_results["data_understanding"] = data_understanding
        print("  [OK] Data understanding complete")
    except Exception as exc:
        logger.error("Data understanding failed: %s", exc, exc_info=True)
        print(f"  [FAIL] Data understanding FAILED: {exc}")
        all_results["data_understanding"] = {"error": str(exc)}

    # ---- (e) EDA and charts -----------------------------------------
    try:
        logger.info("Step [e] Running EDA ...")
        print(">> Running EDA & generating charts ...")
        eda_results = run_eda(df, summary_df, output_dir)
        all_results["eda"] = eda_results
        print("  [OK] EDA complete")
    except Exception as exc:
        logger.error("EDA failed: %s", exc, exc_info=True)
        print(f"  [FAIL] EDA FAILED: {exc}")
        all_results["eda"] = {"error": str(exc)}

    # ---- (f) Financial analysis -------------------------------------
    try:
        logger.info("Step [f] Running financial analysis ...")
        print(">> Running financial analysis ...")
        financial_results = generate_financial_report(summary_df, salary_per_person=salary, commission_rate=commission)
        all_results["financial"] = financial_results
        print("  [OK] Financial analysis complete")
    except Exception as exc:
        logger.error("Financial analysis failed: %s", exc, exc_info=True)
        print(f"  [FAIL] Financial analysis FAILED: {exc}")
        financial_results = {"error": str(exc)}
        all_results["financial"] = financial_results

    # ---- (g) Forecasting --------------------------------------------
    try:
        logger.info("Step [g] Running forecasting ...")
        print(">> Running forecast analysis ...")
        # generate_forecast_report expects (sales_series, output_dir)
        sales_series = df.set_index("Month")["Sales"].astype(float)
        sales_series.index = sales_series.index.astype(int)
        forecast_results = generate_forecast_report(sales_series, charts_dir)
        all_results["forecast"] = forecast_results
        print("  [OK] Forecast analysis complete")
    except Exception as exc:
        logger.error("Forecasting failed: %s", exc, exc_info=True)
        print(f"  [FAIL] Forecasting FAILED: {exc}")
        forecast_results = {"error": str(exc)}
        all_results["forecast"] = forecast_results

    # ---- (h) Staffing analysis --------------------------------------
    try:
        logger.info("Step [h] Running staffing analysis ...")
        print(">> Running staffing analysis ...")
        # generate_staffing_report expects (summary_df, forecast_sales_list)
        # Extract forecast values from forecast_results
        if "error" not in forecast_results:
            forecast_sales_list = forecast_results.get("best_model", {}).get(
                "forecast_values", [100.0] * 6
            )
        else:
            # Fallback: use average of last 6 months
            forecast_sales_list = [float(summary_df["Sales"].iloc[-6:].mean())] * 6
        staffing_results = generate_staffing_report(df, forecast_sales_list, salary_per_employee=salary, commission_rate=commission)
        all_results["staffing"] = staffing_results
        print("  [OK] Staffing analysis complete")
    except Exception as exc:
        logger.error("Staffing analysis failed: %s", exc, exc_info=True)
        print(f"  [FAIL] Staffing analysis FAILED: {exc}")
        staffing_results = {"error": str(exc)}
        all_results["staffing"] = staffing_results

    # ---- (e2) Generate scenario comparison chart --------------------
    try:
        if "error" not in staffing_results:
            scenarios = staffing_results.get("scenario_comparison", {})
            if scenarios:
                from visualization import plot_scenario_comparison
                scenario_margins = {
                    f"Scenario {k}: {v.get('name', k)}": v.get("total_gross_margin_6m", 0)
                    for k, v in scenarios.items()
                }
                plot_scenario_comparison(scenario_margins, charts_dir)
                print("  [OK] Scenario comparison chart generated")
    except Exception as exc:
        logger.error("Scenario chart generation failed: %s", exc, exc_info=True)
        print(f"  [FAIL] Scenario chart FAILED: {exc}")

    # ---- (i) Generate recommendation text ---------------------------
    try:
        logger.info("Step [i] Generating recommendation text ...")
        print(">> Generating recommendation ...")
        recommendation_text = generate_recommendation_text(
            staffing_results, financial_results, forecast_results,
            salary=salary, commission=commission
        )
        all_results["recommendation_text"] = recommendation_text
        print("  [OK] Recommendation generated")
    except Exception as exc:
        logger.error("Recommendation generation failed: %s", exc, exc_info=True)
        print(f"  [FAIL] Recommendation generation FAILED: {exc}")
        recommendation_text = f"Recommendation generation failed: {exc}"
        all_results["recommendation_text"] = recommendation_text

    # ---- (j) Save recommendation to file ----------------------------
    output_paths: dict[str, str] = {}
    try:
        rec_path = os.path.join(output_dir, "recommendation.txt")
        with open(rec_path, "w", encoding="utf-8") as fh:
            fh.write(recommendation_text)
        output_paths["recommendation"] = rec_path
        logger.info("Recommendation saved -> %s", rec_path)
        print(f"  [OK] Recommendation saved -> {rec_path}")
    except Exception as exc:
        logger.error("Failed to save recommendation: %s", exc, exc_info=True)
        print(f"  [FAIL] Failed to save recommendation: {exc}")

    # ---- (k) Save analysis results CSV ------------------------------
    try:
        csv_path = save_analysis_results(summary_df, financial_results, output_dir)
        output_paths["analysis_results_csv"] = csv_path
    except Exception as exc:
        logger.error("Failed to save analysis CSV: %s", exc, exc_info=True)
        print(f"  [FAIL] Failed to save analysis CSV: {exc}")

    # ---- (l) Generate PDF report ------------------------------------
    try:
        logger.info("Step [l] Generating PDF report ...")
        print(">> Generating PDF report ...")
        pdf_path = generate_pdf_report(
            all_results=all_results,
            summary_df=summary_df,
            charts_dir=charts_dir,
            output_path=str(reports_dir / "Recommendation_Report.pdf"),
            salary=salary,
            commission=commission
        )
        output_paths["pdf_report"] = pdf_path
        print(f"  [OK] PDF report saved -> {pdf_path}")
    except Exception as exc:
        logger.error("PDF report generation failed: %s", exc, exc_info=True)
        print(f"  [FAIL] PDF report FAILED: {exc}")

    all_results["output_paths"] = output_paths

    # ---- (m) Done ---------------------------------------------------
    elapsed = (datetime.now() - pipeline_start).total_seconds()
    logger.info("Pipeline finished in %.1f s", elapsed)
    print(f"\n{'=' * 70}")
    print(f"  PIPELINE COMPLETE -- elapsed {elapsed:.1f}s")
    print(f"  Outputs -> {output_dir}")
    print(f"{'=' * 70}\n")

    return all_results


# CLI entry-point

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="RevInsight -- Branded-Clothing Store Staffing Analysis",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data",
            "RevI-Test.csv",
        ),
        help="Path to the input CSV data file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "output",
        ),
        help="Base directory for analysis outputs.",
    )
    args = parser.parse_args()

    results = run_full_analysis(filepath=args.data, output_base_dir=args.output)

    if "error" in results:
        logger.critical("Pipeline terminated with error: %s", results["error"])
    else:
        logger.info("All stages completed -- results keys: %s", list(results.keys()))
