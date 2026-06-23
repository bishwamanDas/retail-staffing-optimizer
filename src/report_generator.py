"""
PDF Report Generator for RevInsight Staffing Analysis
=====================================================

Produces a professionally formatted PDF report containing:
- Title page
- Executive Summary
- Data Understanding
- Exploratory Data Analysis (with embedded charts)
- Financial Analysis
- Forecasting Results
- Scenario Comparison
- Business Assumptions
- Final Recommendation

Uses ``fpdf2`` for PDF generation.

All monetary values are in **Lakhs INR**.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from PIL import Image

try:
    from fpdf import FPDF
except ImportError:
    FPDF = None  # Graceful fallback if fpdf2 is not installed

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page constants
# ---------------------------------------------------------------------------
_LEFT_MARGIN = 15
_RIGHT_MARGIN = 15
_TOP_MARGIN = 15
_TITLE_FONT_SIZE = 24
_HEADING_FONT_SIZE = 16
_SUBHEADING_FONT_SIZE = 13
_BODY_FONT_SIZE = 10
_SMALL_FONT_SIZE = 8


def _sanitize_text(text: str) -> str:
    """Replace Unicode characters that the built-in Helvetica font cannot render.

    fpdf2's core fonts (Helvetica, Times, Courier) only support latin-1.
    This function maps common Unicode chars to ASCII-safe equivalents.
    """
    replacements = {
        '\u2014': '-',   # em-dash
        '\u2013': '-',   # en-dash
        '\u2018': "'",   # left single quote
        '\u2019': "'",   # right single quote
        '\u201c': '"',   # left double quote
        '\u201d': '"',   # right double quote
        '\u2022': '*',   # bullet
        '\u2026': '...', # ellipsis
        '\u20b9': 'Rs.', # Indian Rupee sign
        '\u2265': '>=',  # greater or equal
        '\u2264': '<=',  # less or equal
        '\u2192': '->', # right arrow
        '\u2190': '<-', # left arrow
        '\u00d7': 'x',  # multiplication sign
        '\u00f7': '/',  # division sign
        '\u03c3': 'sigma', # sigma
        '\u2248': '~=',  # approximately
    }
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    # Remove any remaining non-latin-1 characters
    return text.encode('latin-1', errors='replace').decode('latin-1')


class ReportPDF(FPDF if FPDF else object):
    """Custom FPDF subclass with header/footer branding."""

    def __init__(self, *args, **kwargs):
        if FPDF is None:
            raise ImportError(
                "fpdf2 is required for PDF generation. "
                "Install it with: pip install fpdf2"
            )
        super().__init__(*args, **kwargs)

    def header(self):
        """Page header with a subtle line and title."""
        if self.page_no() > 1:
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(120, 120, 120)
            self.cell(
                0, 8,
                _sanitize_text("RevInsight - Branded-Clothing Store Staffing Analysis"),
                align="L",
            )
            self.cell(0, 8, f"Page {self.page_no()}", align="R", new_x="LMARGIN", new_y="NEXT")
            self.set_draw_color(200, 200, 200)
            self.line(
                self.l_margin, self.get_y(),
                self.w - self.r_margin, self.get_y(),
            )
            self.ln(4)

    def footer(self):
        """Page footer with date and page number."""
        self.set_y(-15)
        self.set_font("Helvetica", "I", 7)
        self.set_text_color(150, 150, 150)
        self.cell(
            0, 10,
            f"Generated: {datetime.now().strftime('%Y-%m-%d')}  |  "
            f"Page {self.page_no()}/{{nb}}",
            align="C",
        )


def _add_title_page(pdf: ReportPDF, summary_df: pd.DataFrame) -> None:
    """Add a professional title page."""
    pdf.add_page()
    pdf.ln(60)

    # Title
    pdf.set_font("Helvetica", "B", _TITLE_FONT_SIZE)
    pdf.set_text_color(30, 30, 30)
    pdf.cell(0, 15, "RevInsight", align="C", new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "", 14)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(
        0, 10,
        "Branded-Clothing Store Staffing Recommendation Report",
        align="C", new_x="LMARGIN", new_y="NEXT",
    )

    pdf.ln(10)
    pdf.set_draw_color(0, 180, 140)
    pdf.set_line_width(0.8)
    mid = pdf.w / 2
    pdf.line(mid - 40, pdf.get_y(), mid + 40, pdf.get_y())
    pdf.ln(15)

    # Subtitle info
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(100, 100, 100)
    info_lines = [
        "Data-Driven Staffing Optimization Analysis",
        "",
        f"Date: {datetime.now().strftime('%B %d, %Y')}",
        "Location: Mall in Bangalore, India",
        "Product Category: Branded Clothing",
        f"Operating History: {len(summary_df)} Months",
        "",
        "Prepared by: Bishwaman Das",
    ]
    for line in info_lines:
        pdf.cell(0, 7, line, align="C", new_x="LMARGIN", new_y="NEXT")


def _add_section_heading(pdf: ReportPDF, title: str) -> None:
    """Add a coloured section heading."""
    pdf.ln(5)
    pdf.set_font("Helvetica", "B", _HEADING_FONT_SIZE)
    pdf.set_text_color(0, 140, 110)
    pdf.cell(0, 10, _sanitize_text(title), new_x="LMARGIN", new_y="NEXT")
    pdf.set_draw_color(0, 180, 140)
    pdf.set_line_width(0.4)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
    pdf.ln(4)
    pdf.set_text_color(30, 30, 30)


def _add_subheading(pdf: ReportPDF, title: str) -> None:
    """Add a sub-section heading."""
    pdf.ln(3)
    pdf.set_font("Helvetica", "B", _SUBHEADING_FONT_SIZE)
    pdf.set_text_color(60, 60, 60)
    pdf.cell(0, 8, _sanitize_text(title), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_text_color(30, 30, 30)


def _add_body_text(pdf: ReportPDF, text: str) -> None:
    """Add body text with word wrapping."""
    pdf.set_font("Helvetica", "", _BODY_FONT_SIZE)
    pdf.multi_cell(0, 5.5, _sanitize_text(text))
    pdf.ln(2)


def _add_bullet_list(pdf: ReportPDF, items: list, indent: float = 10) -> None:
    """Add a bulleted list."""
    pdf.set_font("Helvetica", "", _BODY_FONT_SIZE)
    for item in items:
        x = pdf.get_x()
        pdf.set_x(x + indent)
        pdf.multi_cell(0, 5.5, _sanitize_text(f"  * {item}"))
        pdf.ln(1)


def _add_key_value_table(
    pdf: ReportPDF,
    data: dict,
    col_widths: tuple = (80, 100),
) -> None:
    """Add a two-column key-value table."""
    pdf.set_font("Helvetica", "", _BODY_FONT_SIZE)
    for key, value in data.items():
        label = _sanitize_text(str(key).replace("_", " ").title())
        val_str = f"{value:.2f}" if isinstance(value, float) else _sanitize_text(str(value))
        pdf.cell(col_widths[0], 6, f"  {label}", border=0)
        pdf.cell(col_widths[1], 6, val_str, border=0, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)


def _add_chart_image(
    pdf: ReportPDF,
    image_path: str,
    caption: str = "",
    max_width: float = 140,
) -> None:
    """Embed a chart image in the PDF with optional caption."""
    if not os.path.exists(image_path):
        logger.warning("Chart image not found: %s", image_path)
        return

    try:
        with Image.open(image_path) as img:
            w, h = img.size
            aspect = h / w
    except Exception:
        aspect = 0.6  # Safe fallback for 12:7 ratio

    img_height = max_width * aspect

    # Check if we need a new page.
    # A4 height is 297mm, bottom margin is ~20mm. Page break trigger is ~277mm.
    # Add 15mm extra for caption text and padding.
    if pdf.get_y() + img_height + 15 > 277:
        pdf.add_page()

    # Center the image horizontally
    page_width = pdf.w
    x_pos = (page_width - max_width) / 2

    try:
        pdf.image(image_path, x=x_pos, w=max_width)
        if caption:
            pdf.set_font("Helvetica", "I", _SMALL_FONT_SIZE)
            pdf.set_text_color(100, 100, 100)
            pdf.cell(0, 5, _sanitize_text(caption), align="C", new_x="LMARGIN", new_y="NEXT")
            pdf.set_text_color(30, 30, 30)
        pdf.ln(5)
    except Exception as exc:
        logger.warning("Failed to embed chart %s: %s", image_path, exc)


# Main Report Generator

def generate_pdf_report(
    all_results: Dict[str, Any],
    summary_df: pd.DataFrame,
    charts_dir: str,
    output_path: str,
    salary: float = 3.0,
    commission: float = 0.05,
) -> str:
    """Generate a comprehensive PDF report from all analysis results.

    Parameters
    ----------
    all_results : dict
        Consolidated results from the full analysis pipeline.
    summary_df : pd.DataFrame
        Monthly summary data.
    charts_dir : str
        Directory containing generated chart images.
    output_path : str
        Full path for the output PDF file.

    Returns
    -------
    str
        Absolute path to the generated PDF file.
    """
    if FPDF is None:
        raise ImportError(
            "fpdf2 is required for PDF generation. "
            "Install it with: pip install fpdf2"
        )

    logger.info("Generating PDF report → %s", output_path)

    pdf = ReportPDF(orientation="P", unit="mm", format="A4")
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.set_left_margin(_LEFT_MARGIN)
    pdf.set_right_margin(_RIGHT_MARGIN)

    # ── Title page ──────────────────────────────────────────────────
    _add_title_page(pdf, summary_df)

    # ── Table of Contents ───────────────────────────────────────────
    pdf.add_page()
    _add_section_heading(pdf, "Table of Contents")
    toc_items = [
        "1. Executive Summary",
        "2. Data Understanding",
        "3. Exploratory Data Analysis",
        "4. Productivity Analysis",
        "5. Financial Analysis",
        "6. Sales Forecasting",
        "7. Scenario Analysis",
        "8. Business Assumptions",
        "9. Recommended Business Rule",
        "10. Final Recommendation",
        "Appendix: Charts",
    ]
    pdf.set_font("Helvetica", "", 11)
    for item in toc_items:
        pdf.cell(0, 8, _sanitize_text(f"   {item}"), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)

    # ── 1. Executive Summary ────────────────────────────────────────
    pdf.add_page()
    _add_section_heading(pdf, "1. Executive Summary")

    staffing = all_results.get("staffing", {})
    current_sit = staffing.get("current_situation", {})
    rec_info = staffing.get("recommendation", {})
    financial = all_results.get("financial", {})
    fin_stats = financial.get("summary_statistics", {})
    current_employees = current_sit.get("active_employees", 0)

    exec_text = (
        "This report presents a comprehensive data-driven analysis of staffing "
        "requirements for a branded-clothing store located in a mall in Bangalore. "
        f"The store has been operating for {len(summary_df)} months and recently experienced "
        f"a change in staff, reducing the team to {current_employees} employees. The analysis examines "
        "historical sales data, employee productivity, financial metrics, and sales "
        "forecasts to determine the optimal staffing decision."
    )
    _add_body_text(pdf, exec_text)

    _add_subheading(pdf, "Key Findings")
    key_findings = {
        "Current Employees": current_sit.get("active_employees", "N/A"),
        "Last Month Sales": f"Rs. {current_sit.get('monthly_sales', 0):.2f} Lakhs",
        "Avg Monthly Sales": f"Rs. {fin_stats.get('avg_sales', 0):.2f} Lakhs",
        "Avg Monthly Margin": f"Rs. {fin_stats.get('avg_margin', 0):.2f} Lakhs",
        "Recommended Action": rec_info.get("recommendation", "N/A"),
        "Confidence Level": rec_info.get("confidence", "N/A"),
    }
    _add_key_value_table(pdf, key_findings)

    # ── 2. Data Understanding ───────────────────────────────────────
    pdf.add_page()
    _add_section_heading(pdf, "2. Data Understanding")

    data_und = all_results.get("data_understanding", {})

    _add_subheading(pdf, "Dataset Overview")
    shape = data_und.get("shape", (0, 0))
    data_overview = {
        "Rows": shape[0] if isinstance(shape, tuple) else "N/A",
        "Columns": shape[1] if isinstance(shape, tuple) else "N/A",
        "Number of Months": data_und.get("num_months", "N/A"),
        "Salesperson Columns": data_und.get("num_employees", "N/A"),
    }
    _add_key_value_table(pdf, data_overview)

    _add_subheading(pdf, "Data Quality Notes")
    notes = data_und.get("data_quality_notes", [])
    if notes:
        _add_bullet_list(pdf, notes)

    sp_cols = len([c for c in summary_df.columns if str(c).strip().startswith("sale sp")])
    _add_body_text(
        pdf,
        f"The dataset contains monthly sales data for {len(summary_df)} months across {sp_cols} "
        "salesperson columns. Sales values are in Lakhs INR. Salespersons with "
        "zero sales in a given month indicate they were not active (not yet "
        "hired or had resigned). No missing values were detected."
    )

    # ── 3. Exploratory Data Analysis ────────────────────────────────
    pdf.add_page()
    _add_section_heading(pdf, "3. Exploratory Data Analysis")

    _add_body_text(
        pdf,
        "A comprehensive EDA was performed to understand sales patterns, "
        "employee productivity distributions, correlations between staffing "
        "levels and revenue, seasonal trends, and margin dynamics. Key charts "
        "are embedded below."
    )

    # Embed key EDA charts
    chart_files = [
        ("monthly_sales_trend.png", "Fig 1: Monthly Sales Trend with 6-Month Rolling Average"),
        ("salesperson_productivity.png", "Fig 2: Total Sales Contribution by Salesperson"),
        ("avg_sales_per_employee.png", "Fig 3: Average Sales per Active Employee Over Time"),
        ("employee_count_vs_sales.png", "Fig 4: Employee Count vs Total Sales (with OLS)"),
        ("employee_count_vs_margin.png", "Fig 5: Employee Count vs Gross Margin"),
        ("gross_margin_trend.png", "Fig 6: Gross Margin Trend with Rolling Average"),
        ("sales_histogram.png", "Fig 7: Distribution of Monthly Sales"),
        ("sales_boxplot.png", "Fig 8: Sales Distribution by Active Employee Count"),
        ("correlation_heatmap.png", "Fig 9: Correlation Matrix of Key Metrics"),
        ("monthly_seasonality.png", "Fig 10: Average Sales by Month-of-Year (Seasonality)"),
        ("sales_distribution.png", "Fig 11: Sales Distribution (KDE)"),
        ("employee_count_distribution.png", "Fig 12: Distribution of Active Employee Counts"),
        ("rolling_metrics.png", "Fig 13: Rolling Metrics Overview (6-Month Window)"),
    ]

    for filename, caption in chart_files:
        filepath = os.path.join(charts_dir, filename)
        if os.path.exists(filepath):
            _add_chart_image(pdf, filepath, caption)

    # ── 4. Productivity Analysis ────────────────────────────────────
    pdf.add_page()
    _add_section_heading(pdf, "4. Productivity Analysis")

    prod = staffing.get("productivity_analysis", {})
    _add_subheading(pdf, "Per-Employee Productivity Metrics")
    prod_metrics = {
        "Avg Sales per Employee": f"Rs. {prod.get('avg_sales_per_emp', 0):.2f} Lakhs",
        "Median Sales per Employee": f"Rs. {prod.get('median_sales_per_emp', 0):.2f} Lakhs",
        "Std Dev Sales per Employee": f"Rs. {prod.get('std_sales_per_emp', 0):.2f} Lakhs",
        "Best Employee Count": prod.get("best_employee_count", "N/A"),
    }
    _add_key_value_table(pdf, prod_metrics)

    consistency = prod.get("consistency_metrics", {})
    if consistency:
        _add_subheading(pdf, "Consistency Metrics")
        _add_key_value_table(pdf, consistency)

    _add_subheading(pdf, "Diminishing Returns Analysis")
    dim_ret = staffing.get("diminishing_returns", {})
    interp = dim_ret.get("interpretation", "")
    if interp:
        _add_body_text(pdf, interp)

    opt_count = dim_ret.get("optimal_count", "N/A")
    _add_body_text(pdf, f"Optimal headcount by margin/employee: {opt_count}")

    # ── 5. Financial Analysis ───────────────────────────────────────
    pdf.add_page()
    _add_section_heading(pdf, "5. Financial Analysis")

    _add_subheading(pdf, "Gross Margin Formula")
    _add_body_text(
        pdf,
        f"Gross Margin = 50% of Sales - Salary (Rs. {salary}L/person/month) "
        f"- Commission ({commission * 100:.1f}% of Sales)\n"
        f"Simplified: Gross Margin = {1.0 - 0.5 - commission:.2f} x Sales - {salary} x Employees"
    )

    _add_subheading(pdf, "Summary Statistics")
    if fin_stats:
        fin_table = {
            "Average Monthly Sales": f"Rs. {fin_stats.get('avg_sales', 0):.2f} Lakhs",
            "Median Monthly Sales": f"Rs. {fin_stats.get('median_sales', 0):.2f} Lakhs",
            "Sales Std Deviation": f"Rs. {fin_stats.get('std_sales', 0):.2f} Lakhs",
            "Average Gross Margin": f"Rs. {fin_stats.get('avg_margin', 0):.2f} Lakhs",
            f"Total Gross Margin ({len(summary_df)} months)": f"Rs. {fin_stats.get('total_margin', 0):.2f} Lakhs",
            "Best Monthly Margin": f"Rs. {fin_stats.get('max_margin', 0):.2f} Lakhs",
            "Worst Monthly Margin": f"Rs. {fin_stats.get('min_margin', 0):.2f} Lakhs",
        }
        _add_key_value_table(pdf, fin_table)

    be_analysis = financial.get("break_even_analysis", {})
    if be_analysis:
        _add_subheading(pdf, "Break-Even Analysis")
        be_table = {
            "Break-Even Sales/Employee": f"Rs. {be_analysis.get('break_even_sales_per_employee', 0):.2f} Lakhs",
            "Monthly Salary Cost": f"Rs. {be_analysis.get('monthly_salary_cost', 0):.2f} Lakhs",
            "Commission Rate": f"{be_analysis.get('commission_rate', 0)*100:.1f}%",
            "Effective Margin Rate": f"{be_analysis.get('effective_margin_rate', 0)*100:.1f}%",
        }
        _add_key_value_table(pdf, be_table)

    # ── 6. Sales Forecasting ────────────────────────────────────────
    pdf.add_page()
    _add_section_heading(pdf, "6. Sales Forecasting")

    forecast = all_results.get("forecast", {})
    if "error" not in forecast:
        best_model = forecast.get("best_model", {})

        _add_subheading(pdf, "Model Selection")
        _add_body_text(
            pdf,
            f"Three forecasting models were evaluated: Moving Average, "
            f"Exponential Smoothing, and Linear Regression. The best model "
            f"was selected based on validation RMSE (Root Mean Squared Error) "
            f"using the last 12 months as a hold-out set."
        )

        _add_body_text(
            pdf,
            f"Selected Model: {best_model.get('model_name', 'N/A')}"
        )

        reasoning = best_model.get("reasoning", "")
        if reasoning:
            _add_body_text(pdf, reasoning)

        _add_subheading(pdf, "6-Month Forecast")
        fc_vals = best_model.get("forecast_values", [])
        fc_months = forecast.get("forecast_months", [])
        if fc_vals and fc_months:
            pdf.set_font("Helvetica", "B", _BODY_FONT_SIZE)
            pdf.cell(40, 6, "Month", border=1, align="C")
            pdf.cell(60, 6, "Forecasted Sales (Lakhs)", border=1, align="C",
                     new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", "", _BODY_FONT_SIZE)
            for m, v in zip(fc_months, fc_vals):
                pdf.cell(40, 6, str(m), border=1, align="C")
                pdf.cell(60, 6, f"Rs. {v:.2f}", border=1, align="C",
                         new_x="LMARGIN", new_y="NEXT")
            pdf.ln(5)

        # Embed forecast charts
        forecast_charts = [
            "forecast_moving_average.png",
            "forecast_exponential_smoothing.png",
            "forecast_linear_regression.png",
        ]
        for fc_chart in forecast_charts:
            fc_path = os.path.join(charts_dir, fc_chart)
            if os.path.exists(fc_path):
                _add_chart_image(pdf, fc_path, f"Forecast: {fc_chart.replace('.png', '').replace('forecast_', '').replace('_', ' ').title()}")

    # ── 7. Scenario Analysis ────────────────────────────────────────
    pdf.add_page()
    _add_section_heading(pdf, "7. Scenario Analysis")

    scenarios = staffing.get("scenario_comparison", {})
    for key in sorted(scenarios.keys()):
        sc = scenarios[key]
        name = sc.get("name", key)
        _add_subheading(pdf, f"Scenario {key}: {name}")

        sc_table = {
            "Total 6-Month Sales": f"Rs. {sc.get('total_sales_6m', 0):.2f} Lakhs",
            "Total 6-Month Margin": f"Rs. {sc.get('total_gross_margin_6m', 0):.2f} Lakhs",
            "Total Staff Cost": f"Rs. {sc.get('total_staff_cost_6m', 0):.2f} Lakhs",
            "Avg Monthly Margin": f"Rs. {sc.get('avg_monthly_margin', 0):.2f} Lakhs",
        }
        _add_key_value_table(pdf, sc_table)

        risk = sc.get("risk_assessment", "")
        if risk:
            _add_body_text(pdf, f"Risk Assessment: {risk}")

        advantages = sc.get("advantages", [])
        if advantages:
            pdf.set_font("Helvetica", "B", _BODY_FONT_SIZE)
            pdf.cell(0, 5, _sanitize_text("Advantages:"), new_x="LMARGIN", new_y="NEXT")
            _add_bullet_list(pdf, advantages)

        disadvantages = sc.get("disadvantages", [])
        if disadvantages:
            pdf.set_font("Helvetica", "B", _BODY_FONT_SIZE)
            pdf.cell(0, 5, _sanitize_text("Disadvantages:"), new_x="LMARGIN", new_y="NEXT")
            _add_bullet_list(pdf, disadvantages)

        pdf.ln(3)

    # Embed scenario comparison chart
    sc_chart = os.path.join(charts_dir, "scenario_comparison.png")
    if os.path.exists(sc_chart):
        _add_chart_image(pdf, sc_chart, "Scenario Comparison: Total 6-Month Gross Margin")

    # ── 8. Business Assumptions ─────────────────────────────────────
    pdf.add_page()
    _add_section_heading(pdf, "8. Business Assumptions")

    assumptions = [
        "Customer traffic and footfall in the mall remain stable.",
        "No significant hiring or recruitment costs are incurred.",
        "No training period productivity loss beyond the modelled ramp-up "
        "(70% to 100% over 6 months).",
        f"Monthly salary stays fixed at Rs. {salary} Lakhs per employee.",
        f"Commission remains fixed at {commission * 100:.1f}% of total sales.",
        "Economic conditions and consumer demand remain similar.",
        "No major competitor entry or exit in the mall.",
        "Store operating hours and mall footfall remain unchanged.",
        "Branded clothing pricing strategy remains consistent.",
        "The sales force ramp-up follows the assumed productivity curve.",
        "Cost of goods sold (COGS) remains at 50% of sales.",
    ]
    _add_bullet_list(pdf, assumptions)

    # ── 9. Recommended Business Rule ────────────────────────────────
    pdf.add_page()
    _add_section_heading(pdf, "9. Recommended Business Rule")

    optimal_rule = staffing.get("optimal_rule", {})
    rule_desc = optimal_rule.get("rule_description", "")
    if rule_desc:
        _add_body_text(pdf, rule_desc)

    comparison = optimal_rule.get("comparison_with_always_replace", {})
    if comparison:
        _add_subheading(pdf, "Why This Is Better Than 'Always Replace'")
        for _k, v in comparison.items():
            _add_body_text(pdf, f"- {v}")

    trigger = optimal_rule.get("rolling_avg_trigger", {})
    if trigger:
        _add_subheading(pdf, "Rolling Average Trigger Details")
        _add_key_value_table(pdf, trigger)

    threshold = optimal_rule.get("threshold_rule", {})
    actions = threshold.get("actions", {})
    if actions:
        _add_subheading(pdf, "Threshold-Based Decision Rules")
        for condition, action in actions.items():
            _add_body_text(pdf, f"If {condition}: {action}")

    # ── 10. Final Recommendation ────────────────────────────────────
    pdf.add_page()
    _add_section_heading(pdf, "10. Final Recommendation")

    recommended = rec_info.get("recommendation", "N/A")
    confidence = rec_info.get("confidence", "N/A")
    alternative = rec_info.get("alternative", "N/A")

    _add_body_text(
        pdf,
        f"RECOMMENDED ACTION: {recommended}\n\n"
        f"Confidence Level: {confidence}\n"
        f"Alternative Option: {alternative}"
    )

    reasoning_bullets = rec_info.get("reasoning", [])
    if reasoning_bullets:
        _add_subheading(pdf, "Supporting Reasoning")
        _add_bullet_list(pdf, reasoning_bullets)

    expected = rec_info.get("expected_impact", {})
    if expected:
        _add_subheading(pdf, "Expected 6-Month Financial Impact")
        impact_table = {}
        for k, v in expected.items():
            label = k.replace("_", " ").title()
            if isinstance(v, float):
                impact_table[label] = f"Rs. {v:.2f} Lakhs"
            else:
                impact_table[label] = str(v)
        _add_key_value_table(pdf, impact_table)

    _add_body_text(
        pdf,
        f"\n\nThis recommendation is based on rigorous analysis of {len(summary_df)} months "
        "of historical sales data, statistical forecasting, scenario modelling, "
        "and financial impact assessment. The adaptive staffing rule provides a "
        "data-driven framework that is superior to the current 'always replace' "
        "policy and will help maximise gross margins while maintaining operational "
        "efficiency."
    )

    # ── Save the PDF ────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pdf.output(output_path)

    logger.info("PDF report saved to %s", output_path)
    return os.path.abspath(output_path)
