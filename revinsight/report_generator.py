"""Report rendering for the RevInsight store-staffing analytics engine.

The module turns the v2.0 analysis payload produced by
:func:`revinsight.pipeline.run_analysis` into two board-ready artefacts:

* a paginated **PDF** report built with ``fpdf2`` (cover page, table of
  contents and ten analytical sections plus appendices), and
* a single-file, responsive, dark-themed **HTML** report whose charts are
  embedded as base64 data URIs so the file can be shared on its own.

Both renderers consume the same intermediate *view model*, so the PDF and
the HTML can never disagree about a number.

Robustness
----------
Every value read from the payload is treated as optional.  A missing key,
a ``None`` value or a non-finite float (``NaN`` / ``inf``) renders as
``"n/a"`` instead of raising.  Chart images are optional too: a missing
file is skipped silently and never aborts a report.

All monetary values are in **Lakhs INR** and are written with the ASCII
``"Rs."`` prefix so that the built-in PDF fonts can always render them.
"""

from __future__ import annotations

import base64
import html
import logging
import math
import os
import struct
import tempfile
from collections.abc import Iterable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from fpdf import FPDF
from fpdf.enums import XPos, YPos

logger = logging.getLogger(__name__)

__all__ = ["generate_pdf_report", "generate_html_report"]


# --------------------------------------------------------------------------- #
# Branding & layout constants                                                  #
# --------------------------------------------------------------------------- #
_PRODUCT: str = "RevInsight"
_SUBTITLE: str = "Store Staffing Analysis"
_RUPEE: str = "Rs."
_LAKHS: str = "L"
_NA: str = "n/a"

_FONT: str = "Helvetica"
_FONT_MONO: str = "Courier"

_LEFT_MARGIN: float = 15.0
_RIGHT_MARGIN: float = 15.0
_TOP_MARGIN: float = 22.0
_BOTTOM_MARGIN: float = 18.0
_CONTENT_WIDTH: float = 210.0 - _LEFT_MARGIN - _RIGHT_MARGIN

_COLOR_PRIMARY: tuple[int, int, int] = (11, 92, 82)
_COLOR_ACCENT: tuple[int, int, int] = (0, 176, 140)
_COLOR_INK: tuple[int, int, int] = (33, 37, 41)
_COLOR_MUTED: tuple[int, int, int] = (108, 117, 125)
_COLOR_HEADER_BG: tuple[int, int, int] = (11, 92, 82)
_COLOR_ZEBRA: tuple[int, int, int] = (243, 249, 247)
_COLOR_GRID: tuple[int, int, int] = (203, 214, 211)
_COLOR_BAND_HIGH: tuple[int, int, int] = (13, 138, 92)
_COLOR_BAND_MEDIUM: tuple[int, int, int] = (191, 134, 0)
_COLOR_BAND_LOW: tuple[int, int, int] = (176, 42, 42)

_FONT_SIZE_TITLE: float = 30.0
_FONT_SIZE_SUBTITLE: float = 13.0
_FONT_SIZE_HEADING: float = 15.0
_FONT_SIZE_SUBHEADING: float = 10.5
_FONT_SIZE_BODY: float = 9.5
_FONT_SIZE_TABLE: float = 8.2
_FONT_SIZE_SMALL: float = 7.4
_FONT_SIZE_MONO: float = 6.0

_MM_PER_POINT: float = 25.4 / 72.0
_LINE_FACTOR: float = 1.32
_CELL_PAD: float = 0.7
_ROW_MIN_HEIGHT: float = 5.6
_CHART_MAX_WIDTH: float = 170.0
_CHART_MAX_HEIGHT: float = 150.0
_IMAGE_BUDGET_BYTES: int = 10 * 1024 * 1024
_MONTH_TABLE_LIMIT: int = 24


# --------------------------------------------------------------------------- #
# Safe accessors                                                               #
# --------------------------------------------------------------------------- #
def _as_number(value: Any) -> float | None:
    """Coerce *value* into a finite float.

    Parameters
    ----------
    value : Any
        The value to coerce.  ``None``, booleans, non-numeric strings and
        non-finite floats are rejected.

    Returns
    -------
    float or None
        The finite numeric value, or ``None`` when *value* cannot be
        represented as a number.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        try:
            number = float(value.strip())
        except (TypeError, ValueError):
            return None
    else:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
    if not math.isfinite(number):
        return None
    return number


def _num(source: Any, key: str, default: float | None = None) -> float | None:
    """Return the numeric value stored under *key* in *source*.

    Parameters
    ----------
    source : Any
        A mapping-like payload fragment.
    key : str
        The key to read.
    default : float or None, optional
        Value returned when the key is absent or not numeric.

    Returns
    -------
    float or None
        The finite numeric value of *key*, or *default*.
    """
    if not isinstance(source, dict):
        return default
    number = _as_number(source.get(key))
    return default if number is None else number


def _text(source: Any, key: str, default: str = _NA) -> str:
    """Return a printable string for *key* in *source*.

    Parameters
    ----------
    source : Any
        A mapping-like payload fragment.
    key : str
        The key to read.
    default : str, optional
        Value returned when the key is absent, ``None`` or blank.

    Returns
    -------
    str
        A human-readable representation of the value.
    """
    if not isinstance(source, dict):
        return default
    value = source.get(key)
    if value is None:
        return default
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, (int, float)):
        number = _as_number(value)
        if number is None:
            return default
        if float(number).is_integer():
            return f"{int(number):,}"
        return f"{number:,.4f}".rstrip("0").rstrip(".")
    text = str(value).strip()
    return text or default


def _pick(source: Any, *keys: str, default: Any = None) -> Any:
    """Return the first value present among *keys* in *source*.

    The v2.0 contract uses snake_case keys while a few upstream
    dataframes still emit ``CamelCase`` column names; this helper keeps
    both spellings working.

    Parameters
    ----------
    source : Any
        A mapping-like payload fragment.
    *keys : str
        Candidate key names, tried in order.
    default : Any, optional
        Value returned when no candidate key is present.

    Returns
    -------
    Any
        The first matching value, or *default*.
    """
    if not isinstance(source, dict):
        return default
    for key in keys:
        if key in source and source[key] is not None:
            return source[key]
    return default


def _rows(source: Any, key: str) -> list[dict[str, Any]]:
    """Return the list of dicts stored under *key* in *source*.

    Parameters
    ----------
    source : Any
        A mapping-like payload fragment.
    key : str
        The key holding a list of records.

    Returns
    -------
    list of dict
        The record list, or an empty list when the key is missing or the
        payload is malformed.
    """
    if not isinstance(source, dict):
        return []
    value = source.get(key)
    if not isinstance(value, (list, tuple)):
        return []
    return [row for row in value if isinstance(row, dict)]


# --------------------------------------------------------------------------- #
# Formatters                                                                   #
# --------------------------------------------------------------------------- #
def _fmt_number(value: Any, digits: int = 2) -> str:
    """Format *value* as a plain number.

    Parameters
    ----------
    value : Any
        The value to format.
    digits : int, optional
        Number of decimal places.

    Returns
    -------
    str
        The formatted number, or ``"n/a"`` when *value* is not numeric.
    """
    number = _as_number(value)
    if number is None:
        return _NA
    if abs(number) >= 100_000:
        return f"{number:,.0f}"
    return f"{number:,.{digits}f}"


def _fmt_money(value: Any, digits: int = 2) -> str:
    """Format *value* as an amount in Lakhs INR.

    Parameters
    ----------
    value : Any
        The value to format.
    digits : int, optional
        Number of decimal places.

    Returns
    -------
    str
        For example ``"Rs. 1,234.56 L"``, or ``"n/a"``.
    """
    number = _as_number(value)
    if number is None:
        return _NA
    return f"{_RUPEE} {number:,.{digits}f} {_LAKHS}"


def _fmt_mixed(value: Any, digits: int = 4) -> str:
    """Format *value* as a number when possible, otherwise as text.

    Parameters
    ----------
    value : Any
        The value to format.
    digits : int, optional
        Number of decimal places used for numeric values.

    Returns
    -------
    str
        The formatted value, or ``"n/a"`` when it is empty.
    """
    number = _as_number(value)
    if number is not None:
        if float(number).is_integer():
            return f"{int(number):,}"
        return f"{number:,.{digits}f}".rstrip("0").rstrip(".")
    if value is None:
        return _NA
    text = str(value).strip()
    return text or _NA


def _fmt_ratio(value: Any, digits: int = 3) -> str:
    """Format *value* as a dimensionless ratio.

    Parameters
    ----------
    value : Any
        The value to format.
    digits : int, optional
        Number of decimal places.

    Returns
    -------
    str
        The formatted ratio, or ``"n/a"``.
    """
    number = _as_number(value)
    if number is None:
        return _NA
    return f"{number:,.{digits}f}"


def _fmt_percent_points(value: Any, digits: int = 1) -> str:
    """Format *value* that is already expressed in percentage points.

    Parameters
    ----------
    value : Any
        The value to format.
    digits : int, optional
        Number of decimal places.

    Returns
    -------
    str
        For example ``"42.0%"``, or ``"n/a"``.
    """
    number = _as_number(value)
    if number is None:
        return _NA
    return f"{number:,.{digits}f}%"


def _fmt_probability(value: Any, digits: int = 1) -> str:
    """Format a probability stored as a fraction.

    Parameters
    ----------
    value : Any
        The value to format.
    digits : int, optional
        Number of decimal places.

    Returns
    -------
    str
        For example ``"87.3%"``, or ``"n/a"``.
    """
    number = _as_number(value)
    if number is None:
        return _NA
    return f"{number * 100:,.{digits}f}%"


def _fmt_flag(value: Any) -> str:
    """Format a boolean-ish value as ``Yes`` / ``No``.

    Parameters
    ----------
    value : Any
        The value to format.

    Returns
    -------
    str
        ``"Yes"``, ``"No"`` or ``"n/a"``.
    """
    if value is None:
        return _NA
    if isinstance(value, bool):
        return "Yes" if value else "No"
    number = _as_number(value)
    if number is None:
        return _NA
    return "Yes" if number != 0 else "No"


_UNICODE_MAP: dict[str, str] = {
    "\u20b9": "Rs.",
    "\u2014": "-",
    "\u2013": "-",
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2022": "*",
    "\u2026": "...",
    "\u2265": ">=",
    "\u2264": "<=",
    "\u2192": "->",
    "\u2190": "<-",
    "\u00d7": "x",
    "\u00f7": "/",
    "\u03c3": "sigma",
    "\u2248": "~=",
    "\u2212": "-",
    "\u2261": "=",
    "\u2260": "!=",
    "\u00b1": "+/-",
    "\u2032": "'",
    "\u2033": '"',
    "\u00a0": " ",
    "\u202f": " ",
    "\u200b": "",
    "\ufeff": "",
    "\u00ad": "",
}


def _sanitize(text: Any) -> str:
    """Return *text* reduced to characters the built-in fonts can render.

    fpdf2's core fonts (Helvetica, Courier) only cover latin-1, so common
    typographic characters are mapped to ASCII equivalents and anything
    else is replaced.

    Parameters
    ----------
    text : Any
        The value to sanitise.

    Returns
    -------
    str
        An ASCII-safe rendering of *text*.
    """
    if text is None:
        return ""
    value = str(text)
    for char, replacement in _UNICODE_MAP.items():
        value = value.replace(char, replacement)
    return value.encode("latin-1", errors="replace").decode("latin-1")


# --------------------------------------------------------------------------- #
# KPI presentation map                                                         #
# --------------------------------------------------------------------------- #
_KPI_LABELS: dict[str, str] = {
    "current_employees": "Current employees",
    "optimal_headcount": "Capacity-optimum headcount",
    "avg_monthly_sales": "Average monthly sales",
    "avg_monthly_margin": "Average monthly gross margin",
    "total_margin": "Total gross margin",
    "best_margin": "Best monthly gross margin",
    "worst_margin": "Worst monthly gross margin",
    "sales_volatility": "Sales volatility (std. dev.)",
    "break_even_sales_per_employee": "Break-even sales per employee",
    "avg_sales_per_employee": "Average sales per employee",
    "productivity_cv": "Productivity variability (CV)",
    "forecast_avg_sales": "Forecast average monthly sales",
    "forecast_avg_margin": "Forecast average monthly margin",
    "best_forecast_model": "Best forecast model",
    "validation_rmse": "Validation RMSE",
    "validation_mape": "Validation MAPE",
    "validation_mae": "Validation MAE",
    "validation_smape": "Validation sMAPE",
    "validation_r2": "Validation R2",
    "diminishing_returns": "Diminishing returns detected",
    "staff_cost_6m": "Total staff cost ({m} months)",
    "payback_months": "Months to payback",
    "roi_pct": "Return on investment",
    "anomalies_detected": "Anomalous months detected",
    "data_completeness": "Data completeness",
    "confidence_level": "Confidence level",
    "seasonality_strength": "Seasonality strength",
    "detected_cycle_months": "Detected seasonal cycle (months)",
    "total_leavers": "Historical leavers",
    "marginal_sales_per_employee": "Marginal sales per employee",
    "marginal_margin_contribution": "Marginal margin contribution",
    "marginal_model_r2": "Marginal model R2",
}

_KPI_KINDS: dict[str, str] = {
    "current_employees": "count",
    "optimal_headcount": "count",
    "avg_monthly_sales": "money",
    "avg_monthly_margin": "money",
    "total_margin": "money",
    "best_margin": "money",
    "worst_margin": "money",
    "sales_volatility": "money",
    "break_even_sales_per_employee": "money",
    "avg_sales_per_employee": "money",
    "productivity_cv": "ratio",
    "forecast_avg_sales": "money",
    "forecast_avg_margin": "money",
    "best_forecast_model": "text",
    "validation_rmse": "money",
    "validation_mape": "points",
    "validation_mae": "money",
    "validation_smape": "points",
    "validation_r2": "ratio",
    "diminishing_returns": "flag",
    "staff_cost_6m": "money",
    "payback_months": "number",
    "roi_pct": "points",
    "anomalies_detected": "count",
    "data_completeness": "points",
    "confidence_level": "text",
    "seasonality_strength": "points",
    "detected_cycle_months": "count",
    "total_leavers": "count",
    "marginal_sales_per_employee": "money",
    "marginal_margin_contribution": "money",
    "marginal_model_r2": "ratio",
}


def _horizon_label(months: int, template: str) -> str:
    """Fill a label template with the projection horizon.

    Parameters
    ----------
    months : int
        The projection horizon in months.
    template : str
        A template containing the placeholder ``{m}``.

    Returns
    -------
    str
        The formatted label.
    """
    try:
        return template.format(m=months)
    except (IndexError, KeyError, ValueError):
        return template


_IMPACT_LABELS: dict[str, str] = {
    "total_sales_6m": "Total sales ({m} months)",
    "total_gross_margin_6m": "Total gross margin ({m} months)",
    "total_staff_cost_6m": "Total staff cost ({m} months)",
    "avg_monthly_margin": "Average monthly gross margin",
    "margin_advantage_over_runner_up": "Margin advantage over runner-up",
    "margin_advantage_vs_do_nothing": "Margin advantage vs do-nothing",
}

_DRIVER_LABELS: dict[str, str] = {
    "sales_consistency_cv": "Sales consistency (CV)",
    "historical_months": "Historical months",
    "headcount_observations": "Headcount observations",
    "margin_gap_to_runner_up": "Margin gap to runner-up",
}


def _fmt_auto(value: Any, digits: int = 4) -> str:
    """Format *value* as an integer when it is whole, else with decimals.

    Parameters
    ----------
    value : Any
        The value to format.
    digits : int, optional
        Number of decimal places for fractional values.

    Returns
    -------
    str
        The formatted value, or ``"n/a"``.
    """
    number = _as_number(value)
    if number is None:
        return _NA
    if float(number).is_integer():
        return f"{int(number):,}"
    return f"{number:,.{digits}f}"


def _horizon_from(results: Any) -> int:
    """Return the projection horizon in months for a results payload.

    Parameters
    ----------
    results : Any
        The analysis payload.

    Returns
    -------
    int
        The horizon, defaulting to 6 months when it is not stated.
    """
    assumptions = _pick(results, "assumptions", default={})
    months = _as_number(_pick(assumptions, "forecast_months", default=6))
    try:
        value = int(months) if months is not None else 6
    except (TypeError, ValueError):
        value = 6
    return value if value > 0 else 6


def _kpi_label(key: str) -> str:
    """Return a friendly label for a KPI key.

    Parameters
    ----------
    key : str
        The raw payload key.

    Returns
    -------
    str
        A human-readable label.
    """
    if key in _KPI_LABELS:
        return _KPI_LABELS[key]
    words = str(key).replace("_", " ").strip()
    return words[:1].upper() + words[1:] if words else _NA


def _kpi_value(key: str, value: Any) -> str:
    """Format a KPI value according to its declared kind.

    Parameters
    ----------
    key : str
        The raw payload key, used to look up the formatting kind.
    value : Any
        The value to format.

    Returns
    -------
    str
        The formatted value.
    """
    kind = _KPI_KINDS.get(key, "auto")
    if kind == "money":
        return _fmt_money(value)
    if kind == "ratio":
        return _fmt_ratio(value)
    if kind == "points":
        return _fmt_percent_points(value)
    if kind == "flag":
        return _fmt_flag(value)
    if kind == "count":
        return _fmt_number(value, 0)
    if kind == "number":
        return _fmt_number(value, 2)
    if kind == "text":
        return _text({"v": value}, "v")
    number = _as_number(value)
    if number is not None:
        return _fmt_number(number, 2)
    return _text({"v": value}, "v")


# --------------------------------------------------------------------------- #
# View-model block helpers                                                     #
# --------------------------------------------------------------------------- #
def _block_text(text: Any, title: str = "") -> dict[str, Any]:
    """Build a paragraph block.

    Parameters
    ----------
    text : Any
        The paragraph content.
    title : str, optional
        An optional lead-in label.

    Returns
    -------
    dict
        A renderable block.
    """
    return {"type": "text", "title": title, "text": str(text)}


def _block_bullets(items: Iterable[Any], title: str = "") -> dict[str, Any]:
    """Build a bulleted-list block.

    Parameters
    ----------
    items : iterable
        The bullet points.
    title : str, optional
        An optional lead-in label.

    Returns
    -------
    dict
        A renderable block.
    """
    return {
        "type": "bullets",
        "title": title,
        "items": [str(item) for item in items],
    }


def _block_kv(title: str, pairs: Sequence[tuple[str, str]]) -> dict[str, Any]:
    """Build a two-column key/value block.

    Parameters
    ----------
    title : str
        The table caption.
    pairs : sequence of (str, str)
        The label/value pairs.

    Returns
    -------
    dict
        A renderable block.
    """
    return {
        "type": "kv",
        "title": title,
        "rows": [[str(label), str(value)] for label, value in pairs],
    }


def _block_table(
    title: str,
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    ratios: Sequence[float] | None = None,
    aligns: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Build a data-table block.

    Parameters
    ----------
    title : str
        The table caption.
    headers : sequence of str
        The column headers.
    rows : sequence of sequence of str
        The table body.
    ratios : sequence of float, optional
        Relative column widths.
    aligns : sequence of str, optional
        Per-column alignment (``"L"``, ``"C"`` or ``"R"``).

    Returns
    -------
    dict
        A renderable block.
    """
    return {
        "type": "table",
        "title": title,
        "headers": [str(header) for header in headers],
        "rows": [[str(cell) for cell in row] for row in rows],
        "ratios": list(ratios) if ratios else None,
        "aligns": list(aligns) if aligns else None,
    }


def _block_chart(stem: str, caption: str = "") -> dict[str, Any]:
    """Build a chart-embedding block.

    Parameters
    ----------
    stem : str
        The chart file stem (without extension).
    caption : str, optional
        The caption rendered underneath the image.

    Returns
    -------
    dict
        A renderable block.
    """
    return {"type": "chart", "stem": stem, "caption": caption}


def _block_badge(text: Any, level: Any = "") -> dict[str, Any]:
    """Build a confidence-badge block.

    Parameters
    ----------
    text : Any
        The badge label.
    level : Any, optional
        The confidence level used to pick the badge colour.

    Returns
    -------
    dict
        A renderable block.
    """
    return {"type": "badge", "text": _text({"v": text}, "v"), "level": str(level)}


def _block_pre(text: Any, title: str = "") -> dict[str, Any]:
    """Build a monospace preformatted block.

    Parameters
    ----------
    text : Any
        The verbatim content.
    title : str, optional
        The table caption.

    Returns
    -------
    dict
        A renderable block.
    """
    return {"type": "pre", "title": title, "text": str(text)}


# --------------------------------------------------------------------------- #
# Section builders                                                             #
# --------------------------------------------------------------------------- #
def _expected_impact_rows(impact: Any, horizon: int = 6) -> list[list[str]]:
    """Return the expected-impact table rows.

    Parameters
    ----------
    impact : Any
        The ``expected_impact`` mapping.
    horizon : int, optional
        The projection horizon in months, used in the labels.

    Returns
    -------
    list of list of str
        Two-column rows (metric, value).
    """
    if not isinstance(impact, dict):
        return []
    rows: list[list[str]] = []
    for key, value in impact.items():
        label = _IMPACT_LABELS.get(key, str(key).replace("_", " ").capitalize())
        label = _horizon_label(horizon, label)
        number = _as_number(value)
        formatted = _fmt_money(number) if number is not None else _text({"v": value}, "v")
        rows.append([label, formatted])
    return rows


def _section_executive(results: dict[str, Any]) -> list[dict[str, Any]]:
    """Build the executive-summary blocks.

    Parameters
    ----------
    results : dict
        The analysis payload.

    Returns
    -------
    list of dict
        Renderable blocks.
    """
    executive = results.get("executive") if isinstance(results, dict) else None
    executive = executive if isinstance(executive, dict) else {}
    kpis = results.get("kpis") if isinstance(results, dict) else None
    kpis = kpis if isinstance(kpis, dict) else {}

    blocks: list[dict[str, Any]] = []
    headline = _text(executive, "headline")
    blocks.append(_block_text(headline, "Headline"))
    blocks.append(_block_badge(_text(executive, "confidence"), _text(executive, "confidence")))

    blocks.append(
        _block_kv(
            "Recommendation at a glance",
            [
                ("Recommended action", _text(executive, "recommendation")),
                ("Recommended scenario", _text(executive, "recommended_scenario")),
                ("Confidence level", _text(executive, "confidence")),
                ("Runner-up option", _text(executive, "runner_up")),
                ("Current employees", _fmt_number(kpis.get("current_employees"), 0)),
                (
                    "Capacity-optimum headcount",
                    _fmt_number(kpis.get("optimal_headcount"), 0),
                ),
            ],
        )
    )

    reasoning = _rows(executive, "reasoning")
    if reasoning:
        blocks.append(
            _block_bullets(
                [_text(item, "text") for item in reasoning],
                "Supporting reasoning",
            )
        )

    horizon = _horizon_from(results)
    impact_rows = _expected_impact_rows(executive.get("expected_impact"), horizon=horizon)
    if impact_rows:
        blocks.append(
            _block_table(
                f"Expected {horizon}-month financial impact",
                ["Metric", "Value"],
                impact_rows,
                ratios=[2.0, 1.0],
                aligns=["L", "R"],
            )
        )

    drivers = executive.get("confidence_drivers")
    if isinstance(drivers, dict):
        pairs = []
        for key, value in drivers.items():
            if key == "note":
                continue
            label = _DRIVER_LABELS.get(key, str(key).replace("_", " ").strip().capitalize())
            number = _as_number(value)
            pairs.append(
                (
                    label,
                    _fmt_auto(number) if number is not None else _text({"v": value}, "v"),
                )
            )
        if pairs:
            blocks.append(_block_kv("Confidence drivers", pairs))
        note = _text(drivers, "note")
        if note != _NA:
            blocks.append(_block_text(note))

    return blocks


def _monthly_table_rows(
    summary_df: pd.DataFrame,
    series: dict[str, Any] | None = None,
    limit: int = _MONTH_TABLE_LIMIT,
) -> tuple[list[list[str]], int, int]:
    """Return the most recent monthly gross-margin rows.

    The monthly summary frame is preferred; when it is missing or empty
    the equivalent ``series`` arrays of the payload are used instead so
    that both renderers show the same table.

    Parameters
    ----------
    summary_df : pd.DataFrame
        The monthly summary produced by the pipeline.
    series : dict, optional
        The ``series`` section of the payload, used as a fallback.
    limit : int, optional
        Maximum number of (most recent) months to include.

    Returns
    -------
    tuple
        ``(rows, shown, total)`` where *rows* holds the formatted table
        body, *shown* the number of rendered months and *total* the number
        of months available.
    """
    if isinstance(summary_df, pd.DataFrame) and not summary_df.empty:
        total = int(len(summary_df))
        tail = summary_df.tail(limit) if limit and limit > 0 else summary_df
        rows: list[list[str]] = []
        for _, record in tail.iterrows():
            rows.append(
                [
                    _fmt_number(record.get("Month"), 0),
                    _fmt_number(record.get("Sales")),
                    _fmt_number(record.get("ActiveEmployees"), 0),
                    _fmt_number(record.get("SalesPerEmployee")),
                    _fmt_number(record.get("GrossMargin")),
                ]
            )
        return rows, len(rows), total

    if not isinstance(series, dict):
        return [], 0, 0

    def _column(name: str) -> list[Any]:
        """Return a list payload column as a plain list."""
        values = series.get(name)
        return list(values) if isinstance(values, (list, tuple)) else []

    months = _column("months")
    sales = _column("sales")
    employees = _column("employees")
    per_employee = _column("sales_per_employee")
    margins = _column("gross_margin")
    length = max(len(months), len(sales), len(employees), len(per_employee), len(margins))
    if not length:
        return [], 0, 0
    start = max(0, length - limit) if limit and limit > 0 else 0
    rows = []
    for index in range(start, length):
        rows.append(
            [
                _fmt_number(months[index] if index < len(months) else None, 0),
                _fmt_number(sales[index] if index < len(sales) else None),
                _fmt_number(employees[index] if index < len(employees) else None, 0),
                _fmt_number(per_employee[index] if index < len(per_employee) else None),
                _fmt_number(margins[index] if index < len(margins) else None),
            ]
        )
    return rows, len(rows), length


def _section_financial(
    results: dict[str, Any],
    summary_df: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Build the financial-highlights blocks.

    Parameters
    ----------
    results : dict
        The analysis payload.
    summary_df : pd.DataFrame
        The monthly summary produced by the pipeline.

    Returns
    -------
    list of dict
        Renderable blocks.
    """
    kpis = results.get("kpis") if isinstance(results, dict) else None
    kpis = kpis if isinstance(kpis, dict) else {}
    assumptions = results.get("assumptions") if isinstance(results, dict) else None
    assumptions = assumptions if isinstance(assumptions, dict) else {}

    blocks: list[dict[str, Any]] = []

    kpi_rows = [[_kpi_label(str(key)), _kpi_value(str(key), value)] for key, value in kpis.items()]
    if kpi_rows:
        blocks.append(
            _block_table(
                "Key performance indicators",
                ["Metric", "Value"],
                kpi_rows,
                ratios=[1.35, 1.0],
                aligns=["L", "R"],
            )
        )

    salary = _num(assumptions, "salary")
    commission = _num(assumptions, "commission")
    cogs = _num(assumptions, "cogs_rate")
    if salary is not None and commission is not None:
        rate = 1.0 - (cogs if cogs is not None else 0.5) - commission
        blocks.append(
            _block_text(
                "Gross margin = (1 - COGS - commission) x Sales - salary x "
                f"employees, i.e. {rate:.2f} x Sales - {_fmt_number(salary)} x "
                "employees, with salary "
                f"{_fmt_money(salary)} per employee per month and commission "
                f"{_fmt_percent_points(commission * 100)} of sales.",
                "Margin formula",
            )
        )

    series = results.get("series")
    series = series if isinstance(series, dict) else {}

    rows, shown, total = _monthly_table_rows(summary_df, series)
    if rows:
        blocks.append(
            _block_table(
                f"Monthly gross margin - most recent {shown} of {total} months",
                [
                    "Month",
                    "Sales",
                    "Active employees",
                    "Sales / employee",
                    "Gross margin",
                ],
                rows,
                ratios=[1.0, 1.25, 1.35, 1.45, 1.35],
                aligns=["C", "R", "R", "R", "R"],
            )
        )
        blocks.append(
            _block_text(
                f"Showing the {shown} most recent months of {total} available "
                "months of history. All values are in Lakhs INR.",
                "Note",
            )
        )

    vacancy = results.get("vacancy_cost")
    if isinstance(vacancy, dict):
        interpretation = _text(vacancy, "interpretation")
        if interpretation != _NA:
            blocks.append(_block_text(interpretation, "Cost of an open position"))

    seasonal_rows = _rows(series, "seasonal_index")
    if seasonal_rows:
        blocks.append(
            _block_table(
                "Seasonal demand profile",
                ["Month", "Name", "Index", "Strength", "Average sales"],
                [
                    [
                        _fmt_number(row.get("month"), 0),
                        _text(row, "name"),
                        _fmt_ratio(row.get("index")),
                        _fmt_ratio(row.get("strength")),
                        _fmt_number(row.get("avg_sales")),
                    ]
                    for row in seasonal_rows
                ],
                ratios=[1.0, 1.0, 1.0, 1.0, 1.4],
                aligns=["C", "L", "R", "R", "R"],
            )
        )

    anomaly_rows = _rows(series, "anomalies")
    if anomaly_rows:
        blocks.append(
            _block_table(
                "Detected anomalies",
                [
                    "Month",
                    "Name",
                    "Sales",
                    "Expected",
                    "Score",
                    "Direction",
                    "Severity",
                ],
                [
                    [
                        _fmt_number(row.get("month"), 0),
                        _text(row, "name"),
                        _fmt_number(row.get("sales")),
                        _fmt_number(row.get("expected")),
                        _fmt_ratio(row.get("score")),
                        _text(row, "direction"),
                        _text(row, "severity"),
                    ]
                    for row in anomaly_rows
                ],
                ratios=[1.0, 1.0, 1.2, 1.2, 1.0, 1.2, 1.0],
                aligns=["C", "L", "R", "R", "R", "L", "L"],
            )
        )

    return blocks


def _section_forecast(results: dict[str, Any]) -> list[dict[str, Any]]:
    """Build the forecast-outlook blocks.

    Parameters
    ----------
    results : dict
        The analysis payload.

    Returns
    -------
    list of dict
        Renderable blocks.
    """
    forecast = results.get("forecast")
    forecast = forecast if isinstance(forecast, dict) else {}
    kpis = results.get("kpis")
    kpis = kpis if isinstance(kpis, dict) else {}

    months = forecast.get("months")
    months = list(months) if isinstance(months, (list, tuple)) else []
    values = list(forecast.get("values") or [])
    lower = list(forecast.get("lower_95") or [])
    upper = list(forecast.get("upper_95") or [])

    # The prediction band width follows the configured confidence level.
    confidence = _as_number(
        _pick(_pick(results, "assumptions", default={}), "confidence_level", default=0.95)
    )
    band_pct = int(round(confidence * 100)) if confidence is not None else 95

    blocks: list[dict[str, Any]] = []
    blocks.append(
        _block_kv(
            "Model selection",
            [
                ("Selected model", _text(forecast, "model")),
                (
                    "Validation RMSE",
                    _fmt_number(
                        _pick(forecast, "validation_rmse", "rmse")
                        if _pick(forecast, "validation_rmse", "rmse") is not None
                        else kpis.get("validation_rmse")
                    ),
                ),
                (
                    "Validation MAPE",
                    _fmt_percent_points(
                        (
                            _pick(forecast, "validation_mape", "mape")
                            if _pick(forecast, "validation_mape", "mape") is not None
                            else kpis.get("validation_mape")
                        ),
                        digits=2,
                    ),
                ),
                ("Forecast horizon (months)", _fmt_number(len(months), 0)),
                ("Residual std. dev.", _fmt_number(forecast.get("residual_std"))),
                ("Methodology", _text(forecast, "methodology")),
            ],
        )
    )

    rows: list[list[str]] = []
    for index in range(max(len(values), len(months))):
        month = months[index] if index < len(months) else index + 1
        rows.append(
            [
                _fmt_number(month, 0),
                _fmt_number(values[index] if index < len(values) else None),
                _fmt_number(lower[index] if index < len(lower) else None),
                _fmt_number(upper[index] if index < len(upper) else None),
            ]
        )
    if rows:
        blocks.append(
            _block_table(
                f"Point forecast with {band_pct}% prediction interval",
                ["Month", "Point forecast", f"Lower {band_pct}%", f"Upper {band_pct}%"],
                rows,
                ratios=[1.0, 1.4, 1.3, 1.3],
                aligns=["C", "R", "R", "R"],
            )
        )

    members = _rows(forecast, "ensemble_members")
    if members:
        blocks.append(
            _block_table(
                "Ensemble composition",
                ["Model", "Weight", "RMSE", "MAE", "MAPE"],
                [
                    [
                        _text(row, "model"),
                        _fmt_probability(row.get("weight"), 1),
                        _fmt_number(row.get("rmse")),
                        _fmt_number(row.get("mae")),
                        _fmt_percent_points(row.get("mape"), 2),
                    ]
                    for row in members
                ],
                ratios=[2.2, 1.0, 1.0, 1.0, 1.0],
                aligns=["L", "R", "R", "R", "R"],
            )
        )

    metrics = _rows(forecast, "metrics")
    if metrics:
        leaderboard = [
            [
                _text(row, "model"),
                _fmt_number(row.get("rmse")),
                _fmt_number(row.get("mae")),
                _fmt_percent_points(row.get("mape"), 2),
                _fmt_number(row.get("mase")),
                _fmt_number(row.get("r2"), 3),
                _fmt_number(row.get("rank"), 0),
            ]
            for row in metrics
        ]
    else:
        leaderboard = [
            [
                _text(row, "model"),
                _fmt_number(row.get("rmse")),
                _fmt_number(row.get("mae")),
                _fmt_percent_points(row.get("mape"), 2),
                _NA,
                _NA,
                _fmt_number(index + 1, 0),
            ]
            for index, row in enumerate(members)
        ]
    if leaderboard:
        blocks.append(
            _block_table(
                "Model leaderboard",
                ["Model", "RMSE", "MAE", "MAPE", "MASE", "R2", "Rank"],
                leaderboard,
                ratios=[2.2, 1.0, 1.0, 1.0, 1.0, 1.0, 0.7],
                aligns=["L", "R", "R", "R", "R", "R", "C"],
            )
        )

    backtest = _rows(forecast, "backtest")
    if backtest:
        blocks.append(
            _block_table(
                "Walk-forward backtest",
                ["Origin month", "Actual", "Predicted", "Error"],
                [
                    [
                        _fmt_number(row.get("origin_month"), 0),
                        _fmt_number(row.get("actual")),
                        _fmt_number(row.get("predicted")),
                        _fmt_number(row.get("error")),
                    ]
                    for row in backtest
                ],
                ratios=[1.2, 1.3, 1.3, 1.2],
                aligns=["C", "R", "R", "R"],
            )
        )

    return blocks


def _section_staffing(results: dict[str, Any]) -> list[dict[str, Any]]:
    """Build the staffing-recommendation blocks.

    Parameters
    ----------
    results : dict
        The analysis payload.

    Returns
    -------
    list of dict
        Renderable blocks.
    """
    executive = results.get("executive")
    executive = executive if isinstance(executive, dict) else {}

    blocks: list[dict[str, Any]] = []
    blocks.append(_block_text(_text(executive, "recommendation"), "Recommended action"))
    blocks.append(_block_badge(_text(executive, "confidence"), _text(executive, "confidence")))
    blocks.append(
        _block_kv(
            "Decision summary",
            [
                ("Recommended action", _text(executive, "recommendation")),
                ("Recommended scenario", _text(executive, "recommended_scenario")),
                ("Confidence level", _text(executive, "confidence")),
                ("Runner-up option", _text(executive, "runner_up")),
            ],
        )
    )

    reasoning = _rows(executive, "reasoning")
    if reasoning:
        blocks.append(
            _block_bullets(
                [_text(item, "text") for item in reasoning],
                "Supporting reasoning",
            )
        )

    horizon = _horizon_from(results)
    impact_rows = _expected_impact_rows(executive.get("expected_impact"), horizon=horizon)
    if impact_rows:
        blocks.append(
            _block_table(
                f"Expected {horizon}-month financial impact",
                ["Metric", "Value"],
                impact_rows,
                ratios=[2.0, 1.0],
                aligns=["L", "R"],
            )
        )

    return blocks


def _section_scenarios(results: dict[str, Any]) -> list[dict[str, Any]]:
    """Build the scenario-comparison blocks.

    Parameters
    ----------
    results : dict
        The analysis payload.

    Returns
    -------
    list of dict
        Renderable blocks.
    """
    scenarios = results.get("scenarios")
    scenarios = scenarios if isinstance(scenarios, dict) else {}
    executive = results.get("executive")
    executive = executive if isinstance(executive, dict) else {}
    horizon = _horizon_from(results)

    blocks: list[dict[str, Any]] = []
    for key in sorted(scenarios.keys()):
        scenario = scenarios[key]
        if not isinstance(scenario, dict):
            continue
        name = _text(scenario, "name", default=str(key))
        blocks.append(_block_text(name, f"Scenario {key}"))
        blocks.append(
            _block_table(
                f"Scenario {key} - {name}",
                ["Metric", "Value"],
                [
                    [
                        f"{horizon}-month gross margin",
                        _fmt_money(scenario.get("total_gross_margin")),
                    ],
                    [
                        "Difference vs best scenario",
                        _fmt_money(scenario.get("delta_vs_best")),
                    ],
                    [
                        f"Total {horizon}-month sales",
                        _fmt_money(scenario.get("total_sales")),
                    ],
                    [
                        f"Total {horizon}-month staff cost",
                        _fmt_money(scenario.get("total_staff_cost")),
                    ],
                    [
                        "Average monthly margin",
                        _fmt_money(scenario.get("avg_monthly_margin")),
                    ],
                    ["P10 margin", _fmt_money(scenario.get("p10_margin"))],
                    ["P50 margin", _fmt_money(scenario.get("p50_margin"))],
                    ["P90 margin", _fmt_money(scenario.get("p90_margin"))],
                    [
                        "P(beats do-nothing baseline)",
                        _fmt_probability(scenario.get("prob_better_than_baseline")),
                    ],
                    ["P(margin > 0)", _fmt_probability(scenario.get("prob_positive"))],
                ],
                ratios=[1.4, 1.0],
                aligns=["L", "R"],
            )
        )
        risk = _text(scenario, "risk")
        if risk != _NA:
            blocks.append(_block_text(risk, "Risk assessment"))
        advantages = _rows(scenario, "advantages")
        if advantages:
            blocks.append(
                _block_bullets(
                    [_text(item, "text") for item in advantages],
                    "Advantages",
                )
            )
        disadvantages = _rows(scenario, "disadvantages")
        if disadvantages:
            blocks.append(
                _block_bullets(
                    [_text(item, "text") for item in disadvantages],
                    "Disadvantages",
                )
            )

    recommended = _text(executive, "recommended_scenario", default="")
    scenario = scenarios.get(recommended) if recommended else None
    if isinstance(scenario, dict):
        monthly = _rows(scenario, "monthly")
        if monthly:
            blocks.append(
                _block_table(
                    f"Recommended scenario {recommended} - month-by-month detail",
                    ["Month", "Employees", "Sales", "Gross margin", "Staff cost"],
                    [
                        [
                            _fmt_number(row.get("month"), 0),
                            _fmt_number(row.get("employees"), 0),
                            _fmt_number(row.get("sales")),
                            _fmt_number(row.get("gross_margin")),
                            _fmt_number(row.get("staff_cost")),
                        ]
                        for row in monthly
                    ],
                    ratios=[1.0, 1.1, 1.4, 1.4, 1.4],
                    aligns=["C", "C", "R", "R", "R"],
                )
            )

    return blocks


def _section_productivity(results: dict[str, Any]) -> list[dict[str, Any]]:
    """Build the productivity and diminishing-returns blocks.

    Parameters
    ----------
    results : dict
        The analysis payload.

    Returns
    -------
    list of dict
        Renderable blocks.
    """
    productivity = results.get("productivity")
    productivity = productivity if isinstance(productivity, dict) else {}
    horizon = _horizon_from(results)

    blocks: list[dict[str, Any]] = []
    detected = _fmt_flag(productivity.get("diminishing_detected"))
    optimal = _fmt_number(productivity.get("optimal_headcount"), 0)
    blocks.append(
        _block_text(
            f"Diminishing returns detected: {detected}. " f"Capacity-optimum headcount: {optimal}.",
            "Productivity summary",
        )
    )

    # Explain the difference between the capacity optimum and the
    # recommended headcount, so the two numbers cannot read as a conflict.
    optimisation = _pick(productivity, "headcount_optimisation", default={})
    optimisation_note = _text(optimisation, "note") if isinstance(optimisation, dict) else ""
    if optimisation_note:
        blocks.append(_block_text(optimisation_note, "Reading the headcount optimum"))

    by_headcount = _rows(productivity, "by_headcount")
    if by_headcount:
        blocks.append(
            _block_table(
                "Performance by headcount",
                [
                    "Headcount",
                    "Avg sales",
                    "Avg sales / employee",
                    "Avg gross margin",
                    "Months observed",
                ],
                [
                    [
                        _fmt_number(row.get("employees"), 0),
                        _fmt_number(row.get("avg_sales")),
                        _fmt_number(row.get("avg_sales_per_employee")),
                        _fmt_number(row.get("avg_gross_margin")),
                        _fmt_number(row.get("months_observed"), 0),
                    ]
                    for row in by_headcount
                ],
                ratios=[1.0, 1.3, 1.6, 1.4, 1.3],
                aligns=["C", "R", "R", "R", "R"],
            )
        )

    marginal = _rows(productivity, "marginal_changes")
    if marginal:
        blocks.append(
            _block_table(
                "Marginal contribution of each additional hire",
                [
                    "From heads",
                    "To heads",
                    "Marginal sales / added",
                    "Change in sales / employee",
                    "Diminishing",
                ],
                [
                    [
                        _fmt_number(row.get("from_employees"), 0),
                        _fmt_number(row.get("to_employees"), 0),
                        _fmt_number(row.get("marginal_sales_per_added")),
                        _fmt_number(row.get("change_in_sales_per_emp")),
                        _fmt_flag(row.get("diminishing")),
                    ]
                    for row in marginal
                ],
                ratios=[1.0, 1.0, 1.6, 1.7, 1.2],
                aligns=["C", "C", "R", "R", "C"],
            )
        )

    curve = _rows(productivity, "headcount_curve")
    if curve:
        blocks.append(
            _block_table(
                "Expected margin by target headcount",
                [
                    "Headcount",
                    f"{horizon}-month margin",
                    "Avg monthly margin",
                    "Margin / employee",
                ],
                [
                    [
                        _fmt_number(row.get("headcount"), 0),
                        _fmt_number(row.get("total_margin")),
                        _fmt_number(row.get("avg_monthly_margin")),
                        _fmt_number(row.get("margin_per_employee")),
                    ]
                    for row in curve
                ],
                ratios=[1.0, 1.4, 1.4, 1.4],
                aligns=["C", "R", "R", "R"],
            )
        )

    marginal_block = productivity.get("marginal_productivity")
    if isinstance(marginal_block, dict) and marginal_block:
        blocks.append(
            _block_kv(
                "Controlled marginal productivity (OLS with trend and season controls)",
                [
                    (
                        "Marginal sales per employee",
                        _fmt_money(marginal_block.get("marginal_sales_per_employee")),
                    ),
                    (
                        "Marginal margin contribution",
                        _fmt_money(marginal_block.get("marginal_margin_contribution")),
                    ),
                    ("p-value", _fmt_ratio(marginal_block.get("p_value"), 4)),
                    ("Model R2", _fmt_ratio(marginal_block.get("r_squared"), 4)),
                    (
                        "Observations",
                        _fmt_number(marginal_block.get("n_observations"), 0),
                    ),
                    ("Model", _text(marginal_block, "model")),
                ],
            )
        )
        note = _text(marginal_block, "note")
        if note != _NA:
            blocks.append(_block_text(note))

    employees = _rows(productivity, "employees")
    if employees:
        blocks.append(
            _block_table(
                "Workforce roster",
                [
                    "Salesperson",
                    "Total sales",
                    "Avg monthly sales",
                    "Months active",
                    "Tenure (months)",
                    "Still active",
                ],
                [
                    [
                        _text(row, "id"),
                        _fmt_number(row.get("total_sales")),
                        _fmt_number(row.get("avg_monthly_sales")),
                        _fmt_number(row.get("months_active"), 0),
                        _fmt_number(row.get("tenure_months"), 0),
                        _fmt_flag(row.get("still_active")),
                    ]
                    for row in employees
                ],
                ratios=[1.6, 1.4, 1.5, 1.1, 1.2, 1.0],
                aligns=["L", "R", "R", "R", "R", "C"],
            )
        )

    attrition = productivity.get("attrition")
    if isinstance(attrition, dict):
        blocks.append(
            _block_kv(
                "Attrition statistics",
                [
                    ("Total leavers", _fmt_number(attrition.get("total_leavers"), 0)),
                    ("Total hires", _fmt_number(attrition.get("total_hires"), 0)),
                    (
                        "Average tenure of leavers (months)",
                        _fmt_number(attrition.get("avg_tenure_months"), 1),
                    ),
                    (
                        "Leavers in last 12 months",
                        _fmt_number(attrition.get("recent_12m_leavers"), 0),
                    ),
                    (
                        "Retention rate",
                        _fmt_probability(attrition.get("retention_rate")),
                    ),
                ],
            )
        )

    return blocks


def _sensitivity_matrix(
    grid: Any,
) -> tuple[list[float], list[float], dict[tuple[float, float], float]]:
    """Convert the sensitivity grid into a salary x commission matrix.

    Parameters
    ----------
    grid : Any
        The list of grid records.

    Returns
    -------
    tuple
        ``(salaries, commissions, cells)`` where *cells* maps
        ``(salary, commission)`` onto the gross margin.
    """
    salaries: list[float] = []
    commissions: list[float] = []
    cells: dict[tuple[float, float], float] = {}
    for row in _rows({"rows": grid}, "rows"):
        salary = _as_number(_pick(row, "salary", "Salary"))
        commission = _as_number(_pick(row, "commission", "CommissionRate"))
        margin = _as_number(_pick(row, "margin", "GrossMargin"))
        if salary is None or commission is None:
            continue
        if salary not in salaries:
            salaries.append(salary)
        if commission not in commissions:
            commissions.append(commission)
        if margin is not None:
            cells[(salary, commission)] = margin
    salaries.sort()
    commissions.sort()
    return salaries, commissions, cells


def _section_risk(results: dict[str, Any]) -> list[dict[str, Any]]:
    """Build the risk and sensitivity blocks.

    Parameters
    ----------
    results : dict
        The analysis payload.

    Returns
    -------
    list of dict
        Renderable blocks.
    """
    sensitivity = results.get("sensitivity")
    sensitivity = sensitivity if isinstance(sensitivity, dict) else {}

    blocks: list[dict[str, Any]] = []

    tornado = sensitivity.get("tornado")
    tornado_rows = _rows(tornado, "variables")
    if not tornado_rows and isinstance(tornado, list):
        tornado_rows = [row for row in tornado if isinstance(row, dict)]
    if tornado_rows:
        blocks.append(
            _block_table(
                "Tornado analysis - one-at-a-time sensitivity",
                ["Variable", "Low", "High", "Low margin", "High margin", "Swing"],
                [
                    [
                        _text(row, "variable"),
                        _fmt_mixed(_pick(row, "low")),
                        _fmt_mixed(_pick(row, "high")),
                        _fmt_money(row.get("low_margin")),
                        _fmt_money(row.get("high_margin")),
                        _fmt_money(row.get("swing")),
                    ]
                    for row in tornado_rows
                ],
                ratios=[1.9, 1.0, 1.0, 1.2, 1.2, 1.1],
                aligns=["L", "R", "R", "R", "R", "R"],
            )
        )

    monte_carlo = sensitivity.get("monte_carlo")
    monte_carlo = monte_carlo if isinstance(monte_carlo, dict) else {}
    if monte_carlo:
        blocks.append(
            _block_kv(
                "Monte-Carlo margin distribution",
                [
                    ("P10 margin", _fmt_money(monte_carlo.get("p10"))),
                    ("P50 margin (median)", _fmt_money(monte_carlo.get("p50"))),
                    ("P90 margin", _fmt_money(monte_carlo.get("p90"))),
                    ("Mean margin", _fmt_money(monte_carlo.get("mean"))),
                    ("Standard deviation", _fmt_money(monte_carlo.get("std"))),
                    (
                        "P(margin > 0)",
                        _fmt_probability(monte_carlo.get("prob_positive")),
                    ),
                    (
                        "Simulations",
                        _fmt_number(monte_carlo.get("n_simulations"), 0),
                    ),
                ],
            )
        )
        stability = monte_carlo.get("decision_stability")
        if isinstance(stability, dict):
            blocks.append(
                _block_kv(
                    "Decision stability",
                    [
                        ("Leading scenario", _text(stability, "leader_label")),
                        (
                            "Margin over runner-up",
                            _fmt_money(stability.get("margin_over_runner_up")),
                        ),
                        ("Decisive", _fmt_flag(stability.get("decisive"))),
                    ],
                )
            )
            note = _text(stability, "note")
            if note != _NA:
                blocks.append(_block_text(note))

    grid = sensitivity.get("grid")
    salaries, commissions, cells = _sensitivity_matrix(grid)
    if salaries and commissions:
        headers = ["Salary \\ Commission"] + [
            _fmt_percent_points(commission * 100) for commission in commissions
        ]
        rows: list[list[str]] = []
        for salary in salaries:
            row = [_fmt_money(salary, 2)]
            for commission in commissions:
                margin = cells.get((salary, commission))
                row.append(_fmt_number(margin) if margin is not None else _NA)
            rows.append(row)
        blocks.append(
            _block_table(
                "Sensitivity grid - gross margin by salary and commission",
                headers,
                rows,
                ratios=[1.6] + [1.0] * len(commissions),
                aligns=["L"] + ["R"] * len(commissions),
            )
        )

    return blocks


def _section_payback(results: dict[str, Any]) -> list[dict[str, Any]]:
    """Build the payback and ROI blocks.

    Parameters
    ----------
    results : dict
        The analysis payload.

    Returns
    -------
    list of dict
        Renderable blocks.
    """
    payback = results.get("payback")
    payback = payback if isinstance(payback, dict) else {}

    blocks: list[dict[str, Any]] = []
    blocks.append(
        _block_kv(
            "Investment case",
            [
                (
                    "Months to payback",
                    _fmt_number(payback.get("months_to_payback"), 2),
                ),
                ("Return on investment", _fmt_percent_points(payback.get("roi_pct"))),
                ("Investment per month", _fmt_money(payback.get("investment"))),
                ("Net benefit", _fmt_money(payback.get("net_benefit"))),
                (
                    "Total contribution over horizon",
                    _fmt_money(payback.get("total_contribution")),
                ),
                ("Ever pays back", _fmt_flag(payback.get("ever_pays_back"))),
            ],
        )
    )

    cumulative = payback.get("cumulative_margin_delta")
    monthly = payback.get("monthly_contribution")
    if isinstance(cumulative, (list, tuple)) and cumulative:
        monthly_values = list(monthly) if isinstance(monthly, (list, tuple)) else []
        rows = []
        for index, value in enumerate(cumulative):
            rows.append(
                [
                    _fmt_number(index + 1, 0),
                    _fmt_number(monthly_values[index] if index < len(monthly_values) else None),
                    _fmt_number(value),
                ]
            )
        blocks.append(
            _block_table(
                "Cumulative contribution of the recommended hire",
                ["Month", "Monthly contribution", "Cumulative contribution"],
                rows,
                ratios=[1.0, 1.5, 1.6],
                aligns=["C", "R", "R"],
            )
        )

    return blocks


def _section_rule(results: dict[str, Any]) -> list[dict[str, Any]]:
    """Build the business-rule blocks.

    Parameters
    ----------
    results : dict
        The analysis payload.

    Returns
    -------
    list of dict
        Renderable blocks.
    """
    optimal_rule = results.get("optimal_rule")
    optimal_rule = optimal_rule if isinstance(optimal_rule, dict) else {}
    backtest = results.get("rule_backtest")
    backtest = backtest if isinstance(backtest, dict) else {}

    blocks: list[dict[str, Any]] = []
    description = _text(optimal_rule, "rule_description")
    blocks.append(_block_text(description, "Recommended business rule"))

    trigger = optimal_rule.get("rolling_avg_trigger")
    if isinstance(trigger, dict):
        hold_band = trigger.get("hold_band")
        if isinstance(hold_band, (list, tuple)) and len(hold_band) == 2:
            band = f"{_fmt_number(hold_band[0])} to {_fmt_number(hold_band[1])}"
        else:
            band = _fmt_mixed(hold_band)
        blocks.append(
            _block_kv(
                "Rolling-average trigger",
                [
                    ("Window (months)", _fmt_number(trigger.get("window_months"), 0)),
                    ("Hire above", _fmt_number(trigger.get("hire_above"))),
                    ("Hold band", band),
                    (
                        "Do not replace below",
                        _fmt_number(trigger.get("do_not_replace_below")),
                    ),
                    (
                        "Current rolling average",
                        _fmt_number(trigger.get("current_rolling_avg")),
                    ),
                ],
            )
        )

    threshold = optimal_rule.get("threshold_rule")
    if isinstance(threshold, dict):
        actions = threshold.get("actions")
        if isinstance(actions, dict) and actions:
            blocks.append(
                _block_table(
                    "Threshold-based decision rules",
                    ["Condition", "Action"],
                    [[str(key), str(value)] for key, value in actions.items()],
                    ratios=[1.6, 1.4],
                    aligns=["L", "L"],
                )
            )

    comparison = optimal_rule.get("comparison_with_always_replace")
    if isinstance(comparison, dict) and comparison:
        blocks.append(
            _block_bullets(
                [str(value) for value in comparison.values()],
                "Why this beats 'always replace'",
            )
        )

    interpretation = _text(backtest, "interpretation")
    if interpretation != _NA:
        blocks.append(_block_text(interpretation, "Backtest interpretation"))

    always = backtest.get("always_replace")
    adaptive = backtest.get("adaptive")
    if isinstance(always, dict) and isinstance(adaptive, dict):
        blocks.append(
            _block_table(
                "Backtest - adaptive rule vs always replace",
                ["Policy", "Total margin", "Avg monthly margin"],
                [
                    [
                        "Always replace",
                        _fmt_money(always.get("total_margin")),
                        _fmt_money(always.get("avg_monthly_margin")),
                    ],
                    [
                        "Adaptive rule",
                        _fmt_money(adaptive.get("total_margin")),
                        _fmt_money(adaptive.get("avg_monthly_margin")),
                    ],
                    [
                        "Improvement",
                        _fmt_money(backtest.get("improvement")),
                        _fmt_percent_points(backtest.get("improvement_pct")),
                    ],
                ],
                ratios=[1.6, 1.3, 1.4],
                aligns=["L", "R", "R"],
            )
        )
        blocks.append(
            _block_kv(
                "Backtest coverage",
                [
                    (
                        "Months compared",
                        _fmt_number(backtest.get("months_compared"), 0),
                    ),
                    (
                        "Months of overstaffing avoided",
                        _fmt_number(backtest.get("avoided_overstaffing_months"), 0),
                    ),
                    (
                        "Share of overstaffing avoided",
                        _fmt_percent_points(backtest.get("avoided_overstaffing_pct")),
                    ),
                ],
            )
        )

    return blocks


def _business_assumptions(
    salary: Any,
    commission: Any,
    cogs_rate: Any = None,
    ramp: Any = None,
) -> list[str]:
    """Return the list of documented business assumptions.

    Every value that can be configured by the caller is read from the
    payload rather than hard-coded, so the text stays truthful when the
    assumptions change.

    Parameters
    ----------
    salary : Any
        Monthly salary per employee (Lakhs INR).
    commission : Any
        Commission rate as a fraction of sales.
    cogs_rate : Any
        Cost-of-goods rate as a fraction of sales.
    ramp : Any
        New-hire productivity ramp factors.

    Returns
    -------
    list of str
        The assumptions.
    """
    salary_text = _fmt_number(salary, 2)
    commission_text = _fmt_percent_points(
        commission * 100 if _as_number(commission) is not None else None
    )
    cogs_value = _as_number(cogs_rate)
    cogs_text = f"{cogs_value * 100:.0f}% of sales" if cogs_value is not None else "50% of sales"

    ramp_text = "the modelled ramp-up curve"
    if isinstance(ramp, (list, tuple)) and len(ramp) >= 2:
        first = _as_number(ramp[0])
        last = _as_number(ramp[-1])
        if first is not None and last is not None:
            ramp_text = f"{first * 100:.0f}% -> {last * 100:.0f}% over {len(ramp)} months"

    return [
        "Customer traffic and footfall remain stable.",
        "No significant hiring or recruitment costs are incurred.",
        f"No training-period productivity loss beyond the modelled ramp-up " f"({ramp_text}).",
        f"Monthly salary stays fixed at {_RUPEE} {salary_text} Lakhs per employee.",
        f"Commission remains fixed at {commission_text} of total sales.",
        f"Cost of goods sold stays at {cogs_text}.",
        "Economic conditions and consumer demand remain similar.",
        "No major competitor entry or exit during the projection period.",
        "Store operating hours remain unchanged.",
        "The pricing and assortment strategy remains consistent.",
        "Sales are treated as exogenous: the analysis prices staffing "
        "decisions against realised demand.",
    ]


# --------------------------------------------------------------------------- #
# Chart helpers                                                                #
# --------------------------------------------------------------------------- #
def _image_size(path: str) -> tuple[int, int] | None:
    """Return the pixel dimensions of the image stored at *path*.

    Only the file header is inspected, so the routine is cheap and does
    not require an imaging library.

    Parameters
    ----------
    path : str
        The image path.

    Returns
    -------
    tuple of (int, int) or None
        ``(width, height)`` in pixels, or ``None`` when the format is not
        recognised.
    """
    try:
        with open(path, "rb") as handle:
            header = handle.read(26)
            if header[:8] == b"\x89PNG\r\n\x1a\n" and header[12:16] == b"IHDR":
                return struct.unpack(">II", header[16:24])
            if header[:2] == b"\xff\xd8":
                return _jpeg_size(handle)
    except (OSError, struct.error, ValueError):
        return None
    return None


def _jpeg_size(handle: Any) -> tuple[int, int] | None:
    """Return the pixel dimensions of the JPEG stream in *handle*.

    Parameters
    ----------
    handle : file object
        A binary file positioned at the start of the JPEG data.

    Returns
    -------
    tuple of (int, int) or None
        ``(width, height)`` in pixels, or ``None`` when the stream cannot
        be parsed.
    """
    handle.seek(2)
    while True:
        marker = handle.read(2)
        if len(marker) < 2 or marker[0] != 0xFF:
            return None
        code = marker[1]
        if code in (0xD8, 0xD9) or 0xD0 <= code <= 0xD7:
            continue
        length_bytes = handle.read(2)
        if len(length_bytes) < 2:
            return None
        length = struct.unpack(">H", length_bytes)[0]
        if code in (
            0xC0,
            0xC1,
            0xC2,
            0xC3,
            0xC5,
            0xC6,
            0xC7,
            0xC9,
            0xCA,
            0xCB,
            0xCD,
            0xCE,
            0xCF,
        ):
            payload = handle.read(5)
            if len(payload) < 5:
                return None
            height, width = struct.unpack(">HH", payload[1:5])
            return int(width), int(height)
        handle.seek(length - 2, os.SEEK_CUR)


def _discover_charts(charts_dir: str, prefix: str = "") -> list[str]:
    """Return the chart stems found in *charts_dir*.

    Parameters
    ----------
    charts_dir : str
        Directory holding the generated PNG charts.
    prefix : str, optional
        Only return stems starting with this prefix.

    Returns
    -------
    list of str
        The sorted chart stems.
    """
    if not charts_dir or not os.path.isdir(charts_dir):
        return []
    stems: list[str] = []
    try:
        entries = os.listdir(charts_dir)
    except OSError:
        return []
    for entry in entries:
        if not entry.lower().endswith(".png"):
            continue
        stem = Path(entry).stem
        if prefix and not stem.startswith(prefix):
            continue
        stems.append(stem)
    return sorted(stems)


def _chart_path(charts_dir: str, stem: str) -> str | None:
    """Return the absolute path of the chart named *stem*.

    Parameters
    ----------
    charts_dir : str
        Directory holding the generated charts.
    stem : str
        The chart stem (without extension).

    Returns
    -------
    str or None
        The absolute path of the chart, or ``None`` when it does not
        exist.
    """
    if not charts_dir or not stem:
        return None
    candidate = os.path.join(charts_dir, f"{stem}.png")
    if not os.path.isfile(candidate):
        return None
    return os.path.abspath(candidate)


def _optimise_images(paths: Sequence[str]) -> dict[str, str]:
    """Return a mapping of source path to the path that should be embedded.

    When the combined size of the chart images is large the images are
    re-encoded as JPEG so that the resulting PDF stays comfortably below
    ~15 MB.  Pillow ships with ``fpdf2``; when it is unavailable the
    original files are used unchanged.

    Parameters
    ----------
    paths : sequence of str
        The chart paths that will be embedded.

    Returns
    -------
    dict
        Mapping of original path to the path to embed.
    """
    mapping = {path: path for path in paths}
    total = 0
    for path in paths:
        try:
            total += os.path.getsize(path)
        except OSError:
            continue
    if total <= _IMAGE_BUDGET_BYTES:
        return mapping
    try:
        from PIL import Image  # type: ignore[import-not-found]
    except Exception:  # noqa: BLE001 - Pillow is optional here
        logger.warning(
            "Chart images total %.1f MB but Pillow is unavailable; "
            "embedding the original PNG files.",
            total / 1_048_576.0,
        )
        return mapping
    temp_dir = tempfile.mkdtemp(prefix="revinsight_charts_")
    for path in paths:
        try:
            with Image.open(path) as image:
                converted = image.convert("RGB")
                target = os.path.join(temp_dir, f"{Path(path).stem}.jpg")
                converted.save(target, "JPEG", quality=80, optimize=True)
            mapping[path] = target
        except Exception as exc:  # noqa: BLE001 - never abort a report
            logger.warning("Could not re-encode %s as JPEG: %s", path, exc)
    logger.info("Re-encoded chart images as JPEG to keep the PDF small.")
    return mapping


# --------------------------------------------------------------------------- #
# View model                                                                   #
# --------------------------------------------------------------------------- #
def _build_view_model(
    results: dict[str, Any],
    summary_df: pd.DataFrame,
    charts_dir: str,
    salary: float,
    commission: float,
) -> dict[str, Any]:
    """Build the renderer-independent view model of the report.

    Parameters
    ----------
    results : dict
        The v2.0 analysis payload.
    summary_df : pd.DataFrame
        The monthly summary produced by the pipeline.
    charts_dir : str
        Directory holding the generated charts.
    salary : float
        Monthly salary per employee (Lakhs INR).
    commission : float
        Commission rate as a fraction of sales.

    Returns
    -------
    dict
        The view model consumed by the PDF and HTML renderers.
    """
    payload = results if isinstance(results, dict) else {}
    dataset = payload.get("dataset")
    dataset = dataset if isinstance(dataset, dict) else {}
    executive = payload.get("executive")
    executive = executive if isinstance(executive, dict) else {}
    assumptions = payload.get("assumptions")
    assumptions = assumptions if isinstance(assumptions, dict) else {}

    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    created_at = _text(payload, "created_at", default=generated)

    declared = payload.get("charts")
    declared_stems: list[str] = []
    if isinstance(declared, dict):
        declared_stems = [str(key) for key in declared.keys()]
    elif isinstance(declared, (list, tuple)):
        declared_stems = [str(item) for item in declared]

    forecast_charts = _discover_charts(charts_dir, prefix="forecast_")
    all_stems: list[str] = []
    for stem in declared_stems + forecast_charts:
        if stem not in all_stems:
            all_stems.append(stem)

    used: list[str] = []

    def chart(stem: str, caption: str = "") -> dict[str, Any]:
        """Register a chart block and remember the stem as consumed."""
        used.append(stem)
        return _block_chart(stem, caption)

    sections: list[dict[str, Any]] = []

    # -- 1. Executive summary ------------------------------------------ #
    sections.append(
        {
            "number": 1,
            "title": "Executive Summary",
            "blocks": _section_executive(payload),
        }
    )

    # -- 2. Financial highlights ---------------------------------------- #
    financial_blocks = _section_financial(payload, summary_df)
    financial_blocks.append(chart("monthly_sales_trend", "Monthly sales with a rolling average"))
    financial_blocks.append(
        chart("gross_margin_trend", "Monthly gross margin with rolling average")
    )
    financial_blocks.append(chart("anomalies", "Anomaly detection on monthly sales"))
    sections.append(
        {
            "number": 2,
            "title": "Financial Highlights",
            "blocks": financial_blocks,
        }
    )

    # -- 3. Forecast outlook -------------------------------------------- #
    forecast_blocks = _section_forecast(payload)
    for stem in forecast_charts:
        forecast_blocks.append(chart(stem, f"Forecast - {stem.replace('_', ' ')}"))
    forecast_blocks.append(
        chart(
            "decomposition",
            "Time-series decomposition (trend / seasonal / residual)",
        )
    )
    sections.append(
        {
            "number": 3,
            "title": "Forecast Outlook",
            "blocks": forecast_blocks,
        }
    )

    # -- 4. Staffing recommendation -------------------------------------- #
    sections.append(
        {
            "number": 4,
            "title": "Staffing Recommendation",
            "blocks": _section_staffing(payload),
        }
    )

    # -- 5. Scenario comparison ----------------------------------------- #
    scenario_blocks = _section_scenarios(payload)
    scenario_blocks.append(
        chart(
            "scenario_comparison",
            f"Total {_horizon_from(payload)}-month gross margin by scenario",
        )
    )
    scenario_blocks.append(chart("scenario_cumulative", "Cumulative gross margin path by scenario"))
    sections.append(
        {
            "number": 5,
            "title": "Scenario Comparison",
            "blocks": scenario_blocks,
        }
    )

    # -- 6. Productivity & diminishing returns --------------------------- #
    productivity_blocks = _section_productivity(payload)
    productivity_blocks.append(chart("avg_sales_per_employee", "Average sales per active employee"))
    productivity_blocks.append(chart("workforce_tenure", "Salesperson tenure"))
    productivity_blocks.append(
        chart("headcount_optimization", "Expected margin vs target headcount")
    )
    sections.append(
        {
            "number": 6,
            "title": "Productivity & Diminishing Returns",
            "blocks": productivity_blocks,
        }
    )

    # -- 7. Risk & sensitivity ------------------------------------------ #
    risk_blocks = _section_risk(payload)
    risk_blocks.append(chart("tornado", "Margin sensitivity (tornado)"))
    sections.append(
        {
            "number": 7,
            "title": "Risk & Sensitivity",
            "blocks": risk_blocks,
        }
    )

    # -- 8. Payback & ROI ----------------------------------------------- #
    sections.append(
        {
            "number": 8,
            "title": "Payback & ROI",
            "blocks": _section_payback(payload),
        }
    )

    # -- 9. Business rule ----------------------------------------------- #
    sections.append(
        {
            "number": 9,
            "title": "Business Rule",
            "blocks": _section_rule(payload),
        }
    )

    # -- 10. Appendix ---------------------------------------------------- #
    appendix_blocks: list[dict[str, Any]] = []
    narrative = _text(payload, "recommendation_text", default="")
    if narrative and narrative != _NA:
        appendix_blocks.append(_block_pre(narrative, "Full narrative report (plain text)"))
    appendix_blocks.append(
        _block_bullets(
            _business_assumptions(
                _pick(assumptions, "salary", default=salary),
                _pick(assumptions, "commission", default=commission),
                _pick(assumptions, "cogs_rate"),
                _pick(assumptions, "new_hire_ramp"),
            ),
            "Business assumptions",
        )
    )
    for stem in all_stems:
        if stem in used:
            continue
        appendix_blocks.append(chart(stem, stem.replace("_", " ").capitalize()))
    sections.append(
        {
            "number": 10,
            "title": "Appendix",
            "blocks": appendix_blocks,
        }
    )

    meta = {
        "product": _PRODUCT,
        "subtitle": _SUBTITLE,
        "dataset": _text(dataset, "name", default=_NA),
        "filename": _text(dataset, "filename", default=_NA),
        "rows": _fmt_number(dataset.get("rows"), 0),
        "columns": _fmt_number(dataset.get("columns"), 0),
        "months": _fmt_number(dataset.get("months"), 0),
        "salespersons": _fmt_number(dataset.get("salespersons"), 0),
        "total_sales": _fmt_money(dataset.get("total_sales")),
        "generated": generated,
        "created_at": created_at,
        "run_id": _text(payload, "run_id", default=_NA),
        "schema_version": _text(payload, "schema_version", default=_NA),
        "headline": _text(executive, "headline", default=_NA),
        "recommendation": _text(executive, "recommendation", default=_NA),
        "confidence": _text(executive, "confidence", default=_NA),
        "salary": _fmt_number(_pick(assumptions, "salary", default=salary), 2),
        "commission": _fmt_percent_points(
            (_pick(assumptions, "commission", default=commission) or 0) * 100
        ),
    }

    return {"meta": meta, "sections": sections, "charts_dir": charts_dir}


# --------------------------------------------------------------------------- #
# PDF rendering                                                                #
# --------------------------------------------------------------------------- #
class _ReportPDF(FPDF):
    """FPDF subclass carrying the RevInsight branding."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialise the document and the page geometry.

        Parameters
        ----------
        *args : tuple
            Positional arguments forwarded to :class:`fpdf.FPDF`.
        **kwargs : dict
            Keyword arguments forwarded to :class:`fpdf.FPDF`.
        """
        super().__init__(*args, **kwargs)
        self.set_margins(_LEFT_MARGIN, _TOP_MARGIN, _RIGHT_MARGIN)
        self.set_auto_page_break(auto=True, margin=_BOTTOM_MARGIN)
        self.meta: dict[str, str] = {}
        self.generated: str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def header(self) -> None:
        """Draw the running header shown on every page but the cover."""
        if self.page_no() <= 1:
            return
        half = (self.w - self.l_margin - self.r_margin) / 2.0
        self.set_font(_FONT, "B", 8)
        self.set_text_color(*_COLOR_PRIMARY)
        self.set_xy(self.l_margin, 9)
        self.cell(half, 6, _sanitize(f"{_PRODUCT} - {_SUBTITLE}"), align="L")
        dataset = self.meta.get("dataset", "")
        right = f"Dataset: {dataset}" if dataset else ""
        self.set_font(_FONT, "", 8)
        self.set_text_color(*_COLOR_MUTED)
        self.cell(
            half,
            6,
            _sanitize(right),
            align="R",
            new_x=XPos.LMARGIN,
            new_y=YPos.NEXT,
        )
        self.set_draw_color(*_COLOR_GRID)
        self.set_line_width(0.3)
        rule_y = self.get_y()
        self.line(self.l_margin, rule_y, self.w - self.r_margin, rule_y)
        self.set_y(max(self.t_margin, rule_y + 1.6))

    def footer(self) -> None:
        """Draw the document footer with the page number."""
        self.set_y(-14)
        self.set_draw_color(*_COLOR_GRID)
        self.set_line_width(0.3)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.set_y(self.get_y() + 1.4)
        self.set_font(_FONT, "", 7)
        self.set_text_color(*_COLOR_MUTED)
        text = (
            f"{_PRODUCT} - {_SUBTITLE}   |   Generated {self.generated}   |   "
            "Page {self.page_no()}/{nb}"
        )
        self.set_x(0)
        self.cell(
            self.w,
            6,
            _sanitize(text).replace("{self.page_no()}", str(self.page_no())),
            align="C",
        )


def _pdf_write_line(
    pdf: _ReportPDF,
    text: str,
    size: float = _FONT_SIZE_BODY,
    style: str = "",
    color: tuple[int, int, int] = _COLOR_INK,
    align: str = "L",
    height: float | None = None,
    x: float | None = None,
    width: float | None = None,
) -> float:
    """Write a single line of text and return the new cursor position.

    Parameters
    ----------
    pdf : _ReportPDF
        The document being rendered.
    text : str
        The text to write.
    size : float, optional
        Font size in points.
    style : str, optional
        Font style (``""``, ``"B"``, ``"I"``).
    color : tuple of int, optional
        Text colour.
    align : str, optional
        Horizontal alignment.
    height : float, optional
        Line height in millimetres.
    x : float, optional
        Left position; defaults to the left margin.
    width : float, optional
        Cell width; defaults to the content width.

    Returns
    -------
    float
        The cursor position after the line.
    """
    left = pdf.l_margin if x is None else x
    span = _CONTENT_WIDTH if width is None else width
    line_height = size * _MM_PER_POINT * 1.45 if height is None else height
    pdf.set_font(_FONT, style, size)
    pdf.set_text_color(*color)
    pdf.set_xy(left, pdf.get_y())
    pdf.cell(
        span,
        line_height,
        _sanitize(text),
        align=align,
        new_x=XPos.LEFT,
        new_y=YPos.NEXT,
    )
    return pdf.get_y()


def _pdf_write_paragraph(
    pdf: _ReportPDF,
    text: str,
    size: float = _FONT_SIZE_BODY,
    style: str = "",
    color: tuple[int, int, int] = _COLOR_INK,
    x: float | None = None,
    width: float | None = None,
) -> float:
    """Write a wrapped paragraph and return the new cursor position.

    Parameters
    ----------
    pdf : _ReportPDF
        The document being rendered.
    text : str
        The paragraph content.
    size : float, optional
        Font size in points.
    style : str, optional
        Font style.
    color : tuple of int, optional
        Text colour.
    x : float, optional
        Left position; defaults to the left margin.
    width : float, optional
        Cell width; defaults to the content width.

    Returns
    -------
    float
        The cursor position after the paragraph.
    """
    left = pdf.l_margin if x is None else x
    span = _CONTENT_WIDTH if width is None else width
    line_height = size * _MM_PER_POINT * _LINE_FACTOR
    pdf.set_font(_FONT, style, size)
    pdf.set_text_color(*color)
    pdf.set_xy(left, pdf.get_y())
    pdf.multi_cell(
        span,
        line_height,
        _sanitize(text),
        align="L",
        new_x=XPos.LEFT,
        new_y=YPos.NEXT,
    )
    return pdf.get_y()


def _pdf_section_heading(pdf: _ReportPDF, number: Any, title: str) -> None:
    """Render a numbered section heading with an accent rule.

    Parameters
    ----------
    pdf : _ReportPDF
        The document being rendered.
    number : Any
        The section number.
    title : str
        The section title.
    """
    pdf.set_y(pdf.get_y() + 4.0)
    label = f"{number}. {title}" if str(number).strip() else str(title)
    _pdf_write_line(
        pdf,
        label,
        size=_FONT_SIZE_HEADING,
        style="B",
        color=_COLOR_PRIMARY,
        height=8.5,
    )
    pdf.set_draw_color(*_COLOR_ACCENT)
    pdf.set_line_width(0.7)
    rule_y = pdf.get_y()
    pdf.line(pdf.l_margin, rule_y, pdf.w - pdf.r_margin, rule_y)
    pdf.set_y(rule_y + 2.6)


def _pdf_block_title(pdf: _ReportPDF, title: str) -> None:
    """Render a small caption above a table or list.

    Parameters
    ----------
    pdf : _ReportPDF
        The document being rendered.
    title : str
        The caption text.
    """
    if not title:
        return
    pdf.set_y(pdf.get_y() + 1.6)
    _pdf_write_line(
        pdf,
        title,
        size=_FONT_SIZE_SUBHEADING,
        style="B",
        color=_COLOR_PRIMARY,
        height=5.4,
    )
    pdf.set_y(pdf.get_y() + 0.6)


def _pdf_badge(
    pdf: _ReportPDF,
    text: str,
    level: str,
) -> None:
    """Render a coloured confidence badge.

    Parameters
    ----------
    pdf : _ReportPDF
        The document being rendered.
    text : str
        The badge label.
    level : str
        The confidence level used to pick the colour.
    """
    normalised = str(level).upper()
    if "HIGH" in normalised:
        colour = _COLOR_BAND_HIGH
    elif "LOW" in normalised:
        colour = _COLOR_BAND_LOW
    else:
        colour = _COLOR_BAND_MEDIUM
    label = f"CONFIDENCE: {text}" if text else "CONFIDENCE"
    width = 46.0
    height = 8.0
    pdf.set_y(pdf.get_y() + 1.5)
    x = pdf.l_margin
    y = pdf.get_y()
    pdf.set_fill_color(*colour)
    pdf.rect(x, y, width, height, style="F", round_corners=True, corner_radius=1.6)
    pdf.set_font(_FONT, "B", 9)
    pdf.set_text_color(255, 255, 255)
    pdf.set_xy(x, y + 1.6)
    pdf.cell(width, 5, _sanitize(label), align="C", new_x=XPos.LEFT, new_y=YPos.TOP)
    pdf.set_y(y + height + 2.4)


def _pdf_bullets(pdf: _ReportPDF, items: Sequence[str]) -> None:
    """Render a bulleted list.

    Parameters
    ----------
    pdf : _ReportPDF
        The document being rendered.
    items : sequence of str
        The bullet points.
    """
    line_height = _FONT_SIZE_BODY * _MM_PER_POINT * _LINE_FACTOR
    indent = 4.0
    width = _CONTENT_WIDTH - indent
    for item in items:
        y = pdf.get_y()
        pdf.set_font(_FONT, "B", _FONT_SIZE_BODY)
        pdf.set_text_color(*_COLOR_ACCENT)
        pdf.set_xy(pdf.l_margin, y)
        pdf.cell(
            indent,
            line_height,
            "-",
            align="L",
            new_x=XPos.LEFT,
            new_y=YPos.TOP,
        )
        pdf.set_font(_FONT, "", _FONT_SIZE_BODY)
        pdf.set_text_color(*_COLOR_INK)
        pdf.set_xy(pdf.l_margin + indent, y)
        pdf.multi_cell(
            width,
            line_height,
            _sanitize(item),
            align="L",
            new_x=XPos.LEFT,
            new_y=YPos.NEXT,
        )
        pdf.set_y(pdf.get_y() + 0.8)


def _pdf_measure_lines(
    pdf: _ReportPDF,
    width: float,
    text: str,
    line_height: float,
) -> int:
    """Return the number of lines *text* needs inside a cell.

    The measurement is performed at a safe cursor position so that the
    dry run can never trigger a page break.

    Parameters
    ----------
    pdf : _ReportPDF
        The document being rendered.
    width : float
        The available width in millimetres.
    text : str
        The text to measure.
    line_height : float
        The line height in millimetres.

    Returns
    -------
    int
        The number of wrapped lines (at least one).
    """
    saved_x, saved_y = pdf.get_x(), pdf.get_y()
    pdf.set_xy(pdf.l_margin, max(pdf.t_margin, 1.0))
    try:
        probe = pdf.multi_cell(
            width,
            line_height,
            text,
            dry_run=True,
            output="LINES",
        )
    except Exception as exc:  # noqa: BLE001 - measurement must never fail
        logger.debug("Could not measure cell text: %s", exc)
        probe = None
    finally:
        pdf.set_xy(saved_x, saved_y)
    if isinstance(probe, list):
        return max(1, len(probe))
    return 1


def _pdf_table(
    pdf: _ReportPDF,
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    ratios: Sequence[float] | None = None,
    aligns: Sequence[str] | None = None,
    font_size: float = _FONT_SIZE_TABLE,
) -> None:
    """Render a bordered table with a repeating header row.

    Parameters
    ----------
    pdf : _ReportPDF
        The document being rendered.
    headers : sequence of str
        The column headers.
    rows : sequence of sequence of str
        The table body.
    ratios : sequence of float, optional
        Relative column widths.
    aligns : sequence of str, optional
        Per-column alignment (``"L"``, ``"C"`` or ``"R"``).
    font_size : float, optional
        Font size in points.
    """
    columns = max(len(headers), 1)
    if ratios and len(ratios) >= columns:
        weights = [float(value) for value in ratios[:columns]]
    else:
        weights = [1.0] * columns
    total_weight = sum(weights) or 1.0
    widths = [_CONTENT_WIDTH * weight / total_weight for weight in weights]
    column_aligns = list(aligns) if aligns and len(aligns) >= columns else []
    while len(column_aligns) < columns:
        column_aligns.append("L")

    line_height = font_size * _MM_PER_POINT * _LINE_FACTOR
    break_y = pdf.h - pdf.b_margin

    def draw_header() -> None:
        """Draw the shaded header row."""
        pdf.set_fill_color(*_COLOR_HEADER_BG)
        pdf.set_draw_color(*_COLOR_GRID)
        pdf.set_line_width(0.25)
        pdf.set_font(_FONT, "B", font_size)
        pdf.set_text_color(255, 255, 255)
        x = pdf.l_margin
        y = pdf.get_y()
        height = line_height + 2 * _CELL_PAD
        for index in range(columns):
            label = headers[index] if index < len(headers) else ""
            pdf.set_xy(x, y)
            pdf.multi_cell(
                widths[index],
                line_height,
                _sanitize(label),
                border=1,
                align="C",
                fill=True,
                new_x=XPos.LEFT,
                new_y=YPos.TOP,
            )
            x += widths[index]
        pdf.set_y(y + height)

    draw_header()

    for row_index, row in enumerate(rows):
        cells = list(row)[:columns]
        cells += [""] * (columns - len(cells))
        pdf.set_font(_FONT, "", font_size)
        line_counts = [
            _pdf_measure_lines(pdf, widths[index], _sanitize(cells[index]), line_height)
            for index in range(columns)
        ]
        row_height = max(
            _ROW_MIN_HEIGHT,
            max(line_counts) * line_height + 2 * _CELL_PAD,
        )
        if pdf.get_y() + row_height > break_y:
            pdf.add_page()
            draw_header()
        pdf.set_draw_color(*_COLOR_GRID)
        pdf.set_line_width(0.25)
        if row_index % 2 == 1:
            pdf.set_fill_color(*_COLOR_ZEBRA)
            fill = True
        else:
            fill = False
        pdf.set_font(_FONT, "", font_size)
        pdf.set_text_color(*_COLOR_INK)
        x = pdf.l_margin
        y = pdf.get_y()
        for index in range(columns):
            cell_height = row_height / line_counts[index]
            pdf.set_xy(x, y)
            pdf.multi_cell(
                widths[index],
                cell_height,
                _sanitize(cells[index]),
                border=1,
                align=column_aligns[index],
                fill=fill,
                new_x=XPos.LEFT,
                new_y=YPos.TOP,
            )
            x += widths[index]
        pdf.set_y(y + row_height)
    pdf.set_y(pdf.get_y() + 1.8)


def _pdf_chart(
    pdf: _ReportPDF,
    path: str | None,
    caption: str,
    max_width: float = _CHART_MAX_WIDTH,
) -> None:
    """Embed a chart image, skipping silently when the file is absent.

    Parameters
    ----------
    pdf : _ReportPDF
        The document being rendered.
    path : str or None
        The absolute path of the image.
    caption : str
        The caption rendered underneath the image.
    max_width : float, optional
        Maximum width in millimetres.
    """
    if not path or not os.path.isfile(path):
        logger.debug("Chart skipped (not found): %s", path)
        return
    size = _image_size(path)
    if size and size[0] > 0:
        image_height = max_width * size[1] / size[0]
    else:
        image_height = max_width * 0.58
    if image_height > _CHART_MAX_HEIGHT:
        scale = _CHART_MAX_HEIGHT / image_height
        image_height *= scale
        max_width *= scale
    caption_height = 6.0 if caption else 0.0
    if pdf.get_y() + image_height + caption_height + 4.0 > pdf.h - pdf.b_margin:
        pdf.add_page()
    x = (pdf.w - max_width) / 2.0
    y = pdf.get_y()
    try:
        pdf.image(path, x=x, y=y, w=max_width, h=image_height)
    except Exception as exc:  # noqa: BLE001 - a chart must never abort a report
        logger.warning("Could not embed chart %s: %s", path, exc)
        return
    pdf.set_y(y + image_height + 1.4)
    if caption:
        _pdf_write_line(
            pdf,
            caption,
            size=_FONT_SIZE_SMALL,
            style="I",
            color=_COLOR_MUTED,
            align="C",
        )
    pdf.set_y(pdf.get_y() + 1.5)


def _pdf_blocks(
    pdf: _ReportPDF,
    blocks: Sequence[dict[str, Any]],
    image_map: dict[str, str],
) -> None:
    """Render a list of view-model blocks into the PDF.

    Parameters
    ----------
    pdf : _ReportPDF
        The document being rendered.
    blocks : sequence of dict
        The blocks to render.
    image_map : dict
        Mapping of chart path to the path that should be embedded.
    """
    for block in blocks:
        kind = block.get("type")
        title = str(block.get("title", ""))
        if kind == "text":
            if title:
                _pdf_block_title(pdf, title)
            _pdf_write_paragraph(pdf, str(block.get("text", "")))
            pdf.set_y(pdf.get_y() + 1.2)
        elif kind == "bullets":
            if title:
                _pdf_block_title(pdf, title)
            _pdf_bullets(pdf, [str(item) for item in block.get("items", [])])
            pdf.set_y(pdf.get_y() + 1.0)
        elif kind == "kv":
            if title:
                _pdf_block_title(pdf, title)
            _pdf_table(
                pdf,
                ["Metric", "Value"],
                block.get("rows", []),
                ratios=[1.35, 1.0],
                aligns=["L", "R"],
            )
        elif kind == "table":
            if title:
                _pdf_block_title(pdf, title)
            _pdf_table(
                pdf,
                block.get("headers", []),
                block.get("rows", []),
                ratios=block.get("ratios"),
                aligns=block.get("aligns"),
            )
        elif kind == "chart":
            stem = str(block.get("stem", ""))
            path = image_map.get(stem)
            _pdf_chart(pdf, path, str(block.get("caption", "")))
        elif kind == "badge":
            _pdf_badge(pdf, str(block.get("text", "")), str(block.get("level", "")))
        elif kind == "pre":
            if title:
                _pdf_block_title(pdf, title)
            pdf.set_font(_FONT_MONO, "", _FONT_SIZE_MONO)
            pdf.set_text_color(*_COLOR_INK)
            pdf.set_xy(pdf.l_margin, pdf.get_y())
            pdf.multi_cell(
                _CONTENT_WIDTH,
                _FONT_SIZE_MONO * _MM_PER_POINT * _LINE_FACTOR,
                _sanitize(str(block.get("text", ""))),
                align="L",
                new_x=XPos.LEFT,
                new_y=YPos.NEXT,
            )
            pdf.set_y(pdf.get_y() + 1.5)


def _pdf_cover(pdf: _ReportPDF, meta: dict[str, Any]) -> None:
    """Render the cover page.

    Parameters
    ----------
    pdf : _ReportPDF
        The document being rendered.
    meta : dict
        The report metadata.
    """
    pdf.add_page()
    pdf.set_fill_color(*_COLOR_PRIMARY)
    pdf.rect(0, 0, pdf.w, 74, style="F")
    pdf.set_fill_color(*_COLOR_ACCENT)
    pdf.rect(0, 74, pdf.w, 2.2, style="F")

    pdf.set_xy(pdf.l_margin, 22)
    pdf.set_font(_FONT, "B", _FONT_SIZE_TITLE)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 13, _sanitize(_PRODUCT), align="L", new_x=XPos.LEFT, new_y=YPos.NEXT)
    pdf.set_font(_FONT, "", _FONT_SIZE_SUBTITLE)
    pdf.set_text_color(206, 240, 232)
    pdf.cell(
        0,
        8,
        _sanitize(_SUBTITLE),
        align="L",
        new_x=XPos.LEFT,
        new_y=YPos.NEXT,
    )
    pdf.set_font(_FONT, "I", 9)
    pdf.set_text_color(168, 214, 205)
    pdf.cell(
        0,
        6,
        _sanitize("Data-driven staffing optimisation for a retail store"),
        align="L",
        new_x=XPos.LEFT,
        new_y=YPos.NEXT,
    )

    pdf.set_y(92)
    _pdf_write_line(
        pdf,
        "EXECUTIVE HEADLINE",
        size=9,
        style="B",
        color=_COLOR_ACCENT,
        height=5,
    )
    pdf.set_y(pdf.get_y() + 1.0)
    headline = str(meta.get("headline", _NA))
    pdf.set_fill_color(*_COLOR_ZEBRA)
    pdf.set_draw_color(*_COLOR_ACCENT)
    pdf.set_line_width(0.6)
    box_top = pdf.get_y()
    line_count = _pdf_measure_lines(
        pdf,
        _CONTENT_WIDTH - 6,
        _sanitize(headline),
        _FONT_SIZE_HEADING * _MM_PER_POINT * _LINE_FACTOR,
    )
    box_height = line_count * _FONT_SIZE_HEADING * _MM_PER_POINT * _LINE_FACTOR + 5.0
    pdf.set_fill_color(*_COLOR_ZEBRA)
    pdf.rect(
        pdf.l_margin,
        box_top,
        _CONTENT_WIDTH,
        box_height,
        style="DF",
        round_corners=True,
        corner_radius=1.8,
    )
    pdf.set_font(_FONT, "B", _FONT_SIZE_HEADING)
    pdf.set_text_color(*_COLOR_PRIMARY)
    pdf.set_xy(pdf.l_margin + 3, box_top + 2.4)
    pdf.multi_cell(
        _CONTENT_WIDTH - 6,
        _FONT_SIZE_HEADING * _MM_PER_POINT * _LINE_FACTOR,
        _sanitize(headline),
        align="L",
        new_x=XPos.LEFT,
        new_y=YPos.NEXT,
    )
    pdf.set_y(box_top + box_height + 6.0)

    details = [
        ("Dataset", str(meta.get("dataset", _NA))),
        ("File", str(meta.get("filename", _NA))),
        ("Rows / columns", f"{meta.get('rows', _NA)} / {meta.get('columns', _NA)}"),
        ("Months of history", str(meta.get("months", _NA))),
        ("Salespersons", str(meta.get("salespersons", _NA))),
        ("Total sales", str(meta.get("total_sales", _NA))),
        ("Generated", str(meta.get("generated", _NA))),
        ("Run created at", str(meta.get("created_at", _NA))),
        ("Run id", str(meta.get("run_id", _NA))),
        ("Schema version", str(meta.get("schema_version", _NA))),
        (
            "Assumptions",
            f"Salary {meta.get('salary', _NA)} L, " f"commission {meta.get('commission', _NA)}",
        ),
    ]
    _pdf_table(
        pdf,
        ["Report detail", "Value"],
        [[label, value] for label, value in details],
        ratios=[1.0, 1.6],
        aligns=["L", "R"],
        font_size=_FONT_SIZE_BODY,
    )

    pdf.set_y(pdf.get_y() + 2.0)
    _pdf_write_paragraph(
        pdf,
        "All monetary values are in Lakhs INR. This report was generated "
        "automatically by the RevInsight analytics engine; every figure is "
        "read directly from the analysis payload.",
        size=_FONT_SIZE_SMALL,
        color=_COLOR_MUTED,
    )


def _pdf_toc(pdf: _ReportPDF, sections: Sequence[dict[str, Any]]) -> None:
    """Render the table of contents.

    Parameters
    ----------
    pdf : _ReportPDF
        The document being rendered.
    sections : sequence of dict
        The report sections.
    """
    pdf.add_page()
    _pdf_section_heading(pdf, "", "Table of Contents")
    pdf.set_y(pdf.get_y() + 1.0)
    for section in sections:
        label = f"{section.get('number')}. {section.get('title')}"
        pdf.set_font(_FONT, "B", 10)
        pdf.set_text_color(*_COLOR_INK)
        pdf.set_xy(pdf.l_margin, pdf.get_y())
        pdf.cell(0, 7, _sanitize(label), align="L", new_x=XPos.LEFT, new_y=YPos.NEXT)
    pdf.set_y(pdf.get_y() + 2.0)
    _pdf_write_paragraph(
        pdf,
        "Each section presents the KPI tables, the analytical detail and "
        "the supporting charts for the corresponding part of the staffing "
        "decision.",
        size=_FONT_SIZE_SMALL,
        color=_COLOR_MUTED,
    )


def _render_pdf(
    model: dict[str, Any],
    output_path: str,
    image_map: dict[str, str],
) -> None:
    """Render the view model into a PDF document.

    Parameters
    ----------
    model : dict
        The view model.
    output_path : str
        Destination file path.
    image_map : dict
        Mapping of chart stem to the image path that should be embedded.
    """
    meta = model.get("meta", {})
    sections = model.get("sections", [])

    pdf = _ReportPDF(orientation="P", unit="mm", format="A4")
    pdf.alias_nb_pages()
    pdf.set_title(f"{_PRODUCT} - {_SUBTITLE}")
    pdf.set_author(f"{_PRODUCT} Analytics")
    pdf.set_creator("RevInsight report_generator")
    pdf.set_subject("Store staffing analysis report")
    pdf.meta = {
        "dataset": str(meta.get("dataset", "")),
    }
    pdf.generated = str(meta.get("generated", pdf.generated))

    _pdf_cover(pdf, meta)
    _pdf_toc(pdf, sections)

    for section in sections:
        pdf.add_page()
        _pdf_section_heading(pdf, section.get("number"), str(section.get("title")))
        _pdf_blocks(pdf, section.get("blocks", []), image_map)

    directory = os.path.dirname(os.path.abspath(output_path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    pdf.output(output_path)


# --------------------------------------------------------------------------- #
# HTML rendering                                                               #
# --------------------------------------------------------------------------- #
_HTML_CSS: str = """
:root{--bg:#0b1015;--panel:#141b23;--panel2:#1b242e;--ink:#e7eef5;--muted:#9fb0c0;
--accent:#00d4aa;--accent2:#4ecdc4;--warn:#ffb454;--bad:#ff6b6b;--line:#26313d}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--bg);color:var(--ink);line-height:1.55;
font:16px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,
sans-serif}
a{color:var(--accent)}
.wrap{max-width:1180px;margin:0 auto;padding:0 20px 64px}
.hero{background:linear-gradient(140deg,#062b26 0%,#0b1015 62%);
border-bottom:1px solid var(--line);padding:40px 20px 32px}
.hero-inner{max-width:1180px;margin:0 auto}
.brand{display:flex;align-items:baseline;gap:12px;flex-wrap:wrap}
.brand h1{margin:0;font-size:2.3rem;letter-spacing:.5px}
.brand span{color:var(--accent);font-size:1.05rem}
.tagline{color:var(--muted);margin:6px 0 20px}
.headline{background:rgba(0,212,170,.09);border:1px solid rgba(0,212,170,.35);
border-left:4px solid var(--accent);border-radius:10px;padding:14px 16px;
font-size:1.05rem}
.headline b{color:var(--accent)}
.meta{display:grid;gap:10px;margin-top:20px;
grid-template-columns:repeat(auto-fit,minmax(180px,1fr))}
.meta div{background:rgba(255,255,255,.04);border:1px solid var(--line);
border-radius:9px;padding:9px 11px}
.meta small{display:block;color:var(--muted);text-transform:uppercase;
letter-spacing:.07em;font-size:.66rem}
.meta span{font-weight:600}
nav.toc{background:var(--panel);border:1px solid var(--line);border-radius:12px;
padding:16px 20px;margin:26px 0}
nav.toc h2{margin:0 0 10px;font-size:.82rem;letter-spacing:.12em;
text-transform:uppercase;color:var(--muted)}
nav.toc ol{margin:0;padding-left:20px;columns:2;column-gap:28px}
nav.toc li{margin:3px 0;break-inside:avoid}
nav.toc a{text-decoration:none}
nav.toc a:hover{text-decoration:underline}
section{margin:34px 0 0;background:var(--panel);border:1px solid var(--line);
border-radius:14px;padding:22px 24px}
section>h2{margin:0 0 4px;font-size:1.32rem;color:var(--ink);
border-bottom:2px solid var(--accent);padding-bottom:8px}
section>h2 .num{color:var(--accent);margin-right:8px}
h3{font-size:.94rem;color:var(--accent);margin:22px 0 8px;letter-spacing:.03em}
p{margin:8px 0 10px}
.kpis{display:grid;gap:10px;margin:12px 0 6px;
grid-template-columns:repeat(auto-fit,minmax(168px,1fr))}
.kpi{background:var(--panel2);border:1px solid var(--line);border-radius:10px;
padding:10px 12px}
.kpi small{display:block;color:var(--muted);font-size:.68rem;text-transform:uppercase;
letter-spacing:.06em;min-height:2.1em}
.kpi b{font-size:1.02rem;color:var(--ink);word-break:break-word}
.badge{display:inline-block;padding:4px 12px;border-radius:999px;font-size:.76rem;
font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:#08131a}
.badge.high{background:#0d8a5c}
.badge.medium{background:#bf8600}
.badge.low{background:#b02a2a}
.table-wrap{overflow-x:auto;margin:6px 0 14px;-webkit-overflow-scrolling:touch}
table{width:100%;border-collapse:collapse;font-size:.86rem;min-width:460px}
caption{caption-side:top;text-align:left;color:var(--accent);font-weight:600;
font-size:.86rem;padding:6px 0}
th,td{padding:7px 10px;border-bottom:1px solid var(--line);text-align:left;
vertical-align:top}
th{background:var(--panel2);color:var(--muted);font-weight:600;font-size:.72rem;
text-transform:uppercase;letter-spacing:.05em;white-space:nowrap}
tbody tr:nth-child(even){background:rgba(255,255,255,.028)}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
ul.plain{margin:6px 0 12px;padding-left:20px}
ul.plain li{margin:4px 0}
figure{margin:16px 0}
figure img{width:100%;height:auto;display:block;border:1px solid var(--line);
border-radius:10px;background:#fff}
figcaption{color:var(--muted);font-size:.8rem;margin-top:6px;text-align:center}
details{margin:10px 0;border:1px solid var(--line);border-radius:10px;
background:var(--panel2)}
details summary{cursor:pointer;padding:11px 14px;font-weight:600;color:var(--accent);
list-style:none}
details summary::-webkit-details-marker{display:none}
details summary:before{content:"\\25B8 ";color:var(--muted)}
details[open] summary:before{content:"\\25BE "}
details .body{padding:0 14px 14px}
pre{margin:0;background:#0a0f14;border:1px solid var(--line);border-radius:8px;
padding:12px;overflow-x:auto;color:#cfe3dc;white-space:pre;
font:12px/1.45 "SFMono-Regular",Consolas,"Liberation Mono",Menlo,monospace}
footer.page{margin:38px 0 0;color:var(--muted);font-size:.78rem;text-align:center;
border-top:1px solid var(--line);padding-top:14px}
@media (max-width:760px){
.wrap{padding:0 12px 44px}
.hero{padding:26px 14px 22px}
.brand h1{font-size:1.75rem}
section{padding:16px 14px;border-radius:11px}
nav.toc ol{columns:1}
.kpis{grid-template-columns:repeat(auto-fit,minmax(140px,1fr))}
table{font-size:.79rem}
th,td{padding:6px 7px}
}
@media print{
body{background:#fff;color:#111}
.hero,section,nav.toc{background:#fff;border-color:#bbb}
.brand h1,section>h2,h3,caption{color:#0b5c52}
.kpi,th{background:#f2f6f5;color:#111}
figure img{background:#fff}
}
"""


def _html_escape(value: Any) -> str:
    """Escape *value* for safe inclusion in an HTML document.

    Parameters
    ----------
    value : Any
        The value to escape.

    Returns
    -------
    str
        The escaped text.
    """
    return html.escape(str(value), quote=True)


def _html_data_uri(path: str | None) -> str | None:
    """Return a base64 data URI for the image at *path*.

    Parameters
    ----------
    path : str or None
        The image path.

    Returns
    -------
    str or None
        The data URI, or ``None`` when the file cannot be read.
    """
    if not path or not os.path.isfile(path):
        return None
    try:
        payload = Path(path).read_bytes()
    except OSError as exc:
        logger.warning("Could not read chart %s: %s", path, exc)
        return None
    suffix = Path(path).suffix.lower()
    mime = "image/png"
    if suffix in (".jpg", ".jpeg"):
        mime = "image/jpeg"
    elif suffix == ".gif":
        mime = "image/gif"
    encoded = base64.b64encode(payload).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _html_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    aligns: Sequence[str] | None = None,
    title: str = "",
    numeric: Sequence[bool] | None = None,
) -> str:
    """Render an HTML table.

    Parameters
    ----------
    headers : sequence of str
        The column headers.
    rows : sequence of sequence of str
        The table body.
    aligns : sequence of str, optional
        Per-column alignment (``"L"``, ``"C"`` or ``"R"``).
    title : str, optional
        The table caption.
    numeric : sequence of bool, optional
        Per-column flag marking right-aligned numeric columns.

    Returns
    -------
    str
        The rendered HTML.
    """
    columns = max(len(headers), 1)
    if aligns and len(aligns) >= columns:
        column_aligns = list(aligns)[:columns]
    else:
        column_aligns = ["L"] * columns
    if numeric and len(numeric) >= columns:
        numeric_flags = list(numeric)[:columns]
    else:
        numeric_flags = [align == "R" for align in column_aligns]

    parts: list[str] = ['<div class="table-wrap"><table>']
    if title:
        parts.append(f"<caption>{_html_escape(title)}</caption>")
    parts.append("<thead><tr>")
    for index in range(columns):
        label = headers[index] if index < len(headers) else ""
        css = ' class="num"' if numeric_flags[index] else ""
        parts.append(f"<th{css}>{_html_escape(label)}</th>")
    parts.append("</tr></thead><tbody>")
    for row in rows:
        cells = list(row)[:columns]
        cells += [""] * (columns - len(cells))
        parts.append("<tr>")
        for index in range(columns):
            css = ' class="num"' if numeric_flags[index] else ""
            parts.append(f"<td{css}>{_html_escape(cells[index])}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table></div>")
    return "".join(parts)


def _html_blocks(
    blocks: Sequence[dict[str, Any]],
    chart_uris: dict[str, str | None],
) -> str:
    """Render a list of view-model blocks into HTML.

    Parameters
    ----------
    blocks : sequence of dict
        The blocks to render.
    chart_uris : dict
        Mapping of chart stem to a base64 data URI.

    Returns
    -------
    str
        The rendered HTML.
    """
    parts: list[str] = []
    for block in blocks:
        kind = block.get("type")
        title = str(block.get("title", ""))
        if kind == "text":
            if title:
                parts.append(f"<h3>{_html_escape(title)}</h3>")
            parts.append(f"<p>{_html_escape(str(block.get('text', '')))}</p>")
        elif kind == "bullets":
            if title:
                parts.append(f"<h3>{_html_escape(title)}</h3>")
            items = "".join(
                f"<li>{_html_escape(str(item))}</li>" for item in block.get("items", [])
            )
            parts.append(f'<ul class="plain">{items}</ul>')
        elif kind == "kv":
            parts.append(
                _html_table(
                    ["Metric", "Value"],
                    block.get("rows", []),
                    aligns=["L", "R"],
                    title=title,
                )
            )
        elif kind == "table":
            parts.append(
                _html_table(
                    block.get("headers", []),
                    block.get("rows", []),
                    aligns=block.get("aligns"),
                    title=title,
                )
            )
        elif kind == "chart":
            stem = str(block.get("stem", ""))
            uri = chart_uris.get(stem)
            if not uri:
                continue
            caption = str(block.get("caption", ""))
            figure = f'<figure><img src="{uri}" alt="{_html_escape(caption or stem)}">'
            if caption:
                figure += f"<figcaption>{_html_escape(caption)}</figcaption>"
            figure += "</figure>"
            parts.append(figure)
        elif kind == "badge":
            level = str(block.get("level", "")).upper()
            css = "medium"
            if "HIGH" in level:
                css = "high"
            elif "LOW" in level:
                css = "low"
            parts.append(
                f'<p><span class="badge {css}">'
                f"{_html_escape(str(block.get('text', '')))}</span></p>"
            )
        elif kind == "pre":
            if title:
                parts.append(f"<h3>{_html_escape(title)}</h3>")
            parts.append(
                f"<details><summary>Show the full plain-text report</summary>"
                '<div class="body"><pre>'
                f'{_html_escape(str(block.get("text", "")))}</pre>'
                f"</div></details>"
            )
    return "".join(parts)


def _html_kpi_grid(kpis: Any) -> str:
    """Render the compact KPI grid.

    Parameters
    ----------
    kpis : Any
        The ``kpis`` mapping of the payload.

    Returns
    -------
    str
        The rendered HTML.
    """
    if not isinstance(kpis, dict) or not kpis:
        return ""
    cards: list[str] = []
    for key, value in kpis.items():
        cards.append(
            '<div class="kpi"><small>'
            f"{_html_escape(_kpi_label(str(key)))}</small>"
            f"<b>{_html_escape(_kpi_value(str(key), value))}</b></div>"
        )
    return f'<div class="kpis">{"".join(cards)}</div>'


def _render_html(
    model: dict[str, Any],
    output_path: str,
    chart_uris: dict[str, str | None],
    kpis: Any,
) -> None:
    """Render the view model into a self-contained HTML document.

    Parameters
    ----------
    model : dict
        The view model.
    output_path : str
        Destination file path.
    chart_uris : dict
        Mapping of chart stem to a base64 data URI.
    kpis : Any
        The ``kpis`` mapping of the payload.
    """
    meta = model.get("meta", {})
    sections = model.get("sections", [])

    toc_items = "".join(
        f'<li><a href="#section-{_html_escape(section.get("number"))}">'
        f"{_html_escape(section.get('number'))}. "
        f"{_html_escape(section.get('title'))}</a></li>"
        for section in sections
    )

    meta_items = [
        ("Dataset", meta.get("dataset", _NA)),
        ("Months of history", meta.get("months", _NA)),
        ("Salespersons", meta.get("salespersons", _NA)),
        ("Total sales", meta.get("total_sales", _NA)),
        ("Generated", meta.get("generated", _NA)),
        ("Run id", meta.get("run_id", _NA)),
    ]
    meta_html = "".join(
        f"<div><small>{_html_escape(label)}</small>" f"<span>{_html_escape(value)}</span></div>"
        for label, value in meta_items
    )

    body_parts: list[str] = []
    for section in sections:
        body_parts.append(
            f'<section id="section-{_html_escape(section.get("number"))}">'
            f'<h2><span class="num">{_html_escape(section.get("number"))}.</span>'
            f"{_html_escape(section.get('title'))}</h2>"
        )
        body_parts.append(_html_blocks(section.get("blocks", []), chart_uris))
        body_parts.append("</section>")

    kpi_grid = _html_kpi_grid(kpis)
    kpi_block = ""
    if kpi_grid:
        kpi_block = (
            '<section id="kpi-summary"><h2>Key performance indicators</h2>' f"{kpi_grid}</section>"
        )

    document = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="description" content="{_html_escape(meta.get('headline', _NA))}">
<title>{_html_escape(_PRODUCT)} - {_html_escape(_SUBTITLE)}</title>
<style>{_HTML_CSS}</style>
</head>
<body>
<header class="hero"><div class="hero-inner">
<div class="brand"><h1>{_html_escape(_PRODUCT)}</h1>
<span>{_html_escape(_SUBTITLE)}</span></div>
<p class="tagline">Data-driven staffing optimisation for a retail store.
All monetary values are in Lakhs INR.</p>
<div class="headline"><b>Headline:</b> {_html_escape(meta.get("headline", _NA))}</div>
<div class="meta">{meta_html}</div>
</div></header>
<div class="wrap">
{kpi_block}
<nav class="toc"><h2>Contents</h2><ol>{toc_items}</ol></nav>
{"".join(body_parts)}
<footer class="page">
{_html_escape(_PRODUCT)} - {_html_escape(_SUBTITLE)} | Dataset
{_html_escape(meta.get("dataset", _NA))} | Generated
{_html_escape(meta.get("generated", _NA))} | Run
{_html_escape(meta.get("run_id", _NA))}
</footer>
</div>
</body>
</html>
"""

    directory = os.path.dirname(os.path.abspath(output_path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(document)


# --------------------------------------------------------------------------- #
# Public API                                                                   #
# --------------------------------------------------------------------------- #
def _collect_chart_paths(
    charts_dir: str,
    results: dict[str, Any],
) -> dict[str, str]:
    """Return a mapping of chart stem to the absolute path of the image.

    Parameters
    ----------
    charts_dir : str
        Directory holding the generated charts.
    results : dict
        The analysis payload.

    Returns
    -------
    dict
        Mapping of stem to absolute path for every chart that exists.
    """
    stems: list[str] = []
    declared = results.get("charts") if isinstance(results, dict) else None
    if isinstance(declared, dict):
        stems.extend(str(key) for key in declared.keys())
    elif isinstance(declared, (list, tuple)):
        stems.extend(str(item) for item in declared)
    stems.extend(_discover_charts(charts_dir, prefix="forecast_"))

    found: dict[str, str] = {}
    for stem in stems:
        if stem in found:
            continue
        path = _chart_path(charts_dir, stem)
        if path:
            found[stem] = path
    return found


def generate_pdf_report(
    all_results: dict,
    summary_df: pd.DataFrame,
    charts_dir: str,
    output_path: str,
    salary: float = 3.0,
    commission: float = 0.05,
) -> str:
    """Generate the RevInsight PDF report from an analysis payload.

    Parameters
    ----------
    all_results : dict
        The v2.0 analysis payload produced by
        :func:`revinsight.pipeline.run_analysis`.
    summary_df : pd.DataFrame
        The monthly summary (``Month``, ``Sales``, ``ActiveEmployees``,
        ``SalesPerEmployee``, ``GrossMargin``).
    charts_dir : str
        Directory holding the generated chart PNGs.  Charts that are
        missing are skipped silently.
    output_path : str
        Full path of the PDF file to write.
    salary : float, optional
        Monthly salary per employee in Lakhs INR (default ``3.0``).
    commission : float, optional
        Commission rate as a fraction of sales (default ``0.05``).

    Returns
    -------
    str
        The absolute path of the written PDF file.

    Raises
    ------
    RuntimeError
        If the report could not be rendered or written.
    """
    logger.info("Generating PDF report -> %s", output_path)
    try:
        results = all_results if isinstance(all_results, dict) else {}
        model = _build_view_model(results, summary_df, charts_dir, salary, commission)
        chart_paths = _collect_chart_paths(charts_dir, results)
        image_map = _optimise_images(list(chart_paths.values()))
        stem_map: dict[str, str] = {}
        for stem, path in chart_paths.items():
            stem_map[stem] = image_map.get(path, path)
        _render_pdf(model, output_path, stem_map)
    except Exception as exc:  # noqa: BLE001 - re-raised with context
        logger.exception("PDF report generation failed")
        raise RuntimeError(f"Failed to generate the PDF report at '{output_path}': {exc}") from exc
    logger.info("PDF report written to %s", os.path.abspath(output_path))
    return os.path.abspath(output_path)


def generate_html_report(
    results: dict,
    output_path: str,
    charts_dir: str = "",
) -> str:
    """Generate the RevInsight HTML report from an analysis payload.

    The result is a single, self-contained, responsive, dark-themed HTML
    file.  Charts are embedded as base64 data URIs so the document can be
    shared on its own.

    Parameters
    ----------
    results : dict
        The v2.0 analysis payload produced by
        :func:`revinsight.pipeline.run_analysis`.
    output_path : str
        Full path of the HTML file to write.
    charts_dir : str, optional
        Directory holding the generated chart PNGs.  Charts that are
        missing are skipped silently.

    Returns
    -------
    str
        The absolute path of the written HTML file.

    Raises
    ------
    RuntimeError
        If the report could not be rendered or written.
    """
    logger.info("Generating HTML report -> %s", output_path)
    try:
        payload = results if isinstance(results, dict) else {}
        model = _build_view_model(payload, pd.DataFrame(), charts_dir, 3.0, 0.05)
        chart_paths = _collect_chart_paths(charts_dir, payload)
        chart_uris: dict[str, str | None] = {
            stem: _html_data_uri(path) for stem, path in chart_paths.items()
        }
        _render_html(model, output_path, chart_uris, payload.get("kpis"))
    except Exception as exc:  # noqa: BLE001 - re-raised with context
        logger.exception("HTML report generation failed")
        raise RuntimeError(f"Failed to generate the HTML report at '{output_path}': {exc}") from exc
    logger.info("HTML report written to %s", os.path.abspath(output_path))
    return os.path.abspath(output_path)
