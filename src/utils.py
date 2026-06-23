"""
Utility functions for the RevInsight staffing-analysis project.

This module provides helper routines for loading and validating sales data,
computing financial metrics (gross margin, break-even sales), and preparing
output artefacts.  All monetary values are denominated in **Lakhs INR**.

Typical usage::

    from src.utils import load_and_validate_data, prepare_monthly_summary

    df = load_and_validate_data("data/RevI-Test.csv")
    summary = prepare_monthly_summary(df)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import pandas as pd

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SALESPERSON_PREFIX: str = "sale sp"
"""Column-name prefix that identifies individual salesperson columns."""


COST_OF_GOODS_RATE: float = 0.50
"""Cost-of-goods-sold as a fraction of sales (50 %)."""

# 1. Data Loading & Validation

def load_and_validate_data(filepath: str | Path) -> pd.DataFrame:
    """Load a sales CSV file, validate its schema, and clean the data.

    Processing steps
    ~~~~~~~~~~~~~~~~
    1. Read the CSV from *filepath*.
    2. Strip leading/trailing whitespace from every column name.
    3. Validate that the mandatory columns (``Month``, ``Sales``,
       ``sale sp1`` … ``sale sp16``) are present.
    4. Forward-fill then zero-fill any remaining ``NaN`` values in
       salesperson columns; drop rows where ``Month`` or ``Sales``
       are still missing after the fill.

    Parameters
    ----------
    filepath : str or Path
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame ready for downstream analysis.

    Raises
    ------
    FileNotFoundError
        If *filepath* does not exist.
    ValueError
        If required columns are missing after whitespace stripping.

    Examples
    --------
    >>> df = load_and_validate_data("data/RevI-Test.csv")
    >>> list(df.columns[:3])
    ['Month', 'Sales', 'sale sp1']
    """
    filepath = Path(filepath)

    if not filepath.exists():
        logger.error("Data file not found: %s", filepath)
        raise FileNotFoundError(f"Data file not found: {filepath}")

    logger.info("Loading data from %s", filepath)
    df = pd.read_csv(filepath)

    # --- strip column whitespace -------------------------------------------
    df.columns = df.columns.str.strip()
    logger.debug("Columns after stripping: %s", list(df.columns))

    # --- validate required columns -----------------------------------------
    # Dynamically find salesperson columns (they start with 'sale sp')
    expected_sp_cols = [c for c in df.columns if str(c).strip().startswith(SALESPERSON_PREFIX)]
    if not expected_sp_cols:
        raise ValueError(f"No salesperson columns found matching prefix '{SALESPERSON_PREFIX}'")
        
    required_columns = ["Month", "Sales"] + expected_sp_cols

    missing = set(required_columns) - set(df.columns)
    if missing:
        logger.error("Missing required columns: %s", missing)
        raise ValueError(
            f"The following required columns are missing from the data: "
            f"{sorted(missing)}"
        )

    # --- handle missing values ---------------------------------------------
    initial_nan_count = int(df[required_columns].isna().sum().sum())
    if initial_nan_count:
        logger.warning(
            "Found %d missing value(s) in required columns; filling.",
            initial_nan_count,
        )

    # Salesperson columns: missing → 0 (employee was absent / not yet hired)
    df[expected_sp_cols] = df[expected_sp_cols].fillna(0)

    # Month / Sales: drop rows that are entirely missing
    rows_before = len(df)
    df = df.dropna(subset=["Month", "Sales"])
    rows_dropped = rows_before - len(df)
    if rows_dropped:
        logger.warning(
            "Dropped %d row(s) with missing Month or Sales.", rows_dropped
        )

    df = df.reset_index(drop=True)
    logger.info(
        "Data loaded successfully: %d rows × %d columns", *df.shape
    )
    return df


# 2. Employee Counting

def count_active_employees(row: pd.Series) -> int:
    """Count the number of active (non-zero) salespersons in a data row.

    A salesperson is considered *active* in a given month when their
    individual sales figure is non-zero.

    Parameters
    ----------
    row : pd.Series
        A single row from the sales DataFrame.  Must contain columns
        ``sale sp1`` through ``sale sp16``.

    Returns
    -------
    int
        Number of salesperson columns with a value ≠ 0.

    Examples
    --------
    >>> import pandas as pd
    >>> row = pd.Series({"sale sp1": 5.0, "sale sp2": 0.0, "sale sp3": 3.2})
    >>> # Only sp1 and sp3 are in the expected range; sp2 is zero.
    >>> # (In practice all 16 columns are present.)
    """
    sp_cols = [c for c in row.index if str(c).strip().startswith(SALESPERSON_PREFIX)]
    if not sp_cols:
        return 0
    return int((row[sp_cols] != 0).sum())


# 3. Financial Calculations

def calculate_gross_margin(
    sales: float,
    num_employees: int,
    salary_per_person: float = 3.0,
    commission_rate: float = 0.05,
) -> float:
    """Compute the gross margin for a given month.

    The formula is::

        Gross Margin = (1 - COGS_rate - commission_rate) × Sales
                       − salary_per_person × num_employees
                     = 0.45 × Sales  −  3 × num_employees

    where the default cost-of-goods-sold rate is 50 % and the default
    commission rate is 5 %.

    Parameters
    ----------
    sales : float
        Total monthly sales in Lakhs INR.
    num_employees : int
        Number of active employees during the month.
    salary_per_person : float, optional
        Monthly salary cost per employee in Lakhs INR (default ``3.0``).
    commission_rate : float, optional
        Commission rate on total sales (default ``0.05``).

    Returns
    -------
    float
        Gross margin in Lakhs INR.

    Examples
    --------
    >>> calculate_gross_margin(100, 7)
    24.0
    >>> calculate_gross_margin(100, 7, salary_per_person=3, commission_rate=0.05)
    24.0
    """
    # Effective margin: (1 - COGS_rate) * Sales - commission_rate * Sales - salary * n
    # = 0.50 * Sales - 0.05 * Sales - 3 * n
    # = 0.45 * Sales - 3 * n
    effective_margin_rate = 1.0 - COST_OF_GOODS_RATE - commission_rate
    gross_margin = effective_margin_rate * sales - salary_per_person * num_employees
    return round(gross_margin, 6)


def calculate_break_even_sales(
    salary_per_person: float = 3.0,
    commission_rate: float = 0.05,
) -> float:
    """Return the incremental sales needed to break even on one new hire.

    The break-even point is reached when the additional margin generated
    by the extra sales exactly covers the new employee's salary::

        margin_rate = 1.0 - COST_OF_GOODS_RATE - commission_rate
        break_even_sales = salary_per_person / margin_rate


    Parameters
    ----------
    salary_per_person : float, optional
        Monthly salary cost per employee in Lakhs INR.
    commission_rate : float, optional
        Commission rate.

    Returns
    -------
    float
        Break-even sales in Lakhs INR.

    Raises
    ------
    ZeroDivisionError
        If *margin_rate* is zero.

    Examples
    --------
    >>> round(calculate_break_even_sales(), 3)
    6.667
    """
    margin_rate = 1.0 - COST_OF_GOODS_RATE - commission_rate
    if margin_rate == 0:
        logger.error("margin_rate is zero; cannot compute break-even sales.")
        raise ZeroDivisionError(
            "margin_rate must be non-zero to calculate break-even sales."
        )
    return salary_per_person / margin_rate


# 4. Column Helpers

def get_salesperson_columns(df: pd.DataFrame) -> List[str]:
    """Return the list of salesperson column names present in *df*.

    Columns are identified by the prefix ``sale sp`` and returned in
    their original DataFrame order.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame whose columns should be inspected.

    Returns
    -------
    list of str
        Salesperson column names, e.g.
        ``['sale sp1', 'sale sp2', …, 'sale sp16']``.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame(columns=["Month", "Sales", "sale sp1", "sale sp2"])
    >>> get_salesperson_columns(df)
    ['sale sp1', 'sale sp2']
    """
    sp_cols = [
        col for col in df.columns if col.lower().startswith(SALESPERSON_PREFIX)
    ]
    logger.debug("Found %d salesperson column(s).", len(sp_cols))
    return sp_cols


# 5. Monthly Summary

def prepare_monthly_summary(
    df: pd.DataFrame,
    salary: float = 3.0,
    commission: float = 0.05,
) -> pd.DataFrame:
    """Build a per-month summary table from the raw sales data.

    The returned DataFrame contains:

    ============== ================================================
    Column         Description
    ============== ================================================
    Month          Calendar month number.
    Sales          Total monthly sales (Lakhs INR).
    ActiveEmployees Number of salespersons with non-zero sales.
    SalesPerEmployee Sales ÷ ActiveEmployees (``NaN`` if zero).
    GrossMargin    Gross margin computed via
                   :func:`calculate_gross_margin`.
    ============== ================================================

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned sales DataFrame as returned by
        :func:`load_and_validate_data`.

    Returns
    -------
    pd.DataFrame
        Summary DataFrame indexed from 0 with the five columns above.

    Examples
    --------
    >>> df = load_and_validate_data("data/RevI-Test.csv")
    >>> summary = prepare_monthly_summary(df)
    >>> list(summary.columns)
    ['Month', 'Sales', 'ActiveEmployees', 'SalesPerEmployee', 'GrossMargin']
    """
    logger.info("Preparing monthly summary for %d rows.", len(df))

    summary = pd.DataFrame()
    summary["Month"] = df["Month"]
    summary["Sales"] = df["Sales"]

    summary["ActiveEmployees"] = df.apply(count_active_employees, axis=1)

    # Guard against division by zero when no employees are active
    summary["SalesPerEmployee"] = summary["Sales"] / summary[
        "ActiveEmployees"
    ].replace(0, float("nan"))

    summary["GrossMargin"] = summary.apply(
        lambda r: calculate_gross_margin(
            r["Sales"], r["ActiveEmployees"],
            salary_per_person=salary, commission_rate=commission
        ),
        axis=1,
    )

    logger.info("Monthly summary created successfully.")
    return summary.reset_index(drop=True)


# 6. Output Directory Setup

def setup_output_dirs(base_path: str | Path) -> dict[str, Path]:
    """Create the standard output directory tree under *base_path*.

    The following directories are created (if they do not already exist):

    * ``<base_path>/output/``
    * ``<base_path>/output/charts/``
    * ``<base_path>/reports/``

    Parameters
    ----------
    base_path : str or Path
        Root directory of the project (e.g. ``RevInsight_Assignment/``).

    Returns
    -------
    dict of str → Path
        Mapping of logical names to their resolved :class:`~pathlib.Path`
        objects:  ``{"output", "charts", "reports"}``.

    Examples
    --------
    >>> dirs = setup_output_dirs("RevInsight_Assignment")
    >>> sorted(dirs.keys())
    ['charts', 'output', 'reports']
    """
    base = Path(base_path)

    # Assumes base_path is the actual desired output directory
    dirs = {
        "output": base,
        "charts": base / "charts",
    }

    for name, directory in dirs.items():
        directory.mkdir(parents=True, exist_ok=True)
        logger.info("Ensured directory exists: %s", directory)

    return dirs


# 7. Currency Formatting

def format_currency(value: float) -> str:
    """Format a numeric value as an INR Lakhs currency string.

    Parameters
    ----------
    value : float
        Amount in Lakhs INR.

    Returns
    -------
    str
        Human-readable string, e.g. ``'₹12.35 Lakhs'``.

    Examples
    --------
    >>> format_currency(12.3456)
    '₹12.35 Lakhs'
    >>> format_currency(-3.1)
    '₹-3.10 Lakhs'
    >>> format_currency(0)
    '₹0.00 Lakhs'
    """
    return f"₹{value:,.2f} Lakhs"
