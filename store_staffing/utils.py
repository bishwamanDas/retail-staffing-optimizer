"""Core data-loading, validation and aggregation helpers.

This module is the single entry point for turning a raw sales CSV into a
clean, analysis-ready monthly summary.  It is deliberately free of any
modelling logic so that it can be reused by the CLI, the REST API and the
test-suite without side effects.

Data contract
-------------
The input CSV must contain:

* ``Month`` — monotonically increasing integer month index (1-based).
* ``Sales`` — total monthly sales in Lakhs INR.
* One or more ``sale spN`` columns — individual salesperson contribution
  in Lakhs INR.  A zero means the salesperson was not on the roster that
  month (not yet hired, on leave, or already left).

All monetary values are denominated in **Lakhs INR**.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "SALESPERSON_PREFIX",
    "COST_OF_GOODS_RATE",
    "load_and_validate_data",
    "count_active_employees",
    "calculate_gross_margin",
    "calculate_break_even_sales",
    "get_salesperson_columns",
    "prepare_monthly_summary",
    "setup_output_dirs",
    "format_currency",
    "infer_month_of_year",
    "to_jsonable",
]

# --------------------------------------------------------------------------- #
# Constants                                                                    #
# --------------------------------------------------------------------------- #

SALESPERSON_PREFIX: str = "sale sp"
"""Column-name prefix identifying individual salesperson columns."""

COST_OF_GOODS_RATE: float = 0.50
"""Default cost-of-goods-sold rate as a fraction of sales (50 %)."""

REQUIRED_CORE_COLUMNS: tuple[str, ...] = ("Month", "Sales")
"""Columns that must exist in every accepted input file."""


# --------------------------------------------------------------------------- #
# 1. Data loading & validation                                                 #
# --------------------------------------------------------------------------- #


def load_and_validate_data(filepath: str | Path) -> pd.DataFrame:
    """Load a sales CSV, validate its schema and clean the data.

    Processing steps
    ----------------
    1. Read the CSV from *filepath*.
    2. Strip leading/trailing whitespace from every column name.
    3. Locate the salesperson columns dynamically (anything starting with
       ``sale sp``), so the pipeline works with any team size.
    4. Coerce ``Month``/``Sales`` to numeric, dropping unusable rows.
    5. Forward-fill then zero-fill ``NaN`` values in salesperson columns
       (a missing month means the salesperson generated no sales).
    6. Sort by ``Month`` and de-duplicate on it.

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
        If required columns are missing, no salesperson column is present,
        or the file contains no usable rows.

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

    # --- strip column whitespace ----------------------------------------- #
    df.columns = [str(c).strip() for c in df.columns]
    logger.debug("Columns after stripping: %s", list(df.columns))

    # --- locate salesperson columns -------------------------------------- #
    sp_cols = [c for c in df.columns if c.lower().startswith(SALESPERSON_PREFIX)]
    if not sp_cols:
        raise ValueError(
            f"No salesperson columns found matching prefix "
            f"'{SALESPERSON_PREFIX}'. Expected at least one column named "
            f"'sale sp1', 'sale sp2', ..."
        )

    # --- validate required core columns ---------------------------------- #
    missing = [c for c in REQUIRED_CORE_COLUMNS if c not in df.columns]
    if missing:
        logger.error("Missing required columns: %s", missing)
        raise ValueError(f"The following required columns are missing from the data: {missing}")

    # --- coerce numeric types -------------------------------------------- #
    df["Month"] = pd.to_numeric(df["Month"], errors="coerce")
    df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce")

    initial_nan = int(df[list(REQUIRED_CORE_COLUMNS)].isna().sum().sum())
    if initial_nan:
        logger.warning("Found %d missing value(s) in required columns.", initial_nan)

    rows_before = len(df)
    df = df.dropna(subset=list(REQUIRED_CORE_COLUMNS))
    rows_dropped = rows_before - len(df)
    if rows_dropped:
        logger.warning("Dropped %d row(s) with missing Month or Sales.", rows_dropped)

    # --- salesperson columns: missing means "no sales this month" --------- #
    df[sp_cols] = df[sp_cols].apply(pd.to_numeric, errors="coerce")
    df[sp_cols] = df[sp_cols].fillna(0.0)

    # --- ordering & duplicates -------------------------------------------- #
    df = df.sort_values("Month").reset_index(drop=True)
    dup_mask = df["Month"].duplicated()
    if dup_mask.any():
        logger.warning(
            "Dropping %d duplicate Month row(s) (keeping the first).",
            int(dup_mask.sum()),
        )
        df = df.loc[~dup_mask].reset_index(drop=True)

    if df.empty:
        raise ValueError("The dataset contains no usable rows after cleaning.")

    df["Month"] = df["Month"].astype(int)
    df["Sales"] = df["Sales"].astype(float)

    logger.info("Data loaded successfully: %d rows x %d columns", *df.shape)
    return df


# --------------------------------------------------------------------------- #
# 2. Employee counting                                                         #
# --------------------------------------------------------------------------- #


def count_active_employees(row: pd.Series) -> int:
    """Count the salespeople active (generating revenue) in a single month.

    A salesperson counts as *active* when their individual sales are
    strictly positive.  Negative values (returns) are not counted.

    Parameters
    ----------
    row : pd.Series
        A single row from the sales DataFrame.

    Returns
    -------
    int
        Number of salesperson columns with a value > 0.

    Examples
    --------
    >>> import pandas as pd
    >>> row = pd.Series({"sale sp1": 5.0, "sale sp2": 0.0, "sale sp3": 3.2})
    >>> count_active_employees(row)
    2
    """
    sp_cols = [c for c in row.index if str(c).strip().lower().startswith(SALESPERSON_PREFIX)]
    if not sp_cols:
        return 0
    return int((row[sp_cols] > 0).sum())


# --------------------------------------------------------------------------- #
# 3. Financial calculations                                                    #
# --------------------------------------------------------------------------- #


def calculate_gross_margin(
    sales: float,
    num_employees: int,
    salary_per_person: float = 3.0,
    commission_rate: float = 0.05,
    cogs_rate: float = COST_OF_GOODS_RATE,
) -> float:
    """Compute the gross margin for a given month.

    The formula is::

        Gross Margin = (1 - COGS_rate - commission_rate) x Sales
                       - salary_per_person x num_employees

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
    cogs_rate : float, optional
        Cost of goods sold rate (default ``0.50``).

    Returns
    -------
    float
        Gross margin in Lakhs INR.

    Examples
    --------
    >>> calculate_gross_margin(100, 7)
    24.0
    """
    effective_margin_rate = 1.0 - cogs_rate - commission_rate
    gross_margin = effective_margin_rate * sales - salary_per_person * num_employees
    return round(float(gross_margin), 6)


def calculate_break_even_sales(
    salary_per_person: float = 3.0,
    commission_rate: float = 0.05,
    cogs_rate: float = COST_OF_GOODS_RATE,
) -> float:
    """Return the sales needed to break even on one new hire.

    The break-even point is reached when the additional margin generated
    by the extra sales exactly covers the new employee's salary::

        break_even_sales = salary_per_person / (1 - COGS_rate - commission_rate)

    Parameters
    ----------
    salary_per_person : float, optional
        Monthly salary cost per employee in Lakhs INR.
    commission_rate : float, optional
        Commission rate.
    cogs_rate : float, optional
        Cost of goods sold rate.

    Returns
    -------
    float
        Break-even sales in Lakhs INR.

    Raises
    ------
    ZeroDivisionError
        If the effective margin rate is zero.

    Examples
    --------
    >>> round(calculate_break_even_sales(), 3)
    6.667
    """
    margin_rate = 1.0 - cogs_rate - commission_rate
    if margin_rate <= 0:
        logger.error("margin_rate is non-positive; cannot compute break-even sales.")
        raise ZeroDivisionError("margin_rate must be positive to calculate break-even sales.")
    return salary_per_person / margin_rate


# --------------------------------------------------------------------------- #
# 4. Column helpers                                                            #
# --------------------------------------------------------------------------- #


def get_salesperson_columns(df: pd.DataFrame) -> list[str]:
    """Return the salesperson column names present in *df*, in file order.

    Columns are identified by the prefix ``sale sp`` (case-insensitive).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame whose columns should be inspected.

    Returns
    -------
    list of str
        Salesperson column names, e.g. ``['sale sp1', 'sale sp2', ...]``.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame(columns=["Month", "Sales", "sale sp1", "sale sp2"])
    >>> get_salesperson_columns(df)
    ['sale sp1', 'sale sp2']
    """
    sp_cols = [col for col in df.columns if str(col).strip().lower().startswith(SALESPERSON_PREFIX)]
    logger.debug("Found %d salesperson column(s).", len(sp_cols))
    return sp_cols


# --------------------------------------------------------------------------- #
# 5. Calendar helpers                                                          #
# --------------------------------------------------------------------------- #


def infer_month_of_year(month_index: int, start_month: int = 1) -> int:
    """Map a 1-based month index onto a calendar month number (1-12).

    The input datasets index months sequentially from ``1`` without calendar
    dates, so the month-of-year is inferred by assuming a January start and
    wrapping every 12 months.

    Parameters
    ----------
    month_index : int
        1-based sequential month number from the dataset.
    start_month : int, optional
        Calendar month (1-12) that ``month_index == 1`` corresponds to.

    Returns
    -------
    int
        Calendar month number in the range 1-12.

    Examples
    --------
    >>> infer_month_of_year(1)
    1
    >>> infer_month_of_year(13)
    1
    """
    if not 1 <= start_month <= 12:
        raise ValueError("start_month must be between 1 and 12")
    return ((int(month_index) - 1 + start_month - 1) % 12) + 1


def month_name(month_number: int) -> str:
    """Return the abbreviated English name for a calendar month number."""
    names = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]
    if not 1 <= month_number <= 12:
        raise ValueError("month_number must be between 1 and 12")
    return names[month_number - 1]


# --------------------------------------------------------------------------- #
# 6. Monthly summary                                                            #
# --------------------------------------------------------------------------- #


def prepare_monthly_summary(
    df: pd.DataFrame,
    salary: float = 3.0,
    commission: float = 0.05,
    cogs_rate: float = COST_OF_GOODS_RATE,
    start_month: int = 1,
) -> pd.DataFrame:
    """Build a per-month summary table from the raw sales data.

    The returned DataFrame contains:

    ================== ==================================================
    Column              Description
    ================== ==================================================
    Month               Sequential month index (1-based).
    MonthOfYear         Calendar month number (1-12).
    Sales               Total monthly sales (Lakhs INR).
    ActiveEmployees     Salespersons with positive sales.
    SalesPerEmployee    Sales / ActiveEmployees (``NaN`` if zero).
    GrossMargin         Gross margin via :func:`calculate_gross_margin`.
    Rolling3Sales       3-month rolling mean of ``Sales``.
    Rolling3SalesPerEmp 3-month rolling mean of ``SalesPerEmployee``.
    ================== ==================================================

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned sales DataFrame as returned by
        :func:`load_and_validate_data`.
    salary : float, optional
        Monthly salary per employee in Lakhs INR.
    commission : float, optional
        Commission rate as a fraction of sales.
    cogs_rate : float, optional
        Cost of goods sold rate.
    start_month : int, optional
        Calendar month corresponding to ``Month == 1``.

    Returns
    -------
    pd.DataFrame
        Summary DataFrame indexed from 0.
    """
    logger.info("Preparing monthly summary for %d rows.", len(df))

    summary = pd.DataFrame(index=df.index)
    summary["Month"] = df["Month"].astype(int)
    summary["MonthOfYear"] = [infer_month_of_year(m, start_month) for m in summary["Month"]]
    summary["Sales"] = df["Sales"].astype(float)

    summary["ActiveEmployees"] = df.apply(count_active_employees, axis=1)

    summary["SalesPerEmployee"] = summary["Sales"] / summary["ActiveEmployees"].replace(
        0, float("nan")
    )

    summary["GrossMargin"] = summary.apply(
        lambda r: calculate_gross_margin(
            r["Sales"],
            r["ActiveEmployees"],
            salary_per_person=salary,
            commission_rate=commission,
            cogs_rate=cogs_rate,
        ),
        axis=1,
    )

    summary["Rolling3Sales"] = summary["Sales"].rolling(3, min_periods=1).mean()
    summary["Rolling3SalesPerEmp"] = summary["SalesPerEmployee"].rolling(3, min_periods=1).mean()

    logger.info("Monthly summary created successfully.")
    return summary.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 7. Output directory setup                                                    #
# --------------------------------------------------------------------------- #


def setup_output_dirs(base_path: str | Path, with_reports: bool = True) -> dict[str, Path]:
    """Create the standard output directory tree under *base_path*.

    The following directories are created (if they do not already exist):

    * ``<base_path>/``            — the output root.
    * ``<base_path>/charts/``     — generated PNG charts.
    * ``<base_path>/reports/``    — generated PDF/HTML reports (optional).

    Parameters
    ----------
    base_path : str or Path
        Root directory for generated artefacts.
    with_reports : bool, optional
        Also create a sibling ``reports`` directory (default ``True``).

    Returns
    -------
    dict of str -> Path
        Mapping of logical names to resolved paths:
        ``{"output", "charts", "reports"}``.
    """
    base = Path(base_path)
    dirs: dict[str, Path] = {
        "output": base,
        "charts": base / "charts",
    }
    if with_reports:
        dirs["reports"] = base / "reports"

    for _name, directory in dirs.items():
        directory.mkdir(parents=True, exist_ok=True)
        logger.debug("Ensured directory exists: %s", directory)

    return dirs


# --------------------------------------------------------------------------- #
# 8. Currency formatting                                                       #
# --------------------------------------------------------------------------- #


def format_currency(value: float) -> str:
    """Format a numeric value as an INR Lakhs currency string.

    Parameters
    ----------
    value : float
        Amount in Lakhs INR.

    Returns
    -------
    str
        Human-readable string, e.g. ``'Rs.12.35 Lakhs'``.

    Examples
    --------
    >>> format_currency(12.3456)
    'Rs.12.35 Lakhs'
    """
    return f"Rs.{value:,.2f} Lakhs"


# --------------------------------------------------------------------------- #
# 9. JSON serialisation helpers                                                #
# --------------------------------------------------------------------------- #


def to_jsonable(obj: Any) -> Any:
    """Recursively convert *obj* into JSON-serialisable primitives.

    Handles ``numpy`` scalars/arrays, ``pandas`` Series/DataFrame,
    ``NaN``/``Infinity`` (mapped to ``None``), ``datetime`` objects and
    arbitrary containers.

    Parameters
    ----------
    obj : Any
        Object to convert.

    Returns
    -------
    Any
        A structure containing only ``dict``/``list``/``str``/``int``/
        ``float``/``bool``/``None``.
    """
    if obj is None:
        return None
    if isinstance(obj, (str, bool, int)):
        return obj
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        return None if (math.isnan(value) or math.isinf(value)) else value
    if isinstance(obj, np.ndarray):
        return [to_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, pd.DataFrame):
        return [to_jsonable(record) for record in obj.to_dict("records")]
    if isinstance(obj, pd.Series):
        return [to_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    if hasattr(obj, "to_dict"):
        try:
            return to_jsonable(obj.to_dict())
        except Exception:  # noqa: BLE001 - defensive fallback
            pass
    if hasattr(obj, "item"):
        try:
            return to_jsonable(obj.item())
        except Exception:  # noqa: BLE001 - defensive fallback
            pass
    return str(obj)


def round_floats(obj: Any, ndigits: int = 4) -> Any:
    """Recursively round every float inside *obj* to *ndigits* decimals.

    Parameters
    ----------
    obj : Any
        Nested structure of primitives.
    ndigits : int, optional
        Number of decimal places to keep (default 4).

    Returns
    -------
    Any
        The same structure with floats rounded.
    """
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else round(obj, ndigits)
    if isinstance(obj, dict):
        return {k: round_floats(v, ndigits) for k, v in obj.items()}
    if isinstance(obj, list):
        return [round_floats(v, ndigits) for v in obj]
    return obj


def safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Divide safely, returning *default* when the denominator is zero.

    Parameters
    ----------
    numerator : float
        Dividend.
    denominator : float
        Divisor.
    default : float, optional
        Value returned when *denominator* is zero.

    Returns
    -------
    float
        The quotient or *default*.
    """
    if denominator == 0 or not np.isfinite(denominator):
        return default
    result = numerator / denominator
    return default if not np.isfinite(result) else float(result)


def chunked(sequence: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    """Yield successive *size*-sized chunks of *sequence*.

    Parameters
    ----------
    sequence : Sequence
        Sequence to split.
    size : int
        Chunk size (must be > 0).

    Yields
    ------
    Sequence
        Chunks of at most *size* elements.
    """
    if size <= 0:
        raise ValueError("size must be > 0")
    for start in range(0, len(sequence), size):
        yield sequence[start : start + size]
