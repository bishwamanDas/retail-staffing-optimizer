"""
Visualization module for RevInsight branded-clothing store staffing analysis.

Generates publication-quality charts covering sales trends, employee
productivity, statistical distributions, correlation analysis, rolling
metrics, and scenario comparisons.  Every chart is saved as a 150-DPI PNG
to the caller-specified output directory.

Typical usage
-------------
>>> from src.visualization import generate_all_charts
>>> generate_all_charts(summary_df, raw_df, "outputs/charts")
"""

# ---------------------------------------------------------------------------
# Force non-interactive backend BEFORE any other matplotlib import
# ---------------------------------------------------------------------------
import matplotlib

matplotlib.use("Agg")

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# ---------------------------------------------------------------------------
# Global style & palette constants
# ---------------------------------------------------------------------------
_STYLE = "seaborn-v0_8-darkgrid"
_DPI = 150
_FIG_SIZE = (9, 5)
_FIG_SIZE_WIDE = (10, 5)
_FIG_SIZE_MULTI = (10, 8)

# Professional modern palette
PALETTE = [
    "#00D4AA",  # teal-green
    "#FF6B6B",  # coral-red
    "#4ECDC4",  # turquoise
    "#45B7D1",  # sky-blue
    "#96CEB4",  # sage-green
    "#FFEAA7",  # gold
    "#DDA0DD",  # plum
    "#98D8C8",  # mint
]

# Mapping for quick access
C_PRIMARY = PALETTE[0]
C_SECONDARY = PALETTE[1]
C_TERTIARY = PALETTE[2]
C_ACCENT = PALETTE[3]
C_MUTED = PALETTE[4]
C_GOLD = PALETTE[5]
C_PLUM = PALETTE[6]
C_MINT = PALETTE[7]


def _apply_style() -> None:
    """Apply the global matplotlib style."""
    try:
        plt.style.use(_STYLE)
    except OSError:
        # Fallback if the exact style name is unavailable
        plt.style.use("seaborn-v0_8")


def _ensure_dir(output_dir: str) -> Path:
    """Create the output directory if it does not exist and return as Path."""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save_and_close(fig: plt.Figure, filepath: Path) -> None:
    """Apply tight layout, save figure, and close it to free memory."""
    fig.tight_layout()
    fig.savefig(str(filepath), dpi=_DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [OK] Saved: {filepath.name}")


# =========================================================================
# 1. Monthly Sales Trend
# =========================================================================
def plot_monthly_sales_trend(summary_df: pd.DataFrame, output_dir: str) -> None:
    """Line chart of monthly total sales with a 6-month rolling average overlay.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Must contain *Month* and *Sales* columns.
    output_dir : str
        Directory where the PNG will be written.
    """
    _apply_style()
    out = _ensure_dir(output_dir)

    fig, ax = plt.subplots(figsize=_FIG_SIZE)
    months = summary_df["Month"]
    sales = summary_df["Sales"]

    ax.plot(
        months,
        sales,
        color=C_PRIMARY,
        linewidth=1.8,
        marker="o",
        markersize=4,
        label="Monthly Sales",
        alpha=0.85,
    )

    rolling = sales.rolling(window=6, min_periods=1).mean()
    ax.plot(
        months,
        rolling,
        color=C_SECONDARY,
        linewidth=2.5,
        linestyle="--",
        label="6-Month Rolling Avg",
    )

    ax.fill_between(months, sales, alpha=0.12, color=C_PRIMARY)
    ax.set_title(
        "Monthly Sales Trend with 6-Month Rolling Average", fontsize=15, fontweight="bold", pad=12
    )
    ax.set_xlabel("Month", fontsize=12)
    ax.set_ylabel("Sales (₹ Lakhs)", fontsize=12)
    ax.legend(fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    _save_and_close(fig, out / "monthly_sales_trend.png")


# =========================================================================
# 2. Salesperson Productivity
# =========================================================================
def plot_salesperson_productivity(raw_df: pd.DataFrame, output_dir: str) -> None:
    """Horizontal bar chart showing total sales contribution by each salesperson.

    Parameters
    ----------
    raw_df : pd.DataFrame
        Must contain columns *sale sp1* … *sale sp16*.
    output_dir : str
        Directory where the PNG will be written.
    """
    _apply_style()
    out = _ensure_dir(output_dir)

    sp_cols = [c for c in raw_df.columns if c.strip().startswith("sale sp")]
    totals = raw_df[sp_cols].sum().sort_values(ascending=True)
    # Clean column names for display
    labels = [c.strip().replace("sale ", "SP ").upper() for c in totals.index]

    colours = (PALETTE * 3)[: len(labels)]  # cycle palette

    fig, ax = plt.subplots(figsize=_FIG_SIZE)
    bars = ax.barh(labels, totals.values, color=colours, edgecolor="white", linewidth=0.5)

    # Value annotations
    for bar, val in zip(bars, totals.values, strict=False):
        ax.text(
            val + totals.max() * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"₹{val:,.1f}L",
            va="center",
            fontsize=9,
            color="white",
        )

    ax.set_title("Total Sales Contribution by Salesperson", fontsize=15, fontweight="bold", pad=12)
    ax.set_xlabel("Total Sales (₹ Lakhs)", fontsize=12)
    ax.set_ylabel("Salesperson", fontsize=12)
    ax.grid(True, axis="x", alpha=0.3)

    _save_and_close(fig, out / "salesperson_productivity.png")


# =========================================================================
# 3. Average Sales per Employee
# =========================================================================
def plot_avg_sales_per_employee(summary_df: pd.DataFrame, output_dir: str) -> None:
    """Line chart of *SalesPerEmployee* over months.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Must contain *Month* and *SalesPerEmployee*.
    output_dir : str
        Directory where the PNG will be written.
    """
    _apply_style()
    out = _ensure_dir(output_dir)

    fig, ax = plt.subplots(figsize=_FIG_SIZE)
    months = summary_df["Month"]
    spe = summary_df["SalesPerEmployee"]

    ax.plot(
        months,
        spe,
        color=C_ACCENT,
        linewidth=2,
        marker="s",
        markersize=5,
        label="Sales / Active Employee",
    )
    ax.fill_between(months, spe, alpha=0.10, color=C_ACCENT)

    ax.set_title(
        "Average Sales per Active Employee Over Time", fontsize=15, fontweight="bold", pad=12
    )
    ax.set_xlabel("Month", fontsize=12)
    ax.set_ylabel("Sales per Employee (₹ Lakhs)", fontsize=12)
    ax.legend(fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    _save_and_close(fig, out / "avg_sales_per_employee.png")


# =========================================================================
# 4. Employee Count vs Sales (scatter + regression)
# =========================================================================
def plot_employee_count_vs_sales(summary_df: pd.DataFrame, output_dir: str) -> None:
    """Scatter plot of *ActiveEmployees* vs *Sales* with OLS regression line.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Must contain *ActiveEmployees* and *Sales*.
    output_dir : str
        Directory where the PNG will be written.
    """
    _apply_style()
    out = _ensure_dir(output_dir)

    fig, ax = plt.subplots(figsize=_FIG_SIZE)
    x = summary_df["ActiveEmployees"].values
    y = summary_df["Sales"].values

    ax.scatter(x, y, color=C_PRIMARY, s=70, edgecolors="white", linewidth=0.6, alpha=0.85, zorder=5)

    # Regression
    slope, intercept, r_value, p_value, _ = stats.linregress(x, y)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(
        x_line,
        intercept + slope * x_line,
        color=C_SECONDARY,
        linewidth=2,
        linestyle="--",
        label=f"OLS  R²={r_value**2:.3f}  p={p_value:.2e}",
    )

    ax.set_title("Employee Count vs Total Sales", fontsize=15, fontweight="bold", pad=12)
    ax.set_xlabel("Active Employees", fontsize=12)
    ax.set_ylabel("Sales (₹ Lakhs)", fontsize=12)
    ax.legend(fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    _save_and_close(fig, out / "employee_count_vs_sales.png")


# =========================================================================
# 5. Employee Count vs Gross Margin (scatter + regression)
# =========================================================================
def plot_employee_count_vs_margin(summary_df: pd.DataFrame, output_dir: str) -> None:
    """Scatter plot of *ActiveEmployees* vs *GrossMargin* with OLS regression line.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Must contain *ActiveEmployees* and *GrossMargin*.
    output_dir : str
        Directory where the PNG will be written.
    """
    _apply_style()
    out = _ensure_dir(output_dir)

    fig, ax = plt.subplots(figsize=_FIG_SIZE)
    x = summary_df["ActiveEmployees"].values
    y = summary_df["GrossMargin"].values

    ax.scatter(x, y, color=C_PLUM, s=70, edgecolors="white", linewidth=0.6, alpha=0.85, zorder=5)

    slope, intercept, r_value, p_value, _ = stats.linregress(x, y)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(
        x_line,
        intercept + slope * x_line,
        color=C_GOLD,
        linewidth=2,
        linestyle="--",
        label=f"OLS  R²={r_value**2:.3f}  p={p_value:.2e}",
    )

    ax.set_title("Employee Count vs Gross Margin", fontsize=15, fontweight="bold", pad=12)
    ax.set_xlabel("Active Employees", fontsize=12)
    ax.set_ylabel("Gross Margin (₹ Lakhs)", fontsize=12)
    ax.legend(fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    _save_and_close(fig, out / "employee_count_vs_margin.png")


# =========================================================================
# 6. Gross Margin Trend
# =========================================================================
def plot_gross_margin_trend(summary_df: pd.DataFrame, output_dir: str) -> None:
    """Line chart of *GrossMargin* over months with 6-month rolling average.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Must contain *Month* and *GrossMargin*.
    output_dir : str
        Directory where the PNG will be written.
    """
    _apply_style()
    out = _ensure_dir(output_dir)

    fig, ax = plt.subplots(figsize=_FIG_SIZE)
    months = summary_df["Month"]
    margin = summary_df["GrossMargin"]

    ax.plot(
        months,
        margin,
        color=C_MUTED,
        linewidth=1.8,
        marker="D",
        markersize=4,
        label="Monthly Gross Margin",
        alpha=0.85,
    )

    rolling = margin.rolling(window=6, min_periods=1).mean()
    ax.plot(
        months,
        rolling,
        color=C_SECONDARY,
        linewidth=2.5,
        linestyle="--",
        label="6-Month Rolling Avg",
    )

    ax.fill_between(months, margin, alpha=0.10, color=C_MUTED)

    ax.set_title(
        "Gross Margin Trend with 6-Month Rolling Average", fontsize=15, fontweight="bold", pad=12
    )
    ax.set_xlabel("Month", fontsize=12)
    ax.set_ylabel("Gross Margin (₹ Lakhs)", fontsize=12)
    ax.legend(fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    _save_and_close(fig, out / "gross_margin_trend.png")


# =========================================================================
# 7. Sales Histogram with KDE
# =========================================================================
def plot_sales_histogram(summary_df: pd.DataFrame, output_dir: str) -> None:
    """Histogram of monthly sales with a kernel-density estimate overlay.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Must contain *Sales*.
    output_dir : str
        Directory where the PNG will be written.
    """
    _apply_style()
    out = _ensure_dir(output_dir)

    fig, ax = plt.subplots(figsize=_FIG_SIZE)
    sales = summary_df["Sales"]

    ax.hist(
        sales,
        bins=15,
        color=C_TERTIARY,
        edgecolor="white",
        linewidth=0.8,
        alpha=0.75,
        density=True,
        label="Histogram",
    )

    # KDE overlay
    kde_x = np.linspace(sales.min() - 10, sales.max() + 10, 300)
    kde = stats.gaussian_kde(sales)
    ax.plot(kde_x, kde(kde_x), color=C_SECONDARY, linewidth=2.5, label="KDE")

    ax.set_title("Distribution of Monthly Sales", fontsize=15, fontweight="bold", pad=12)
    ax.set_xlabel("Sales (₹ Lakhs)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.legend(fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    _save_and_close(fig, out / "sales_histogram.png")


# =========================================================================
# 8. Sales Box Plot by Employee Count
# =========================================================================
def plot_sales_boxplot(summary_df: pd.DataFrame, output_dir: str) -> None:
    """Box plot of *Sales* grouped by *ActiveEmployees*.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Must contain *Sales* and *ActiveEmployees*.
    output_dir : str
        Directory where the PNG will be written.
    """
    _apply_style()
    out = _ensure_dir(output_dir)

    fig, ax = plt.subplots(figsize=_FIG_SIZE)

    groups = summary_df.groupby("ActiveEmployees")["Sales"].apply(list)
    labels = [str(int(k)) for k in groups.index]
    data = groups.values

    bp = ax.boxplot(
        data,
        patch_artist=True,
        notch=False,
        medianprops={"color": C_SECONDARY, "linewidth": 2},
        whiskerprops={"color": "gray"},
        capprops={"color": "gray"},
    )
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)

    # Colour boxes
    box_colors = (PALETTE * 5)[: len(data)]
    for patch, colour in zip(bp["boxes"], box_colors, strict=False):
        patch.set_facecolor(colour)
        patch.set_alpha(0.75)

    ax.set_title(
        "Sales Distribution by Active Employee Count", fontsize=15, fontweight="bold", pad=12
    )
    ax.set_xlabel("Active Employee Count", fontsize=12)
    ax.set_ylabel("Sales (₹ Lakhs)", fontsize=12)
    ax.grid(True, axis="y", alpha=0.3)

    _save_and_close(fig, out / "sales_boxplot.png")


# =========================================================================
# 9. Correlation Heatmap
# =========================================================================
def plot_correlation_heatmap(summary_df: pd.DataFrame, output_dir: str) -> None:
    """Heatmap of Pearson correlations among key numeric columns.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Must contain *Sales*, *ActiveEmployees*, *GrossMargin*,
        and *SalesPerEmployee*.
    output_dir : str
        Directory where the PNG will be written.
    """
    _apply_style()
    out = _ensure_dir(output_dir)

    cols = ["Sales", "ActiveEmployees", "GrossMargin", "SalesPerEmployee"]
    corr = summary_df[cols].corr()

    fig, ax = plt.subplots(figsize=(9, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        vmin=-1,
        vmax=1,
        center=0,
        annot=True,
        fmt=".2f",
        linewidths=1,
        square=True,
        ax=ax,
        annot_kws={"fontsize": 12, "fontweight": "bold"},
        cbar_kws={"shrink": 0.8},
    )

    ax.set_title(
        "Correlation Matrix – Key Staffing Metrics", fontsize=15, fontweight="bold", pad=14
    )

    _save_and_close(fig, out / "correlation_heatmap.png")


# =========================================================================
# 10. Monthly Seasonality
# =========================================================================
def plot_monthly_seasonality(summary_df: pd.DataFrame, output_dir: str) -> None:
    """Bar chart of average sales by simulated month-of-year.

    Uses ``month % 12 + 1`` to derive a cyclic 1–12 month indicator.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Must contain *Month* and *Sales*.
    output_dir : str
        Directory where the PNG will be written.
    """
    _apply_style()
    out = _ensure_dir(output_dir)

    df = summary_df.copy()
    df["MonthOfYear"] = (df["Month"] % 12) + 1

    seasonal = df.groupby("MonthOfYear")["Sales"].mean()

    month_labels = [
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

    fig, ax = plt.subplots(figsize=_FIG_SIZE)
    colours = (PALETTE * 2)[:12]
    bars = ax.bar(
        seasonal.index, seasonal.values, color=colours, edgecolor="white", linewidth=0.6, width=0.7
    )

    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(month_labels, fontsize=11)

    # Value labels on bars
    for bar, val in zip(bars, seasonal.values, strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"₹{val:.1f}L",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_title(
        "Average Sales by Month-of-Year (Seasonal Pattern)", fontsize=15, fontweight="bold", pad=12
    )
    ax.set_xlabel("Month of Year", fontsize=12)
    ax.set_ylabel("Average Sales (₹ Lakhs)", fontsize=12)
    ax.grid(True, axis="y", alpha=0.3)

    _save_and_close(fig, out / "monthly_seasonality.png")


# =========================================================================
# 11. Sales Distribution (KDE only)
# =========================================================================
def plot_sales_distribution(summary_df: pd.DataFrame, output_dir: str) -> None:
    """Kernel-density estimate of the sales distribution.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Must contain *Sales*.
    output_dir : str
        Directory where the PNG will be written.
    """
    _apply_style()
    out = _ensure_dir(output_dir)

    fig, ax = plt.subplots(figsize=_FIG_SIZE)
    sales = summary_df["Sales"]

    kde_x = np.linspace(sales.min() - 15, sales.max() + 15, 500)
    kde = stats.gaussian_kde(sales)
    density = kde(kde_x)

    ax.plot(kde_x, density, color=C_PRIMARY, linewidth=2.5)
    ax.fill_between(kde_x, density, alpha=0.20, color=C_PRIMARY)

    # Mark mean & median
    mean_val = sales.mean()
    median_val = sales.median()
    ax.axvline(
        mean_val, color=C_SECONDARY, linestyle="--", linewidth=1.5, label=f"Mean = ₹{mean_val:.1f}L"
    )
    ax.axvline(
        median_val, color=C_GOLD, linestyle=":", linewidth=1.5, label=f"Median = ₹{median_val:.1f}L"
    )

    ax.set_title("Sales Distribution (KDE)", fontsize=15, fontweight="bold", pad=12)
    ax.set_xlabel("Sales (₹ Lakhs)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.legend(fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    _save_and_close(fig, out / "sales_distribution.png")


# =========================================================================
# 12. Employee Count Distribution (Histogram)
# =========================================================================
def plot_employee_count_distribution(summary_df: pd.DataFrame, output_dir: str) -> None:
    """Histogram of *ActiveEmployees* counts across months.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Must contain *ActiveEmployees*.
    output_dir : str
        Directory where the PNG will be written.
    """
    _apply_style()
    out = _ensure_dir(output_dir)

    fig, ax = plt.subplots(figsize=_FIG_SIZE)
    emp = summary_df["ActiveEmployees"]

    unique_counts = sorted(emp.unique())
    bins = np.arange(min(unique_counts) - 0.5, max(unique_counts) + 1.5, 1)

    ax.hist(
        emp, bins=bins, color=C_ACCENT, edgecolor="white", linewidth=0.8, alpha=0.80, rwidth=0.85
    )

    ax.set_title("Distribution of Active Employee Counts", fontsize=15, fontweight="bold", pad=12)
    ax.set_xlabel("Active Employee Count", fontsize=12)
    ax.set_ylabel("Frequency (Months)", fontsize=12)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.grid(True, axis="y", alpha=0.3)

    _save_and_close(fig, out / "employee_count_distribution.png")


# =========================================================================
# 13. Rolling Metrics – Multi-panel
# =========================================================================
def plot_rolling_metrics(summary_df: pd.DataFrame, output_dir: str) -> None:
    """Three-panel figure showing 6-month rolling Sales, GrossMargin,
    and SalesPerEmployee.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Must contain *Month*, *Sales*, *GrossMargin*, *SalesPerEmployee*.
    output_dir : str
        Directory where the PNG will be written.
    """
    _apply_style()
    out = _ensure_dir(output_dir)

    fig, axes = plt.subplots(3, 1, figsize=_FIG_SIZE_MULTI, sharex=True)

    months = summary_df["Month"]
    metrics = [
        ("Sales", "Rolling Sales (₹ Lakhs)", C_PRIMARY),
        ("GrossMargin", "Rolling Gross Margin (₹ Lakhs)", C_MUTED),
        ("SalesPerEmployee", "Rolling Sales / Employee (₹ Lakhs)", C_ACCENT),
    ]

    for ax, (col, ylabel, colour) in zip(axes, metrics, strict=False):
        raw = summary_df[col]
        rolling = raw.rolling(window=6, min_periods=1).mean()

        ax.plot(months, raw, color=colour, linewidth=1.2, alpha=0.45, label="Monthly")
        ax.plot(months, rolling, color=colour, linewidth=2.5, label="6-Mo Rolling Avg")
        ax.fill_between(months, rolling, alpha=0.10, color=colour)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.legend(fontsize=9, loc="upper left", framealpha=0.85)
        ax.grid(True, alpha=0.3)

    axes[0].set_title(
        "Rolling Metrics Overview (6-Month Window)", fontsize=15, fontweight="bold", pad=12
    )
    axes[-1].set_xlabel("Month", fontsize=12)

    _save_and_close(fig, out / "rolling_metrics.png")


# =========================================================================
# 14. Scenario Comparison
# =========================================================================
def plot_scenario_comparison(scenarios_dict: dict[str, float], output_dir: str) -> None:
    """Grouped bar chart comparing total gross margin across staffing scenarios.

    Parameters
    ----------
    scenarios_dict : dict
        Mapping of scenario names to total gross margin values.
        Example: ``{"Baseline": 1200.5, "+1 Employee": 1350.2, ...}``
    output_dir : str
        Directory where the PNG will be written.
    """
    _apply_style()
    out = _ensure_dir(output_dir)

    names = list(scenarios_dict.keys())
    values = list(scenarios_dict.values())
    colours = (PALETTE * 3)[: len(names)]

    fig, ax = plt.subplots(figsize=_FIG_SIZE)
    bars = ax.bar(names, values, color=colours, edgecolor="white", linewidth=0.6, width=0.55)

    # Value annotations
    for bar, val in zip(bars, values, strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(values) * 0.01,
            f"₹{val:,.1f}L",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_title("Scenario Comparison – Total Gross Margin", fontsize=15, fontweight="bold", pad=12)
    ax.set_xlabel("Scenario", fontsize=12)
    ax.set_ylabel("Total Gross Margin (₹ Lakhs)", fontsize=12)
    ax.grid(True, axis="y", alpha=0.3)

    # Rotate labels if many scenarios
    if len(names) > 4:
        plt.xticks(rotation=25, ha="right")

    _save_and_close(fig, out / "scenario_comparison.png")


# =========================================================================
# 15. Time-series decomposition
# =========================================================================
def plot_decomposition(
    decomposition: dict[str, object],
    output_dir: str,
) -> None:
    """Plot the observed series with its trend and seasonal components.

    Parameters
    ----------
    decomposition : dict
        Output of :func:`revinsight.seasonality.decompose_series` with
        ``observed``, ``trend``, ``seasonal`` and ``residual`` keys.
    output_dir : str
        Directory where the PNG will be written.
    """
    _apply_style()
    out = _ensure_dir(output_dir)

    observed = np.asarray(decomposition.get("observed", []), dtype=float)
    trend = np.asarray(decomposition.get("trend", []), dtype=float)
    seasonal = np.asarray(decomposition.get("seasonal", []), dtype=float)
    if observed.size == 0:
        print("  [SKIP] decomposition.png (no data)")
        return

    x = np.arange(1, observed.size + 1)
    fig, axes = plt.subplots(4, 1, figsize=(10, 9), sharex=True)

    axes[0].plot(x, observed, color=C_PRIMARY, linewidth=1.4)
    axes[0].set_title("Observed", fontsize=12, fontweight="bold")

    axes[1].plot(x, trend, color=C_ACCENT, linewidth=1.8)
    axes[1].set_title("Trend", fontsize=12, fontweight="bold")

    axes[2].plot(x, seasonal, color=C_SECONDARY, linewidth=1.4)
    axes[2].axhline(0, color="grey", linewidth=0.8, linestyle="--")
    axes[2].set_title("Seasonal", fontsize=12, fontweight="bold")

    axes[3].plot(
        x, np.asarray(decomposition.get("residual", []), dtype=float), color=C_GOLD, linewidth=1.2
    )
    axes[3].axhline(0, color="grey", linewidth=0.8, linestyle="--")
    axes[3].set_title("Residual", fontsize=12, fontweight="bold")

    for ax in axes:
        ax.grid(True, alpha=0.3)
    axes[3].set_xlabel("Month Number", fontsize=11)

    fig.suptitle(
        f"Time-Series Decomposition ({decomposition.get('method', 'stl').upper()})",
        fontsize=14,
        fontweight="bold",
    )
    _save_and_close(fig, out / "decomposition.png")


# =========================================================================
# 16. Anomaly detection
# =========================================================================
def plot_anomalies(
    summary_df: pd.DataFrame,
    anomalies: list,
    output_dir: str,
) -> None:
    """Highlight detected anomalous months on the sales series.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Monthly summary containing ``Month`` and ``Sales``.
    anomalies : list
        Anomaly records from :func:`revinsight.anomaly.detect_anomalies`.
    output_dir : str
        Directory where the PNG will be written.
    """
    _apply_style()
    out = _ensure_dir(output_dir)

    if "Month" not in summary_df.columns or "Sales" not in summary_df.columns:
        print("  [SKIP] anomalies.png (missing columns)")
        return

    months = summary_df["Month"].to_numpy()
    sales = summary_df["Sales"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=_FIG_SIZE_WIDE)
    ax.plot(months, sales, color=C_PRIMARY, linewidth=1.4, label="Monthly Sales")

    if anomalies:
        ax_m = [a.get("month") for a in anomalies if a.get("month") in set(months)]
        ax_v = [float(sales[list(months).index(m)]) for m in ax_m]
        ax.scatter(
            ax_m,
            ax_v,
            s=90,
            color=C_SECONDARY,
            zorder=5,
            edgecolor="white",
            linewidth=1.2,
            label="Flagged anomaly",
        )

    ax.set_title("Anomaly Detection on Monthly Sales", fontsize=15, fontweight="bold")
    ax.set_xlabel("Month Number", fontsize=12)
    ax.set_ylabel("Sales (Lakhs)", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    _save_and_close(fig, out / "anomalies.png")


# =========================================================================
# 17. Workforce tenure
# =========================================================================
def plot_workforce_tenure(workforce: dict[str, object], output_dir: str) -> None:
    """Plot per-salesperson tenure and revenue contribution.

    Parameters
    ----------
    workforce : dict
        Output of :func:`revinsight.people.analyze_workforce`.
    output_dir : str
        Directory where the PNG will be written.
    """
    _apply_style()
    out = _ensure_dir(output_dir)

    employees = list(workforce.get("employees", []))  # type: ignore[arg-type]
    if not employees:
        print("  [SKIP] workforce_tenure.png (no employees)")
        return

    top = employees[:20][::-1]
    labels = [str(e.get("id", "?")) for e in top]
    tenure = [float(e.get("tenure_months", 0) or 0) for e in top]
    colours = [C_MINT if e.get("still_active") else C_SECONDARY for e in top]

    fig, ax = plt.subplots(figsize=(9, max(5, 0.32 * len(top) + 1.6)))
    ax.barh(labels, tenure, color=colours, edgecolor="white", linewidth=0.6)
    ax.set_title("Salesperson Tenure (teal = currently active)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Months on Roster", fontsize=11)
    ax.grid(True, axis="x", alpha=0.3)
    _save_and_close(fig, out / "workforce_tenure.png")


# =========================================================================
# 18. Headcount optimisation curve
# =========================================================================
def plot_headcount_optimization(curve: list, output_dir: str, horizon: int = 6) -> None:
    """Plot expected margin against target headcount.

    Parameters
    ----------
    curve : list
        Rows produced by :func:`revinsight.staffing_model.optimize_headcount`.
    output_dir : str
        Directory where the PNG will be written.
    horizon : int, optional
        Projection horizon in months, used in the axis label.
    """
    _apply_style()
    out = _ensure_dir(output_dir)

    rows = [r for r in curve if r.get("headcount")]
    if not rows:
        print("  [SKIP] headcount_optimization.png (no curve)")
        return

    headcounts = [int(r["headcount"]) for r in rows]
    margins = [float(r["total_margin"]) for r in rows]
    best_idx = int(np.argmax(margins))

    fig, ax = plt.subplots(figsize=_FIG_SIZE)
    ax.plot(headcounts, margins, color=C_ACCENT, linewidth=2.0, marker="o", markersize=4)
    ax.scatter(
        [headcounts[best_idx]],
        [margins[best_idx]],
        s=140,
        color=C_SECONDARY,
        zorder=5,
        edgecolor="white",
        linewidth=1.4,
        label=f"Optimal: {headcounts[best_idx]}",
    )
    ax.set_title("Expected Gross Margin vs Target Headcount", fontsize=15, fontweight="bold")
    ax.set_xlabel("Target Headcount", fontsize=12)
    ax.set_ylabel(f"{horizon}-Month Gross Margin (Lakhs)", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    _save_and_close(fig, out / "headcount_optimization.png")


# =========================================================================
# 19. Cumulative scenario margin
# =========================================================================
def plot_scenario_cumulative(scenarios: dict[str, object], output_dir: str) -> None:
    """Plot the cumulative gross margin path of every staffing scenario.

    Parameters
    ----------
    scenarios : dict
        Mapping of scenario key -> scenario dict with ``monthly_breakdown``.
    output_dir : str
        Directory where the PNG will be written.
    """
    _apply_style()
    out = _ensure_dir(output_dir)

    if not scenarios:
        print("  [SKIP] scenario_cumulative.png (no scenarios)")
        return

    fig, ax = plt.subplots(figsize=_FIG_SIZE_WIDE)
    colours = PALETTE * 3
    plotted = 0

    for idx, (key, scenario) in enumerate(scenarios.items()):
        breakdown = scenario.get("monthly_breakdown") or scenario.get("monthly") or []
        if not breakdown:
            continue
        months = [m.get("month") for m in breakdown]
        cumulative = np.cumsum([float(m.get("gross_margin", 0)) for m in breakdown])
        ax.plot(
            months,
            cumulative,
            marker="o",
            markersize=4,
            linewidth=2.0,
            color=colours[idx % len(colours)],
            label=f"{key} - {scenario.get('name', '')}",
        )
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        print("  [SKIP] scenario_cumulative.png (no breakdown)")
        return

    ax.set_title("Cumulative Gross Margin by Staffing Scenario", fontsize=15, fontweight="bold")
    ax.set_xlabel("Forecast Month", fontsize=12)
    ax.set_ylabel("Cumulative Gross Margin (Lakhs)", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    _save_and_close(fig, out / "scenario_cumulative.png")


# =========================================================================
# 20. Tornado sensitivity
# =========================================================================
def plot_tornado(tornado: dict[str, object], output_dir: str) -> None:
    """Plot a tornado chart of the margin sensitivity to key assumptions.

    Parameters
    ----------
    tornado : dict
        Output of :func:`revinsight.sensitivity.tornado_analysis`.
    output_dir : str
        Directory where the PNG will be written.
    """
    _apply_style()
    out = _ensure_dir(output_dir)

    variables = list(tornado.get("variables", []))  # type: ignore[arg-type]
    if not variables:
        print("  [SKIP] tornado.png (no variables)")
        return

    names = [str(v["variable"]) for v in variables][::-1]
    lows = [float(v["low_margin"]) for v in variables][::-1]
    highs = [float(v["high_margin"]) for v in variables][::-1]
    base = float(tornado.get("base_margin", 0.0))

    fig, ax = plt.subplots(figsize=_FIG_SIZE_WIDE)
    y = np.arange(len(names))
    ax.barh(
        y,
        np.array(highs) - base,
        left=base,
        color=C_PRIMARY,
        edgecolor="white",
        linewidth=0.6,
        label="Upside",
    )
    ax.barh(
        y,
        np.array(lows) - base,
        left=base,
        color=C_SECONDARY,
        edgecolor="white",
        linewidth=0.6,
        label="Downside",
    )
    ax.axvline(base, color="#2c3e50", linewidth=1.4, linestyle="--", label=f"Base: {base:,.1f} L")

    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_title("Margin Sensitivity (Tornado)", fontsize=15, fontweight="bold")
    ax.set_xlabel("Monthly Gross Margin (Lakhs)", fontsize=12)
    ax.grid(True, axis="x", alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    _save_and_close(fig, out / "tornado.png")


# =========================================================================
# Master orchestrator
# =========================================================================
def generate_all_charts(
    summary_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    output_dir: str,
    scenarios_dict: dict[str, float] | None = None,
    extra_charts: dict[str, object] | None = None,
) -> dict[str, str]:
    """Generate **all** RevInsight charts and save them to *output_dir*.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Aggregated monthly data with columns:
        ``Month, Sales, ActiveEmployees, SalesPerEmployee, GrossMargin``.
    raw_df : pd.DataFrame
        Original CSV data with individual salesperson columns
        (``sale sp1`` ... ``sale sp16``).
    output_dir : str
        Destination folder for PNG files (created if absent).
    scenarios_dict : dict, optional
        Mapping ``{scenario_name: total_gross_margin}`` used for the
        scenario comparison chart.  Skipped when *None*.
    extra_charts : dict, optional
        Mapping of chart key -> zero-argument callable that renders an
        additional chart.  Each callable receives the output directory.
        Failures are logged and never abort the chart run.

    Returns
    -------
    dict
        Dictionary mapping chart names to their file paths.
    """
    print(f"\n{'='*60}")
    print("  RevInsight -- Generating Publication-Quality Charts")
    print(f"{'='*60}")
    print(f"  Output directory: {output_dir}\n")

    chart_paths: dict[str, str] = {}
    out = _ensure_dir(output_dir)

    # 1  -- Monthly sales trend
    plot_monthly_sales_trend(summary_df, output_dir)
    chart_paths["monthly_sales_trend"] = str(out / "monthly_sales_trend.png")

    # 2  -- Salesperson productivity
    plot_salesperson_productivity(raw_df, output_dir)
    chart_paths["salesperson_productivity"] = str(out / "salesperson_productivity.png")

    # 3  -- Avg sales per employee
    plot_avg_sales_per_employee(summary_df, output_dir)
    chart_paths["avg_sales_per_employee"] = str(out / "avg_sales_per_employee.png")

    # 4  -- Employee count vs sales
    plot_employee_count_vs_sales(summary_df, output_dir)
    chart_paths["employee_count_vs_sales"] = str(out / "employee_count_vs_sales.png")

    # 5  -- Employee count vs margin
    plot_employee_count_vs_margin(summary_df, output_dir)
    chart_paths["employee_count_vs_margin"] = str(out / "employee_count_vs_margin.png")

    # 6  -- Gross margin trend
    plot_gross_margin_trend(summary_df, output_dir)
    chart_paths["gross_margin_trend"] = str(out / "gross_margin_trend.png")

    # 7  -- Sales histogram
    plot_sales_histogram(summary_df, output_dir)
    chart_paths["sales_histogram"] = str(out / "sales_histogram.png")

    # 8  -- Sales box plot
    plot_sales_boxplot(summary_df, output_dir)
    chart_paths["sales_boxplot"] = str(out / "sales_boxplot.png")

    # 9  -- Correlation heatmap
    plot_correlation_heatmap(summary_df, output_dir)
    chart_paths["correlation_heatmap"] = str(out / "correlation_heatmap.png")

    # 10 -- Monthly seasonality
    plot_monthly_seasonality(summary_df, output_dir)
    chart_paths["monthly_seasonality"] = str(out / "monthly_seasonality.png")

    # 11 -- Sales distribution (KDE)
    plot_sales_distribution(summary_df, output_dir)
    chart_paths["sales_distribution"] = str(out / "sales_distribution.png")

    # 12 -- Employee count distribution
    plot_employee_count_distribution(summary_df, output_dir)
    chart_paths["employee_count_distribution"] = str(out / "employee_count_distribution.png")

    # 13 -- Rolling metrics (multi-panel)
    plot_rolling_metrics(summary_df, output_dir)
    chart_paths["rolling_metrics"] = str(out / "rolling_metrics.png")

    # 14 -- Scenario comparison (optional)
    if scenarios_dict is not None:
        plot_scenario_comparison(scenarios_dict, output_dir)
        chart_paths["scenario_comparison"] = str(out / "scenario_comparison.png")
    else:
        print("  [SKIP] scenario_comparison.png (no scenarios_dict)")

    # 15+ -- Optional charts supplied by the pipeline
    if extra_charts:
        for key, renderer in extra_charts.items():
            try:
                renderer(output_dir)
                chart_paths[key] = str(out / f"{key}.png")
            except Exception as exc:  # noqa: BLE001 - a chart must never break the run
                print(f"  [SKIP] {key}.png ({exc})")

    print(f"\n{'='*60}")
    print("  [OK] All charts generated successfully.")
    print(f"{'='*60}\n")

    return chart_paths
