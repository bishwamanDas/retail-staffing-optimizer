"""Edge case boundary tests for financial formulas."""

import pytest

from store_staffing.financial_analysis import (
    calculate_break_even_analysis,
    calculate_payback_period,
    calculate_roi,
    compute_gross_margin,
)


def test_gross_margin_zero_sales():
    """With zero sales, margin should strictly equal negative salary cost."""
    margin = compute_gross_margin(
        sales=0.0,
        employees=5,
        salary_per_person=3.0,
        commission_rate=0.05,
        cogs_rate=0.50,
    )
    assert margin == -15.0


def test_gross_margin_zero_employees():
    """With zero employees, margin is net revenue without labor cost."""
    margin = compute_gross_margin(
        sales=100.0,
        employees=0,
        salary_per_person=3.0,
        commission_rate=0.05,
        cogs_rate=0.50,
    )
    assert margin == pytest.approx(45.0)


def test_break_even_with_custom_commission():
    """Verify break-even calculation respects varied commission tiers."""
    be = calculate_break_even_analysis(
        salary_per_person=3.0,
        commission_rate=0.10,
        cogs_rate=0.50,
    )
    # net margin rate = 1 - 0.50 - 0.10 = 0.40
    # break-even = 3.0 / 0.40 = 7.5
    assert be["break_even_sales_per_employee"] == pytest.approx(7.5)


def test_payback_period_non_covering_sales():
    """When sales never cover costs, ever_pays_back should be False."""
    payback = calculate_payback_period(
        incremental_sales_by_month=[1.0, 1.0, 1.0],
        salary_per_person=3.0,
        commission_rate=0.05,
        cogs_rate=0.50,
    )
    assert payback["ever_pays_back"] is False
    assert payback["months_to_payback"] is None


def test_roi_calculation():
    """Verify ROI formula consistency."""
    roi = calculate_roi(incremental_sales_total=27.0, total_cost=18.0)
    assert roi["roi_pct"] == pytest.approx(50.0)
    assert roi["net_benefit"] == pytest.approx(9.0)
