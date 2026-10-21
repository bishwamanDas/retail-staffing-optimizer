"""Tests for the configuration and utility layers."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from store_staffing.config import AnalysisConfig, config_from_env
from store_staffing.utils import (
    calculate_break_even_sales,
    calculate_gross_margin,
    count_active_employees,
    format_currency,
    get_salesperson_columns,
    infer_month_of_year,
    load_and_validate_data,
    month_name,
    prepare_monthly_summary,
    round_floats,
    safe_div,
    setup_output_dirs,
    to_jsonable,
    compute_file_sha256,
)


# --------------------------------------------------------------------------- #
# Configuration                                                                #
# --------------------------------------------------------------------------- #
class TestAnalysisConfig:
    """Behaviour of :class:`store_staffing.config.AnalysisConfig`."""

    def test_defaults_are_valid(self) -> None:
        config = AnalysisConfig()
        config.validate()
        assert config.salary_per_employee == 3.0
        assert config.commission_rate == 0.05
        assert config.forecast_months == 6

    def test_effective_margin_rate(self) -> None:
        config = AnalysisConfig(commission_rate=0.10, cogs_rate=0.50)
        assert config.effective_margin_rate == pytest.approx(0.40)

    def test_break_even_sales(self) -> None:
        config = AnalysisConfig()
        assert config.break_even_sales_per_employee == pytest.approx(3.0 / 0.45)

    def test_replace_ignores_none(self) -> None:
        config = AnalysisConfig().replace(salary_per_employee=None, commission_rate=0.08)
        assert config.commission_rate == 0.08
        assert config.salary_per_employee == 3.0

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"salary_per_employee": 0},
            {"salary_per_employee": -1},
            {"commission_rate": 1.5},
            {"cogs_rate": 0.9, "commission_rate": 0.2},
            {"forecast_months": 0},
            {"validation_months": 1},
            {"backtest_origins": 0},
            {"confidence_level": 1.5},
            {"n_simulations": 10},
            {"min_headcount": 0},
            {"max_headcount": 0, "min_headcount": 2},
        ],
    )
    def test_invalid_configs_raise(self, kwargs: dict) -> None:
        with pytest.raises(ValueError):
            AnalysisConfig(**kwargs).validate()

    def test_to_dict_is_jsonable(self) -> None:
        payload = AnalysisConfig().to_dict()
        assert isinstance(payload["new_hire_ramp"], list)
        assert payload["effective_margin_rate"] == pytest.approx(0.45)

    def test_config_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SSA_SALARY", "4.25")
        monkeypatch.setenv("SSA_COMMISSION", "0.08")
        config = config_from_env()
        assert config.salary_per_employee == 4.25
        assert config.commission_rate == 0.08

    def test_config_from_env_ignores_garbage(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SSA_SALARY", "not-a-number")
        config = config_from_env()
        assert config.salary_per_employee == 3.0


# --------------------------------------------------------------------------- #
# Data loading                                                                 #
# --------------------------------------------------------------------------- #
class TestDataLoading:
    """Behaviour of :func:`store_staffing.utils.load_and_validate_data`."""

    def test_loads_sample_dataset(self, raw_df: pd.DataFrame) -> None:
        assert len(raw_df) == 81
        assert "Month" in raw_df.columns
        assert "Sales" in raw_df.columns
        assert raw_df["Month"].is_monotonic_increasing
        assert not raw_df.isna().any().any()

    def test_raises_for_missing_file(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError):
            load_and_validate_data(tmp_path / "nope.csv")

    def test_raises_without_salesperson_columns(self, tmp_path) -> None:
        path = tmp_path / "bad.csv"
        path.write_text("Month,Sales\n1,100\n2,110\n", encoding="utf-8")
        with pytest.raises(ValueError, match="salesperson"):
            load_and_validate_data(path)

    def test_raises_without_core_columns(self, tmp_path) -> None:
        path = tmp_path / "bad.csv"
        path.write_text("Month,sale sp1\n1,100\n", encoding="utf-8")
        with pytest.raises(ValueError, match="missing"):
            load_and_validate_data(path)

    def test_fills_missing_salesperson_values(self, tmp_path) -> None:
        path = tmp_path / "gaps.csv"
        path.write_text("Month,Sales,sale sp1,sale sp2\n1,100,60,\n2,110,55,55\n", encoding="utf-8")
        df = load_and_validate_data(path)
        assert df.loc[0, "sale sp2"] == 0.0

    def test_drops_rows_with_missing_sales(self, tmp_path) -> None:
        path = tmp_path / "gaps.csv"
        path.write_text(
            "Month,Sales,sale sp1,sale sp2\n1,100,60,40\n2,,55,55\n3,120,60,60\n",
            encoding="utf-8",
        )
        df = load_and_validate_data(path)
        assert len(df) == 2

    def test_sorts_and_deduplicates(self, tmp_path) -> None:
        path = tmp_path / "dupes.csv"
        path.write_text(
            "Month,Sales,sale sp1,sale sp2\n2,110,55,55\n1,100,60,40\n1,101,60,41\n",
            encoding="utf-8",
        )
        df = load_and_validate_data(path)
        assert list(df["Month"]) == [1, 2]
        assert df.loc[0, "Sales"] == 100.0

    def test_handles_arbitrary_team_sizes(self, tmp_path) -> None:
        path = tmp_path / "team.csv"
        path.write_text(
            "Month,Sales,sale sp1,sale sp2,sale sp3\n1,100,40,35,25\n", encoding="utf-8"
        )
        df = load_and_validate_data(path)
        assert len(get_salesperson_columns(df)) == 3


# --------------------------------------------------------------------------- #
# Calculations                                                                 #
# --------------------------------------------------------------------------- #
class TestCalculations:
    """Behaviour of the financial and aggregation helpers."""

    def test_count_active_employees(self) -> None:
        row = pd.Series({"sale sp1": 5.0, "sale sp2": 0.0, "sale sp3": 3.2})
        assert count_active_employees(row) == 2

    def test_count_active_ignores_returns(self) -> None:
        row = pd.Series({"sale sp1": -2.0, "sale sp2": 4.0})
        assert count_active_employees(row) == 1

    def test_gross_margin_formula(self) -> None:
        assert calculate_gross_margin(100, 7) == pytest.approx(24.0)
        assert calculate_gross_margin(100, 7, 3.0, 0.05) == pytest.approx(24.0)

    def test_gross_margin_respects_cogs(self) -> None:
        assert calculate_gross_margin(100, 7, 3.0, 0.05, 0.60) == pytest.approx(14.0)

    def test_break_even_sales(self) -> None:
        assert calculate_break_even_sales() == pytest.approx(6.666666, rel=1e-5)

    def test_break_even_raises_on_zero_margin(self) -> None:
        with pytest.raises(ZeroDivisionError):
            calculate_break_even_sales(3.0, 0.2, 0.8)

    def test_infer_month_of_year(self) -> None:
        assert infer_month_of_year(1) == 1
        assert infer_month_of_year(12) == 12
        assert infer_month_of_year(13) == 1
        assert infer_month_of_year(3, start_month=4) == 6

    def test_infer_month_of_year_rejects_bad_start(self) -> None:
        with pytest.raises(ValueError):
            infer_month_of_year(1, start_month=13)

    def test_month_name(self) -> None:
        assert month_name(1) == "Jan"
        assert month_name(12) == "Dec"
        with pytest.raises(ValueError):
            month_name(0)

    def test_format_currency(self) -> None:
        assert format_currency(12.3456) == "Rs.12.35 Lakhs"


# --------------------------------------------------------------------------- #
# Monthly summary                                                              #
# --------------------------------------------------------------------------- #
class TestMonthlySummary:
    """Behaviour of :func:`store_staffing.utils.prepare_monthly_summary`."""

    def test_columns_present(self, summary_df: pd.DataFrame) -> None:
        for column in (
            "Month",
            "MonthOfYear",
            "Sales",
            "ActiveEmployees",
            "SalesPerEmployee",
            "GrossMargin",
            "Rolling3Sales",
            "Rolling3SalesPerEmp",
        ):
            assert column in summary_df.columns

    def test_row_count_matches_source(self, raw_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
        assert len(summary_df) == len(raw_df)

    def test_active_employees_matches_manual_count(
        self, raw_df: pd.DataFrame, summary_df: pd.DataFrame
    ) -> None:
        for idx in (0, len(raw_df) - 1):
            sp_cols = [c for c in raw_df.columns if c.startswith("sale sp")]
            expected = int((raw_df.iloc[idx][sp_cols] > 0).sum())
            assert summary_df.iloc[idx]["ActiveEmployees"] == expected

    def test_gross_margin_matches_helper(self, summary_df: pd.DataFrame) -> None:
        for idx in (0, len(summary_df) - 1):
            row = summary_df.iloc[idx]
            expected = calculate_gross_margin(row["Sales"], row["ActiveEmployees"])
            assert row["GrossMargin"] == pytest.approx(expected)

    def test_sales_per_employee_is_nan_when_no_staff(self) -> None:
        df = pd.DataFrame({"Month": [1], "Sales": [100.0], "sale sp1": [0.0]})
        summary = prepare_monthly_summary(df)
        assert math.isnan(summary.loc[0, "SalesPerEmployee"])

    def test_month_of_year_wraps(self) -> None:
        df = pd.DataFrame(
            {"Month": [1, 13, 25], "Sales": [1.0, 2.0, 3.0], "sale sp1": [1.0, 2.0, 3.0]}
        )
        summary = prepare_monthly_summary(df)
        assert list(summary["MonthOfYear"]) == [1, 1, 1]


# --------------------------------------------------------------------------- #
# Output helpers                                                               #
# --------------------------------------------------------------------------- #
class TestOutputHelpers:
    """Behaviour of the directory and serialisation helpers."""

    def test_setup_output_dirs_creates_tree(self, tmp_path) -> None:
        dirs = setup_output_dirs(tmp_path / "out")
        assert (tmp_path / "out").exists()
        assert (tmp_path / "out" / "charts").exists()
        assert (tmp_path / "out" / "reports").exists()
        assert set(dirs) == {"output", "charts", "reports"}

    def test_setup_output_dirs_without_reports(self, tmp_path) -> None:
        dirs = setup_output_dirs(tmp_path / "out", with_reports=False)
        assert "reports" not in dirs

    def test_to_jsonable_handles_numpy(self) -> None:
        payload = {
            "int": np.int64(3),
            "float": np.float64(1.5),
            "bool": np.bool_(True),
            "array": np.array([1.0, 2.0]),
            "nan": float("nan"),
            "inf": float("inf"),
        }
        result = to_jsonable(payload)
        assert result["int"] == 3
        assert result["float"] == 1.5
        assert result["bool"] is True
        assert result["array"] == [1.0, 2.0]
        assert result["nan"] is None
        assert result["inf"] is None

    def test_to_jsonable_handles_dataframe(self) -> None:
        df = pd.DataFrame({"a": [1, 2], "b": [3.0, 4.0]})
        result = to_jsonable(df)
        assert result == [{"a": 1, "b": 3.0}, {"a": 2, "b": 4.0}]

    def test_round_floats(self) -> None:
        assert round_floats({"x": 1.23456789, "y": [2.987654]}, 3) == {
            "x": 1.235,
            "y": [2.988],
        }

    def test_safe_div(self) -> None:
        assert safe_div(10, 4) == 2.5
        assert safe_div(10, 0) == 0.0
        assert safe_div(10, 0, default=-1) == -1.0

    def test_compute_file_sha256(self, sample_csv) -> None:
        digest = compute_file_sha256(sample_csv)
        assert isinstance(digest, str)
        assert len(digest) == 64
        assert digest == compute_file_sha256(sample_csv)

