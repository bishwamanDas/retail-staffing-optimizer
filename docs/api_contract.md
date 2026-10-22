# RevInsight Analytics API — Contract v2.0

Base URL: `/api/v1` · Content type: `application/json` unless stated.

All monetary values are **Lakhs INR**. All floats are rounded to 4 dp.

---

## `GET /api/v1/health`

```json
{ "status": "ok", "version": "2.0.0", "uptime_seconds": 12.34, "runs_cached": 3 }
```

---

## `GET /api/v1/datasets`

```json
{ "datasets": [
  { "id": "RevI-Test", "label": "RevI-Test", "filename": "RevI-Test.csv",
    "rows": 81, "columns": 18, "months": 81, "salespersons": 16,
    "description": "Primary sample dataset" }
] }
```

---

## `POST /api/v1/analyses`

`multipart/form-data`

| field | type | default | notes |
|---|---|---|---|
| `dataset` | string | `RevI-Test` | dataset id from `/datasets` |
| `file` | file | – | optional CSV upload (wins over `dataset`) |
| `salary` | float | `3.0` | monthly salary / employee (Lakhs) |
| `commission` | float | `0.05` | commission fraction of sales |
| `forecast_months` | int | `6` | forecast horizon |
| `start_month` | int | `1` | calendar month that dataset month 1 = |

Returns **201** with the Analysis payload (below). **422** on invalid input.

---

## `GET /api/v1/analyses`

```json
{ "runs": [
  { "run_id": "0f3c...", "dataset": "RevI-Test", "created_at": "2026-10-22T11:13:50Z",
    "recommendation": "Hire Immediately (8 from month 1)", "confidence": "MEDIUM" }
] }
```

---

## `GET /api/v1/analyses/{run_id}` → Analysis payload · **404** if unknown.

## `DELETE /api/v1/analyses/{run_id}` → `{ "deleted": true }`

## `GET /api/v1/analyses/{run_id}/report.pdf` → `application/pdf`
## `GET /api/v1/analyses/{run_id}/report.html` → `text/html`
## `GET /api/v1/analyses/{run_id}/results.csv` → `text/csv`
## `GET /api/v1/analyses/{run_id}/charts/{chart_name}.png` → `image/png`

`{chart_name}` is one of the keys of `charts` in the Analysis payload.

---

## `POST /api/v1/simulate`

```json
{ "run_id": "0f3c...", "salary": 3.0, "commission": 0.05, "horizon": 12,
  "decision": "auto", "sales_shock_pct": 0.0 }
```

`decision` ∈ `auto` | `hold` | `hire` | `conditional`.

```json
{ "run_id": "0f3c...", "assumptions": { "salary": 3.0, "commission": 0.05, "horizon": 12, "decision": "auto", "sales_shock_pct": 0.0 },
  "scenarios": {
    "A": { "key": "A", "name": "Don't Hire (Stay at 7)", "total_sales": 0.0,
           "total_gross_margin": 0.0, "total_staff_cost": 0.0, "avg_monthly_margin": 0.0,
           "delta_vs_best": 0.0,
           "monthly": [ { "month": 1, "employees": 7, "sales": 0.0,
             "gross_margin": 0.0, "staff_cost": 0.0 } ],
           "risk": "", "advantages": [], "disadvantages": [],
           "p10_margin": 0.0, "p50_margin": 0.0, "p90_margin": 0.0,
           "margin_std": 0.0, "prob_positive": 0.0,
           "prob_better_than_baseline": 0.0, "in_scope": true },
    "B": {}, "C": {}, "D": {} },
  "best": "B",
  "break_even_sales_per_employee": 6.67,
  "payback_months": 2.4,
  "note": "..." }
```

---

## `GET /api/v1/analyses/{run_id}/sensitivity`

```json
{ "tornado": [ { "variable": "Monthly salary", "low": 2.5, "high": 3.5,
                 "low_margin": 0.0, "high_margin": 0.0, "swing": 0.0 } ],
  "monte_carlo": { "p10": 0.0, "p50": 0.0, "p90": 0.0, "mean": 0.0, "std": 0.0,
                   "prob_positive": 0.0, "prob_better_than_hold": 0.0 },
  "grid": [ { "salary": 2.0, "commission_rate": 0.03, "effective_margin_rate": 0.47,
              "gross_margin": 0.0, "break_even_sales_per_emp": 0.0,
              "margin_vs_base": 0.0, "recommended_action": "..." } ] }
```

---

# Analysis payload

```jsonc
{
  "schema_version": "2.0",
  "run_id": "0f3c...",
  "created_at": "2026-10-22T11:13:50Z",
  "dataset": { "name": "RevI-Test", "filename": "RevI-Test.csv", "rows": 81, "columns": 18,
               "months": 81, "salespersons": 16, "total_sales": 8375.8,
               "first_month": 1, "last_month": 81, "completeness": 1.0 },
  "assumptions": { "salary": 3.0, "commission": 0.05, "cogs_rate": 0.5,
                   "forecast_months": 6, "confidence_level": 0.95, "start_month": 1 },

  "executive": {
    "recommendation": "Hire Immediately (8 from month 1)",
    "recommended_scenario": "B",
    "confidence": "MEDIUM",
    "runner_up": "Scenario D: Conditional Hire (trigger >= 116.18 L sales)",
    "headline": "...",
    "reasoning": ["bullet", "bullet"],
    "confidence_drivers": { "sales_consistency_cv": 0.25, "historical_months": 81,
                             "headcount_observations": 4,
                             "margin_gap_to_runner_up": 0.0, "note": "" },
    "expected_impact": { "total_sales_6m": 0.0, "total_gross_margin_6m": 0.0,
                         "total_staff_cost_6m": 0.0, "avg_monthly_margin": 0.0,
                         "margin_advantage_over_runner_up": 0.0,
                         "margin_advantage_vs_do_nothing": 0.0 }
  },

  "kpis": {
    "current_employees": 7, "optimal_headcount": 7,
    "avg_monthly_sales": 0.0, "avg_monthly_margin": 0.0, "total_margin": 0.0,
    "best_margin": 0.0, "worst_margin": 0.0, "sales_volatility": 0.0,
    "break_even_sales_per_employee": 0.0, "avg_sales_per_employee": 0.0,
    "productivity_cv": 0.0, "forecast_avg_sales": 0.0, "forecast_avg_margin": 0.0,
    "best_forecast_model": "Exponential Smoothing", "validation_rmse": 0.0,
    "validation_mape": 0.0, "validation_mae": 0.0, "validation_smape": 0.0,
    "validation_r2": 0.0, "diminishing_returns": true,
    "staff_cost_6m": 0.0, "payback_months": 0.0, "roi_pct": 0.0,
    "anomalies_detected": 4, "data_completeness": 1.0, "confidence_level": "MEDIUM",
    "seasonality_strength": 0.86, "detected_cycle_months": 6, "total_leavers": 9,
    "marginal_sales_per_employee": 11.25, "marginal_margin_contribution": 2.06,
    "marginal_model_r2": 0.89
  },

  "series": {
    "months": [1], "month_of_year": [1], "sales": [], "employees": [],
    "sales_per_employee": [], "gross_margin": [], "rolling_sales": [],
    "rolling_sales_per_employee": [], "trend": [], "seasonal": [], "residual": [],
    "seasonal_index": [ { "month": 1, "name": "Jan", "index": 1.02, "strength": 0.31,
                          "avg_sales": 0.0, "months_observed": 0 } ],
    "anomalies": [ { "month": 12, "name": "Dec", "sales": 0.0, "expected": 0.0,
                     "score": 0.0, "direction": "spike", "severity": "high" } ],
    "headcount_timeline": [ { "month": 1, "employees": 3, "joined": ["sp1"], "left": [] } ]
  },

  "forecast": {
    "model": "Exponential Smoothing",
    "model_candidates": 7,
    "months": [82], "values": [], "lower_95": [], "upper_95": [],
    "ensemble_members": [ { "model": "Exponential Smoothing", "weight": 0.62, "rmse": 0.0 } ],
    "metrics": [ { "model": "Exponential Smoothing", "rmse": 0.0, "mae": 0.0,
                   "mape": 0.0, "mase": 0.0, "r2": 0.0, "rank": 1 } ],
    "backtest": [ { "model": "Ensemble (Top 3)", "origin_month": 70,
                    "actual": 0.0, "predicted": 0.0, "error": 0.0,
                    "actual_path": [], "predicted_path": [], "steps": 6 } ],
    "residual_std": 0.0,
    "methodology": "Rolling-origin cross-validation (4 origins, 12-month horizon)"
  },

  "scenarios": {
    "A": { "key": "A", "name": "Don't Hire (Stay at 7)", "total_sales": 0.0,
           "total_gross_margin": 0.0, "total_staff_cost": 0.0, "avg_monthly_margin": 0.0,
           "delta_vs_best": 0.0, "delta_vs_baseline": 0.0,
           "monthly": [ { "month": 1, "employees": 7, "sales": 0.0, "gross_margin": 0.0, "staff_cost": 0.0 } ],
           "risk": "", "advantages": [], "disadvantages": [],
           "margin_std": 0.0, "p10_margin": 0.0, "p50_margin": 0.0, "p90_margin": 0.0,
           "prob_positive": 0.0, "prob_better_than_baseline": 0.0, "in_scope": true,
           "threshold": 0.0, "triggered": false, "trigger_month": null },
    "B": {}, "C": {}, "D": {}
  },

  "productivity": {
    "by_headcount": [ { "employees": 3, "avg_sales": 0.0, "avg_sales_per_employee": 0.0,
                        "avg_gross_margin": 0.0, "months_observed": 0,
                        "marginal_sales_per_added": 0.0 } ],
    "diminishing_detected": true,
    "marginal_changes": [ { "from_employees": 3, "to_employees": 4,
                            "marginal_sales_per_added": 0.0,
                            "change_in_sales_per_emp": 0.0, "diminishing": false } ],
    "employees": [ { "id": "sp1", "total_sales": 0.0, "avg_monthly_sales": 0.0,
                     "months_active": 0, "first_month": 1, "last_month": 81,
                     "still_active": true, "avg_sales_when_active": 0.0,
                     "peak_month_sales": 0.0, "share_of_sales": 0.0 } ],
    "attrition": { "total_leavers": 9, "avg_tenure_months": 0.0, "recent_12m_leavers": 2,
                   "timeline": [ { "month": 45, "joined": ["sp7"], "left": ["sp3"] } ] },
    "headcount_curve": [ { "headcount": 7, "total_margin": 0.0,
                           "avg_monthly_margin": 0.0, "margin_per_employee": 0.0 } ],
    "optimal_headcount": 8,
    "headcount_optimisation": {
      "optimal_headcount": 8, "optimal_total_margin": 0.0,
      "marginal_sales_per_employee": 0.0, "marginal_source": "first_difference",
      "interpretation": "", "search_bounds": [3, 9], "note": ""
    },
    "marginal_productivity": {
      "marginal_sales_per_employee": 0.0, "marginal_margin_contribution": 0.0,
      "p_value": 0.0, "r_squared": 0.0, "n_observations": 81,
      "model": "first_difference", "observed_marginal_step": 0.0, "note": ""
    }
  },

  "sensitivity": {
    "tornado": [ { "variable": "Monthly salary", "low": 2.5, "high": 3.5,
                   "low_margin": 0.0, "high_margin": 0.0, "swing": 0.0,
                   "delta_vs_base_low": 0.0, "delta_vs_base_high": 0.0 } ],
    "tornado_base_margin": 0.0,
    "monte_carlo": { "p10": 0.0, "p50": 0.0, "p90": 0.0, "mean": 0.0, "std": 0.0,
                     "prob_positive": 0.0, "prob_better_than_hold": 0.0,
                     "delta_vs_do_nothing": 0.0, "n_simulations": 5000,
                     "decision_stability": { "leader": "B", "leader_label": "",
                       "margin_over_runner_up": 0.0, "decisive": true, "note": "" },
                     "per_scenario": { "A": {}, "B": {} } },
    "grid": [ { "salary": 2.0, "commission_rate": 0.03, "effective_margin_rate": 0.47,
                "gross_margin": 0.0, "break_even_sales_per_emp": 0.0,
                "margin_vs_base": 0.0, "recommended_action": "Healthy margin" } ]
  },

  "payback": { "months_to_payback": 2.4, "investment": 3.0, "roi_pct": 42.0,
               "net_benefit": 0.0, "monthly_contribution": [],
               "cumulative_contribution": [], "cumulative_margin_delta": [] },

  "optimal_rule": { "optimal_employee_count": 7, "rule_description": "",
                    "rolling_avg_trigger": {}, "threshold_rule": {} },
  "rule_backtest": { "always_replace": {}, "adaptive": {}, "improvement": 0.0,
                     "improvement_pct": 0.0, "avoided_overstaffing_pct": 0.0,
                     "avoided_overstaffing_months": 0, "months_compared": 81,
                     "interpretation": "" },
  "vacancy_cost": { "open_positions": 1, "marginal_sales_per_employee": 0.0,
                    "estimated_lost_sales_per_month": 0.0,
                    "estimated_margin_cost_per_month": 0.0,
                    "monthly_salary_saving": 0.0, "net_monthly_impact": 0.0,
                    "annualised_cost": 0.0, "confidence": "HIGH",
                    "interpretation": "" },

  "charts": { "monthly_sales_trend": "monthly_sales_trend" },
  "artifacts": { "pdf_url": "", "html_url": "", "csv_url": "", "charts_url": "" },
  "recommendation_text": "..."
}
```

## Error shape

```json
{ "detail": "human readable message", "code": "invalid_input" }
```

Status codes: `400` bad request · `404` unknown run · `422` validation error ·
`500` unexpected failure (body still uses the error shape).
