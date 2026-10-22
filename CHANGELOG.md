# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Nothing yet.

## [2.1.0] - 2026-10-22

### Added
- **MIT `LICENSE`** with a note that the bundled datasets are synthetic.
- **Dashboard screenshots** in `docs/screenshots/` and a Screenshots section in the README.
- **On-demand report rendering** — `POST /api/v1/analyses/{id}/report` renders the
  charts and the PDF/HTML/CSV artefacts, and the `GET` artefact routes render them
  automatically if they are missing. `POST /api/v1/analyses` now returns in ~6 s
  instead of 17-36 s because the report is no longer built inline.
- **Seasonal-naive extension for the simulator** — the what-if horizon slider now
  explores 1-24 months instead of being clamped to the model's forecast window, and
  the note states which months are extrapolated.

### Changed
- **Redesigned the dashboard theme** — neutral graphite surfaces with a single brass
  accent, hairline borders instead of glows, smaller radii and a desaturated chart
  palette, replacing the blue-gradient look in both the dark and light themes.
- **Every displayed figure is now derived from the payload** — the dataset default,
  the money-unit example, the scenario count, the prediction-band level, the horizon
  labels, the ramp description, the model-zoo size and the business-assumption prose
  are all computed instead of hard-coded.
- **The headcount optimum is labelled explicitly** as *capacity optimum* alongside the
  *decide now* headcount, and the report explains why the two differ.

### Fixed
- **Simulator callouts no longer accumulate** — each run replaces the previous note
  instead of appending a new one.
- **The salary × commission heat map renders again** — the sensitivity grid now uses
  snake_case field names end to end, matching the published API contract.
- **The walk-forward backtest table and chart render again** — the API exposes a
  scalar per origin plus the full path.
- **The "Download report" link follows the current run** after a re-analysis.
- **Slider changes made during an in-flight simulation are no longer dropped**, and
  the status line reports the horizon the server actually used.
- **The headline can no longer contradict the recommendation** — it branches on the
  recommended scenario, so a "don't hire" outcome never claims a hiring gain.
- **No horizontal page scroll on mobile** — the long verdict badge wraps, three
  tables gained scroll containers, the recommendation text breaks long rules, and
  the table layout fits narrow screens.

## [2.0.0] - 2026-10-22

### Added
- **Model zoo with walk-forward cross-validation** — a single `forecasting` module now
  evaluates moving average, linear regression, exponential smoothing (Holt-Winters),
  and seasonal-naive candidates on a rolling-origin backtest, selecting the winner on
  RMSE/MAPE and reporting per-model error metrics plus prediction intervals.
- **Controlled marginal-productivity staffing model** — `staffing_model` replaces the
  v1 heuristic with a documented productivity curve, diminishing-returns saturation,
  new-hire ramp-up factors and an explicit objective (gross margin) optimised over
  headcount scenarios.
- **Seasonality, anomaly and workforce analytics** — new `seasonality`, `anomaly` and
  `people` modules provide calendar decomposition, robust anomaly detection and
  per-salesperson productivity/tenure analytics.
- **Sensitivity and Monte-Carlo analysis** — `sensitivity` produces a tornado chart
  block, a two-way parameter grid and a seeded probabilistic margin distribution
  (tornado, grid and Monte-Carlo blocks are exposed through the API).
- **FastAPI service and static web dashboard** — `revinsight.api` serves a versioned
  REST API under `/api/v1` (health, datasets, analyses, charts, PDF/HTML/CSV artefacts,
  simulation and sensitivity endpoints) and hosts the `web/` single-page dashboard from
  the same origin, so the whole product runs as one process.
- **Professional PDF and HTML reporting** — `report_generator` produces a branded PDF
  recommendation report and an HTML twin, plus a plain-text recommendation and a CSV
  export of the monthly results.
- **Packaging and CLI** — PEP 621 `pyproject.toml` with a `revinsight` console script,
  a structured `revinsight.cli` (`run` / `serve` sub-commands) and a thin `run.py`
  wrapper that keeps `python run.py data/RevI-Test.csv` working.
- **Centralised configuration** — `revinsight.config.AnalysisConfig` holds every
  business assumption, is validated on construction and can be overridden through
  `REVINSIGHT_*` environment variables.
- **Test suite** — `pytest` suite covering configuration, forecasting, financial
  analysis, the staffing model, the pipeline and the HTTP API.
- **CI/CD and packaging infrastructure** — GitHub Actions workflows (test matrix on
  Python 3.10-3.12, ruff/black lint job, container build job, manual Render deploy),
  a multi-stage non-root `Dockerfile` with a python-based health check,
  `docker-compose.yml`, `Procfile`, `render.yaml`, `fly.toml`, `Makefile`,
  `requirements.txt` / `requirements-dev.txt` split, `.env.example` and a
  `Keep a Changelog` history.

### Changed
- The project was reorganised from a flat `src/` package into the importable
  `revinsight/` package, with `revinsight.api` added as a sub-package.
- All configuration now flows through a single validated object instead of being
  scattered across modules and CLI flags.
- Forecast model selection is driven by a rolling-origin backtest instead of a single
  fixed hold-out window.
- The recommendation output was expanded from a single scenario comparison to a
  full decision set (hold / hire / conditional) with confidence and payback period.

### Deprecated
- The v1 `src/` module paths (`src.forecasting`, `src.staffing_model`, ...) are no
  longer importable; use the `revinsight.*` equivalents.

## [1.0.0] - 2025-06-01

### Added
- First public release of the store staffing analyzer: CSV ingestion with dynamic
  sales-team detection, exploratory data analysis and chart generation.
- Three-model forecasting comparison (moving average, linear regression, exponential
  smoothing) evaluated on a 12-month hold-out window with automatic model selection
  on RMSE.
- Financial modelling of gross margin, break-even sales per employee and payback
  period using salary and commission assumptions.
- Multi-scenario staffing comparison (do-not-hire, hire-now, hire-later,
  conditional-hire) with a new-hire ramp-up curve and an adaptive staffing rule.
- Professional PDF recommendation report, CSV metrics export and plain-text summary.
- Command-line entry point (`run.py`) with configurable salary, commission and
  forecast-horizon parameters.

[Unreleased]: https://github.com/bishwamanDas/retail-staffing-optimizer/compare/v2.1.0...HEAD
[2.1.0]: https://github.com/bishwamanDas/retail-staffing-optimizer/compare/v2.0.0...v2.1.0
[2.0.0]: https://github.com/bishwamanDas/retail-staffing-optimizer/compare/v1.0.0...v2.0.0
[1.0.0]: https://github.com/bishwamanDas/retail-staffing-optimizer/releases/tag/v1.0.0
