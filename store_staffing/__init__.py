"""Store Staffing Analyzer — retail staffing analytics.

A data-driven staffing optimisation engine for retail stores.

The package is organised in layers:

``store_staffing.config``
    Business + modelling assumptions (:class:`~store_staffing.config.AnalysisConfig`).
``store_staffing.utils``
    Data loading, validation and aggregation helpers.
``store_staffing.forecasting``
    Model zoo with walk-forward validation, ensembling and intervals.
``store_staffing.financial_analysis``
    Gross margin, break-even, sensitivity and payback analytics.
``store_staffing.staffing_model``
    Productivity, diminishing returns and scenario optimisation.
``store_staffing.seasonality`` / ``store_staffing.anomaly`` / ``store_staffing.people``
    Calendar structure, anomaly detection and workforce analytics.
``store_staffing.pipeline``
    End-to-end orchestration returning a JSON-serialisable result.
``store_staffing.api``
    FastAPI application exposing the analytics engine over HTTP.
``store_staffing.cli``
    Command-line entry point.
"""

from __future__ import annotations

__version__ = "2.1.0"
__author__ = "Bishwaman Das"
__all__ = [
    "__version__",
    "AnalysisConfig",
    "DEFAULT_CONFIG",
    "config_from_env",
]


def __getattr__(name: str):
    """Lazily expose the configuration helpers without eager imports."""
    if name in {"AnalysisConfig", "DEFAULT_CONFIG", "config_from_env"}:
        from . import config

        return getattr(config, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
