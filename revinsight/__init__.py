"""RevInsight — Store Staffing Analytics.

A data-driven staffing optimisation engine for retail stores.

The package is organised in layers:

``revinsight.config``
    Business + modelling assumptions (:class:`~revinsight.config.AnalysisConfig`).
``revinsight.utils``
    Data loading, validation and aggregation helpers.
``revinsight.forecasting``
    Model zoo with walk-forward validation, ensembling and intervals.
``revinsight.financial_analysis``
    Gross margin, break-even, sensitivity and payback analytics.
``revinsight.staffing_model``
    Productivity, diminishing returns and scenario optimisation.
``revinsight.seasonality`` / ``revinsight.anomaly`` / ``revinsight.people``
    Calendar structure, anomaly detection and workforce analytics.
``revinsight.pipeline``
    End-to-end orchestration returning a JSON-serialisable result.
``revinsight.api``
    FastAPI application exposing the analytics engine over HTTP.
``revinsight.cli``
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
