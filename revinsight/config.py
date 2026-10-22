"""Centralised configuration for the RevInsight analytics engine.

All tunable business assumptions live here so that the CLI, the REST API
and the web dashboard share a single source of truth.  Values may be
overridden through environment variables (prefix ``REVINSIGHT_``) or
explicit keyword arguments.

Every monetary value is denominated in **Lakhs INR** (1 Lakh = 100,000 INR).
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any

__all__ = [
    "AnalysisConfig",
    "DEFAULT_CONFIG",
    "config_from_env",
]


def _env_float(name: str, default: float) -> float:
    """Read a float from the environment, falling back to *default*."""
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    """Read an int from the environment, falling back to *default*."""
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(float(raw))
    except ValueError:
        return default


@dataclass(frozen=True)
class AnalysisConfig:
    """Immutable bundle of business + modelling assumptions.

    Attributes
    ----------
    salary_per_employee:
        Monthly fixed salary per salesperson, in Lakhs INR.
    commission_rate:
        Commission paid to staff as a fraction of monthly sales (0.05 = 5 %).
    cogs_rate:
        Cost of goods sold as a fraction of sales (0.50 = 50 %).
    forecast_months:
        Length of the forward-looking forecast horizon.
    validation_months:
        Size of the hold-out window used by the rolling-origin backtest.
    backtest_origins:
        Number of walk-forward evaluation origins per model.
    confidence_level:
        Central coverage of the prediction interval (0.95 = 95 %).
    n_simulations:
        Monte-Carlo draws used for the probabilistic margin analysis.
    random_seed:
        Seed for reproducible Monte-Carlo results.
    new_hire_ramp:
        Productivity multipliers for a newly hired employee, month 1 first.
    min_headcount:
        Lower bound on headcount considered by the optimiser.
    max_headcount:
        Upper bound on headcount considered by the optimiser.
    """

    salary_per_employee: float = 3.0
    commission_rate: float = 0.05
    cogs_rate: float = 0.50
    forecast_months: int = 6
    validation_months: int = 12
    backtest_origins: int = 4
    confidence_level: float = 0.95
    n_simulations: int = 5_000
    random_seed: int = 42
    new_hire_ramp: tuple[float, ...] = (0.70, 0.75, 0.85, 0.90, 0.95, 1.00)
    min_headcount: int = 1
    max_headcount: int = 30

    # ------------------------------------------------------------------ #
    # Derived quantities                                                  #
    # ------------------------------------------------------------------ #
    @property
    def effective_margin_rate(self) -> float:
        """Net margin retained per rupee of sales after COGS and commission."""
        return 1.0 - self.cogs_rate - self.commission_rate

    @property
    def break_even_sales_per_employee(self) -> float:
        """Sales required to cover one employee's salary (Lakhs INR)."""
        rate = self.effective_margin_rate
        if rate <= 0:
            raise ValueError("effective margin rate must be positive to compute break-even sales")
        return self.salary_per_employee / rate

    def validate(self) -> None:
        """Raise :class:`ValueError` if any assumption is out of range.

        Raises
        ------
        ValueError
            If the configuration is internally inconsistent.
        """
        if self.salary_per_employee <= 0:
            raise ValueError("salary_per_employee must be > 0")
        if not 0 <= self.commission_rate < 1:
            raise ValueError("commission_rate must be in [0, 1)")
        if not 0 <= self.cogs_rate < 1:
            raise ValueError("cogs_rate must be in [0, 1)")
        if self.commission_rate + self.cogs_rate >= 1:
            raise ValueError("cogs_rate + commission_rate must be < 1 (no margin left over)")
        if self.forecast_months < 1:
            raise ValueError("forecast_months must be >= 1")
        if self.validation_months < 2:
            raise ValueError("validation_months must be >= 2")
        if self.backtest_origins < 1:
            raise ValueError("backtest_origins must be >= 1")
        if not 0 < self.confidence_level < 1:
            raise ValueError("confidence_level must be in (0, 1)")
        if self.n_simulations < 100:
            raise ValueError("n_simulations must be >= 100")
        if not self.new_hire_ramp:
            raise ValueError("new_hire_ramp must not be empty")
        if any(f <= 0 for f in self.new_hire_ramp):
            raise ValueError("new_hire_ramp factors must all be > 0")
        if self.min_headcount < 1:
            raise ValueError("min_headcount must be >= 1")
        if self.max_headcount < self.min_headcount:
            raise ValueError("max_headcount must be >= min_headcount")

    # ------------------------------------------------------------------ #
    # Serialisation                                                       #
    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation of the configuration."""
        data = asdict(self)
        data["new_hire_ramp"] = list(self.new_hire_ramp)
        data["effective_margin_rate"] = self.effective_margin_rate
        data["break_even_sales_per_employee"] = round(self.break_even_sales_per_employee, 4)
        return data

    def replace(self, **overrides: Any) -> AnalysisConfig:
        """Return a copy of this config with *overrides* applied.

        ``None`` values are ignored, which lets API handlers pass optional
        user-supplied parameters straight through.
        """
        clean = {k: v for k, v in overrides.items() if v is not None}
        if not clean:
            return self
        current = asdict(self)
        current["new_hire_ramp"] = tuple(self.new_hire_ramp)
        current.update(clean)
        return AnalysisConfig(**current)


DEFAULT_CONFIG = AnalysisConfig()


def config_from_env(**overrides: Any) -> AnalysisConfig:
    """Build an :class:`AnalysisConfig` from environment variables.

    Recognised variables (all optional):

    * ``REVINSIGHT_SALARY`` — monthly salary per employee (Lakhs).
    * ``REVINSIGHT_COMMISSION`` — commission rate as a fraction.
    * ``REVINSIGHT_COGS`` — cost of goods sold rate.
    * ``REVINSIGHT_FORECAST_MONTHS`` — forecast horizon.
    * ``REVINSIGHT_SEED`` — random seed for Monte-Carlo sampling.

    Parameters
    ----------
    **overrides:
        Explicit values that take precedence over the environment.

    Returns
    -------
    AnalysisConfig
        A validated configuration object.
    """
    cfg = AnalysisConfig(
        salary_per_employee=_env_float("REVINSIGHT_SALARY", DEFAULT_CONFIG.salary_per_employee),
        commission_rate=_env_float("REVINSIGHT_COMMISSION", DEFAULT_CONFIG.commission_rate),
        cogs_rate=_env_float("REVINSIGHT_COGS", DEFAULT_CONFIG.cogs_rate),
        forecast_months=_env_int("REVINSIGHT_FORECAST_MONTHS", DEFAULT_CONFIG.forecast_months),
        random_seed=_env_int("REVINSIGHT_SEED", DEFAULT_CONFIG.random_seed),
    )
    cfg = cfg.replace(**overrides)
    cfg.validate()
    return cfg
