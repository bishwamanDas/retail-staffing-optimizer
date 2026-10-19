# ADR 0001: Walk-Forward Cross-Validation for Forecasting Model Selection

## Status
Accepted

## Context
Traditional machine learning and cross-validation techniques (like k-fold or a single static train/test split) violate time-series independence. In real deployment, models forecast into the future based only on historical data up to a given decision point. A model that performs well on an arbitrary single test slice often fails when applied iteratively over varying economic conditions and seasons.

## Decision
We implement rolling-origin **walk-forward cross-validation** (`plan_backtest`) across the candidate forecasting models:
1. Candidate origins move forward by 1 month across recent history.
2. At each origin, models train only on data prior to that origin and forecast a fixed $H$-month horizon (default $H=6$).
3. Performance is measured across all origins using multi-step RMSE, MAE, and MAPE.
4. The ensemble weights are computed directly using inverse-RMSE weighting over these walk-forward errors.

## Consequences
- **Pros**: Completely eliminates data leakage; accurately reflects production forecast degradation over longer horizons; prevents overfitting.
- **Cons**: Computationally heavier during pipeline runs ($O(M \times K)$ where $M$ is model count and $K$ is origins), mitigated by multi-threaded background processing in FastAPI.
