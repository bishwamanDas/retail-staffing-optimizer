"""Performance benchmarking script for the Store Staffing Analyzer engine.

Measures wall-clock latency for:
* Data loading and validation
* Seasonal decomposition & cycle detection
* Walk-forward cross-validation forecasting zoo
* Financial and scenario optimization
* End-to-end pipeline execution
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from .config import DEFAULT_CONFIG
from .pipeline import run_analysis
from .utils import load_and_validate_data, prepare_monthly_summary


def benchmark_pipeline(csv_path: str | Path, runs: int = 3) -> dict[str, Any]:
    """Profile the analytics pipeline across repeated executions.

    Parameters
    ----------
    csv_path : str or Path
        Path to the target dataset CSV.
    runs : int
        Number of warm runs to average (default: 3).

    Returns
    -------
    dict
        Latency metrics in milliseconds.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Benchmark dataset not found: {path}")

    # 1. Loading & validation
    t0 = time.perf_counter()
    df = load_and_validate_data(path)
    summary_df = prepare_monthly_summary(df)
    load_ms = (time.perf_counter() - t0) * 1000.0

    # 2. Pipeline runs
    durations: list[float] = []
    for _ in range(runs):
        start = time.perf_counter()
        _ = run_analysis(filepath=str(path), config=DEFAULT_CONFIG)
        durations.append((time.perf_counter() - start) * 1000.0)

    avg_ms = sum(durations) / len(durations)
    min_ms = min(durations)
    max_ms = max(durations)

    return {
        "dataset": path.name,
        "rows": len(df),
        "months": len(summary_df),
        "load_and_prep_ms": round(load_ms, 2),
        "avg_pipeline_ms": round(avg_ms, 2),
        "min_pipeline_ms": round(min_ms, 2),
        "max_pipeline_ms": round(max_ms, 2),
        "runs": runs,
    }


if __name__ == "__main__":
    import json
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else "data/RevI-Test.csv"
    results = benchmark_pipeline(target)
    print(json.dumps(results, indent=2))
