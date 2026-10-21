"""Edge-case tests for robust anomaly and structural break detection."""

import numpy as np
import pandas as pd
import pytest

from store_staffing.anomaly import _robust_scale, detect_anomalies


def test_robust_scale_flat_series():
    """Verify fallback scale when all observations are identical (MAD == 0)."""
    flat = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
    median, scale = _robust_scale(flat)
    assert median == 100.0
    # When variance is zero, fallback scale is 1.0 to prevent division by zero
    assert scale == 1.0


def test_detect_anomalies_flat_series():
    """Flat sales should produce zero anomalous months."""
    df = pd.DataFrame({
        "Month": list(range(1, 13)),
        "Sales": [50.0] * 12,
    })
    result = detect_anomalies(df)
    assert result["n_anomalies"] == 0
    assert len(result["anomalies"]) == 0


def test_detect_anomalies_single_spike():
    """A huge sales outlier should be flagged with high severity."""
    sales = [100.0] * 15
    sales[7] = 500.0  # Massive spike in Month 8
    df = pd.DataFrame({
        "Month": list(range(1, 16)),
        "Sales": sales,
    })
    result = detect_anomalies(df, threshold=3.0)
    assert result["n_anomalies"] >= 1
    flagged_months = [a["month"] for a in result["anomalies"]]
    assert 8 in flagged_months
