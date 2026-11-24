import numpy as np
import pandas as pd

from src.train_forecast import FEATURE_COLUMNS, build_features


def test_feature_generation_is_time_ordered_and_complete():
    dates = pd.date_range("2025-01-01", periods=80, freq="D")
    frame = pd.DataFrame({"Close": np.linspace(40000, 50000, 80)}, index=dates)
    features = build_features(frame)
    assert set(FEATURE_COLUMNS).issubset(features.columns)
    assert not features[FEATURE_COLUMNS].isna().any().any()
    assert features.index.is_monotonic_increasing
