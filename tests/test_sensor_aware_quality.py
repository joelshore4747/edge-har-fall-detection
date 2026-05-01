import numpy as np
import pandas as pd

from pipeline.preprocess.config import PreprocessConfig
from pipeline.preprocess.quality import infer_active_sensor_columns, window_quality_summary


def test_accel_only_window_not_penalized_for_missing_gyro():
    df = pd.DataFrame(
        {
            "timestamp": [0.00, 0.02, 0.04, 0.06],
            "ax": [0.1, 0.2, 0.3, 0.4],
            "ay": [0.0, 0.1, 0.2, 0.3],
            "az": [9.8, 9.8, 9.8, 9.8],
            "gx": [np.nan, np.nan, np.nan, np.nan],
            "gy": [np.nan, np.nan, np.nan, np.nan],
            "gz": [np.nan, np.nan, np.nan, np.nan],
        }
    )
    cfg = PreprocessConfig(max_missing_ratio_per_window=0.20)

    active = infer_active_sensor_columns(df)
    summary = window_quality_summary(df, cfg, active_sensor_cols=active)

    assert active == ["ax", "ay", "az"]
    assert summary["active_sensor_columns"] == ["ax", "ay", "az"]
    assert summary["missing_ratio"] == 0.0
    assert summary["is_acceptable"] is True


def test_gyro_missingness_counts_when_group_has_usable_gyro():
    group_df = pd.DataFrame(
        {
            "timestamp": [0.00, 0.02, 0.04, 0.06],
            "ax": [0.1, 0.2, 0.3, 0.4],
            "ay": [0.1, 0.2, 0.3, 0.4],
            "az": [9.8, 9.8, 9.8, 9.8],
            "gx": [0.1, 0.2, 0.3, 0.4],
            "gy": [0.1, 0.2, 0.3, 0.4],
            "gz": [0.1, 0.2, 0.3, 0.4],
        }
    )
    # Window has a partial gyro dropout (should be penalized because the group is gyro-capable).
    window_df = group_df.copy()
    window_df.loc[1, "gx"] = np.nan

    cfg = PreprocessConfig(max_missing_ratio_per_window=0.20)
    active = infer_active_sensor_columns(group_df)
    summary = window_quality_summary(window_df, cfg, active_sensor_cols=active)

    assert active == ["ax", "ay", "az", "gx", "gy", "gz"]
    assert summary["missing_ratio"] > 0.0
    # 1 missing over 24 cells => ~0.0417, still acceptable at 20%
    assert summary["is_acceptable"] is True
