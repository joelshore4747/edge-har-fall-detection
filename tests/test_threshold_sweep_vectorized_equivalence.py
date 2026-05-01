from dataclasses import replace

import pandas as pd
import pytest

from metrics.fall_metrics import compute_fall_detection_metrics
from pipeline.fall.threshold_detector import FallThresholdConfig, detect_fall_from_features
from scripts.sweep_fall_thresholds import (
    FEATURE_COLUMNS,
    evaluate_config_vectorized,
    prepare_vectorized_sweep_data,
)


def _rowwise_metrics(df: pd.DataFrame, cfg) -> dict:
    y_true = df["true_label"].astype(str).tolist()
    y_pred: list[str] = []
    for _, row in df.iterrows():
        features = {k: row.get(k) for k in FEATURE_COLUMNS if k in df.columns}
        decision = detect_fall_from_features(features, cfg)
        y_pred.append(str(decision["predicted_label"]))
    return compute_fall_detection_metrics(y_true, y_pred)


def test_vectorized_eval_matches_rowwise_detector_logic():
    df = pd.DataFrame(
        {
            "true_label": ["fall", "non_fall", "fall", "non_fall", "fall", "non_fall", "fall", "non_fall"],
            "peak_acc": [20.0, 11.0, 18.0, 22.0, 15.0, 14.0, 30.0, 9.0],
            "peak_over_mean_ratio": [1.30, 1.40, 1.10, 1.50, 1.28, 1.18, 1.35, 1.60],
            "jerk_peak": [45.0, 20.0, 65.0, 55.0, 40.0, 15.0, 90.0, 10.0],
            "gyro_peak": [10.0, 80.0, 20.0, 70.0, 5.0, 40.0, 2.0, 120.0],
            "post_impact_dyn_mean": [0.8, 1.2, None, 0.6, 1.6, 0.4, 0.7, None],
            "post_impact_dyn_ratio_mean": [0.05, 0.12, 0.08, 0.10, 0.30, 0.04, 0.07, None],
            "post_impact_motion": [1.0, 1.1, 0.9, 0.5, 1.8, 0.3, 0.6, 0.2],
            "post_impact_variance": [0.08, 0.07, 0.15, 0.03, 0.25, 0.02, 0.09, 0.01],
            "jerk_rms": [8.0, 9.0, 12.0, 7.0, 11.0, 6.0, 10.0, 5.0],
            "post_impact_available": [True, True, True, True, True, True, True, False],
        }
    )

    baseline = FallThresholdConfig(
        impact_peak_acc_threshold=14.0,
        impact_peak_ratio_threshold=1.25,
        require_support_stage=False,
        jerk_peak_threshold=30.0,
        gyro_peak_threshold=None,
        support_logic="any",
        require_confirm_stage=True,
        confirm_logic="all",
        confirm_post_dyn_ratio_mean_max=0.12,
        confirm_requires_post_impact=True,
        confirm_post_dyn_mean_max=None,
        confirm_post_var_max=0.15,
        confirm_post_jerk_rms_max=None,
        post_impact_motion_max=None,
        post_impact_variance_max=0.15,
        post_impact_motion_ratio_max=0.98,
        post_impact_skip_samples=2,
    )

    configs = [
        baseline,
        replace(
            baseline,
            require_support_stage=True,
            jerk_peak_threshold=50.0,
            gyro_peak_threshold=60.0,  # tests support_logic="any" with gyro fallback
        ),
        replace(
            baseline,
            confirm_logic="any",
            confirm_post_dyn_ratio_mean_max=0.20,
            confirm_post_var_max=0.10,
            post_impact_variance_max=0.10,
        ),
    ]

    data = prepare_vectorized_sweep_data(df)
    for cfg in configs:
        vectorized = evaluate_config_vectorized(data, cfg)["metrics"]
        rowwise = _rowwise_metrics(df, cfg)
        for key in ["tn", "fp", "fn", "tp", "accuracy", "sensitivity", "specificity", "precision", "f1"]:
            assert vectorized[key] == pytest.approx(rowwise[key])
