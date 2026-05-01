from dataclasses import replace

import pandas as pd

from pipeline.fall.threshold_detector import default_fall_threshold_config
from scripts.sweep_fall_thresholds import evaluate_config_vectorized, prepare_vectorized_sweep_data


def test_sisfall_vectorized_sweep_has_nonzero_true_positive_for_reasonable_ratio_threshold():
    # Synthetic SisFall-like magnitudes (large peak_acc), but unit-robust ratio confirmation.
    df = pd.DataFrame(
        {
            "true_label": ["fall", "fall", "non_fall", "non_fall", "fall", "non_fall"],
            "peak_acc": [1200.0, 980.0, 600.0, 520.0, 1500.0, 430.0],
            "peak_over_mean_ratio": [2.2, 1.9, 1.3, 1.1, 2.5, 1.05],
            "jerk_peak": [2500.0, 2100.0, 900.0, 700.0, 2800.0, 600.0],
            "gyro_peak": [1300.0, 1000.0, 300.0, 250.0, 1400.0, 200.0],
            "post_impact_motion": [350.0, 330.0, 410.0, 390.0, 340.0, 420.0],
            "post_impact_variance": [12000.0, 14000.0, 18000.0, 20000.0, 10000.0, 22000.0],
            "post_impact_dyn_ratio_mean": [0.10, 0.16, 0.38, 0.42, 0.08, 0.45],
            "post_impact_available": [True, True, True, True, True, True],
        }
    )
    data = prepare_vectorized_sweep_data(df)
    base_cfg = default_fall_threshold_config("SISFALL")

    cfgs = [
        replace(
            base_cfg,
            impact_peak_acc_threshold=500.0,
            confirm_post_dyn_ratio_mean_max=0.20,
            confirm_post_var_max=25000.0,
            require_support_stage=False,
        ),
        replace(
            base_cfg,
            impact_peak_acc_threshold=700.0,
            confirm_post_dyn_ratio_mean_max=0.15,
            confirm_post_var_max=20000.0,
            require_support_stage=True,
            jerk_peak_threshold=1800.0,
        ),
    ]

    tp_values = [int(evaluate_config_vectorized(data, cfg)["metrics"]["tp"]) for cfg in cfgs]
    assert max(tp_values) > 0
