import numpy as np

from pipeline.fall.threshold_detector import (
    FallThresholdConfig,
    detect_fall_from_features,
    detect_fall_window,
)


def test_detect_fall_from_features_obvious_fall_case():
    cfg = FallThresholdConfig(
        impact_peak_acc_threshold=14.0,
        impact_peak_ratio_threshold=1.2,
        require_support_stage=False,
        require_confirm_stage=True,
        confirm_post_dyn_ratio_mean_max=None,
        post_impact_motion_max=13.0,
        post_impact_variance_max=25.0,
        post_impact_motion_ratio_max=0.95,
    )
    feats = {
        "peak_acc": 25.0,
        "mean_acc": 10.0,
        "peak_over_mean_ratio": 2.5,
        "post_impact_motion": 10.5,
        "post_impact_variance": 2.0,
    }
    out = detect_fall_from_features(feats, cfg)
    assert out["predicted_label"] == "fall"
    assert out["stage_impact_pass"] is True
    assert out["stage_confirm_pass"] is True
    assert out["detector_reason"] == "fall_detected"


def test_detect_fall_from_features_obvious_non_fall_case():
    cfg = FallThresholdConfig(
        impact_peak_acc_threshold=14.0,
        impact_peak_ratio_threshold=1.2,
        require_support_stage=False,
        require_confirm_stage=True,
        confirm_post_dyn_ratio_mean_max=None,
        post_impact_motion_max=13.0,
        post_impact_variance_max=25.0,
        post_impact_motion_ratio_max=0.95,
    )
    feats = {
        "peak_acc": 11.5,
        "mean_acc": 9.9,
        "peak_over_mean_ratio": 1.16,
        "post_impact_motion": 10.0,
        "post_impact_variance": 1.0,
    }
    out = detect_fall_from_features(feats, cfg)
    assert out["predicted_label"] == "non_fall"
    assert out["stage_impact_pass"] is False
    assert out["detector_reason"] == "failed_impact_stage"


def test_detect_fall_window_returns_details():
    acc_mag = np.array([9.8, 10.0, 11.0, 24.0, 11.5, 10.2, 9.9, 9.8], dtype=float)
    window = {
        "window_id": 1,
        "dataset_name": "MOBIFALL",
        "subject_id": "M01",
        "session_id": "trial_01",
        "source_file": "fixture.csv",
        "task_type": "fall",
        "start_ts": 0.0,
        "end_ts": (len(acc_mag) - 1) / 50.0,
        "n_samples": len(acc_mag),
        "sensor_payload": {"acc_magnitude": acc_mag},
    }
    cfg = FallThresholdConfig(
        impact_peak_acc_threshold=14.0,
        impact_peak_ratio_threshold=1.2,
        require_support_stage=False,
        require_confirm_stage=True,
        confirm_post_dyn_ratio_mean_max=None,
        post_impact_motion_max=13.0,
        post_impact_variance_max=25.0,
        post_impact_motion_ratio_max=0.95,
    )
    out = detect_fall_window(window, config=cfg, default_sampling_rate_hz=50.0)
    assert "features" in out and "decision" in out and "config" in out
    assert out["decision"]["predicted_label"] in {"fall", "non_fall"}
    assert "peak_acc" in out["features"]
