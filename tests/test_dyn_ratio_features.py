import numpy as np
import pytest

from pipeline.fall.features import extract_fall_window_features


def _window(acc_mag: np.ndarray, *, fs: float = 50.0) -> dict:
    n = int(len(acc_mag))
    return {
        "window_id": 0,
        "dataset_name": "TEST",
        "subject_id": "S1",
        "session_id": "sess1",
        "source_file": "fixture.csv",
        "task_type": "fall",
        "start_ts": 0.0,
        "end_ts": (n - 1) / fs if n > 1 else 0.0,
        "n_samples": n,
        "sensor_payload": {"acc_magnitude": acc_mag},
    }


def test_post_impact_dyn_ratio_mean_constant_signal_near_zero():
    acc_mag = np.full(16, 9.81, dtype=float)
    feats = extract_fall_window_features(_window(acc_mag), default_sampling_rate_hz=50.0, post_impact_skip_samples=2)

    assert feats["post_impact_available"] is True
    assert feats["post_impact_dyn_ratio_mean"] == pytest.approx(0.0, abs=1e-8)
    assert feats["post_impact_dyn_ratio_rms"] == pytest.approx(0.0, abs=1e-8)


def test_post_impact_dyn_ratio_mean_varying_signal_positive():
    acc_mag = np.array([9.81, 9.90, 10.10, 12.20, 10.60, 9.90, 9.70, 9.85, 9.81], dtype=float)
    feats = extract_fall_window_features(_window(acc_mag), default_sampling_rate_hz=50.0, post_impact_skip_samples=1)

    assert feats["post_impact_available"] is True
    assert feats["post_impact_dyn_ratio_mean"] > 0.0
    assert feats["post_impact_dyn_ratio_rms"] >= feats["post_impact_dyn_ratio_mean"]
