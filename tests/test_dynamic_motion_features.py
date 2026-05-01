import numpy as np
import pytest

from pipeline.fall.features import extract_fall_window_features


def _make_window(acc_magnitude: np.ndarray, *, fs: float = 50.0) -> dict:
    n = int(len(acc_magnitude))
    end_ts = (n - 1) / fs if n > 1 else 0.0
    return {
        "window_id": 0,
        "dataset_name": "TEST",
        "subject_id": "S1",
        "session_id": "sess1",
        "source_file": "fixture.csv",
        "task_type": "fall",
        "start_ts": 0.0,
        "end_ts": end_ts,
        "n_samples": n,
        "sensor_payload": {"acc_magnitude": acc_magnitude},
    }


def test_dynamic_motion_features_constant_gravity_signal_near_zero():
    acc_mag = np.full(16, 9.81, dtype=float)
    window = _make_window(acc_mag)

    feats = extract_fall_window_features(window, default_sampling_rate_hz=50.0, post_impact_skip_samples=2)

    assert feats["g_reference"] == pytest.approx(9.81, abs=1e-6)
    assert feats["post_impact_dyn_mean"] == pytest.approx(0.0, abs=1e-6)
    assert feats["post_impact_dyn_rms"] == pytest.approx(0.0, abs=1e-6)


def test_dynamic_motion_features_varying_signal_is_positive():
    acc_mag = np.array([9.81, 9.90, 10.20, 12.00, 10.40, 9.60, 9.50, 9.90, 9.81], dtype=float)
    window = _make_window(acc_mag)

    feats = extract_fall_window_features(window, default_sampling_rate_hz=50.0, post_impact_skip_samples=1)

    assert feats["post_impact_dyn_mean"] > 0.0
    assert feats["post_impact_dyn_rms"] > 0.0
    assert feats["post_impact_dyn_rms"] >= feats["post_impact_dyn_mean"]


def test_dynamic_motion_features_auto_g_reference_for_g_units():
    acc_mag = np.array([1.0, 1.02, 0.98, 1.05, 0.97, 1.01], dtype=float)
    window = _make_window(acc_mag)

    feats = extract_fall_window_features(window, default_sampling_rate_hz=50.0, post_impact_skip_samples=1)

    assert feats["g_reference"] == pytest.approx(1.0, abs=1e-6)
    assert feats["post_impact_dyn_mean"] >= 0.0
