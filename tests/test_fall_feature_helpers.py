import numpy as np
import pytest

from pipeline.fall.features import extract_fall_window_features


def _make_window_with_payload(payload: dict, *, fs: float = 50.0) -> dict:
    n = len(next(iter(payload.values()))) if payload else 0
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
        "sensor_payload": payload,
    }


def test_extract_fall_window_features_acc_magnitude_only():
    # Obvious impact around index 3, then settling motion.
    acc_mag = np.array([9.8, 10.0, 11.0, 25.0, 12.0, 10.5, 9.9, 9.8], dtype=float)
    window = _make_window_with_payload({"acc_magnitude": acc_mag}, fs=50.0)

    feats = extract_fall_window_features(window, default_sampling_rate_hz=50.0, post_impact_skip_samples=1)

    assert feats["peak_acc"] == pytest.approx(25.0)
    assert feats["mean_acc"] == pytest.approx(float(np.mean(acc_mag)))
    assert feats["impact_index"] == pytest.approx(3.0)
    assert feats["impact_time_offset_s"] == pytest.approx(3 / 50.0)
    assert feats["post_impact_motion"] == pytest.approx(float(np.mean(acc_mag[4:])))
    assert feats["post_impact_samples"] == pytest.approx(float(len(acc_mag[4:])))
    assert feats["jerk_peak"] > 0.0
    assert np.isnan(feats["gyro_peak"])


def test_extract_fall_window_features_derives_magnitudes_from_axes():
    ax = np.array([0.0, 0.1, 2.0, 0.1], dtype=float)
    ay = np.array([0.0, 0.1, 2.0, 0.1], dtype=float)
    az = np.array([9.8, 9.8, 9.8, 9.8], dtype=float)
    gx = np.array([0.0, 0.2, 3.0, 0.1], dtype=float)
    gy = np.array([0.0, 0.2, 3.0, 0.1], dtype=float)
    gz = np.array([0.0, 0.2, 3.0, 0.1], dtype=float)

    window = _make_window_with_payload(
        {"ax": ax, "ay": ay, "az": az, "gx": gx, "gy": gy, "gz": gz},
        fs=50.0,
    )
    feats = extract_fall_window_features(window, default_sampling_rate_hz=50.0)

    assert feats["peak_acc"] > 9.8
    assert feats["gyro_peak"] == pytest.approx(float(np.max(np.sqrt(gx**2 + gy**2 + gz**2))))
    assert feats["window_sampling_rate_hz"] == pytest.approx(50.0)
