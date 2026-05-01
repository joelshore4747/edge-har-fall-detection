import numpy as np
import pytest

from pipeline.features.time_domain import compute_time_domain_features, extract_time_domain_features_for_window


def test_compute_time_domain_features_known_values():
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)
    feats = compute_time_domain_features(x, prefix="x")

    assert feats["x_mean"] == pytest.approx(2.5)
    assert feats["x_std"] == pytest.approx(np.std(x, ddof=0))
    assert feats["x_min"] == pytest.approx(1.0)
    assert feats["x_max"] == pytest.approx(4.0)
    assert feats["x_median"] == pytest.approx(2.5)
    assert feats["x_rms"] == pytest.approx(np.sqrt(np.mean(x**2)))
    assert feats["x_iqr"] == pytest.approx(1.5)


def test_extract_time_domain_features_for_window_multiple_channels():
    payload = {
        "ax": np.array([0.0, 1.0, 2.0], dtype=float),
        "acc_magnitude": np.array([1.0, 1.0, 1.0], dtype=float),
    }
    feats = extract_time_domain_features_for_window(payload)

    assert "ax_mean" in feats
    assert "ax_std" in feats
    assert "acc_magnitude_rms" in feats
    assert feats["acc_magnitude_std"] == pytest.approx(0.0)


def test_compute_time_domain_features_ignores_nans():
    x = np.array([1.0, np.nan, 3.0], dtype=float)
    feats = compute_time_domain_features(x, prefix="x")
    assert feats["x_mean"] == pytest.approx(2.0)
    assert feats["x_min"] == pytest.approx(1.0)
    assert feats["x_max"] == pytest.approx(3.0)
