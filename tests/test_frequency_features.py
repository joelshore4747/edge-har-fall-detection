import numpy as np
import pytest

from pipeline.features.frequency_domain import compute_frequency_features, extract_frequency_features_for_window


def test_dominant_frequency_detects_simple_sine():
    fs = 50.0
    t = np.arange(0, 2.0, 1.0 / fs)
    x = np.sin(2 * np.pi * 2.0 * t)  # 2 Hz

    feats = compute_frequency_features(x, sampling_rate_hz=fs, prefix="sig")
    assert feats["sig_dominant_freq_hz"] == pytest.approx(2.0, abs=0.25)
    assert feats["sig_spectral_energy"] > 0.0
    assert 0.0 <= feats["sig_spectral_entropy"] <= 1.0


def test_frequency_features_constant_signal_is_stable():
    fs = 50.0
    x = np.ones(64, dtype=float)
    feats = compute_frequency_features(x, sampling_rate_hz=fs, prefix="sig")
    assert feats["sig_dominant_freq_hz"] == pytest.approx(0.0)
    assert feats["sig_spectral_energy"] == pytest.approx(0.0)
    assert feats["sig_spectral_entropy"] == pytest.approx(0.0)


def test_extract_frequency_features_window_acc_magnitude():
    fs = 50.0
    t = np.arange(0, 1.28, 1.0 / fs)
    payload = {"acc_magnitude": np.sin(2 * np.pi * 1.5 * t)}
    feats = extract_frequency_features_for_window(payload, sampling_rate_hz=fs)
    assert "acc_magnitude_dominant_freq_hz" in feats
    assert "acc_magnitude_spectral_energy" in feats


def test_step_cadence_recovers_known_frequency():
    fs = 50.0
    t = np.arange(0, 4.0, 1.0 / fs)
    # Gait-like waveform: fundamental 1.8 Hz + first harmonic.
    x = np.sin(2 * np.pi * 1.8 * t) + 0.3 * np.sin(2 * np.pi * 3.6 * t)
    feats = compute_frequency_features(x, sampling_rate_hz=fs, prefix="acc_magnitude")
    assert feats["acc_magnitude_step_cadence_hz"] == pytest.approx(1.8, abs=0.1)
    assert feats["acc_magnitude_step_cadence_strength"] > 0.5


def test_step_cadence_is_window_length_invariant():
    """A 4 s pocket window and a 2 s hand window of the same gait yield the
    same cadence within the bin tolerance — that's why the per-placement
    window-length change is safe to mix in a single feature table.
    """
    fs = 50.0

    def cadence_for_duration(duration: float) -> float:
        t = np.arange(0, duration, 1.0 / fs)
        x = np.sin(2 * np.pi * 1.8 * t)
        feats = compute_frequency_features(x, sampling_rate_hz=fs, prefix="acc")
        return feats["acc_step_cadence_hz"]

    cadence_2s = cadence_for_duration(2.0)
    cadence_4s = cadence_for_duration(4.0)
    assert cadence_2s == pytest.approx(cadence_4s, abs=0.1)


def test_step_cadence_low_strength_for_noise():
    rng = np.random.default_rng(0)
    feats = compute_frequency_features(rng.normal(size=200), sampling_rate_hz=50.0, prefix="acc")
    assert feats["acc_step_cadence_strength"] < 0.5


def test_step_cadence_zero_signal():
    feats = compute_frequency_features(np.zeros(100), sampling_rate_hz=50.0, prefix="acc")
    assert feats["acc_step_cadence_hz"] == pytest.approx(0.0)
    assert feats["acc_step_cadence_strength"] == pytest.approx(0.0)
