from __future__ import annotations

from typing import Mapping

import numpy as np


def _finite(signal) -> np.ndarray:
    arr = np.asarray(signal, dtype=float).reshape(-1)
    return arr[np.isfinite(arr)]


def _nan_freq_block(prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_dominant_freq_hz": np.nan,
        f"{prefix}_spectral_energy": np.nan,
        f"{prefix}_spectral_entropy": np.nan,
        f"{prefix}_bandpower_0p3_1hz": np.nan,
        f"{prefix}_bandpower_1_3hz": np.nan,
        f"{prefix}_bandpower_3_8hz": np.nan,
        f"{prefix}_bandpower_step_share": np.nan,
        f"{prefix}_bandpower_high_share": np.nan,
        f"{prefix}_dominant_power_share": np.nan,
        f"{prefix}_step_cadence_hz": np.nan,
        f"{prefix}_step_cadence_strength": np.nan,
    }


def _step_cadence_from_autocorr(
    x: np.ndarray,
    *,
    sampling_rate_hz: float,
    min_hz: float = 0.5,
    max_hz: float = 3.0,
) -> tuple[float, float]:
    """Find the dominant gait frequency by autocorrelation peak.

    Returns ``(cadence_hz, strength)`` where ``strength`` is the peak
    autocorrelation value at the cadence lag, normalised by the zero-lag
    energy (so it sits in roughly [0, 1] regardless of signal amplitude).

    Why autocorr in addition to FFT dominant_freq: short windows give the
    FFT poor frequency resolution (a 2 s window gives 0.5 Hz bins), so two
    distinct gait frequencies (1.6 Hz vs 2.0 Hz) collapse into the same
    bin. Time-domain autocorrelation finds the peak at sub-bin resolution
    and the strength tells the classifier whether the cadence is a real
    periodic structure or noise.
    """
    n = int(x.size)
    if n < 8 or not np.isfinite(sampling_rate_hz) or sampling_rate_hz <= 0:
        return float("nan"), float("nan")

    min_lag = max(1, int(round(sampling_rate_hz / max_hz)))
    max_lag = min(n - 1, int(round(sampling_rate_hz / min_hz)))
    if max_lag <= min_lag:
        return float("nan"), float("nan")

    centered = x - np.mean(x)
    energy = float(np.dot(centered, centered))
    if energy <= 0:
        return float("nan"), float("nan")

    full = np.correlate(centered, centered, mode="full")
    mid = full.size // 2
    autocorr = full[mid:] / energy

    band = autocorr[min_lag : max_lag + 1]
    if band.size == 0:
        return float("nan"), float("nan")
    peak_idx_local = int(np.argmax(band))
    peak_lag = min_lag + peak_idx_local
    strength = float(band[peak_idx_local])
    cadence_hz = float(sampling_rate_hz / peak_lag) if peak_lag > 0 else float("nan")
    # Strength can briefly exceed 1.0 due to numerical noise on tiny signals;
    # clamp so downstream consumers can rely on a [0, 1] range.
    strength = max(0.0, min(1.0, strength))
    return cadence_hz, strength


def _bandpower(freqs: np.ndarray, power: np.ndarray, low_hz: float, high_hz: float) -> float:
    if freqs.size == 0 or power.size == 0:
        return 0.0
    mask = (freqs >= float(low_hz)) & (freqs < float(high_hz))
    if not np.any(mask):
        return 0.0
    return float(np.sum(power[mask]))


def compute_frequency_features(
    signal,
    *,
    sampling_rate_hz: float,
    prefix: str,
    include_entropy: bool = True,
) -> dict[str, float]:
    x = _finite(signal)
    if x.size < 4 or not np.isfinite(sampling_rate_hz) or sampling_rate_hz <= 0:
        return _nan_freq_block(prefix)

    x = x - np.mean(x)
    if np.allclose(x, 0.0):
        out = {
            f"{prefix}_dominant_freq_hz": 0.0,
            f"{prefix}_spectral_energy": 0.0,
            f"{prefix}_spectral_entropy": 0.0 if include_entropy else np.nan,
            f"{prefix}_bandpower_0p3_1hz": 0.0,
            f"{prefix}_bandpower_1_3hz": 0.0,
            f"{prefix}_bandpower_3_8hz": 0.0,
            f"{prefix}_bandpower_step_share": 0.0,
            f"{prefix}_bandpower_high_share": 0.0,
            f"{prefix}_dominant_power_share": 0.0,
            f"{prefix}_step_cadence_hz": 0.0,
            f"{prefix}_step_cadence_strength": 0.0,
        }
        return out

    n = int(x.size)
    freqs = np.fft.rfftfreq(n, d=1.0 / float(sampling_rate_hz))
    fft_vals = np.fft.rfft(x)
    power = np.abs(fft_vals) ** 2

    if freqs.size > 1:
        pos_freqs = freqs[1:]
        pos_power = power[1:]
    else:
        pos_freqs = freqs
        pos_power = power

    if pos_power.size == 0:
        return _nan_freq_block(prefix)

    dom_idx = int(np.argmax(pos_power))
    dominant_freq = float(pos_freqs[dom_idx]) if pos_freqs.size > dom_idx else np.nan
    dominant_power = float(pos_power[dom_idx]) if pos_power.size > dom_idx else 0.0

    spectral_energy = float(np.sum(power) / n)

    spectral_entropy = np.nan
    if include_entropy:
        total = float(np.sum(pos_power))
        if total <= 0:
            spectral_entropy = 0.0
        else:
            p_norm = pos_power / total
            p_nonzero = p_norm[p_norm > 0]
            raw_entropy = -float(np.sum(p_nonzero * np.log2(p_nonzero)))
            denom = float(np.log2(max(2, pos_power.size)))
            spectral_entropy = float(raw_entropy / denom) if denom > 0 else 0.0

    low_band = _bandpower(pos_freqs, pos_power, 0.3, 1.0)
    step_band = _bandpower(pos_freqs, pos_power, 1.0, 3.0)
    high_band = _bandpower(pos_freqs, pos_power, 3.0, 8.0)

    total_pos_power = float(np.sum(pos_power))
    if total_pos_power > 0:
        step_share = float(step_band / total_pos_power)
        high_share = float(high_band / total_pos_power)
        dominant_share = float(dominant_power / total_pos_power)
    else:
        step_share = 0.0
        high_share = 0.0
        dominant_share = 0.0

    cadence_hz, cadence_strength = _step_cadence_from_autocorr(
        x, sampling_rate_hz=float(sampling_rate_hz)
    )

    return {
        f"{prefix}_dominant_freq_hz": dominant_freq,
        f"{prefix}_spectral_energy": spectral_energy,
        f"{prefix}_spectral_entropy": spectral_entropy,
        f"{prefix}_bandpower_0p3_1hz": low_band,
        f"{prefix}_bandpower_1_3hz": step_band,
        f"{prefix}_bandpower_3_8hz": high_band,
        f"{prefix}_bandpower_step_share": step_share,
        f"{prefix}_bandpower_high_share": high_share,
        f"{prefix}_dominant_power_share": dominant_share,
        f"{prefix}_step_cadence_hz": cadence_hz,
        f"{prefix}_step_cadence_strength": cadence_strength,
    }


def extract_frequency_features_for_window(
    sensor_payload: Mapping[str, np.ndarray],
    *,
    sampling_rate_hz: float,
) -> dict[str, float]:
    features: dict[str, float] = {}

    if "acc_magnitude" in sensor_payload:
        features.update(
            compute_frequency_features(
                sensor_payload["acc_magnitude"],
                sampling_rate_hz=sampling_rate_hz,
                prefix="acc_magnitude",
            )
        )

    if "gyro_magnitude" in sensor_payload:
        features.update(
            compute_frequency_features(
                sensor_payload["gyro_magnitude"],
                sampling_rate_hz=sampling_rate_hz,
                prefix="gyro_magnitude",
            )
        )

    return features