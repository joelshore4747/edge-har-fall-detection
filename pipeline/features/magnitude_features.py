from __future__ import annotations

from typing import Mapping

import numpy as np


def _as_numeric(signal) -> np.ndarray:
    arr = np.asarray(signal, dtype=float).reshape(-1)
    return arr


def _finite(signal) -> np.ndarray:
    arr = _as_numeric(signal)
    return arr[np.isfinite(arr)]


def signal_magnitude_area(ax, ay, az) -> float:
    x = _as_numeric(ax)
    y = _as_numeric(ay)
    z = _as_numeric(az)
    n = min(len(x), len(y), len(z))
    if n == 0:
        return float("nan")
    tri = np.vstack([x[:n], y[:n], z[:n]]).T
    tri = tri[np.all(np.isfinite(tri), axis=1)]
    if tri.size == 0:
        return float("nan")
    return float(np.mean(np.abs(tri).sum(axis=1)))


def _jerk(signal, sampling_rate_hz: float) -> np.ndarray:
    x = _finite(signal)
    if x.size < 2 or not np.isfinite(sampling_rate_hz) or sampling_rate_hz <= 0:
        return np.array([], dtype=float)
    return np.diff(x) * float(sampling_rate_hz)


def _jerk_features(signal, *, prefix: str, sampling_rate_hz: float) -> dict[str, float]:
    x = _finite(signal)
    j = _jerk(x, sampling_rate_hz)
    if j.size == 0:
        return {
            f"{prefix}_jerk_mean_abs": np.nan,
            f"{prefix}_jerk_rms": np.nan,
            f"{prefix}_jerk_std": np.nan,
            f"{prefix}_mean_abs_diff": np.nan,
        }
    return {
        f"{prefix}_jerk_mean_abs": float(np.mean(np.abs(j))),
        f"{prefix}_jerk_rms": float(np.sqrt(np.mean(np.square(j)))),
        f"{prefix}_jerk_std": float(np.std(j, ddof=0)),
        f"{prefix}_mean_abs_diff": float(np.mean(np.abs(np.diff(x)))),
    }


def extract_magnitude_features_for_window(
    sensor_payload: Mapping[str, np.ndarray],
    *,
    sampling_rate_hz: float,
) -> dict[str, float]:
    features: dict[str, float] = {}

    if all(ch in sensor_payload for ch in ("ax", "ay", "az")):
        features["acc_sma"] = signal_magnitude_area(
            sensor_payload["ax"], sensor_payload["ay"], sensor_payload["az"]
        )

    if "acc_magnitude" in sensor_payload:
        acc_mag = _finite(sensor_payload["acc_magnitude"])
        if acc_mag.size > 0:
            features["acc_magnitude_range"] = float(np.max(acc_mag) - np.min(acc_mag))
            mean_val = float(np.mean(acc_mag))
            std_val = float(np.std(acc_mag, ddof=0))
            features["acc_magnitude_cv"] = float(std_val / mean_val) if mean_val != 0 else np.nan
        else:
            features["acc_magnitude_range"] = np.nan
            features["acc_magnitude_cv"] = np.nan
        features.update(_jerk_features(sensor_payload["acc_magnitude"], prefix="acc_magnitude", sampling_rate_hz=sampling_rate_hz))

    if "gyro_magnitude" in sensor_payload:
        gyro_mag = _finite(sensor_payload["gyro_magnitude"])
        if gyro_mag.size > 0:
            features["gyro_magnitude_range"] = float(np.max(gyro_mag) - np.min(gyro_mag))
            mean_val = float(np.mean(gyro_mag))
            std_val = float(np.std(gyro_mag, ddof=0))
            features["gyro_magnitude_cv"] = float(std_val / mean_val) if mean_val != 0 else np.nan
        else:
            features["gyro_magnitude_range"] = np.nan
            features["gyro_magnitude_cv"] = np.nan
        features.update(_jerk_features(sensor_payload["gyro_magnitude"], prefix="gyro_magnitude", sampling_rate_hz=sampling_rate_hz))

    return features
