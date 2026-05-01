from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np


def _as_finite_1d(signal) -> np.ndarray:
    arr = np.asarray(signal, dtype=float).reshape(-1)
    if arr.size == 0:
        return arr
    return arr[np.isfinite(arr)]


def _nan_feature_block(prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_mean": np.nan,
        f"{prefix}_mean_abs": np.nan,
        f"{prefix}_std": np.nan,
        f"{prefix}_min": np.nan,
        f"{prefix}_max": np.nan,
        f"{prefix}_median": np.nan,
        f"{prefix}_rms": np.nan,
        f"{prefix}_iqr": np.nan,
        f"{prefix}_energy": np.nan,
    }


def compute_time_domain_features(signal, *, prefix: str) -> dict[str, float]:
    x = _as_finite_1d(signal)
    if x.size == 0:
        return _nan_feature_block(prefix)

    q25, q75 = np.percentile(x, [25, 75])
    return {
        f"{prefix}_mean": float(np.mean(x)),
        f"{prefix}_mean_abs": float(np.mean(np.abs(x))),
        f"{prefix}_std": float(np.std(x, ddof=0)),
        f"{prefix}_min": float(np.min(x)),
        f"{prefix}_max": float(np.max(x)),
        f"{prefix}_median": float(np.median(x)),
        f"{prefix}_rms": float(np.sqrt(np.mean(np.square(x)))),
        f"{prefix}_iqr": float(q75 - q25),
        f"{prefix}_energy": float(np.sum(np.square(x))),
    }


def extract_time_domain_features_for_window(
    sensor_payload: Mapping[str, np.ndarray],
    *,
    channels: Iterable[str] | None = None,
) -> dict[str, float]:
    if channels is None:
        preferred = ["ax", "ay", "az", "acc_magnitude", "gx", "gy", "gz", "gyro_magnitude"]
        channels = [c for c in preferred if c in sensor_payload]

    features: dict[str, float] = {}
    for channel in channels:
        if channel not in sensor_payload:
            continue
        features.update(compute_time_domain_features(sensor_payload[channel], prefix=channel))
    return features
