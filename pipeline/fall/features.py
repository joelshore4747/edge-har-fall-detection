"""Interpretable fall-oriented feature helpers for threshold baselines."""

from __future__ import annotations

from typing import Any

import numpy as np


def _as_float_array(values) -> np.ndarray:
    return np.asarray(values, dtype=float).reshape(-1)


def _finite(values) -> np.ndarray:
    arr = _as_float_array(values)
    return arr[np.isfinite(arr)]


def _vector_magnitude(x, y, z) -> np.ndarray:
    x = _as_float_array(x)
    y = _as_float_array(y)
    z = _as_float_array(z)
    n = min(len(x), len(y), len(z))
    if n == 0:
        return np.array([], dtype=float)
    return np.sqrt(x[:n] ** 2 + y[:n] ** 2 + z[:n] ** 2)


def _infer_sampling_rate_hz(window: dict[str, Any], default_sampling_rate_hz: float | None) -> float | None:
    if default_sampling_rate_hz is not None and np.isfinite(default_sampling_rate_hz):
        return float(default_sampling_rate_hz)

    n_samples = int(window.get("n_samples", 0) or 0)
    start_ts = window.get("start_ts")
    end_ts = window.get("end_ts")
    if n_samples >= 2 and start_ts is not None and end_ts is not None:
        try:
            duration = float(end_ts) - float(start_ts)
            if duration > 0:
                return float((n_samples - 1) / duration)
        except Exception:
            return None
    return None


def _safe_stat(fn, arr: np.ndarray) -> float:
    x = arr[np.isfinite(arr)] if arr.size else arr
    if x.size == 0:
        return float("nan")
    return float(fn(x))


def _infer_g_reference(acc_finite: np.ndarray, explicit_g_reference: float | None) -> float:
    """Infer gravity reference for dynamic-motion features.

    Default is 9.81 (m/s^2). If magnitudes look like ``g`` units, use 1.0.
    """
    if explicit_g_reference is not None and np.isfinite(float(explicit_g_reference)):
        return float(explicit_g_reference)
    if acc_finite.size == 0:
        return 9.81

    median_acc = float(np.median(acc_finite))
    peak_acc = float(np.max(acc_finite))
    # Heuristic: values around 1 indicate data likely expressed in g-units.
    if 0.5 <= median_acc <= 2.5 and peak_acc < 6.0:
        return 1.0
    return 9.81


def extract_fall_window_features(
    window: dict[str, Any],
    *,
    default_sampling_rate_hz: float | None = None,
    post_impact_skip_samples: int = 2,
    g_reference: float | None = None,
) -> dict[str, float | bool]:
    """Extract a compact set of fall-oriented indicators from one window.

    These features are intentionally simple and interpretable for a threshold baseline.
    """
    payload = window.get("sensor_payload", {}) or {}
    if not isinstance(payload, dict):
        raise ValueError("window['sensor_payload'] must be a dict")

    if "acc_magnitude" in payload:
        acc_mag = _as_float_array(payload["acc_magnitude"])
    elif all(ch in payload for ch in ("ax", "ay", "az")):
        acc_mag = _vector_magnitude(payload["ax"], payload["ay"], payload["az"])
    else:
        raise ValueError("Fall feature extraction requires acc_magnitude or ax/ay/az in sensor_payload")

    if "gyro_magnitude" in payload:
        gyro_mag = _as_float_array(payload["gyro_magnitude"])
    elif all(ch in payload for ch in ("gx", "gy", "gz")):
        gyro_mag = _vector_magnitude(payload["gx"], payload["gy"], payload["gz"])
    else:
        gyro_mag = np.array([], dtype=float)

    acc_finite = _finite(acc_mag)
    gravity_ref = _infer_g_reference(acc_finite, g_reference)
    if acc_finite.size == 0:
        return {
            "window_sampling_rate_hz": float("nan"),
            "g_reference": gravity_ref,
            "acc_baseline": float("nan"),
            "peak_acc": float("nan"),
            "mean_acc": float("nan"),
            "acc_variance": float("nan"),
            "peak_minus_mean": float("nan"),
            "peak_over_mean_ratio": float("nan"),
            "impact_index": -1.0,
            "impact_time_offset_s": float("nan"),
            "post_impact_motion": float("nan"),
            "post_impact_variance": float("nan"),
            "post_impact_samples": 0.0,
            "post_impact_available": False,
            "post_impact_dyn_mean": float("nan"),
            "post_impact_dyn_rms": float("nan"),
            "post_impact_dyn_ratio_mean": float("nan"),
            "post_impact_dyn_ratio_rms": float("nan"),
            "jerk_peak": float("nan"),
            "jerk_mean": float("nan"),
            "jerk_rms": float("nan"),
            "gyro_peak": float("nan"),
            "gyro_mean": float("nan"),
        }

    # Keep index in original (possibly NaN-containing) array for post-impact slicing.
    finite_mask = np.isfinite(acc_mag)
    finite_indices = np.flatnonzero(finite_mask)
    peak_idx_finite = int(np.argmax(acc_mag[finite_mask]))
    impact_index = int(finite_indices[peak_idx_finite]) if finite_indices.size else int(np.argmax(acc_finite))

    peak_acc = float(np.nanmax(acc_mag))
    mean_acc = _safe_stat(np.mean, acc_mag)
    acc_variance = _safe_stat(np.var, acc_mag)
    acc_baseline = _safe_stat(np.median, acc_mag)
    peak_minus_mean = float(peak_acc - mean_acc) if np.isfinite(mean_acc) else float("nan")
    peak_over_mean_ratio = float(peak_acc / mean_acc) if np.isfinite(mean_acc) and mean_acc != 0 else float("nan")

    tail_start = min(len(acc_mag), impact_index + max(1, int(post_impact_skip_samples)))
    post_tail = acc_mag[tail_start:]
    post_tail_finite = _finite(post_tail)
    post_impact_motion = _safe_stat(np.mean, post_tail_finite)
    post_impact_variance = _safe_stat(np.var, post_tail_finite)
    post_impact_samples = float(post_tail_finite.size)
    post_impact_available = bool(post_tail_finite.size > 0)
    post_tail_dyn = np.abs(post_tail_finite - gravity_ref)
    post_impact_dyn_mean = _safe_stat(np.mean, post_tail_dyn)
    post_impact_dyn_rms = _safe_stat(lambda x: np.sqrt(np.mean(np.square(x))), post_tail_dyn)
    if np.isfinite(acc_baseline):
        acc_dyn_ratio = np.abs(acc_mag - acc_baseline) / (float(acc_baseline) + 1e-9)
    else:
        acc_dyn_ratio = np.full(len(acc_mag), np.nan, dtype=float)
    post_tail_dyn_ratio = acc_dyn_ratio[tail_start:]
    post_tail_dyn_ratio_finite = _finite(post_tail_dyn)
    post_impact_dyn_ratio_mean = _safe_stat(np.mean, post_tail_dyn_ratio_finite)
    post_impact_dyn_ratio_rms = _safe_stat(lambda x: np.sqrt(np.mean(np.square(x))), post_tail_dyn_ratio_finite)

    fs = _infer_sampling_rate_hz(window, default_sampling_rate_hz)
    if fs is None or not np.isfinite(fs) or fs <= 0:
        jerk = np.diff(acc_finite) if acc_finite.size >= 2 else np.array([], dtype=float)
        fs_out = float("nan")
    else:
        jerk = np.diff(acc_finite) * float(fs) if acc_finite.size >= 2 else np.array([], dtype=float)
        fs_out = float(fs)

    jerk_peak = _safe_stat(np.max, np.abs(jerk))
    jerk_mean = _safe_stat(np.mean, np.abs(jerk))
    jerk_rms = _safe_stat(lambda x: np.sqrt(np.mean(np.square(x))), jerk)

    gyro_peak = _safe_stat(np.max, np.abs(gyro_mag)) if gyro_mag.size else float("nan")
    gyro_mean = _safe_stat(np.mean, np.abs(gyro_mag)) if gyro_mag.size else float("nan")

    impact_time_offset_s = float("nan")
    if len(acc_mag) > 0 and np.isfinite(fs_out) and fs_out > 0:
        impact_time_offset_s = float(impact_index / fs_out)

    return {
        "window_sampling_rate_hz": fs_out,
        "g_reference": gravity_ref,
        "acc_baseline": acc_baseline,
        "peak_acc": peak_acc,
        "mean_acc": mean_acc,
        "acc_variance": acc_variance,
        "peak_minus_mean": peak_minus_mean,
        "peak_over_mean_ratio": peak_over_mean_ratio,
        "impact_index": float(impact_index),
        "impact_time_offset_s": impact_time_offset_s,
        "post_impact_motion": post_impact_motion,
        "post_impact_variance": post_impact_variance,
        "post_impact_samples": post_impact_samples,
        "post_impact_available": post_impact_available,
        "post_impact_dyn_mean": post_impact_dyn_mean,
        "post_impact_dyn_rms": post_impact_dyn_rms,
        "post_impact_dyn_ratio_mean": post_impact_dyn_ratio_mean,
        "post_impact_dyn_ratio_rms": post_impact_dyn_ratio_rms,
        "jerk_peak": jerk_peak,
        "jerk_mean": jerk_mean,
        "jerk_rms": jerk_rms,
        "gyro_peak": gyro_peak,
        "gyro_mean": gyro_mean,
    }
