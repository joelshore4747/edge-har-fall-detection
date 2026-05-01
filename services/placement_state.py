from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from pipeline.preprocess import append_derived_channels


@dataclass(slots=True)
class PlacementStateConfig:
    target_rate_hz: float = 50.0
    window_seconds: float = 2.5
    step_seconds: float = 1.0

    smoothing_window: int = 5
    bridge_short_runs: int = 2
    min_reposition_windows: int = 3

    locomotion_min_acc_std: float = 0.22
    locomotion_freq_low_hz: float = 0.9
    locomotion_freq_high_hz: float = 3.2

    on_surface_acc_std_max: float = 0.10
    on_surface_gyro_mean_max: float = 0.04
    on_surface_jerk_mean_max: float = 0.80

    reposition_gyro_peak_min: float = 4.2
    reposition_jerk_mean_min: float = 18.0
    reposition_acc_std_min: float = 2.8


PLACEMENT_STATES = [
    "in_pocket",
    "in_hand",
    "on_surface",
    "arm_mounted",
    "repositioning",
    "unknown",
]


def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for col in ["gx", "gy", "gz"]:
        if col not in out.columns:
            out[col] = 0.0

    for col in ["dataset_name", "subject_id", "session_id"]:
        if col not in out.columns:
            out[col] = "unknown"

    return out


def _safe_mode(series: pd.Series) -> str:
    s = series.dropna().astype(str)
    if s.empty:
        return "unknown"
    return str(s.mode(dropna=True).iloc[0])


def _dominant_frequency_hz(values: np.ndarray, rate_hz: float) -> float | None:
    if len(values) < 8:
        return None

    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 8:
        return None

    x = x - np.mean(x)
    if np.allclose(x, 0.0):
        return None

    fft = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), d=1.0 / rate_hz)

    valid = (freqs >= 0.5) & (freqs <= 4.0)
    if not np.any(valid):
        return None

    amps = np.abs(fft[valid])
    if amps.size == 0:
        return None

    best_idx = int(np.argmax(amps))
    return float(freqs[valid][best_idx])


def _weighted_mode(labels: list[str], weights: list[float]) -> str:
    if not labels:
        return "unknown"

    score: dict[str, float] = {}
    for label, weight in zip(labels, weights, strict=False):
        key = str(label).strip() or "unknown"
        score[key] = score.get(key, 0.0) + float(weight)

    if not score:
        return "unknown"

    return max(score.items(), key=lambda item: (item[1], item[0] != "unknown"))[0]


def _bridge_short_runs(labels: list[str], max_run_length: int) -> list[str]:
    if not labels or max_run_length <= 0:
        return labels

    out = list(labels)
    n = len(out)
    i = 0

    while i < n:
        j = i + 1
        while j < n and out[j] == out[i]:
            j += 1

        run_len = j - i
        prev_label = out[i - 1] if i > 0 else None
        next_label = out[j] if j < n else None

        if (
                run_len <= max_run_length
                and prev_label is not None
                and next_label is not None
                and prev_label == next_label
        ):
            for k in range(i, j):
                out[k] = prev_label

        i = j

    return out


def _smooth_labels(
        labels: list[str],
        confidences: list[float],
        *,
        window: int,
        bridge_short_runs: int,
) -> list[str]:
    if not labels:
        return []

    if window <= 1:
        return _bridge_short_runs(labels, bridge_short_runs)

    half = max(1, window // 2)
    out: list[str] = []

    for i in range(len(labels)):
        lo = max(0, i - half)
        hi = min(len(labels), i + half + 1)

        local_labels = [str(value).strip() or "unknown" for value in labels[lo:hi]]
        local_weights: list[float] = []
        for value in confidences[lo:hi]:
            try:
                weight = float(value)
            except (TypeError, ValueError):
                weight = 0.0
            local_weights.append(weight if np.isfinite(weight) and weight > 0 else 1.0)

        out.append(_weighted_mode(local_labels, local_weights))

    return _bridge_short_runs(out, bridge_short_runs)


def _suppress_short_repositioning_runs(
        labels: list[str],
        *,
        min_reposition_windows: int,
) -> list[str]:
    if not labels:
        return []

    out = list(labels)
    n = len(out)
    i = 0

    while i < n:
        j = i + 1
        while j < n and out[j] == out[i]:
            j += 1

        run_label = out[i]
        run_len = j - i

        if run_label == "repositioning" and run_len < min_reposition_windows:
            prev_label = out[i - 1] if i > 0 else None
            next_label = out[j] if j < n else None

            replacement = None
            if prev_label is not None and next_label is not None and prev_label == next_label:
                replacement = prev_label
            elif prev_label is not None and prev_label != "repositioning":
                replacement = prev_label
            elif next_label is not None and next_label != "repositioning":
                replacement = next_label

            if replacement is not None:
                for k in range(i, j):
                    out[k] = replacement

        i = j

    return out


def _refine_placement_labels(
        labels: list[str],
        confidences: list[float],
        *,
        cfg: PlacementStateConfig,
) -> list[str]:
    if not labels:
        return []

    refined = _smooth_labels(
        labels,
        confidences,
        window=cfg.smoothing_window,
        bridge_short_runs=cfg.bridge_short_runs,
    )
    refined = _suppress_short_repositioning_runs(
        refined,
        min_reposition_windows=cfg.min_reposition_windows,
    )
    refined = _bridge_short_runs(refined, cfg.bridge_short_runs)
    return refined


def _classify_window(
        *,
        acc_std: float,
        gyro_mean: float,
        gyro_peak: float,
        jerk_mean: float,
        dom_freq_hz: float | None,
        cfg: PlacementStateConfig,
) -> tuple[str, float, str]:
    locomotion_like = (
            dom_freq_hz is not None
            and cfg.locomotion_freq_low_hz <= dom_freq_hz <= cfg.locomotion_freq_high_hz
            and acc_std >= cfg.locomotion_min_acc_std
    )

    on_surface_like = (
            acc_std < cfg.on_surface_acc_std_max
            and gyro_mean < cfg.on_surface_gyro_mean_max
            and jerk_mean < cfg.on_surface_jerk_mean_max
    )
    if on_surface_like:
        return "on_surface", 0.94, "very low accel, gyro, and jerk variability"

    strong_reposition_burst = (
            gyro_peak >= cfg.reposition_gyro_peak_min
            or jerk_mean >= cfg.reposition_jerk_mean_min
            or acc_std >= cfg.reposition_acc_std_min
    )

    # Repositioning must be high-evidence and not simply normal rhythmic walking.
    if strong_reposition_burst and not locomotion_like:
        return "repositioning", 0.88, "strong burst pattern inconsistent with stable locomotion"

    if locomotion_like:
        if gyro_mean >= 1.55 or acc_std >= 1.45:
            return "arm_mounted", 0.75, "rhythmic locomotion with strong swing/rotation"
        if gyro_mean >= 0.95:
            return "in_hand", 0.70, "rhythmic locomotion with moderate hand rotation"
        return "in_pocket", 0.80, "rhythmic locomotion with comparatively stable rotation"

    if gyro_mean >= 1.05 and acc_std >= 0.35 and jerk_mean < 12.0:
        return "in_hand", 0.62, "moderate rotational activity without large bursts"

    if 0.15 <= acc_std <= 1.20 and gyro_mean < 0.95:
        return "in_pocket", 0.58, "moderate motion without strong hand swing"

    return "unknown", 0.36, "heuristics inconclusive"


def infer_placement_state_from_dataframe(
        source_df: pd.DataFrame,
        *,
        config: PlacementStateConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Infer placement state from a dataframe already resampled to the target rate."""
    cfg = config or PlacementStateConfig()

    df = _ensure_columns(source_df)
    df = append_derived_channels(df)

    if df.empty:
        return pd.DataFrame(), {
            "window_count": 0,
            "placement_state": "unknown",
            "placement_confidence": 0.0,
            "state_fraction": 0.0,
            "state_counts": {},
            "config": asdict(cfg),
        }

    if "acc_magnitude" not in df.columns:
        df["acc_magnitude"] = np.sqrt(
            pd.to_numeric(df["ax"], errors="coerce").fillna(0.0) ** 2
            + pd.to_numeric(df["ay"], errors="coerce").fillna(0.0) ** 2
            + pd.to_numeric(df["az"], errors="coerce").fillna(0.0) ** 2
        )

    df["gyro_magnitude"] = np.sqrt(
        pd.to_numeric(df["gx"], errors="coerce").fillna(0.0) ** 2
        + pd.to_numeric(df["gy"], errors="coerce").fillna(0.0) ** 2
        + pd.to_numeric(df["gz"], errors="coerce").fillna(0.0) ** 2
    )

    win_size = max(8, int(round(cfg.window_seconds * cfg.target_rate_hz)))
    step = max(4, int(round(cfg.step_seconds * cfg.target_rate_hz)))

    rows: list[dict[str, Any]] = []

    for session_id, group in df.groupby("session_id", dropna=False, sort=False):
        group = group.sort_values("timestamp", kind="stable").reset_index(drop=True)

        for start in range(0, max(1, len(group) - win_size + 1), step):
            chunk = group.iloc[start : start + win_size].copy()
            if len(chunk) < win_size:
                continue

            acc_mag = pd.to_numeric(chunk["acc_magnitude"], errors="coerce").to_numpy(dtype=float)
            gyro_mag = pd.to_numeric(chunk["gyro_magnitude"], errors="coerce").to_numpy(dtype=float)

            acc_std = float(np.nanstd(acc_mag))
            gyro_mean = float(np.nanmean(gyro_mag))
            gyro_peak = float(np.nanmax(gyro_mag))

            if len(acc_mag) > 1:
                jerk = np.abs(np.diff(acc_mag)) * float(cfg.target_rate_hz)
                jerk_mean = float(np.nanmean(jerk))
            else:
                jerk_mean = 0.0

            dom_freq_hz = _dominant_frequency_hz(acc_mag, float(cfg.target_rate_hz))
            state, conf, reason = _classify_window(
                acc_std=acc_std,
                gyro_mean=gyro_mean,
                gyro_peak=gyro_peak,
                jerk_mean=jerk_mean,
                dom_freq_hz=dom_freq_hz,
                cfg=cfg,
            )

            rows.append(
                {
                    "dataset_name": str(chunk["dataset_name"].iloc[0]),
                    "subject_id": str(chunk["subject_id"].iloc[0]),
                    "session_id": str(session_id),
                    "start_ts": float(pd.to_numeric(chunk["timestamp"], errors="coerce").min()),
                    "end_ts": float(pd.to_numeric(chunk["timestamp"], errors="coerce").max()),
                    "midpoint_ts": float(pd.to_numeric(chunk["timestamp"], errors="coerce").mean()),
                    "placement_state": state,
                    "placement_confidence": conf,
                    "placement_reason": reason,
                    "acc_std": acc_std,
                    "gyro_mean": gyro_mean,
                    "gyro_peak": gyro_peak,
                    "jerk_mean": jerk_mean,
                    "dominant_frequency_hz": dom_freq_hz,
                }
            )

    placement_df = pd.DataFrame(rows)
    if placement_df.empty:
        return placement_df, {
            "window_count": 0,
            "placement_state": "unknown",
            "placement_confidence": 0.0,
            "state_fraction": 0.0,
            "state_counts": {},
            "config": asdict(cfg),
        }

    final_labels: list[str] = []
    final_confidences: list[float] = []

    refined_parts: list[pd.DataFrame] = []
    for _, group in placement_df.groupby("session_id", dropna=False, sort=False):
        group = group.sort_values("midpoint_ts", kind="stable").reset_index(drop=True).copy()

        labels = group["placement_state"].astype(str).tolist()
        confidences = (
            pd.to_numeric(group["placement_confidence"], errors="coerce")
            .fillna(0.0)
            .astype(float)
            .tolist()
        )

        refined = _refine_placement_labels(
            labels,
            confidences,
            cfg=cfg,
        )
        group["placement_state_smoothed"] = refined
        refined_parts.append(group)

        final_labels.extend(refined)
        final_confidences.extend(confidences)

    placement_df = pd.concat(refined_parts, ignore_index=True, sort=False)
    placement_df = placement_df.sort_values(
        ["session_id", "midpoint_ts"], kind="stable"
    ).reset_index(drop=True)

    state_counts = (
        placement_df["placement_state_smoothed"]
        .astype(str)
        .value_counts(dropna=False)
        .to_dict()
    )

    top_state = str(
        placement_df["placement_state_smoothed"].astype(str).mode(dropna=True).iloc[0]
    )
    top_mask = placement_df["placement_state_smoothed"].astype(str).eq(top_state)
    top_fraction = float(top_mask.mean())
    top_conf = float(
        pd.to_numeric(
            placement_df.loc[top_mask, "placement_confidence"],
            errors="coerce",
        ).mean()
    )

    summary = {
        "window_count": int(len(placement_df)),
        "placement_state": top_state,
        "placement_confidence": top_conf,
        "state_fraction": top_fraction,
        "state_counts": {str(k): int(v) for k, v in state_counts.items()},
        "config": asdict(cfg),
    }

    return placement_df.reset_index(drop=True), summary
