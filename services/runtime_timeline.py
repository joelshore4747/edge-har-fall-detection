from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(slots=True)
class RuntimeTimelineConfig:
    join_tolerance_seconds: float = 1.25

    smoothing_window: int = 7
    bridge_short_runs: int = 2

    elevated_fall_probability_threshold: float = 0.55
    fall_probability_threshold: float = 0.82

    max_point_gap_seconds: float = 2.5
    merge_same_label_gap_seconds: float = 1.5

    min_activity_event_duration_seconds: float = 2.5
    min_stationary_event_duration_seconds: float = 3.0
    min_placement_event_duration_seconds: float = 3.0
    min_fall_event_duration_seconds: float = 1.0

    min_event_points: int = 2
    min_fall_points: int = 2
    repositioning_min_points: int = 3

    refinement_passes: int = 3

    display_merge_gap_seconds: float = 2.0
    display_min_activity_duration_seconds: float = 4.0
    display_min_stationary_duration_seconds: float = 4.0
    display_min_placement_duration_seconds: float = 5.0
    display_min_fall_duration_seconds: float = 1.5
    display_min_points: int = 3
    display_repositioning_min_duration_seconds: float = 5.0
    display_repositioning_min_points: int = 4
    display_compression_passes: int = 4


@dataclass(slots=True)
class RuntimeTimelineResult:
    point_timeline: pd.DataFrame
    raw_timeline_events: pd.DataFrame
    timeline_events: pd.DataFrame
    transition_events: pd.DataFrame
    session_summaries: pd.DataFrame
    narrative_summary: dict[str, Any]


PLACEMENT_NORMALISATION = {
    "in_pocket": "pocket",
    "pocket": "pocket",
    "in_hand": "hand",
    "hand": "hand",
    "arm_mounted": "hand",
    "on_surface": "desk",
    "desk": "desk",
    "bag": "bag",
    "repositioning": "repositioning",
    "unknown": "unknown",
}

ACTIVITY_NORMALISATION = {
    "walking": "walking",
    "walk": "walking",
    "running": "walking",
    "jogging": "walking",
    "locomotion": "walking",

    "stairs": "stairs",
    "upstairs": "stairs",
    "downstairs": "stairs",
    "walking_upstairs": "stairs",
    "walking_downstairs": "stairs",
    "ascending_stairs": "stairs",
    "descending_stairs": "stairs",
    "stairs_up": "stairs",
    "stairs_down": "stairs",

    "standing": "static",
    "sitting": "static",
    "lying": "static",
    "laying": "static",
    "static": "static",
    "stationary": "static",
    "idle": "static",

    "phone_handling": "other",
    "sit_down_transition": "other",
    "other_transition": "other",
    "other": "other",
    "fall": "other",

    "unknown": "unknown",
}

STATIC_ACTIVITY_LABELS = {"static"}
MOVING_ACTIVITY_LABELS = {"walking", "stairs"}
FINAL_ACTIVITY_LABELS = {"static", "walking", "stairs", "other", "unknown"}


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


def _coerce_bool(value: Any) -> bool | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y"}:
            return True
        if lowered in {"false", "0", "no", "n"}:
            return False
        return None
    try:
        return bool(value)
    except Exception:
        return None


def _coerce_label(value: Any, *, default: str = "unknown") -> str:
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except TypeError:
        pass
    text = str(value).strip().lower()
    return text if text else default

def _normalise_activity_label(value: Any) -> str:
    label = _coerce_label(value)
    return ACTIVITY_NORMALISATION.get(label, "other" if label != "unknown" else "unknown")

def _normalise_placement_label(value: Any) -> str:
    label = _coerce_label(value)
    return PLACEMENT_NORMALISATION.get(label, label)


def _humanise_label(label: str) -> str:
    text = str(label).strip().replace("_", " ")
    return text if text else "unknown"


def _weighted_mode(labels: list[str], weights: list[float]) -> str:
    score: dict[str, float] = {}
    for label, weight in zip(labels, weights, strict=False):
        clean_label = _coerce_label(label)
        score[clean_label] = score.get(clean_label, 0.0) + float(weight)
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


def smooth_label_sequence(
        labels: list[str],
        *,
        weights: list[float] | None = None,
        window: int = 7,
        bridge_short_runs: int = 2,
) -> list[str]:
    if not labels:
        return []

    base_labels = [_coerce_label(label) for label in labels]
    if window <= 1:
        return _bridge_short_runs(base_labels, bridge_short_runs)

    half = max(1, window // 2)
    weights = weights or [1.0] * len(base_labels)
    smoothed: list[str] = []

    for idx in range(len(base_labels)):
        lo = max(0, idx - half)
        hi = min(len(base_labels), idx + half + 1)

        local_labels = base_labels[lo:hi]
        local_weights: list[float] = []
        for value in weights[lo:hi]:
            weight = _coerce_float(value)
            local_weights.append(weight if weight is not None and weight > 0 else 1.0)

        smoothed.append(_weighted_mode(local_labels, local_weights))

    return _bridge_short_runs(smoothed, bridge_short_runs)


def _standardise_har_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "session_id",
                "dataset_name",
                "subject_id",
                "midpoint_ts",
                "activity_label",
                "activity_confidence",
            ]
        )

    out = pd.DataFrame()
    out["session_id"] = df.get("session_id", "unknown_session").astype(str)
    out["dataset_name"] = df.get("dataset_name", "unknown_dataset").astype(str)
    out["subject_id"] = df.get("subject_id", "unknown_subject").astype(str)
    out["midpoint_ts"] = pd.to_numeric(df.get("midpoint_ts"), errors="coerce")
    out["activity_label"] = df.get("predicted_label", "unknown").map(_normalise_activity_label)
    out["activity_confidence"] = pd.to_numeric(df.get("predicted_confidence"), errors="coerce")
    out = out.dropna(subset=["midpoint_ts"]).reset_index(drop=True)
    return out


def _standardise_fall_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "session_id",
                "dataset_name",
                "subject_id",
                "midpoint_ts",
                "fall_label",
                "fall_probability",
                "predicted_is_fall",
            ]
        )

    out = pd.DataFrame()
    out["session_id"] = df.get("session_id", "unknown_session").astype(str)
    out["dataset_name"] = df.get("dataset_name", "unknown_dataset").astype(str)
    out["subject_id"] = df.get("subject_id", "unknown_subject").astype(str)
    out["midpoint_ts"] = pd.to_numeric(df.get("midpoint_ts"), errors="coerce")
    out["fall_label"] = df.get("predicted_label", "unknown").map(_coerce_label)
    out["fall_probability"] = pd.to_numeric(df.get("predicted_probability"), errors="coerce")

    if "predicted_is_fall" in df.columns:
        out["predicted_is_fall"] = df["predicted_is_fall"].map(_coerce_bool)
    else:
        out["predicted_is_fall"] = out["fall_label"].eq("fall")

    out = out.dropna(subset=["midpoint_ts"]).reset_index(drop=True)
    return out


def _standardise_placement_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "session_id",
                "dataset_name",
                "subject_id",
                "midpoint_ts",
                "placement_label",
                "placement_confidence",
            ]
        )

    if "placement_state_smoothed" in df.columns:
        placement_series = df["placement_state_smoothed"]
    else:
        placement_series = df.get("placement_state", pd.Series([], dtype="string"))

    out = pd.DataFrame()
    out["session_id"] = df.get("session_id", "unknown_session").astype(str)
    out["dataset_name"] = df.get("dataset_name", "unknown_dataset").astype(str)
    out["subject_id"] = df.get("subject_id", "unknown_subject").astype(str)
    out["midpoint_ts"] = pd.to_numeric(df.get("midpoint_ts"), errors="coerce")
    out["placement_label"] = placement_series.map(_normalise_placement_label)
    out["placement_confidence"] = pd.to_numeric(df.get("placement_confidence"), errors="coerce")
    out = out.dropna(subset=["midpoint_ts"]).reset_index(drop=True)
    return out


def _standardise_grouped_fall_events(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "event_id",
                "session_id",
                "dataset_name",
                "subject_id",
                "event_start_ts",
                "event_end_ts",
                "peak_probability",
                "mean_probability",
                "median_probability",
            ]
        )

    out = pd.DataFrame()
    out["event_id"] = df.get("event_id", pd.Series(dtype="string")).astype(str)
    out["session_id"] = df.get("session_id", "unknown_session").astype(str)
    out["dataset_name"] = df.get("dataset_name", "unknown_dataset").astype(str)
    out["subject_id"] = df.get("subject_id", "unknown_subject").astype(str)
    out["event_start_ts"] = pd.to_numeric(df.get("event_start_ts"), errors="coerce")
    out["event_end_ts"] = pd.to_numeric(df.get("event_end_ts"), errors="coerce")
    out["peak_probability"] = pd.to_numeric(df.get("peak_probability"), errors="coerce")
    out["mean_probability"] = pd.to_numeric(df.get("mean_probability"), errors="coerce")
    out["median_probability"] = pd.to_numeric(df.get("median_probability"), errors="coerce")
    out = out.dropna(subset=["event_start_ts", "event_end_ts"]).reset_index(drop=True)
    return out


def _build_base_points(
        har_df: pd.DataFrame,
        fall_df: pd.DataFrame,
        placement_df: pd.DataFrame,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []

    for frame in (har_df, fall_df, placement_df):
        if frame.empty:
            continue
        part = frame[["session_id", "dataset_name", "subject_id", "midpoint_ts"]].copy()
        parts.append(part)

    if not parts:
        return pd.DataFrame(
            columns=["session_id", "dataset_name", "subject_id", "midpoint_ts"]
        )

    base = pd.concat(parts, ignore_index=True, sort=False)
    base = base.dropna(subset=["midpoint_ts"]).copy()
    base["session_id"] = base["session_id"].astype(str)
    base["dataset_name"] = base["dataset_name"].astype(str)
    base["subject_id"] = base["subject_id"].astype(str)
    base["midpoint_ts"] = pd.to_numeric(base["midpoint_ts"], errors="coerce")
    base = base.dropna(subset=["midpoint_ts"]).copy()
    base = (
        base.sort_values(["session_id", "midpoint_ts"], kind="stable")
        .drop_duplicates(subset=["session_id", "midpoint_ts"], keep="first")
        .reset_index(drop=True)
    )
    return base


def _merge_nearest_fields(
        base: pd.DataFrame,
        source: pd.DataFrame,
        *,
        tolerance_seconds: float,
        columns: list[str],
) -> pd.DataFrame:
    if base.empty or source.empty:
        return base.copy()

    merged_parts: list[pd.DataFrame] = []

    source = source.copy()
    source["midpoint_ts"] = pd.to_numeric(source["midpoint_ts"], errors="coerce").astype(float)
    source = source.dropna(subset=["midpoint_ts"])
    source = source.sort_values(["session_id", "midpoint_ts"], kind="stable").reset_index(drop=True)

    for session_id, base_group in base.groupby("session_id", dropna=False, sort=False):
        base_group = base_group.copy()
        base_group["midpoint_ts"] = pd.to_numeric(base_group["midpoint_ts"], errors="coerce").astype(float)
        base_group = base_group.dropna(subset=["midpoint_ts"])
        base_group = base_group.sort_values("midpoint_ts", kind="stable").reset_index(drop=True)

        source_group = source[source["session_id"].astype(str) == str(session_id)].copy()

        if source_group.empty:
            merged_parts.append(base_group)
            continue

        source_group["midpoint_ts"] = pd.to_numeric(
            source_group["midpoint_ts"], errors="coerce"
        ).astype(float)
        source_group = source_group.dropna(subset=["midpoint_ts"])
        source_group = source_group.sort_values("midpoint_ts", kind="stable").reset_index(drop=True)

        keep = ["midpoint_ts", *columns]
        merged = pd.merge_asof(
            base_group,
            source_group[keep],
            on="midpoint_ts",
            direction="nearest",
            tolerance=float(tolerance_seconds),
        )
        merged_parts.append(merged)

    out = pd.concat(merged_parts, ignore_index=True, sort=False)
    return out.sort_values(["session_id", "midpoint_ts"], kind="stable").reset_index(drop=True)


def _segment_boundaries(
        times: list[float],
        activities: list[str],
        placements: list[str],
        elevated: list[bool],
        *,
        max_gap_seconds: float,
) -> list[tuple[int, int]]:
    if not times:
        return []

    bounds: list[tuple[int, int]] = []
    start = 0

    for idx in range(1, len(times)):
        gap = float(times[idx]) - float(times[idx - 1])
        changed = (
                activities[idx] != activities[idx - 1]
                or placements[idx] != placements[idx - 1]
                or elevated[idx] != elevated[idx - 1]
                or gap > float(max_gap_seconds)
        )
        if changed:
            bounds.append((start, idx))
            start = idx

    bounds.append((start, len(times)))
    return bounds


def _segment_duration(times: list[float], start: int, end: int) -> float:
    if end - start <= 1:
        return 0.0
    return max(0.0, float(times[end - 1]) - float(times[start]))


def _broad_activity_group(label: str) -> str:
    clean = _normalise_activity_label(label)
    if clean in MOVING_ACTIVITY_LABELS:
        return "moving"
    if clean in STATIC_ACTIVITY_LABELS:
        return "static"
    if clean == "unknown":
        return "unknown"
    return clean


def _refine_session_points(session_df: pd.DataFrame, cfg: RuntimeTimelineConfig) -> pd.DataFrame:
    if session_df.empty:
        return session_df

    working = session_df.sort_values("midpoint_ts", kind="stable").reset_index(drop=True).copy()

    times = working["midpoint_ts"].astype(float).tolist()
    activities = working["smoothed_activity_label"].astype(str).tolist()
    placements = working["smoothed_placement_label"].astype(str).tolist()
    fall_probs = pd.to_numeric(working["fall_probability"], errors="coerce").fillna(0.0).tolist()
    elevated = working["elevated_fall"].fillna(False).astype(bool).tolist()

    for _ in range(max(1, cfg.refinement_passes)):
        changed_any = False
        bounds = _segment_boundaries(
            times,
            activities,
            placements,
            elevated,
            max_gap_seconds=cfg.max_point_gap_seconds,
        )

        for seg_idx, (start, end) in enumerate(bounds):
            point_count = end - start
            duration = _segment_duration(times, start, end)
            activity = activities[start]
            placement = placements[start]
            segment_elevated = elevated[start]
            peak_prob = max(fall_probs[start:end]) if point_count > 0 else 0.0

            prev_bounds = bounds[seg_idx - 1] if seg_idx > 0 else None
            next_bounds = bounds[seg_idx + 1] if seg_idx + 1 < len(bounds) else None

            prev_activity = activities[prev_bounds[0]] if prev_bounds else None
            next_activity = activities[next_bounds[0]] if next_bounds else None
            prev_placement = placements[prev_bounds[0]] if prev_bounds else None
            next_placement = placements[next_bounds[0]] if next_bounds else None
            prev_elevated = elevated[prev_bounds[0]] if prev_bounds else None
            next_elevated = elevated[next_bounds[0]] if next_bounds else None

            short_activity = (
                    point_count < cfg.min_event_points
                    or (
                            activity not in STATIC_ACTIVITY_LABELS
                            and duration < cfg.min_activity_event_duration_seconds
                    )
                    or (
                            activity in STATIC_ACTIVITY_LABELS
                            and duration < cfg.min_stationary_event_duration_seconds
                    )
            )

            short_placement = (
                    point_count < cfg.min_event_points
                    or duration < cfg.min_placement_event_duration_seconds
            )

            short_reposition = (
                    placement == "repositioning"
                    and (
                            point_count < cfg.repositioning_min_points
                            or duration < cfg.min_placement_event_duration_seconds
                    )
            )

            weak_fall_segment = (
                    segment_elevated
                    and (
                            point_count < cfg.min_fall_points
                            or duration < cfg.min_fall_event_duration_seconds
                            or peak_prob < cfg.fall_probability_threshold
                    )
            )

            if weak_fall_segment:
                replacement = False
                if prev_elevated is False and next_elevated is False:
                    for idx in range(start, end):
                        elevated[idx] = False
                    replacement = True
                elif peak_prob < cfg.elevated_fall_probability_threshold:
                    for idx in range(start, end):
                        elevated[idx] = False
                    replacement = True

                if replacement:
                    changed_any = True
                    continue

            if short_reposition:
                replacement_label: str | None = None
                if (
                        prev_placement is not None
                        and next_placement is not None
                        and prev_placement == next_placement
                        and prev_placement != "repositioning"
                ):
                    replacement_label = prev_placement
                elif prev_placement is not None and prev_placement != "repositioning":
                    replacement_label = prev_placement
                elif next_placement is not None and next_placement != "repositioning":
                    replacement_label = next_placement

                if replacement_label is not None:
                    for idx in range(start, end):
                        placements[idx] = replacement_label
                    changed_any = True
                    continue

            if (
                    short_placement
                    and prev_placement is not None
                    and next_placement is not None
                    and prev_placement == next_placement
                    and placement != prev_placement
            ):
                for idx in range(start, end):
                    placements[idx] = prev_placement
                changed_any = True
                continue

            if (
                    short_activity
                    and prev_activity is not None
                    and next_activity is not None
                    and prev_activity == next_activity
                    and activity != prev_activity
            ):
                for idx in range(start, end):
                    activities[idx] = prev_activity
                changed_any = True
                continue

            if (
                    prev_activity is not None
                    and next_activity is not None
                    and _broad_activity_group(prev_activity) == _broad_activity_group(next_activity)
                    and prev_placement is not None
                    and next_placement is not None
                    and prev_placement == next_placement
                    and prev_elevated is not None
                    and next_elevated is not None
                    and prev_elevated == next_elevated
                    and (point_count <= cfg.bridge_short_runs + 1 or duration < cfg.min_activity_event_duration_seconds)
            ):
                replacement_activity = prev_activity
                if _broad_activity_group(prev_activity) == "moving":
                    replacement_activity = prev_activity

                for idx in range(start, end):
                    activities[idx] = replacement_activity
                    placements[idx] = prev_placement
                    elevated[idx] = prev_elevated
                changed_any = True
                continue

        if not changed_any:
            break

    working["activity_label"] = activities
    working["placement_label"] = placements
    working["elevated_fall"] = elevated
    return working


def _build_point_timeline(
        har_df: pd.DataFrame,
        fall_df: pd.DataFrame,
        placement_df: pd.DataFrame,
        *,
        config: RuntimeTimelineConfig,
) -> pd.DataFrame:
    base = _build_base_points(har_df, fall_df, placement_df)
    if base.empty:
        return pd.DataFrame()

    out = _merge_nearest_fields(
        base,
        har_df,
        tolerance_seconds=config.join_tolerance_seconds,
        columns=["activity_label", "activity_confidence"],
    )
    out = _merge_nearest_fields(
        out,
        placement_df,
        tolerance_seconds=config.join_tolerance_seconds,
        columns=["placement_label", "placement_confidence"],
    )
    out = _merge_nearest_fields(
        out,
        fall_df,
        tolerance_seconds=config.join_tolerance_seconds,
        columns=["fall_label", "fall_probability", "predicted_is_fall"],
    )

    out["activity_label"] = out["activity_label"].map(_coerce_label)
    out["placement_label"] = out["placement_label"].map(_normalise_placement_label)
    out["activity_confidence"] = pd.to_numeric(out["activity_confidence"], errors="coerce")
    out["placement_confidence"] = pd.to_numeric(out["placement_confidence"], errors="coerce")
    out["fall_probability"] = pd.to_numeric(out["fall_probability"], errors="coerce")
    out["predicted_is_fall"] = out["predicted_is_fall"].map(_coerce_bool)

    parts: list[pd.DataFrame] = []
    for _, group in out.groupby("session_id", dropna=False, sort=False):
        group = group.sort_values("midpoint_ts", kind="stable").reset_index(drop=True).copy()

        activity_labels = group["activity_label"].astype(str).tolist()
        activity_weights = group["activity_confidence"].fillna(1.0).tolist()
        placement_labels = group["placement_label"].astype(str).tolist()
        placement_weights = group["placement_confidence"].fillna(1.0).tolist()

        group["smoothed_activity_label"] = smooth_label_sequence(
            activity_labels,
            weights=activity_weights,
            window=config.smoothing_window,
            bridge_short_runs=config.bridge_short_runs,
        )
        group["smoothed_placement_label"] = smooth_label_sequence(
            placement_labels,
            weights=placement_weights,
            window=config.smoothing_window,
            bridge_short_runs=config.bridge_short_runs,
        )

        group["elevated_fall"] = (
                                         group["fall_probability"].fillna(0.0)
                                         >= float(config.elevated_fall_probability_threshold)
                                 ) | group["predicted_is_fall"].fillna(False)

        refined = _refine_session_points(group, config)
        parts.append(refined)

    point_timeline = pd.concat(parts, ignore_index=True, sort=False)
    point_timeline = point_timeline.sort_values(["session_id", "midpoint_ts"], kind="stable").reset_index(drop=True)
    return point_timeline


def _determine_event_kind(
        *,
        activity_label: str,
        placement_label: str,
        likely_fall: bool,
) -> str:
    if likely_fall:
        return "fall_like_event"
    if placement_label == "repositioning":
        return "placement_change"
    if activity_label in STATIC_ACTIVITY_LABELS:
        return "stationary"
    return "activity"


def _describe_event(
        *,
        activity_label: str,
        placement_label: str,
        event_kind: str,
        likely_fall: bool,
) -> str:
    activity_text = _humanise_label(activity_label)
    placement_text = _humanise_label(placement_label)

    if likely_fall or event_kind == "fall_like_event":
        if placement_label == "unknown":
            return f"Fall-like event detected during {activity_text}."
        return f"Fall-like event detected during {activity_text} with phone in {placement_text}."

    if event_kind == "placement_change":
        if placement_label == "repositioning":
            return "Phone repositioning detected."
        return f"Placement changed; phone now appears to be in {placement_text}."

    if event_kind == "stationary":
        if placement_label == "unknown":
            return f"Stationary period ({activity_text})."
        return f"Stationary period ({activity_text}) with phone in {placement_text}."

    if placement_label == "unknown":
        return f"{activity_text.capitalize()} detected."
    return f"{activity_text.capitalize()} with phone in {placement_text}."


def _overlapping_grouped_falls(
        grouped_fall_events: pd.DataFrame,
        *,
        session_id: str,
        start_ts: float,
        end_ts: float,
) -> pd.DataFrame:
    if grouped_fall_events.empty:
        return pd.DataFrame()

    group = grouped_fall_events[grouped_fall_events["session_id"].astype(str) == str(session_id)].copy()
    if group.empty:
        return group

    overlap_mask = (group["event_start_ts"] <= float(end_ts)) & (
            group["event_end_ts"] >= float(start_ts)
    )
    return group[overlap_mask].reset_index(drop=True)


def _build_timeline_events(
        point_timeline: pd.DataFrame,
        grouped_fall_events: pd.DataFrame,
        *,
        config: RuntimeTimelineConfig,
) -> pd.DataFrame:
    if point_timeline.empty:
        return pd.DataFrame()

    events: list[dict[str, Any]] = []

    for session_id, group in point_timeline.groupby("session_id", dropna=False, sort=False):
        group = group.sort_values("midpoint_ts", kind="stable").reset_index(drop=True)
        if group.empty:
            continue

        times = group["midpoint_ts"].astype(float).tolist()
        activities = group["activity_label"].astype(str).tolist()
        placements = group["placement_label"].astype(str).tolist()
        elevated = group["elevated_fall"].fillna(False).astype(bool).tolist()
        bounds = _segment_boundaries(
            times,
            activities,
            placements,
            elevated,
            max_gap_seconds=config.max_point_gap_seconds,
        )

        event_counter = 0
        for start, end in bounds:
            seg = group.iloc[start:end].copy()
            if seg.empty:
                continue

            start_ts = float(seg["midpoint_ts"].iloc[0])
            end_ts = float(seg["midpoint_ts"].iloc[-1])
            duration = max(0.0, end_ts - start_ts)
            point_count = int(len(seg))

            activity_label = _coerce_label(seg["activity_label"].iloc[0])
            placement_label = _normalise_placement_label(seg["placement_label"].iloc[0])

            point_peak_prob = (
                float(seg["fall_probability"].max(skipna=True))
                if seg["fall_probability"].notna().any()
                else np.nan
            )
            point_mean_prob = (
                float(seg["fall_probability"].mean(skipna=True))
                if seg["fall_probability"].notna().any()
                else np.nan
            )
            elevated_count = int(seg["elevated_fall"].fillna(False).astype(bool).sum())

            overlap = _overlapping_grouped_falls(
                grouped_fall_events,
                session_id=str(session_id),
                start_ts=start_ts,
                end_ts=end_ts,
            )

            grouped_peak_prob = (
                float(overlap["peak_probability"].max(skipna=True))
                if not overlap.empty and overlap["peak_probability"].notna().any()
                else np.nan
            )

            likely_fall = False
            if not overlap.empty:
                likely_fall = True
            elif (
                    elevated_count >= int(config.min_fall_points)
                    and duration >= float(config.min_fall_event_duration_seconds)
                    and pd.notna(point_peak_prob)
                    and point_peak_prob >= float(config.fall_probability_threshold)
            ):
                likely_fall = True

            event_kind = _determine_event_kind(
                activity_label=activity_label,
                placement_label=placement_label,
                likely_fall=likely_fall,
            )

            event_counter += 1
            related_ids = overlap["event_id"].astype(str).tolist() if not overlap.empty else []

            events.append(
                {
                    "event_id": f"{session_id}_timeline_{event_counter:03d}",
                    "session_id": str(session_id),
                    "dataset_name": str(seg["dataset_name"].iloc[0]),
                    "subject_id": str(seg["subject_id"].iloc[0]),
                    "start_ts": start_ts,
                    "end_ts": end_ts,
                    "duration_seconds": duration,
                    "midpoint_ts": (start_ts + end_ts) / 2.0,
                    "point_count": point_count,
                    "activity_label": activity_label,
                    "placement_label": placement_label,
                    "activity_confidence_mean": float(seg["activity_confidence"].mean(skipna=True))
                    if seg["activity_confidence"].notna().any()
                    else np.nan,
                    "placement_confidence_mean": float(seg["placement_confidence"].mean(skipna=True))
                    if seg["placement_confidence"].notna().any()
                    else np.nan,
                    "fall_probability_peak": max(
                        [v for v in [point_peak_prob, grouped_peak_prob] if pd.notna(v)],
                        default=np.nan,
                    ),
                    "fall_probability_mean": point_mean_prob,
                    "likely_fall": likely_fall,
                    "event_kind": event_kind,
                    "related_grouped_fall_event_ids": related_ids,
                    "description": _describe_event(
                        activity_label=activity_label,
                        placement_label=placement_label,
                        event_kind=event_kind,
                        likely_fall=likely_fall,
                    ),
                }
            )

    out = pd.DataFrame(events)
    if out.empty:
        return out

    return out.sort_values(["session_id", "start_ts"], kind="stable").reset_index(drop=True)


def _combine_related_ids(a: Any, b: Any) -> list[str]:
    out: list[str] = []
    for raw in (a, b):
        if isinstance(raw, list):
            out.extend(str(item) for item in raw)
    return sorted(set(out))


def _merge_event_dicts(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    merged = dict(left)
    merged["end_ts"] = float(right["end_ts"])
    merged["duration_seconds"] = max(0.0, float(merged["end_ts"]) - float(merged["start_ts"]))
    merged["midpoint_ts"] = (float(merged["start_ts"]) + float(merged["end_ts"])) / 2.0
    merged["point_count"] = int(left.get("point_count", 0)) + int(right.get("point_count", 0))

    for key in ["activity_confidence_mean", "placement_confidence_mean", "fall_probability_mean"]:
        a = _coerce_float(left.get(key))
        b = _coerce_float(right.get(key))
        vals = [v for v in [a, b] if v is not None]
        merged[key] = float(np.mean(vals)) if vals else np.nan

    a_peak = _coerce_float(left.get("fall_probability_peak"))
    b_peak = _coerce_float(right.get("fall_probability_peak"))
    peaks = [v for v in [a_peak, b_peak] if v is not None]
    merged["fall_probability_peak"] = max(peaks) if peaks else np.nan

    merged["likely_fall"] = bool(left.get("likely_fall")) or bool(right.get("likely_fall"))
    merged["related_grouped_fall_event_ids"] = _combine_related_ids(
        left.get("related_grouped_fall_event_ids"),
        right.get("related_grouped_fall_event_ids"),
    )
    return merged


def _is_short_display_event(event: dict[str, Any], cfg: RuntimeTimelineConfig) -> bool:
    duration = _coerce_float(event.get("duration_seconds")) or 0.0
    point_count = int(event.get("point_count", 0))
    event_kind = _coerce_label(event.get("event_kind"))
    placement = _normalise_placement_label(event.get("placement_label"))
    likely_fall = bool(event.get("likely_fall"))

    if likely_fall or event_kind == "fall_like_event":
        return duration < cfg.display_min_fall_duration_seconds or point_count < cfg.min_fall_points

    if placement == "repositioning":
        return (
                duration < cfg.display_repositioning_min_duration_seconds
                or point_count < cfg.display_repositioning_min_points
        )

    if event_kind == "placement_change":
        return (
                duration < cfg.display_min_placement_duration_seconds
                or point_count < cfg.display_min_points
        )

    activity = _coerce_label(event.get("activity_label"))
    if activity in STATIC_ACTIVITY_LABELS:
        return (
                duration < cfg.display_min_stationary_duration_seconds
                or point_count < cfg.display_min_points
        )

    return duration < cfg.display_min_activity_duration_seconds or point_count < cfg.display_min_points


def _events_compatible_for_merge(
        left: dict[str, Any],
        right: dict[str, Any],
        cfg: RuntimeTimelineConfig,
) -> bool:
    gap = float(right["start_ts"]) - float(left["end_ts"])
    if gap > float(cfg.display_merge_gap_seconds):
        return False

    left_fall = bool(left.get("likely_fall"))
    right_fall = bool(right.get("likely_fall"))
    if left_fall != right_fall:
        return False
    if left_fall and right_fall:
        return True

    left_kind = _coerce_label(left.get("event_kind"))
    right_kind = _coerce_label(right.get("event_kind"))
    left_activity = _coerce_label(left.get("activity_label"))
    right_activity = _coerce_label(right.get("activity_label"))
    left_place = _normalise_placement_label(left.get("placement_label"))
    right_place = _normalise_placement_label(right.get("placement_label"))

    if left_activity == right_activity and left_place == right_place:
        return True

    if (
            _broad_activity_group(left_activity) == "moving"
            and _broad_activity_group(right_activity) == "moving"
            and left_place == right_place
            and left_kind == "activity"
            and right_kind == "activity"
    ):
        return True

    if (
            left_kind == "stationary"
            and right_kind == "stationary"
            and left_place == right_place
    ):
        return True

    return False


def _compress_session_event_rows(
        rows: list[dict[str, Any]],
        cfg: RuntimeTimelineConfig,
) -> list[dict[str, Any]]:
    if not rows:
        return rows

    current_rows = [dict(row) for row in rows]

    for _ in range(max(1, cfg.display_compression_passes)):
        changed = False

        # Pass 1: merge obviously compatible neighbors.
        merged: list[dict[str, Any]] = []
        current = current_rows[0]

        for row in current_rows[1:]:
            if _events_compatible_for_merge(current, row, cfg):
                current = _merge_event_dicts(current, row)
                changed = True
            else:
                merged.append(current)
                current = row
        merged.append(current)
        current_rows = merged

        if len(current_rows) <= 1:
            break

        # Pass 2: absorb short middle events into neighbors when possible.
        i = 0
        absorbed: list[dict[str, Any]] = []
        while i < len(current_rows):
            row = current_rows[i]

            if i == 0 or i == len(current_rows) - 1:
                absorbed.append(row)
                i += 1
                continue

            if not _is_short_display_event(row, cfg):
                absorbed.append(row)
                i += 1
                continue

            prev_row = absorbed[-1]
            next_row = current_rows[i + 1]

            can_merge_prev_next = _events_compatible_for_merge(prev_row, next_row, cfg)

            if can_merge_prev_next:
                merged_prev = absorbed.pop()
                merged_prev = _merge_event_dicts(merged_prev, row)
                merged_prev = _merge_event_dicts(merged_prev, next_row)
                absorbed.append(merged_prev)
                changed = True
                i += 2
                continue

            if _events_compatible_for_merge(prev_row, row, cfg):
                merged_prev = absorbed.pop()
                absorbed.append(_merge_event_dicts(merged_prev, row))
                changed = True
                i += 1
                continue

            if _events_compatible_for_merge(row, next_row, cfg):
                current_rows[i + 1] = _merge_event_dicts(row, next_row)
                changed = True
                i += 1
                continue

            absorbed.append(row)
            i += 1

        current_rows = absorbed

        if not changed:
            break

    return current_rows


def _refresh_event_fields(events_df: pd.DataFrame) -> pd.DataFrame:
    if events_df.empty:
        return events_df

    refreshed_parts: list[pd.DataFrame] = []

    for session_id, group in events_df.groupby("session_id", dropna=False, sort=False):
        group = group.sort_values("start_ts", kind="stable").reset_index(drop=True).copy()

        for idx in range(len(group)):
            group.at[idx, "event_id"] = f"{session_id}_timeline_{idx + 1:03d}"
            activity = _coerce_label(group.at[idx, "activity_label"])
            placement = _normalise_placement_label(group.at[idx, "placement_label"])
            likely_fall = bool(group.at[idx, "likely_fall"])
            kind = _determine_event_kind(
                activity_label=activity,
                placement_label=placement,
                likely_fall=likely_fall,
            )
            group.at[idx, "event_kind"] = kind
            group.at[idx, "description"] = _describe_event(
                activity_label=activity,
                placement_label=placement,
                event_kind=kind,
                likely_fall=likely_fall,
            )
            group.at[idx, "duration_seconds"] = max(
                0.0, float(group.at[idx, "end_ts"]) - float(group.at[idx, "start_ts"])
            )
            group.at[idx, "midpoint_ts"] = (
                                                   float(group.at[idx, "start_ts"]) + float(group.at[idx, "end_ts"])
                                           ) / 2.0

        refreshed_parts.append(group)

    return pd.concat(refreshed_parts, ignore_index=True, sort=False).sort_values(
        ["session_id", "start_ts"], kind="stable"
    ).reset_index(drop=True)


def _compress_timeline_events(
        events_df: pd.DataFrame,
        *,
        config: RuntimeTimelineConfig,
) -> pd.DataFrame:
    if events_df.empty:
        return events_df

    compressed_parts: list[pd.DataFrame] = []

    for _, group in events_df.groupby("session_id", dropna=False, sort=False):
        group = group.sort_values("start_ts", kind="stable").reset_index(drop=True)
        rows = group.to_dict(orient="records")
        compressed_rows = _compress_session_event_rows(rows, config)
        compressed_parts.append(pd.DataFrame(compressed_rows))

    out = pd.concat(compressed_parts, ignore_index=True, sort=False)
    out = _refresh_event_fields(out)
    return out


def detect_timeline_transitions(events_df: pd.DataFrame) -> pd.DataFrame:
    empty_columns = [
        "transition_id",
        "session_id",
        "dataset_name",
        "subject_id",
        "transition_ts",
        "from_event_id",
        "to_event_id",
        "transition_kind",
        "from_activity_label",
        "to_activity_label",
        "from_placement_label",
        "to_placement_label",
        "description",
    ]

    if events_df.empty:
        return pd.DataFrame(columns=empty_columns)

    transitions: list[dict[str, Any]] = []

    for session_id, group in events_df.groupby("session_id", dropna=False, sort=False):
        group = group.sort_values("start_ts", kind="stable").reset_index(drop=True)
        for idx in range(1, len(group)):
            prev = group.iloc[idx - 1]
            curr = group.iloc[idx]

            from_activity = _coerce_label(prev["activity_label"])
            to_activity = _coerce_label(curr["activity_label"])
            from_placement = _normalise_placement_label(prev["placement_label"])
            to_placement = _normalise_placement_label(curr["placement_label"])

            activity_changed = from_activity != to_activity
            placement_changed = from_placement != to_placement
            fall_state_changed = bool(prev["likely_fall"]) != bool(curr["likely_fall"])

            if not (activity_changed or placement_changed or fall_state_changed):
                continue

            if activity_changed and placement_changed:
                transition_kind = "activity_and_placement_change"
                description = (
                    f"Activity changed from {_humanise_label(from_activity)} to "
                    f"{_humanise_label(to_activity)} and placement changed from "
                    f"{_humanise_label(from_placement)} to {_humanise_label(to_placement)}."
                )
            elif activity_changed:
                transition_kind = "activity_change"
                description = (
                    f"Activity changed from {_humanise_label(from_activity)} "
                    f"to {_humanise_label(to_activity)}."
                )
            elif placement_changed:
                transition_kind = "placement_change"
                description = (
                    f"Placement changed from {_humanise_label(from_placement)} "
                    f"to {_humanise_label(to_placement)}."
                )
            else:
                transition_kind = "fall_state_change"
                description = "Fall state changed across consecutive timeline events."

            transitions.append(
                {
                    "transition_id": f"{session_id}_transition_{idx:03d}",
                    "session_id": str(session_id),
                    "dataset_name": str(curr["dataset_name"]),
                    "subject_id": str(curr["subject_id"]),
                    "transition_ts": float(_coerce_float(curr["start_ts"]) or 0.0),
                    "from_event_id": str(prev["event_id"]),
                    "to_event_id": str(curr["event_id"]),
                    "transition_kind": transition_kind,
                    "from_activity_label": from_activity,
                    "to_activity_label": to_activity,
                    "from_placement_label": from_placement,
                    "to_placement_label": to_placement,
                    "description": description,
                }
            )

    out = pd.DataFrame(transitions)
    if out.empty:
        return pd.DataFrame(columns=empty_columns)
    return out.sort_values(["session_id", "transition_ts"], kind="stable").reset_index(drop=True)


def _dominant_label_by_duration(
        events_df: pd.DataFrame,
        label_col: str,
        *,
        exclude_unknown: bool = True,
) -> str:
    if events_df.empty or label_col not in events_df.columns:
        return "unknown"

    working = events_df.copy()
    working[label_col] = working[label_col].map(_coerce_label)
    if exclude_unknown:
        working = working[working[label_col] != "unknown"].copy()
    if working.empty:
        return "unknown"

    duration_by_label = (
        working.groupby(label_col)["duration_seconds"]
        .sum(min_count=1)
        .sort_values(ascending=False)
    )
    if duration_by_label.empty:
        return "unknown"
    return str(duration_by_label.index[0])


def build_session_narrative_summary(
        events_df: pd.DataFrame,
        transition_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if events_df.empty:
        empty = pd.DataFrame(
            columns=[
                "session_id",
                "dataset_name",
                "subject_id",
                "total_duration_seconds",
                "event_count",
                "transition_count",
                "fall_event_count",
                "dominant_activity_label",
                "dominant_placement_label",
                "highest_fall_probability",
                "summary_text",
            ]
        )
        return empty, {
            "session_count": 0,
            "total_event_count": 0,
            "total_transition_count": 0,
            "total_fall_event_count": 0,
            "sessions": [],
        }

    summaries: list[dict[str, Any]] = []

    for session_id, group in events_df.groupby("session_id", dropna=False, sort=False):
        group = group.sort_values("start_ts", kind="stable").reset_index(drop=True)
        if transition_df.empty or "session_id" not in transition_df.columns:
            trans = pd.DataFrame()
        else:
            trans = transition_df[
                transition_df["session_id"].astype(str) == str(session_id)
                ].copy()

        dominant_activity = _dominant_label_by_duration(group, "activity_label")
        dominant_placement = _dominant_label_by_duration(group, "placement_label")
        fall_event_count = int(group["likely_fall"].fillna(False).astype(bool).sum())
        highest_fall_probability = (
            float(group["fall_probability_peak"].max(skipna=True))
            if group["fall_probability_peak"].notna().any()
            else np.nan
        )
        total_duration_seconds = float(
            pd.to_numeric(group["duration_seconds"], errors="coerce").fillna(0.0).sum()
        )

        summary_text = (
            f"Mostly {_humanise_label(dominant_activity)} with phone in "
            f"{_humanise_label(dominant_placement)}; "
            f"{len(group)} timeline events, {len(trans)} transitions"
        )
        if fall_event_count > 0:
            summary_text += f", {fall_event_count} fall-like event(s)."
        else:
            summary_text += ", no fall-like events."

        summaries.append(
            {
                "session_id": str(session_id),
                "dataset_name": str(group["dataset_name"].iloc[0]),
                "subject_id": str(group["subject_id"].iloc[0]),
                "total_duration_seconds": total_duration_seconds,
                "event_count": int(len(group)),
                "transition_count": int(len(trans)),
                "fall_event_count": fall_event_count,
                "dominant_activity_label": dominant_activity,
                "dominant_placement_label": dominant_placement,
                "highest_fall_probability": highest_fall_probability,
                "summary_text": summary_text,
            }
        )

    session_summaries = pd.DataFrame(summaries).sort_values(
        ["session_id"], kind="stable"
    ).reset_index(drop=True)

    total_fall_events = (
        int(session_summaries["fall_event_count"].sum())
        if "fall_event_count" in session_summaries.columns
        else 0
    )

    narrative_summary = {
        "session_count": int(len(session_summaries)),
        "total_event_count": int(len(events_df)),
        "total_transition_count": int(len(transition_df)),
        "total_fall_event_count": total_fall_events,
        "sessions": session_summaries.to_dict(orient="records"),
    }

    return session_summaries, narrative_summary


def build_runtime_timeline_events(
        *,
        har_windows: pd.DataFrame,
        fall_windows: pd.DataFrame,
        placement_windows: pd.DataFrame,
        grouped_fall_events: pd.DataFrame | None = None,
        config: RuntimeTimelineConfig | None = None,
) -> RuntimeTimelineResult:
    cfg = config or RuntimeTimelineConfig()

    har_df = _standardise_har_df(har_windows)
    fall_df = _standardise_fall_df(fall_windows)
    placement_df = _standardise_placement_df(placement_windows)
    grouped_df = _standardise_grouped_fall_events(
        grouped_fall_events if grouped_fall_events is not None else pd.DataFrame()
    )

    point_timeline = _build_point_timeline(
        har_df,
        fall_df,
        placement_df,
        config=cfg,
    )

    raw_timeline_events = _build_timeline_events(
        point_timeline,
        grouped_df,
        config=cfg,
    )
    timeline_events = _compress_timeline_events(
        raw_timeline_events,
        config=cfg,
    )

    transition_events = detect_timeline_transitions(timeline_events)
    session_summaries, narrative_summary = build_session_narrative_summary(
        timeline_events,
        transition_events,
    )

    return RuntimeTimelineResult(
        point_timeline=point_timeline,
        raw_timeline_events=raw_timeline_events,
        timeline_events=timeline_events,
        transition_events=transition_events,
        session_summaries=session_summaries,
        narrative_summary=narrative_summary,
    )
