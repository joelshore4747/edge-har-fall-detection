from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.runtime_timeline import RuntimeTimelineConfig, build_runtime_timeline_events


def _build_har_df(rows: list[dict]) -> pd.DataFrame:
    base = {
        "dataset_name": "APP_RUNTIME_TEST",
        "subject_id": "subject_001",
        "session_id": "session_001",
    }
    return pd.DataFrame([{**base, **row} for row in rows])


def _build_placement_df(rows: list[dict]) -> pd.DataFrame:
    base = {
        "dataset_name": "APP_RUNTIME_TEST",
        "subject_id": "subject_001",
        "session_id": "session_001",
    }
    return pd.DataFrame([{**base, **row} for row in rows])


def _build_fall_df(rows: list[dict]) -> pd.DataFrame:
    base = {
        "dataset_name": "APP_RUNTIME_TEST",
        "subject_id": "subject_001",
        "session_id": "session_001",
    }
    return pd.DataFrame([{**base, **row} for row in rows])


def _default_config() -> RuntimeTimelineConfig:
    return RuntimeTimelineConfig(
        smoothing_window=7,
        bridge_short_runs=2,
        max_point_gap_seconds=2.5,
        merge_same_label_gap_seconds=1.5,
        display_merge_gap_seconds=2.0,
        display_min_activity_duration_seconds=4.0,
        display_min_stationary_duration_seconds=4.0,
        display_min_placement_duration_seconds=5.0,
        display_min_fall_duration_seconds=1.5,
        display_repositioning_min_duration_seconds=5.0,
        display_repositioning_min_points=4,
        display_min_points=3,
        fall_probability_threshold=0.82,
        elevated_fall_probability_threshold=0.55,
        min_fall_points=2,
        min_fall_event_duration_seconds=1.0,
    )


@pytest.mark.integration
def test_short_repositioning_burst_is_suppressed() -> None:
    times = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    har_df = _build_har_df(
        [
            {
                "midpoint_ts": t,
                "predicted_label": "walking",
                "predicted_confidence": 0.90,
            }
            for t in times
        ]
    )

    placement_labels = [
        "in_pocket",
        "in_pocket",
        "in_pocket",
        "repositioning",
        "repositioning",
        "in_pocket",
        "in_pocket",
        "in_pocket",
        "in_pocket",
        "in_pocket",
    ]
    placement_df = _build_placement_df(
        [
            {
                "midpoint_ts": t,
                "placement_state": label,
                "placement_state_smoothed": label,
                "placement_confidence": 0.85,
            }
            for t, label in zip(times, placement_labels, strict=True)
        ]
    )

    fall_df = _build_fall_df(
        [
            {
                "midpoint_ts": t,
                "predicted_label": "non_fall",
                "predicted_probability": 0.05,
                "predicted_is_fall": False,
            }
            for t in times
        ]
    )

    result = build_runtime_timeline_events(
        har_windows=har_df,
        placement_windows=placement_df,
        fall_windows=fall_df,
        grouped_fall_events=pd.DataFrame(),
        config=_default_config(),
    )

    assert len(result.timeline_events) >= 1
    assert "repositioning" not in set(result.timeline_events["placement_label"].astype(str))


@pytest.mark.integration
def test_nearby_fall_spikes_collapse_to_single_fall_event() -> None:
    times = list(range(12))

    har_df = _build_har_df(
        [
            {
                "midpoint_ts": t,
                "predicted_label": "walking" if t < 9 else "sitting",
                "predicted_confidence": 0.88,
            }
            for t in times
        ]
    )

    placement_df = _build_placement_df(
        [
            {
                "midpoint_ts": t,
                "placement_state": "in_pocket",
                "placement_state_smoothed": "in_pocket",
                "placement_confidence": 0.82,
            }
            for t in times
        ]
    )

    probs = [0.04, 0.05, 0.10, 0.18, 0.83, 0.91, 0.79, 0.88, 0.14, 0.09, 0.04, 0.03]
    fall_df = _build_fall_df(
        [
            {
                "midpoint_ts": t,
                "predicted_label": "fall" if p >= 0.55 else "non_fall",
                "predicted_probability": p,
                "predicted_is_fall": p >= 0.55,
            }
            for t, p in zip(times, probs, strict=True)
        ]
    )

    result = build_runtime_timeline_events(
        har_windows=har_df,
        placement_windows=placement_df,
        fall_windows=fall_df,
        grouped_fall_events=pd.DataFrame(),
        config=_default_config(),
    )

    fall_events = result.timeline_events[
        result.timeline_events["likely_fall"].fillna(False).astype(bool)
    ]
    assert len(fall_events) == 1


@pytest.mark.integration
def test_short_middle_activity_flip_merges_back_into_neighbors() -> None:
    times = list(range(11))
    labels = [
        "walking",
        "walking",
        "walking",
        "walking_downstairs",
        "walking_downstairs",
        "walking",
        "walking",
        "walking",
        "walking",
        "walking",
        "walking",
    ]

    har_df = _build_har_df(
        [
            {
                "midpoint_ts": t,
                "predicted_label": label,
                "predicted_confidence": 0.86,
            }
            for t, label in zip(times, labels, strict=True)
        ]
    )

    placement_df = _build_placement_df(
        [
            {
                "midpoint_ts": t,
                "placement_state": "in_pocket",
                "placement_state_smoothed": "in_pocket",
                "placement_confidence": 0.80,
            }
            for t in times
        ]
    )

    fall_df = _build_fall_df(
        [
            {
                "midpoint_ts": t,
                "predicted_label": "non_fall",
                "predicted_probability": 0.03,
                "predicted_is_fall": False,
            }
            for t in times
        ]
    )

    result = build_runtime_timeline_events(
        har_windows=har_df,
        placement_windows=placement_df,
        fall_windows=fall_df,
        grouped_fall_events=pd.DataFrame(),
        config=_default_config(),
    )

    assert len(result.timeline_events) == 1
    only_event = result.timeline_events.iloc[0]
    assert str(only_event["activity_label"]) == "walking"


@pytest.mark.integration
def test_display_timeline_is_shorter_than_point_timeline_for_noisy_session() -> None:
    times = list(range(20))

    har_labels = [
        "walking",
        "walking",
        "walking",
        "walking_downstairs",
        "walking",
        "walking",
        "walking",
        "walking",
        "walking_upstairs",
        "walking",
        "walking",
        "walking",
        "sitting",
        "standing",
        "sitting",
        "sitting",
        "sitting",
        "sitting",
        "sitting",
        "sitting",
    ]
    har_df = _build_har_df(
        [
            {
                "midpoint_ts": t,
                "predicted_label": label,
                "predicted_confidence": 0.84,
            }
            for t, label in zip(times, har_labels, strict=True)
        ]
    )

    placement_labels = [
        "in_pocket",
        "in_pocket",
        "repositioning",
        "in_pocket",
        "in_pocket",
        "in_pocket",
        "in_hand",
        "in_pocket",
        "in_pocket",
        "in_pocket",
        "in_pocket",
        "in_pocket",
        "on_surface",
        "on_surface",
        "on_surface",
        "on_surface",
        "on_surface",
        "on_surface",
        "on_surface",
        "on_surface",
    ]
    placement_df = _build_placement_df(
        [
            {
                "midpoint_ts": t,
                "placement_state": label,
                "placement_state_smoothed": label,
                "placement_confidence": 0.82,
            }
            for t, label in zip(times, placement_labels, strict=True)
        ]
    )

    probs = [
        0.02,
        0.03,
        0.04,
        0.06,
        0.05,
        0.04,
        0.81,
        0.87,
        0.08,
        0.04,
        0.03,
        0.02,
        0.05,
        0.04,
        0.03,
        0.03,
        0.02,
        0.02,
        0.02,
        0.02,
    ]
    fall_df = _build_fall_df(
        [
            {
                "midpoint_ts": t,
                "predicted_label": "fall" if p >= 0.55 else "non_fall",
                "predicted_probability": p,
                "predicted_is_fall": p >= 0.55,
            }
            for t, p in zip(times, probs, strict=True)
        ]
    )

    result = build_runtime_timeline_events(
        har_windows=har_df,
        placement_windows=placement_df,
        fall_windows=fall_df,
        grouped_fall_events=pd.DataFrame(),
        config=_default_config(),
    )

    assert len(result.point_timeline) == len(times)
    assert len(result.timeline_events) < len(result.point_timeline)
    assert len(result.transition_events) <= max(0, len(result.timeline_events) - 1)

    durations = pd.to_numeric(result.timeline_events["duration_seconds"], errors="coerce")
    assert durations.notna().all()
    assert (durations >= 0).all()