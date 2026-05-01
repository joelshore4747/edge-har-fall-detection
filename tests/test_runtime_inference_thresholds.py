"""Unit tests for per-placement fall-threshold plumbing.

Verifies that:

- When ``event_thresholds_by_placement`` is ``None``, behaviour matches the
  legacy single-threshold path exactly (monotonic change invariant).
- When a mapping is supplied and the dataframe carries a ``placement``
  column, per-row thresholds are applied, with a fallback to the scalar for
  any placement not listed.
- ``_mark_positive_windows`` accepts either a scalar or a Series.
"""

from __future__ import annotations

import pandas as pd

from services.runtime_inference import (
    _group_runtime_fall_events,
    _mark_positive_windows,
    _resolve_threshold_series,
)


def _build_fall_frame(rows: list[tuple[str, float]]) -> pd.DataFrame:
    """Build a tiny fall_df with one window per row at monotonically increasing time."""
    out = []
    for i, (placement, prob) in enumerate(rows):
        out.append(
            {
                "session_id": f"sess_{i}",
                "dataset_name": "PHONE_TEST",
                "subject_id": "tester",
                "placement": placement,
                "start_ts": float(i),
                "end_ts": float(i) + 1.0,
                "midpoint_ts": float(i) + 0.5,
                "predicted_probability": float(prob),
                "predicted_is_fall": prob >= 0.5,
            }
        )
    return pd.DataFrame(out)


def test_scalar_threshold_matches_legacy_path():
    df = _build_fall_frame([("pocket", 0.6), ("pocket", 0.4)])
    legacy = _mark_positive_windows(df, 0.5)
    resolved = _resolve_threshold_series(df, probability_threshold=0.5, thresholds_by_placement=None)
    assert resolved == 0.5  # scalar pass-through
    new_path = _mark_positive_windows(df, resolved)
    assert legacy.tolist() == new_path.tolist() == [True, False]


def test_per_placement_thresholds_applied_per_row():
    df = _build_fall_frame(
        [
            ("pocket", 0.55),  # pocket threshold 0.5 → positive
            ("desk", 0.55),    # desk threshold 0.80 → negative
            ("hand", 0.70),    # hand threshold 0.65 → positive
            ("pocket", 0.45),  # pocket threshold 0.5 → negative
        ]
    )
    thresholds = {"pocket": 0.5, "hand": 0.65, "desk": 0.80}
    resolved = _resolve_threshold_series(df, probability_threshold=0.5, thresholds_by_placement=thresholds)
    assert isinstance(resolved, pd.Series)
    assert resolved.tolist() == [0.5, 0.80, 0.65, 0.5]
    positives = _mark_positive_windows(df, resolved)
    assert positives.tolist() == [True, False, True, False]


def test_missing_placement_column_falls_back_to_scalar():
    df = _build_fall_frame([("pocket", 0.6), ("hand", 0.4)]).drop(columns=["placement"])
    resolved = _resolve_threshold_series(
        df,
        probability_threshold=0.5,
        thresholds_by_placement={"pocket": 0.9},
    )
    assert resolved == 0.5


def test_unknown_placement_falls_back_to_scalar_threshold():
    df = _build_fall_frame([("bag", 0.55)])  # 'bag' not in mapping
    resolved = _resolve_threshold_series(
        df,
        probability_threshold=0.5,
        thresholds_by_placement={"pocket": 0.9},
    )
    assert isinstance(resolved, pd.Series)
    assert resolved.tolist() == [0.5]
    positives = _mark_positive_windows(df, resolved)
    assert positives.tolist() == [True]


def test_grouped_events_honour_per_placement_thresholds():
    """Two sessions, same prob — only the pocket one meets its lower threshold."""
    df = pd.DataFrame(
        [
            # pocket session: 3 windows at 0.55, threshold 0.5 → all positive → event.
            {
                "session_id": "pocket_session",
                "dataset_name": "d",
                "subject_id": "s",
                "placement": "pocket",
                "start_ts": 0.0, "end_ts": 1.0, "midpoint_ts": 0.5,
                "predicted_probability": 0.55, "predicted_is_fall": True,
            },
            {
                "session_id": "pocket_session",
                "dataset_name": "d",
                "subject_id": "s",
                "placement": "pocket",
                "start_ts": 1.0, "end_ts": 2.0, "midpoint_ts": 1.5,
                "predicted_probability": 0.55, "predicted_is_fall": True,
            },
            # desk session: 3 windows at 0.55, threshold 0.80 → no positives → no event.
            {
                "session_id": "desk_session",
                "dataset_name": "d",
                "subject_id": "s",
                "placement": "desk",
                "start_ts": 10.0, "end_ts": 11.0, "midpoint_ts": 10.5,
                "predicted_probability": 0.55, "predicted_is_fall": True,
            },
            {
                "session_id": "desk_session",
                "dataset_name": "d",
                "subject_id": "s",
                "placement": "desk",
                "start_ts": 11.0, "end_ts": 12.0, "midpoint_ts": 11.5,
                "predicted_probability": 0.55, "predicted_is_fall": True,
            },
        ]
    )

    events = _group_runtime_fall_events(
        df,
        probability_threshold=0.5,
        merge_gap_seconds=0.5,
        min_windows=2,
        max_event_duration_seconds=10.0,
        thresholds_by_placement={"pocket": 0.5, "desk": 0.80},
    )
    # Only the pocket session should have produced a grouped event.
    assert list(events["session_id"].unique()) == ["pocket_session"]