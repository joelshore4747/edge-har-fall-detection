"""Event-level fall metrics for threshold baseline analysis.

Window-level false alarms can overstate alarm burden when adjacent windows represent
one continuous trigger. These helpers cluster consecutive fall windows into events
within each logical sequence and compute event-level precision/recall counts.
"""

from __future__ import annotations

import pandas as pd


DEFAULT_GROUP_COLS = ["dataset_name", "subject_id", "session_id", "source_file"]

def _group_columns(df: pd.DataFrame, group_cols: list[str] | None = None) -> list[str]:
    cols = group_cols or DEFAULT_GROUP_COLS
    return [c for c in cols if c in df.columns]



def _sort_for_events(df: pd.DataFrame) -> pd.DataFrame:
    working = df.copy()
    sort_cols = [c for c in ["start_ts", "timestamp", "window_id", "row_index"] if c in working.columns]
    if sort_cols:
        working = working.sort_values(sort_cols, kind="stable")
    return working.reset_index(drop=True)



def cluster_positive_events(
    df: pd.DataFrame,
    *,
    label_col: str,
    positive_label: str = "fall",
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Cluster consecutive positive windows into event intervals."""
    if df.empty or label_col not in df.columns:
        return pd.DataFrame(
            columns=[
                "event_id",
                "dataset_name",
                "subject_id",
                "session_id",
                "source_file",
                "group_key",
                "start_row_pos",
                "end_row_pos",
                "n_windows",
                "start_ts",
                "end_ts",
            ]
        )

    groups = _group_columns(df, group_cols)
    grouped = df.groupby(groups, dropna=False, sort=False) if groups else [((), df)]

    events: list[dict] = []
    event_id = 0

    for group_key, group_df in grouped:
        ordered = _sort_for_events(group_df)
        mask = ordered[label_col].astype(str).eq(positive_label).tolist()
        if not mask:
            continue

        start = None
        for idx, is_pos in enumerate(mask):
            if is_pos and start is None:
                start = idx
            if (not is_pos) and start is not None:
                end = idx - 1
                segment = ordered.iloc[start : end + 1]
                events.append(
                    {
                        "event_id": int(event_id),
                        "dataset_name": segment.get("dataset_name", pd.Series([None])).iloc[0],
                        "subject_id": segment.get("subject_id", pd.Series([None])).iloc[0],
                        "session_id": segment.get("session_id", pd.Series([None])).iloc[0],
                        "source_file": segment.get("source_file", pd.Series([None])).iloc[0],
                        "group_key": tuple(group_key) if isinstance(group_key, tuple) else (group_key,),
                        "start_row_pos": int(start),
                        "end_row_pos": int(end),
                        "n_windows": int(end - start + 1),
                        "start_ts": float(segment["start_ts"].iloc[0]) if "start_ts" in segment.columns and pd.notna(segment["start_ts"].iloc[0]) else None,
                        "end_ts": float(segment["end_ts"].iloc[-1]) if "end_ts" in segment.columns and pd.notna(segment["end_ts"].iloc[-1]) else None,
                    }
                )
                event_id += 1
                start = None

        if start is not None:
            end = len(mask) - 1
            segment = ordered.iloc[start : end + 1]
            events.append(
                {
                    "event_id": int(event_id),
                    "dataset_name": segment.get("dataset_name", pd.Series([None])).iloc[0],
                    "subject_id": segment.get("subject_id", pd.Series([None])).iloc[0],
                    "session_id": segment.get("session_id", pd.Series([None])).iloc[0],
                    "source_file": segment.get("source_file", pd.Series([None])).iloc[0],
                    "group_key": tuple(group_key) if isinstance(group_key, tuple) else (group_key,),
                    "start_row_pos": int(start),
                    "end_row_pos": int(end),
                    "n_windows": int(end - start + 1),
                    "start_ts": float(segment["start_ts"].iloc[0]) if "start_ts" in segment.columns and pd.notna(segment["start_ts"].iloc[0]) else None,
                    "end_ts": float(segment["end_ts"].iloc[-1]) if "end_ts" in segment.columns and pd.notna(segment["end_ts"].iloc[-1]) else None,
                }
            )
            event_id += 1

    return pd.DataFrame(events)



def _interval_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return not (a_end < b_start or b_end < a_start)



def compute_event_level_metrics(
    predictions_df: pd.DataFrame,
    *,
    true_label_col: str = "true_label",
    predicted_label_col: str = "predicted_label",
    positive_label: str = "fall",
    group_cols: list[str] | None = None,
) -> dict:
    """Compute event-level metrics by clustering consecutive fall windows."""
    if predictions_df.empty:
        return {
            "predicted_fall_events_count": 0,
            "true_fall_events_count": 0,
            "true_positive_events_count": 0,
            "false_positive_events_count": 0,
            "false_negative_events_count": 0,
            "event_precision": 0.0,
            "event_recall": 0.0,
            "predicted_events": pd.DataFrame(),
            "true_events": pd.DataFrame(),
            "matched_pred_event_ids": [],
            "matched_true_event_ids": [],
        }

    pred_events = cluster_positive_events(
        predictions_df,
        label_col=predicted_label_col,
        positive_label=positive_label,
        group_cols=group_cols,
    )
    true_events = cluster_positive_events(
        predictions_df,
        label_col=true_label_col,
        positive_label=positive_label,
        group_cols=group_cols,
    )

    matched_pred_ids: set[int] = set()
    matched_true_ids: set[int] = set()

    if not pred_events.empty and not true_events.empty:
        for _, pred_event in pred_events.iterrows():
            pred_group = pred_event.get("group_key")
            cand = true_events[true_events["group_key"] == pred_group]
            for _, true_event in cand.iterrows():
                if _interval_overlap(
                    int(pred_event["start_row_pos"]),
                    int(pred_event["end_row_pos"]),
                    int(true_event["start_row_pos"]),
                    int(true_event["end_row_pos"]),
                ):
                    matched_pred_ids.add(int(pred_event["event_id"]))
                    matched_true_ids.add(int(true_event["event_id"]))
                    break

    tp_events = int(len(matched_pred_ids))
    fp_events = int(len(pred_events) - tp_events)
    fn_events = int(len(true_events) - len(matched_true_ids))

    precision = float(tp_events / (tp_events + fp_events)) if (tp_events + fp_events) > 0 else 0.0
    recall = float(tp_events / (tp_events + fn_events)) if (tp_events + fn_events) > 0 else 0.0

    return {
        "predicted_fall_events_count": int(len(pred_events)),
        "true_fall_events_count": int(len(true_events)),
        "true_positive_events_count": tp_events,
        "false_positive_events_count": fp_events,
        "false_negative_events_count": fn_events,
        "event_precision": precision,
        "event_recall": recall,
        "predicted_events": pred_events,
        "true_events": true_events,
        "matched_pred_event_ids": sorted(int(v) for v in matched_pred_ids),
        "matched_true_event_ids": sorted(int(v) for v in matched_true_ids),
    }
