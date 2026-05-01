#!/usr/bin/env python3
"""Group overlapping fall-positive runtime windows into candidate events.

This version is deliberately less aggressive than the first pass:
- smaller default merge gap
- requires at least 2 positive windows by default
- can split very long merged regions into smaller sub-events

Why:
- overlapping-window models often produce runs of positives around one motion episode
- but merging too aggressively can turn stairs / prolonged motion into one giant "fall event"
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Group fall-positive runtime windows into events")
    parser.add_argument(
        "--fall-csv",
        default="results/validation/phone1_fall.csv",
        help="Path to fall replay CSV",
    )
    parser.add_argument(
        "--out-csv",
        default="results/validation/phone1_fall_grouped_events.csv",
        help="Path to grouped event CSV",
    )
    parser.add_argument(
        "--probability-threshold",
        type=float,
        default=None,
        help=(
            "Override probability threshold for defining positive windows. "
            "If omitted, uses predicted_is_fall when present, else predicted_label=='fall', "
            "else falls back to >= 0.5 probability."
        ),
    )
    parser.add_argument(
        "--merge-gap-seconds",
        type=float,
        default=0.25,
        help="Merge positive windows into one event only if the gap between them is <= this value",
    )
    parser.add_argument(
        "--min-windows",
        type=int,
        default=2,
        help="Minimum number of positive windows required to keep an event",
    )
    parser.add_argument(
        "--max-event-duration-seconds",
        type=float,
        default=4.0,
        help=(
            "If a merged region lasts longer than this, split it into smaller sub-events "
            "using the same merge-gap rule."
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="How many top events to print",
    )
    return parser.parse_args()


def _resolve(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    return path


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    return pd.read_csv(path)


def _coerce_bool_like(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    lowered = series.astype(str).str.lower()
    if lowered.isin({"true", "false"}).all():
        return lowered.map({"true": True, "false": False}).astype(bool)
    return series.astype(bool)


def _mark_positive_windows(df: pd.DataFrame, probability_threshold: float | None) -> pd.Series:
    if probability_threshold is not None and "predicted_probability" in df.columns:
        prob = pd.to_numeric(df["predicted_probability"], errors="coerce").fillna(0.0)
        return prob >= float(probability_threshold)

    if "predicted_is_fall" in df.columns:
        return _coerce_bool_like(df["predicted_is_fall"]).fillna(False)

    if "predicted_label" in df.columns:
        return df["predicted_label"].astype(str).str.lower().eq("fall")

    if "predicted_probability" in df.columns:
        prob = pd.to_numeric(df["predicted_probability"], errors="coerce").fillna(0.0)
        return prob >= 0.5

    raise ValueError(
        "Could not determine positive windows. Need one of: predicted_is_fall, predicted_label, predicted_probability."
    )


def _event_time_columns(df: pd.DataFrame) -> tuple[str, str, str]:
    start_col = "start_ts" if "start_ts" in df.columns else "midpoint_ts"
    end_col = "end_ts" if "end_ts" in df.columns else "midpoint_ts"
    midpoint_col = "midpoint_ts" if "midpoint_ts" in df.columns else start_col
    return start_col, end_col, midpoint_col


def _build_event_record(
    event_df: pd.DataFrame,
    *,
    session_id: str,
    start_col: str,
    end_col: str,
    midpoint_col: str,
    event_idx: int,
) -> dict[str, object]:
    prob = (
        pd.to_numeric(event_df["predicted_probability"], errors="coerce")
        if "predicted_probability" in event_df.columns
        else pd.Series([float("nan")] * len(event_df))
    )

    event_start = float(pd.to_numeric(event_df[start_col], errors="coerce").min())
    event_end = float(pd.to_numeric(event_df[end_col], errors="coerce").max())

    return {
        "event_id": f"{session_id}_event_{event_idx:03d}",
        "dataset_name": str(event_df["dataset_name"].iloc[0]),
        "subject_id": str(event_df["subject_id"].iloc[0]),
        "session_id": str(session_id),
        "event_start_ts": event_start,
        "event_end_ts": event_end,
        "event_duration_seconds": float(event_end - event_start),
        "n_positive_windows": int(len(event_df)),
        "peak_probability": float(prob.max()) if prob.notna().any() else float("nan"),
        "mean_probability": float(prob.mean()) if prob.notna().any() else float("nan"),
        "median_probability": float(prob.median()) if prob.notna().any() else float("nan"),
        "first_midpoint_ts": float(pd.to_numeric(event_df[midpoint_col], errors="coerce").min()),
        "last_midpoint_ts": float(pd.to_numeric(event_df[midpoint_col], errors="coerce").max()),
    }


def _split_long_event(
    event_df: pd.DataFrame,
    *,
    session_id: str,
    start_col: str,
    end_col: str,
    midpoint_col: str,
    merge_gap_seconds: float,
    max_event_duration_seconds: float,
    min_windows: int,
    next_event_idx: int,
) -> tuple[list[dict[str, object]], int]:
    """Split an overlong merged event into smaller chunks.

    Strategy:
    - walk through already-positive windows in time order
    - start a new sub-event if either:
      1) the local gap is larger than merge_gap_seconds, or
      2) adding the next window would push event duration beyond max_event_duration_seconds
    """
    rows = event_df.sort_values(start_col, kind="stable").reset_index(drop=True)
    records: list[dict[str, object]] = []

    current_rows: list[pd.Series] = []
    current_start = None
    current_end = None

    def flush() -> None:
        nonlocal current_rows, current_start, current_end, next_event_idx, records
        if not current_rows:
            return
        chunk_df = pd.DataFrame(current_rows)
        if len(chunk_df) >= int(min_windows):
            records.append(
                _build_event_record(
                    chunk_df,
                    session_id=session_id,
                    start_col=start_col,
                    end_col=end_col,
                    midpoint_col=midpoint_col,
                    event_idx=next_event_idx,
                )
            )
            next_event_idx += 1
        current_rows = []
        current_start = None
        current_end = None

    for _, row in rows.iterrows():
        row_start = float(row[start_col])
        row_end = float(row[end_col])

        if not current_rows:
            current_rows.append(row)
            current_start = row_start
            current_end = row_end
            continue

        gap = row_start - float(current_end)
        proposed_end = max(float(current_end), row_end)
        proposed_duration = proposed_end - float(current_start)

        if gap > float(merge_gap_seconds) or proposed_duration > float(max_event_duration_seconds):
            flush()
            current_rows.append(row)
            current_start = row_start
            current_end = row_end
        else:
            current_rows.append(row)
            current_end = proposed_end

    flush()
    return records, next_event_idx


def group_fall_events(
    fall_df: pd.DataFrame,
    *,
    probability_threshold: float | None,
    merge_gap_seconds: float,
    min_windows: int,
    max_event_duration_seconds: float,
) -> pd.DataFrame:
    df = fall_df.copy()

    start_col, end_col, midpoint_col = _event_time_columns(df)
    for col in [start_col, end_col, midpoint_col]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "session_id" not in df.columns:
        df["session_id"] = "unknown_session"
    if "dataset_name" not in df.columns:
        df["dataset_name"] = "unknown_dataset"
    if "subject_id" not in df.columns:
        df["subject_id"] = "unknown_subject"

    df["is_positive_window"] = _mark_positive_windows(df, probability_threshold)
    df = df[df["is_positive_window"]].copy()

    if df.empty:
        return pd.DataFrame(
            columns=[
                "event_id",
                "dataset_name",
                "subject_id",
                "session_id",
                "event_start_ts",
                "event_end_ts",
                "event_duration_seconds",
                "n_positive_windows",
                "peak_probability",
                "mean_probability",
                "median_probability",
                "first_midpoint_ts",
                "last_midpoint_ts",
            ]
        )

    df = df.sort_values(["session_id", start_col, midpoint_col], kind="stable").reset_index(drop=True)

    preliminary_groups: list[pd.DataFrame] = []

    for _, group in df.groupby("session_id", dropna=False, sort=False):
        group = group.reset_index(drop=True)

        current_rows: list[pd.Series] = []
        current_end = None

        def flush_current() -> None:
            nonlocal current_rows, current_end
            if current_rows:
                preliminary_groups.append(pd.DataFrame(current_rows))
            current_rows = []
            current_end = None

        for _, row in group.iterrows():
            row_start = float(row[start_col])
            row_end = float(row[end_col])

            if not current_rows:
                current_rows.append(row)
                current_end = row_end
                continue

            gap = row_start - float(current_end)
            if gap <= float(merge_gap_seconds):
                current_rows.append(row)
                current_end = max(float(current_end), row_end)
            else:
                flush_current()
                current_rows.append(row)
                current_end = row_end

        flush_current()

    events: list[dict[str, object]] = []
    event_idx = 0

    for event_df in preliminary_groups:
        session_id = str(event_df["session_id"].iloc[0])

        event_start = float(pd.to_numeric(event_df[start_col], errors="coerce").min())
        event_end = float(pd.to_numeric(event_df[end_col], errors="coerce").max())
        event_duration = event_end - event_start

        if event_duration > float(max_event_duration_seconds):
            split_records, event_idx = _split_long_event(
                event_df,
                session_id=session_id,
                start_col=start_col,
                end_col=end_col,
                midpoint_col=midpoint_col,
                merge_gap_seconds=merge_gap_seconds,
                max_event_duration_seconds=max_event_duration_seconds,
                min_windows=min_windows,
                next_event_idx=event_idx,
            )
            events.extend(split_records)
        else:
            if len(event_df) >= int(min_windows):
                events.append(
                    _build_event_record(
                        event_df,
                        session_id=session_id,
                        start_col=start_col,
                        end_col=end_col,
                        midpoint_col=midpoint_col,
                        event_idx=event_idx,
                    )
                )
                event_idx += 1

    out = pd.DataFrame(events)
    if out.empty:
        return out

    out = out.sort_values(
        ["peak_probability", "mean_probability", "event_duration_seconds", "event_start_ts"],
        ascending=[False, False, True, True],
        kind="stable",
    ).reset_index(drop=True)
    return out


def main() -> int:
    args = parse_args()

    fall_csv = _resolve(args.fall_csv)
    out_csv = _resolve(args.out_csv)

    fall_df = _load_csv(fall_csv)
    grouped = group_fall_events(
        fall_df,
        probability_threshold=args.probability_threshold,
        merge_gap_seconds=args.merge_gap_seconds,
        min_windows=args.min_windows,
        max_event_duration_seconds=args.max_event_duration_seconds,
    )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    grouped.to_csv(out_csv, index=False)

    print(f"Saved grouped event CSV to: {out_csv}")
    print(f"Grouped events: {len(grouped)}")

    if grouped.empty:
        print("No grouped events found.")
        return 0

    cols = [
        c for c in [
            "event_id",
            "session_id",
            "event_start_ts",
            "event_end_ts",
            "event_duration_seconds",
            "n_positive_windows",
            "peak_probability",
            "mean_probability",
        ]
        if c in grouped.columns
    ]

    print()
    print(f"Top {min(args.top_k, len(grouped))} grouped events:")
    print(grouped.loc[:, cols].head(args.top_k).to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())