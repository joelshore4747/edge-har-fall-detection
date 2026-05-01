#!/usr/bin/env python3
"""Create a manual annotation template for a runtime phone session.

Purpose:
- turn one replayed runtime session into an editable annotation CSV
- pre-seed the file with candidate event intervals from grouped fall events
- optionally add regular background intervals so you can label walking / stairs / sitting / fall

This is the bridge from:
- raw runtime model output
to:
- interpretable, session-level ground truth for phone evaluation
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_LABEL_OPTIONS = [
    "unknown",
    "walking",
    "stairs",
    "standing",
    "sitting",
    "sit_down_transition",
    "phone_handling",
    "fall",
    "other",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a runtime annotation template")
    parser.add_argument(
        "--timeline-csv",
        default="results/validation/phone1_timeline.csv",
        help="Combined runtime timeline CSV",
    )
    parser.add_argument(
        "--grouped-events-csv",
        default="results/validation/phone1_fall_grouped_events.csv",
        help="Grouped runtime fall-event CSV",
    )
    parser.add_argument(
        "--out-csv",
        default="results/validation/phone1_annotation_template.csv",
        help="Output annotation template CSV",
    )
    parser.add_argument(
        "--include-background-intervals",
        action="store_true",
        help="Add evenly spaced background intervals across the whole session",
    )
    parser.add_argument(
        "--background-interval-seconds",
        type=float,
        default=10.0,
        help="Interval size when --include-background-intervals is used",
    )
    parser.add_argument(
        "--event-padding-seconds",
        type=float,
        default=1.0,
        help="Padding added before and after grouped events in the annotation template",
    )
    parser.add_argument(
        "--top-k-events",
        type=int,
        default=15,
        help="How many grouped candidate events to include",
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


def _timeline_time_bounds(timeline_df: pd.DataFrame) -> tuple[float, float]:
    if "midpoint_ts" not in timeline_df.columns:
        raise ValueError("Timeline CSV must contain midpoint_ts")

    ts = pd.to_numeric(timeline_df["midpoint_ts"], errors="coerce").dropna()
    if ts.empty:
        raise ValueError("Timeline CSV has no usable midpoint_ts values")

    return float(ts.min()), float(ts.max())


def _make_event_rows(
    grouped_df: pd.DataFrame,
    *,
    event_padding_seconds: float,
    top_k_events: int,
) -> list[dict[str, object]]:
    if grouped_df.empty:
        return []

    working = grouped_df.copy()

    for col in ["event_start_ts", "event_end_ts", "peak_probability", "mean_probability"]:
        if col in working.columns:
            working[col] = pd.to_numeric(working[col], errors="coerce")

    sort_cols = [c for c in ["peak_probability", "mean_probability"] if c in working.columns]
    if sort_cols:
        working = working.sort_values(sort_cols, ascending=[False] * len(sort_cols), kind="stable")
    working = working.head(top_k_events).reset_index(drop=True)

    rows: list[dict[str, object]] = []
    for idx, row in working.iterrows():
        start_ts = float(row["event_start_ts"]) if pd.notna(row.get("event_start_ts")) else float("nan")
        end_ts = float(row["event_end_ts"]) if pd.notna(row.get("event_end_ts")) else float("nan")

        padded_start = max(0.0, start_ts - float(event_padding_seconds))
        padded_end = end_ts + float(event_padding_seconds)

        rows.append(
            {
                "row_type": "candidate_event",
                "priority_rank": int(idx + 1),
                "source_event_id": row.get("event_id", f"event_{idx:03d}"),
                "session_id": row.get("session_id", ""),
                "start_ts": padded_start,
                "end_ts": padded_end,
                "suggested_label": "unknown",
                "final_label": "",
                "confidence_hint": row.get("peak_probability", ""),
                "model_hint": "fall_candidate",
                "notes": "",
                "label_options": "|".join(DEFAULT_LABEL_OPTIONS),
            }
        )

    return rows


def _make_background_rows(
    *,
    start_ts: float,
    end_ts: float,
    interval_seconds: float,
    session_id: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    current = float(start_ts)
    idx = 0

    while current < end_ts:
        interval_end = min(float(end_ts), current + float(interval_seconds))
        rows.append(
            {
                "row_type": "background_interval",
                "priority_rank": int(1000 + idx),
                "source_event_id": "",
                "session_id": session_id,
                "start_ts": float(current),
                "end_ts": float(interval_end),
                "suggested_label": "unknown",
                "final_label": "",
                "confidence_hint": "",
                "model_hint": "background_review",
                "notes": "",
                "label_options": "|".join(DEFAULT_LABEL_OPTIONS),
            }
        )
        current = interval_end
        idx += 1

    return rows


def _select_session_id(timeline_df: pd.DataFrame, grouped_df: pd.DataFrame) -> str:
    if "session_id" in grouped_df.columns and not grouped_df["session_id"].dropna().empty:
        return str(grouped_df["session_id"].dropna().iloc[0])
    if "session_id" in timeline_df.columns and not timeline_df["session_id"].dropna().empty:
        return str(timeline_df["session_id"].dropna().iloc[0])
    return "unknown_session"


def main() -> int:
    args = parse_args()

    timeline_csv = _resolve(args.timeline_csv)
    grouped_events_csv = _resolve(args.grouped_events_csv)
    out_csv = _resolve(args.out_csv)

    timeline_df = _load_csv(timeline_csv)
    grouped_df = _load_csv(grouped_events_csv)

    session_start, session_end = _timeline_time_bounds(timeline_df)
    session_id = _select_session_id(timeline_df, grouped_df)

    rows: list[dict[str, object]] = []

    rows.extend(
        _make_event_rows(
            grouped_df,
            event_padding_seconds=args.event_padding_seconds,
            top_k_events=args.top_k_events,
        )
    )

    if args.include_background_intervals:
        rows.extend(
            _make_background_rows(
                start_ts=session_start,
                end_ts=session_end,
                interval_seconds=args.background_interval_seconds,
                session_id=session_id,
            )
        )

    if not rows:
        rows.append(
            {
                "row_type": "manual",
                "priority_rank": 1,
                "source_event_id": "",
                "session_id": session_id,
                "start_ts": session_start,
                "end_ts": session_end,
                "suggested_label": "unknown",
                "final_label": "",
                "confidence_hint": "",
                "model_hint": "",
                "notes": "",
                "label_options": "|".join(DEFAULT_LABEL_OPTIONS),
            }
        )

    out_df = pd.DataFrame(rows)
    out_df = out_df.sort_values(["priority_rank", "start_ts"], kind="stable").reset_index(drop=True)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)

    print(f"Saved annotation template CSV to: {out_csv}")
    print(f"Rows written: {len(out_df)}")
    print(f"Session bounds: {session_start:.3f}s -> {session_end:.3f}s")
    print(f"Session id: {session_id}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())