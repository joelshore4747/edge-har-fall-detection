#!/usr/bin/env python3
"""Compare manual runtime annotations against grouped fall events and HAR timeline output.

Purpose:
- check whether annotated fall intervals were detected by grouped fall events
- identify what false-positive fall events overlap with (stairs, sit_down_transition, etc.)
- inspect HAR predictions inside annotated background intervals

Inputs:
- annotation template CSV (manually filled)
- grouped fall events CSV
- combined runtime timeline CSV

Outputs:
- fall event comparison CSV
- HAR interval comparison CSV
- JSON summary report
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare runtime annotations against replay outputs")
    parser.add_argument(
        "--annotation-csv",
        default="results/validation/phone1_annotation_template.csv",
        help="Filled annotation template CSV",
    )
    parser.add_argument(
        "--grouped-events-csv",
        default="results/validation/phone1_fall_grouped_events.csv",
        help="Grouped fall events CSV",
    )
    parser.add_argument(
        "--timeline-csv",
        default="results/validation/phone1_timeline.csv",
        help="Combined runtime timeline CSV",
    )
    parser.add_argument(
        "--fall-events-out",
        default="results/validation/phone1_fall_event_comparison.csv",
        help="CSV path for grouped fall event comparison",
    )
    parser.add_argument(
        "--har-intervals-out",
        default="results/validation/phone1_har_interval_comparison.csv",
        help="CSV path for HAR interval comparison",
    )
    parser.add_argument(
        "--report-out",
        default="results/validation/phone1_annotation_comparison_report.json",
        help="JSON summary report path",
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


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _clean_annotations(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for col in ["start_ts", "end_ts"]:
        if col not in out.columns:
            raise ValueError(f"Annotation CSV missing required column: {col}")
        out[col] = pd.to_numeric(out[col], errors="coerce")

    if "final_label" not in out.columns:
        raise ValueError("Annotation CSV missing required column: final_label")

    out["final_label"] = out["final_label"].astype(str).str.strip()
    out = out[out["final_label"].ne("")].copy()
    out = out[out["final_label"].str.lower().ne("unknown")].copy()

    if "row_type" not in out.columns:
        out["row_type"] = "manual"

    if "priority_rank" not in out.columns:
        out["priority_rank"] = 999999

    return out.reset_index(drop=True)


def _time_overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    start = max(float(a_start), float(b_start))
    end = min(float(a_end), float(b_end))
    return max(0.0, end - start)


def _best_annotation_overlap(
    event_start: float,
    event_end: float,
    annotations_df: pd.DataFrame,
) -> dict[str, Any] | None:
    overlaps: list[dict[str, Any]] = []

    for _, row in annotations_df.iterrows():
        ann_start = float(row["start_ts"])
        ann_end = float(row["end_ts"])
        overlap = _time_overlap(event_start, event_end, ann_start, ann_end)
        if overlap <= 0:
            continue

        row_type = str(row.get("row_type", "manual"))
        # Prefer candidate_event over background_interval when overlap is similar.
        row_type_priority = 0 if row_type == "candidate_event" else 1

        overlaps.append(
            {
                "row_type": row_type,
                "row_type_priority": row_type_priority,
                "final_label": str(row["final_label"]),
                "notes": str(row.get("notes", "")),
                "start_ts": ann_start,
                "end_ts": ann_end,
                "overlap_seconds": float(overlap),
                "priority_rank": int(row.get("priority_rank", 999999)),
            }
        )

    if not overlaps:
        return None

    overlaps_df = pd.DataFrame(overlaps).sort_values(
        ["overlap_seconds", "row_type_priority", "priority_rank"],
        ascending=[False, True, True],
        kind="stable",
    )
    return overlaps_df.iloc[0].to_dict()


def _compare_grouped_fall_events(
    grouped_df: pd.DataFrame,
    annotations_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    required = {"event_start_ts", "event_end_ts"}
    missing = required - set(grouped_df.columns)
    if missing:
        raise ValueError(f"Grouped events CSV missing required columns: {sorted(missing)}")

    working = grouped_df.copy()
    working["event_start_ts"] = pd.to_numeric(working["event_start_ts"], errors="coerce")
    working["event_end_ts"] = pd.to_numeric(working["event_end_ts"], errors="coerce")

    rows: list[dict[str, Any]] = []
    for _, row in working.iterrows():
        event_start = float(row["event_start_ts"])
        event_end = float(row["event_end_ts"])
        best = _best_annotation_overlap(event_start, event_end, annotations_df)

        matched_label = best["final_label"] if best is not None else "no_overlap"
        matched_notes = best["notes"] if best is not None else ""
        matched_row_type = best["row_type"] if best is not None else ""
        overlap_seconds = float(best["overlap_seconds"]) if best is not None else 0.0

        if matched_label == "fall":
            event_role = "true_positive_candidate"
        elif matched_label == "no_overlap":
            event_role = "unmatched_candidate"
        else:
            event_role = "false_positive_candidate"

        rows.append(
            {
                "event_id": row.get("event_id", ""),
                "session_id": row.get("session_id", ""),
                "event_start_ts": event_start,
                "event_end_ts": event_end,
                "event_duration_seconds": float(row.get("event_duration_seconds", event_end - event_start)),
                "n_positive_windows": int(row.get("n_positive_windows", 0)),
                "peak_probability": float(row.get("peak_probability", float("nan"))),
                "mean_probability": float(row.get("mean_probability", float("nan"))),
                "matched_annotation_label": matched_label,
                "matched_annotation_row_type": matched_row_type,
                "matched_annotation_notes": matched_notes,
                "overlap_seconds": overlap_seconds,
                "event_role": event_role,
            }
        )

    out_df = pd.DataFrame(rows)
    summary = {
        "n_grouped_events": int(len(out_df)),
        "event_role_counts": out_df["event_role"].astype(str).value_counts(dropna=False).to_dict()
        if not out_df.empty
        else {},
        "matched_annotation_label_counts": out_df["matched_annotation_label"].astype(str).value_counts(dropna=False).to_dict()
        if not out_df.empty
        else {},
    }
    return out_df, summary


def _evaluate_annotated_fall_intervals(
    annotations_df: pd.DataFrame,
    grouped_df: pd.DataFrame,
) -> pd.DataFrame:
    fall_annotations = annotations_df[annotations_df["final_label"].astype(str).eq("fall")].copy()
    if fall_annotations.empty:
        return pd.DataFrame(
            columns=[
                "annotation_start_ts",
                "annotation_end_ts",
                "detected",
                "best_event_id",
                "best_event_peak_probability",
                "best_overlap_seconds",
                "notes",
            ]
        )

    grouped = grouped_df.copy()
    grouped["event_start_ts"] = pd.to_numeric(grouped["event_start_ts"], errors="coerce")
    grouped["event_end_ts"] = pd.to_numeric(grouped["event_end_ts"], errors="coerce")

    rows: list[dict[str, Any]] = []
    for _, ann in fall_annotations.iterrows():
        ann_start = float(ann["start_ts"])
        ann_end = float(ann["end_ts"])

        best_event = None
        best_overlap = 0.0

        for _, event in grouped.iterrows():
            overlap = _time_overlap(
                ann_start,
                ann_end,
                float(event["event_start_ts"]),
                float(event["event_end_ts"]),
            )
            if overlap > best_overlap:
                best_overlap = float(overlap)
                best_event = event

        rows.append(
            {
                "annotation_start_ts": ann_start,
                "annotation_end_ts": ann_end,
                "detected": bool(best_overlap > 0),
                "best_event_id": best_event.get("event_id", "") if best_event is not None else "",
                "best_event_peak_probability": float(best_event.get("peak_probability", float("nan")))
                if best_event is not None
                else float("nan"),
                "best_overlap_seconds": float(best_overlap),
                "notes": str(ann.get("notes", "")),
            }
        )

    return pd.DataFrame(rows)


def _select_har_annotation_rows(annotations_df: pd.DataFrame) -> pd.DataFrame:
    background = annotations_df[annotations_df["row_type"].astype(str).eq("background_interval")].copy()
    if not background.empty:
        return background.reset_index(drop=True)
    return annotations_df.reset_index(drop=True)


def _compare_har_against_annotations(
    timeline_df: pd.DataFrame,
    annotations_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if "midpoint_ts" not in timeline_df.columns:
        raise ValueError("Timeline CSV missing midpoint_ts")
    if "har_predicted_label" not in timeline_df.columns:
        raise ValueError("Timeline CSV missing har_predicted_label")

    tl = timeline_df.copy()
    tl["midpoint_ts"] = pd.to_numeric(tl["midpoint_ts"], errors="coerce")
    tl = tl.dropna(subset=["midpoint_ts"]).reset_index(drop=True)

    ann_rows = _select_har_annotation_rows(annotations_df)
    rows: list[dict[str, Any]] = []

    for _, ann in ann_rows.iterrows():
        ann_start = float(ann["start_ts"])
        ann_end = float(ann["end_ts"])
        window_df = tl[(tl["midpoint_ts"] >= ann_start) & (tl["midpoint_ts"] <= ann_end)].copy()

        if window_df.empty:
            rows.append(
                {
                    "row_type": str(ann.get("row_type", "")),
                    "session_id": str(ann.get("session_id", "")),
                    "annotation_label": str(ann["final_label"]),
                    "annotation_notes": str(ann.get("notes", "")),
                    "annotation_start_ts": ann_start,
                    "annotation_end_ts": ann_end,
                    "n_timeline_rows": 0,
                    "dominant_har_label": "",
                    "dominant_har_fraction": float("nan"),
                    "har_label_counts": "",
                    "mean_har_confidence": float("nan"),
                }
            )
            continue

        counts = window_df["har_predicted_label"].astype(str).value_counts(dropna=False)
        dominant_label = str(counts.index[0])
        dominant_fraction = float(counts.iloc[0] / len(window_df))

        conf = (
            pd.to_numeric(window_df["har_predicted_confidence"], errors="coerce")
            if "har_predicted_confidence" in window_df.columns
            else pd.Series([float("nan")] * len(window_df))
        )

        rows.append(
            {
                "row_type": str(ann.get("row_type", "")),
                "session_id": str(ann.get("session_id", "")),
                "annotation_label": str(ann["final_label"]),
                "annotation_notes": str(ann.get("notes", "")),
                "annotation_start_ts": ann_start,
                "annotation_end_ts": ann_end,
                "n_timeline_rows": int(len(window_df)),
                "dominant_har_label": dominant_label,
                "dominant_har_fraction": dominant_fraction,
                "har_label_counts": "; ".join([f"{k}:{int(v)}" for k, v in counts.items()]),
                "mean_har_confidence": float(conf.mean()) if conf.notna().any() else float("nan"),
            }
        )

    out_df = pd.DataFrame(rows)

    summary = {}
    if not out_df.empty:
        pair_counts = (
            out_df.groupby(["annotation_label", "dominant_har_label"], dropna=False)
            .size()
            .reset_index(name="count")
            .sort_values(["annotation_label", "count"], ascending=[True, False], kind="stable")
        )
        summary["annotation_to_har_dominant_counts"] = pair_counts.to_dict(orient="records")

    return out_df, summary


def main() -> int:
    args = parse_args()

    annotation_csv = _resolve(args.annotation_csv)
    grouped_events_csv = _resolve(args.grouped_events_csv)
    timeline_csv = _resolve(args.timeline_csv)
    fall_events_out = _resolve(args.fall_events_out)
    har_intervals_out = _resolve(args.har_intervals_out)
    report_out = _resolve(args.report_out)

    annotations_raw = _load_csv(annotation_csv)
    grouped_df = _load_csv(grouped_events_csv)
    timeline_df = _load_csv(timeline_csv)

    annotations_df = _clean_annotations(annotations_raw)

    fall_compare_df, fall_summary = _compare_grouped_fall_events(grouped_df, annotations_df)
    fall_detect_df = _evaluate_annotated_fall_intervals(annotations_df, grouped_df)

    har_compare_df, har_summary = _compare_har_against_annotations(timeline_df, annotations_df)

    fall_events_out.parent.mkdir(parents=True, exist_ok=True)
    har_intervals_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.parent.mkdir(parents=True, exist_ok=True)

    fall_compare_df.to_csv(fall_events_out, index=False)
    har_compare_df.to_csv(har_intervals_out, index=False)

    report = {
        "evaluation_name": "runtime_annotation_comparison",
        "inputs": {
            "annotation_csv": str(annotation_csv),
            "grouped_events_csv": str(grouped_events_csv),
            "timeline_csv": str(timeline_csv),
        },
        "annotation_summary": {
            "n_annotation_rows_used": int(len(annotations_df)),
            "annotation_label_counts": annotations_df["final_label"].astype(str).value_counts(dropna=False).to_dict()
            if not annotations_df.empty
            else {},
        },
        "fall_event_summary": fall_summary,
        "annotated_fall_detection": {
            "n_fall_annotations": int(len(fall_detect_df)),
            "fall_annotation_rows": _json_safe(fall_detect_df),
            "n_detected": int(fall_detect_df["detected"].astype(bool).sum()) if not fall_detect_df.empty else 0,
        },
        "har_interval_summary": har_summary,
        "outputs": {
            "fall_events_out": str(fall_events_out),
            "har_intervals_out": str(har_intervals_out),
            "report_out": str(report_out),
        },
    }

    report_out.write_text(json.dumps(_json_safe(report), indent=2), encoding="utf-8")

    print(f"Saved fall event comparison CSV to: {fall_events_out}")
    print(f"Saved HAR interval comparison CSV to: {har_intervals_out}")
    print(f"Saved summary report JSON to: {report_out}")
    print()

    print("Annotation label counts:")
    if not annotations_df.empty:
        for label, count in annotations_df["final_label"].astype(str).value_counts(dropna=False).items():
            print(f"  {label}: {int(count)}")
    else:
        print("  no usable annotations")

    print()
    print("Grouped fall event roles:")
    if not fall_compare_df.empty:
        for role, count in fall_compare_df["event_role"].astype(str).value_counts(dropna=False).items():
            print(f"  {role}: {int(count)}")
    else:
        print("  no grouped fall events")

    if not fall_detect_df.empty:
        print()
        print("Annotated fall detection:")
        for _, row in fall_detect_df.iterrows():
            print(
                f"  fall interval {row['annotation_start_ts']:.2f}-{row['annotation_end_ts']:.2f}s "
                f"detected={bool(row['detected'])} "
                f"best_event={row['best_event_id']} "
                f"overlap={row['best_overlap_seconds']:.2f}s "
                f"peak_prob={row['best_event_peak_probability']}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())