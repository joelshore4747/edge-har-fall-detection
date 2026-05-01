from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


HAR_RUNTIME_TO_TARGET = {
    "walking": "locomotion",
    "walk": "locomotion",

    "stairs": "stairs",
    "upstairs": "stairs",
    "downstairs": "stairs",

    "standing": "static",
    "sitting": "static",
    "static": "static",
    "lying": "static",
    "laying": "static",

    "other": "other",
    "phone_handling": "other",
    "sit_down_transition": "other",
    "other_transition": "other",
}

FALL_NEGATIVE_LABELS = {
    "walking",
    "walk",
    "stairs",
    "upstairs",
    "downstairs",
    "standing",
    "sitting",
    "static",
    "lying",
    "laying",
    "sit_down_transition",
    "phone_handling",
    "other",
    "other_transition",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build phone runtime labeled window tables")
    parser.add_argument(
        "--annotation-csv",
        default="results/validation/phone1_annotation_template.csv",
        help="Filled annotation CSV",
    )
    parser.add_argument(
        "--har-csv",
        default="results/validation/phone1_har.csv",
        help="HAR replay CSV",
    )
    parser.add_argument(
        "--fall-csv",
        default="results/validation/phone1_fall.csv",
        help="Fall replay CSV",
    )
    parser.add_argument(
        "--har-out",
        default="results/validation/phone1_har_labeled_windows.csv",
        help="Output HAR labeled window CSV",
    )
    parser.add_argument(
        "--fall-out",
        default="results/validation/phone1_fall_labeled_windows.csv",
        help="Output fall labeled window CSV",
    )
    parser.add_argument(
        "--summary-out",
        default="results/validation/phone1_runtime_training_summary.json",
        help="Output summary JSON",
    )
    parser.add_argument(
        "--min-overlap-fraction",
        type=float,
        default=0.25,
        help=(
            "Minimum fraction of a window that must overlap an annotation interval "
            "before the window is considered labeled"
        ),
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

    label_series = out["final_label"].astype("string").str.strip().str.lower()
    label_series = label_series.replace({
        "": pd.NA,
        "unknown": pd.NA,
        "nan": pd.NA,
        "none": pd.NA,
        "<na>": pd.NA,
    })
    out["final_label"] = label_series
    out = out[out["final_label"].notna()]

    if "row_type" not in out.columns:
        out["row_type"] = "manual"

    if "priority_rank" not in out.columns:
        out["priority_rank"] = 999999

    return out.reset_index(drop=True)


def _window_time_columns(df: pd.DataFrame) -> tuple[str, str]:
    start_col = "start_ts" if "start_ts" in df.columns else None
    end_col = "end_ts" if "end_ts" in df.columns else None

    if start_col is None or end_col is None:
        raise ValueError("Window CSV must contain start_ts and end_ts")

    return start_col, end_col


def _time_overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    start = max(float(a_start), float(b_start))
    end = min(float(a_end), float(b_end))
    return max(0.0, end - start)


def _best_annotation_for_window(
    *,
    window_start: float,
    window_end: float,
    annotations_df: pd.DataFrame,
    min_overlap_fraction: float,
) -> dict[str, Any] | None:
    window_duration = max(1e-9, float(window_end - window_start))
    rows: list[dict[str, Any]] = []

    for _, row in annotations_df.iterrows():
        ann_start = float(row["start_ts"])
        ann_end = float(row["end_ts"])
        overlap = _time_overlap(window_start, window_end, ann_start, ann_end)
        if overlap <= 0:
            continue

        overlap_fraction = float(overlap / window_duration)

        row_type = str(row.get("row_type", "manual")).strip().lower()
        row_type_priority = 0 if row_type == "manual" else 1

        rows.append(
            {
                "row_type": row_type,
                "row_type_priority": row_type_priority,
                "priority_rank": int(row.get("priority_rank", 999999)),
                "final_label": str(row["final_label"]),
                "notes": str(row.get("notes", "")),
                "annotation_start_ts": ann_start,
                "annotation_end_ts": ann_end,
                "overlap_seconds": float(overlap),
                "overlap_fraction": float(overlap_fraction),
            }
        )

    if not rows:
        return None

    overlaps_df = pd.DataFrame(rows).sort_values(
        ["overlap_fraction", "overlap_seconds", "row_type_priority", "priority_rank"],
        ascending=[False, False, True, True],
        kind="stable",
    ).reset_index(drop=True)

    best = overlaps_df.iloc[0].to_dict()
    if float(best["overlap_fraction"]) < float(min_overlap_fraction):
        return None
    return best


def _attach_annotations_to_windows(
    window_df: pd.DataFrame,
    annotations_df: pd.DataFrame,
    *,
    min_overlap_fraction: float,
) -> pd.DataFrame:
    out = window_df.copy()
    start_col, end_col = _window_time_columns(out)

    out[start_col] = pd.to_numeric(out[start_col], errors="coerce")
    out[end_col] = pd.to_numeric(out[end_col], errors="coerce")

    ann_label: list[str | None] = []
    ann_notes: list[str] = []
    ann_row_type: list[str] = []
    ann_overlap_seconds: list[float] = []
    ann_overlap_fraction: list[float] = []

    for _, row in out.iterrows():
        best = _best_annotation_for_window(
            window_start=float(row[start_col]),
            window_end=float(row[end_col]),
            annotations_df=annotations_df,
            min_overlap_fraction=min_overlap_fraction,
        )
        if best is None:
            ann_label.append(None)
            ann_notes.append("")
            ann_row_type.append("")
            ann_overlap_seconds.append(0.0)
            ann_overlap_fraction.append(0.0)
        else:
            ann_label.append(str(best["final_label"]))
            ann_notes.append(str(best["notes"]))
            ann_row_type.append(str(best["row_type"]))
            ann_overlap_seconds.append(float(best["overlap_seconds"]))
            ann_overlap_fraction.append(float(best["overlap_fraction"]))

    out["annotation_label"] = pd.Series(ann_label, index=out.index, dtype="string")
    out["annotation_notes"] = ann_notes
    out["annotation_row_type"] = ann_row_type
    out["annotation_overlap_seconds"] = ann_overlap_seconds
    out["annotation_overlap_fraction"] = ann_overlap_fraction
    out["is_labeled_from_annotation"] = out["annotation_label"].notna()

    return out


def _build_har_labeled_table(window_df: pd.DataFrame) -> pd.DataFrame:
    out = window_df.copy()

    out["har_runtime_label"] = out["annotation_label"].astype("string")
    out["har_target_label"] = out["har_runtime_label"].map(HAR_RUNTIME_TO_TARGET).astype("string")
    out["use_for_har_supervision"] = out["har_target_label"].notna()

    return out


def _build_fall_labeled_table(window_df: pd.DataFrame) -> pd.DataFrame:
    out = window_df.copy()

    def map_fall_label(label: Any) -> str | None:
        if pd.isna(label):
            return None

        label_s = str(label).strip().lower().replace("-", "_").replace(" ", "_")

        if label_s == "fall":
            return "fall"
        if label_s in FALL_NEGATIVE_LABELS:
            return "non_fall"
        return None

    out["fall_runtime_label"] = out["annotation_label"].astype("string")
    out["fall_target_label"] = out["fall_runtime_label"].map(map_fall_label).astype("string")
    out["fall_hard_negative_type"] = out["fall_runtime_label"].where(
        out["fall_target_label"].astype("string").eq("non_fall"),
        pd.NA,
    ).astype("string")
    out["use_for_fall_supervision"] = out["fall_target_label"].notna()

    return out


def _summary_counts(df: pd.DataFrame, col: str) -> dict[str, int]:
    if col not in df.columns:
        return {}
    series = df[col].dropna().astype(str)
    if series.empty:
        return {}
    return {str(k): int(v) for k, v in series.value_counts(dropna=False).items()}


def main() -> int:
    args = parse_args()

    annotation_csv = _resolve(args.annotation_csv)
    har_csv = _resolve(args.har_csv)
    fall_csv = _resolve(args.fall_csv)
    har_out = _resolve(args.har_out)
    fall_out = _resolve(args.fall_out)
    summary_out = _resolve(args.summary_out)

    annotations_raw = _load_csv(annotation_csv)
    har_df = _load_csv(har_csv)
    fall_df = _load_csv(fall_csv)

    annotations_df = _clean_annotations(annotations_raw)

    har_labeled = _attach_annotations_to_windows(
        har_df,
        annotations_df,
        min_overlap_fraction=args.min_overlap_fraction,
    )
    har_labeled = _build_har_labeled_table(har_labeled)

    fall_labeled = _attach_annotations_to_windows(
        fall_df,
        annotations_df,
        min_overlap_fraction=args.min_overlap_fraction,
    )
    fall_labeled = _build_fall_labeled_table(fall_labeled)

    har_out.parent.mkdir(parents=True, exist_ok=True)
    fall_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.parent.mkdir(parents=True, exist_ok=True)

    har_labeled.to_csv(har_out, index=False)
    fall_labeled.to_csv(fall_out, index=False)

    summary = {
        "evaluation_name": "phone_runtime_training_set_build",
        "inputs": {
            "annotation_csv": str(annotation_csv),
            "har_csv": str(har_csv),
            "fall_csv": str(fall_csv),
            "min_overlap_fraction": float(args.min_overlap_fraction),
        },
        "annotations": {
            "rows_used": int(len(annotations_df)),
            "label_counts": _summary_counts(annotations_df, "final_label"),
        },
        "har": {
            "rows_total": int(len(har_labeled)),
            "rows_with_annotation": int(har_labeled["is_labeled_from_annotation"].astype(bool).sum()),
            "rows_for_har_supervision": int(har_labeled["use_for_har_supervision"].astype(bool).sum()),
            "runtime_label_counts": _summary_counts(har_labeled, "har_runtime_label"),
            "target_label_counts": _summary_counts(har_labeled, "har_target_label"),
        },
        "fall": {
            "rows_total": int(len(fall_labeled)),
            "rows_with_annotation": int(fall_labeled["is_labeled_from_annotation"].astype(bool).sum()),
            "rows_for_fall_supervision": int(fall_labeled["use_for_fall_supervision"].astype(bool).sum()),
            "runtime_label_counts": _summary_counts(fall_labeled, "fall_runtime_label"),
            "target_label_counts": _summary_counts(fall_labeled, "fall_target_label"),
            "hard_negative_type_counts": _summary_counts(fall_labeled, "fall_hard_negative_type"),
        },
        "outputs": {
            "har_out": str(har_out),
            "fall_out": str(fall_out),
            "summary_out": str(summary_out),
        },
    }

    summary_out.write_text(json.dumps(_json_safe(summary), indent=2), encoding="utf-8")

    print(f"Saved HAR labeled windows to: {har_out}")
    print(f"Saved fall labeled windows to: {fall_out}")
    print(f"Saved summary JSON to: {summary_out}")
    print()
    print("HAR supervision rows:", summary["har"]["rows_for_har_supervision"])
    print("Fall supervision rows:", summary["fall"]["rows_for_fall_supervision"])
    print("Fall hard-negative types:", summary["fall"]["hard_negative_type_counts"])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
