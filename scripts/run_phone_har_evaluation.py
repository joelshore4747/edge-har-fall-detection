#!/usr/bin/env python3
"""Evaluate public-data HAR training against annotated phone windows."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.confusion_matrix_plots import plot_confusion_matrix, save_confusion_matrix_csv  # noqa: E402
from metrics.classification import compute_classification_metrics  # noqa: E402
from models.har.baselines import DEFAULT_HAR_LABEL_ORDER  # noqa: E402
from models.har.train_har import (  # noqa: E402
    DEFAULT_HAR_ALLOWED_LABELS,
    filter_har_training_rows,
    train_har_random_forest,
)
from pipeline.features import build_feature_table, feature_table_schema_summary  # noqa: E402
from pipeline.ingest import load_pamap2, load_uci_har  # noqa: E402
from pipeline.ingest.runtime_phone_folder import load_runtime_phone_folder  # noqa: E402
from pipeline.preprocess import (  # noqa: E402
    PreprocessConfig,
    append_derived_channels,
    resample_dataframe,
    window_dataframe,
)


PHONE_RUNTIME_TO_TARGET = {
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
    "phone_handling": "other",
    "sit_down_transition": "other",
    "other_transition": "other",
    "other": "other",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phone-labelled HAR evaluation")
    parser.add_argument("--train-dataset", required=True, choices=["uci_har", "pamap2", "both"])
    parser.add_argument("--ucihar-path", default="data/raw/UCIHAR_Dataset/UCI-HAR Dataset")
    parser.add_argument("--pamap2-path", default="data/raw/PAMAP2_Dataset")
    parser.add_argument("--phone-folder", required=True, help="Folder containing Accelerometer.csv / Gyroscope.csv")
    parser.add_argument("--annotation-csv", required=True, help="Filled annotation CSV with start_ts, end_ts, final_label")
    parser.add_argument("--target-rate", type=float, default=50.0)
    parser.add_argument("--window-size", type=int, default=128)
    parser.add_argument("--step-size", type=int, default=64)
    parser.add_argument("--min-overlap-fraction", type=float, default=0.25)
    parser.add_argument("--results-root", default="results/validation")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--skip-plots", action="store_true")
    return parser.parse_args()


def _resolve(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    return path


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


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")


def _build_run_id(train_dataset: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"phone_har_eval_{train_dataset}__{ts}"


def _map_phone_runtime_label(label: Any) -> str | None:
    if label is None:
        return None
    try:
        if pd.isna(label):
            return None
    except TypeError:
        pass

    text = str(label).strip().lower().replace("-", "_").replace(" ", "_")
    if text in {"", "unknown", "nan", "<na>", "none"}:
        return None
    return PHONE_RUNTIME_TO_TARGET.get(text)


def _load_annotations(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path).copy()

    for col in ["start_ts", "end_ts"]:
        if col not in df.columns:
            raise ValueError(f"Annotation CSV missing required column: {col}")
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "final_label" not in df.columns:
        raise ValueError("Annotation CSV missing required column: final_label")

    df["final_label_raw"] = df["final_label"].astype("string").str.strip().str.lower()
    df["phone_target_label"] = df["final_label_raw"].map(_map_phone_runtime_label).astype("string")

    if "row_type" not in df.columns:
        df["row_type"] = "manual"
    if "priority_rank" not in df.columns:
        df["priority_rank"] = 999999

    df = df.dropna(subset=["start_ts", "end_ts"]).copy()
    df = df[df["phone_target_label"].notna()].reset_index(drop=True)

    if df.empty:
        raise ValueError("No usable annotated rows remained after phone label mapping")

    return df


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
                "phone_target_label": str(row["phone_target_label"]),
                "final_label_raw": str(row["final_label_raw"]),
                "row_type": row_type,
                "row_type_priority": row_type_priority,
                "priority_rank": int(row.get("priority_rank", 999999)),
                "overlap_seconds": float(overlap),
                "overlap_fraction": float(overlap_fraction),
                "annotation_start_ts": ann_start,
                "annotation_end_ts": ann_end,
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


def _attach_phone_annotations(
    feature_df: pd.DataFrame,
    annotations_df: pd.DataFrame,
    *,
    min_overlap_fraction: float,
) -> pd.DataFrame:
    out = feature_df.copy()

    required_cols = {"start_ts", "end_ts"}
    missing = required_cols - set(out.columns)
    if missing:
        raise ValueError(f"Feature table missing required time columns: {sorted(missing)}")

    target_labels: list[str | None] = []
    raw_labels: list[str | None] = []
    overlap_seconds: list[float] = []
    overlap_fraction: list[float] = []
    annotation_row_type: list[str] = []

    for _, row in out.iterrows():
        start_ts = row.get("start_ts")
        end_ts = row.get("end_ts")

        if pd.isna(start_ts) or pd.isna(end_ts):
            target_labels.append(None)
            raw_labels.append(None)
            overlap_seconds.append(0.0)
            overlap_fraction.append(0.0)
            annotation_row_type.append("")
            continue

        best = _best_annotation_for_window(
            window_start=float(start_ts),
            window_end=float(end_ts),
            annotations_df=annotations_df,
            min_overlap_fraction=min_overlap_fraction,
        )

        if best is None:
            target_labels.append(None)
            raw_labels.append(None)
            overlap_seconds.append(0.0)
            overlap_fraction.append(0.0)
            annotation_row_type.append("")
        else:
            target_labels.append(str(best["phone_target_label"]))
            raw_labels.append(str(best["final_label_raw"]))
            overlap_seconds.append(float(best["overlap_seconds"]))
            overlap_fraction.append(float(best["overlap_fraction"]))
            annotation_row_type.append(str(best["row_type"]))

    out["phone_target_label"] = pd.Series(target_labels, index=out.index, dtype="string")
    out["phone_annotation_label_raw"] = pd.Series(raw_labels, index=out.index, dtype="string")
    out["phone_annotation_overlap_seconds"] = overlap_seconds
    out["phone_annotation_overlap_fraction"] = overlap_fraction
    out["phone_annotation_row_type"] = annotation_row_type
    out["is_phone_labeled"] = out["phone_target_label"].notna()

    return out


def _prepare_public_feature_table(
    dataset_name: str,
    dataset_path: Path,
    *,
    target_rate: float,
    window_size: int,
    step_size: int,
) -> pd.DataFrame:
    if dataset_name == "uci_har":
        df = load_uci_har(dataset_path)
    elif dataset_name == "pamap2":
        df = load_pamap2(dataset_path)
    else:
        raise ValueError(dataset_name)

    cfg = PreprocessConfig(target_sampling_rate_hz=target_rate)

    resampled = resample_dataframe(
        df,
        target_rate_hz=cfg.target_sampling_rate_hz,
        interpolation_method=cfg.interpolation_method,
    )
    resampled = append_derived_channels(resampled)

    windows = window_dataframe(
        resampled,
        window_size=window_size,
        step_size=step_size,
        config=cfg,
    )

    feature_df = build_feature_table(
        windows,
        filter_unacceptable=True,
        default_sampling_rate_hz=target_rate,
    )

    feature_df = filter_har_training_rows(
        feature_df,
        label_col="label_mapped_majority",
        allowed_labels=DEFAULT_HAR_ALLOWED_LABELS,
        require_acceptable=True,
    )

    if feature_df.empty:
        raise ValueError(f"Empty public HAR feature table for {dataset_name}")

    return feature_df.reset_index(drop=True)


def _prepare_phone_feature_table(
    phone_folder: Path,
    annotations_csv: Path,
    *,
    target_rate: float,
    window_size: int,
    step_size: int,
    min_overlap_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw_phone_df = load_runtime_phone_folder(phone_folder)

    cfg = PreprocessConfig(target_sampling_rate_hz=target_rate)

    resampled = resample_dataframe(
        raw_phone_df,
        target_rate_hz=cfg.target_sampling_rate_hz,
        interpolation_method=cfg.interpolation_method,
    )
    resampled = append_derived_channels(resampled)

    windows = window_dataframe(
        resampled,
        window_size=window_size,
        step_size=step_size,
        config=cfg,
    )

    feature_df = build_feature_table(
        windows,
        filter_unacceptable=False,
        default_sampling_rate_hz=target_rate,
    )

    annotations_df = _load_annotations(annotations_csv)
    feature_df = _attach_phone_annotations(
        feature_df,
        annotations_df,
        min_overlap_fraction=min_overlap_fraction,
    )

    phone_eval_df = feature_df[feature_df["is_phone_labeled"].fillna(False)].copy().reset_index(drop=True)
    phone_eval_df = phone_eval_df[phone_eval_df["phone_target_label"].isin(DEFAULT_HAR_ALLOWED_LABELS)].reset_index(drop=True)

    if phone_eval_df.empty:
        raise ValueError("No phone windows remained after annotation alignment")

    return raw_phone_df, feature_df, phone_eval_df


def _concat_training_tables(args: argparse.Namespace) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    if args.train_dataset in {"uci_har", "both"}:
        frames.append(
            _prepare_public_feature_table(
                "uci_har",
                _resolve(args.ucihar_path),
                target_rate=args.target_rate,
                window_size=args.window_size,
                step_size=args.step_size,
            )
        )

    if args.train_dataset in {"pamap2", "both"}:
        frames.append(
            _prepare_public_feature_table(
                "pamap2",
                _resolve(args.pamap2_path),
                target_rate=args.target_rate,
                window_size=args.window_size,
                step_size=args.step_size,
            )
        )

    if not frames:
        raise ValueError("No training datasets selected")

    return pd.concat(frames, ignore_index=True)


def _prepare_eval_matrix(
    eval_df: pd.DataFrame,
    feature_columns: list[str],
    fill_values: dict[str, float],
) -> pd.DataFrame:
    X = eval_df.copy()

    for col in feature_columns:
        if col not in X.columns:
            X[col] = pd.NA

    X = X[feature_columns].copy()
    X = X.fillna(fill_values).fillna(0.0)
    return X


def main() -> int:
    args = parse_args()

    phone_folder = _resolve(args.phone_folder)
    annotation_csv = _resolve(args.annotation_csv)

    if not phone_folder.exists():
        print(f"ERROR: phone folder not found: {phone_folder}")
        return 1
    if not annotation_csv.exists():
        print(f"ERROR: annotation CSV not found: {annotation_csv}")
        return 1

    run_id = args.run_id or _build_run_id(args.train_dataset)
    results_root = _resolve(args.results_root)
    run_dir = results_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Preparing public training data: {args.train_dataset}")
    public_train_df = _concat_training_tables(args)

    print("Preparing phone feature table and aligning annotations")
    raw_phone_df, full_phone_feature_df, phone_eval_df = _prepare_phone_feature_table(
        phone_folder,
        annotation_csv,
        target_rate=args.target_rate,
        window_size=args.window_size,
        step_size=args.step_size,
        min_overlap_fraction=args.min_overlap_fraction,
    )

    print(
        f"public_train_rows={len(public_train_df)} "
        f"phone_raw_rows={len(raw_phone_df)} "
        f"phone_feature_rows={len(full_phone_feature_df)} "
        f"phone_labeled_rows={len(phone_eval_df)}"
    )

    model, feature_columns, fill_values, y_train = train_har_random_forest(
        public_train_df,
        label_col="label_mapped_majority",
        allowed_labels=DEFAULT_HAR_ALLOWED_LABELS,
        require_acceptable=True,
    )

    X_phone = _prepare_eval_matrix(phone_eval_df, feature_columns, fill_values)
    y_true = phone_eval_df["phone_target_label"].astype(str).reset_index(drop=True)
    y_pred = pd.Series(model.predict(X_phone), dtype="string").astype(str).reset_index(drop=True)

    metrics = compute_classification_metrics(
        y_true.tolist(),
        y_pred.tolist(),
        labels=DEFAULT_HAR_ALLOWED_LABELS,
    )

    predictions_df = phone_eval_df.copy()
    predictions_df["predicted_label"] = y_pred
    predictions_df["is_correct"] = predictions_df["phone_target_label"].astype(str) == predictions_df["predicted_label"].astype(str)

    predictions_csv = run_dir / "phone_har_predictions.csv"
    predictions_df.to_csv(predictions_csv, index=False)

    save_confusion_matrix_csv(
        metrics["confusion_matrix"],
        DEFAULT_HAR_ALLOWED_LABELS,
        run_dir / "phone_har_confusion_matrix.csv",
    )

    if not args.skip_plots:
        try:
            plot_confusion_matrix(
                metrics["confusion_matrix"],
                DEFAULT_HAR_ALLOWED_LABELS,
                title=f"Phone HAR Confusion Matrix ({args.train_dataset} train)",
                out_path=run_dir / "phone_har_confusion_matrix.png",
            )
        except Exception as exc:  # noqa: BLE001
            print(f"plot warning: {type(exc).__name__}: {exc}")

    summary = {
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "train_dataset": args.train_dataset,
        "public_train_rows": int(len(public_train_df)),
        "public_train_label_counts": public_train_df["label_mapped_majority"].astype(str).value_counts(dropna=False).to_dict(),
        "phone_raw_rows": int(len(raw_phone_df)),
        "phone_feature_rows": int(len(full_phone_feature_df)),
        "phone_labeled_rows": int(len(phone_eval_df)),
        "phone_label_counts": phone_eval_df["phone_target_label"].astype(str).value_counts(dropna=False).to_dict(),
        "feature_table_schema_summary": feature_table_schema_summary(phone_eval_df),
        "label_order": list(DEFAULT_HAR_ALLOWED_LABELS),
        "metrics": metrics,
        "artifacts": {
            "predictions_csv": str(predictions_csv),
            "confusion_csv": str(run_dir / "phone_har_confusion_matrix.csv"),
            "confusion_png": str(run_dir / "phone_har_confusion_matrix.png")
            if (run_dir / "phone_har_confusion_matrix.png").exists()
            else None,
        },
    }

    _save_json(run_dir / "phone_har_evaluation.json", summary)

    print("\nPhone HAR evaluation summary")
    print(f"accuracy={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f}")
    print(f"phone_labeled_rows={len(phone_eval_df)}")
    print(f"saved_to={run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
