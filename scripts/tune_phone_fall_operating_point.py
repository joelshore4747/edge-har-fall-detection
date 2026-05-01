#!/usr/bin/env python3
"""Tune the runtime operating threshold for phone fall predictions.

Purpose:
- sweep probability thresholds over saved phone prediction outputs
- measure the precision/recall/F1 tradeoff without retraining
- inspect false-positive breakdown by phone hard-negative type
- choose a runtime threshold for deployment

Expected input:
- CSV produced by scripts/train_fall_with_phone_hard_negatives.py
  e.g. results/validation/fall_phone_adaptation_phone_predictions_light.csv

By default this tunes the adapted model probabilities, but it can also tune the baseline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from metrics.fall_metrics import compute_fall_detection_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune phone fall operating threshold")
    parser.add_argument(
        "--predictions-csv",
        default="results/validation/fall_phone_adaptation_phone_predictions_light.csv",
        help="Phone prediction comparison CSV from adaptation script",
    )
    parser.add_argument(
        "--model",
        choices=["adapted", "baseline"],
        default="adapted",
        help="Which model probabilities to tune",
    )
    parser.add_argument(
        "--threshold-start",
        type=float,
        default=0.05,
        help="Start of threshold sweep",
    )
    parser.add_argument(
        "--threshold-stop",
        type=float,
        default=0.95,
        help="End of threshold sweep (inclusive if it lands exactly on a step)",
    )
    parser.add_argument(
        "--threshold-step",
        type=float,
        default=0.05,
        help="Step size for threshold sweep",
    )
    parser.add_argument(
        "--select-by",
        choices=["f1", "precision", "recall", "balanced_score", "specificity"],
        default="f1",
        help="Metric used to choose the best threshold",
    )
    parser.add_argument(
        "--min-recall",
        type=float,
        default=0.0,
        help="Optional minimum recall constraint when choosing the best threshold",
    )
    parser.add_argument(
        "--max-false-positives",
        type=int,
        default=-1,
        help="Optional max false positives constraint (-1 disables)",
    )
    parser.add_argument(
        "--out-csv",
        default="results/validation/phone_fall_operating_point_sweep.csv",
        help="Threshold sweep output CSV",
    )
    parser.add_argument(
        "--out-json",
        default="results/validation/phone_fall_operating_point_best.json",
        help="Best-threshold summary JSON",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="How many top thresholds to print",
    )
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


def _load_predictions(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Predictions CSV not found: {path}")
    df = pd.read_csv(path)
    if "true_label" not in df.columns:
        raise ValueError("Predictions CSV must contain true_label")
    return df


def _required_probability_column(model_name: str) -> str:
    return f"{model_name}_predicted_probability"


def _predicted_label_column(model_name: str) -> str:
    return f"{model_name}_predicted_label"


def _compute_metrics(
    y_true_labels: pd.Series,
    y_prob: np.ndarray,
    *,
    threshold: float,
) -> dict[str, Any]:
    y_pred_labels = np.where(y_prob >= float(threshold), "fall", "non_fall")

    metrics = compute_fall_detection_metrics(
        y_true_labels.astype(str).tolist(),
        pd.Series(y_pred_labels, dtype="string").astype(str).tolist(),
        positive_label="fall",
        negative_label="non_fall",
    )

    y_true_bin = y_true_labels.astype(str).map({"fall": 1, "non_fall": 0}).astype(int).to_numpy()
    if len(np.unique(y_true_bin)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true_bin, y_prob))
        metrics["average_precision"] = float(average_precision_score(y_true_bin, y_prob))
    else:
        metrics["roc_auc"] = float("nan")
        metrics["average_precision"] = float("nan")

    metrics["brier_score"] = float(brier_score_loss(y_true_bin, y_prob))
    metrics["probability_threshold"] = float(threshold)
    metrics["balanced_score"] = float((metrics["precision"] + metrics["sensitivity"]) / 2.0)
    return metrics


def _false_positive_breakdown(
    df: pd.DataFrame,
    y_pred_labels: np.ndarray,
) -> dict[str, int]:
    working = df.copy()
    working["runtime_predicted_label"] = pd.Series(y_pred_labels, index=working.index, dtype="string")

    fp_df = working[
        working["true_label"].astype(str).eq("non_fall")
        & working["runtime_predicted_label"].astype(str).eq("fall")
    ].copy()

    if fp_df.empty:
        return {}

    if "fall_hard_negative_type" in fp_df.columns:
        s = fp_df["fall_hard_negative_type"].dropna().astype(str)
        if not s.empty:
            return {str(k): int(v) for k, v in s.value_counts(dropna=False).items()}

    if "annotation_label" in fp_df.columns:
        s = fp_df["annotation_label"].dropna().astype(str)
        if not s.empty:
            return {str(k): int(v) for k, v in s.value_counts(dropna=False).items()}

    return {"all_false_positives": int(len(fp_df))}


def _threshold_grid(start: float, stop: float, step: float) -> list[float]:
    if step <= 0:
        raise ValueError("threshold-step must be > 0")
    values: list[float] = []
    cur = float(start)
    while cur <= float(stop) + 1e-12:
        values.append(round(cur, 10))
        cur += float(step)
    return values


def _pick_best_threshold(
    sweep_df: pd.DataFrame,
    *,
    metric_name: str,
    min_recall: float,
    max_false_positives: int,
) -> pd.Series:
    working = sweep_df.copy()

    if min_recall > 0:
        working = working[working["sensitivity"] >= float(min_recall)].copy()

    if max_false_positives >= 0:
        working = working[working["fp"] <= int(max_false_positives)].copy()

    if working.empty:
        raise ValueError(
            "No thresholds satisfy the requested constraints. "
            "Relax min_recall or max_false_positives."
        )

    sort_cols = [metric_name, "precision", "sensitivity", "specificity", "probability_threshold"]
    ascending = [False, False, False, False, True]
    working = working.sort_values(sort_cols, ascending=ascending, kind="stable").reset_index(drop=True)
    return working.iloc[0]


def main() -> int:
    args = parse_args()

    predictions_csv = _resolve(args.predictions_csv)
    out_csv = _resolve(args.out_csv)
    out_json = _resolve(args.out_json)

    df = _load_predictions(predictions_csv)

    prob_col = _required_probability_column(args.model)
    if prob_col not in df.columns:
        raise ValueError(f"Predictions CSV missing required column: {prob_col}")

    y_true = df["true_label"].astype(str)
    y_prob = pd.to_numeric(df[prob_col], errors="coerce")
    valid_mask = y_true.isin(["fall", "non_fall"]) & y_prob.notna()
    df = df.loc[valid_mask].reset_index(drop=True)
    y_true = df["true_label"].astype(str)
    y_prob = pd.to_numeric(df[prob_col], errors="coerce").to_numpy(dtype=float)

    thresholds = _threshold_grid(args.threshold_start, args.threshold_stop, args.threshold_step)

    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        metrics = _compute_metrics(y_true, y_prob, threshold=float(threshold))
        y_pred_labels = np.where(y_prob >= float(threshold), "fall", "non_fall")
        fp_breakdown = _false_positive_breakdown(df, y_pred_labels)

        row = dict(metrics)
        row["false_positive_breakdown"] = json.dumps(fp_breakdown, sort_keys=True)
        rows.append(row)

    sweep_df = pd.DataFrame(rows)

    best_row = _pick_best_threshold(
        sweep_df,
        metric_name=args.select_by,
        min_recall=float(args.min_recall),
        max_false_positives=int(args.max_false_positives),
    )

    best_threshold = float(best_row["probability_threshold"])
    best_pred_labels = np.where(y_prob >= best_threshold, "fall", "non_fall")
    best_fp_breakdown = _false_positive_breakdown(df, best_pred_labels)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    sweep_df.to_csv(out_csv, index=False)

    summary = {
        "evaluation_name": "phone_fall_operating_point_tuning",
        "inputs": {
            "predictions_csv": str(predictions_csv),
            "model": args.model,
            "threshold_start": float(args.threshold_start),
            "threshold_stop": float(args.threshold_stop),
            "threshold_step": float(args.threshold_step),
            "select_by": args.select_by,
            "min_recall": float(args.min_recall),
            "max_false_positives": int(args.max_false_positives),
        },
        "best_threshold": best_threshold,
        "best_metrics": {k: _json_safe(v) for k, v in best_row.to_dict().items()},
        "best_false_positive_breakdown": best_fp_breakdown,
        "outputs": {
            "out_csv": str(out_csv),
            "out_json": str(out_json),
        },
    }

    out_json.write_text(json.dumps(_json_safe(summary), indent=2), encoding="utf-8")

    ranked = sweep_df.sort_values(
        [args.select_by, "precision", "sensitivity", "specificity", "probability_threshold"],
        ascending=[False, False, False, False, True],
        kind="stable",
    ).reset_index(drop=True)

    print(f"Saved threshold sweep CSV to: {out_csv}")
    print(f"Saved best-threshold JSON to: {out_json}")
    print()
    print(f"Best threshold for model={args.model}: {best_threshold:.2f}")
    print(f"Selected by: {args.select_by}")
    print(
        f"F1={best_row['f1']:.4f} "
        f"precision={best_row['precision']:.4f} "
        f"recall={best_row['sensitivity']:.4f} "
        f"specificity={best_row['specificity']:.4f} "
        f"fp={int(best_row['fp'])} "
        f"tp={int(best_row['tp'])}"
    )
    print(f"False-positive breakdown: {best_fp_breakdown}")
    print()
    print(f"Top {min(args.top_k, len(ranked))} thresholds:")
    cols = [
        "probability_threshold",
        "f1",
        "precision",
        "sensitivity",
        "specificity",
        "fp",
        "tp",
        "fn",
        "tn",
    ]
    print(ranked.loc[:, cols].head(args.top_k).to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())