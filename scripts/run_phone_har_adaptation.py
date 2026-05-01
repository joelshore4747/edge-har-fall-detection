#!/usr/bin/env python3
"""Evaluate HAR phone adaptation: public-only vs public+phone-labeled adaptation."""

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
from models.har.train_har import (  # noqa: E402
    DEFAULT_HAR_ALLOWED_LABELS,
    filter_har_training_rows,
    select_feature_columns,
)
from models.har.baselines import train_random_forest_classifier  # noqa: E402
from pipeline.features import feature_table_schema_summary  # noqa: E402
from scripts.run_phone_har_evaluation import (  # noqa: E402
    _prepare_phone_feature_table,
    _prepare_public_feature_table,
    _resolve,
    _save_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phone HAR adaptation experiment")
    parser.add_argument("--train-dataset", required=True, choices=["uci_har", "pamap2", "both"])
    parser.add_argument("--ucihar-path", default="data/raw/UCIHAR_Dataset/UCI-HAR Dataset")
    parser.add_argument("--pamap2-path", default="data/raw/PAMAP2_Dataset")
    parser.add_argument("--phone-folder", required=True)
    parser.add_argument("--annotation-csv", required=True)
    parser.add_argument("--target-rate", type=float, default=50.0)
    parser.add_argument("--window-size", type=int, default=128)
    parser.add_argument("--step-size", type=int, default=64)
    parser.add_argument("--min-overlap-fraction", type=float, default=0.25)
    parser.add_argument(
        "--adapt-fracs",
        default="0.1,0.25,0.5",
        help="Comma-separated fractions of earliest labeled phone windows to use for adaptation training",
    )
    parser.add_argument(
        "--holdout-frac",
        type=float,
        default=0.4,
        help="Final fraction of phone labeled windows reserved as held-out test set",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--results-root", default="results/validation")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--skip-plots", action="store_true")
    return parser.parse_args()


def _build_run_id(train_dataset: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"phone_har_adapt_{train_dataset}__{ts}"


def _concat_public_training(args: argparse.Namespace) -> pd.DataFrame:
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
        raise ValueError("No public HAR training data selected")

    return pd.concat(frames, ignore_index=True)


def _parse_fracs(raw: str) -> list[float]:
    out: list[float] = []
    for part in str(raw).split(","):
        text = part.strip()
        if not text:
            continue
        value = float(text)
        if not (0.0 < value < 1.0):
            raise ValueError(f"Invalid adaptation fraction: {value}")
        out.append(value)
    if not out:
        raise ValueError("No adaptation fractions provided")
    return out


def _chronological_phone_split(phone_eval_df: pd.DataFrame, *, holdout_frac: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    if phone_eval_df.empty:
        raise ValueError("phone_eval_df is empty")
    if not (0.0 < holdout_frac < 1.0):
        raise ValueError("holdout_frac must be in (0,1)")

    working = phone_eval_df.copy()
    if "midpoint_ts" in working.columns:
        working["midpoint_ts"] = pd.to_numeric(working["midpoint_ts"], errors="coerce")
        working = working.sort_values(["midpoint_ts", "window_id"], kind="stable")
    else:
        working = working.sort_values(["window_id"], kind="stable")
    working = working.reset_index(drop=True)

    n = len(working)
    test_n = max(1, int(round(n * holdout_frac)))
    train_n = n - test_n
    if train_n < 1:
        raise ValueError("Not enough phone labeled rows after holdout split")

    adapt_pool = working.iloc[:train_n].reset_index(drop=True)
    holdout_test = working.iloc[train_n:].reset_index(drop=True)
    return adapt_pool, holdout_test


def _subset_phone_adapt_pool(adapt_pool: pd.DataFrame, frac: float) -> pd.DataFrame:
    n = len(adapt_pool)
    k = max(1, int(round(n * frac)))
    return adapt_pool.iloc[:k].reset_index(drop=True)


def _fit_and_score(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    train_label_col: str,
    test_label_col: str,
    label_order: list[str],
    random_state: int,
) -> tuple[dict[str, Any], pd.DataFrame, list[str], dict[str, float]]:
    if train_df.empty:
        raise ValueError("Training dataframe is empty")
    if test_df.empty:
        raise ValueError("Test dataframe is empty")

    feature_cols = select_feature_columns(train_df)
    feature_cols = [c for c in feature_cols if c in test_df.columns]
    if not feature_cols:
        raise ValueError("No overlapping feature columns between train and test")

    X_train = train_df[feature_cols].copy()
    y_train = train_df[train_label_col].astype(str).copy()

    fill_values = X_train.median(numeric_only=True).to_dict()
    fill_values = {str(k): float(v) for k, v in fill_values.items()}

    X_train = X_train.fillna(fill_values).fillna(0.0)

    X_test = test_df.copy()
    for col in feature_cols:
        if col not in X_test.columns:
            X_test[col] = pd.NA
    X_test = X_test[feature_cols].copy().fillna(fill_values).fillna(0.0)

    model = train_random_forest_classifier(X_train, y_train, random_state=random_state)

    y_true = test_df[test_label_col].astype(str).reset_index(drop=True)
    y_pred = pd.Series(model.predict(X_test), dtype="string").astype(str).reset_index(drop=True)

    metrics = compute_classification_metrics(
        y_true.tolist(),
        y_pred.tolist(),
        labels=label_order,
    )

    preds_df = test_df.copy()
    preds_df["predicted_label"] = y_pred
    preds_df["is_correct"] = preds_df[test_label_col].astype(str) == preds_df["predicted_label"].astype(str)

    return metrics, preds_df, feature_cols, fill_values


def main() -> int:
    args = parse_args()

    adapt_fracs = _parse_fracs(args.adapt_fracs)
    phone_folder = _resolve(args.phone_folder)
    annotation_csv = _resolve(args.annotation_csv)

    run_id = args.run_id or _build_run_id(args.train_dataset)
    results_root = _resolve(args.results_root)
    run_dir = results_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    public_train_df = _concat_public_training(args)

    _, _, phone_eval_df = _prepare_phone_feature_table(
        phone_folder,
        annotation_csv,
        target_rate=args.target_rate,
        window_size=args.window_size,
        step_size=args.step_size,
        min_overlap_fraction=args.min_overlap_fraction,
    )

    phone_eval_df = phone_eval_df.copy()
    phone_eval_df = phone_eval_df[phone_eval_df["phone_target_label"].isin(DEFAULT_HAR_ALLOWED_LABELS)].reset_index(drop=True)

    adapt_pool, holdout_test = _chronological_phone_split(
        phone_eval_df,
        holdout_frac=args.holdout_frac,
    )

    print(
        f"public_train_rows={len(public_train_df)} "
        f"phone_labeled_rows={len(phone_eval_df)} "
        f"adapt_pool_rows={len(adapt_pool)} "
        f"holdout_test_rows={len(holdout_test)}"
    )

    runs: dict[str, Any] = {}

    # Public-only baseline on held-out phone test
    baseline_metrics, baseline_preds_df, baseline_feature_cols, baseline_fill_values = _fit_and_score(
        public_train_df,
        holdout_test,
        train_label_col="label_mapped_majority",
        test_label_col="phone_target_label",
        label_order=list(DEFAULT_HAR_ALLOWED_LABELS),
        random_state=args.random_state,
    )
    baseline_name = "public_only"
    runs[baseline_name] = {
        "metrics": baseline_metrics,
        "train_rows": int(len(public_train_df)),
        "test_rows": int(len(holdout_test)),
        "feature_columns_count": int(len(baseline_feature_cols)),
        "fill_values_count": int(len(baseline_fill_values)),
    }
    baseline_preds_df.to_csv(run_dir / f"{baseline_name}_predictions.csv", index=False)
    save_confusion_matrix_csv(
        baseline_metrics["confusion_matrix"],
        list(DEFAULT_HAR_ALLOWED_LABELS),
        run_dir / f"{baseline_name}_confusion_matrix.csv",
    )

    # Public + phone adaptation
    for frac in adapt_fracs:
        phone_adapt_df = _subset_phone_adapt_pool(adapt_pool, frac)
        phone_adapt_df = phone_adapt_df.copy()
        phone_adapt_df["label_mapped_majority"] = phone_adapt_df["phone_target_label"].astype("string")

        combined_train = pd.concat([public_train_df, phone_adapt_df], ignore_index=True)
        combined_train = filter_har_training_rows(
            combined_train,
            label_col="label_mapped_majority",
            allowed_labels=DEFAULT_HAR_ALLOWED_LABELS,
            require_acceptable=False,
        )

        name = f"public_plus_phone_{int(round(frac * 100))}pct"
        metrics, preds_df, feature_cols, fill_values = _fit_and_score(
            combined_train,
            holdout_test,
            train_label_col="label_mapped_majority",
            test_label_col="phone_target_label",
            label_order=list(DEFAULT_HAR_ALLOWED_LABELS),
            random_state=args.random_state,
        )

        runs[name] = {
            "metrics": metrics,
            "train_rows": int(len(combined_train)),
            "phone_adapt_rows": int(len(phone_adapt_df)),
            "test_rows": int(len(holdout_test)),
            "feature_columns_count": int(len(feature_cols)),
            "fill_values_count": int(len(fill_values)),
        }

        preds_df.to_csv(run_dir / f"{name}_predictions.csv", index=False)
        save_confusion_matrix_csv(
            metrics["confusion_matrix"],
            list(DEFAULT_HAR_ALLOWED_LABELS),
            run_dir / f"{name}_confusion_matrix.csv",
        )

        if not args.skip_plots:
            try:
                plot_confusion_matrix(
                    metrics["confusion_matrix"],
                    list(DEFAULT_HAR_ALLOWED_LABELS),
                    title=f"Phone HAR Adaptation ({name})",
                    out_path=run_dir / f"{name}_confusion_matrix.png",
                )
            except Exception as exc:  # noqa: BLE001
                print(f"plot warning for {name}: {type(exc).__name__}: {exc}")

    summary = {
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "train_dataset": args.train_dataset,
        "adapt_fracs": adapt_fracs,
        "holdout_frac": float(args.holdout_frac),
        "public_train_rows": int(len(public_train_df)),
        "phone_eval_rows": int(len(phone_eval_df)),
        "phone_adapt_pool_rows": int(len(adapt_pool)),
        "phone_holdout_test_rows": int(len(holdout_test)),
        "phone_label_counts_full": phone_eval_df["phone_target_label"].astype(str).value_counts(dropna=False).to_dict(),
        "phone_label_counts_holdout": holdout_test["phone_target_label"].astype(str).value_counts(dropna=False).to_dict(),
        "feature_table_schema_summary": feature_table_schema_summary(phone_eval_df),
        "label_order": list(DEFAULT_HAR_ALLOWED_LABELS),
        "runs": runs,
    }

    _save_json(run_dir / "phone_har_adaptation_summary.json", summary)

    print("\nPhone HAR adaptation summary")
    for name, payload in runs.items():
        metrics = payload["metrics"]
        print(
            f"{name}: accuracy={metrics['accuracy']:.4f} "
            f"macro_f1={metrics['macro_f1']:.4f} "
            f"train_rows={payload['train_rows']}"
        )

    print(f"saved_to={run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())