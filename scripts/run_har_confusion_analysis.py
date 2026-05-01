#!/usr/bin/env python3
"""Detailed HAR confusion analysis for within- and cross-dataset runs."""

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


from models.har.train_har import filter_har_training_rows, DEFAULT_HAR_ALLOWED_LABELS
from models.har.baselines import DEFAULT_HAR_LABEL_ORDER, heuristic_har_predict, train_random_forest_classifier
from models.har.train_har import select_feature_columns
from pipeline.features import build_feature_table
from pipeline.ingest import load_pamap2, load_uci_har
from pipeline.preprocess import PreprocessConfig, append_derived_channels, resample_dataframe, window_dataframe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HAR confusion analysis")
    parser.add_argument("--ucihar-path", default="data/raw/UCIHAR_Dataset/UCI-HAR Dataset")
    parser.add_argument("--pamap2-path", default="data/raw/PAMAP2_Dataset")
    parser.add_argument("--target-rate", type=float, default=50.0)
    parser.add_argument("--window-size", type=int, default=128)
    parser.add_argument("--step-size", type=int, default=64)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--out-json",
        default="results/validation/har_confusion_analysis.json",
        help="Output JSON path",
    )
    return parser.parse_args()


def _resolve(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    return path


def _prepare_feature_table(dataset: str, path: Path, target_rate: float, window_size: int, step_size: int) -> pd.DataFrame:
    if dataset == "ucihar":
        df = load_uci_har(path)
    elif dataset == "pamap2":
        df = load_pamap2(path)
    else:
        raise ValueError(dataset)

    cfg = PreprocessConfig(target_sampling_rate_hz=target_rate)

    resampled = resample_dataframe(
        df,
        target_rate_hz=cfg.target_sampling_rate_hz,
        interpolation_method=cfg.interpolation_method,
    )
    resampled = append_derived_channels(resampled)

    windows = window_dataframe(resampled, window_size=window_size, step_size=step_size, config=cfg)

    feat = build_feature_table(
        windows,
        filter_unacceptable=True,
        default_sampling_rate_hz=target_rate,
    )

    feat = filter_har_training_rows(
        feat,
        label_col="label_mapped_majority",
        allowed_labels=DEFAULT_HAR_ALLOWED_LABELS,
        require_acceptable=True,
    )

    if feat.empty:
        raise ValueError(f"Empty feature table for {dataset}")
    return feat


def _ordered_shared_labels(a: pd.DataFrame, b: pd.DataFrame) -> list[str]:
    a_labels = set(a["label_mapped_majority"].astype(str))
    b_labels = set(b["label_mapped_majority"].astype(str))
    shared = a_labels & b_labels

    ordered = [label for label in DEFAULT_HAR_LABEL_ORDER if label in shared]
    ordered.extend(sorted(shared - set(ordered)))
    return ordered


def _filter_labels(df: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    return df[df["label_mapped_majority"].astype(str).isin(labels)].reset_index(drop=True).copy()


def _confusion_df(y_true: pd.Series, y_pred: pd.Series, labels: list[str]) -> pd.DataFrame:
    yt = y_true.astype(str)
    yp = y_pred.astype(str)
    matrix = pd.crosstab(
        pd.Categorical(yt, categories=labels, ordered=True),
        pd.Categorical(yp, categories=labels, ordered=True),
        rownames=["true"],
        colnames=["pred"],
        dropna=False,
    )
    return matrix


def _row_normalized(df: pd.DataFrame) -> pd.DataFrame:
    denom = df.sum(axis=1).replace(0, 1)
    return df.div(denom, axis=0)


def _per_class_recall(confusion: pd.DataFrame) -> dict[str, float]:
    out: dict[str, float] = {}
    for label in confusion.index:
        total = float(confusion.loc[label].sum())
        correct = float(confusion.loc[label, label]) if label in confusion.columns else 0.0
        out[str(label)] = 0.0 if total == 0 else correct / total
    return out


def _top_confusions(confusion: pd.DataFrame, *, top_k: int = 10) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for true_label in confusion.index:
        for pred_label in confusion.columns:
            if true_label == pred_label:
                continue
            count = int(confusion.loc[true_label, pred_label])
            if count <= 0:
                continue
            items.append(
                {
                    "true_label": str(true_label),
                    "pred_label": str(pred_label),
                    "count": count,
                }
            )
    items.sort(key=lambda x: x["count"], reverse=True)
    return items[:top_k]


def _rf_predict(train_df: pd.DataFrame, test_df: pd.DataFrame, labels: list[str], random_state: int) -> pd.Series:
    train_use = _filter_labels(train_df, labels)
    test_use = _filter_labels(test_df, labels)

    if train_use.empty or test_use.empty:
        raise ValueError("Empty train/test table after label filtering in _rf_predict")

    if train_use["label_mapped_majority"].astype(str).nunique(dropna=True) < 2:
        raise ValueError("RF confusion analysis requires at least 2 training classes")

    feature_cols = select_feature_columns(train_use)
    feature_cols = [c for c in feature_cols if c in test_use.columns]
    X_train = train_use[feature_cols].copy()
    X_test = test_use[feature_cols].copy()
    y_train = train_use["label_mapped_majority"].astype(str).copy()

    fill_values = X_train.median(numeric_only=True).to_dict()
    X_train = X_train.fillna(fill_values).fillna(0.0)
    X_test = X_test.fillna(fill_values).fillna(0.0)

    model = train_random_forest_classifier(X_train, y_train, random_state=random_state)
    preds = pd.Series(model.predict(X_test), index=test_use.index, dtype="string")
    return preds


def _heuristic_predict(test_df: pd.DataFrame, labels: list[str]) -> pd.Series:
    test_use = _filter_labels(test_df, labels)
    preds = heuristic_har_predict(test_use).astype(str)
    return preds


def _block(name: str, y_true: pd.Series, y_pred: pd.Series, labels: list[str]) -> dict[str, Any]:
    label_set = set(labels)

    y_true = y_true.astype(str)
    y_pred = y_pred.astype(str)

    y_pred = y_pred.where(y_pred.isin(label_set), other="other_out_of_scope")
    y_true = y_true.where(y_true.isin(label_set), other="other_out_of_scope")

    effective_labels = list(labels)
    if "other_out_of_scope" in set(y_true) or "other_out_of_scope" in set(y_pred):
        effective_labels = [*labels, "other_out_of_scope"]

    confusion = _confusion_df(y_true, y_pred, effective_labels)
    confusion_norm = _row_normalized(confusion)

    return {
        "name": name,
        "labels": effective_labels,
        "confusion_counts": confusion.to_dict(orient="index"),
        "confusion_row_normalized": confusion_norm.round(4).to_dict(orient="index"),
        "per_class_recall": _per_class_recall(confusion),
        "top_confusions": _top_confusions(confusion),
    }


def main() -> int:
    args = parse_args()

    uci_path = _resolve(args.ucihar_path)
    pamap_path = _resolve(args.pamap2_path)
    out_json = _resolve(args.out_json)

    print("Preparing feature tables...")
    uci_df = _prepare_feature_table("ucihar", uci_path, args.target_rate, args.window_size, args.step_size)
    pamap_df = _prepare_feature_table("pamap2", pamap_path, args.target_rate, args.window_size, args.step_size)

    within_uci_labels = sorted(uci_df["label_mapped_majority"].astype(str).unique().tolist())
    within_pamap_labels = sorted(pamap_df["label_mapped_majority"].astype(str).unique().tolist())
    shared_labels = _ordered_shared_labels(uci_df, pamap_df)

    # Within UCI
    uci_within_true = _filter_labels(uci_df, within_uci_labels)["label_mapped_majority"].astype(str)
    uci_within_heur = _heuristic_predict(uci_df, within_uci_labels)
    uci_within_rf = _rf_predict(uci_df, uci_df, within_uci_labels, args.random_state)

    # Within PAMAP2
    pamap_within_true = _filter_labels(pamap_df, within_pamap_labels)["label_mapped_majority"].astype(str)
    pamap_within_heur = _heuristic_predict(pamap_df, within_pamap_labels)
    pamap_within_rf = _rf_predict(pamap_df, pamap_df, within_pamap_labels, args.random_state)

    # Cross UCI -> PAMAP2
    pamap_shared_true = _filter_labels(pamap_df, shared_labels)["label_mapped_majority"].astype(str)
    uci_to_pamap_heur = _heuristic_predict(pamap_df, shared_labels)
    uci_to_pamap_rf = _rf_predict(uci_df, pamap_df, shared_labels, args.random_state)

    # Cross PAMAP2 -> UCI
    uci_shared_true = _filter_labels(uci_df, shared_labels)["label_mapped_majority"].astype(str)
    pamap_to_uci_heur = _heuristic_predict(uci_df, shared_labels)
    pamap_to_uci_rf = _rf_predict(pamap_df, uci_df, shared_labels, args.random_state)

    payload = {
        "within_dataset": {
            "UCIHAR_heuristic": _block("UCIHAR_heuristic", uci_within_true, uci_within_heur, within_uci_labels),
            "UCIHAR_rf_train_eq_test": _block("UCIHAR_rf_train_eq_test", uci_within_true, uci_within_rf, within_uci_labels),
            "PAMAP2_heuristic": _block("PAMAP2_heuristic", pamap_within_true, pamap_within_heur, within_pamap_labels),
            "PAMAP2_rf_train_eq_test": _block("PAMAP2_rf_train_eq_test", pamap_within_true, pamap_within_rf, within_pamap_labels),
        },
        "cross_dataset": {
            "UCIHAR_to_PAMAP2_heuristic": _block("UCIHAR_to_PAMAP2_heuristic", pamap_shared_true, uci_to_pamap_heur, shared_labels),
            "UCIHAR_to_PAMAP2_rf": _block("UCIHAR_to_PAMAP2_rf", pamap_shared_true, uci_to_pamap_rf, shared_labels),
            "PAMAP2_to_UCIHAR_heuristic": _block("PAMAP2_to_UCIHAR_heuristic", uci_shared_true, pamap_to_uci_heur, shared_labels),
            "PAMAP2_to_UCIHAR_rf": _block("PAMAP2_to_UCIHAR_rf", uci_shared_true, pamap_to_uci_rf, shared_labels),
        },
        "shared_labels": shared_labels,
        "note": "Within-dataset RF here is train==test confusion only for class-pattern inspection, not headline performance.",
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Saved HAR confusion analysis to: {out_json}")
    print(f"Shared labels: {shared_labels}")
    print("Top cross-dataset RF confusions:")
    for item in payload["cross_dataset"]["UCIHAR_to_PAMAP2_rf"]["top_confusions"][:5]:
        print(f"  UCI->PAMAP RF: true={item['true_label']} pred={item['pred_label']} count={item['count']}")
    for item in payload["cross_dataset"]["PAMAP2_to_UCIHAR_rf"]["top_confusions"][:5]:
        print(f"  P->U RF: true={item['true_label']} pred={item['pred_label']} count={item['count']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())