#!/usr/bin/env python3
"""Evaluate simple unsupervised domain-adaptation baselines for HAR LODO transfer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from metrics.classification import compute_classification_metrics
from models.har.baselines import train_random_forest_classifier
from models.har.train_har import select_feature_columns
from scripts.run_har_cross_dataset_eval import (
    _filter_to_labels,
    _json_safe,
    _ordered_shared_labels,
    _prepare_feature_table,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run RF and lightweight unsupervised domain-adaptation baselines "
            "on UCIHAR<->PAMAP2 HAR transfer."
        )
    )
    parser.add_argument("--ucihar-path", default="data/raw/UCIHAR_Dataset")
    parser.add_argument("--pamap2-path", default="data/raw/PAMAP2_Dataset")
    parser.add_argument("--pamap2-include-optional", action="store_true")
    parser.add_argument("--target-rate", type=float, default=50.0)
    parser.add_argument("--window-size", type=int, default=128)
    parser.add_argument("--step-size", type=int, default=64)
    parser.add_argument("--keep-unacceptable", action="store_true")
    parser.add_argument("--ucihar-sample-limit", type=int, default=0, help="0 = full dataset")
    parser.add_argument("--pamap2-sample-limit", type=int, default=0, help="0 = full dataset")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--bootstrap-confidence", type=float, default=0.95)
    parser.add_argument(
        "--out-json",
        default="results/validation/har_domain_adaptation_eval.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--out-csv",
        default="results/validation/har_domain_adaptation_eval.csv",
        help="Output summary CSV path",
    )
    parser.add_argument(
        "--out-md",
        default="results/validation/har_domain_adaptation_eval.md",
        help="Output summary Markdown path",
    )
    return parser.parse_args()


def _resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    return path


def _safe_std(values: pd.DataFrame) -> pd.Series:
    std = values.std(axis=0, ddof=0)
    return std.replace([np.inf, -np.inf], np.nan).fillna(0.0).mask(lambda s: s <= 1e-12, 1.0)


def _dataset_zscore(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Standardise source and target with their own unlabeled domain statistics."""
    train_mean = X_train.mean(axis=0)
    train_std = _safe_std(X_train)
    test_mean = X_test.mean(axis=0)
    test_std = _safe_std(X_test)

    return (
        (X_train - train_mean) / train_std,
        (X_test - test_mean) / test_std,
        {
            "uses_target_labels": False,
            "target_statistics": "feature means/stds computed on unlabeled target windows",
        },
    )


def _combined_zscore(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    combined = pd.concat([X_train, X_test], axis=0)
    mean = combined.mean(axis=0)
    std = _safe_std(combined)
    return (X_train - mean) / std, (X_test - mean) / std


def _subject_group_key(df: pd.DataFrame) -> pd.Series:
    dataset = (
        df["dataset_name"].astype(str)
        if "dataset_name" in df.columns
        else pd.Series(["UNKNOWN"] * len(df), index=df.index, dtype="string")
    )
    subject = (
        df["subject_id"].astype(str)
        if "subject_id" in df.columns
        else pd.Series(["UNKNOWN_SUBJECT"] * len(df), index=df.index, dtype="string")
    )
    return (dataset + "::" + subject).astype("string")


def _subject_zscore(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Standardise each subject independently, using no labels from the target domain."""

    def normalise_by_group(X: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=X.index, columns=X.columns, dtype=float)
        groups = _subject_group_key(df)
        for _, idx in groups.groupby(groups, sort=False).groups.items():
            group_idx = list(idx)
            group = X.loc[group_idx]
            out.loc[group_idx] = (group - group.mean(axis=0)) / _safe_std(group)
        return out.fillna(0.0)

    return (
        normalise_by_group(X_train, train_df),
        normalise_by_group(X_test, test_df),
        {
            "uses_target_labels": False,
            "target_statistics": "feature means/stds computed per unlabeled target subject",
        },
    )


def _matrix_sqrt_psd(matrix: np.ndarray, *, inverse: bool, eps: float) -> np.ndarray:
    eigvals, eigvecs = np.linalg.eigh(matrix)
    eigvals = np.clip(eigvals, eps, None)
    if inverse:
        diag = np.diag(1.0 / np.sqrt(eigvals))
    else:
        diag = np.diag(np.sqrt(eigvals))
    return eigvecs @ diag @ eigvecs.T


def _coral_align(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    *,
    eps: float = 1e-3,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Align source covariance to unlabeled target covariance using CORAL."""
    source = X_train.to_numpy(dtype=float)
    target = X_test.to_numpy(dtype=float)

    source_mean = source.mean(axis=0, keepdims=True)
    target_mean = target.mean(axis=0, keepdims=True)
    source_centered = source - source_mean
    target_centered = target - target_mean

    n_features = source.shape[1]
    source_cov = np.cov(source_centered, rowvar=False) + eps * np.eye(n_features)
    target_cov = np.cov(target_centered, rowvar=False) + eps * np.eye(n_features)

    transform = _matrix_sqrt_psd(source_cov, inverse=True, eps=eps) @ _matrix_sqrt_psd(
        target_cov,
        inverse=False,
        eps=eps,
    )
    aligned_source = source_centered @ transform + target_mean

    return (
        pd.DataFrame(aligned_source, index=X_train.index, columns=X_train.columns),
        X_test.copy(),
        {
            "uses_target_labels": False,
            "target_statistics": "feature mean/covariance computed on unlabeled target windows",
            "regularisation_eps": float(eps),
        },
    )


def _pca_basis(
    X: np.ndarray,
    *,
    n_components: int,
) -> np.ndarray:
    if n_components < 1:
        raise ValueError("n_components must be >= 1")
    _, _, vt = np.linalg.svd(X, full_matrices=False)
    return vt[:n_components].T


def _components_for_variance(
    X: np.ndarray,
    *,
    variance_threshold: float,
    max_components: int,
) -> int:
    _, singular_values, _ = np.linalg.svd(X, full_matrices=False)
    variances = singular_values**2
    total = float(variances.sum())
    if total <= 1e-12:
        return 1
    cumulative = np.cumsum(variances) / total
    requested = int(np.searchsorted(cumulative, variance_threshold, side="left") + 1)
    return max(1, min(requested, max_components, X.shape[1], X.shape[0]))


def _subspace_align(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    *,
    variance_threshold: float = 0.95,
    max_components: int = 50,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Align PCA source subspace to PCA target subspace using unlabeled target features."""
    train_z, test_z, _ = _dataset_zscore(X_train, X_test)
    source = train_z.to_numpy(dtype=float)
    target = test_z.to_numpy(dtype=float)

    source_components = _components_for_variance(
        source,
        variance_threshold=variance_threshold,
        max_components=max_components,
    )
    target_components = _components_for_variance(
        target,
        variance_threshold=variance_threshold,
        max_components=max_components,
    )
    n_components = max(source_components, target_components)
    n_components = min(n_components, source.shape[0], target.shape[0], source.shape[1], target.shape[1])

    source_basis = _pca_basis(source, n_components=n_components)
    target_basis = _pca_basis(target, n_components=n_components)
    alignment = source_basis.T @ target_basis

    aligned_source = source @ source_basis @ alignment
    projected_target = target @ target_basis
    columns = [f"sa_component_{idx:03d}" for idx in range(n_components)]

    return (
        pd.DataFrame(aligned_source, index=X_train.index, columns=columns),
        pd.DataFrame(projected_target, index=X_test.index, columns=columns),
        {
            "uses_target_labels": False,
            "target_statistics": "feature means/stds and PCA basis computed on unlabeled target windows",
            "source_components_for_variance": int(source_components),
            "target_components_for_variance": int(target_components),
            "components_used": int(n_components),
            "variance_threshold": float(variance_threshold),
            "max_components": int(max_components),
            "preprocessing": "source and target z-scored with their own unlabeled domain statistics before PCA",
        },
    )


def _domain_classifier_importance_weights(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    *,
    random_state: int,
    min_weight: float = 0.2,
    max_weight: float = 5.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, dict[str, Any]]:
    """Estimate source sample weights from a source-vs-target domain classifier."""
    train_z, test_z = _combined_zscore(X_train, X_test)
    domain_X = pd.concat([train_z, test_z], axis=0)
    domain_y = np.concatenate([np.zeros(len(train_z), dtype=int), np.ones(len(test_z), dtype=int)])

    domain_model = LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=random_state,
    )
    domain_model.fit(domain_X, domain_y)

    source_target_prob = domain_model.predict_proba(train_z)[:, 1]
    source_target_prob = np.clip(source_target_prob, 1e-6, 1.0 - 1e-6)
    raw_weights = source_target_prob / (1.0 - source_target_prob)
    clipped_weights = np.clip(raw_weights, min_weight, max_weight)
    normalised_weights = clipped_weights / float(np.mean(clipped_weights))
    weights = pd.Series(normalised_weights, index=X_train.index, dtype=float)

    domain_prob = domain_model.predict_proba(domain_X)[:, 1]
    effective_sample_size = float((weights.sum() ** 2) / np.square(weights.to_numpy()).sum())

    return (
        X_train.copy(),
        X_test.copy(),
        weights,
        {
            "uses_target_labels": False,
            "target_statistics": "unlabeled target features used to train a source-vs-target domain classifier",
            "weighting": "source sample_weight = clipped odds p(target|x) / p(source|x), normalised to mean 1",
            "domain_classifier": "LogisticRegression(class_weight='balanced') on combined z-scored features",
            "domain_classifier_auc_training": float(roc_auc_score(domain_y, domain_prob)),
            "min_weight": float(weights.min()),
            "max_weight": float(weights.max()),
            "mean_weight": float(weights.mean()),
            "std_weight": float(weights.std(ddof=0)),
            "effective_sample_size": effective_sample_size,
            "clip_range": [float(min_weight), float(max_weight)],
        },
    )


def _encode_labels(values: pd.Series | list[str] | np.ndarray, labels: list[str]) -> np.ndarray:
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    return np.asarray([label_to_idx[str(value)] for value in list(values)], dtype=np.int16)


def _macro_f1_encoded(y_true: np.ndarray, y_pred: np.ndarray, *, n_labels: int) -> float:
    scores: list[float] = []
    for label_idx in range(n_labels):
        true_is_label = y_true == label_idx
        pred_is_label = y_pred == label_idx
        tp = int(np.sum(true_is_label & pred_is_label))
        fp = int(np.sum(~true_is_label & pred_is_label))
        fn = int(np.sum(true_is_label & ~pred_is_label))
        denominator = (2 * tp) + fp + fn
        scores.append(0.0 if denominator == 0 else (2 * tp) / denominator)
    return float(np.mean(scores)) if scores else float("nan")


def _paired_bootstrap_delta_macro_f1(
    *,
    y_true: pd.Series,
    baseline_pred: pd.Series,
    candidate_pred: pd.Series,
    labels: list[str],
    n_resamples: int,
    confidence: float,
    random_state: int,
) -> dict[str, Any]:
    if not (len(y_true) == len(baseline_pred) == len(candidate_pred)):
        raise ValueError("y_true, baseline_pred, and candidate_pred must align")

    y_true_encoded = _encode_labels(y_true.astype(str).tolist(), labels)
    baseline_encoded = _encode_labels(baseline_pred.astype(str).tolist(), labels)
    candidate_encoded = _encode_labels(candidate_pred.astype(str).tolist(), labels)
    n = int(len(y_true_encoded))
    n_labels = int(len(labels))

    baseline_score = _macro_f1_encoded(y_true_encoded, baseline_encoded, n_labels=n_labels)
    candidate_score = _macro_f1_encoded(y_true_encoded, candidate_encoded, n_labels=n_labels)
    point = float(candidate_score - baseline_score)

    if n == 0 or int(n_resamples) <= 0:
        return {
            "metric": "paired_bootstrap_delta_macro_f1_vs_rf_baseline",
            "point": point,
            "lower": float("nan"),
            "upper": float("nan"),
            "confidence": float(confidence),
            "n_resamples": int(n_resamples),
            "n": n,
        }

    rng = np.random.default_rng(random_state)
    samples = np.empty(int(n_resamples), dtype=float)
    for sample_idx in range(int(n_resamples)):
        idx = rng.integers(0, n, size=n)
        sampled_true = y_true_encoded[idx]
        sampled_baseline = baseline_encoded[idx]
        sampled_candidate = candidate_encoded[idx]
        samples[sample_idx] = _macro_f1_encoded(
            sampled_true,
            sampled_candidate,
            n_labels=n_labels,
        ) - _macro_f1_encoded(
            sampled_true,
            sampled_baseline,
            n_labels=n_labels,
        )

    alpha = (1.0 - float(confidence)) / 2.0
    return {
        "metric": "paired_bootstrap_delta_macro_f1_vs_rf_baseline",
        "point": point,
        "lower": float(np.quantile(samples, alpha)),
        "upper": float(np.quantile(samples, 1.0 - alpha)),
        "confidence": float(confidence),
        "n_resamples": int(n_resamples),
        "n": n,
    }


def _prepare_matrices(
    *,
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    labels: list[str],
) -> dict[str, Any]:
    source_use = _filter_to_labels(source_df, labels)
    target_use = _filter_to_labels(target_df, labels)

    feature_cols = select_feature_columns(source_use)
    feature_cols = [col for col in feature_cols if col in target_use.columns]
    if not feature_cols:
        raise ValueError("No overlapping feature columns for adaptation evaluation")

    X_train = source_use[feature_cols].copy()
    X_test = target_use[feature_cols].copy()
    y_train = source_use["label_mapped_majority"].astype(str).copy()
    y_test = target_use["label_mapped_majority"].astype(str).copy()

    fill_values = X_train.median(numeric_only=True).to_dict()
    fill_values = {str(k): float(v) for k, v in fill_values.items()}
    X_train = X_train.fillna(fill_values).fillna(0.0)
    X_test = X_test.fillna(fill_values).fillna(0.0)

    return {
        "source_use": source_use,
        "target_use": target_use,
        "feature_cols": feature_cols,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "fill_values": fill_values,
    }


def _evaluate_rf(
    *,
    method: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    labels: list[str],
    random_state: int,
    transform_metadata: dict[str, Any],
    sample_weight: pd.Series | np.ndarray | None = None,
) -> dict[str, Any]:
    model = train_random_forest_classifier(
        X_train,
        y_train,
        random_state=random_state,
        sample_weight=sample_weight,
    )
    y_pred = pd.Series(model.predict(X_test), index=X_test.index, dtype="string")
    metrics = compute_classification_metrics(
        y_test.astype(str).tolist(),
        y_pred.astype(str).tolist(),
        labels=labels,
    )
    return {
        "method": method,
        "metrics": metrics,
        "transform": transform_metadata,
        "predictions": y_pred.astype(str).tolist(),
        "predictions_preview": y_pred.head(10).astype(str).tolist(),
    }


def _direction_eval(
    *,
    source_name: str,
    source_df: pd.DataFrame,
    target_name: str,
    target_df: pd.DataFrame,
    random_state: int,
    bootstrap_resamples: int,
    bootstrap_confidence: float,
) -> dict[str, Any]:
    labels = _ordered_shared_labels(source_df, target_df)
    matrices = _prepare_matrices(source_df=source_df, target_df=target_df, labels=labels)
    X_train = matrices["X_train"]
    X_test = matrices["X_test"]
    y_train = matrices["y_train"]
    y_test = matrices["y_test"]

    target_z_train, target_z_test, target_z_metadata = _dataset_zscore(X_train, X_test)
    subject_z_train, subject_z_test, subject_z_metadata = _subject_zscore(
        X_train,
        X_test,
        matrices["source_use"],
        matrices["target_use"],
    )
    coral_train, coral_test, coral_metadata = _coral_align(X_train, X_test)
    subspace_train, subspace_test, subspace_metadata = _subspace_align(X_train, X_test)
    weighted_X_train, weighted_X_test, source_weights, weighted_metadata = _domain_classifier_importance_weights(
        X_train,
        X_test,
        random_state=random_state,
    )

    methods: dict[str, dict[str, Any]] = {
        "rf_baseline": {
            "X_train": X_train.copy(),
            "X_test": X_test.copy(),
            "metadata": {"uses_target_labels": False, "target_statistics": "none"},
            "sample_weight": None,
        },
        "rf_target_zscore": {
            "X_train": target_z_train,
            "X_test": target_z_test,
            "metadata": target_z_metadata,
            "sample_weight": None,
        },
        "rf_subject_zscore": {
            "X_train": subject_z_train,
            "X_test": subject_z_test,
            "metadata": subject_z_metadata,
            "sample_weight": None,
        },
        "rf_coral": {
            "X_train": coral_train,
            "X_test": coral_test,
            "metadata": coral_metadata,
            "sample_weight": None,
        },
        "rf_subspace_align": {
            "X_train": subspace_train,
            "X_test": subspace_test,
            "metadata": subspace_metadata,
            "sample_weight": None,
        },
        "rf_importance_weighted": {
            "X_train": weighted_X_train,
            "X_test": weighted_X_test,
            "metadata": weighted_metadata,
            "sample_weight": source_weights,
        },
    }

    results: dict[str, Any] = {}
    baseline_macro_f1: float | None = None
    predictions_by_method: dict[str, pd.Series] = {}
    for method, spec in methods.items():
        result = _evaluate_rf(
            method=method,
            X_train=spec["X_train"],
            X_test=spec["X_test"],
            y_train=y_train,
            y_test=y_test,
            labels=labels,
            random_state=random_state,
            transform_metadata=spec["metadata"],
            sample_weight=spec["sample_weight"],
        )
        if method == "rf_baseline":
            baseline_macro_f1 = float(result["metrics"]["macro_f1"])
        if baseline_macro_f1 is not None:
            result["delta_macro_f1_vs_baseline"] = float(result["metrics"]["macro_f1"]) - baseline_macro_f1
        results[method] = result
        predictions_by_method[method] = pd.Series(result["predictions"], index=y_test.index, dtype="string")

    baseline_pred = predictions_by_method["rf_baseline"]
    for idx, (method, result) in enumerate(results.items()):
        if method == "rf_baseline":
            result["paired_delta_macro_f1_ci"] = {
                "metric": "paired_bootstrap_delta_macro_f1_vs_rf_baseline",
                "point": 0.0,
                "lower": 0.0,
                "upper": 0.0,
                "confidence": float(bootstrap_confidence),
                "n_resamples": int(bootstrap_resamples),
                "n": int(len(y_test)),
            }
            continue
        result["paired_delta_macro_f1_ci"] = _paired_bootstrap_delta_macro_f1(
            y_true=y_test,
            baseline_pred=baseline_pred,
            candidate_pred=predictions_by_method[method],
            labels=labels,
            n_resamples=int(bootstrap_resamples),
            confidence=float(bootstrap_confidence),
            random_state=int(random_state) + 1009 + idx,
        )

    return {
        "source_dataset": source_name,
        "target_dataset": target_name,
        "shared_labels_used": labels,
        "train_rows": int(len(matrices["source_use"])),
        "test_rows": int(len(matrices["target_use"])),
        "feature_columns_count": int(len(matrices["feature_cols"])),
        "train_label_counts": matrices["source_use"]["label_mapped_majority"].astype(str).value_counts().to_dict(),
        "test_label_counts": matrices["target_use"]["label_mapped_majority"].astype(str).value_counts().to_dict(),
        "target_labels_sequence": y_test.astype(str).tolist(),
        "methods": results,
    }


def _summary_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for direction_key, direction in payload["directions"].items():
        for method, result in direction["methods"].items():
            metrics = result["metrics"]
            ci = result.get(
                "paired_delta_macro_f1_ci",
                {
                    "lower": 0.0,
                    "upper": 0.0,
                    "confidence": 0.95,
                },
            )
            rows.append(
                {
                    "direction": direction_key,
                    "source": direction["source_dataset"],
                    "target": direction["target_dataset"],
                    "method": method,
                    "macro_f1": float(metrics["macro_f1"]),
                    "accuracy": float(metrics["accuracy"]),
                    "delta_macro_f1_vs_baseline": float(result.get("delta_macro_f1_vs_baseline", 0.0)),
                    "delta_macro_f1_ci_lower": float(ci["lower"]),
                    "delta_macro_f1_ci_upper": float(ci["upper"]),
                    "delta_macro_f1_ci_confidence": float(ci["confidence"]),
                    "support_total": int(metrics["support_total"]),
                    "uses_target_labels": bool(result["transform"].get("uses_target_labels", False)),
                }
            )
    return rows


def _write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    lines = [
        "# HAR Domain Adaptation Evaluation",
        "",
        "| Direction | Method | Macro F1 | Δ vs RF | Paired 95% CI for Δ | Accuracy | Uses target labels |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in df.to_dict(orient="records"):
        lines.append(
            "| {direction} | `{method}` | {macro_f1:.3f} | {delta:+.3f} | [{lower:+.3f}, {upper:+.3f}] | {accuracy:.3f} | {labels} |".format(
                direction=row["direction"],
                method=row["method"],
                macro_f1=row["macro_f1"],
                delta=row["delta_macro_f1_vs_baseline"],
                lower=row["delta_macro_f1_ci_lower"],
                upper=row["delta_macro_f1_ci_upper"],
                accuracy=row["accuracy"],
                labels="yes" if row["uses_target_labels"] else "no",
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    ucihar_path = _resolve_path(args.ucihar_path)
    pamap2_path = _resolve_path(args.pamap2_path)
    out_json = _resolve_path(args.out_json)
    out_csv = _resolve_path(args.out_csv)
    out_md = _resolve_path(args.out_md)

    for dataset_name, path in [("UCIHAR", ucihar_path), ("PAMAP2", pamap2_path)]:
        if not path.exists():
            print(f"ERROR: dataset path not found for {dataset_name}: {path}")
            return 1

    print("Preparing UCIHAR feature table...")
    ucihar_df, ucihar_summary = _prepare_feature_table(
        dataset_key="uci_har",
        path=ucihar_path,
        sample_limit=int(args.ucihar_sample_limit),
        pamap2_include_optional=False,
        target_rate=float(args.target_rate),
        window_size=int(args.window_size),
        step_size=int(args.step_size),
        keep_unacceptable=bool(args.keep_unacceptable),
    )

    print("Preparing PAMAP2 feature table...")
    pamap2_df, pamap2_summary = _prepare_feature_table(
        dataset_key="pamap2",
        path=pamap2_path,
        sample_limit=int(args.pamap2_sample_limit),
        pamap2_include_optional=bool(args.pamap2_include_optional),
        target_rate=float(args.target_rate),
        window_size=int(args.window_size),
        step_size=int(args.step_size),
        keep_unacceptable=bool(args.keep_unacceptable),
    )

    feature_tables = {
        "UCIHAR": ucihar_df,
        "PAMAP2": pamap2_df,
    }

    directions = {
        "UCIHAR_to_PAMAP2": _direction_eval(
            source_name="UCIHAR",
            source_df=feature_tables["UCIHAR"],
            target_name="PAMAP2",
            target_df=feature_tables["PAMAP2"],
            random_state=int(args.random_state),
            bootstrap_resamples=int(args.bootstrap_resamples),
            bootstrap_confidence=float(args.bootstrap_confidence),
        ),
        "PAMAP2_to_UCIHAR": _direction_eval(
            source_name="PAMAP2",
            source_df=feature_tables["PAMAP2"],
            target_name="UCIHAR",
            target_df=feature_tables["UCIHAR"],
            random_state=int(args.random_state),
            bootstrap_resamples=int(args.bootstrap_resamples),
            bootstrap_confidence=float(args.bootstrap_confidence),
        ),
    }

    payload = {
        "evaluation_name": "har_domain_adaptation_eval",
        "methodological_framing": (
            "DAGHAR-inspired lightweight domain-adaptation mitigation for the measured "
            "UCIHAR/PAMAP2 HAR generalisation cliff. Target-domain statistics are computed "
            "without target labels."
        ),
        "preprocessing": {
            "target_rate_hz": float(args.target_rate),
            "window_size": int(args.window_size),
            "step_size": int(args.step_size),
            "keep_unacceptable": bool(args.keep_unacceptable),
            "random_state": int(args.random_state),
            "bootstrap_resamples": int(args.bootstrap_resamples),
            "bootstrap_confidence": float(args.bootstrap_confidence),
        },
        "datasets": {
            "UCIHAR": ucihar_summary,
            "PAMAP2": pamap2_summary,
        },
        "directions": directions,
        "methods": {
            "rf_baseline": "Random forest on source features, evaluated directly on target features.",
            "rf_target_zscore": "Source and target feature columns z-scored with their own unlabeled domain statistics.",
            "rf_subject_zscore": "Feature columns z-scored within each subject group, including unlabeled target subjects.",
            "rf_coral": "CORAL covariance alignment: source features are aligned to unlabeled target mean/covariance before RF training.",
            "rf_subspace_align": "Subspace Alignment: source and target are z-scored separately, PCA subspaces are fitted with unlabeled target features, and the source subspace is aligned to the target subspace before RF training.",
            "rf_importance_weighted": "Domain-classifier importance weighting: train a source-vs-target classifier on unlabeled features, then up-weight source rows that look target-like before RF training.",
        },
        "notes": [
            "No target-domain labels are used by the adaptation transforms.",
            "Delta intervals are paired non-parametric bootstrap CIs over target windows, resampling the RF baseline and adapted-method predictions together.",
            "All methods use the same shared-label target rows as the Chapter 4 LODO evaluation.",
            "This is a lightweight mitigation experiment, not a full DAGHAR benchmark reproduction.",
        ],
    }

    rows = _summary_rows(payload)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    _write_markdown(out_md, rows)

    print()
    print(f"Saved HAR domain adaptation JSON to: {out_json}")
    print(f"Saved HAR domain adaptation CSV to: {out_csv}")
    print(f"Saved HAR domain adaptation Markdown to: {out_md}")
    for row in rows:
        print(
            "{direction} {method}: macro_f1={macro_f1:.4f} delta={delta:+.4f} ci=[{lower:+.4f}, {upper:+.4f}]".format(
                direction=row["direction"],
                method=row["method"],
                macro_f1=row["macro_f1"],
                delta=row["delta_macro_f1_vs_baseline"],
                lower=row["delta_macro_f1_ci_lower"],
                upper=row["delta_macro_f1_ci_upper"],
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
