#!/usr/bin/env python3
"""Build the v2 and v3 fall detector artifacts from the already-trained
per-kind artifacts (HGB + XGB + RF).

Both v2 and v3 use identical subject-aware splits + seeds + feature extraction
as `scripts/run_fall_artifact_train.py` — the only difference is the
inference-time architecture:

    v2: Calibrated best-of-three + F2 threshold
        - pick the held-out-F1 winner among {hgb, xgb, rf}
        - isotonic-calibrate its probabilities on the held-out val fold
          (CalibratedClassifierCV with cv="prefit")
        - re-tune the decision threshold on the val fold for F2 (beta=2.0)

    v3: Soft-voting ensemble of all three + F2 threshold
        - average class-1 probabilities across HGB + XGB + RF
        - re-tune a single F2 threshold on the val fold

Outputs:
    artifacts/fall_v2/model.joblib
    artifacts/fall_v3/model.joblib
    results/validation/fall_artifact_eval_v2.json (+ predictions.csv)
    results/validation/fall_artifact_eval_v3.json (+ predictions.csv)
    results/validation/fall_artifact_eval_v2_v3_comparison.json
    artifacts/fall_detector.joblib  (copy of v2 or v3 whichever has higher held-out F2)
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import fbeta_score

try:
    from sklearn.frozen import FrozenEstimator
except Exception:  # pragma: no cover - older sklearn fallback
    FrozenEstimator = None

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Reuse everything from the training script.
from scripts.run_fall_artifact_train import (  # noqa: E402
    DEFAULT_INNER_VAL_SIZE,
    DEFAULT_OUTER_TEST_SIZE,
    DEFAULT_RANDOM_STATE,
    DEFAULT_STEP_SIZE,
    DEFAULT_TARGET_RATE_HZ,
    DEFAULT_THRESHOLD_GRID,
    DEFAULT_WINDOW_SIZE,
    FEATURE_COLUMNS,
    NEGATIVE_LABEL,
    POSITIVE_LABEL,
    _binarise,
    _binary_metrics_with_probs,
    _json_safe,
    _library_versions,
    _load_and_feature_extract,
    _positive_proba,
    _prepare_feature_matrix,
    _resolve_path,
    _split_subject_aware,
)
from models.fall.ensemble import SoftVotingFallEnsemble  # noqa: E402


POSITIVE_INDEX = 1
DEFAULT_BETA = 2.0
KIND_ORDER = ("hgb", "xgb", "rf")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--mobifall-path", default="data/raw/MOBIACT_Dataset/MobiFall_Dataset_v2.0")
    p.add_argument("--sisfall-path", default="data/raw/SISFALL_Dataset/SisFall_dataset")
    p.add_argument("--target-rate", type=float, default=DEFAULT_TARGET_RATE_HZ)
    p.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE)
    p.add_argument("--step-size", type=int, default=DEFAULT_STEP_SIZE)
    p.add_argument("--outer-test-size", type=float, default=DEFAULT_OUTER_TEST_SIZE)
    p.add_argument("--inner-val-size", type=float, default=DEFAULT_INNER_VAL_SIZE)
    p.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    p.add_argument("--artifact-dir", default="artifacts")
    p.add_argument("--report-dir", default="results/validation")
    p.add_argument(
        "--runtime-artifact-out",
        default="artifacts/fall_detector.joblib",
        help="Where to copy the v2/v3 winner for runtime use.",
    )
    p.add_argument("--beta", type=float, default=DEFAULT_BETA, help="F-beta for threshold tuning")
    return p.parse_args()


@dataclass
class KindArtifact:
    kind: str
    model: Any
    trained_threshold: float


def _load_kind_artifact(kind: str, artifact_dir: Path) -> KindArtifact:
    path = artifact_dir / f"fall_detector_{kind}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Missing trained artifact for {kind}: {path}")
    blob = joblib.load(path)
    return KindArtifact(
        kind=kind,
        model=blob["model"],
        trained_threshold=float(blob.get("probability_threshold", 0.5)),
    )


def _tune_fbeta_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    grid: list[float],
    beta: float,
) -> tuple[float, list[dict[str, float]]]:
    rows: list[dict[str, float]] = []
    best_t = 0.5
    best_score = -1.0
    for t in grid:
        y_pred = (y_prob >= t).astype(int)
        score = float(fbeta_score(y_true, y_pred, beta=beta, zero_division=0))
        rows.append({"threshold": float(t), f"f{beta:g}": float(score)})
        if score > best_score:
            best_score = score
            best_t = float(t)
    return best_t, rows


def _evaluate_holdout_from_probs(
    *,
    test_df: pd.DataFrame,
    y_prob: np.ndarray,
    threshold: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    y_true = _binarise(test_df["true_label"])
    combined = _binary_metrics_with_probs(y_true, y_prob, threshold=threshold)

    per_dataset: dict[str, dict[str, Any]] = {}
    for ds_name, sub in test_df.groupby("dataset_name", dropna=False):
        idx = sub.index.to_numpy()
        sub_y = y_true[idx]
        sub_prob = y_prob[idx]
        per_dataset[str(ds_name)] = _binary_metrics_with_probs(
            sub_y, sub_prob, threshold=threshold
        )

    pred_df = test_df[
        [
            c
            for c in [
                "dataset_name",
                "subject_id",
                "session_id",
                "window_id",
                "start_ts",
                "end_ts",
                "true_label",
            ]
            if c in test_df.columns
        ]
    ].copy()
    pred_df["predicted_probability"] = y_prob
    pred_df["predicted_is_fall"] = (y_prob >= threshold)
    pred_df["predicted_label"] = np.where(
        pred_df["predicted_is_fall"], POSITIVE_LABEL, NEGATIVE_LABEL
    )
    pred_df["probability_threshold_used"] = float(threshold)

    return {"combined": combined, "per_dataset": per_dataset}, pred_df


def _save_v2_v3_artifact(
    *,
    model_payload: Any,
    artifact_kind_label: str,
    used_features: list[str],
    probability_threshold: float,
    out_path: Path,
    metadata: dict[str, Any],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model_type": type(model_payload).__name__,
        "task_type": "fall",
        "model": model_payload,
        "used_features": list(used_features),
        "positive_label": POSITIVE_LABEL,
        "negative_label": NEGATIVE_LABEL,
        "probability_threshold": float(probability_threshold),
        "metadata": {**metadata, "artifact_kind": artifact_kind_label},
    }
    joblib.dump(artifact, out_path)


def main() -> int:
    args = parse_args()
    beta = float(args.beta)

    mobifall_path = _resolve_path(args.mobifall_path)
    sisfall_path = _resolve_path(args.sisfall_path)
    artifact_dir = _resolve_path(args.artifact_dir)
    report_dir = _resolve_path(args.report_dir)
    runtime_artifact_out = _resolve_path(args.runtime_artifact_out)

    if not mobifall_path.exists():
        raise FileNotFoundError(f"MobiFall path not found: {mobifall_path}")
    if not sisfall_path.exists():
        raise FileNotFoundError(f"SisFall path not found: {sisfall_path}")

    print("Loading per-kind trained artifacts (HGB + XGB + RF)...")
    kind_artifacts: dict[str, KindArtifact] = {
        k: _load_kind_artifact(k, artifact_dir) for k in KIND_ORDER
    }

    print("Rebuilding feature tables (same config as training)...")
    mobi_df, mobi_summary = _load_and_feature_extract(
        "mobifall",
        mobifall_path,
        target_rate=args.target_rate,
        window_size=args.window_size,
        step_size=args.step_size,
    )
    sis_df, sis_summary = _load_and_feature_extract(
        "sisfall",
        sisfall_path,
        target_rate=args.target_rate,
        window_size=args.window_size,
        step_size=args.step_size,
    )
    combined_df = pd.concat([mobi_df, sis_df], ignore_index=True)
    print(
        f"Combined windows={len(combined_df)} "
        f"label_counts={combined_df['true_label'].value_counts().to_dict()}"
    )

    train_outer_df, test_df, outer_split = _split_subject_aware(
        combined_df,
        test_size=args.outer_test_size,
        random_state=args.random_state,
    )
    train_inner_df, val_df, inner_split = _split_subject_aware(
        train_outer_df,
        test_size=args.inner_val_size,
        random_state=args.random_state,
    )
    print(f"Outer train={len(train_outer_df)} test={len(test_df)}")
    print(f"Inner train={len(train_inner_df)} val={len(val_df)}")

    X_val = _prepare_feature_matrix(val_df, FEATURE_COLUMNS)
    y_val = _binarise(val_df["true_label"])
    X_test = _prepare_feature_matrix(test_df, FEATURE_COLUMNS)
    y_test = _binarise(test_df["true_label"])

    # --- compute per-kind positive probabilities on val and test.
    val_probs: dict[str, np.ndarray] = {
        k: _positive_proba(ka.model, X_val) for k, ka in kind_artifacts.items()
    }
    test_probs: dict[str, np.ndarray] = {
        k: _positive_proba(ka.model, X_test) for k, ka in kind_artifacts.items()
    }

    # Raw per-kind metrics at each kind's trained threshold. Validation metrics
    # are used for model selection; held-out metrics are for reporting only.
    per_kind_test_f1: dict[str, float] = {}
    per_kind_val_fbeta: dict[str, float] = {}
    for kind, ka in kind_artifacts.items():
        val_pred = (val_probs[kind] >= ka.trained_threshold).astype(int)
        per_kind_val_fbeta[kind] = float(
            fbeta_score(y_val, val_pred, beta=beta, zero_division=0)
        )
        m = _binary_metrics_with_probs(
            y_test, test_probs[kind], threshold=ka.trained_threshold
        )
        per_kind_test_f1[kind] = float(m["f1"])
        print(
            f"  [{kind}] at trained-threshold={ka.trained_threshold:.2f}  "
            f"test_f1={m['f1']:.4f}  test_auc={m['roc_auc']:.4f}  "
            f"sens={m['sensitivity']:.4f}  spec={m['specificity']:.4f}"
        )

    # ========================================================================
    # V2: calibrate best-of-three + tune F-beta threshold on val
    # ========================================================================
    print("\n=== Building v2: calibrated best-single + F{:.0f} threshold ===".format(beta))
    winner_kind = max(per_kind_val_fbeta.keys(), key=lambda k: per_kind_val_fbeta[k])
    print(
        f"  winner-by-validation-F{beta:g}: [{winner_kind}]  "
        f"(val_f{beta:g}={per_kind_val_fbeta[winner_kind]:.4f})"
    )

    # Compatibility note: sklearn >=1.8 removed cv="prefit". FrozenEstimator is
    # the supported replacement for calibrating an already-fitted estimator.
    base_model = kind_artifacts[winner_kind].model
    print(f"  Fitting isotonic calibration on val_rows={len(val_df)}...")
    if FrozenEstimator is not None:
        v2_calibrated = CalibratedClassifierCV(
            estimator=FrozenEstimator(base_model),
            method="isotonic",
        )
    else:
        v2_calibrated = CalibratedClassifierCV(
            estimator=base_model,
            method="isotonic",
            cv="prefit",
        )
    v2_calibrated.fit(X_val, y_val)
    v2_val_prob = _positive_proba(v2_calibrated, X_val)
    v2_test_prob = _positive_proba(v2_calibrated, X_test)

    v2_threshold, v2_tuning = _tune_fbeta_threshold(
        y_val, v2_val_prob, grid=DEFAULT_THRESHOLD_GRID, beta=beta
    )
    v2_val_metrics = _binary_metrics_with_probs(y_val, v2_val_prob, threshold=v2_threshold)
    v2_heldout, v2_predictions = _evaluate_holdout_from_probs(
        test_df=test_df, y_prob=v2_test_prob, threshold=v2_threshold
    )
    print(
        f"  v2 val_f1={v2_val_metrics['f1']:.4f}  threshold={v2_threshold:.2f}  "
        f"heldout_f1={v2_heldout['combined']['f1']:.4f}  "
        f"auc={v2_heldout['combined']['roc_auc']:.4f}  "
        f"sens={v2_heldout['combined']['sensitivity']:.4f}  "
        f"spec={v2_heldout['combined']['specificity']:.4f}"
    )
    for ds, m in v2_heldout["per_dataset"].items():
        print(f"  v2 {ds}: f1={m['f1']:.4f} auc={m['roc_auc']:.4f}")

    # ========================================================================
    # V3: soft-vote ensemble of all three + tune F-beta threshold on val
    # ========================================================================
    print("\n=== Building v3: soft-vote ensemble (HGB+XGB+RF) + F{:.0f} threshold ===".format(beta))
    v3_val_prob = np.mean(np.stack([val_probs[k] for k in KIND_ORDER], axis=0), axis=0)
    v3_test_prob = np.mean(np.stack([test_probs[k] for k in KIND_ORDER], axis=0), axis=0)

    v3_threshold, v3_tuning = _tune_fbeta_threshold(
        y_val, v3_val_prob, grid=DEFAULT_THRESHOLD_GRID, beta=beta
    )
    v3_val_metrics = _binary_metrics_with_probs(y_val, v3_val_prob, threshold=v3_threshold)
    v3_heldout, v3_predictions = _evaluate_holdout_from_probs(
        test_df=test_df, y_prob=v3_test_prob, threshold=v3_threshold
    )
    print(
        f"  v3 val_f1={v3_val_metrics['f1']:.4f}  threshold={v3_threshold:.2f}  "
        f"heldout_f1={v3_heldout['combined']['f1']:.4f}  "
        f"auc={v3_heldout['combined']['roc_auc']:.4f}  "
        f"sens={v3_heldout['combined']['sensitivity']:.4f}  "
        f"spec={v3_heldout['combined']['specificity']:.4f}"
    )
    for ds, m in v3_heldout["per_dataset"].items():
        print(f"  v3 {ds}: f1={m['f1']:.4f} auc={m['roc_auc']:.4f}")

    # ========================================================================
    # Persist
    # ========================================================================
    created_utc = datetime.now(timezone.utc).isoformat()
    common_metadata = {
        "created_utc": created_utc,
        "train_source_composition": {
            "mobifall": mobi_summary,
            "sisfall": sis_summary,
            "combined_rows": int(len(combined_df)),
        },
        "feature_columns": list(FEATURE_COLUMNS),
        "library_versions": _library_versions(),
        "training_config": {
            "target_rate_hz": float(args.target_rate),
            "window_size": int(args.window_size),
            "step_size": int(args.step_size),
            "outer_test_size": float(args.outer_test_size),
            "inner_val_size": float(args.inner_val_size),
            "random_state": int(args.random_state),
            "threshold_beta": beta,
        },
    }

    v2_artifact_path = artifact_dir / "fall_v2" / "model.joblib"
    v3_artifact_path = artifact_dir / "fall_v3" / "model.joblib"

    v2_metadata = {
        **common_metadata,
        "artifact_version": "fall_v2_calibrated_best",
        "process": "calibrated_best_of_three + F{:g}_threshold".format(beta),
        "winner_kind": winner_kind,
        "calibration": {"method": "isotonic", "cv": "prefit_on_val_fold"},
        "status": "experimental",
        "methodology_note": (
            "Model kind and runtime winner are selected by validation F-beta, "
            "not held-out metrics. Calibration still uses the saved candidate "
            "estimator and should be revisited before final dissertation claims."
        ),
        "threshold_tuning": {
            "beta": beta,
            "selected_threshold": v2_threshold,
            "grid_top10": sorted(v2_tuning, key=lambda r: r[f"f{beta:g}"], reverse=True)[:10],
            "val_metrics": v2_val_metrics,
        },
    }
    v3_metadata = {
        **common_metadata,
        "artifact_version": "fall_v3_soft_vote_ensemble",
        "process": "soft_vote_ensemble_hgb_xgb_rf + F{:g}_threshold".format(beta),
        "base_model_kinds": list(KIND_ORDER),
        "status": "experimental",
        "threshold_tuning": {
            "beta": beta,
            "selected_threshold": v3_threshold,
            "grid_top10": sorted(v3_tuning, key=lambda r: r[f"f{beta:g}"], reverse=True)[:10],
            "val_metrics": v3_val_metrics,
        },
    }

    _save_v2_v3_artifact(
        model_payload=v2_calibrated,
        artifact_kind_label="fall_v2_calibrated_best",
        used_features=FEATURE_COLUMNS,
        probability_threshold=v2_threshold,
        out_path=v2_artifact_path,
        metadata=v2_metadata,
    )
    print(f"Saved v2 artifact -> {v2_artifact_path}")

    ensemble = SoftVotingFallEnsemble({k: kind_artifacts[k].model for k in KIND_ORDER})
    _save_v2_v3_artifact(
        model_payload=ensemble,
        artifact_kind_label="fall_v3_soft_vote_ensemble",
        used_features=FEATURE_COLUMNS,
        probability_threshold=v3_threshold,
        out_path=v3_artifact_path,
        metadata=v3_metadata,
    )
    print(f"Saved v3 artifact -> {v3_artifact_path}")

    # Per-artifact reports.
    v2_report = {
        "evaluation_name": "fall_artifact_eval_v2",
        "created_utc": created_utc,
        "process": v2_metadata["process"],
        "winner_kind": winner_kind,
        "per_kind_test_f1": per_kind_test_f1,
        "outer_split": outer_split,
        "inner_split": inner_split,
        "threshold_tuning": v2_metadata["threshold_tuning"],
        "held_out_metrics": v2_heldout,
        "artifact": {
            "path": str(v2_artifact_path),
            "probability_threshold": v2_threshold,
            "used_features": list(FEATURE_COLUMNS),
            "library_versions": common_metadata["library_versions"],
        },
    }
    v3_report = {
        "evaluation_name": "fall_artifact_eval_v3",
        "created_utc": created_utc,
        "process": v3_metadata["process"],
        "base_model_kinds": list(KIND_ORDER),
        "per_kind_test_f1": per_kind_test_f1,
        "outer_split": outer_split,
        "inner_split": inner_split,
        "threshold_tuning": v3_metadata["threshold_tuning"],
        "held_out_metrics": v3_heldout,
        "artifact": {
            "path": str(v3_artifact_path),
            "probability_threshold": v3_threshold,
            "used_features": list(FEATURE_COLUMNS),
            "library_versions": common_metadata["library_versions"],
        },
    }

    v2_report_path = report_dir / "fall_artifact_eval_v2.json"
    v3_report_path = report_dir / "fall_artifact_eval_v3.json"
    v2_pred_path = report_dir / "fall_artifact_eval_v2_predictions.csv"
    v3_pred_path = report_dir / "fall_artifact_eval_v3_predictions.csv"

    report_dir.mkdir(parents=True, exist_ok=True)
    v2_report_path.write_text(json.dumps(_json_safe(v2_report), indent=2), encoding="utf-8")
    v3_report_path.write_text(json.dumps(_json_safe(v3_report), indent=2), encoding="utf-8")
    v2_predictions.to_csv(v2_pred_path, index=False)
    v3_predictions.to_csv(v3_pred_path, index=False)
    print(f"Saved v2 report -> {v2_report_path}")
    print(f"Saved v3 report -> {v3_report_path}")

    # Decide runtime winner by validation F-beta. Held-out F-beta is reported
    # below, but must not select the artifact.
    v2_val_fbeta = float(
        fbeta_score(
            y_val,
            (v2_val_prob >= v2_threshold).astype(int),
            beta=beta,
            zero_division=0,
        )
    )
    v3_val_fbeta = float(
        fbeta_score(
            y_val,
            (v3_val_prob >= v3_threshold).astype(int),
            beta=beta,
            zero_division=0,
        )
    )
    v2_heldout_fbeta = float(
        fbeta_score(
            y_test,
            (v2_test_prob >= v2_threshold).astype(int),
            beta=beta,
            zero_division=0,
        )
    )
    v3_heldout_fbeta = float(
        fbeta_score(
            y_test,
            (v3_test_prob >= v3_threshold).astype(int),
            beta=beta,
            zero_division=0,
        )
    )
    runtime_winner = "v2" if v2_val_fbeta >= v3_val_fbeta else "v3"
    runtime_winner_path = v2_artifact_path if runtime_winner == "v2" else v3_artifact_path
    runtime_artifact_out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(runtime_winner_path, runtime_artifact_out)
    print(
        f"\nRuntime winner (by validation F{beta:g}): {runtime_winner}  "
        f"({runtime_winner_path.name}) -> {runtime_artifact_out}"
    )

    comparison = {
        "evaluation_name": "fall_artifact_v2_v3_comparison",
        "created_utc": created_utc,
        "config": {
            **common_metadata["training_config"],
            "mobifall_path": str(mobifall_path),
            "sisfall_path": str(sisfall_path),
        },
        "per_kind_test_f1_before_v2_v3": per_kind_test_f1,
        "per_kind_validation_fbeta_before_v2_v3": per_kind_val_fbeta,
        "v2": {
            "process": v2_metadata["process"],
            "winner_kind": winner_kind,
            "selected_threshold": v2_threshold,
            "val_metrics": v2_val_metrics,
            "held_out_metrics": v2_heldout,
            "validation_fbeta": v2_val_fbeta,
            "held_out_fbeta": v2_heldout_fbeta,
            "artifact_path": str(v2_artifact_path),
            "report_path": str(v2_report_path),
        },
        "v3": {
            "process": v3_metadata["process"],
            "base_model_kinds": list(KIND_ORDER),
            "selected_threshold": v3_threshold,
            "val_metrics": v3_val_metrics,
            "held_out_metrics": v3_heldout,
            "validation_fbeta": v3_val_fbeta,
            "held_out_fbeta": v3_heldout_fbeta,
            "artifact_path": str(v3_artifact_path),
            "report_path": str(v3_report_path),
        },
        "runtime_winner": runtime_winner,
        "runtime_artifact": str(runtime_artifact_out),
    }
    comparison_path = report_dir / "fall_artifact_eval_v2_v3_comparison.json"
    comparison_path.write_text(json.dumps(_json_safe(comparison), indent=2), encoding="utf-8")
    print(f"Saved comparison -> {comparison_path}")

    print("\n=== Summary (held-out) ===")
    print(f"{'variant':<8} {'process':<45} {'f1':>7} {'fbeta':>7} {'auc':>7} {'sens':>7} {'spec':>7}")
    print(
        f"{'v2':<8} {v2_metadata['process']:<45} "
        f"{v2_heldout['combined']['f1']:>7.4f} {v2_heldout_fbeta:>7.4f} "
        f"{v2_heldout['combined']['roc_auc']:>7.4f} "
        f"{v2_heldout['combined']['sensitivity']:>7.4f} "
        f"{v2_heldout['combined']['specificity']:>7.4f}"
    )
    print(
        f"{'v3':<8} {v3_metadata['process']:<45} "
        f"{v3_heldout['combined']['f1']:>7.4f} {v3_heldout_fbeta:>7.4f} "
        f"{v3_heldout['combined']['roc_auc']:>7.4f} "
        f"{v3_heldout['combined']['sensitivity']:>7.4f} "
        f"{v3_heldout['combined']['specificity']:>7.4f}"
    )
    print(f"runtime default -> {runtime_winner}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
