import numpy as np
import pandas as pd
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_har_domain_adaptation_eval import (
    _coral_align,
    _dataset_zscore,
    _domain_classifier_importance_weights,
    _paired_bootstrap_delta_macro_f1,
    _subspace_align,
    _subject_zscore,
    _summary_rows,
)


def test_dataset_zscore_uses_unlabelled_domain_statistics():
    X_train = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [10.0, 12.0, 14.0]})
    X_test = pd.DataFrame({"a": [100.0, 120.0, 140.0], "b": [2.0, 2.0, 2.0]})

    train_z, test_z, metadata = _dataset_zscore(X_train, X_test)

    assert metadata["uses_target_labels"] is False
    assert np.allclose(train_z.mean(axis=0), [0.0, 0.0])
    assert np.allclose(test_z["a"].mean(), 0.0)
    assert np.allclose(test_z["b"], [0.0, 0.0, 0.0])


def test_subject_zscore_normalises_each_subject_group_without_labels():
    X = pd.DataFrame(
        {
            "a": [1.0, 3.0, 10.0, 14.0],
            "b": [2.0, 4.0, 20.0, 28.0],
        },
        index=[10, 11, 12, 13],
    )
    df = pd.DataFrame(
        {
            "dataset_name": ["D", "D", "D", "D"],
            "subject_id": ["s1", "s1", "s2", "s2"],
        },
        index=X.index,
    )

    train_z, test_z, metadata = _subject_zscore(X, X, df, df)

    assert metadata["uses_target_labels"] is False
    assert np.allclose(train_z.loc[[10, 11]].mean(axis=0), [0.0, 0.0])
    assert np.allclose(train_z.loc[[12, 13]].mean(axis=0), [0.0, 0.0])
    assert np.allclose(test_z.loc[[10, 11]].mean(axis=0), [0.0, 0.0])


def test_coral_align_returns_finite_source_aligned_to_target_mean():
    X_train = pd.DataFrame(
        {
            "a": [0.0, 1.0, 2.0, 3.0],
            "b": [1.0, 1.5, 3.0, 4.5],
            "c": [10.0, 11.0, 13.0, 16.0],
        }
    )
    X_test = pd.DataFrame(
        {
            "a": [100.0, 101.0, 103.0, 106.0],
            "b": [20.0, 22.0, 23.0, 25.0],
            "c": [5.0, 8.0, 13.0, 21.0],
        }
    )

    aligned_train, aligned_test, metadata = _coral_align(X_train, X_test)

    assert metadata["uses_target_labels"] is False
    assert aligned_train.shape == X_train.shape
    assert aligned_test.equals(X_test)
    assert np.isfinite(aligned_train.to_numpy()).all()
    assert np.allclose(aligned_train.mean(axis=0), X_test.mean(axis=0))


def test_subspace_align_returns_matching_finite_components_without_labels():
    X_train = pd.DataFrame(
        {
            "a": [0.0, 1.0, 2.0, 3.0, 4.0],
            "b": [1.0, 1.2, 1.5, 1.9, 2.4],
            "c": [10.0, 11.0, 13.0, 16.0, 20.0],
            "d": [5.0, 4.0, 3.0, 2.0, 1.0],
        }
    )
    X_test = pd.DataFrame(
        {
            "a": [100.0, 101.0, 103.0, 106.0, 110.0],
            "b": [20.0, 22.0, 23.0, 25.0, 28.0],
            "c": [5.0, 8.0, 13.0, 21.0, 34.0],
            "d": [50.0, 47.0, 43.0, 38.0, 32.0],
        }
    )

    aligned_train, projected_test, metadata = _subspace_align(
        X_train,
        X_test,
        variance_threshold=0.90,
        max_components=3,
    )

    assert metadata["uses_target_labels"] is False
    assert aligned_train.shape == projected_test.shape
    assert aligned_train.shape[1] == metadata["components_used"]
    assert 1 <= metadata["components_used"] <= 3
    assert np.isfinite(aligned_train.to_numpy()).all()
    assert np.isfinite(projected_test.to_numpy()).all()
    assert aligned_train.columns.tolist() == projected_test.columns.tolist()


def test_domain_classifier_importance_weights_are_positive_and_normalised():
    X_train = pd.DataFrame(
        {
            "a": [0.0, 0.5, 1.0, 1.5, 2.0, 2.5],
            "b": [1.0, 1.1, 1.3, 1.4, 1.6, 1.7],
        }
    )
    X_test = pd.DataFrame(
        {
            "a": [1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
            "b": [1.4, 1.6, 1.7, 1.9, 2.0, 2.2],
        }
    )

    weighted_train, weighted_test, weights, metadata = _domain_classifier_importance_weights(
        X_train,
        X_test,
        random_state=42,
        min_weight=0.5,
        max_weight=2.0,
    )

    assert metadata["uses_target_labels"] is False
    assert weighted_train.equals(X_train)
    assert weighted_test.equals(X_test)
    assert weights.index.equals(X_train.index)
    assert np.isfinite(weights.to_numpy()).all()
    assert (weights > 0.0).all()
    assert np.isclose(weights.mean(), 1.0)
    assert np.isclose(metadata["mean_weight"], 1.0)
    assert metadata["min_weight"] == float(weights.min())
    assert metadata["max_weight"] == float(weights.max())
    assert 0.0 <= metadata["domain_classifier_auc_training"] <= 1.0
    assert 0.0 < metadata["effective_sample_size"] <= len(X_train)


def test_paired_bootstrap_delta_macro_f1_reports_delta_interval():
    y_true = pd.Series(["static", "static", "locomotion", "locomotion", "stairs", "stairs"])
    baseline = pd.Series(["static", "locomotion", "locomotion", "static", "stairs", "locomotion"])
    candidate = pd.Series(["static", "static", "locomotion", "locomotion", "stairs", "stairs"])

    ci = _paired_bootstrap_delta_macro_f1(
        y_true=y_true,
        baseline_pred=baseline,
        candidate_pred=candidate,
        labels=["static", "locomotion", "stairs"],
        n_resamples=50,
        confidence=0.95,
        random_state=42,
    )

    assert ci["metric"] == "paired_bootstrap_delta_macro_f1_vs_rf_baseline"
    assert ci["point"] > 0.0
    assert ci["lower"] <= ci["point"] <= ci["upper"]
    assert ci["n_resamples"] == 50
    assert ci["n"] == 6


def test_summary_rows_include_delta_and_label_usage_flags():
    payload = {
        "directions": {
            "A_to_B": {
                "source_dataset": "A",
                "target_dataset": "B",
                "methods": {
                    "rf_baseline": {
                        "metrics": {"macro_f1": 0.4, "accuracy": 0.5, "support_total": 10},
                        "delta_macro_f1_vs_baseline": 0.0,
                        "paired_delta_macro_f1_ci": {
                            "lower": 0.0,
                            "upper": 0.0,
                            "confidence": 0.95,
                        },
                        "transform": {"uses_target_labels": False},
                    },
                    "rf_coral": {
                        "metrics": {"macro_f1": 0.46, "accuracy": 0.55, "support_total": 10},
                        "delta_macro_f1_vs_baseline": 0.06,
                        "paired_delta_macro_f1_ci": {
                            "lower": 0.02,
                            "upper": 0.10,
                            "confidence": 0.95,
                        },
                        "transform": {"uses_target_labels": False},
                    },
                },
            }
        }
    }

    rows = _summary_rows(payload)

    assert rows == [
        {
            "direction": "A_to_B",
            "source": "A",
            "target": "B",
            "method": "rf_baseline",
            "macro_f1": 0.4,
            "accuracy": 0.5,
            "delta_macro_f1_vs_baseline": 0.0,
            "delta_macro_f1_ci_lower": 0.0,
            "delta_macro_f1_ci_upper": 0.0,
            "delta_macro_f1_ci_confidence": 0.95,
            "support_total": 10,
            "uses_target_labels": False,
        },
        {
            "direction": "A_to_B",
            "source": "A",
            "target": "B",
            "method": "rf_coral",
            "macro_f1": 0.46,
            "accuracy": 0.55,
            "delta_macro_f1_vs_baseline": 0.06,
            "delta_macro_f1_ci_lower": 0.02,
            "delta_macro_f1_ci_upper": 0.10,
            "delta_macro_f1_ci_confidence": 0.95,
            "support_total": 10,
            "uses_target_labels": False,
        },
    ]
