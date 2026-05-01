from metrics.classification import compute_classification_metrics
from scripts.validate_lesson5_har import summarize_evaluation_from_metrics


def test_classification_metrics_include_support_counts():
    y_true = ["static", "static", "locomotion", "stairs"]
    y_pred = ["static", "locomotion", "locomotion", "other"]

    metrics = compute_classification_metrics(y_true, y_pred, labels=["static", "locomotion", "stairs", "other"])

    assert "support_total" in metrics
    assert metrics["support_total"] == 4
    assert "per_class_support" in metrics
    assert metrics["per_class_support"]["static"] == 2
    assert metrics["per_class_support"]["locomotion"] == 1
    assert metrics["per_class"]["static"]["support"] == 2


def test_validation_evaluation_summary_exposes_per_class_support():
    metrics_payload = {
        "heuristic": {
            "metrics": {
                "accuracy": 0.5,
                "macro_f1": 0.4,
                "support_total": 4,
                "per_class_support": {"static": 2, "locomotion": 2},
                "per_class": {
                    "static": {"precision": 1.0, "recall": 0.5, "support": 2},
                    "locomotion": {"precision": 0.0, "recall": 0.0, "support": 2},
                },
                "confusion_matrix": [[1, 1], [2, 0]],
            }
        },
        "random_forest": {
            "metrics": {
                "accuracy": 0.75,
                "macro_f1": 0.7,
                "support_total": 4,
                "per_class": {
                    "static": {"precision": 1.0, "recall": 1.0, "support": 2},
                    "locomotion": {"precision": 0.67, "recall": 0.5, "support": 2},
                },
                "confusion_matrix": [[2, 0], [1, 1]],
            }
        },
    }
    summary = summarize_evaluation_from_metrics(metrics_payload, artifacts_found={"confusion_matrix_random_forest_csv": True})

    heur = summary["heuristic_baseline"]
    rf = summary["rf_baseline"]
    assert heur["support_total"] == 4
    assert heur["per_class_support"]["static"] == 2
    assert rf["per_class_support"]["locomotion"] == 2
    assert summary["confusion_matrix_available"] is True
