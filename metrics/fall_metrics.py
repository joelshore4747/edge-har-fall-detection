"""Safety-relevant metrics for threshold-based fall detection baselines."""

from __future__ import annotations

from typing import Sequence

from sklearn.metrics import confusion_matrix


def compute_fall_detection_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    *,
    positive_label: str = "fall",
    negative_label: str = "non_fall",
) -> dict:
    labels = [negative_label, positive_label]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    if cm.shape != (2, 2):
        # Defensive fallback for degenerate cases; confusion_matrix with fixed labels should still return 2x2.
        tn = fp = fn = tp = 0
    else:
        tn, fp, fn, tp = (int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1]))

    total = tn + fp + fn + tp
    accuracy = (tn + tp) / total if total else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0  # fall recall
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = (2 * precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) else 0.0

    per_class_support = {
        negative_label: int(tn + fp),
        positive_label: int(tp + fn),
    }
    per_class_precision = {
        negative_label: float(tn / (tn + fn)) if (tn + fn) else 0.0,
        positive_label: float(precision),
    }
    per_class_recall = {
        negative_label: float(specificity),
        positive_label: float(sensitivity),
    }

    return {
        "labels": labels,
        "confusion_matrix": cm.tolist(),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "accuracy": float(accuracy),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision),
        "f1": float(f1),
        "support_total": int(total),
        "per_class_support": per_class_support,
        "per_class_precision": per_class_precision,
        "per_class_recall": per_class_recall,
    }
