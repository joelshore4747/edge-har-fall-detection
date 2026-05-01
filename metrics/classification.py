"""Classification metrics helpers for dissertation experiments."""

from __future__ import annotations

from typing import Iterable, Sequence

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support


def _label_order(y_true: Sequence[str], y_pred: Sequence[str], labels: Iterable[str] | None = None) -> list[str]:
    if labels is not None:
        return [str(v) for v in labels]
    seen = sorted({str(v) for v in list(y_true) + list(y_pred)})
    return seen


def confusion_matrix_dataframe(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    *,
    labels: Iterable[str] | None = None,
) -> pd.DataFrame:
    label_list = _label_order(y_true, y_pred, labels=labels)
    cm = confusion_matrix(y_true, y_pred, labels=label_list)
    return pd.DataFrame(cm, index=label_list, columns=label_list)


def compute_classification_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    *,
    labels: Iterable[str] | None = None,
) -> dict:
    """Compute a compact, JSON-serializable metrics summary."""
    label_list = _label_order(y_true, y_pred, labels=labels)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=label_list,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=label_list)

    return {
        "labels": label_list,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=label_list, average="macro", zero_division=0)),
        "support_total": int(sum(int(v) for v in support)),
        "per_class_support": {
            label: int(support[idx])
            for idx, label in enumerate(label_list)
        },
        "per_class": {
            label: {
                "precision": float(precision[idx]),
                "recall": float(recall[idx]),
                "f1": float(f1[idx]),
                "support": int(support[idx]),
            }
            for idx, label in enumerate(label_list)
        },
        "confusion_matrix": cm.tolist(),
    }
