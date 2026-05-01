"""Confusion matrix export and plotting helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def confusion_matrix_to_dataframe(cm: np.ndarray | list[list[int]], labels: Iterable[str]) -> pd.DataFrame:
    arr = np.asarray(cm)
    label_list = [str(v) for v in labels]
    return pd.DataFrame(arr, index=label_list, columns=label_list)


def save_confusion_matrix_csv(
    cm: np.ndarray | list[list[int]],
    labels: Iterable[str],
    out_path: str | Path,
) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = confusion_matrix_to_dataframe(cm, labels)
    df.to_csv(out)
    return out


def plot_confusion_matrix(
    cm: np.ndarray | list[list[int]],
    labels: Iterable[str],
    *,
    title: str = "Confusion Matrix",
    normalize: bool = False,
    out_path: str | Path | None = None,
) -> Path | None:
    """Plot a confusion matrix using matplotlib.

    If matplotlib is unavailable, raises the import error so the caller can decide
    whether to skip plotting. CSV export is available separately.
    """
    import matplotlib.pyplot as plt

    arr = np.asarray(cm, dtype=float)
    label_list = [str(v) for v in labels]
    if normalize:
        row_sums = arr.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        arr = arr / row_sums

    side = max(5.5, min(10.0, 3.5 + 0.75 * len(label_list)))
    fig, ax = plt.subplots(figsize=(side, side * 0.85))
    im = ax.imshow(arr, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(label_list)))
    ax.set_yticks(range(len(label_list)))
    ax.set_xticklabels(label_list, rotation=45, ha="right")
    ax.set_yticklabels(label_list)

    vmax = float(np.max(arr)) if arr.size else 0.0
    text_threshold = vmax * 0.5 if vmax > 0 else 0.0
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            val = arr[i, j]
            txt = f"{val:.2f}" if normalize else f"{int(val)}"
            color = "white" if val > text_threshold else "black"
            ax.text(j, i, txt, ha="center", va="center", color=color, fontsize=9)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()

    if out_path is None:
        return None

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out
