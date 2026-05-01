"""Non-parametric bootstrap CIs for classification metrics.

Used by the results writeup so every reported number carries a 95% interval.
With small datasets (74 sessions, 20 fall events) point estimates alone are
misleading — bootstrap is the cheapest way to make that uncertainty explicit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np


@dataclass(frozen=True)
class CI:
    point: float
    lower: float
    upper: float
    n: int

    def __str__(self) -> str:
        return f"{self.point:.3f} [{self.lower:.3f}, {self.upper:.3f}]"


def bootstrap_metric(
    y_true: Sequence,
    y_pred: Sequence,
    metric_fn: Callable[[Sequence, Sequence], float],
    *,
    n_resamples: int = 2000,
    confidence: float = 0.95,
    random_state: int = 42,
) -> CI:
    """Resample (y_true, y_pred) pairs with replacement; compute the metric on each."""
    y_true_arr = np.asarray(list(y_true))
    y_pred_arr = np.asarray(list(y_pred))
    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError("y_true and y_pred must align")
    n = y_true_arr.shape[0]
    if n == 0:
        return CI(point=float("nan"), lower=float("nan"), upper=float("nan"), n=0)

    rng = np.random.default_rng(random_state)
    point = float(metric_fn(y_true_arr, y_pred_arr))
    samples = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        samples[i] = metric_fn(y_true_arr[idx], y_pred_arr[idx])
    alpha = (1.0 - confidence) / 2.0
    lower = float(np.quantile(samples, alpha))
    upper = float(np.quantile(samples, 1.0 - alpha))
    return CI(point=point, lower=lower, upper=upper, n=n)
