"""Paired significance tests for model comparisons on shared examples."""

from __future__ import annotations

from dataclasses import dataclass
from math import comb, erfc, sqrt
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class McNemarResult:
    """Exact McNemar test result for two classifiers on paired examples."""

    statistic: float
    p_value: float
    n: int
    both_correct: int
    a_correct_b_wrong: int
    a_wrong_b_correct: int
    both_wrong: int

    @property
    def discordant(self) -> int:
        return self.a_correct_b_wrong + self.a_wrong_b_correct


def _as_bool_array(values: Sequence[object], name: str) -> np.ndarray:
    arr = np.asarray(list(values), dtype=bool)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return arr


def _exact_two_sided_binomial_p_value(k: int, n: int) -> float:
    """Two-sided exact binomial test under p=0.5 for McNemar discordants."""
    if n == 0:
        return 1.0
    tail = min(k, n - k)
    probability = sum(comb(n, i) for i in range(tail + 1)) / (2**n)
    return min(1.0, 2.0 * probability)


def mcnemar_test(
    y_true: Sequence[object],
    y_pred_a: Sequence[object],
    y_pred_b: Sequence[object],
) -> McNemarResult:
    """Run an exact paired McNemar test over two aligned prediction vectors.

    The test is applied to correctness, not directly to class labels:
    ``a_correct_b_wrong`` counts examples where model A is correct and model B
    is wrong; ``a_wrong_b_correct`` counts the reverse. The exact p-value tests
    whether those two discordant counts are symmetric.
    """

    true_arr = np.asarray(list(y_true))
    pred_a_arr = np.asarray(list(y_pred_a))
    pred_b_arr = np.asarray(list(y_pred_b))
    if true_arr.shape != pred_a_arr.shape or true_arr.shape != pred_b_arr.shape:
        raise ValueError("y_true, y_pred_a, and y_pred_b must align")
    if true_arr.ndim != 1:
        raise ValueError("inputs must be one-dimensional")

    a_correct = pred_a_arr == true_arr
    b_correct = pred_b_arr == true_arr
    both_correct = int(np.sum(a_correct & b_correct))
    a_correct_b_wrong = int(np.sum(a_correct & ~b_correct))
    a_wrong_b_correct = int(np.sum(~a_correct & b_correct))
    both_wrong = int(np.sum(~a_correct & ~b_correct))
    discordant = a_correct_b_wrong + a_wrong_b_correct

    if discordant == 0:
        statistic = 0.0
    else:
        statistic = (abs(a_correct_b_wrong - a_wrong_b_correct) - 1) ** 2 / discordant
    p_value = _exact_two_sided_binomial_p_value(a_wrong_b_correct, discordant)

    return McNemarResult(
        statistic=float(statistic),
        p_value=float(p_value),
        n=int(true_arr.shape[0]),
        both_correct=both_correct,
        a_correct_b_wrong=a_correct_b_wrong,
        a_wrong_b_correct=a_wrong_b_correct,
        both_wrong=both_wrong,
    )


def mcnemar_chi_square_p_value(result: McNemarResult) -> float:
    """Continuity-corrected chi-square p-value for df=1, for reporting only."""
    if result.discordant == 0:
        return 1.0
    return float(erfc(sqrt(result.statistic / 2.0)))


def binary_from_labels(values: Sequence[object], positive_label: str = "fall") -> np.ndarray:
    """Convert fall/non-fall string labels into a boolean prediction vector."""
    arr = np.asarray([str(value).strip().lower() == positive_label for value in values], dtype=bool)
    return _as_bool_array(arr, "values")
