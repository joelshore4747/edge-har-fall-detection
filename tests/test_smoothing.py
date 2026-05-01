"""Tests for ``scripts.lib.smoothing``."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.lib.smoothing import (
    smooth_probs,
    smoothed_majority,
)


def test_rolling_mean_kills_single_window_spike():
    probs = np.array([0.1, 0.1, 0.1, 0.1, 0.95, 0.1, 0.1, 0.1, 0.1])
    smoothed = smooth_probs(probs, mode="rolling_mean", window=5)
    # Spike must be reduced and the floor must be lifted only modestly.
    assert smoothed.max() < probs.max()
    assert smoothed.max() < 0.5  # well below the spike
    assert smoothed.min() >= 0.0


def test_none_mode_is_identity():
    probs = np.array([0.0, 0.5, 1.0])
    np.testing.assert_array_equal(smooth_probs(probs, mode="none"), probs)


def test_rolling_mean_2d_renormalises_rows():
    probs = np.array([
        [0.7, 0.2, 0.1],
        [0.6, 0.3, 0.1],
        [0.1, 0.8, 0.1],
        [0.6, 0.3, 0.1],
        [0.7, 0.2, 0.1],
    ])
    smoothed = smooth_probs(probs, mode="rolling_mean", window=3)
    assert smoothed.shape == probs.shape
    np.testing.assert_allclose(smoothed.sum(axis=1), 1.0, atol=1e-9)


def test_smoothed_majority_returns_label_from_classes():
    probs = np.array([
        [0.6, 0.4],
        [0.7, 0.3],
        [0.6, 0.4],
        [0.6, 0.4],
        [0.7, 0.3],
    ])
    label = smoothed_majority(probs, classes=["walk", "fall"], mode="rolling_mean", window=3)
    assert label == "walk"


def test_hmm_smoothing_passes_through_when_steady():
    # Steady high probability stays high.
    probs = np.full(10, 0.9)
    smoothed = smooth_probs(probs, mode="hmm", window=5)
    assert smoothed.shape == probs.shape
    assert smoothed.min() > 0.8


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        smooth_probs(np.array([0.5]), mode="kalman")  # type: ignore[arg-type]


def test_empty_array_returns_empty():
    out = smooth_probs(np.array([]), mode="rolling_mean", window=5)
    assert out.size == 0
