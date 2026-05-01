"""Per-window probability smoothing for HAR / fall detection.

The classifier emits one probability per window. Single noisy windows
(transient impacts, brief sensor saturation) inflate the per-session
``max_p_fall`` and flip per-session activity majority votes. Smoothing the
probability sequence before aggregation removes lone spikes while preserving
sustained signal.

Two modes:

- ``rolling_mean`` (default): a centered moving average over ``window``
  windows. Cheap, deterministic, removes single-window spikes.
- ``hmm``: a small two-state Viterbi smoother (low / high) that emits a
  binary state sequence; the per-window probability is replaced by the
  posterior of the high state. Better at clean step-up/step-down boundaries
  but adds dependencies and is harder to tune.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

SmoothingMode = Literal["none", "rolling_mean", "hmm"]


def smooth_probs(
    probs: np.ndarray,
    *,
    mode: SmoothingMode = "rolling_mean",
    window: int = 5,
) -> np.ndarray:
    """Smooth a 1-D or 2-D array of per-window probabilities.

    ``probs`` may be shape ``(n_windows,)`` or ``(n_windows, n_classes)``.
    For 2-D input each class column is smoothed independently and the result
    is renormalised so each row sums to 1.
    """
    arr = np.asarray(probs, dtype=float)
    if arr.size == 0:
        return arr
    if mode == "none":
        return arr
    if mode == "rolling_mean":
        return _rolling_mean(arr, window=window)
    if mode == "hmm":
        return _hmm_smooth_binary(arr, window=window)
    raise ValueError(f"Unknown smoothing mode: {mode!r}")


def _rolling_mean(arr: np.ndarray, *, window: int) -> np.ndarray:
    if window <= 1:
        return arr
    if arr.ndim == 1:
        smoothed = _rolling_mean_1d(arr, window=window)
    elif arr.ndim == 2:
        smoothed = np.column_stack(
            [_rolling_mean_1d(arr[:, c], window=window) for c in range(arr.shape[1])]
        )
        # Renormalise so rows still sum to 1 (rolling mean is per-column,
        # which preserves sums in expectation but not exactly at edges).
        row_sums = smoothed.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        smoothed = smoothed / row_sums
    else:
        raise ValueError(f"smooth_probs expects 1-D or 2-D input, got shape {arr.shape}")
    return smoothed


def _rolling_mean_1d(x: np.ndarray, *, window: int) -> np.ndarray:
    n = x.shape[0]
    if n == 0 or window <= 1:
        return x
    # Centered rolling mean with edge handling: use the largest available
    # symmetric window at the edges (so a single noisy first window cannot
    # dominate). This is equivalent to convolving with a uniform kernel and
    # dividing by the actual count of summed elements per position.
    kernel = np.ones(window, dtype=float)
    sums = np.convolve(x, kernel, mode="same")
    counts = np.convolve(np.ones(n, dtype=float), kernel, mode="same")
    return sums / counts


def _hmm_smooth_binary(probs: np.ndarray, *, window: int) -> np.ndarray:
    """Two-state HMM (low / high) on a binary or multinomial probability array.

    For 2-D input, the high state corresponds to ``argmax`` differing from the
    "rest" class (taken as class 0). The output preserves the input shape; for
    binary input ``probs[:, 1]`` is replaced with the posterior of state high.
    """
    if probs.ndim == 1:
        emit_high = np.clip(probs, 1e-6, 1 - 1e-6)
        emit_low = 1.0 - emit_high
        post_high = _viterbi_posterior(emit_low, emit_high, window=window)
        return post_high
    if probs.ndim == 2:
        # Treat class 0 as the "rest" / low state; the rest collectively as
        # high. This is a coarse smoother — for fine-grained class smoothing
        # use rolling_mean instead.
        emit_high = np.clip(probs[:, 1:].sum(axis=1), 1e-6, 1 - 1e-6)
        emit_low = 1.0 - emit_high
        post_high = _viterbi_posterior(emit_low, emit_high, window=window)
        out = np.empty_like(probs)
        out[:, 0] = 1.0 - post_high
        # Distribute the "high" mass across the original non-zero classes
        # in proportion to their input weight, so the smoother only sharpens
        # the rest-vs-active boundary and doesn't reshuffle which active
        # class wins on a given window.
        active = probs[:, 1:]
        active_sum = active.sum(axis=1, keepdims=True)
        active_sum[active_sum == 0] = 1.0
        out[:, 1:] = active * (post_high[:, None] / active_sum)
        return out
    raise ValueError(f"smooth_probs expects 1-D or 2-D input, got shape {probs.shape}")


def _viterbi_posterior(
    emit_low: np.ndarray,
    emit_high: np.ndarray,
    *,
    window: int,
) -> np.ndarray:
    """Forward smoothing (not full Viterbi) for a 2-state HMM.

    Transition probabilities are derived from ``window``: roughly
    ``1 - 1/window`` for staying in a state, balancing responsiveness against
    over-smoothing. This is a fixed prior — there's no data-driven Baum-Welch
    estimation, which keeps the smoother deterministic across runs.
    """
    n = emit_low.shape[0]
    if n == 0:
        return emit_low
    stay = max(0.5, 1.0 - 1.0 / max(window, 2))
    switch = 1.0 - stay
    # Forward pass.
    fwd = np.zeros((n, 2), dtype=float)
    fwd[0, 0] = 0.5 * emit_low[0]
    fwd[0, 1] = 0.5 * emit_high[0]
    fwd[0] /= fwd[0].sum() or 1.0
    for t in range(1, n):
        prev_low, prev_high = fwd[t - 1]
        fwd[t, 0] = (prev_low * stay + prev_high * switch) * emit_low[t]
        fwd[t, 1] = (prev_low * switch + prev_high * stay) * emit_high[t]
        s = fwd[t].sum()
        if s > 0:
            fwd[t] /= s
    return fwd[:, 1]


def smoothed_majority(
    probs: np.ndarray,
    classes: list[str],
    *,
    mode: SmoothingMode = "rolling_mean",
    window: int = 5,
) -> str:
    """Return the majority-vote label after smoothing per-window probabilities.

    Equivalent to ``classes[argmax(smoothed.sum(axis=0))]`` for sessions where
    a single label should be assigned.
    """
    smoothed = smooth_probs(probs, mode=mode, window=window)
    if smoothed.ndim == 1:
        # Binary case: 0 = rest, 1 = active.
        return classes[1] if smoothed.mean() >= 0.5 else classes[0]
    totals = smoothed.sum(axis=0)
    return classes[int(np.argmax(totals))]
