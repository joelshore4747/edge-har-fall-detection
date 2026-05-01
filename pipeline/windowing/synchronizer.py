"""Paired HAR + Fall window synchronizer.

Runs the two independent resample+window passes that
``services.runtime_inference._prepare_{har,fall}_branch`` would each do on
their own and emits:

- ``har_windows`` / ``fall_windows`` — the raw windowed outputs (``list[dict]``
  as produced by :func:`pipeline.preprocess.window.window_dataframe`), suitable
  for handing to the existing feature-table builders.
- ``har_resampled`` / ``fall_resampled`` — the resampled dataframes the branches
  would otherwise rebuild, kept here so downstream summaries can report row
  counts without a second pass.
- ``pairing`` — a one-row-per-fall-window table of nearest-HAR matches,
  computed once from window midpoints. Consumers join HAR predictions onto
  fall windows via ``fall_window_id`` + ``har_window_id`` instead of running a
  second post-hoc :func:`pandas.merge_asof` after inference.

Rates stay independent (HAR 50 Hz / Fall 100 Hz by convention); the
synchronizer only aligns windows, it does not harmonize sample rates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from pipeline.preprocess import (
    PreprocessConfig,
    append_derived_channels,
    resample_dataframe,
    window_dataframe,
)


@dataclass(slots=True, frozen=True)
class BranchWindowConfig:
    """Window parameters for a single branch (HAR or Fall)."""

    target_rate_hz: float
    window_size: int
    step_size: int


@dataclass(slots=True)
class SynchronizedWindows:
    har_windows: list[dict[str, Any]]
    fall_windows: list[dict[str, Any]]
    har_resampled: pd.DataFrame
    fall_resampled: pd.DataFrame
    pairing: pd.DataFrame
    har_config: BranchWindowConfig
    fall_config: BranchWindowConfig
    tolerance_seconds: float
    stats: dict[str, Any] = field(default_factory=dict)


def _resample_and_window(
    source_df: pd.DataFrame,
    *,
    cfg: BranchWindowConfig,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    resampled = resample_dataframe(source_df, target_rate_hz=cfg.target_rate_hz)
    resampled = append_derived_channels(resampled)
    preprocess_cfg = PreprocessConfig(target_sampling_rate_hz=cfg.target_rate_hz)
    windows = window_dataframe(
        resampled,
        window_size=cfg.window_size,
        step_size=cfg.step_size,
        config=preprocess_cfg,
    )
    return resampled, windows


def _windows_to_midpoint_frame(
    windows: list[dict[str, Any]],
    *,
    window_id_col: str,
) -> pd.DataFrame:
    """Project the subset of window fields used for pairing."""
    if not windows:
        return pd.DataFrame(
            columns=[window_id_col, "session_id", "midpoint_ts"]
        )

    records: list[dict[str, Any]] = []
    for idx, w in enumerate(windows):
        midpoint = w.get("midpoint_ts")
        if midpoint is None:
            start_ts = w.get("start_ts")
            end_ts = w.get("end_ts")
            if start_ts is not None and end_ts is not None:
                midpoint = (float(start_ts) + float(end_ts)) / 2.0
            else:
                midpoint = float(idx)
        records.append(
            {
                window_id_col: w.get("window_id", idx),
                "session_id": w.get("session_id") or "unknown_session",
                "midpoint_ts": float(midpoint),
            }
        )
    frame = pd.DataFrame.from_records(records)
    frame["session_id"] = frame["session_id"].astype(str)
    frame["midpoint_ts"] = pd.to_numeric(frame["midpoint_ts"], errors="coerce")
    return frame


def _build_pairing(
    *,
    fall_frame: pd.DataFrame,
    har_frame: pd.DataFrame,
    tolerance_seconds: float,
) -> pd.DataFrame:
    """One row per fall window; nearest HAR window within ``tolerance_seconds``.

    Missing matches (either because a session has no HAR windows, or because
    no HAR window falls within tolerance) come through with ``har_window_id``
    NaN and ``delta_ts`` NaN. This mirrors the semantics of the post-hoc
    alignment that ``merge_asof(..., direction='nearest', tolerance=...)``
    currently produces inside ``runtime_inference``.
    """
    if fall_frame.empty:
        return pd.DataFrame(
            columns=[
                "session_id",
                "fall_window_id",
                "har_window_id",
                "fall_midpoint_ts",
                "har_midpoint_ts",
                "delta_ts",
            ]
        )

    fall_frame = fall_frame.rename(columns={"midpoint_ts": "fall_midpoint_ts"}).copy()
    fall_frame["_orig_order"] = np.arange(len(fall_frame))

    if har_frame.empty:
        out = fall_frame[["session_id", "fall_window_id", "fall_midpoint_ts"]].copy()
        out["har_window_id"] = pd.NA
        out["har_midpoint_ts"] = pd.NA
        out["delta_ts"] = pd.NA
        return out.reset_index(drop=True)

    har_frame = har_frame.rename(columns={"midpoint_ts": "har_midpoint_ts"}).copy()

    merged_parts: list[pd.DataFrame] = []
    session_ids = sorted(set(fall_frame["session_id"]) | set(har_frame["session_id"]))
    for sid in session_ids:
        f = fall_frame[fall_frame["session_id"] == sid].copy()
        h = har_frame[har_frame["session_id"] == sid].copy()
        if f.empty:
            continue
        if h.empty:
            f["har_window_id"] = pd.NA
            f["har_midpoint_ts"] = pd.NA
            merged_parts.append(f)
            continue

        f = f.sort_values("fall_midpoint_ts", kind="stable").reset_index(drop=True)
        h = h.sort_values("har_midpoint_ts", kind="stable").reset_index(drop=True)

        merged = pd.merge_asof(
            f,
            h[["har_window_id", "har_midpoint_ts"]],
            left_on="fall_midpoint_ts",
            right_on="har_midpoint_ts",
            direction="nearest",
            tolerance=float(tolerance_seconds),
        )
        merged_parts.append(merged)

    if not merged_parts:
        return pd.DataFrame(
            columns=[
                "session_id",
                "fall_window_id",
                "har_window_id",
                "fall_midpoint_ts",
                "har_midpoint_ts",
                "delta_ts",
            ]
        )

    out = pd.concat(merged_parts, ignore_index=True, sort=False)
    out = out.sort_values("_orig_order", kind="stable").reset_index(drop=True)
    out = out.drop(columns=["_orig_order"])
    out["delta_ts"] = (
        pd.to_numeric(out["har_midpoint_ts"], errors="coerce")
        - pd.to_numeric(out["fall_midpoint_ts"], errors="coerce")
    ).abs()
    return out[
        [
            "session_id",
            "fall_window_id",
            "har_window_id",
            "fall_midpoint_ts",
            "har_midpoint_ts",
            "delta_ts",
        ]
    ]


def synchronize_windows(
    source_df: pd.DataFrame,
    *,
    har_cfg: BranchWindowConfig,
    fall_cfg: BranchWindowConfig,
    tolerance_seconds: float = 1.0,
) -> SynchronizedWindows:
    """Run the HAR + Fall resample+window passes and emit a pairing table.

    Parameters
    ----------
    source_df : DataFrame
        Raw ingested samples (whatever the caller already passes to the two
        branches today).
    har_cfg, fall_cfg : BranchWindowConfig
        Per-branch window parameters. Typically resolved from the artifact
        metadata via the Phase-C ``_artifact_{har,fall}_preprocess`` helpers.
    tolerance_seconds : float
        Maximum |HAR_midpoint - Fall_midpoint| for a pair to be recorded.
        Beyond this the fall window is emitted with no HAR match (NaN).
    """
    har_resampled, har_windows = _resample_and_window(source_df, cfg=har_cfg)
    fall_resampled, fall_windows = _resample_and_window(source_df, cfg=fall_cfg)

    har_frame = _windows_to_midpoint_frame(har_windows, window_id_col="har_window_id")
    fall_frame = _windows_to_midpoint_frame(fall_windows, window_id_col="fall_window_id")

    pairing = _build_pairing(
        fall_frame=fall_frame,
        har_frame=har_frame,
        tolerance_seconds=tolerance_seconds,
    )

    matched_count = int(pairing["har_window_id"].notna().sum()) if not pairing.empty else 0
    stats: dict[str, Any] = {
        "har_window_count": int(len(har_windows)),
        "fall_window_count": int(len(fall_windows)),
        "paired_fall_windows": matched_count,
        "unpaired_fall_windows": int(len(fall_windows) - matched_count),
        "tolerance_seconds": float(tolerance_seconds),
    }

    return SynchronizedWindows(
        har_windows=har_windows,
        fall_windows=fall_windows,
        har_resampled=har_resampled,
        fall_resampled=fall_resampled,
        pairing=pairing,
        har_config=har_cfg,
        fall_config=fall_cfg,
        tolerance_seconds=float(tolerance_seconds),
        stats=stats,
    )