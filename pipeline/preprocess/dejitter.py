"""Timestamp-jitter cleanup applied before resampling.

Some mobile clients (notably an iOS Core Motion startup race in the
edge-fall-detection app) emit a single sample with ``timestamp=0`` *before*
the first real motion callback fires. The next sample then arrives at the
device's actual monotonic clock, often tens or hundreds of seconds later.

If left in place, that phantom sample inflates ``last_ts - first_ts``
(used as the session duration), creates an enormous interpolated gap that
the model reads as a long flat 'static' run, and pushes the dashboard
timeline x-axis far beyond the real recording length.

This module detects and removes that phantom so downstream consumers
(training, runtime inference, the dashboard) all see the real data span.
"""

from __future__ import annotations

import pandas as pd


PHANTOM_GAP_THRESHOLD_SECONDS = 5.0


def drop_phantom_leading_samples(
    df: pd.DataFrame,
    *,
    timestamp_col: str = "timestamp",
    max_gap_seconds: float = PHANTOM_GAP_THRESHOLD_SECONDS,
) -> pd.DataFrame:
    """Drop a leading sample that is separated from the rest by a large gap.

    Returns the input unchanged if there is no qualifying phantom: fewer than
    two rows, missing timestamp column, or the leading gap is below
    ``max_gap_seconds``. Only ever drops *one* leading row — this is
    deliberately surgical so it cannot corrupt sessions whose real data
    happens to start with a brief pause.
    """
    if df is None or len(df) < 2 or timestamp_col not in df.columns:
        return df
    ts = pd.to_numeric(df[timestamp_col], errors="coerce")
    if ts.isna().iloc[0] or ts.isna().iloc[1]:
        return df
    gap = float(ts.iloc[1]) - float(ts.iloc[0])
    if gap <= max_gap_seconds:
        return df
    return df.iloc[1:].reset_index(drop=True)
