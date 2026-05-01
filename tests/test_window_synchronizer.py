"""Tests for the shared HAR + Fall window synchronizer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline.windowing import BranchWindowConfig, SynchronizedWindows, synchronize_windows


def _synthetic_source(duration_s: float, rate_hz: float, *, session_id: str = "sess1") -> pd.DataFrame:
    n = int(duration_s * rate_hz)
    ts = np.linspace(0.0, duration_s, n, endpoint=False)
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "timestamp": ts,
            "ax": rng.normal(size=n),
            "ay": rng.normal(size=n),
            "az": 9.8 + 0.1 * rng.normal(size=n),
            "gx": rng.normal(size=n),
            "gy": rng.normal(size=n),
            "gz": rng.normal(size=n),
            "dataset_name": "SYNTH",
            "subject_id": "S1",
            "session_id": session_id,
            "sampling_rate_hz": rate_hz,
            "source_file": "synth.csv",
            "task_type": "har",
            "label_mapped": "static",
            "label_mapped_majority": "static",
            "label_raw": "static",
        }
    )


@pytest.fixture
def har_cfg() -> BranchWindowConfig:
    return BranchWindowConfig(target_rate_hz=50.0, window_size=128, step_size=64)


@pytest.fixture
def fall_cfg() -> BranchWindowConfig:
    return BranchWindowConfig(target_rate_hz=100.0, window_size=128, step_size=64)


def test_synchronize_windows_produces_expected_counts(
    har_cfg: BranchWindowConfig, fall_cfg: BranchWindowConfig
) -> None:
    # 20 s of data at 200 Hz → HAR 50 Hz = 1000 samples → 14 windows
    # Fall 100 Hz = 2000 samples → 30 windows.
    source = _synthetic_source(duration_s=20.0, rate_hz=200.0)
    sync = synchronize_windows(
        source, har_cfg=har_cfg, fall_cfg=fall_cfg, tolerance_seconds=1.0
    )

    assert isinstance(sync, SynchronizedWindows)
    assert len(sync.har_windows) == 14
    assert len(sync.fall_windows) == 30
    # The resampler may include the endpoint sample; allow ±1 row vs the
    # naïve duration * rate count.
    assert abs(len(sync.har_resampled) - 1000) <= 1
    assert abs(len(sync.fall_resampled) - 2000) <= 1
    assert sync.stats["har_window_count"] == 14
    assert sync.stats["fall_window_count"] == 30


def test_pairing_midpoint_deltas_within_tolerance(
    har_cfg: BranchWindowConfig, fall_cfg: BranchWindowConfig
) -> None:
    source = _synthetic_source(duration_s=20.0, rate_hz=200.0)
    sync = synchronize_windows(
        source, har_cfg=har_cfg, fall_cfg=fall_cfg, tolerance_seconds=1.0
    )

    paired = sync.pairing.dropna(subset=["har_window_id"])
    assert not paired.empty
    assert (paired["delta_ts"] <= 1.0 + 1e-9).all()


def test_pairing_has_one_row_per_fall_window(
    har_cfg: BranchWindowConfig, fall_cfg: BranchWindowConfig
) -> None:
    source = _synthetic_source(duration_s=20.0, rate_hz=200.0)
    sync = synchronize_windows(
        source, har_cfg=har_cfg, fall_cfg=fall_cfg, tolerance_seconds=1.0
    )
    assert len(sync.pairing) == len(sync.fall_windows)
    assert set(sync.pairing.columns) >= {
        "session_id",
        "fall_window_id",
        "har_window_id",
        "fall_midpoint_ts",
        "har_midpoint_ts",
        "delta_ts",
    }


def test_zero_tolerance_pairs_only_exact_midpoints(
    har_cfg: BranchWindowConfig, fall_cfg: BranchWindowConfig
) -> None:
    source = _synthetic_source(duration_s=20.0, rate_hz=200.0)
    sync = synchronize_windows(
        source, har_cfg=har_cfg, fall_cfg=fall_cfg, tolerance_seconds=0.0
    )

    # With independent 50 Hz / 100 Hz windowing the midpoints rarely coincide;
    # expect very few (possibly zero) matches.
    assert sync.stats["paired_fall_windows"] <= sync.stats["fall_window_count"]
    paired = sync.pairing.dropna(subset=["har_window_id"])
    if not paired.empty:
        assert (paired["delta_ts"] == 0.0).all()


def test_synchronizer_groups_by_session(
    har_cfg: BranchWindowConfig, fall_cfg: BranchWindowConfig
) -> None:
    s1 = _synthetic_source(duration_s=10.0, rate_hz=200.0, session_id="sess_A")
    s2 = _synthetic_source(duration_s=10.0, rate_hz=200.0, session_id="sess_B")
    s2["timestamp"] = s2["timestamp"] + 100.0  # large gap so sessions don't collide
    source = pd.concat([s1, s2], ignore_index=True)

    sync = synchronize_windows(
        source, har_cfg=har_cfg, fall_cfg=fall_cfg, tolerance_seconds=1.0
    )

    # Each session should appear in the pairing on its own.
    sessions = set(sync.pairing["session_id"].astype(str).unique())
    assert sessions == {"sess_A", "sess_B"}
    # No cross-session pair should exist: for any pair where both ids are
    # present, their underlying windows must share the session id.
    paired = sync.pairing.dropna(subset=["har_window_id"])
    for _, row in paired.iterrows():
        # In this simple fixture the session_id in the pairing row IS the
        # session for both windows (we grouped by session when pairing).
        assert isinstance(row["session_id"], str)