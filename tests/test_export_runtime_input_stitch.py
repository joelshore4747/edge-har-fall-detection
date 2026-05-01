from pathlib import Path

import pandas as pd

from pipeline.ingest import load_mobifall
from scripts.export_runtime_input import _label_changes, _stitch_segments


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_stitch_segments_produces_transitions_and_monotonic_timestamps():
    fixture_path = REPO_ROOT / "tests" / "fixtures" / "mobifall_sample.csv"
    df = load_mobifall(fixture_path)

    seg_a = df[df["label_mapped"] == "non_fall"].copy()
    seg_b = df[df["label_mapped"] == "fall"].copy()

    assert not seg_a.empty
    assert not seg_b.empty

    seg_a["session_id"] = "segA"
    seg_b["session_id"] = "segB"

    stitched_df, summary = _stitch_segments(
        pd.concat([seg_a, seg_b], ignore_index=True),
        min_label_changes=1,
        max_rows=1000,
        seed=123,
        stitch_max_segments=2,
        target_rate=50.0,
        min_rows_per_segment=1,
    )

    required_cols = {"timestamp", "ax", "ay", "az", "label_mapped"}
    assert required_cols.issubset(stitched_df.columns)
    assert "stitched_segment_index" in stitched_df.columns
    assert "stitched_source_session" in stitched_df.columns

    diffs = stitched_df["timestamp"].diff().dropna()
    assert (diffs > 0).all()

    assert _label_changes(stitched_df["label_mapped"].astype(str)) >= 1
    assert summary["label_changes"] >= 1
    assert summary["segments_used"] >= 2
    assert summary["timestamp_min"] is not None
    assert summary["timestamp_max"] is not None
