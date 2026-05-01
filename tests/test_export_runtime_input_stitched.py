from pathlib import Path

import pandas as pd

from pipeline.ingest import load_mobifall
from scripts.export_runtime_input import _filter_required, _stitch_adl_fall


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_stitched_adl_fall_exports_monotonic_with_gap(tmp_path: Path) -> None:
    fixture_path = REPO_ROOT / "tests" / "fixtures" / "mobifall_sample.csv"
    df = load_mobifall(fixture_path)
    df = _filter_required(df)

    stitched_df, summary = _stitch_adl_fall(
        df,
        max_rows=20,
        gap_seconds=2.0,
        min_fall_triggers=0,
        fall_search_limit=5,
        seed=123,
        target_rate=50.0,
        window_size=2,
        step_size=1,
        min_rows_per_segment=1,
    )

    out_path = tmp_path / "stitched.csv"
    stitched_df.to_csv(out_path, index=False)

    loaded = pd.read_csv(out_path)
    required_cols = {"timestamp", "ax", "ay", "az", "label_mapped"}
    assert required_cols.issubset(loaded.columns)
    assert "stitched_segment_role" in loaded.columns

    diffs = loaded["timestamp"].diff().dropna()
    assert (diffs > 0).all()

    labels = set(loaded["label_mapped"].astype(str).tolist())
    assert "fall" in labels
    assert "non_fall" in labels

    adl_max = loaded.loc[loaded["stitched_segment_role"] == "adl", "timestamp"].max()
    fall_min = loaded.loc[loaded["stitched_segment_role"] == "fall", "timestamp"].min()
    assert fall_min - adl_max >= 2.0

    assert summary["rows_total"] == len(stitched_df)
    assert summary["labels_present"].get("fall", 0) > 0
    assert summary["labels_present"].get("non_fall", 0) > 0
