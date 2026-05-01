import pandas as pd

from pipeline.preprocess.resample import estimate_sampling_rate, summarize_sampling_rate_by_group


def test_grouped_sampling_rate_summary_avoids_global_timestamp_reset_bias():
    rows = []
    # Two logical sessions with near-identical timestamp ranges (both start near 0).
    # A global sort interleaves rows from both sessions and can produce tiny deltas,
    # but per-session estimation should remain close to ~100 Hz.
    for session_idx, offset in enumerate([0.0, 1e-6], start=1):
        for i in range(6):
            rows.append(
                {
                    "dataset_name": "MOBIFALL",
                    "subject_id": "sub1",
                    "session_id": f"session_{session_idx}",
                    "source_file": f"file_{session_idx}.txt",
                    "timestamp": (i * 0.01) + offset,
                    "ax": 0.1 * i,
                    "ay": 0.2 * i,
                    "az": 0.3 * i,
                    "label_raw": "fall" if i >= 3 else "ADL",
                    "label_mapped": "fall" if i >= 3 else "non_fall",
                }
            )

    df = pd.DataFrame(rows)

    global_rate = estimate_sampling_rate(df)
    assert global_rate is not None
    assert global_rate > 1000.0  # Demonstrates why global estimate is misleading here.

    summary = summarize_sampling_rate_by_group(df)
    assert summary["groups_checked"] == 2
    assert 95.0 <= float(summary["median_hz"]) <= 105.0
    assert 95.0 <= float(summary["min_hz"]) <= 105.0
    assert 95.0 <= float(summary["max_hz"]) <= 105.0
    assert len(summary["estimated_rates_hz_preview"]) == 2
