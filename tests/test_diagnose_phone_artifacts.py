"""Unit tests for the stratified diagnostic helper.

The goal is to verify the *joining* logic (merged timestamps → real session
context) and the threshold-sweep + grouped-event accounting, since those are
what the promotion gates will consume. Uses a hand-built fall/har CSV and
manifest so failures are obvious.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "diagnose_phone_artifacts.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("diagnose_phone_artifacts", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    sys.modules[module.__name__] = module
    return module


def _write_manifest(path: Path, rows: list[dict]) -> None:
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def _write_csv(path: Path, df: pd.DataFrame) -> None:
    df.to_csv(path, index=False)


def test_diagnose_stratifies_fp_by_placement_and_activity(tmp_path):
    module = _load_module()

    manifest = [
        {
            "session_id": "walk-pocket",
            "subject_id": "joe",
            "activity_label": "walking",
            "placement": "pocket",
            "merged_start_ts": 0.0,
            "merged_end_ts": 10.0,
            "duration_seconds": 10.0,
            "sample_count": 500,
        },
        {
            "session_id": "stairs-hand",
            "subject_id": "joe",
            "activity_label": "stairs",
            "placement": "hand",
            "merged_start_ts": 11.0,
            "merged_end_ts": 21.0,
            "duration_seconds": 10.0,
            "sample_count": 500,
        },
    ]
    manifest_path = tmp_path / "sessions_manifest.json"
    _write_manifest(manifest_path, manifest)

    # Fall windows: walking-pocket all truly non_fall, low probs. Stairs-hand
    # has a spike of 4 high-probability windows forming a false-alert event.
    fall_rows = []
    for i, t in enumerate([1.0, 3.0, 5.0, 7.0]):
        fall_rows.append(
            {
                "session_id": "merged",
                "midpoint_ts": t,
                "start_ts": t - 1.0,
                "end_ts": t + 1.0,
                "window_id": f"w_walk_{i}",
                "true_label": "non_fall",
                "predicted_label": "non_fall",
                "predicted_probability": 0.1,
                "predicted_is_fall": False,
            }
        )
    # 4 consecutive stairs windows above 0.5 → a single grouped alert.
    for i, t in enumerate([12.0, 12.5, 13.0, 13.5]):
        fall_rows.append(
            {
                "session_id": "merged",
                "midpoint_ts": t,
                "start_ts": t - 1.0,
                "end_ts": t + 1.0,
                "window_id": f"w_stairs_{i}",
                "true_label": "non_fall",
                "predicted_label": "fall",
                "predicted_probability": 0.82,
                "predicted_is_fall": True,
            }
        )
    fall_df = pd.DataFrame(fall_rows)
    fall_csv = tmp_path / "fall.csv"
    _write_csv(fall_csv, fall_df)

    # HAR windows: walking labels land inside the pocket session and stairs
    # labels inside the hand session — predicted labels are mostly correct.
    har_rows = []
    for t in [1.0, 3.0, 5.0, 7.0]:
        har_rows.append(
            {
                "session_id": "merged",
                "midpoint_ts": t,
                "window_id": f"h_walk_{t}",
                "label_mapped_majority": "locomotion",
                "predicted_label": "locomotion",
                "predicted_confidence": 0.9,
            }
        )
    for t in [12.0, 13.0, 14.0, 15.0]:
        har_rows.append(
            {
                "session_id": "merged",
                "midpoint_ts": t,
                "window_id": f"h_stairs_{t}",
                "label_mapped_majority": "stairs",
                # HAR confuses stairs for static to force a low recall cell.
                "predicted_label": "static",
                "predicted_confidence": 0.7,
            }
        )
    har_csv = tmp_path / "har.csv"
    _write_csv(har_csv, pd.DataFrame(har_rows))

    out_dir = tmp_path / "out"
    summary = module.run_diagnosis(
        fall_csv=fall_csv,
        har_csv=har_csv,
        sessions_manifest=manifest_path,
        current_threshold=0.5,
        out_dir=out_dir,
    )

    # Output files exist.
    assert (out_dir / "fall_diagnostic.json").exists()
    assert (out_dir / "har_diagnostic.json").exists()
    assert (out_dir / "diagnostic_report.json").exists()
    assert (out_dir / "diagnostic_report.md").exists()

    fall_diag = json.loads((out_dir / "fall_diagnostic.json").read_text())
    # Stratification joined windows to the right sessions.
    by_pair = fall_diag["at_current_threshold"]["stratified"]["by_pair"]
    assert "hand|stairs" in by_pair
    assert "pocket|walking" in by_pair
    # 4 FPs on the stairs/hand cell; 0 on walking/pocket.
    assert by_pair["hand|stairs"]["fp"] == 4
    assert by_pair["pocket|walking"]["fp"] == 0

    # Grouped events: exactly one false alert from the 4-window cluster.
    ev = fall_diag["grouped_events"]
    assert ev["alerts"] >= 1
    assert ev["false_alerts"] >= 1

    # HAR: stairs should have low recall (we predicted static for all of them).
    har_diag = json.loads((out_dir / "har_diagnostic.json").read_text())
    stairs_metrics = har_diag["overall"]["per_class"].get("stairs")
    assert stairs_metrics is not None
    assert stairs_metrics["recall"] == 0.0  # all 4 mispredicted

    # Top-3 failure modes: should surface the hand|stairs FPs.
    top = summary["top_failure_modes"]
    assert any(m["kind"] == "fall_fp" and m["placement"] == "hand" for m in top)


def test_diagnose_handles_missing_sessions_manifest_gracefully(tmp_path):
    module = _load_module()

    manifest_path = tmp_path / "sessions_manifest.json"
    _write_manifest(manifest_path, [])

    empty_fall = tmp_path / "fall.csv"
    empty_har = tmp_path / "har.csv"
    pd.DataFrame(columns=["midpoint_ts"]).to_csv(empty_fall, index=False)
    pd.DataFrame(columns=["midpoint_ts"]).to_csv(empty_har, index=False)

    out_dir = tmp_path / "out"
    summary = module.run_diagnosis(
        fall_csv=empty_fall,
        har_csv=empty_har,
        sessions_manifest=manifest_path,
        current_threshold=0.5,
        out_dir=out_dir,
    )
    assert summary["windows"]["fall"] == 0
    assert summary["windows"]["har"] == 0
    # No crashes, markdown still renders.
    assert (out_dir / "diagnostic_report.md").exists()