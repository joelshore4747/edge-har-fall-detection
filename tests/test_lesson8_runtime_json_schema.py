import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_lesson8_runtime_dry_run_outputs_json(tmp_path: Path):
    out_json = tmp_path / "runtime_summary.json"
    fixture_path = REPO_ROOT / "tests" / "fixtures" / "timeseries_irregular.csv"

    cmd = [
        sys.executable,
        "scripts/run_runtime_on_csv.py",
        "--input",
        str(fixture_path),
        "--target-rate",
        "50",
        "--window-size",
        "8",
        "--step-size",
        "4",
        "--run-name",
        "lesson8_runtime_fixture",
        "--out-json",
        str(out_json),
        "--dry-run",
    ]
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"

    assert out_json.exists(), f"Expected output JSON at {out_json}"
    payload = json.loads(out_json.read_text(encoding="utf-8"))

    required_keys = {
        "run_name",
        "run_id",
        "input_path",
        "window_params",
        "counts_by_event_type",
        "counts_by_label",
        "ts_min",
        "ts_max",
        "events_preview",
    }
    assert required_keys.issubset(payload.keys())
    assert payload["run_name"] == "lesson8_runtime_fixture"
    assert payload["window_params"]["target_rate"] == 50.0
    assert "window_size" in payload["window_params"]
    assert "step_size" in payload["window_params"]
    assert payload["counts_by_event_type"]
    assert sum(payload["counts_by_event_type"].values()) >= 1
    assert "run_start" in payload["counts_by_event_type"]
    assert "activity_current" in payload["counts_by_event_type"]

    events_preview = payload["events_preview"]
    assert isinstance(events_preview, list)
    assert len(events_preview) >= 1
    for event in events_preview:
        for key in {"ts", "event_type", "label", "payload"}:
            assert key in event
