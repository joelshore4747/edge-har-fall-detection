import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_threshold_sweep_smoke_runs_on_mobifall_fixture(tmp_path: Path):
    run_id = "fall_threshold_sweep_fixture_smoke"
    results_root = tmp_path / "results"
    fixture_path = REPO_ROOT / "tests" / "fixtures" / "mobifall_sample.csv"

    cmd = [
        sys.executable,
        "scripts/sweep_fall_thresholds.py",
        "--dataset",
        "mobifall",
        "--path",
        str(fixture_path),
        "--sample-limit",
        "0",
        "--target-rate",
        "50",
        "--window-size",
        "2",
        "--step-size",
        "1",
        "--impact-thresholds",
        "8,12",
        "--confirm-post-dyn-mean-max",
        "1.0,2.0",
        "--confirm-post-var-max",
        "0.1,0.2",
        "--jerk-thresholds",
        "0,20",
        "--random-state",
        "42",
        "--test-size",
        "0.3",
        "--results-root",
        str(results_root),
        "--run-id",
        run_id,
        "--skip-plots",
    ]

    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"

    run_dir = results_root / run_id
    csv_path = run_dir / "threshold_sweep_results.csv"
    json_path = run_dir / "threshold_sweep_results.json"

    assert csv_path.exists()
    assert json_path.exists()

    df = pd.read_csv(csv_path)
    assert not df.empty
    assert {"impact_threshold", "confirm_post_dyn_mean_max", "confirm_post_var_max", "jerk_threshold", "f1", "false_alarms_count"}.issubset(df.columns)

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["sweep_name"] == "lesson7_threshold_sweep"
    assert payload["dataset"] == "mobifall"
    assert payload["sweep_summary"]["configs_evaluated"] == len(df)
