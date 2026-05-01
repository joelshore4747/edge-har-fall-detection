import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_fall_threshold_baseline_smoke_runs_on_mobifall_fixture(tmp_path: Path):
    run_id = "fall_threshold_fixture_smoke"
    results_root = tmp_path / "results"
    fixture_path = REPO_ROOT / "tests" / "fixtures" / "mobifall_sample.csv"

    cmd = [
        sys.executable,
        "scripts/run_fall_threshold_baseline.py",
        "--dataset",
        "mobifall",
        "--path",
        str(fixture_path),
        "--sample-limit",
        "0",
        "--target-rate",
        "50",
        "--skip-plots",
        "--results-root",
        str(results_root),
        "--run-id",
        run_id,
    ]
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"

    run_dir = results_root / run_id
    metrics_json = run_dir / "metrics.json"
    false_alarm_csv = run_dir / "false_alarms.csv"
    cm_csv = run_dir / "confusion_matrix_threshold_fall.csv"
    preds_csv = run_dir / "predictions_windows.csv"
    test_preds_csv = run_dir / "test_predictions_windows.csv"

    assert metrics_json.exists()
    assert false_alarm_csv.exists()
    assert cm_csv.exists()
    assert preds_csv.exists()
    assert test_preds_csv.exists()

    payload = json.loads(metrics_json.read_text(encoding="utf-8"))
    assert payload["dataset"] == "mobifall"
    assert "metrics" in payload
    assert "accuracy" in payload["metrics"]
    assert "sensitivity" in payload["metrics"]
    assert "specificity" in payload["metrics"]
    assert "false_alarm_summary" in payload
    assert "split" in payload
    assert "strategy" in payload["split"]
