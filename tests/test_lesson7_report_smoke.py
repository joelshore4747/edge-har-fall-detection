from pathlib import Path
import subprocess
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _pick_newest_dir(candidates: list[Path]) -> Path | None:
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def test_lesson7_report_smoke(tmp_path: Path):
    validation_dirs = [
        path
        for path in (REPO_ROOT / "results" / "runs").glob("lesson7_fall_validation_sisfall__*")
        if path.is_dir() and (path / "test_predictions_windows.csv").exists()
    ]
    validation_dir = _pick_newest_dir(validation_dirs)
    if validation_dir is None:
        pytest.skip("No SisFall validation run dir available for report smoke test.")

    sweep_dirs = []
    for pattern in ("fall_threshold_sweep_sisfall_VAR_ONLY__*", "fall_threshold_sweep_sisfall_RATIO_TUNED__*"):
        candidates = [path for path in (REPO_ROOT / "results" / "runs").glob(pattern) if path.is_dir()]
        newest = _pick_newest_dir(candidates)
        if newest is not None:
            sweep_dirs.append(newest)
    if not sweep_dirs:
        pytest.skip("No SisFall sweep run dirs available for report smoke test.")

    out_json = tmp_path / "lesson7_sisfall_report.json"
    out_md = tmp_path / "lesson7_sisfall_report.md"

    cmd = [
        sys.executable,
        "scripts/lesson7_report.py",
        "--sisfall-validation-run-dir",
        str(validation_dir),
        "--sisfall-sweep-run-dirs",
        *[str(path) for path in sweep_dirs],
        "--output-json",
        str(out_json),
        "--output-md",
        str(out_md),
    ]
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert out_json.exists()
    assert out_md.exists()

    payload = out_json.read_text(encoding="utf-8")
    assert "report_name" in payload
    assert "quantiles" in payload
    assert "sweep_summaries" in payload
