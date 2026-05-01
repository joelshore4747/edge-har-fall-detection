import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_mobifall_acc(path: Path, rows: list[tuple[float, float, float, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["@DATA"]
    for ts, ax, ay, az in rows:
        lines.append(f"{ts},{ax},{ay},{az}")
    path.write_text("\n".join(lines), encoding="utf-8")


def test_exporter_stitch_mobifall_strong(tmp_path: Path) -> None:
    root = tmp_path / "MobiFall_Dataset_v2.0"
    adl_file = root / "sub1" / "ADL" / "WAL" / "WAL_acc_1_1.txt"
    fall_file = root / "sub1" / "FALLS" / "FOL" / "FOL_acc_1_1.txt"

    adl_rows = [(i * 0.01, 0.1, 0.1, 9.7) for i in range(12)]
    fall_rows = [(i * 0.01, 30.0 if 4 <= i <= 8 else 5.0, 0.5, 0.5) for i in range(12)]
    _write_mobifall_acc(adl_file, adl_rows)
    _write_mobifall_acc(fall_file, fall_rows)

    out_csv = tmp_path / "stitched_strongfall.csv"
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "export_runtime_input.py"),
        "--dataset",
        "mobifall",
        "--path",
        str(root),
        "--out",
        str(out_csv),
        "--stitch-adl-fall-strong",
        "--target-rate",
        "50",
        "--window-size",
        "4",
        "--step-size",
        "2",
        "--max-rows-adl",
        "20",
        "--max-rows-fall",
        "20",
        "--min-fall-detections",
        "1",
    ]
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"

    assert out_csv.exists()
    df = pd.read_csv(out_csv)
    assert {"fall", "non_fall"}.issubset(set(df["label_mapped"].astype(str)))

    report_path = out_csv.with_suffix(".report.json")
    assert report_path.exists()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report.get("qualifying_fall_windows", 0) >= 1
