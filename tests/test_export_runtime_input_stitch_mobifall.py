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


def test_exporter_stitch_mobifall(tmp_path: Path) -> None:
    root = tmp_path / "MobiFall_Dataset_v2.0"
    adl_file = root / "sub1" / "ADL" / "WAL" / "WAL_acc_1_1.txt"
    fall_file = root / "sub1" / "FALLS" / "FOL" / "FOL_acc_1_1.txt"

    _write_mobifall_acc(
        adl_file,
        [
            (0.0, 0.1, 0.1, 9.7),
            (0.01, 0.2, 0.1, 9.6),
            (0.02, 0.15, 0.1, 9.5),
            (0.03, 0.1, 0.2, 9.8),
        ],
    )
    _write_mobifall_acc(
        fall_file,
        [
            (0.0, 12.0, 5.0, 20.0),
            (0.01, 15.0, 6.0, 25.0),
            (0.02, 9.0, 4.0, 18.0),
            (0.03, 8.0, 3.0, 17.0),
        ],
    )

    out_csv = tmp_path / "stitched.csv"
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "export_runtime_input.py"),
        "--dataset",
        "mobifall",
        "--path",
        str(root),
        "--out",
        str(out_csv),
        "--stitch-adl-fall",
        "--sample-limit",
        "0",
        "--max-rows",
        "100",
    ]
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"

    assert out_csv.exists()
    df = pd.read_csv(out_csv)
    assert {"fall", "non_fall"}.issubset(set(df["label_mapped"].astype(str)))
    assert (df["timestamp"].diff().dropna() > 0).all()

    meta_path = Path(str(out_csv) + ".meta.json")
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta.get("chosen_adl_file")
    assert meta.get("chosen_fall_file")
    assert meta.get("acc_mag_max_fall") >= meta.get("acc_mag_max_adl")
