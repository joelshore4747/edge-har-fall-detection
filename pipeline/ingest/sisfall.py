"""SisFall loader (Chapter 3).

Supports:
- simplified CSV extracts for tests (Chapter 3)
- SisFall raw text files / directory traversal with filename-based label inference (Chapter 3 patch)

Harmonization choice:
- treat the first 3 columns as ADXL345 accelerometer raw counts
- treat the next 3 columns as ITG3200 gyroscope raw counts
- convert to approximate physical units before final schema normalization

Notes:
- the accelerometer conversion uses ADXL345 full-resolution sensitivity (~3.9 mg/LSB)
- the gyroscope conversion uses ITG3200 sensitivity (14.375 LSB / deg/s)
- this keeps SisFall closer to the same physical space as PAMAP2, MobiFall, and later phone data
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from pipeline.ingest.common import (
    apply_label_mapping,
    finalize_ingest_dataframe,
    inject_metadata,
    normalize_columns,
    read_csv_any,
)
from pipeline.preprocess.units import GRAVITY_M_S2, normalize_sensor_units
from pipeline.schema import TASK_FALL

# SisFall sensor-scale assumptions for the currently parsed channels.
# These are based on the published hardware used for SisFall:
# - ADXL345 accelerometer: ~3.9 mg/LSB in full-resolution mode
# - ITG3200 gyroscope: 14.375 LSB / (deg/s)
SISFALL_ADXL345_G_PER_LSB = 0.0039
SISFALL_ITG3200_LSB_PER_DEG_S = 14.375


def _parse_sisfall_line(line: str) -> list[str] | None:
    text = line.strip().rstrip(";").strip()
    if not text:
        return None
    parts = [p.strip() for p in text.split(",")]
    if len(parts) < 6:
        return None
    return parts


def _convert_sisfall_raw_units(df: pd.DataFrame) -> pd.DataFrame:
    """Convert SisFall raw sensor counts to approximate physical units.

    Loader assumption:
    - cols 0:2 are ADXL345 raw accel counts
    - cols 3:5 are ITG3200 raw gyro counts

    Output target:
    - accel in m/s^2
    - gyro in rad/s
    """
    out = df.copy()

    # Raw ADXL345 counts -> g -> m/s^2
    accel_factor_m_s2_per_lsb = SISFALL_ADXL345_G_PER_LSB * GRAVITY_M_S2
    for col in ("ax", "ay", "az"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce") * accel_factor_m_s2_per_lsb

    # Raw ITG3200 counts -> deg/s
    for col in ("gx", "gy", "gz"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce") / SISFALL_ITG3200_LSB_PER_DEG_S

    # deg/s -> rad/s through the shared normalization module
    out = normalize_sensor_units(
        out,
        source_accel_unit="m_s2",
        source_gyro_unit="deg_s",
        target_accel_unit="m_s2",
        target_gyro_unit="rad_s",
    )

    notes = list(out.attrs.get("loader_notes", []))
    notes.append(
        "SisFall loader converts assumed ADXL345 raw accel counts (~3.9 mg/LSB) to m/s^2."
    )
    notes.append(
        "SisFall loader converts assumed ITG3200 raw gyro counts (14.375 LSB/deg/s) to rad/s."
    )
    notes.append(
        "Column mapping assumption: first 3 numeric channels are accel, next 3 are gyro."
    )
    out.attrs["loader_notes"] = notes

    out.attrs["unit_normalization"] = {
        "source_accel_unit": "sisfall_raw_adxl345_counts",
        "source_gyro_unit": "sisfall_raw_itg3200_counts",
        "target_accel_unit": "m_s2",
        "target_gyro_unit": "rad_s",
    }

    return out


def _read_sisfall_txt(file_path: Path, sampling_rate_hz: float = 200.0) -> pd.DataFrame:
    """Parse a SisFall raw text file.

    SisFall rows commonly contain 9 numeric channels. For Chapter 3, we map:
    - first 3 -> ax, ay, az
    - next 3 -> gx, gy, gz
    Remaining channels are ignored for now.
    """
    rows: list[list[str]] = []
    for line in file_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = _parse_sisfall_line(line)
        if parts is None:
            continue
        rows.append(parts)

    if not rows:
        raise ValueError(f"No parseable SisFall rows found in file: {file_path}")

    width = max(len(r) for r in rows)
    padded = [r + [np.nan] * (width - len(r)) for r in rows]
    raw_df = pd.DataFrame(padded)

    if raw_df.shape[1] < 6:
        raise ValueError(f"SisFall file has too few columns for accel+gyro parsing: {file_path}")

    df = pd.DataFrame(
        {
            "ax": pd.to_numeric(raw_df.iloc[:, 0], errors="coerce"),
            "ay": pd.to_numeric(raw_df.iloc[:, 1], errors="coerce"),
            "az": pd.to_numeric(raw_df.iloc[:, 2], errors="coerce"),
            "gx": pd.to_numeric(raw_df.iloc[:, 3], errors="coerce"),
            "gy": pd.to_numeric(raw_df.iloc[:, 4], errors="coerce"),
            "gz": pd.to_numeric(raw_df.iloc[:, 5], errors="coerce"),
        }
    )

    df = _convert_sisfall_raw_units(df)
    df["timestamp"] = np.arange(len(df), dtype=float) / float(sampling_rate_hz)
    return df


def _infer_sisfall_file_metadata(file_path: Path) -> dict[str, object]:
    subject_id = file_path.parent.name if file_path.parent.name else "unknown_subject"
    stem = file_path.stem
    first_token = stem.split("_")[0] if "_" in stem else stem
    label_raw = "fall" if first_token.upper().startswith("F") else "ADL"
    return {
        "subject_id": subject_id,
        "session_id": stem,
        "label_raw": label_raw,
        "source_file": str(file_path),
        "placement": "waist",
        "sampling_rate_hz": 200.0,
    }


def _iter_sisfall_txt_files(root_path: Path) -> Iterable[Path]:
    for path in sorted(root_path.rglob("*.txt")):
        if path.name.lower() == "readme.txt":
            continue
        if not path.parent.name.upper().startswith(("SA", "SE")):
            continue
        yield path


def _load_sisfall_real_file(file_path: Path) -> pd.DataFrame:
    df = _read_sisfall_txt(file_path, sampling_rate_hz=200.0)
    meta = _infer_sisfall_file_metadata(file_path)
    for key, value in meta.items():
        df[key] = value
    return df


def _load_sisfall_directory(path: Path, *, max_files: int | None = None) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    parsed_count = 0
    for file_path in _iter_sisfall_txt_files(path):
        if max_files is not None and parsed_count >= max_files:
            break
        try:
            frames.append(_load_sisfall_real_file(file_path))
            parsed_count += 1
        except Exception:
            continue
    if not frames:
        raise ValueError(f"No parseable SisFall files found under: {path}")
    return pd.concat(frames, ignore_index=True)


def load_sisfall(path: str | Path, *, validate: bool = True, max_files: int | None = None) -> pd.DataFrame:
    src_path = Path(path)
    if src_path.is_dir():
        df = _load_sisfall_directory(src_path, max_files=max_files)
        df = normalize_columns(df)
    elif src_path.suffix.lower() == ".txt":
        df = _load_sisfall_real_file(src_path)
        df = normalize_columns(df)
    else:
        df = read_csv_any(src_path)
        df = normalize_columns(df)

    rename_map = {
        "subject": "subject_id",
        "session": "session_id",
        "time": "timestamp",
        "label": "label_raw",
        "activity": "label_raw",
        "class": "label_raw",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    if "subject_id" not in df.columns or "label_raw" not in df.columns:
        raise ValueError("SisFall loader requires 'subject_id' and 'label_raw' columns in simplified CSV")

    apply_label_mapping(df, task_type=TASK_FALL)
    inject_metadata(
        df,
        dataset_name="SISFALL",
        task_type=TASK_FALL,
        source_file=src_path,
        placement="waist",
        sampling_rate_hz=200.0,
    )

    return finalize_ingest_dataframe(df, validate=validate)