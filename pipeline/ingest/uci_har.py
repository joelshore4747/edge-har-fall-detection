"""UCI HAR loader (Chapter 3).

Supports:
- simplified CSV input used in tests/manual extracts (Chapter 3)
- UCI HAR dataset directory parsing via inertial signal files (Chapter 3 patch)

Important harmonization choice:
- when loading the real UCI HAR directory, use TOTAL acceleration rather than BODY acceleration
- convert total acceleration from g to m/s^2

Why:
- BODY acceleration has gravity removed, so stationary windows sit near 0 acceleration magnitude
- PAMAP2 and later phone capture will be much closer to raw/total acceleration with gravity present
- for cross-dataset transfer, total acceleration is the better canonical choice here
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.ingest.common import (
    apply_label_mapping,
    finalize_ingest_dataframe,
    inject_metadata,
    normalize_columns,
    read_csv_any,
)
from pipeline.preprocess.units import normalize_sensor_units
from pipeline.schema import TASK_HAR


def _find_uci_har_root(path: Path) -> Path:
    """Resolve the canonical UCI HAR dataset directory."""
    if (path / "train").exists() and (path / "test").exists():
        return path
    nested = path / "UCI-HAR Dataset"
    if nested.exists() and (nested / "train").exists():
        return nested
    raise ValueError(f"Could not locate UCI HAR root directory under: {path}")


def _read_vector_txt(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=r"\s+", header=None, engine="python")


def _read_series_txt(path: Path) -> pd.Series:
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    return df.iloc[:, 0]


def _load_uci_har_split(
    root: Path,
    split: str,
    *,
    max_windows: int | None = None,
) -> pd.DataFrame:
    split_dir = root / split
    signals_dir = split_dir / "Inertial Signals"
    if not signals_dir.exists():
        raise ValueError(f"Missing UCI HAR inertial signals directory: {signals_dir}")

    subject = _read_series_txt(split_dir / f"subject_{split}.txt")
    y = _read_series_txt(split_dir / f"y_{split}.txt")
    labels_map_df = pd.read_csv(
        root / "activity_labels.txt",
        sep=r"\s+",
        header=None,
        names=["id", "label"],
        engine="python",
    )
    labels_map = {int(row.id): str(row.label) for row in labels_map_df.itertuples(index=False)}

    acc_x = _read_vector_txt(signals_dir / f"total_acc_x_{split}.txt")
    acc_y = _read_vector_txt(signals_dir / f"total_acc_y_{split}.txt")
    acc_z = _read_vector_txt(signals_dir / f"total_acc_z_{split}.txt")

    # Body gyro is already the standard angular velocity signal provided by UCI HAR.
    gyro_x = _read_vector_txt(signals_dir / f"body_gyro_x_{split}.txt")
    gyro_y = _read_vector_txt(signals_dir / f"body_gyro_y_{split}.txt")
    gyro_z = _read_vector_txt(signals_dir / f"body_gyro_z_{split}.txt")

    n_windows = len(subject)
    if max_windows is not None:
        n_windows = min(n_windows, int(max_windows))

    subject = subject.iloc[:n_windows].reset_index(drop=True)
    y = y.iloc[:n_windows].reset_index(drop=True)
    acc_x = acc_x.iloc[:n_windows].reset_index(drop=True)
    acc_y = acc_y.iloc[:n_windows].reset_index(drop=True)
    acc_z = acc_z.iloc[:n_windows].reset_index(drop=True)
    gyro_x = gyro_x.iloc[:n_windows].reset_index(drop=True)
    gyro_y = gyro_y.iloc[:n_windows].reset_index(drop=True)
    gyro_z = gyro_z.iloc[:n_windows].reset_index(drop=True)

    n_samples_per_window = acc_x.shape[1]
    total_rows = n_windows * n_samples_per_window

    # Flatten pre-windowed sequences to sample rows while preserving their window/session identity.
    sample_idx = np.tile(np.arange(n_samples_per_window, dtype=int), n_windows)
    window_ids = np.repeat(np.arange(n_windows, dtype=int), n_samples_per_window)

    labels_raw = [labels_map.get(int(v), str(v)) for v in y.astype(int).tolist()]

    df = pd.DataFrame(
        {
            "subject_id": np.repeat(subject.astype(str).to_numpy(), n_samples_per_window),
            "session_id": np.array([f"{split}_w{idx:05d}" for idx in window_ids], dtype=object),
            "timestamp": sample_idx.astype(float) / 50.0,
            "ax": acc_x.to_numpy(dtype=float).reshape(total_rows),
            "ay": acc_y.to_numpy(dtype=float).reshape(total_rows),
            "az": acc_z.to_numpy(dtype=float).reshape(total_rows),
            "gx": gyro_x.to_numpy(dtype=float).reshape(total_rows),
            "gy": gyro_y.to_numpy(dtype=float).reshape(total_rows),
            "gz": gyro_z.to_numpy(dtype=float).reshape(total_rows),
            "label_raw": np.repeat(np.array(labels_raw, dtype=object), n_samples_per_window),
            "source_file": str(signals_dir),
            "placement": "waist",
            "sampling_rate_hz": 50.0,
            "row_index": np.arange(total_rows, dtype=int),
        }
    )

    df = normalize_sensor_units(
        df,
        source_accel_unit="g",
        source_gyro_unit="rad_s",
        target_accel_unit="m_s2",
        target_gyro_unit="rad_s",
    )
    return df


def _load_uci_har_directory(
    path: Path,
    *,
    max_windows_per_split: int | None = None,
) -> pd.DataFrame:
    root = _find_uci_har_root(path)
    frames: list[pd.DataFrame] = []

    for split in ["train", "test"]:
        frames.append(_load_uci_har_split(root, split, max_windows=max_windows_per_split))
    out = pd.concat(frames, ignore_index=True)
    out.attrs["is_prewindowed"] = True
    out.attrs["prewindowed_source"] = "UCI HAR flattened windows"
    out.attrs["loader_notes"] = [
        "UCI HAR loader uses total_acc_* signals for cross-dataset harmonization.",
        "Accelerometer values are converted from g to m/s^2.",
        "Gyroscope uses body_gyro_* signals from the source dataset.",
    ]
    return out


def load_uci_har(
    path: str | Path,
    *,
    validate: bool = True,
    max_windows_per_split: int | None = None,
) -> pd.DataFrame:
    src_path = Path(path)
    if src_path.is_dir():
        df = _load_uci_har_directory(src_path, max_windows_per_split=max_windows_per_split)
        df = normalize_columns(df)
    else:
        df = read_csv_any(src_path)
        df = normalize_columns(df)

    rename_map = {
        "subject": "subject_id",
        "subjectid": "subject_id",
        "session": "session_id",
        "time": "timestamp",
        "activity": "label_raw",
        "label": "label_raw",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    if "subject_id" not in df.columns:
        raise ValueError("UCI HAR loader requires 'subject_id' (or 'subject') column in CSV fixture/extract")
    if "label_raw" not in df.columns:
        raise ValueError("UCI HAR loader requires 'label_raw' (or 'activity'/'label') column")

    apply_label_mapping(df, task_type=TASK_HAR)
    inject_metadata(
        df,
        dataset_name="UCIHAR",
        task_type=TASK_HAR,
        source_file=src_path,
        placement="waist",
        sampling_rate_hz=50.0,
    )

    out = finalize_ingest_dataframe(df, validate=validate)
    if src_path.is_dir():
        out.attrs["is_prewindowed"] = True
        out.attrs["prewindowed_source"] = "UCI HAR flattened windows"
        out.attrs["loader_notes"] = [
            "UCI HAR loader uses total_acc_* signals for cross-dataset harmonization.",
            "Accelerometer values are converted from g to m/s^2.",
            "Gyroscope uses body_gyro_* signals from the source dataset.",
        ]
    return out