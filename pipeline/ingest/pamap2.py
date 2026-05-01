"""PAMAP2 loader (Chapter 3 + readiness completion patch).

Supports:
- simplified CSV extracts for tests and manual slices (existing behavior)
- real PAMAP2 Protocol `.dat` parsing for readiness auditing and preprocessing

Scope choice for the common dissertation pipeline:
- selects a single IMU source (hand/wrist IMU) for the common schema
- keeps a coarse HAR label mapping suitable for cross-dataset harmonization
- preserves raw activity ID + activity name in ``label_raw``

This keeps the loader practical and testable while leaving room for later expansion
to multi-placement fusion.
"""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd

from pipeline.ingest.common import (
    apply_label_mapping,
    finalize_ingest_dataframe,
    inject_metadata,
    normalize_columns,
    read_csv_any,
)
from pipeline.schema import TASK_HAR


_PAMAP2_COLS_54 = [
    "timestamp",
    "activity_id",
    "heart_rate",
    "hand_temp",
    "hand_acc16_x",
    "hand_acc16_y",
    "hand_acc16_z",
    "hand_acc6_x",
    "hand_acc6_y",
    "hand_acc6_z",
    "hand_gyro_x",
    "hand_gyro_y",
    "hand_gyro_z",
    "hand_mag_x",
    "hand_mag_y",
    "hand_mag_z",
    "hand_orient_1",
    "hand_orient_2",
    "hand_orient_3",
    "hand_orient_4",
    "chest_temp",
    "chest_acc16_x",
    "chest_acc16_y",
    "chest_acc16_z",
    "chest_acc6_x",
    "chest_acc6_y",
    "chest_acc6_z",
    "chest_gyro_x",
    "chest_gyro_y",
    "chest_gyro_z",
    "chest_mag_x",
    "chest_mag_y",
    "chest_mag_z",
    "chest_orient_1",
    "chest_orient_2",
    "chest_orient_3",
    "chest_orient_4",
    "ankle_temp",
    "ankle_acc16_x",
    "ankle_acc16_y",
    "ankle_acc16_z",
    "ankle_acc6_x",
    "ankle_acc6_y",
    "ankle_acc6_z",
    "ankle_gyro_x",
    "ankle_gyro_y",
    "ankle_gyro_z",
    "ankle_mag_x",
    "ankle_mag_y",
    "ankle_mag_z",
    "ankle_orient_1",
    "ankle_orient_2",
    "ankle_orient_3",
    "ankle_orient_4",
]

_PAMAP2_ACTIVITY_NAMES: dict[int, str] = {
    0: "other_transition",
    1: "lying",
    2: "sitting",
    3: "standing",
    4: "walking",
    5: "running",
    6: "cycling",
    7: "nordic_walking",
    9: "watching_tv",
    10: "computer_work",
    11: "car_driving",
    12: "ascending_stairs",
    13: "descending_stairs",
    16: "vacuum_cleaning",
    17: "ironing",
    18: "folding_laundry",
    19: "house_cleaning",
    20: "playing_soccer",
    24: "rope_jumping",
}

_PAMAP2_COARSE_MAP: dict[int, str] = {
    1: "static",
    2: "static",
    3: "static",
    9: "static",
    10: "static",
    11: "static",
    4: "locomotion",
    5: "locomotion",
    6: "locomotion",
    7: "locomotion",
    12: "stairs",
    13: "stairs",
    # All other / transitional / household / sport activities default to "other"
}


def _pamap2_activity_name(activity_id: int) -> str:
    return _PAMAP2_ACTIVITY_NAMES.get(int(activity_id), "unknown_activity")


def _map_pamap2_activity(activity_id: int) -> str:
    return _PAMAP2_COARSE_MAP.get(int(activity_id), "other")


def _parse_subject_id_from_path(path: Path) -> str:
    match = re.search(r"subject(\d+)", path.stem, flags=re.IGNORECASE)
    if not match:
        return path.stem
    return match.group(1)


def _discover_pamap2_dat_files(root: Path, *, include_optional: bool = False) -> list[Path]:
    if root.is_file() and root.suffix.lower() == ".dat":
        return [root]

    # Allow either the dataset root (contains Protocol/Optional) or the Protocol folder itself.
    protocol_dir = root / "Protocol" if (root / "Protocol").exists() else root
    discovered: list[Path] = []

    if protocol_dir.exists():
        discovered.extend(sorted(protocol_dir.glob("subject*.dat")))

    if include_optional:
        optional_dir = root / "Optional" if (root / "Optional").exists() else None
        if optional_dir and optional_dir.exists():
            discovered.extend(sorted(optional_dir.glob("subject*.dat")))

    # If root is "Optional" directly and include_optional is False, still parse what was requested.
    if not discovered and root.exists():
        discovered.extend(sorted(root.glob("subject*.dat")))

    return discovered


def _load_pamap2_dat_file(path: str | Path, *, validate: bool = True) -> pd.DataFrame:
    src_path = Path(path)
    if src_path.suffix.lower() != ".dat":
        raise ValueError(f"PAMAP2 .dat parser expects a .dat file, got: {src_path}")

    # Pragmatic selection for the common schema:
    # hand IMU acc16 + hand gyroscope.
    usecols = [
        "timestamp",
        "activity_id",
        "hand_acc16_x",
        "hand_acc16_y",
        "hand_acc16_z",
        "hand_gyro_x",
        "hand_gyro_y",
        "hand_gyro_z",
    ]
    usecol_idx = [_PAMAP2_COLS_54.index(c) for c in usecols]
    raw = pd.read_csv(
        src_path,
        sep=r"\s+",
        header=None,
        names=_PAMAP2_COLS_54,
        usecols=usecol_idx,
        na_values=["NaN"],
        engine="python",
    )

    raw = raw.rename(
        columns={
            "hand_acc16_x": "ax",
            "hand_acc16_y": "ay",
            "hand_acc16_z": "az",
            "hand_gyro_x": "gx",
            "hand_gyro_y": "gy",
            "hand_gyro_z": "gz",
        }
    )

    # Drop rows that are unusable for the common schema (missing timestamp/activity or accel axes).
    raw["timestamp"] = pd.to_numeric(raw["timestamp"], errors="coerce")
    raw["activity_id"] = pd.to_numeric(raw["activity_id"], errors="coerce")
    for col in ["ax", "ay", "az", "gx", "gy", "gz"]:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    raw = raw.dropna(subset=["timestamp", "activity_id", "ax", "ay", "az"]).reset_index(drop=True)

    if raw.empty:
        raise ValueError(f"PAMAP2 file produced no valid rows after parsing/filtering: {src_path}")

    activity_ids = raw["activity_id"].astype(int)
    activity_names = activity_ids.map(_pamap2_activity_name)
    raw["label_raw"] = activity_ids.astype(str) + ":" + activity_names.astype(str)
    raw["label_mapped"] = activity_ids.map(_map_pamap2_activity)

    subject_id = _parse_subject_id_from_path(src_path)
    session_prefix = "optional" if "optional" in str(src_path.parent).lower() else "protocol"
    raw["subject_id"] = subject_id
    raw["session_id"] = f"{session_prefix}_subject{subject_id}"
    raw["row_index"] = np.arange(len(raw), dtype=int)
    raw["source_file"] = str(src_path)
    raw["placement"] = "wrist"  # hand IMU chosen for the common schema
    raw["sampling_rate_hz"] = 100.0

    inject_metadata(
        raw,
        dataset_name="PAMAP2",
        task_type=TASK_HAR,
        source_file=src_path,
        placement="wrist",
        sampling_rate_hz=100.0,
    )

    out = finalize_ingest_dataframe(raw, validate=validate)
    out.attrs["loader_notes"] = [
        "PAMAP2 Protocol parser selected hand IMU acc16 + gyro columns for the common schema.",
        "Rows missing timestamp/activity_id/ax/ay/az are dropped during parsing.",
    ]
    out.attrs["pamap2_sensor_source"] = "hand_imu_acc16_plus_gyro"
    return out


def _load_pamap2_dir(
    root: Path,
    *,
    validate: bool = True,
    max_files: int | None = None,
    include_optional: bool = False,
) -> pd.DataFrame:
    dat_files = _discover_pamap2_dat_files(root, include_optional=include_optional)
    if not dat_files:
        raise ValueError(f"No PAMAP2 subject*.dat files found under: {root}")

    if max_files is not None and max_files > 0:
        dat_files = dat_files[:max_files]

    frames: list[pd.DataFrame] = []
    failures: list[str] = []
    for dat_file in dat_files:
        try:
            frames.append(_load_pamap2_dat_file(dat_file, validate=False))
        except Exception as exc:  # noqa: BLE001 - aggregate failures for audit/inspection use
            failures.append(f"{dat_file}: {type(exc).__name__}: {exc}")

    if not frames:
        raise ValueError(
            "Failed to parse all discovered PAMAP2 .dat files. "
            + (f"First error: {failures[0]}" if failures else "")
        )

    out = pd.concat(frames, ignore_index=True)
    # Recreate row_index per source file group for deterministic local indexing.
    if "source_file" in out.columns:
        out["row_index"] = out.groupby("source_file", dropna=False, sort=False).cumcount().astype(int)
    out = finalize_ingest_dataframe(out, validate=validate)
    notes = [
        "PAMAP2 real parser loaded Protocol .dat files using hand IMU acc16 + gyro columns.",
        f"Files requested={len(dat_files)}, loaded={len(frames)}, failed={len(failures)}.",
    ]
    if include_optional:
        notes.append("Optional folder inclusion enabled.")
    if failures:
        notes.append("Some files failed during directory parse; inspect audit output for details.")
    out.attrs["loader_notes"] = notes
    out.attrs["pamap2_file_failures"] = failures
    out.attrs["pamap2_sensor_source"] = "hand_imu_acc16_plus_gyro"
    return out


def load_pamap2(
    path: str | Path,
    *,
    validate: bool = True,
    max_files: int | None = None,
    include_optional: bool = False,
) -> pd.DataFrame:
    src_path = Path(path)
    if src_path.is_dir():
        return _load_pamap2_dir(src_path, validate=validate, max_files=max_files, include_optional=include_optional)
    if src_path.suffix.lower() == ".dat":
        return _load_pamap2_dat_file(src_path, validate=validate)

    df = read_csv_any(src_path)
    df = normalize_columns(df)

    rename_map = {
        "subject": "subject_id",
        "session": "session_id",
        "time": "timestamp",
        "activity": "label_raw",
        "activity_name": "label_raw",
        "label": "label_raw",
        "hand_ax": "ax",
        "hand_ay": "ay",
        "hand_az": "az",
        "hand_gx": "gx",
        "hand_gy": "gy",
        "hand_gz": "gz",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    if "subject_id" not in df.columns or "label_raw" not in df.columns:
        raise ValueError("PAMAP2 loader requires 'subject_id' and 'label_raw' columns in simplified CSV")

    apply_label_mapping(df, task_type=TASK_HAR)
    inject_metadata(
        df,
        dataset_name="PAMAP2",
        task_type=TASK_HAR,
        source_file=src_path,
        placement="wrist",  # scaffold default for fixture extracts; real dataset has multiple placements
        sampling_rate_hz=100.0,
    )

    out = finalize_ingest_dataframe(df, validate=validate)
    out.attrs["loader_notes"] = [
        "PAMAP2 loaded from simplified CSV extract (fixture/manual slice path), not raw Protocol .dat parser."
    ]
    return out
