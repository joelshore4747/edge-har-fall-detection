"""MobiFall loader (dissertation-grade rewrite).

Goals:
- support real MobiFall directory traversal
- pair accelerometer + gyroscope streams per trial when available
- optionally retain orientation if present
- derive per-trial metadata and estimated sampling rate
- keep simplified CSV fallback for tests/manual validation

Design notes:
- accelerometer is required for a trial to be loaded
- gyroscope/orientation are optional and merged when present
- session IDs are trial-level, not sensor-file-level
- skipped files are tolerated during directory traversal, but only trials
  with valid accelerometer data are emitted
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
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
from pipeline.schema import TASK_FALL
from pipeline.preprocess.units import normalize_sensor_units

_SENSOR_TOKEN_RE = re.compile(r"_(acc|gyro|ori|orientation)_", flags=re.IGNORECASE)


@dataclass(slots=True)
class _TrialFiles:
    acc: Path | None = None
    gyro: Path | None = None
    orientation: Path | None = None


def _coerce_mobifall_timestamps_to_seconds(raw_ts: pd.Series) -> pd.Series:
    """
    Convert raw timestamps to relative seconds using a scale heuristic.

    MobiFall exports can differ in timestamp scale; this keeps the loader robust.
    """
    ts = pd.to_numeric(raw_ts, errors="coerce")
    if ts.dropna().empty:
        return ts

    diffs = ts.diff().dropna()
    diffs = diffs[diffs > 0]

    scale = 1.0
    if not diffs.empty:
        median_diff = float(diffs.median())
        if median_diff > 1e6:
            scale = 1e9
        elif median_diff > 1e3:
            scale = 1e6
        elif median_diff > 10:
            scale = 1e3

    ts = (ts - ts.dropna().iloc[0]) / scale
    return ts


def _estimate_sampling_rate_hz(timestamps_s: pd.Series) -> float:
    ts = pd.to_numeric(timestamps_s, errors="coerce").dropna()
    if len(ts) < 2:
        return float("nan")
    diffs = ts.diff().dropna()
    diffs = diffs[diffs > 0]
    if diffs.empty:
        return float("nan")
    median_dt = float(diffs.median())
    if median_dt <= 0:
        return float("nan")
    return float(1.0 / median_dt)


def _parse_mobifall_sensor_txt(file_path: Path, sensor_kind: str) -> pd.DataFrame:
    """
    Parse a MobiFall sensor txt file into a DataFrame.

    Expected data rows after @DATA:
      timestamp,x,y,z
    """
    records: list[list[str]] = []
    in_data = False

    for line in file_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        text = line.strip()
        if not text:
            continue
        if text.upper() == "@DATA":
            in_data = True
            continue
        if not in_data or text.startswith("#"):
            continue

        parts = [p.strip() for p in text.split(",")]
        if len(parts) < 4:
            continue
        records.append(parts[:4])

    if not records:
        raise ValueError(f"No parseable sensor records found in MobiFall file: {file_path}")

    df = pd.DataFrame(records, columns=["timestamp", "x", "y", "z"])
    df["timestamp"] = _coerce_mobifall_timestamps_to_seconds(df["timestamp"])

    rename_map = {
        "acc": {"x": "ax", "y": "ay", "z": "az"},
        "gyro": {"x": "gx", "y": "gy", "z": "gz"},
        "orientation": {"x": "ox", "y": "oy", "z": "oz"},
    }
    if sensor_kind not in rename_map:
        raise ValueError(f"Unknown sensor kind: {sensor_kind}")

    df = df.rename(columns=rename_map[sensor_kind])

    for col in df.columns:
        if col != "timestamp":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def _sensor_kind_from_name(file_path: Path) -> str | None:
    name = file_path.name.lower()
    if "_acc_" in name:
        return "acc"
    if "_gyro_" in name:
        return "gyro"
    if "_ori_" in name or "_orientation_" in name:
        return "orientation"
    return None


def _trial_key_from_file(file_path: Path) -> str:
    """
    Convert e.g.
      BSC_acc_4_1.txt  -> BSC_4_1
      BSC_gyro_4_1.txt -> BSC_4_1
    so different sensor files map to the same trial.
    """
    stem = file_path.stem
    return _SENSOR_TOKEN_RE.sub("_", stem)


def _infer_mobifall_trial_metadata(file_path: Path) -> dict[str, object]:
    parts_lower = [p.lower() for p in file_path.parts]
    subject_id = next((p for p in file_path.parts if p.lower().startswith("sub")), "unknown_subject")

    activity_code = file_path.parent.name
    category = "unknown"
    if "adl" in parts_lower:
        category = "ADL"
    elif "falls" in parts_lower:
        category = "fall"

    label_raw = "fall" if category == "fall" else ("ADL" if category == "ADL" else activity_code)
    session_id = f"{activity_code}:{_trial_key_from_file(file_path)}"

    return {
        "subject_id": subject_id,
        "session_id": session_id,
        "label_raw": label_raw,
        "placement": "pocket",
    }


def _iter_mobifall_sensor_files(root_path: Path) -> Iterable[Path]:
    for path in sorted(root_path.rglob("*.txt")):
        if _sensor_kind_from_name(path) is None:
            continue
        yield path


def _discover_trial_groups(root_path: Path) -> list[tuple[str, _TrialFiles]]:
    """
    Group sensor files by logical trial.

    Key is built from relative parent folder + trial key, so parallel activity
    names in different folders do not collide.
    """
    grouped: dict[str, _TrialFiles] = {}

    for path in _iter_mobifall_sensor_files(root_path):
        rel_parent = str(path.parent.relative_to(root_path))
        trial_key = _trial_key_from_file(path)
        group_key = f"{rel_parent}::{trial_key}"

        if group_key not in grouped:
            grouped[group_key] = _TrialFiles()

        kind = _sensor_kind_from_name(path)
        if kind == "acc":
            grouped[group_key].acc = path
        elif kind == "gyro":
            grouped[group_key].gyro = path
        elif kind == "orientation":
            grouped[group_key].orientation = path

    return sorted(grouped.items(), key=lambda x: x[0])


def _merge_streams_on_timestamp(
    acc_df: pd.DataFrame,
    gyro_df: pd.DataFrame | None = None,
    ori_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Align optional streams onto the accelerometer timeline.

    We keep accelerometer timestamps as the primary timeline because the
    current fall-feature pipeline is accelerometer-first.
    """
    base = acc_df.sort_values("timestamp").reset_index(drop=True).copy()

    def _merge_optional(left: pd.DataFrame, right: pd.DataFrame | None) -> pd.DataFrame:
        if right is None or right.empty:
            return left

        right_sorted = right.sort_values("timestamp").reset_index(drop=True)
        left_ts_hz = _estimate_sampling_rate_hz(left["timestamp"])
        right_ts_hz = _estimate_sampling_rate_hz(right_sorted["timestamp"])

        dt_candidates = []
        for hz in (left_ts_hz, right_ts_hz):
            if np.isfinite(hz) and hz > 0:
                dt_candidates.append(1.0 / hz)

        tolerance = max(dt_candidates) * 0.75 if dt_candidates else 0.02
        tolerance = max(float(tolerance), 0.005)

        return pd.merge_asof(
            left.sort_values("timestamp"),
            right_sorted.sort_values("timestamp"),
            on="timestamp",
            direction="nearest",
            tolerance=tolerance,
        )

    merged = _merge_optional(base, gyro_df)
    merged = _merge_optional(merged, ori_df)
    return merged.reset_index(drop=True)


def _load_mobifall_trial(
    trial_files: _TrialFiles,
    *,
    root_path: Path,
) -> pd.DataFrame:
    if trial_files.acc is None:
        raise ValueError("MobiFall trial must contain an accelerometer file")

    acc_df = _parse_mobifall_sensor_txt(trial_files.acc, sensor_kind="acc")
    gyro_df = (
        _parse_mobifall_sensor_txt(trial_files.gyro, sensor_kind="gyro")
        if trial_files.gyro is not None
        else None
    )
    ori_df = (
        _parse_mobifall_sensor_txt(trial_files.orientation, sensor_kind="orientation")
        if trial_files.orientation is not None
        else None
    )

    merged = _merge_streams_on_timestamp(acc_df, gyro_df=gyro_df, ori_df=ori_df)

    merged = normalize_sensor_units(
        merged,
        source_accel_unit="m_s2",
        source_gyro_unit=None,
        target_accel_unit="m_s2",
        target_gyro_unit="rad_s",
    )
    notes = list(merged.attrs.get("loader_notes", []))
    notes.append(
        "MobiFall accelerometer is currently treated as already aligned to m/s^2; "
        "gyroscope is left unconverted pending dataset audit."
    )
    merged.attrs["loader_notes"] = notes
    meta = _infer_mobifall_trial_metadata(trial_files.acc)
    for key, value in meta.items():
        merged[key] = value

    merged["source_file"] = str(trial_files.acc)
    merged["source_file_acc"] = str(trial_files.acc)
    merged["source_file_gyro"] = str(trial_files.gyro) if trial_files.gyro is not None else None
    merged["source_file_orientation"] = (
        str(trial_files.orientation) if trial_files.orientation is not None else None
    )

    merged["has_gyro"] = trial_files.gyro is not None
    merged["has_orientation"] = trial_files.orientation is not None
    merged["sampling_rate_hz"] = _estimate_sampling_rate_hz(merged["timestamp"])

    return merged


def _load_mobifall_directory(path: Path, *, max_files: int | None = None) -> pd.DataFrame:
    """
    Load all parseable MobiFall trials under a directory.

    max_files limits the number of *trials*, not raw txt files.
    """
    frames: list[pd.DataFrame] = []
    discovered = _discover_trial_groups(path)

    for idx, (_group_key, trial_files) in enumerate(discovered):
        if max_files is not None and idx >= max_files:
            break

        if trial_files.acc is None:
            continue

        try:
            frames.append(_load_mobifall_trial(trial_files, root_path=path))
        except Exception:
            continue

    if not frames:
        raise ValueError(f"No parseable MobiFall trials found under: {path}")

    return pd.concat(frames, ignore_index=True)


def _find_single_file_trial(src_path: Path) -> _TrialFiles:
    """
    Build a trial group from one provided sensor file by looking for siblings.
    """
    kind = _sensor_kind_from_name(src_path)
    if kind is None:
        raise ValueError(f"Not a recognised MobiFall sensor file: {src_path}")

    trial_key = _trial_key_from_file(src_path)
    candidates = list(src_path.parent.glob(f"{trial_key}*.txt"))

    trial_files = _TrialFiles()
    for path in sorted(candidates):
        candidate_kind = _sensor_kind_from_name(path)
        if candidate_kind == "acc":
            trial_files.acc = path
        elif candidate_kind == "gyro":
            trial_files.gyro = path
        elif candidate_kind == "orientation":
            trial_files.orientation = path

    # Ensure the originally requested file is preserved even if glob misses something unusual.
    if kind == "acc":
        trial_files.acc = src_path
    elif kind == "gyro":
        trial_files.gyro = src_path
    elif kind == "orientation":
        trial_files.orientation = src_path

    return trial_files


def load_mobifall(path: str | Path, *, validate: bool = True, max_files: int | None = None) -> pd.DataFrame:
    """
    Load MobiFall into the common schema.

    Supports:
    - full directory loading with trial grouping
    - single real sensor txt file loading with sibling pairing
    - simplified CSV fallback
    """
    src_path = Path(path)

    if src_path.is_dir():
        df = _load_mobifall_directory(src_path, max_files=max_files)
        df = normalize_columns(df)
    elif src_path.suffix.lower() == ".txt" and _sensor_kind_from_name(src_path) is not None:
        trial_files = _find_single_file_trial(src_path)
        df = _load_mobifall_trial(trial_files, root_path=src_path.parent)
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
        raise ValueError("MobiFall loader requires 'subject_id' and 'label_raw' columns in simplified CSV")

    apply_label_mapping(df, task_type=TASK_FALL)
    inject_metadata(
        df,
        dataset_name="MOBIFALL",
        task_type=TASK_FALL,
        source_file=src_path,
        placement="pocket",
        sampling_rate_hz=87.0,
    )

    return finalize_ingest_dataframe(df, validate=validate)