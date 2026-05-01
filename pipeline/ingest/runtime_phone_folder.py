from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pipeline.ingest.common import normalize_columns, read_csv_any
from pipeline.preprocess.units import normalize_sensor_units

PLACEMENT_MAP = {
    "in_pocket": "pocket",
    "pocket": "pocket",
    "in_hand": "hand",
    "hand": "hand",
    "arm_mounted": "hand",
    "on_surface": "desk",
    "desk": "desk",
    "bag": "bag",
    "unknown": "unknown",
}


def _normalise_placement(value: object, default: str = "unknown") -> str:
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except TypeError:
        pass

    label = str(value).strip().lower().replace(" ", "_").replace("-", "_")
    return PLACEMENT_MAP.get(label, label if label else default)

@dataclass(slots=True)
class RuntimePhoneFolderConfig:
    dataset_name: str = "PHONE_RUNTIME"
    task_type: str = "runtime"
    subject_id: str = "phone_subject"
    session_id: str | None = None
    placement: str = "pocket"

    # For your current iPhone export, these are the correct defaults.
    accel_unit: str = "m_s2"
    gyro_unit: str = "rad_s"

    # Prefer relative time from seconds_elapsed when present.
    prefer_seconds_elapsed: bool = True
    merge_tolerance_seconds: float = 0.02  # ~20ms
    sort_by_timestamp: bool = True


def _load_csv(path: Path) -> pd.DataFrame:
    df = read_csv_any(path)
    return normalize_columns(df)


def _find_file(folder: Path, filename: str) -> Path | None:
    candidate = folder / filename
    return candidate if candidate.exists() else None


def _parse_metadata(folder: Path) -> dict[str, Any]:
    meta_path = _find_file(folder, "Metadata.csv")
    if meta_path is None:
        return {}

    meta_df = _load_csv(meta_path)
    if meta_df.empty:
        return {}

    row = meta_df.iloc[0].to_dict()
    return {str(k): v for k, v in row.items()}


def _extract_nominal_rate_hz(metadata: dict[str, Any]) -> float | None:
    # sampleRateMs looks like: 10|10|10|10||10|10
    raw = metadata.get("sampleratems")
    if raw is None:
        raw = metadata.get("sample_rate_ms")
    if raw is None:
        return None

    parts = [p for p in str(raw).split("|") if str(p).strip()]
    numeric: list[float] = []
    for part in parts:
        try:
            ms = float(part)
            if ms > 0:
                numeric.append(ms)
        except Exception:
            continue

    if not numeric:
        return None

    # Use the first available nominal rate.
    return float(1000.0 / numeric[0])


def _ensure_relative_timestamp(df: pd.DataFrame, *, prefer_seconds_elapsed: bool) -> pd.Series:
    if prefer_seconds_elapsed and "seconds_elapsed" in df.columns:
        ts = pd.to_numeric(df["seconds_elapsed"], errors="coerce")
        if ts.notna().any():
            return ts.astype(float)

    if "time" not in df.columns:
        raise ValueError("Sensor CSV must contain either seconds_elapsed or time")

    raw_time = pd.to_numeric(df["time"], errors="coerce")
    if raw_time.dropna().empty:
        raise ValueError("No usable timestamps found in sensor CSV")

    # Your current files appear to use nanoseconds since epoch in `time`.
    median_abs = float(raw_time.dropna().abs().median())
    if median_abs >= 1e17:
        ts = raw_time / 1e9
    elif median_abs >= 1e14:
        ts = raw_time / 1e6
    elif median_abs >= 1e11:
        ts = raw_time / 1e3
    else:
        ts = raw_time.astype(float)

    first = ts.dropna().iloc[0]
    return (ts - float(first)).astype(float)


def _prepare_accel_df(path: Path, *, prefer_seconds_elapsed: bool) -> pd.DataFrame:
    df = _load_csv(path)

    required = {"x", "y", "z"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Accelerometer.csv missing required columns: {sorted(missing)}")

    out = pd.DataFrame()
    out["timestamp"] = _ensure_relative_timestamp(df, prefer_seconds_elapsed=prefer_seconds_elapsed)
    out["ax"] = pd.to_numeric(df["x"], errors="coerce")
    out["ay"] = pd.to_numeric(df["y"], errors="coerce")
    out["az"] = pd.to_numeric(df["z"], errors="coerce")
    return out


def _prepare_gyro_df(path: Path, *, prefer_seconds_elapsed: bool) -> pd.DataFrame:
    df = _load_csv(path)

    required = {"x", "y", "z"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Gyroscope.csv missing required columns: {sorted(missing)}")

    out = pd.DataFrame()
    out["timestamp"] = _ensure_relative_timestamp(df, prefer_seconds_elapsed=prefer_seconds_elapsed)
    out["gx"] = pd.to_numeric(df["x"], errors="coerce")
    out["gy"] = pd.to_numeric(df["y"], errors="coerce")
    out["gz"] = pd.to_numeric(df["z"], errors="coerce")
    return out


def _infer_sampling_rate_from_timestamp(ts: pd.Series) -> float | None:
    vals = pd.to_numeric(ts, errors="coerce").dropna().to_numpy(dtype=float)
    if vals.size < 3:
        return None

    diffs = np.diff(vals)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if diffs.size == 0:
        return None

    median_dt = float(np.median(diffs))
    if median_dt <= 0:
        return None
    return float(1.0 / median_dt)


def _merge_sensor_streams(
    accel_df: pd.DataFrame,
    gyro_df: pd.DataFrame,
    *,
    tolerance_seconds: float,
) -> pd.DataFrame:
    accel_df = accel_df.sort_values("timestamp", kind="stable").reset_index(drop=True)
    gyro_df = gyro_df.sort_values("timestamp", kind="stable").reset_index(drop=True)

    merged = pd.merge_asof(
        accel_df,
        gyro_df,
        on="timestamp",
        direction="nearest",
        tolerance=float(tolerance_seconds),
    )

    for col in ["gx", "gy", "gz"]:
        if col not in merged.columns:
            merged[col] = np.nan

    return merged


def load_runtime_phone_folder(
    folder_path: str | Path,
    *,
    config: RuntimePhoneFolderConfig | None = None,
) -> pd.DataFrame:
    """Load a phone-export folder and return canonical replay schema."""
    cfg = config or RuntimePhoneFolderConfig()
    folder = Path(folder_path)

    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"Runtime phone folder not found or not a directory: {folder}")

    accel_path = _find_file(folder, "Accelerometer.csv")
    gyro_path = _find_file(folder, "Gyroscope.csv")

    if accel_path is None:
        raise FileNotFoundError(f"Missing Accelerometer.csv in folder: {folder}")
    if gyro_path is None:
        raise FileNotFoundError(f"Missing Gyroscope.csv in folder: {folder}")

    metadata = _parse_metadata(folder)

    accel_df = _prepare_accel_df(accel_path, prefer_seconds_elapsed=cfg.prefer_seconds_elapsed)
    gyro_df = _prepare_gyro_df(gyro_path, prefer_seconds_elapsed=cfg.prefer_seconds_elapsed)

    merged = _merge_sensor_streams(
        accel_df,
        gyro_df,
        tolerance_seconds=cfg.merge_tolerance_seconds,
    )

    # Remove rows with no usable accel sample.
    merged = merged.dropna(subset=["timestamp", "ax", "ay", "az"]).reset_index(drop=True)

    # Normalize onto canonical physical units.
    merged = normalize_sensor_units(
        merged,
        source_accel_unit=cfg.accel_unit,
        source_gyro_unit=cfg.gyro_unit,
        target_accel_unit="m_s2",
        target_gyro_unit="rad_s",
    )

    session_id = cfg.session_id or folder.name
    sampling_rate_hz = _extract_nominal_rate_hz(metadata)
    if sampling_rate_hz is None:
        sampling_rate_hz = _infer_sampling_rate_from_timestamp(merged["timestamp"])

    merged["dataset_name"] = cfg.dataset_name
    merged["subject_id"] = cfg.subject_id
    merged["session_id"] = session_id
    merged["task_type"] = cfg.task_type
    merged["placement"] = _normalise_placement(cfg.placement)
    merged["sampling_rate_hz"] = sampling_rate_hz if sampling_rate_hz is not None else np.nan
    merged["source_file"] = str(folder)
    merged["label_raw"] = pd.NA
    merged["label_mapped"] = pd.NA

    for col in ["dataset_name", "subject_id", "session_id", "task_type", "placement"]:
        if col in merged.columns:
            merged[col] = merged[col].astype("string").str.strip()

    merged["row_index"] = np.arange(len(merged), dtype=int)

    if cfg.sort_by_timestamp:
        merged = merged.sort_values("timestamp", kind="stable").reset_index(drop=True)

    merged.attrs["phone_folder_metadata"] = metadata
    notes = list(merged.attrs.get("loader_notes", []))
    notes.append("Runtime phone folder adapter merged calibrated Accelerometer.csv and Gyroscope.csv.")
    notes.append("Timestamp uses seconds_elapsed when available, otherwise normalized absolute time.")
    notes.append(f"Accelerometer source unit set to: {cfg.accel_unit}")
    notes.append(f"Gyroscope source unit set to: {cfg.gyro_unit}")


    if sampling_rate_hz is not None:
        notes.append(f"Sampling rate inferred/set to approximately: {sampling_rate_hz:.3f} Hz")
    merged.attrs["loader_notes"] = notes


    for col in ["dataset_name", "subject_id", "session_id", "task_type", "placement"]:
        if col in merged.columns:
            merged[col] = merged[col].astype("string").str.strip()

    keep_order = [
        "timestamp",
        "ax",
        "ay",
        "az",
        "gx",
        "gy",
        "gz",
        "row_index",
        "dataset_name",
        "subject_id",
        "session_id",
        "task_type",
        "placement",
        "sampling_rate_hz",
        "source_file",
        "label_raw",
        "label_mapped",
    ]
    ordered_existing = [c for c in keep_order if c in merged.columns]
    remaining = [c for c in merged.columns if c not in ordered_existing]
    merged = merged[ordered_existing + remaining].reset_index(drop=True)

    return merged


def export_runtime_phone_folder_csv(
    folder_path: str | Path,
    output_path: str | Path,
    *,
    config: RuntimePhoneFolderConfig | None = None,
) -> Path:
    """Load a phone-export folder, normalize it, and save as canonical CSV."""
    df = load_runtime_phone_folder(folder_path, config=config)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return out_path