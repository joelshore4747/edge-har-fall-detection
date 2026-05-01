from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from pipeline.ingest.common import normalize_columns, read_csv_any
from pipeline.preprocess.units import (
    accel_magnitude,
    gyro_magnitude,
    normalize_sensor_units,
)

DEFAULT_PHONE_COLUMN_ALIASES: dict[str, list[str]] = {
    "timestamp": [
        "timestamp",
        "time",
        "ts",
        "t",
        "unix_time",
        "unix_timestamp",
        "epoch",
        "epoch_s",
        "epoch_ms",
        "epoch_us",
        "epoch_ns",
        "elapsed_seconds",
        "elapsed_time",
    ],
    "ax": [
        "ax",
        "accel_x",
        "acc_x",
        "accelerometer_x",
        "acceleration_x",
        "lin_acc_x",
        "linear_acceleration_x",
        "user_acc_x",
        "user_accel_x",
        "user_acceleration_x",
        "x_acceleration",
        "x_accel",
    ],
    "ay": [
        "ay",
        "accel_y",
        "acc_y",
        "accelerometer_y",
        "acceleration_y",
        "lin_acc_y",
        "linear_acceleration_y",
        "user_acc_y",
        "user_accel_y",
        "user_acceleration_y",
        "y_acceleration",
        "y_accel",
    ],
    "az": [
        "az",
        "accel_z",
        "acc_z",
        "accelerometer_z",
        "acceleration_z",
        "lin_acc_z",
        "linear_acceleration_z",
        "user_acc_z",
        "user_accel_z",
        "user_acceleration_z",
        "z_acceleration",
        "z_accel",
    ],
    "gx": [
        "gx",
        "gyro_x",
        "gyroscope_x",
        "rotation_rate_x",
        "rot_x",
        "omega_x",
        "x_rotation_rate",
        "x_gyro",
    ],
    "gy": [
        "gy",
        "gyro_y",
        "gyroscope_y",
        "rotation_rate_y",
        "rot_y",
        "omega_y",
        "y_rotation_rate",
        "y_gyro",
    ],
    "gz": [
        "gz",
        "gyro_z",
        "gyroscope_z",
        "rotation_rate_z",
        "rot_z",
        "omega_z",
        "z_rotation_rate",
        "z_gyro",
    ],
    "label_raw": [
        "label_raw",
        "label",
        "activity",
        "activity_label",
        "class",
        "annotation",
        "event",
    ],
    "subject_id": [
        "subject_id",
        "subject",
        "participant",
        "participant_id",
        "user_id",
        "user",
    ],
    "session_id": [
        "session_id",
        "session",
        "recording_id",
        "trial_id",
        "clip_id",
        "record_id",
    ],
    "placement": [
        "placement",
        "phone_placement",
        "device_placement",
        "location",
        "carry_position",
    ],
}

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
class RuntimePhoneCsvConfig:
    dataset_name: str = "PHONE_RUNTIME"
    task_type: str = "runtime"
    subject_id: str = "phone_subject"
    session_id: str = "phone_session"
    placement: str = "pocket"

    accel_unit: str = "auto"  # auto | g | m_s2
    gyro_unit: str = "auto"   # auto | rad_s | deg_s
    timestamp_unit: str = "auto"  # auto | s | ms | us | ns

    sampling_rate_hz: float | None = None

    sort_by_timestamp: bool = True
    make_timestamp_relative: bool = True

    column_aliases: dict[str, list[str]] = field(
        default_factory=lambda: {k: list(v) for k, v in DEFAULT_PHONE_COLUMN_ALIASES.items()}
    )


def _first_matching_column(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    available = {str(c): str(c) for c in columns}
    for candidate in candidates:
        if candidate in available:
            return available[candidate]
    return None


def _rename_phone_columns(df: pd.DataFrame, config: RuntimePhoneCsvConfig) -> pd.DataFrame:
    rename_map: dict[str, str] = {}
    for canonical, aliases in config.column_aliases.items():
        match = _first_matching_column(df.columns, aliases)
        if match is not None and match != canonical:
            rename_map[match] = canonical

    out = df.rename(columns=rename_map).copy()

    # Preserve canonical names if already present.
    for canonical in config.column_aliases:
        if canonical in df.columns and canonical not in out.columns:
            out[canonical] = df[canonical]

    return out


def _coerce_numeric_columns(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _infer_timestamp_scale(ts: pd.Series) -> str:
    vals = pd.to_numeric(ts, errors="coerce").dropna()
    if vals.empty:
        return "s"

    median_abs = float(vals.abs().median())
    if median_abs >= 1e17:
        return "ns"
    if median_abs >= 1e14:
        return "us"
    if median_abs >= 1e11:
        return "ms"
    return "s"


def _normalize_timestamp(ts: pd.Series, timestamp_unit: str, *, make_relative: bool) -> pd.Series:
    vals = pd.to_numeric(ts, errors="coerce")

    unit = timestamp_unit
    if unit == "auto":
        unit = _infer_timestamp_scale(vals)

    if unit == "ns":
        vals = vals / 1e9
    elif unit == "us":
        vals = vals / 1e6
    elif unit == "ms":
        vals = vals / 1e3
    elif unit == "s":
        pass
    else:
        raise ValueError(f"Unsupported timestamp_unit: {timestamp_unit}")

    if make_relative:
        first = vals.dropna().iloc[0] if vals.dropna().size > 0 else None
        if first is not None:
            vals = vals - float(first)

    return vals.astype(float)


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


def _infer_accel_unit(df: pd.DataFrame) -> str:
    mag = accel_magnitude(df).dropna()
    if mag.empty:
        return "m_s2"

    q50 = float(mag.quantile(0.50))
    q99 = float(mag.quantile(0.99))

    if q50 < 3.0 and q99 < 8.0:
        return "g"
    return "m_s2"


def _infer_gyro_unit(df: pd.DataFrame) -> str:
    mag = gyro_magnitude(df).dropna()
    if mag.empty:
        return "rad_s"

    q99 = float(mag.quantile(0.99))
    if q99 < 20.0:
        return "rad_s"
    return "deg_s"


def _ensure_sensor_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["ax", "ay", "az", "gx", "gy", "gz"]:
        if col not in out.columns:
            out[col] = np.nan
    return out


def _inject_runtime_metadata(df: pd.DataFrame, config: RuntimePhoneCsvConfig, source_file: Path) -> pd.DataFrame:
    out = df.copy()

    if "dataset_name" not in out.columns:
        out["dataset_name"] = config.dataset_name
    else:
        out["dataset_name"] = out["dataset_name"].fillna(config.dataset_name)

    if "subject_id" not in out.columns:
        out["subject_id"] = config.subject_id
    else:
        out["subject_id"] = out["subject_id"].fillna(config.subject_id)

    if "session_id" not in out.columns:
        out["session_id"] = config.session_id
    else:
        out["session_id"] = out["session_id"].fillna(config.session_id)

    if "task_type" not in out.columns:
        out["task_type"] = config.task_type
    else:
        out["task_type"] = out["task_type"].fillna(config.task_type)

    if "placement" not in out.columns:
        out["placement"] = config.placement
    else:
        out["placement"] = out["placement"].fillna(config.placement)

    out["placement"] = out["placement"].map(_normalise_placement)

    if "source_file" not in out.columns:
        out["source_file"] = str(source_file)
    else:
        out["source_file"] = out["source_file"].fillna(str(source_file))

    return out


def load_runtime_phone_csv(
    path: str | Path,
    *,
    config: RuntimePhoneCsvConfig | None = None,
) -> pd.DataFrame:
    """Load a raw phone-export CSV and return the normalized project schema."""
    cfg = config or RuntimePhoneCsvConfig()
    src_path = Path(path)

    if not src_path.exists():
        raise FileNotFoundError(f"Runtime phone CSV not found: {src_path}")

    df = read_csv_any(src_path)
    df = normalize_columns(df)
    df = _rename_phone_columns(df, cfg)
    df = _ensure_sensor_columns(df)
    df = _coerce_numeric_columns(df, ["timestamp", "ax", "ay", "az", "gx", "gy", "gz"])

    if "timestamp" not in df.columns or df["timestamp"].dropna().empty:
        if cfg.sampling_rate_hz is None:
            raise ValueError("CSV has no usable timestamp column and no sampling_rate_hz was provided")
        df["timestamp"] = np.arange(len(df), dtype=float) / float(cfg.sampling_rate_hz)
    else:
        df["timestamp"] = _normalize_timestamp(
            df["timestamp"],
            cfg.timestamp_unit,
            make_relative=cfg.make_timestamp_relative,
        )

    if cfg.sort_by_timestamp and "timestamp" in df.columns:
        df = df.sort_values("timestamp", kind="stable").reset_index(drop=True)

    inferred_sampling_rate = _infer_sampling_rate_from_timestamp(df["timestamp"])
    sampling_rate_hz = float(cfg.sampling_rate_hz) if cfg.sampling_rate_hz is not None else inferred_sampling_rate
    if sampling_rate_hz is None:
        sampling_rate_hz = np.nan
    df["sampling_rate_hz"] = sampling_rate_hz

    if "label_raw" in df.columns and "label_mapped" not in df.columns:
        df["label_mapped"] = pd.NA

    accel_unit = cfg.accel_unit
    if accel_unit == "auto":
        accel_unit = _infer_accel_unit(df)

    gyro_unit = cfg.gyro_unit
    if gyro_unit == "auto":
        gyro_unit = _infer_gyro_unit(df)

    df = normalize_sensor_units(
        df,
        source_accel_unit=accel_unit,
        source_gyro_unit=gyro_unit,
        target_accel_unit="m_s2",
        target_gyro_unit="rad_s",
    )

    df = _inject_runtime_metadata(df, cfg, src_path)

    if "label_raw" not in df.columns:
        df["label_raw"] = pd.NA
    if "label_mapped" not in df.columns:
        df["label_mapped"] = pd.NA

    for col in ["dataset_name", "subject_id", "session_id", "task_type", "placement", "label_raw"]:
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip()

    notes = list(df.attrs.get("loader_notes", []))
    notes.append("Runtime phone CSV adapter normalized raw export into canonical replay schema.")
    notes.append(f"Accelerometer source unit inferred/set to: {accel_unit}")
    notes.append(f"Gyroscope source unit inferred/set to: {gyro_unit}")
    notes.append("Timestamps are stored in seconds and are relative to session start when enabled.")
    df.attrs["loader_notes"] = notes
    df["row_index"] = np.arange(len(df), dtype=int)
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
    ordered_existing = [c for c in keep_order if c in df.columns]
    remaining = [c for c in df.columns if c not in ordered_existing]
    df = df[ordered_existing + remaining].reset_index(drop=True)

    return df



def export_runtime_phone_csv(
    input_path: str | Path,
    output_path: str | Path,
    *,
    config: RuntimePhoneCsvConfig | None = None,
) -> Path:
    df = load_runtime_phone_csv(input_path, config=config)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return out_path