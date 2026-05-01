"""MobiAct v2.0 loader (external cross-dataset evaluation, Chapter 3 / Chapter 8).

Scope:
- read the Annotated Data half of the MobiAct v2 release
  (data/raw/MobiAct_Dataset_v2.0/Annotated Data/<ACTIVITY>/<trial>.csv)
- emit the same canonical schema as the existing fall loaders
- map labels per-row using the in-file ``label`` column rather than the folder
  name, because scenario files (e.g. SBE, SBW) contain mixed ADL labels and
  the fall-trial files contain pre-fall STD and post-fall LYI segments

Unit assumptions (verified against the released CSVs):
- accelerometer is already in m/s^2 (resting az ~= 9.8)
- gyroscope is already in rad/s (sub-radian magnitudes during static periods)
- sampling rate is ~200 Hz, derived per-trial from ``rel_time`` so no hard-code
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Iterable, Literal

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
from pipeline.schema import TASK_FALL, TASK_HAR

# MobiAct v2 fall-trial label codes that mark the fall event itself.
# Source: Vavoulas et al., "The MobiAct Dataset: Recognition of Activities of
# Daily Living using Smartphones" — fall types are FOL, FKL, SDL, BSC.
# Other codes such as LYI (post-fall lying), STD (pre-fall standing), and
# JUM/SBE/SBW (scenario or ADL) are non-fall under the binary task.
MOBIACT_V2_FALL_CODES: frozenset[str] = frozenset({"FKL", "FOL", "SDL", "BSC"})

# Map MobiAct v2 activity codes to tokens that pipeline.ingest.common.map_har_label
# recognises (case-insensitive exact match, then keyword fallbacks). Without this
# expansion, codes like "STD" would not match the substring rule "stand" and would
# fall through to "other".
# - locomotion: walking, jogging
# - static: sitting, standing, lying-related
# - stairs: stair ascent / descent
# - other: jumps, transitions, fall codes (under task=har they are *not* HAR
#   activities — the meta-classifier is asked to recognise locomotion/static/stairs)
_MOBIACT_V2_HAR_EXPANSION: dict[str, str] = {
    "WAL": "walking",
    "JOG": "jogging",
    "STD": "standing",
    "SIT": "sitting",
    "LYI": "lying",
    "STU": "ascending_stairs",
    "STN": "descending_stairs",
    "JUM": "other",
    "CHU": "other",
    "CSI": "other",
    "CSO": "other",
    "SCH": "other",
    "SLH": "lying",
    "SLW": "lying",
    "SRH": "other",
    # Fall codes are not HAR activities — collapse to "other" under task=har.
    "FKL": "other",
    "FOL": "other",
    "SDL": "other",
    "BSC": "other",
    "SBE": "other",
    "SBW": "other",
}

_FILENAME_RE = re.compile(
    r"^(?P<activity>[A-Z]{3})_(?P<subject>\d+)_(?P<trial>\d+)_annotated\.csv$",
    flags=re.IGNORECASE,
)


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


def _parse_filename(file_path: Path) -> dict[str, str]:
    match = _FILENAME_RE.match(file_path.name)
    if match is None:
        # Fall back to folder name + stem so we still emit something usable.
        return {
            "activity_code": file_path.parent.name.upper(),
            "subject_token": file_path.stem,
            "trial": "0",
        }
    return {
        "activity_code": match.group("activity").upper(),
        "subject_token": match.group("subject"),
        "trial": match.group("trial"),
    }


def _read_annotated_csv(file_path: Path) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    expected = {
        "rel_time",
        "acc_x", "acc_y", "acc_z",
        "gyro_x", "gyro_y", "gyro_z",
        "label",
    }
    missing = expected.difference(df.columns)
    if missing:
        raise ValueError(
            f"MobiAct v2 file {file_path} missing required columns: {sorted(missing)}"
        )

    out = pd.DataFrame(
        {
            "timestamp": pd.to_numeric(df["rel_time"], errors="coerce"),
            "ax": pd.to_numeric(df["acc_x"], errors="coerce"),
            "ay": pd.to_numeric(df["acc_y"], errors="coerce"),
            "az": pd.to_numeric(df["acc_z"], errors="coerce"),
            "gx": pd.to_numeric(df["gyro_x"], errors="coerce"),
            "gy": pd.to_numeric(df["gyro_y"], errors="coerce"),
            "gz": pd.to_numeric(df["gyro_z"], errors="coerce"),
            "label_raw": df["label"].astype(str).str.strip().str.upper(),
        }
    )

    # MobiAct v2 ships m/s^2 for accel and rad/s for gyro; no conversion needed.
    # Run normalize_sensor_units anyway so downstream code can rely on the
    # ``unit_normalization`` attrs being populated consistently.
    out = normalize_sensor_units(
        out,
        source_accel_unit="m_s2",
        source_gyro_unit="rad_s",
        target_accel_unit="m_s2",
        target_gyro_unit="rad_s",
    )
    notes = list(out.attrs.get("loader_notes", []))
    notes.append(
        "MobiAct v2 Annotated CSVs ship accelerometer in m/s^2 and gyroscope in rad/s; "
        "no unit conversion applied."
    )
    out.attrs["loader_notes"] = notes
    return out


def _label_raw_for_binary(activity_code: str, per_row_label: str) -> str:
    """Return a coarse per-row label suitable for the shared fall mapper.

    Strategy:
    - rows whose in-file label is one of the four fall codes => "fall"
    - everything else => "non_fall" (this includes pre-fall STD, post-fall LYI,
      scenario ADLs such as WAL/JOG/JUM/SIT, and transitions)
    The activity_code argument is unused at the row level but kept for future
    sanity audits (e.g. flagging rows where the trial folder claims FKL but
    the row label disagrees).
    """
    return "fall" if per_row_label in MOBIACT_V2_FALL_CODES else "non_fall"


def _label_raw_for_har(per_row_label: str) -> str:
    """Expand a MobiAct v2 activity code to a token map_har_label() can match."""
    return _MOBIACT_V2_HAR_EXPANSION.get(per_row_label, "other")


def _load_single_trial(file_path: Path, *, task: Literal["fall", "har"] = "fall") -> pd.DataFrame:
    meta = _parse_filename(file_path)
    df = _read_annotated_csv(file_path)

    if task == "har":
        df["label_raw"] = [_label_raw_for_har(code) for code in df["label_raw"]]
    else:
        # The shared fall label mapper accepts free-form raw labels and falls back
        # to non_fall for unknowns; we pre-collapse here so the audit trail
        # ("label_raw") is informative without being noisy.
        df["label_raw"] = [
            _label_raw_for_binary(meta["activity_code"], code)
            for code in df["label_raw"]
        ]

    subject_id = f"sub_{int(meta['subject_token']):02d}" if meta["subject_token"].isdigit() else meta["subject_token"]
    df["subject_id"] = subject_id
    df["session_id"] = f"{meta['activity_code']}_{meta['subject_token']}_{meta['trial']}"
    df["sampling_rate_hz"] = _estimate_sampling_rate_hz(df["timestamp"])
    return df


def _iter_annotated_csv_files(root_path: Path) -> Iterable[Path]:
    """Yield CSVs round-robin across activity folders.

    A purely alphabetical traversal biases small ``max_files`` smokes toward
    a single activity folder (BSC, ~191 files), giving only fall+static
    labels and missing locomotion/stairs entirely. Round-robin across
    activity directories keeps smoke samples representative for both fall
    and HAR tasks.
    """
    activity_dirs = sorted(p for p in root_path.iterdir() if p.is_dir())
    if not activity_dirs:
        # Fall back to a flat scan (e.g. when a single-activity directory is passed).
        for path in sorted(root_path.rglob("*_annotated.csv")):
            if path.is_file():
                yield path
        return

    file_lists = [
        sorted(d.glob("*_annotated.csv")) for d in activity_dirs
    ]
    iterators = [iter(files) for files in file_lists]
    while iterators:
        next_round: list = []
        for it in iterators:
            try:
                yield next(it)
            except StopIteration:
                continue
            next_round.append(it)
        iterators = next_round


def _load_annotated_directory(
    path: Path,
    *,
    max_files: int | None = None,
    task: Literal["fall", "har"] = "fall",
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    parsed_count = 0
    for file_path in _iter_annotated_csv_files(path):
        if max_files is not None and parsed_count >= max_files:
            break
        try:
            frames.append(_load_single_trial(file_path, task=task))
            parsed_count += 1
        except Exception:
            continue
    if not frames:
        raise ValueError(f"No parseable MobiAct v2 trials found under: {path}")
    return pd.concat(frames, ignore_index=True)


def load_mobiact_v2(
    path: str | Path,
    *,
    validate: bool = True,
    max_files: int | None = None,
    task: Literal["fall", "har"] = "fall",
) -> pd.DataFrame:
    """Load MobiAct v2 Annotated Data into the common ingestion schema.

    Parameters
    ----------
    path
        Either the ``Annotated Data`` directory (recursive trial discovery)
        or a single ``*_annotated.csv`` file. A simplified CSV with
        ``subject_id`` / ``label_raw`` columns is also accepted as a fallback
        for fixture-driven tests.
    validate
        When True (default), runs the shared ingest validator before return.
    max_files
        Optional limit on the number of trial CSVs loaded — used for smoke
        tests and the ``--max-files`` flag in the eval wrapper.
    task
        ``"fall"`` (default) routes the per-row label through the binary
        fall/non_fall mapper. ``"har"`` expands MobiAct activity codes to the
        canonical HAR taxonomy ``{static, locomotion, stairs, other}``.
    """
    if task not in ("fall", "har"):
        raise ValueError(f"task must be 'fall' or 'har', got: {task!r}")

    src_path = Path(path)

    if src_path.is_dir():
        df = _load_annotated_directory(src_path, max_files=max_files, task=task)
        df = normalize_columns(df)
    elif src_path.suffix.lower() == ".csv" and src_path.name.endswith("_annotated.csv"):
        df = _load_single_trial(src_path, task=task)
        df = normalize_columns(df)
    else:
        # Simplified CSV fallback for tests / manual fixtures.
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
        raise ValueError(
            "MobiAct v2 loader requires 'subject_id' and 'label_raw' columns "
            "in simplified CSV"
        )

    task_type = TASK_FALL if task == "fall" else TASK_HAR
    apply_label_mapping(df, task_type=task_type)
    inject_metadata(
        df,
        dataset_name="MOBIACT_V2",
        task_type=task_type,
        source_file=src_path,
        placement="pocket",
        sampling_rate_hz=200.0,
    )

    return finalize_ingest_dataframe(df, validate=validate)
