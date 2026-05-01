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
from pipeline.schema import TASK_HAR


WISDM_ACTIVITY_LABELS: dict[int, str] = {
    0: "downstairs",
    1: "jogging",
    2: "sitting",
    3: "standing",
    4: "upstairs",
    5: "walking",
}

WISDM_NOMINAL_SAMPLING_RATE_HZ = 20.0
_WISDM_SESSION_GAP_SECONDS = 1.0


def _resolve_wisdm_file(path: Path, *, split: str | None = None) -> tuple[Path, str]:
    if path.is_file():
        split_name = split or path.stem.lower()
        return path, split_name

    if not path.is_dir():
        raise ValueError(f"WISDM path does not exist: {path}")

    requested = (split or "").strip().lower()
    if requested in {"", "all"}:
        raise ValueError("Directory resolution requires an explicit split file ('train' or 'test').")

    candidate = path / f"{requested}.csv"
    if not candidate.exists():
        raise ValueError(f"Could not find WISDM split file under {path}: {requested}.csv")
    return candidate, requested


def _map_activity_code(raw_value: object) -> str:
    try:
        numeric = int(float(raw_value))
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Unsupported WISDM Activity value: {raw_value!r}") from exc

    if numeric not in WISDM_ACTIVITY_LABELS:
        raise ValueError(f"Unknown WISDM Activity code: {numeric}")
    return WISDM_ACTIVITY_LABELS[numeric]


def _estimate_sampling_rate_hz(timestamp_seconds: pd.Series) -> float:
    ts = pd.to_numeric(timestamp_seconds, errors="coerce").dropna()
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


def _summarize_sampling_rates(df: pd.DataFrame) -> dict[str, float | int]:
    rates = pd.to_numeric(df.get("sampling_rate_hz"), errors="coerce").dropna()
    if rates.empty:
        return {
            "count": 0,
            "min_hz": float("nan"),
            "median_hz": float("nan"),
            "max_hz": float("nan"),
        }

    return {
        "count": int(len(rates)),
        "min_hz": float(rates.min()),
        "median_hz": float(rates.median()),
        "max_hz": float(rates.max()),
    }


def _build_session_ids(
    df: pd.DataFrame,
    *,
    split_name: str,
    split_on_label_change: bool = True,
) -> tuple[pd.DataFrame, dict[str, int | float]]:
    working = df.copy()

    original_rows = int(len(working))
    working["timestamp"] = pd.to_numeric(working["timestamp"], errors="coerce")
    working = working.dropna(subset=["timestamp", "label_raw", "ax", "ay", "az"]).reset_index(drop=True)

    dropped_rows = original_rows - len(working)
    if working.empty:
        raise ValueError("WISDM frame is empty after numeric coercion")

    prev_ts = working["timestamp"].shift()
    prev_label = working["label_raw"].shift()
    diffs = working["timestamp"].diff()

    session_start = (
        prev_ts.isna()
        | diffs.le(0.0)
        | diffs.gt(_WISDM_SESSION_GAP_SECONDS)
    )

    if split_on_label_change:
        session_start = session_start | working["label_raw"].ne(prev_label)

    session_number = session_start.cumsum().astype(int)

    working["_session_number"] = session_number
    working["session_id"] = session_number.map(lambda n: f"{split_name}_session_{int(n):05d}")

    working["timestamp"] = (
        working["timestamp"]
        - working.groupby("_session_number", dropna=False, sort=False)["timestamp"].transform("first")
    )

    rate_by_session = (
        working.groupby("_session_number", dropna=False, sort=False)["timestamp"]
        .apply(_estimate_sampling_rate_hz)
        .to_dict()
    )

    working["sampling_rate_hz"] = working["_session_number"].map(rate_by_session).astype(float)

    fallback_mask = working["sampling_rate_hz"].isna()
    fallback_rows = int(fallback_mask.sum())

    working["sampling_rate_hz"] = working["sampling_rate_hz"].fillna(WISDM_NOMINAL_SAMPLING_RATE_HZ)

    stats = {
        "original_rows": int(original_rows),
        "rows_after_dropna": int(len(working)),
        "dropped_rows_after_numeric_coercion": int(dropped_rows),
        "num_sessions": int(working["session_id"].nunique()),
        "sampling_rate_fallback_rows": int(fallback_rows),
    }

    working = working.drop(columns=["_session_number"])
    return working, stats


def _load_wisdm_csv(
    path: Path,
    *,
    split_name: str,
    validate: bool = True,
    max_sessions: int | None = None,
    split_on_label_change: bool = True,
) -> pd.DataFrame:
    df = read_csv_any(path)
    df = normalize_columns(df)

    rename_map = {
        "acc_x": "ax",
        "acc_y": "ay",
        "acc_z": "az",
        "activity": "activity_code",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    required = {"ax", "ay", "az", "timestamp", "activity_code"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"WISDM loader missing required columns: {sorted(missing)}")

    raw_rows = int(len(df))

    for col in ["ax", "ay", "az", "timestamp"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["label_raw"] = df["activity_code"].apply(_map_activity_code).astype("string")

    # This export does not contain stable person IDs.
    # Preserve split identity but explicitly mark that this is not a true subject identifier.
    df["subject_id"] = f"split_{split_name}"
    df["source_file"] = str(path)
    df["row_index"] = np.arange(len(df), dtype=int)

    df, session_stats = _build_session_ids(
        df,
        split_name=split_name,
        split_on_label_change=split_on_label_change,
    )

    if max_sessions is not None and max_sessions > 0:
        keep_sessions = (
            df["session_id"].dropna().astype(str).drop_duplicates().tolist()[: int(max_sessions)]
        )
        df = df[df["session_id"].astype(str).isin(keep_sessions)].reset_index(drop=True)
        df["row_index"] = np.arange(len(df), dtype=int)

    raw_label_counts = df["label_raw"].value_counts(dropna=False).to_dict()

    apply_label_mapping(df, task_type=TASK_HAR)
    mapped_label_counts = df["label_mapped"].value_counts(dropna=False).to_dict()

    inject_metadata(
        df,
        dataset_name="WISDM",
        task_type=TASK_HAR,
        source_file=path,
        placement="pocket",
        sampling_rate_hz=WISDM_NOMINAL_SAMPLING_RATE_HZ,
    )

    out = finalize_ingest_dataframe(df, validate=validate)

    session_lengths = out.groupby("session_id", dropna=False).size() if "session_id" in out.columns else pd.Series(dtype=int)
    sampling_stats = _summarize_sampling_rates(out)

    out.attrs["loader_notes"] = [
        "WISDM loader expects the checked-in train.csv/test.csv accelerometer export.",
        "No user IDs are present in this WISDM export; subject_id preserves the provided split name only.",
        f"Session IDs are derived from timestamp resets, gaps larger than {_WISDM_SESSION_GAP_SECONDS:.1f} second(s)"
        + (" and label changes." if split_on_label_change else "."),
        "WISDM is accelerometer-only in this baseline export.",
    ]
    out.attrs["wisdm_split_name"] = split_name
    out.attrs["wisdm_has_true_subject_ids"] = False
    out.attrs["wisdm_split_on_label_change"] = bool(split_on_label_change)
    out.attrs["wisdm_loader_stats"] = {
        "source_file": str(path),
        "raw_rows_read": raw_rows,
        **session_stats,
        "rows_after_session_limit": int(len(out)),
        "num_sessions_after_session_limit": int(out["session_id"].nunique()) if "session_id" in out.columns else 0,
        "median_rows_per_session": float(session_lengths.median()) if not session_lengths.empty else float("nan"),
        "min_rows_per_session": int(session_lengths.min()) if not session_lengths.empty else 0,
        "max_rows_per_session": int(session_lengths.max()) if not session_lengths.empty else 0,
        "sampling_rate_summary": sampling_stats,
    }
    out.attrs["wisdm_label_distribution_raw"] = raw_label_counts
    out.attrs["wisdm_label_distribution_mapped"] = mapped_label_counts
    return out


def load_wisdm(
    path: str | Path,
    *,
    validate: bool = True,
    split: str | None = None,
    max_sessions: int | None = None,
    split_on_label_change: bool = True,
) -> pd.DataFrame:
    src_path = Path(path)

    if src_path.is_dir():
        if split is not None:
            csv_path, split_name = _resolve_wisdm_file(src_path, split=split)
            return _load_wisdm_csv(
                csv_path,
                split_name=split_name,
                validate=validate,
                max_sessions=max_sessions,
                split_on_label_change=split_on_label_change,
            )

        frames = [
            _load_wisdm_csv(
                src_path / "train.csv",
                split_name="train",
                validate=False,
                max_sessions=max_sessions,
                split_on_label_change=split_on_label_change,
            ),
            _load_wisdm_csv(
                src_path / "test.csv",
                split_name="test",
                validate=False,
                max_sessions=max_sessions,
                split_on_label_change=split_on_label_change,
            ),
        ]
        out = pd.concat(frames, ignore_index=True)
        out = finalize_ingest_dataframe(out, validate=validate)

        train_stats = frames[0].attrs.get("wisdm_loader_stats", {})
        test_stats = frames[1].attrs.get("wisdm_loader_stats", {})
        combined_session_lengths = (
            out.groupby("session_id", dropna=False).size() if "session_id" in out.columns else pd.Series(dtype=int)
        )

        out.attrs["loader_notes"] = [
            "WISDM combined loader concatenates the provided train.csv and test.csv exports.",
            "No user IDs are present in this WISDM export; subject_id preserves the provided split name only.",
            f"Session IDs are derived from timestamp resets, gaps larger than {_WISDM_SESSION_GAP_SECONDS:.1f} second(s)"
            + (" and label changes." if split_on_label_change else "."),
            "WISDM is accelerometer-only in this baseline export.",
        ]
        out.attrs["wisdm_has_true_subject_ids"] = False
        out.attrs["wisdm_split_on_label_change"] = bool(split_on_label_change)
        out.attrs["wisdm_loader_stats"] = {
            "raw_rows_read": int(train_stats.get("raw_rows_read", 0)) + int(test_stats.get("raw_rows_read", 0)),
            "rows_after_session_limit": int(len(out)),
            "num_sessions_after_session_limit": int(out["session_id"].nunique()) if "session_id" in out.columns else 0,
            "median_rows_per_session": float(combined_session_lengths.median()) if not combined_session_lengths.empty else float("nan"),
            "min_rows_per_session": int(combined_session_lengths.min()) if not combined_session_lengths.empty else 0,
            "max_rows_per_session": int(combined_session_lengths.max()) if not combined_session_lengths.empty else 0,
            "sampling_rate_summary": _summarize_sampling_rates(out),
            "train_split_stats": train_stats,
            "test_split_stats": test_stats,
        }
        return out

    csv_path, split_name = _resolve_wisdm_file(src_path, split=split)
    return _load_wisdm_csv(
        csv_path,
        split_name=split_name,
        validate=validate,
        max_sessions=max_sessions,
        split_on_label_change=split_on_label_change,
    )
