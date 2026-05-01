#!/usr/bin/env python3
"""Export a contiguous runtime-input CSV from real datasets using common schema loaders.

For segmented datasets (e.g., MobiFall/SisFall), enable --stitch or --mode stitched_adl_fall
to concatenate multiple labeled sessions into one monotonic stream with label transitions.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import pandas as pd
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.ingest import load_mobifall, load_pamap2, load_sisfall, load_uci_har  # noqa: E402
from pipeline.fall.threshold_detector import default_fall_threshold_config, detect_fall_window  # noqa: E402
from pipeline.preprocess import PreprocessConfig, append_derived_channels, resample_dataframe, window_dataframe  # noqa: E402
from pipeline.features import build_feature_table  # noqa: E402
from models.har.baselines import heuristic_har_predict  # noqa: E402
from pipeline.schema import COMMON_SCHEMA_COLUMNS  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export runtime input CSV from real datasets. Use --stitch to join segmented "
            "sessions (e.g., MobiFall/SisFall) into a single stream with label transitions."
        )
    )
    parser.add_argument("--dataset", required=True, choices=["pamap2", "mobifall", "sisfall", "uci_har"])
    parser.add_argument("--path", required=True, help="Root dataset path")
    parser.add_argument("--out", required=True, help="Output CSV path")
    parser.add_argument(
        "--mode",
        choices=[
            "contiguous",
            "stitched_adl_fall",
            "stitched_adl_then_fall",
            "stitched_adl_then_fall_balanced",
            "stitched_adl_then_fall_strong",
            "stitched_adl_fall_strong",
        ],
        default="contiguous",
        help="Export mode: contiguous slice or stitched ADL+FALL stream.",
    )
    parser.add_argument("--max-rows", type=int, default=200000)
    parser.add_argument("--min-label-changes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--target-rate",
        type=float,
        default=50.0,
        help="Target sampling rate for timestamp normalization and fall detection (dt = 1/target_rate).",
    )
    parser.add_argument("--window-size", type=int, default=128)
    parser.add_argument("--step-size", type=int, default=64)
    parser.add_argument("--gap-seconds", type=float, default=0.5)
    parser.add_argument("--min-fall-triggers", type=int, default=1)
    parser.add_argument("--fall-search-limit", type=int, default=50)
    parser.add_argument("--sample-limit", type=int, default=0)
    parser.add_argument(
        "--stitch-adl-fall",
        action="store_true",
        help="Alias for --mode stitched_adl_fall.",
    )
    parser.add_argument(
        "--stitch-adl-fall-strong",
        action="store_true",
        help="Alias for --mode stitched_adl_fall_strong.",
    )
    parser.add_argument("--max-rows-adl", type=int, default=120000)
    parser.add_argument("--max-rows-fall", type=int, default=80000)
    parser.add_argument("--min-fall-detections", type=int, default=1)
    parser.add_argument(
        "--stitch",
        action="store_true",
        help="Stitch multiple labeled segments into a single stream (recommended for segmented datasets).",
    )
    parser.add_argument(
        "--stitch-max-segments",
        type=int,
        default=6,
        help="Maximum number of segments to stitch when --stitch is enabled.",
    )
    return parser.parse_args()


def _resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    return path


def _load_dataset(name: str, path: Path) -> pd.DataFrame:
    if name == "pamap2":
        return load_pamap2(path)
    if name == "mobifall":
        return load_mobifall(path)
    if name == "sisfall":
        return load_sisfall(path)
    if name == "uci_har":
        return load_uci_har(path)
    raise ValueError(f"Unsupported dataset: {name}")


def _filter_required(df: pd.DataFrame) -> pd.DataFrame:
    required = ["timestamp", "ax", "ay", "az", "label_mapped"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in dataset: {', '.join(missing)}")
    filtered = df.dropna(subset=required).reset_index(drop=True)
    if filtered.empty:
        raise ValueError("No rows remain after filtering required columns")
    return filtered


def _sort_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    sort_cols = [c for c in ["subject_id", "session_id", "timestamp"] if c in df.columns]
    if sort_cols:
        return df.sort_values(by=sort_cols, kind="mergesort").reset_index(drop=True)
    return df.reset_index(drop=True)

def _label_changes(labels: pd.Series) -> int:
    if labels.empty:
        return 0
    changes = labels.ne(labels.shift()).sum() - 1
    return int(max(changes, 0))


def _segment_label_mode(labels: pd.Series) -> str:
    if labels.empty:
        return "unknown"
    counts = labels.astype(str).value_counts()
    if counts.empty:
        return "unknown"
    return str(counts.idxmax())

def _normalize_fall_label(value: object) -> str:
    text = str(value).strip().lower()
    if "fall" in text and "non" not in text:
        return "fall"
    return "non_fall"


def _segment_group_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in ["dataset_name", "subject_id", "session_id", "source_file"] if c in df.columns]


def _determine_dt(df: pd.DataFrame, target_rate: float | None) -> float:
    if target_rate and target_rate > 0:
        return 1.0 / float(target_rate)
    if "sampling_rate_hz" in df.columns:
        rates = pd.to_numeric(df["sampling_rate_hz"], errors="coerce").dropna()
        if not rates.empty:
            median_rate = float(rates.median())
            if median_rate > 0:
                return 1.0 / median_rate
    return 0.02


def _build_local_time(series: pd.Series, *, dt: float) -> pd.Series:
    ts = pd.to_numeric(series, errors="coerce")
    if ts.notna().all():
        if len(ts) <= 1:
            return ts - float(ts.iloc[0]) if len(ts) == 1 else pd.Series([], dtype=float)
        diffs = ts.diff().dropna()
        if not diffs.empty and (diffs > 0).all():
            return ts - float(ts.iloc[0])
    return pd.Series([idx * dt for idx in range(len(series))], index=series.index, dtype=float)

def _acc_magnitude_variance(df: pd.DataFrame) -> float:
    for col in ["ax", "ay", "az"]:
        if col not in df.columns:
            return 0.0
    ax = pd.to_numeric(df["ax"], errors="coerce")
    ay = pd.to_numeric(df["ay"], errors="coerce")
    az = pd.to_numeric(df["az"], errors="coerce")
    acc_mag = (ax**2 + ay**2 + az**2) ** 0.5
    return float(acc_mag.var(skipna=True)) if not acc_mag.empty else 0.0


def _acc_magnitude_peak(df: pd.DataFrame) -> float:
    for col in ["ax", "ay", "az"]:
        if col not in df.columns:
            return float("nan")
    ax = pd.to_numeric(df["ax"], errors="coerce")
    ay = pd.to_numeric(df["ay"], errors="coerce")
    az = pd.to_numeric(df["az"], errors="coerce")
    acc_mag = (ax**2 + ay**2 + az**2) ** 0.5
    return float(acc_mag.max(skipna=True)) if not acc_mag.empty else float("nan")

def _select_contiguous_slice(group: pd.DataFrame, *, target_rows: int, rng: random.Random) -> pd.DataFrame:
    working = group.copy()
    if "timestamp" in working.columns:
        working["timestamp"] = pd.to_numeric(working["timestamp"], errors="coerce")
        working = working.sort_values("timestamp", kind="mergesort")
    working = working.reset_index(drop=True)
    if len(working) <= target_rows:
        return working
    max_start = len(working) - target_rows
    start = rng.randint(0, max_start)
    return working.iloc[start : start + target_rows].reset_index(drop=True)


def _select_slice_around_window(
    df: pd.DataFrame,
    *,
    window_index: int | None,
    window_size: int,
    step_size: int,
    target_rows: int,
) -> pd.DataFrame:
    if df.empty:
        return df
    if len(df) <= target_rows:
        return df.reset_index(drop=True)
    target_rows = max(int(target_rows), int(window_size))
    if window_index is None:
        return df.iloc[:target_rows].reset_index(drop=True)
    window_start = int(window_index) * int(step_size)
    window_end = window_start + int(window_size)
    max_start = max(0, min(window_start, len(df) - target_rows))
    min_start = max(0, min(window_end - target_rows, len(df) - target_rows))
    if min_start > max_start:
        start = max_start
    else:
        start = int((min_start + max_start) // 2)
    return df.iloc[start : start + target_rows].reset_index(drop=True)


def _prepare_windows(
    df: pd.DataFrame,
    *,
    target_rate: float,
    window_size: int,
    step_size: int,
) -> list[dict[str, object]]:
    resampled = resample_dataframe(df, target_rate_hz=target_rate)
    resampled = append_derived_channels(resampled)
    preprocess_cfg = PreprocessConfig(target_sampling_rate_hz=target_rate)
    return window_dataframe(resampled, window_size=window_size, step_size=step_size, config=preprocess_cfg)


def _count_fall_triggers(
    df: pd.DataFrame,
    *,
    dataset_name: str | None,
    target_rate: float,
    window_size: int,
    step_size: int,
) -> tuple[int, float, int]:
    if len(df) < window_size:
        return 0, float("nan"), 0
    windows = _prepare_windows(df, target_rate=target_rate, window_size=window_size, step_size=step_size)
    detector_cfg = default_fall_threshold_config(dataset_name=dataset_name)
    predicted = 0
    max_peak = float("-inf")
    for window in windows:
        result = detect_fall_window(window, config=detector_cfg, default_sampling_rate_hz=target_rate)
        features = result.get("features", {}) or {}
        decision = result.get("decision", {}) or {}
        peak = features.get("peak_acc")
        try:
            peak_val = float(peak)
            if peak_val > max_peak:
                max_peak = peak_val
        except Exception:
            pass
        if decision.get("predicted_is_fall"):
            predicted += 1
    if max_peak == float("-inf"):
        max_peak = float("nan")
    return predicted, max_peak, len(windows)

def _window_peak_acc_mag(df: pd.DataFrame, *, window_size: int, step_size: int) -> list[float]:
    if df.empty or window_size <= 0 or step_size <= 0:
        return []
    acc = np.sqrt(
        pd.to_numeric(df["ax"], errors="coerce").to_numpy(dtype=float) ** 2
        + pd.to_numeric(df["ay"], errors="coerce").to_numpy(dtype=float) ** 2
        + pd.to_numeric(df["az"], errors="coerce").to_numpy(dtype=float) ** 2
    )
    peaks: list[float] = []
    for start in range(0, len(acc) - window_size + 1, step_size):
        end = start + window_size
        peak = float(np.nanmax(acc[start:end]))
        if math.isfinite(peak):
            peaks.append(peak)
    return peaks


def _scan_peak_acc_windows(
    df: pd.DataFrame,
    *,
    dataset_name: str | None,
    target_rate: float,
    window_size: int,
    step_size: int,
) -> dict[str, object] | None:
    if df.empty:
        return None
    resampled = resample_dataframe(df, target_rate_hz=target_rate)
    if len(resampled) < window_size:
        return None
    preprocess_cfg = PreprocessConfig(target_sampling_rate_hz=target_rate)
    windows = window_dataframe(resampled, window_size=window_size, step_size=step_size, config=preprocess_cfg)
    if not windows:
        return None
    detector_cfg = default_fall_threshold_config(dataset_name=dataset_name)
    max_peak = float("-inf")
    max_idx = None
    predicted = 0
    for idx, window in enumerate(windows):
        result = detect_fall_window(window, config=detector_cfg, default_sampling_rate_hz=target_rate)
        features = result.get("features", {}) or {}
        decision = result.get("decision", {}) or {}
        peak = features.get("peak_acc")
        try:
            peak_val = float(peak)
        except Exception:
            continue
        if peak_val > max_peak:
            max_peak = peak_val
            max_idx = idx
        if decision.get("predicted_is_fall"):
            predicted += 1
    if max_idx is None:
        return None
    return {
        "max_peak_acc": max_peak,
        "max_window_index": int(max_idx),
        "predicted_fall_windows": int(predicted),
        "n_windows": int(len(windows)),
    }


def _rank_mobifall_files_by_peak(
    files: list[Path],
    *,
    target_rate: float,
    window_size: int,
    step_size: int,
    dataset_name: str | None,
) -> list[dict[str, object]]:
    ranked: list[dict[str, object]] = []
    for path in files:
        df = _filter_required(load_mobifall(path))
        stats = _scan_peak_acc_windows(
            df,
            dataset_name=dataset_name,
            target_rate=target_rate,
            window_size=window_size,
            step_size=step_size,
        )
        if stats is None:
            continue
        stats["source_file"] = str(path)
        ranked.append(stats)
    return ranked


def _acc_mag_window_stats(
    df: pd.DataFrame,
    *,
    window_size: int,
    step_size: int,
) -> tuple[float, float, int, int | None]:
    peaks = _window_peak_acc_mag(df, window_size=window_size, step_size=step_size)
    if not peaks:
        return float("nan"), float("nan"), 0, None
    arr = np.asarray(peaks, dtype=float)
    max_idx = int(np.nanargmax(arr))
    max_peak = float(arr[max_idx])
    p95_peak = float(np.nanpercentile(arr, 95))
    return max_peak, p95_peak, int(len(peaks)), max_idx


def _window_peak_flags(
    df: pd.DataFrame,
    *,
    window_size: int,
    step_size: int,
    impact_threshold: float,
) -> tuple[list[float], list[bool]]:
    peaks = _window_peak_acc_mag(df, window_size=window_size, step_size=step_size)
    flags = [bool(peak >= impact_threshold) for peak in peaks]
    return peaks, flags


def _best_slice_by_qualifying_windows(
    df: pd.DataFrame,
    *,
    window_size: int,
    step_size: int,
    impact_threshold: float,
    max_rows: int,
) -> tuple[pd.DataFrame, int]:
    if df.empty:
        return df, 0
    peaks, flags = _window_peak_flags(
        df,
        window_size=window_size,
        step_size=step_size,
        impact_threshold=impact_threshold,
    )
    n_windows = len(peaks)
    if n_windows == 0:
        return df.iloc[: max_rows].reset_index(drop=True), 0
    if len(df) <= max_rows:
        return df.reset_index(drop=True), int(sum(flags))

    rows = int(max_rows)
    windows_per_slice = max(1, int((rows - window_size) // step_size + 1))
    windows_per_slice = min(windows_per_slice, n_windows)
    prefix = [0]
    for flag in flags:
        prefix.append(prefix[-1] + int(flag))

    best_count = -1
    best_k = 0
    max_k = n_windows - windows_per_slice
    for k in range(0, max_k + 1):
        count = prefix[k + windows_per_slice] - prefix[k]
        if count > best_count:
            best_count = count
            best_k = k

    start_idx = best_k * step_size
    end_idx = min(len(df), start_idx + rows)
    return df.iloc[start_idx:end_idx].reset_index(drop=True), int(best_count)


def _count_detector_pass_windows(
    df: pd.DataFrame,
    *,
    target_rate: float,
    window_size: int,
    step_size: int,
) -> int:
    if df.empty:
        return 0
    working = append_derived_channels(df.copy())
    windows = window_dataframe(
        working,
        window_size=window_size,
        step_size=step_size,
        config=PreprocessConfig(target_sampling_rate_hz=target_rate),
    )
    if not windows:
        return 0
    detector_cfg = default_fall_threshold_config()
    count = 0
    for window in windows:
        result = detect_fall_window(window, config=detector_cfg, default_sampling_rate_hz=target_rate)
        decision = result.get("decision", {}) or {}
        if decision.get("predicted_is_fall"):
            count += 1
    return int(count)


def _scan_mobifall_session_acc_mag(
    path: Path,
    *,
    target_rate: float,
    window_size: int,
    step_size: int,
) -> dict[str, object] | None:
    df = _filter_required(load_mobifall(path))
    resampled = resample_dataframe(df, target_rate_hz=target_rate)
    max_peak, p95_peak, n_windows, max_idx = _acc_mag_window_stats(
        resampled,
        window_size=window_size,
        step_size=step_size,
    )
    if n_windows <= 0:
        return None
    subject_id = str(df["subject_id"].iloc[0]) if "subject_id" in df.columns else None
    session_id = str(df["session_id"].iloc[0]) if "session_id" in df.columns else None
    predicted, _, _ = _count_fall_triggers(
        df,
        dataset_name="MOBIFALL",
        target_rate=target_rate,
        window_size=window_size,
        step_size=step_size,
    )
    return {
        "source_file": str(path),
        "subject_id": subject_id,
        "session_id": session_id,
        "max_peak_acc_mag": float(max_peak),
        "p95_peak_acc_mag": float(p95_peak),
        "n_windows": int(n_windows),
        "max_window_index": max_idx if max_idx is None else int(max_idx),
        "predicted_fall_windows": int(predicted),
    }


def _scan_mobifall_sessions_acc_mag(
    files: list[Path],
    *,
    target_rate: float,
    window_size: int,
    step_size: int,
) -> list[dict[str, object]]:
    stats: list[dict[str, object]] = []
    for path in files:
        result = _scan_mobifall_session_acc_mag(
            path,
            target_rate=target_rate,
            window_size=window_size,
            step_size=step_size,
        )
        if result is None:
            continue
        stats.append(result)
    return stats


def _peak_stats(peaks: list[float]) -> tuple[float, float]:
    if not peaks:
        return float("nan"), float("nan")
    arr = np.asarray(peaks, dtype=float)
    mean = float(np.nanmean(arr))
    p95 = float(np.nanpercentile(arr, 95))
    return mean, p95


def _har_label_changes_in_windows(
    df: pd.DataFrame,
    *,
    target_rate: float,
    window_size: int,
    step_size: int,
) -> int:
    windows = _prepare_windows(df, target_rate=target_rate, window_size=window_size, step_size=step_size)
    if not windows:
        return 0
    feature_df = build_feature_table(
        windows,
        filter_unacceptable=False,
        default_sampling_rate_hz=target_rate,
    )
    if feature_df.empty:
        return 0
    preds = heuristic_har_predict(feature_df).astype("string")
    return _label_changes(pd.Series(preds))


def _mobifall_tag_status(text: str) -> tuple[bool, bool]:
    tag = text.upper()
    exclude = ["JOG", "JUM", "CSO"]
    prefer = ["WAL", "WALK"]
    is_excluded = any(t in tag for t in exclude)
    is_preferred = any(t in tag for t in prefer)
    return is_excluded, is_preferred


def _find_balanced_adl_slice(
    files: list[Path],
    *,
    target_rate: float,
    window_size: int,
    step_size: int,
    target_rows: int,
    rng: random.Random,
) -> tuple[pd.DataFrame, dict[str, object], float, float] | None:
    candidates: list[tuple[int, Path]] = []
    for path in files:
        tag_text = f"{path.name} {path.parent.name}"
        excluded, preferred = _mobifall_tag_status(tag_text)
        if excluded:
            continue
        score = 1 if preferred else 0
        candidates.append((score, path))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0], reverse=True)
    ordered = [p for _, p in candidates]

    for path in ordered:
        df = _filter_required(load_mobifall(path))
        df = resample_dataframe(df, target_rate_hz=target_rate)
        df["label_mapped"] = "non_fall"

        if len(df) < window_size:
            continue

        slice_attempts = 5 if len(df) > target_rows else 1
        for _ in range(slice_attempts):
            candidate = _select_contiguous_slice(df, target_rows=target_rows, rng=rng)
            peaks = _window_peak_acc_mag(candidate, window_size=window_size, step_size=step_size)
            mean_peak, p95_peak = _peak_stats(peaks)
            if not (mean_peak <= 14 and p95_peak <= 18):
                continue
            har_changes = _har_label_changes_in_windows(
                candidate,
                target_rate=target_rate,
                window_size=window_size,
                step_size=step_size,
            )
            if har_changes < 1:
                continue
            meta = {
                "source_file": str(path),
                "har_label_changes": int(har_changes),
            }
            return candidate, meta, mean_peak, p95_peak
    return None


def _find_balanced_fall_slice(
    files: list[Path],
    *,
    target_rate: float,
    window_size: int,
    step_size: int,
    target_rows: int,
    rng: random.Random,
    min_p95: float,
    adl_p95: float,
) -> tuple[pd.DataFrame, dict[str, object], float, float, int] | None:
    best: tuple[pd.DataFrame, dict[str, object], float, float, int] | None = None

    for path in files:
        df = _filter_required(load_mobifall(path))
        df = resample_dataframe(df, target_rate_hz=target_rate)
        df["label_mapped"] = "fall"
        if len(df) < window_size:
            continue
        candidate = _select_contiguous_slice(df, target_rows=target_rows, rng=rng)
        peaks = _window_peak_acc_mag(candidate, window_size=window_size, step_size=step_size)
        mean_peak, p95_peak = _peak_stats(peaks)
        if not math.isfinite(p95_peak):
            continue
        predicted, _, _ = _count_fall_triggers(
            candidate,
            dataset_name="MOBIFALL",
            target_rate=target_rate,
            window_size=window_size,
            step_size=step_size,
        )
        if p95_peak >= min_p95 and p95_peak > adl_p95 + 3.0 and predicted >= 1:
            meta = {
                "source_file": str(path),
                "predicted_fall_windows": int(predicted),
            }
            return candidate, meta, mean_peak, p95_peak, predicted
        # Track best available candidate (highest p95) for fallback.
        if best is None or p95_peak > best[3]:
            meta = {
                "source_file": str(path),
                "predicted_fall_windows": int(predicted),
            }
            best = (candidate, meta, mean_peak, p95_peak, predicted)
    return best


def _stitch_adl_then_fall_balanced_mobifall(
    root: Path,
    *,
    max_rows: int,
    seed: int,
    target_rate: float,
    window_size: int,
    step_size: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    adl_files = _discover_mobifall_files(root, "adl")
    fall_files = _discover_mobifall_files(root, "fall")
    if not adl_files or not fall_files:
        raise ValueError("No ADL/FALLS accelerometer files found for MOBIFALL.")

    rng = random.Random(seed)
    target_rows = max_rows // 2

    adl_candidate = _find_balanced_adl_slice(
        adl_files,
        target_rate=target_rate,
        window_size=window_size,
        step_size=step_size,
        target_rows=target_rows,
        rng=rng,
    )
    if adl_candidate is None:
        raise ValueError("No ADL slice met low-impact criteria.")
    adl_df, adl_meta, adl_mean, adl_p95 = adl_candidate

    fall_candidate = _find_balanced_fall_slice(
        fall_files,
        target_rate=target_rate,
        window_size=window_size,
        step_size=step_size,
        target_rows=target_rows,
        rng=rng,
        min_p95=22.0,
        adl_p95=adl_p95,
    )
    if fall_candidate is None or fall_candidate[3] < 20.0:
        fall_candidate = _find_balanced_fall_slice(
            fall_files,
            target_rate=target_rate,
            window_size=window_size,
            step_size=step_size,
            target_rows=target_rows,
            rng=rng,
            min_p95=20.0,
            adl_p95=adl_p95,
        )
    if fall_candidate is None:
        raise ValueError("No FALL slice met high-impact criteria.")
    fall_df, fall_meta, fall_mean, fall_p95, fall_predicted = fall_candidate

    dt = 1.0 / float(target_rate)
    adl_time = _build_local_time(adl_df["timestamp"], dt=dt)
    adl_df = adl_df.copy()
    adl_df["timestamp"] = (adl_time - float(adl_time.min())).astype(float)
    adl_df["stitched_segment_role"] = "adl"

    fall_time = _build_local_time(fall_df["timestamp"], dt=dt)
    fall_df = fall_df.copy()
    fall_df["timestamp"] = (fall_time - float(fall_time.min()) + float(adl_df["timestamp"].max()) + 2.0).astype(float)
    fall_df["stitched_segment_role"] = "fall"

    stitched_df = pd.concat([adl_df, fall_df], ignore_index=True)
    label_counts = {
        str(label): int(count)
        for label, count in stitched_df["label_mapped"].astype(str).value_counts().to_dict().items()
    }

    summary = {
        "rows": int(len(stitched_df)),
        "label_counts": label_counts,
        "adl_source_file": adl_meta.get("source_file"),
        "fall_source_file": fall_meta.get("source_file"),
        "adl_peak_mean": float(adl_mean),
        "adl_peak_p95": float(adl_p95),
        "fall_peak_mean": float(fall_mean),
        "fall_peak_p95": float(fall_p95),
        "note": "stitched_adl_then_fall_balanced",
        "fall_candidate_predicted_fall_windows": int(fall_predicted),
    }
    return stitched_df, summary


def _stitch_adl_then_fall_mobifall(
    root: Path,
    *,
    max_rows: int,
    gap_seconds: float,
    seed: int,
    target_rate: float,
    window_size: int,
    step_size: int,
) -> tuple[pd.DataFrame, dict[str, object], dict[str, object]]:
    adl_files = _discover_mobifall_files(root, "adl")
    fall_files = _discover_mobifall_files(root, "fall")
    if not adl_files or not fall_files:
        raise ValueError("No ADL/FALLS accelerometer files found for MOBIFALL.")

    dataset_name = "MOBIFALL"
    adl_ranked = _rank_mobifall_files_by_peak(
        adl_files,
        target_rate=target_rate,
        window_size=window_size,
        step_size=step_size,
        dataset_name=dataset_name,
    )
    fall_ranked = _rank_mobifall_files_by_peak(
        fall_files,
        target_rate=target_rate,
        window_size=window_size,
        step_size=step_size,
        dataset_name=dataset_name,
    )
    if not adl_ranked:
        raise ValueError("No ADL files produced peak_acc stats.")
    if not fall_ranked:
        raise ValueError("No FALL files produced peak_acc stats.")

    adl_ranked.sort(key=lambda item: float(item.get("max_peak_acc", float("inf"))))
    fall_ranked.sort(key=lambda item: float(item.get("max_peak_acc", float("-inf"))), reverse=True)

    chosen_adl = adl_ranked[0]
    chosen_fall = fall_ranked[0]

    adl_df = _filter_required(load_mobifall(chosen_adl["source_file"]))
    fall_df = _filter_required(load_mobifall(chosen_fall["source_file"]))
    adl_resampled = resample_dataframe(adl_df, target_rate_hz=target_rate)
    fall_resampled = resample_dataframe(fall_df, target_rate_hz=target_rate)

    target_rows = max_rows // 2
    adl_resampled = _select_slice_around_window(
        adl_resampled,
        window_index=chosen_adl.get("max_window_index"),
        window_size=window_size,
        step_size=step_size,
        target_rows=target_rows,
    )
    fall_resampled = _select_slice_around_window(
        fall_resampled,
        window_index=chosen_fall.get("max_window_index"),
        window_size=window_size,
        step_size=step_size,
        target_rows=target_rows,
    )

    adl_resampled = adl_resampled.copy()
    fall_resampled = fall_resampled.copy()
    adl_resampled["label_mapped"] = "non_fall"
    adl_resampled["label_raw"] = "ADL"
    adl_resampled["task_type"] = "adl"
    fall_resampled["label_mapped"] = "fall"
    fall_resampled["label_raw"] = "fall"
    fall_resampled["task_type"] = "fall"

    dt = 1.0 / float(target_rate)
    adl_time = _build_local_time(adl_resampled["timestamp"], dt=dt)
    adl_time = adl_time - float(adl_time.min()) if not adl_time.empty else adl_time
    adl_resampled["timestamp"] = adl_time.astype(float)
    adl_resampled["stitched_segment_role"] = "adl"

    fall_time = _build_local_time(fall_resampled["timestamp"], dt=dt)
    fall_time = fall_time - float(fall_time.min()) if not fall_time.empty else fall_time
    offset = float(adl_resampled["timestamp"].max()) + float(gap_seconds) if not adl_resampled["timestamp"].empty else float(gap_seconds)
    fall_resampled["timestamp"] = (fall_time + offset).astype(float)
    fall_resampled["stitched_segment_role"] = "fall"

    stitched_df = pd.concat([adl_resampled, fall_resampled], ignore_index=True)
    label_counts = {
        str(label): int(count)
        for label, count in stitched_df["label_mapped"].astype(str).value_counts().to_dict().items()
    }

    summary = {
        "rows_total": int(len(stitched_df)),
        "rows_adl": int(len(adl_resampled)),
        "rows_fall": int(len(fall_resampled)),
        "ts_min": float(stitched_df["timestamp"].min()) if not stitched_df["timestamp"].empty else None,
        "ts_max": float(stitched_df["timestamp"].max()) if not stitched_df["timestamp"].empty else None,
        "gap_seconds": float(gap_seconds),
        "labels_present": label_counts,
        "adl_peak_acc_max": float(chosen_adl.get("max_peak_acc", float("nan"))),
        "fall_peak_acc_max": float(chosen_fall.get("max_peak_acc", float("nan"))),
    }

    report = {
        "mode": "stitched_adl_then_fall",
        "adl_source_file": chosen_adl.get("source_file"),
        "fall_source_file": chosen_fall.get("source_file"),
        "adl_peak_acc_max": float(chosen_adl.get("max_peak_acc", float("nan"))),
        "fall_peak_acc_max": float(chosen_fall.get("max_peak_acc", float("nan"))),
        "adl_peak_window_index": int(chosen_adl.get("max_window_index", -1)),
        "fall_peak_window_index": int(chosen_fall.get("max_window_index", -1)),
        "adl_predicted_fall_windows": int(chosen_adl.get("predicted_fall_windows", 0)),
        "fall_predicted_fall_windows": int(chosen_fall.get("predicted_fall_windows", 0)),
        "window_size": int(window_size),
        "step_size": int(step_size),
        "target_rate": float(target_rate),
        "rows_adl": int(len(adl_resampled)),
        "rows_fall": int(len(fall_resampled)),
        "rows_total": int(len(stitched_df)),
    }
    return stitched_df, summary, report


def _stitch_adl_then_fall_strong_mobifall(
    root: Path,
    *,
    max_rows: int,
    gap_seconds: float,
    seed: int,
    target_rate: float,
    window_size: int,
    step_size: int,
) -> tuple[pd.DataFrame, dict[str, object], dict[str, object]]:
    adl_files = _discover_mobifall_files(root, "adl")
    fall_files = _discover_mobifall_files(root, "fall")
    if not adl_files or not fall_files:
        raise ValueError("No ADL/FALLS accelerometer files found for MOBIFALL.")

    adl_stats = _scan_mobifall_sessions_acc_mag(
        adl_files,
        target_rate=target_rate,
        window_size=window_size,
        step_size=step_size,
    )
    fall_stats = _scan_mobifall_sessions_acc_mag(
        fall_files,
        target_rate=target_rate,
        window_size=window_size,
        step_size=step_size,
    )
    if not adl_stats:
        raise ValueError("No ADL sessions produced peak stats.")
    if not fall_stats:
        raise ValueError("No FALL sessions produced peak stats.")

    adl_sorted = sorted(adl_stats, key=lambda item: float(item.get("max_peak_acc_mag", float("inf"))))
    adl_candidates = [item for item in adl_sorted if int(item.get("predicted_fall_windows", 0)) == 0] or adl_sorted
    adl_idx = int(0.25 * (len(adl_candidates) - 1)) if len(adl_candidates) > 1 else 0
    chosen_adl = adl_candidates[adl_idx]

    fall_sorted = sorted(fall_stats, key=lambda item: float(item.get("max_peak_acc_mag", float("-inf"))), reverse=True)
    fall_predicted = [item for item in fall_sorted if int(item.get("predicted_fall_windows", 0)) > 0]
    chosen_fall = fall_predicted[0] if fall_predicted else fall_sorted[0]

    adl_df = _filter_required(load_mobifall(chosen_adl["source_file"]))
    fall_df = _filter_required(load_mobifall(chosen_fall["source_file"]))
    adl_resampled = resample_dataframe(adl_df, target_rate_hz=target_rate)
    fall_resampled = resample_dataframe(fall_df, target_rate_hz=target_rate)

    target_rows = max_rows // 2
    adl_resampled = _select_slice_around_window(
        adl_resampled,
        window_index=chosen_adl.get("max_window_index"),
        window_size=window_size,
        step_size=step_size,
        target_rows=target_rows,
    )
    fall_resampled = _select_slice_around_window(
        fall_resampled,
        window_index=chosen_fall.get("max_window_index"),
        window_size=window_size,
        step_size=step_size,
        target_rows=max(1, max_rows - len(adl_resampled)),
    )

    adl_resampled = adl_resampled.copy()
    fall_resampled = fall_resampled.copy()
    adl_resampled["label_mapped"] = "non_fall"
    adl_resampled["label_raw"] = "ADL"
    adl_resampled["task_type"] = "adl"
    fall_resampled["label_mapped"] = "fall"
    fall_resampled["label_raw"] = "fall"
    fall_resampled["task_type"] = "fall"

    dt = 1.0 / float(target_rate)
    adl_time = _build_local_time(adl_resampled["timestamp"], dt=dt)
    adl_time = adl_time - float(adl_time.min()) if not adl_time.empty else adl_time
    adl_resampled["timestamp"] = adl_time.astype(float)
    adl_resampled["stitched_segment_role"] = "adl"

    fall_time = _build_local_time(fall_resampled["timestamp"], dt=dt)
    fall_time = fall_time - float(fall_time.min()) if not fall_time.empty else fall_time
    offset = float(adl_resampled["timestamp"].max()) + float(gap_seconds) if not adl_resampled["timestamp"].empty else float(gap_seconds)
    fall_resampled["timestamp"] = (fall_time + offset).astype(float)
    fall_resampled["stitched_segment_role"] = "fall"

    stitched_df = pd.concat([adl_resampled, fall_resampled], ignore_index=True)
    label_counts = {
        str(label): int(count)
        for label, count in stitched_df["label_mapped"].astype(str).value_counts().to_dict().items()
    }

    stitched_windows = window_dataframe(
        stitched_df,
        window_size=window_size,
        step_size=step_size,
        config=PreprocessConfig(target_sampling_rate_hz=target_rate),
    )
    window_label_counts: dict[str, int] = {}
    for window in stitched_windows:
        label = str(window.get("label_mapped_majority") or "unknown")
        window_label_counts[label] = window_label_counts.get(label, 0) + 1

    adl_peak_max, adl_peak_p95, _, _ = _acc_mag_window_stats(
        adl_resampled, window_size=window_size, step_size=step_size
    )
    fall_peak_max, fall_peak_p95, _, _ = _acc_mag_window_stats(
        fall_resampled, window_size=window_size, step_size=step_size
    )

    summary = {
        "rows_total": int(len(stitched_df)),
        "rows_adl": int(len(adl_resampled)),
        "rows_fall": int(len(fall_resampled)),
        "ts_min": float(stitched_df["timestamp"].min()) if not stitched_df["timestamp"].empty else None,
        "ts_max": float(stitched_df["timestamp"].max()) if not stitched_df["timestamp"].empty else None,
        "gap_seconds": float(gap_seconds),
        "labels_present": label_counts,
    }

    report = {
        "chosen_adl_session_id": chosen_adl.get("session_id"),
        "chosen_fall_session_id": chosen_fall.get("session_id"),
        "chosen_adl_source_file": chosen_adl.get("source_file"),
        "chosen_fall_source_file": chosen_fall.get("source_file"),
        "adl_peak_acc_mag_max": float(adl_peak_max),
        "adl_peak_acc_mag_p95": float(adl_peak_p95),
        "fall_peak_acc_mag_max": float(fall_peak_max),
        "fall_peak_acc_mag_p95": float(fall_peak_p95),
        "window_counts_by_label": window_label_counts,
    }
    return stitched_df, summary, report


def _stitch_adl_fall_strong_mobifall(
    root: Path,
    *,
    max_rows_adl: int,
    max_rows_fall: int,
    min_fall_detections: int,
    target_rate: float,
    window_size: int,
    step_size: int,
) -> tuple[pd.DataFrame, dict[str, object], dict[str, object]]:
    adl_files = _discover_mobifall_files(root, "adl")
    fall_files = _discover_mobifall_files(root, "fall")
    if not adl_files or not fall_files:
        raise ValueError("No ADL/FALLS accelerometer files found for MOBIFALL.")

    impact_threshold = default_fall_threshold_config(dataset_name="MOBIFALL").impact_peak_acc_threshold

    adl_candidates: list[dict[str, object]] = []
    for path in adl_files:
        df = _filter_required(load_mobifall(path))
        resampled = resample_dataframe(df, target_rate_hz=target_rate)
        max_peak, p95_peak, n_windows, _ = _acc_mag_window_stats(
            resampled, window_size=window_size, step_size=step_size
        )
        if n_windows <= 0:
            continue
        adl_candidates.append(
            {
                "source_file": str(path),
                "session_id": str(df["session_id"].iloc[0]) if "session_id" in df.columns else None,
                "max_peak": float(max_peak),
                "p95_peak": float(p95_peak),
            }
        )

    if not adl_candidates:
        raise ValueError("No ADL sessions produced usable windows for stitching.")

    adl_candidates.sort(key=lambda item: float(item.get("max_peak", float("inf"))))
    chosen_adl = adl_candidates[0]
    adl_df = _filter_required(load_mobifall(chosen_adl["source_file"]))
    adl_resampled = resample_dataframe(adl_df, target_rate_hz=target_rate)
    if len(adl_resampled) > int(max_rows_adl):
        adl_resampled = adl_resampled.iloc[: int(max_rows_adl)].reset_index(drop=True)

    fall_candidates: list[dict[str, object]] = []
    best_failed_count = 0
    for path in fall_files:
        df = _filter_required(load_mobifall(path))
        resampled = resample_dataframe(df, target_rate_hz=target_rate)
        if resampled.empty:
            continue
        fall_slice, qualifying_count = _best_slice_by_qualifying_windows(
            resampled,
            window_size=window_size,
            step_size=step_size,
            impact_threshold=float(impact_threshold),
            max_rows=int(max_rows_fall),
        )
        if qualifying_count < min_fall_detections:
            best_failed_count = max(best_failed_count, int(qualifying_count))
            continue
        max_peak, p95_peak, _, _ = _acc_mag_window_stats(
            fall_slice, window_size=window_size, step_size=step_size
        )
        detector_pass_windows = _count_detector_pass_windows(
            fall_slice,
            target_rate=target_rate,
            window_size=window_size,
            step_size=step_size,
        )
        fall_candidates.append(
            {
                "source_file": str(path),
                "session_id": str(df["session_id"].iloc[0]) if "session_id" in df.columns else None,
                "qualifying_windows": int(qualifying_count),
                "max_peak": float(max_peak),
                "p95_peak": float(p95_peak),
                "detector_pass_windows": int(detector_pass_windows),
            }
        )

    if not fall_candidates:
        raise ValueError(
            "No FALL sessions met min-fall-detections. "
            f"Highest qualifying window count observed: {best_failed_count}. "
            "Try lowering --min-fall-detections or expanding the search dataset."
        )

    eligible_candidates = [item for item in fall_candidates if int(item.get("detector_pass_windows", 0)) >= 1]
    if not eligible_candidates:
        eligible_candidates = fall_candidates

    eligible_candidates.sort(
        key=lambda item: (int(item.get("qualifying_windows", 0)), float(item.get("max_peak", float('-inf')))),
        reverse=True,
    )
    chosen_fall = eligible_candidates[0]
    fall_df = _filter_required(load_mobifall(chosen_fall["source_file"]))
    fall_resampled_full = resample_dataframe(fall_df, target_rate_hz=target_rate)
    fall_resampled, qualifying_count = _best_slice_by_qualifying_windows(
        fall_resampled_full,
        window_size=window_size,
        step_size=step_size,
        impact_threshold=float(impact_threshold),
        max_rows=int(max_rows_fall),
    )

    dt = 1.0 / float(target_rate)
    adl_time = _build_local_time(adl_resampled["timestamp"], dt=dt)
    adl_time = adl_time - float(adl_time.min()) if not adl_time.empty else adl_time
    adl_resampled = adl_resampled.copy()
    adl_resampled["timestamp"] = adl_time.astype(float)
    adl_resampled["stitched_segment_role"] = "adl"

    fall_time = _build_local_time(fall_resampled["timestamp"], dt=dt)
    fall_time = fall_time - float(fall_time.min()) if not fall_time.empty else fall_time
    offset = float(adl_resampled["timestamp"].max()) + dt if not adl_resampled["timestamp"].empty else dt
    fall_resampled = fall_resampled.copy()
    fall_resampled["timestamp"] = (fall_time + offset).astype(float)
    fall_resampled["stitched_segment_role"] = "fall"

    stitched_df = pd.concat([adl_resampled, fall_resampled], ignore_index=True)
    label_counts = {
        str(label): int(count)
        for label, count in stitched_df["label_mapped"].astype(str).value_counts().to_dict().items()
    }

    summary = {
        "rows_total": int(len(stitched_df)),
        "rows_adl": int(len(adl_resampled)),
        "rows_fall": int(len(fall_resampled)),
        "labels_present": label_counts,
    }
    report = {
        "chosen_adl_source_file": chosen_adl.get("source_file"),
        "chosen_adl_session_id": chosen_adl.get("session_id"),
        "chosen_fall_source_file": chosen_fall.get("source_file"),
        "chosen_fall_session_id": chosen_fall.get("session_id"),
        "qualifying_fall_windows": int(qualifying_count),
        "counts_by_label": label_counts,
    }
    return stitched_df, summary, report


def _discover_mobifall_files(root: Path, category: str) -> list[Path]:
    category = category.lower()
    files: list[Path] = []
    for path in sorted(root.rglob("*_acc_*.txt")):
        parts = [p.lower() for p in path.parts]
        if category == "adl" and "adl" in parts:
            files.append(path)
        elif category == "fall" and ("falls" in parts or "fall" in parts):
            files.append(path)
    return files


def _stitch_adl_fall_mobifall(
    root: Path,
    *,
    max_rows: int,
    gap_seconds: float,
    min_fall_triggers: int,
    fall_search_limit: int,
    seed: int,
    target_rate: float,
    window_size: int,
    step_size: int,
    sample_limit: int,
) -> tuple[pd.DataFrame, dict[str, object], dict[str, object]]:
    adl_files = _discover_mobifall_files(root, "adl")
    fall_files = _discover_mobifall_files(root, "fall")
    if sample_limit and sample_limit > 0:
        adl_files = adl_files[: sample_limit]
        fall_files = fall_files[: sample_limit]
    if not adl_files or not fall_files:
        raise ValueError("No ADL/FALLS accelerometer files found for MOBIFALL.")

    rng = random.Random(seed)
    rng.shuffle(adl_files)

    chosen_adl = None
    adl_df = None
    for path in adl_files:
        df = _filter_required(load_mobifall(path))
        if len(df) >= window_size:
            chosen_adl = path
            adl_df = df
            break
    if chosen_adl is None:
        chosen_adl = adl_files[0]
        adl_df = _filter_required(load_mobifall(chosen_adl))

    best_fall = None
    best_peak = float("-inf")
    best_pred = 0
    best_windows = 0

    for path in fall_files[: max(1, int(fall_search_limit))]:
        df = _filter_required(load_mobifall(path))
        predicted, peak_acc, n_windows = _count_fall_triggers(
            df,
            dataset_name="MOBIFALL",
            target_rate=target_rate,
            window_size=window_size,
            step_size=step_size,
        )
        raw_peak = _acc_magnitude_peak(df)
        effective_peak = peak_acc if math.isfinite(float(peak_acc)) else raw_peak
        if effective_peak > best_peak:
            best_peak = effective_peak
            best_fall = (path, df, predicted, n_windows)
        if predicted >= int(min_fall_triggers):
            best_fall = (path, df, predicted, n_windows)
            best_pred = predicted
            best_windows = n_windows
            break

    if best_fall is None:
        raise ValueError("No usable fall files found for MOBIFALL.")

    chosen_fall, fall_df, best_pred, best_windows = best_fall
    if best_pred < int(min_fall_triggers):
        print("WARNING: No fall file met min_fall_triggers; using highest peak_acc fall file.")

    adl_resampled = resample_dataframe(adl_df, target_rate_hz=target_rate)
    fall_resampled = resample_dataframe(fall_df, target_rate_hz=target_rate)

    target_rows = max_rows // 2
    adl_resampled = _select_contiguous_slice(adl_resampled, target_rows=target_rows, rng=rng)
    remaining = max_rows - len(adl_resampled)
    fall_resampled = _select_contiguous_slice(fall_resampled, target_rows=max(1, remaining), rng=rng)

    adl_resampled["label_mapped"] = "non_fall"
    fall_resampled["label_mapped"] = "fall"

    dt = 1.0 / float(target_rate)
    adl_time = _build_local_time(adl_resampled["timestamp"], dt=dt)
    adl_time = adl_time - float(adl_time.min()) if not adl_time.empty else adl_time
    adl_resampled["timestamp"] = adl_time.astype(float)
    adl_resampled["stitched_segment_role"] = "adl"

    fall_time = _build_local_time(fall_resampled["timestamp"], dt=dt)
    fall_time = fall_time - float(fall_time.min()) if not fall_time.empty else fall_time
    offset = float(adl_resampled["timestamp"].max()) + float(gap_seconds) if not adl_resampled["timestamp"].empty else float(gap_seconds)
    fall_resampled["timestamp"] = (fall_time + offset).astype(float)
    fall_resampled["stitched_segment_role"] = "fall"

    stitched_df = pd.concat([adl_resampled, fall_resampled], ignore_index=True)
    label_counts = {
        str(label): int(count)
        for label, count in stitched_df["label_mapped"].astype(str).value_counts().to_dict().items()
    }

    acc_mag_max_adl = _acc_magnitude_peak(adl_resampled)
    acc_mag_max_fall = _acc_magnitude_peak(fall_resampled)

    meta = {
        "chosen_adl_file": str(chosen_adl),
        "chosen_fall_file": str(chosen_fall),
        "rows_adl": int(len(adl_resampled)),
        "rows_fall": int(len(fall_resampled)),
        "rows_total": int(len(stitched_df)),
        "acc_mag_max_adl": acc_mag_max_adl,
        "acc_mag_max_fall": acc_mag_max_fall,
        "label_counts": label_counts,
    }

    summary = {
        "rows_total": int(len(stitched_df)),
        "rows_adl": int(len(adl_resampled)),
        "rows_fall": int(len(fall_resampled)),
        "ts_min": float(stitched_df["timestamp"].min()) if not stitched_df["timestamp"].empty else None,
        "ts_max": float(stitched_df["timestamp"].max()) if not stitched_df["timestamp"].empty else None,
        "gap_seconds": float(gap_seconds),
        "labels_present": label_counts,
        "fall_candidate_predicted_fall_windows": int(best_pred),
    }
    return stitched_df, summary, meta


def _stitch_adl_fall(
    df: pd.DataFrame,
    *,
    max_rows: int,
    gap_seconds: float,
    min_fall_triggers: int,
    fall_search_limit: int,
    seed: int,
    target_rate: float,
    window_size: int,
    step_size: int,
    min_rows_per_segment: int | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    group_cols = _segment_group_columns(df)
    if min_rows_per_segment is None:
        min_rows_per_segment = max(int(window_size), 256)

    label_norm = df["label_mapped"].apply(_normalize_fall_label)
    adl_df_all = df[label_norm == "non_fall"].copy()
    fall_df_all = df[label_norm == "fall"].copy()

    def _build_segments(frame: pd.DataFrame) -> list[dict[str, object]]:
        if frame.empty:
            return []
        group_iter = frame.groupby(group_cols, dropna=False, sort=False) if group_cols else [(None, frame)]
        segs: list[dict[str, object]] = []
        for key, group in group_iter:
            group = group.reset_index(drop=True)
            if len(group) < min_rows_per_segment:
                continue
            segs.append(
                {
                    "key": key,
                    "df": group,
                    "rows": int(len(group)),
                    "peak_acc": _acc_magnitude_peak(group),
                    "subject_id": str(group["subject_id"].iloc[0]) if "subject_id" in group.columns else None,
                    "session_id": str(group["session_id"].iloc[0]) if "session_id" in group.columns else None,
                    "source_file": str(group["source_file"].iloc[0]) if "source_file" in group.columns else None,
                }
            )
        return segs

    adl_segments = _build_segments(adl_df_all)
    fall_segments = _build_segments(fall_df_all)
    if not adl_segments or not fall_segments:
        raise ValueError("Unable to find both ADL and FALL segments for stitching.")

    rng = random.Random(seed)
    rng.shuffle(adl_segments)
    def _peak_sort_value(seg: dict[str, object]) -> float:
        try:
            val = float(seg.get("peak_acc"))
        except Exception:
            return float("-inf")
        return val if math.isfinite(val) else float("-inf")

    fall_segments.sort(key=_peak_sort_value, reverse=True)

    target_rows = max_rows // 2
    adl_segment = None
    for seg in adl_segments:
        if _acc_magnitude_variance(seg["df"]) <= 0:
            continue
        adl_segment = seg
        break
    if adl_segment is None:
        raise ValueError("No usable ADL segment found with motion variance.")

    adl_df = _select_contiguous_slice(adl_segment["df"], target_rows=target_rows, rng=rng)
    adl_len = len(adl_df)
    fall_target_rows = max(1, max_rows - adl_len)

    dataset_name = str(df["dataset_name"].iloc[0]) if "dataset_name" in df.columns else None

    selected_fall = None
    selected_predicted = 0
    selected_windows = 0
    fallback = None
    fallback_peak = float("-inf")

    for seg in fall_segments[: int(fall_search_limit)]:
        candidate_df = _select_contiguous_slice(seg["df"], target_rows=fall_target_rows, rng=rng)
        predicted, peak_acc, n_windows = _count_fall_triggers(
            candidate_df,
            dataset_name=dataset_name,
            target_rate=target_rate,
            window_size=window_size,
            step_size=step_size,
        )
        if peak_acc > fallback_peak:
            fallback_peak = peak_acc
            fallback = (seg, candidate_df, predicted, n_windows)
        if predicted >= int(min_fall_triggers):
            selected_fall = (seg, candidate_df, predicted, n_windows)
            selected_predicted = predicted
            selected_windows = n_windows
            break

    if selected_fall is None and fallback is not None:
        seg, candidate_df, predicted, n_windows = fallback
        selected_fall = (seg, candidate_df, predicted, n_windows)
        selected_predicted = predicted
        selected_windows = n_windows
        print(
            "WARNING: No fall segment met min_fall_triggers; using fallback with highest peak_acc."
        )

    if selected_fall is None:
        raise ValueError("No usable fall segment found for stitching.")

    fall_seg, fall_df, _, _ = selected_fall

    dt = 1.0 / float(target_rate)
    adl_time = _build_local_time(adl_df["timestamp"], dt=dt)
    adl_time = adl_time - float(adl_time.min()) if not adl_time.empty else adl_time
    adl_df = adl_df.copy()
    adl_df["timestamp"] = adl_time.astype(float)
    adl_df["stitched_segment_role"] = "adl"

    fall_time = _build_local_time(fall_df["timestamp"], dt=dt)
    fall_time = fall_time - float(fall_time.min()) if not fall_time.empty else fall_time
    gap = float(gap_seconds)
    offset = float(adl_df["timestamp"].max()) + gap if not adl_df["timestamp"].empty else gap
    fall_df = fall_df.copy()
    fall_df["timestamp"] = (fall_time + offset).astype(float)
    fall_df["stitched_segment_role"] = "fall"

    stitched_df = pd.concat([adl_df, fall_df], ignore_index=True)
    label_counts = {
        str(label): int(count)
        for label, count in stitched_df["label_mapped"].astype(str).value_counts().to_dict().items()
    }

    summary = {
        "rows_total": int(len(stitched_df)),
        "rows_adl": int(len(adl_df)),
        "rows_fall": int(len(fall_df)),
        "ts_min": float(stitched_df["timestamp"].min()) if not stitched_df["timestamp"].empty else None,
        "ts_max": float(stitched_df["timestamp"].max()) if not stitched_df["timestamp"].empty else None,
        "gap_seconds": float(gap_seconds),
        "labels_present": label_counts,
        "fall_candidate_predicted_fall_windows": int(selected_predicted),
        "fall_candidate_windows": int(selected_windows),
        "adl_source": {
            "subject_id": adl_segment.get("subject_id"),
            "session_id": adl_segment.get("session_id"),
            "source_file": adl_segment.get("source_file"),
        },
        "fall_source": {
            "subject_id": fall_seg.get("subject_id"),
            "session_id": fall_seg.get("session_id"),
            "source_file": fall_seg.get("source_file"),
        },
    }
    return stitched_df, summary


def _stitch_segments(
    df: pd.DataFrame,
    *,
    min_label_changes: int,
    max_rows: int,
    seed: int,
    stitch_max_segments: int,
    target_rate: float | None,
    min_rows_per_segment: int = 256,
) -> tuple[pd.DataFrame, dict[str, object]]:
    group_cols = _segment_group_columns(df)
    if group_cols:
        grouped = list(df.groupby(group_cols, dropna=False, sort=False))
    else:
        grouped = [("all", df)]

    segments: list[dict[str, object]] = []
    for key, group in grouped:
        group = group.reset_index(drop=True)
        if len(group) < min_rows_per_segment:
            continue
        label = _segment_label_mode(group["label_mapped"])
        segments.append(
            {
                "key": key,
                "label": label,
                "rows": int(len(group)),
                "df": group,
                "subject_id": str(group["subject_id"].iloc[0]) if "subject_id" in group.columns else None,
                "session_id": str(group["session_id"].iloc[0]) if "session_id" in group.columns else None,
                "source_file": str(group["source_file"].iloc[0]) if "source_file" in group.columns else None,
            }
        )

    segments = [seg for seg in segments if int(seg["rows"]) <= max_rows]
    if not segments:
        raise ValueError("No usable segments found for stitching within max_rows")

    rng = random.Random(seed)
    remaining = segments[:]
    start = rng.choice(remaining)
    remaining.remove(start)

    selected = [start]
    total_rows = int(start["rows"])  # type: ignore[assignment]
    label_changes = 0
    last_label = str(start["label"])

    while (
        remaining
        and total_rows < max_rows
        and len(selected) < stitch_max_segments
        and label_changes < min_label_changes
    ):
        candidates = [seg for seg in remaining if total_rows + int(seg["rows"]) <= max_rows]
        if not candidates:
            break
        preferred = [seg for seg in candidates if str(seg["label"]) != last_label]
        next_seg = rng.choice(preferred or candidates)
        remaining.remove(next_seg)
        selected.append(next_seg)
        total_rows += int(next_seg["rows"])
        if str(next_seg["label"]) != last_label:
            label_changes += 1
        last_label = str(next_seg["label"])

    dt = _determine_dt(df, target_rate)
    stitched_frames: list[pd.DataFrame] = []
    offset = 0.0
    last_ts = None

    for idx, seg in enumerate(selected):
        seg_df = seg["df"].copy()  # type: ignore[assignment]
        seg_df = seg_df.reset_index(drop=True)
        local_time = _build_local_time(seg_df["timestamp"], dt=dt)
        if idx == 0:
            shifted = local_time
        else:
            if last_ts is None:
                last_ts = 0.0
            offset = float(last_ts) + dt
            shifted = local_time + offset
        seg_df["timestamp"] = shifted.astype(float)
        seg_df["stitched_segment_index"] = int(idx)
        source_session = (
            str(seg.get("session_id"))
            if seg.get("session_id") not in (None, "None", "<NA>")
            else str(seg.get("source_file") or "unknown")
        )
        seg_df["stitched_source_session"] = source_session
        stitched_frames.append(seg_df)
        if len(shifted) > 0:
            last_ts = float(shifted.iloc[-1])

    stitched_df = pd.concat(stitched_frames, ignore_index=True)
    stitched_df["session_id"] = f"stitched_{seed}"

    label_counts = {
        str(label): int(count)
        for label, count in stitched_df["label_mapped"].astype(str).value_counts().to_dict().items()
    }
    actual_label_changes = _label_changes(stitched_df["label_mapped"].astype(str))

    summary_segments = []
    for idx, seg in enumerate(selected):
        summary_segments.append(
            {
                "segment_index": idx,
                "segment_label": str(seg["label"]),
                "rows": int(seg["rows"]),
                "subject_id": str(seg.get("subject_id")) if seg.get("subject_id") is not None else None,
                "session_id": str(seg.get("session_id")) if seg.get("session_id") is not None else None,
                "source_file": str(seg.get("source_file")) if seg.get("source_file") is not None else None,
            }
        )

    summary: dict[str, object] = {
        "dataset": str(stitched_df["dataset_name"].iloc[0]) if "dataset_name" in stitched_df.columns else None,
        "rows": int(len(stitched_df)),
        "segments_used": int(len(selected)),
        "label_changes": int(actual_label_changes),
        "label_counts": label_counts,
        "segments": summary_segments,
        "timestamp_min": float(stitched_df["timestamp"].min()) if not stitched_df["timestamp"].empty else None,
        "timestamp_max": float(stitched_df["timestamp"].max()) if not stitched_df["timestamp"].empty else None,
        "dt_used": float(dt),
    }
    if actual_label_changes < min_label_changes:
        summary["warning"] = "Unable to reach requested label_changes with available segments."

    return stitched_df, summary


def _find_contiguous_slice(
    df: pd.DataFrame,
    *,
    min_label_changes: int,
    max_rows: int,
) -> tuple[int, int, int] | None:
    labels = df["label_mapped"].astype(str).tolist()
    n = len(labels)
    if n == 0:
        return None

    change = [0] * n
    for i in range(1, n):
        change[i] = 1 if labels[i] != labels[i - 1] else 0

    prefix = [0] * n
    for i in range(1, n):
        prefix[i] = prefix[i - 1] + change[i]

    left = 0
    for right in range(n):
        while left < right and (right - left + 1) > max_rows:
            left += 1
        transitions = 0 if right == left else prefix[right] - prefix[left]
        if transitions >= min_label_changes:
            return left, right, transitions
    return None


def _choose_slice(
    df: pd.DataFrame,
    *,
    min_label_changes: int,
    max_rows: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    group_cols = [c for c in ["subject_id", "session_id"] if c in df.columns]
    if group_cols:
        groups = list(df.groupby(group_cols, dropna=False, sort=False))
    else:
        groups = [("all", df)]

    rng = random.Random(seed)
    rng.shuffle(groups)

    for key, group in groups:
        group = group.reset_index(drop=True)
        result = _find_contiguous_slice(group, min_label_changes=min_label_changes, max_rows=max_rows)
        if result is None:
            continue
        start, end, transitions = result
        slice_df = group.iloc[start : end + 1].copy()
        info = {
            "group_key": key,
            "row_count": len(slice_df),
            "label_transitions": transitions,
        }
        return slice_df, info

    raise ValueError(
        "No contiguous slice found with at least "
        f"{min_label_changes} label changes within {max_rows} rows."
    )


def main() -> int:
    args = parse_args()
    if args.stitch_adl_fall:
        args.mode = "stitched_adl_fall"
    if args.stitch_adl_fall_strong:
        args.mode = "stitched_adl_fall_strong"
    dataset_path = _resolve_path(args.path)
    out_path = _resolve_path(args.out)

    if args.mode == "stitched_adl_fall":
        if args.dataset == "mobifall":
            stitched_df, summary, meta = _stitch_adl_fall_mobifall(
                dataset_path,
                max_rows=int(args.max_rows),
                gap_seconds=float(args.gap_seconds),
                min_fall_triggers=int(args.min_fall_triggers),
                fall_search_limit=int(args.fall_search_limit),
                seed=int(args.seed),
                target_rate=float(args.target_rate),
                window_size=int(args.window_size),
                step_size=int(args.step_size),
                sample_limit=int(args.sample_limit),
            )
            meta_path = Path(str(out_path) + ".meta.json")
            meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        else:
            df = _load_dataset(args.dataset, dataset_path)
            df = _filter_required(df)
            stitched_df, summary = _stitch_adl_fall(
                df,
                max_rows=int(args.max_rows),
                gap_seconds=float(args.gap_seconds),
                min_fall_triggers=int(args.min_fall_triggers),
                fall_search_limit=int(args.fall_search_limit),
                seed=int(args.seed),
                target_rate=float(args.target_rate),
                window_size=int(args.window_size),
                step_size=int(args.step_size),
            )
            meta_path = None

        output_cols = [c for c in COMMON_SCHEMA_COLUMNS if c in stitched_df.columns]
        extra_cols = [c for c in ["stitched_segment_role"] if c in stitched_df.columns]
        if output_cols:
            stitched_df = stitched_df[output_cols + extra_cols]

        out_path.parent.mkdir(parents=True, exist_ok=True)
        stitched_df.to_csv(out_path, index=False)

        print(f"Wrote {len(stitched_df)} rows to {out_path}")
        if meta_path is not None:
            print(f"Wrote meta: {meta_path}")
        print(json.dumps(summary, indent=2))
    elif args.mode == "stitched_adl_then_fall":
        if args.dataset != "mobifall":
            raise ValueError("--mode stitched_adl_then_fall is only supported for mobifall.")
        stitched_df, summary, report = _stitch_adl_then_fall_mobifall(
            dataset_path,
            max_rows=int(args.max_rows),
            gap_seconds=float(args.gap_seconds),
            seed=int(args.seed),
            target_rate=float(args.target_rate),
            window_size=int(args.window_size),
            step_size=int(args.step_size),
        )
        output_cols = [c for c in COMMON_SCHEMA_COLUMNS if c in stitched_df.columns]
        extra_cols = [c for c in ["stitched_segment_role"] if c in stitched_df.columns]
        if output_cols:
            stitched_df = stitched_df[output_cols + extra_cols]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        stitched_df.to_csv(out_path, index=False)
        report_path = out_path.with_suffix(".report.json")
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Wrote {len(stitched_df)} rows to {out_path}")
        print(f"Wrote report: {report_path}")
        print(json.dumps(summary, indent=2))
    elif args.mode == "stitched_adl_then_fall_strong":
        if args.dataset != "mobifall":
            raise ValueError("--mode stitched_adl_then_fall_strong is only supported for mobifall.")
        stitched_df, summary, report = _stitch_adl_then_fall_strong_mobifall(
            dataset_path,
            max_rows=int(args.max_rows),
            gap_seconds=float(args.gap_seconds),
            seed=int(args.seed),
            target_rate=float(args.target_rate),
            window_size=int(args.window_size),
            step_size=int(args.step_size),
        )
        output_cols = [c for c in COMMON_SCHEMA_COLUMNS if c in stitched_df.columns]
        extra_cols = [c for c in ["stitched_segment_role"] if c in stitched_df.columns]
        if output_cols:
            stitched_df = stitched_df[output_cols + extra_cols]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        stitched_df.to_csv(out_path, index=False)
        report_path = REPO_ROOT / "results" / "validation" / "mobifall_stitch_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Wrote {len(stitched_df)} rows to {out_path}")
        print(f"Wrote report: {report_path}")
        print(json.dumps(summary, indent=2))
    elif args.mode == "stitched_adl_fall_strong":
        if args.dataset != "mobifall":
            raise ValueError("--mode stitched_adl_fall_strong is only supported for mobifall.")
        stitched_df, summary, report = _stitch_adl_fall_strong_mobifall(
            dataset_path,
            max_rows_adl=int(args.max_rows_adl),
            max_rows_fall=int(args.max_rows_fall),
            min_fall_detections=int(args.min_fall_detections),
            target_rate=float(args.target_rate),
            window_size=int(args.window_size),
            step_size=int(args.step_size),
        )
        output_cols = [c for c in COMMON_SCHEMA_COLUMNS if c in stitched_df.columns]
        extra_cols = [c for c in ["stitched_segment_role"] if c in stitched_df.columns]
        if output_cols:
            stitched_df = stitched_df[output_cols + extra_cols]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        stitched_df.to_csv(out_path, index=False)
        report_path = out_path.with_suffix(".report.json")
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Wrote {len(stitched_df)} rows to {out_path}")
        print(f"Wrote report: {report_path}")
        print(json.dumps(summary, indent=2))
    elif args.mode == "stitched_adl_then_fall_balanced":
        if args.dataset != "mobifall":
            raise ValueError("--mode stitched_adl_then_fall_balanced is only supported for mobifall.")
        stitched_df, summary = _stitch_adl_then_fall_balanced_mobifall(
            dataset_path,
            max_rows=int(args.max_rows),
            seed=int(args.seed),
            target_rate=float(args.target_rate),
            window_size=int(args.window_size),
            step_size=int(args.step_size),
        )
        output_cols = [c for c in COMMON_SCHEMA_COLUMNS if c in stitched_df.columns]
        extra_cols = [c for c in ["stitched_segment_role"] if c in stitched_df.columns]
        if output_cols:
            stitched_df = stitched_df[output_cols + extra_cols]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        stitched_df.to_csv(out_path, index=False)
        print(f"Wrote {len(stitched_df)} rows to {out_path}")
        print(json.dumps(summary, indent=2))
    elif args.stitch:
        df = _load_dataset(args.dataset, dataset_path)
        df = _filter_required(df)
        stitched_df, summary = _stitch_segments(
            df,
            min_label_changes=int(args.min_label_changes),
            max_rows=int(args.max_rows),
            seed=int(args.seed),
            stitch_max_segments=int(args.stitch_max_segments),
            target_rate=args.target_rate,
        )
        output_cols = [c for c in COMMON_SCHEMA_COLUMNS if c in stitched_df.columns]
        extra_cols = [c for c in ["stitched_segment_index", "stitched_source_session"] if c in stitched_df.columns]
        if output_cols:
            stitched_df = stitched_df[output_cols + extra_cols]

        out_path.parent.mkdir(parents=True, exist_ok=True)
        stitched_df.to_csv(out_path, index=False)
        summary_path = Path(str(out_path) + ".summary.json")
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        print(f"Wrote {len(stitched_df)} rows to {out_path}")
        print(f"Wrote summary: {summary_path}")
    else:
        df = _load_dataset(args.dataset, dataset_path)
        df = _filter_required(df)
        df = _sort_dataframe(df)
        slice_df, info = _choose_slice(
            df,
            min_label_changes=int(args.min_label_changes),
            max_rows=int(args.max_rows),
            seed=int(args.seed),
        )

        output_cols = [c for c in COMMON_SCHEMA_COLUMNS if c in slice_df.columns]
        if output_cols:
            slice_df = slice_df[output_cols]

        out_path.parent.mkdir(parents=True, exist_ok=True)
        slice_df.to_csv(out_path, index=False)

        print(f"Wrote {len(slice_df)} rows to {out_path}")
        print(f"Selected slice info: {info}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
