#!/usr/bin/env python3
"""Real-data readiness audit for Chapter 3.

This script complements fixture tests by auditing *real dataset paths* for:
- file discovery vs successful loading
- metadata plausibility (subject/session/source_file)
- label mapping plausibility
- grouped resampling + grouped windowing behavior
- weather CSV parsing/quality checks

The audit is intentionally lightweight (sample-limited) so it can be run frequently.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Callable, Iterable

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.ingest import (  # noqa: E402
    load_mobifall,
    load_pamap2,
    load_sisfall,
    load_uci_har,
    load_weather_csv,
)
from pipeline.preprocess import (  # noqa: E402
    PreprocessConfig,
    append_derived_channels,
    estimate_sampling_rate,
    resample_dataframe,
    window_dataframe,
)
from pipeline.validation import validate_ingestion_dataframe  # noqa: E402


LoaderFn = Callable[..., pd.DataFrame]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a real-data readiness audit across available datasets")
    parser.add_argument("--mobifall-root", default="data/raw/MOBIACT_Dataset/MobiFall_Dataset_v2.0")
    parser.add_argument("--sisfall-root", default="data/raw/SISFALL_Dataset/SisFall_dataset")
    parser.add_argument("--ucihar-root", default="data/raw/UCIHAR_Dataset/UCI-HAR Dataset")
    parser.add_argument("--pamap2-root", default="data/raw/PAMAP2_Dataset")
    parser.add_argument("--weather-root", default="data/raw/weather")
    parser.add_argument("--sample-limit", type=int, default=3, help="Limit files/windows loaded per dataset for quick audits")
    parser.add_argument(
        "--full-audit",
        action="store_true",
        help="Process all discovered files for supported datasets (equivalent to --sample-limit 0).",
    )
    parser.add_argument(
        "--pamap2-include-optional",
        action="store_true",
        help="Include PAMAP2 Optional/ subject*.dat files in addition to Protocol/ files.",
    )
    parser.add_argument("--target-rate", type=float, default=50.0)
    parser.add_argument("--window-size", type=int, default=None, help="Override window size in samples")
    parser.add_argument("--step-size", type=int, default=None, help="Override step size in samples")
    parser.add_argument("--json-out", default=None, help="Optional JSON output file for the full audit report")
    parser.add_argument("--print-pretty", action="store_true", help="Pretty-print summaries (default is compact JSON per dataset)")
    return parser.parse_args()


def _resolve_path(path_str: str | None) -> Path | None:
    if path_str is None:
        return None
    p = Path(path_str)
    if not p.is_absolute():
        p = (REPO_ROOT / p).resolve()
    return p


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _series_counts(series: pd.Series, limit: int | None = None) -> dict[str, int]:
    counts = series.astype(str).value_counts(dropna=False)
    if limit is not None:
        counts = counts.head(limit)
    return {str(k): int(v) for k, v in counts.items()}


def _null_counts(df: pd.DataFrame, cols: Iterable[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for col in cols:
        if col in df.columns:
            out[col] = int(df[col].isna().sum())
    return out


def _default_window_sizes(df: pd.DataFrame, cfg: PreprocessConfig, window_size: int | None, step_size: int | None) -> tuple[int, int]:
    if window_size is not None:
        return int(window_size), int(step_size or max(1, window_size // 2))

    # For sample-limited audit slices, pick a small per-group-friendly window to ensure
    # at least one window can be generated if groups are short.
    group_cols = [c for c in ["dataset_name", "subject_id", "session_id", "source_file"] if c in df.columns]
    min_group_size: int | None = None
    if len(df) > 0 and group_cols:
        try:
            min_group_size = int(df.groupby(group_cols, dropna=False, sort=False).size().min())
        except Exception:
            min_group_size = None

    if len(df) >= cfg.window_size_samples and (min_group_size is None or min_group_size >= cfg.window_size_samples):
        return cfg.window_size_samples, cfg.step_size_samples

    if min_group_size is not None:
        w = max(2, min(32, min_group_size))
    else:
        w = max(2, min(32, len(df)))
    s = int(step_size or max(1, w // 2))
    return int(w), s


def _sampling_rate_group_summary(df: pd.DataFrame, max_groups: int = 10) -> dict:
    group_cols = [c for c in ["dataset_name", "subject_id", "session_id", "source_file"] if c in df.columns]
    if "timestamp" not in df.columns or df.empty:
        return {"groups_checked": 0, "estimated_rates_hz": [], "median_hz": None, "min_hz": None, "max_hz": None}

    rates: list[float] = []
    if group_cols:
        for idx, (_, g) in enumerate(df.groupby(group_cols, dropna=False, sort=False)):
            if idx >= max_groups:
                break
            rate = estimate_sampling_rate(g)
            if rate is not None:
                rates.append(float(rate))
    else:
        rate = estimate_sampling_rate(df)
        if rate is not None:
            rates.append(float(rate))

    if not rates:
        return {"groups_checked": 0, "estimated_rates_hz": [], "median_hz": None, "min_hz": None, "max_hz": None}
    s = pd.Series(rates, dtype=float)
    return {
        "groups_checked": int(len(rates)),
        "estimated_rates_hz": [round(float(v), 6) for v in rates],
        "median_hz": float(s.median()),
        "min_hz": float(s.min()),
        "max_hz": float(s.max()),
    }


def _preprocess_audit(df: pd.DataFrame, *, target_rate: float, window_size: int | None, step_size: int | None) -> dict:
    cfg = PreprocessConfig(target_sampling_rate_hz=target_rate)
    rows_before = int(len(df))

    resampled = resample_dataframe(df, target_rate_hz=target_rate)
    resampled = append_derived_channels(resampled, include_acc=True, include_gyro=True)
    w_size, s_size = _default_window_sizes(resampled, cfg, window_size, step_size)
    windows = window_dataframe(resampled, window_size=w_size, step_size=s_size, config=cfg)

    return {
        "rows_before_resampling": rows_before,
        "rows_after_resampling": int(len(resampled)),
        "window_size_samples": int(w_size),
        "step_size_samples": int(s_size),
        "windows_total": int(len(windows)),
        "windows_accepted": int(sum(1 for w in windows if bool(w.get("is_acceptable")))),
        "windows_rejected": int(sum(1 for w in windows if not bool(w.get("is_acceptable")))),
        "window_majority_label_counts": (
            _series_counts(pd.DataFrame({"label": [w.get("label_mapped_majority") for w in windows]})["label"])
            if windows
            else {}
        ),
        "window_quality_preview": [
            {
                "window_id": int(w.get("window_id", -1)),
                "subject_id": _json_safe(w.get("subject_id")),
                "session_id": _json_safe(w.get("session_id")),
                "source_file": _json_safe(w.get("source_file")),
                "label_mapped_majority": _json_safe(w.get("label_mapped_majority")),
                "missing_ratio": float(w.get("missing_ratio", 0.0)),
                "is_acceptable": bool(w.get("is_acceptable", False)),
                "has_large_gap": bool(w.get("has_large_gap", False)),
                "n_gaps": int(w.get("n_gaps", 0)),
            }
            for w in windows[:5]
        ],
    }


def _basic_inertial_dataframe_summary(df: pd.DataFrame) -> dict:
    summary = {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "subjects_count": int(df["subject_id"].nunique(dropna=True)) if "subject_id" in df.columns else 0,
        "sessions_count": int(df["session_id"].nunique(dropna=True)) if "session_id" in df.columns else 0,
        "source_files_count": int(df["source_file"].nunique(dropna=True)) if "source_file" in df.columns else 0,
        "label_raw_counts": _series_counts(df["label_raw"]) if "label_raw" in df.columns else {},
        "label_mapped_counts": _series_counts(df["label_mapped"]) if "label_mapped" in df.columns else {},
        "null_counts_key_columns": _null_counts(
            df, ["timestamp", "ax", "ay", "az", "gx", "gy", "gz", "label_raw", "label_mapped", "subject_id", "session_id", "source_file"]
        ),
        "estimated_sampling_rate_summary": _sampling_rate_group_summary(df),
    }
    if hasattr(df, "attrs") and df.attrs:
        prewindowed = bool(df.attrs.get("is_prewindowed", False))
        if prewindowed:
            summary["prewindowed_source"] = True
            summary["prewindowed_note"] = str(df.attrs.get("prewindowed_source", "pre-windowed source"))
    return summary


def _validation_summary(df: pd.DataFrame) -> dict:
    result = validate_ingestion_dataframe(df)
    return {
        "is_valid": bool(result.is_valid),
        "errors": list(result.errors),
        "warnings": list(result.warnings),
    }


def _discover_mobifall_files(root: Path) -> list[Path]:
    return sorted(root.rglob("*_acc_*.txt"))


def _discover_sisfall_files(root: Path) -> list[Path]:
    out: list[Path] = []
    for p in sorted(root.rglob("*.txt")):
        if p.name.lower() == "readme.txt":
            continue
        if not p.parent.name.upper().startswith(("SA", "SE")):
            continue
        out.append(p)
    return out


def _audit_file_based_dataset(
    *,
    dataset_name: str,
    root: Path,
    discover_files: Callable[[Path], list[Path]],
    file_loader: LoaderFn,
    sample_limit: int,
    target_rate: float,
    window_size: int | None,
    step_size: int | None,
    discovery_pattern_note: str | None = None,
    file_loader_kwargs: dict | None = None,
) -> dict:
    discovered = discover_files(root)
    files_considered = discovered[: sample_limit if sample_limit > 0 else None]
    frames: list[pd.DataFrame] = []
    failures: list[dict] = []
    file_loader_kwargs = file_loader_kwargs or {}

    for path in files_considered:
        try:
            frames.append(file_loader(path, **file_loader_kwargs))
        except Exception as exc:  # noqa: BLE001 - audit should collect and continue
            failures.append({"file": str(path), "error": f"{type(exc).__name__}: {exc}"})

    if frames:
        df = pd.concat(frames, ignore_index=True)
        validation = _validation_summary(df)
        preprocess = _preprocess_audit(df, target_rate=target_rate, window_size=window_size, step_size=step_size)
        data_summary = _basic_inertial_dataframe_summary(df)
    else:
        df = None
        validation = {"is_valid": False, "errors": ["No files loaded successfully"], "warnings": []}
        preprocess = {}
        data_summary = {}

    return {
        "dataset": dataset_name,
        "root_path": str(root),
        "status": "ok" if frames else "failed",
        "discovery": {
            "files_discovered": int(len(discovered)),
            "files_considered": int(len(files_considered)),
            "sample_limit": int(sample_limit),
            "discovery_pattern_note": discovery_pattern_note or "dataset-specific file discovery",
        },
        "loading": {
            "files_loaded_successfully": int(len(frames)),
            "files_failed": int(len(failures)),
            "failed_files_preview": failures[:10],
        },
        "validation": validation,
        "data_summary": data_summary,
        "preprocessing_audit": preprocess,
        "notes": [],
    }


def _audit_ucihar(
    root: Path,
    *,
    sample_limit: int,
    target_rate: float,
    window_size: int | None,
    step_size: int | None,
) -> dict:
    discovered_files = sorted(root.rglob("*.txt"))
    try:
        df = load_uci_har(root, max_windows_per_split=sample_limit)
        validation = _validation_summary(df)
        data_summary = _basic_inertial_dataframe_summary(df)
        preprocess = _preprocess_audit(df, target_rate=target_rate, window_size=window_size, step_size=step_size)
        status = "ok"
        error = None
    except Exception as exc:  # noqa: BLE001
        df = None
        validation = {"is_valid": False, "errors": [f"{type(exc).__name__}: {exc}"], "warnings": []}
        data_summary = {}
        preprocess = {}
        status = "failed"
        error = f"{type(exc).__name__}: {exc}"

    notes = [
        "UCI HAR is treated as a pre-windowed dataset and flattened to sample rows for common-schema compatibility.",
        "source_file values refer to split inertial-signal directories, not individual rows/windows files.",
    ]
    if error:
        notes.append(error)

    return {
        "dataset": "UCIHAR",
        "root_path": str(root),
        "status": status,
        "discovery": {
            "files_discovered": int(len(discovered_files)),
            "files_considered": int(len(discovered_files)),
            "sample_limit": int(sample_limit),
        },
        "loading": {
            "files_loaded_successfully": int(data_summary.get("source_files_count", 0)) if data_summary else 0,
            "files_failed": 0 if status == "ok" else 1,
            "failed_files_preview": [],
        },
        "validation": validation,
        "data_summary": data_summary,
        "preprocessing_audit": preprocess,
        "notes": notes,
    }


def _audit_pamap2(
    root: Path,
    *,
    sample_limit: int,
    target_rate: float,
    window_size: int | None,
    step_size: int | None,
    include_optional: bool = False,
) -> dict:
    if not root.exists():
        return _missing_dataset_summary("PAMAP2", root)

    # Discover Protocol first; Optional inclusion can be enabled explicitly.
    def _discover_pamap2(root_path: Path) -> list[Path]:
        candidates: list[Path] = []
        if (root_path / "Protocol").exists():
            candidates.extend(sorted((root_path / "Protocol").glob("subject*.dat")))
        elif root_path.name.lower() == "protocol":
            candidates.extend(sorted(root_path.glob("subject*.dat")))
        else:
            # Direct file or fallback folder of .dat files
            if root_path.is_file() and root_path.suffix.lower() == ".dat":
                candidates.append(root_path)
            else:
                candidates.extend(sorted(root_path.glob("subject*.dat")))

        if include_optional:
            if (root_path / "Optional").exists():
                candidates.extend(sorted((root_path / "Optional").glob("subject*.dat")))
            elif root_path.name.lower() == "optional":
                # already covered above if root is Optional
                pass
        return candidates

    summary = _audit_file_based_dataset(
        dataset_name="PAMAP2",
        root=root,
        discover_files=_discover_pamap2,
        file_loader=load_pamap2,
        sample_limit=sample_limit,
        target_rate=target_rate,
        window_size=window_size,
        step_size=step_size,
        discovery_pattern_note="PAMAP2 Protocol/subject*.dat files (Optional included only if requested)",
    )
    summary["notes"] = [
        "PAMAP2 real parser uses hand IMU acc16 + gyroscope columns for the common schema.",
        "Protocol folder is audited by default; Optional folder can be included explicitly.",
    ]
    if include_optional:
        summary["notes"].append("Optional folder inclusion enabled for this audit run.")
    return summary


def _weather_file_summary(df: pd.DataFrame, *, time_col: str = "time") -> dict:
    out: dict = {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "null_counts": {str(k): int(v) for k, v in df.isna().sum().items()},
    }
    if time_col in df.columns:
        out["time_min"] = _json_safe(df[time_col].min())
        out["time_max"] = _json_safe(df[time_col].max())
        out["duplicate_timestamp_count"] = int(df.duplicated(subset=[time_col]).sum())
    if "pressure_msl" in df.columns:
        series = pd.to_numeric(df["pressure_msl"], errors="coerce")
        out["pressure_msl_stats"] = {
            "count": int(series.notna().sum()),
            "mean": _json_safe(series.mean()),
            "min": _json_safe(series.min()),
            "max": _json_safe(series.max()),
        }
    if "location_name" in df.columns:
        out["location_name"] = str(df["location_name"].iloc[0]) if not df.empty else None
    if "source_file" in df.columns:
        out["source_file"] = str(df["source_file"].iloc[0]) if not df.empty else None
    return out


def _discover_weather_csvs(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.csv") if p.is_file())


def _audit_weather_root(root: Path, *, sample_limit: int) -> dict:
    discovered = _discover_weather_csvs(root)
    considered = discovered[: sample_limit if sample_limit > 0 else None]
    per_file: list[dict] = []
    failures: list[dict] = []
    loaded_frames: list[pd.DataFrame] = []

    for path in considered:
        try:
            df = load_weather_csv(path)
            loaded_frames.append(df)
            per_file.append(_weather_file_summary(df))
        except Exception as exc:  # noqa: BLE001
            failures.append({"file": str(path), "error": f"{type(exc).__name__}: {exc}"})

    combined = pd.concat(loaded_frames, ignore_index=True) if loaded_frames else pd.DataFrame()
    by_location = (
        {
            str(k): {
                "rows": int(len(g)),
                "files": int(g["source_file"].nunique(dropna=True)) if "source_file" in g.columns else None,
                "time_min": _json_safe(g["time"].min()) if "time" in g.columns else None,
                "time_max": _json_safe(g["time"].max()) if "time" in g.columns else None,
                "duplicate_timestamps": int(g.duplicated(subset=["time"]).sum()) if "time" in g.columns else None,
            }
            for k, g in combined.groupby("location_name", dropna=False, sort=True)
        }
        if not combined.empty and "location_name" in combined.columns
        else {}
    )

    return {
        "dataset": "WEATHER_CSVS",
        "root_path": str(root),
        "status": "ok" if (loaded_frames or not considered) else "failed",
        "discovery": {
            "files_discovered": int(len(discovered)),
            "files_considered": int(len(considered)),
            "sample_limit": int(sample_limit),
        },
        "loading": {
            "files_loaded_successfully": int(len(loaded_frames)),
            "files_failed": int(len(failures)),
            "failed_files_preview": failures[:10],
        },
        "data_summary": {
            "combined_rows": int(len(combined)),
            "locations_count": int(combined["location_name"].nunique(dropna=True)) if not combined.empty and "location_name" in combined.columns else 0,
            "per_location": by_location,
            "per_file_preview": per_file[:10],
        },
        "notes": [
            "Weather audit checks parsing/time/nulls/duplicates only. Context alignment and feature engineering are later chapters."
        ],
    }


def _audit_inertial_dataset(
    *,
    dataset_name: str,
    root: Path,
    sample_limit: int,
    target_rate: float,
    window_size: int | None,
    step_size: int | None,
) -> dict:
    if dataset_name == "MOBIFALL":
        return _audit_file_based_dataset(
            dataset_name=dataset_name,
            root=root,
            discover_files=_discover_mobifall_files,
            file_loader=load_mobifall,
            sample_limit=sample_limit,
            target_rate=target_rate,
            window_size=window_size,
            step_size=step_size,
            discovery_pattern_note="MobiFall accelerometer files discovered recursively (*_acc_*.txt)",
        )
    if dataset_name == "SISFALL":
        return _audit_file_based_dataset(
            dataset_name=dataset_name,
            root=root,
            discover_files=_discover_sisfall_files,
            file_loader=load_sisfall,
            sample_limit=sample_limit,
            target_rate=target_rate,
            window_size=window_size,
            step_size=step_size,
            discovery_pattern_note="SisFall subject text files under SA*/SE* folders",
        )
    if dataset_name == "UCIHAR":
        return _audit_ucihar(
            root,
            sample_limit=sample_limit,
            target_rate=target_rate,
            window_size=window_size,
            step_size=step_size,
        )
    raise ValueError(dataset_name)


def _missing_dataset_summary(name: str, root: Path | None) -> dict:
    return {
        "dataset": name,
        "root_path": str(root) if root is not None else None,
        "status": "skipped",
        "notes": ["Missing path / skipped"],
    }


def _print_dataset_summary(summary: dict, *, pretty: bool) -> None:
    if pretty:
        print(f"\n=== {summary.get('dataset')} ===")
        print(json.dumps(_json_safe(summary), indent=2, default=str))
    else:
        print(json.dumps(_json_safe(summary), default=str))


def main() -> int:
    args = parse_args()
    if args.full_audit:
        args.sample_limit = 0

    mobifall_root = _resolve_path(args.mobifall_root)
    sisfall_root = _resolve_path(args.sisfall_root)
    ucihar_root = _resolve_path(args.ucihar_root)
    pamap2_root = _resolve_path(args.pamap2_root)
    weather_root = _resolve_path(args.weather_root)

    report = {
        "audit_name": "lesson4_real_data_readiness_audit",
        "repo_root": str(REPO_ROOT),
        "config": {
            "sample_limit": int(args.sample_limit),
            "target_rate": float(args.target_rate),
            "window_size": args.window_size,
            "step_size": args.step_size,
        },
        "datasets": [],
        "notes": [
            "Fixture tests are useful but do not prove real-data readiness.",
            "This audit checks discovery/loading/metadata/preprocessing behavior on real paths or reports clear skips.",
        ],
    }

    dataset_roots = [
        ("MOBIFALL", mobifall_root),
        ("SISFALL", sisfall_root),
        ("UCIHAR", ucihar_root),
    ]
    for dataset_name, root in dataset_roots:
        if root is None or not root.exists():
            summary = _missing_dataset_summary(dataset_name, root)
        else:
            try:
                summary = _audit_inertial_dataset(
                    dataset_name=dataset_name,
                    root=root,
                    sample_limit=args.sample_limit,
                    target_rate=args.target_rate,
                    window_size=args.window_size,
                    step_size=args.step_size,
                )
            except Exception as exc:  # noqa: BLE001
                summary = {
                    "dataset": dataset_name,
                    "root_path": str(root),
                    "status": "failed",
                    "notes": [f"Unhandled audit error: {type(exc).__name__}: {exc}"],
                }
        report["datasets"].append(summary)
        _print_dataset_summary(summary, pretty=args.print_pretty)

    # PAMAP2 Protocol parser is supported; Optional files can be included explicitly.
    if pamap2_root is None or not pamap2_root.exists():
        pamap2_summary = _missing_dataset_summary("PAMAP2", pamap2_root)
    else:
        pamap2_summary = _audit_pamap2(
            pamap2_root,
            sample_limit=args.sample_limit,
            target_rate=args.target_rate,
            window_size=args.window_size,
            step_size=args.step_size,
            include_optional=args.pamap2_include_optional,
        )
    report["datasets"].append(pamap2_summary)
    _print_dataset_summary(pamap2_summary, pretty=args.print_pretty)

    # Weather CSVs
    if weather_root is None or not weather_root.exists():
        weather_summary = _missing_dataset_summary("WEATHER_CSVS", weather_root)
    else:
        try:
            weather_sample_limit = 0 if args.sample_limit == 0 else max(1, args.sample_limit)
            weather_summary = _audit_weather_root(weather_root, sample_limit=weather_sample_limit)
        except Exception as exc:  # noqa: BLE001
            weather_summary = {
                "dataset": "WEATHER_CSVS",
                "root_path": str(weather_root),
                "status": "failed",
                "notes": [f"Unhandled audit error: {type(exc).__name__}: {exc}"],
            }
    report["datasets"].append(weather_summary)
    _print_dataset_summary(weather_summary, pretty=args.print_pretty)

    if args.json_out:
        out_path = Path(args.json_out)
        if not out_path.is_absolute():
            out_path = (REPO_ROOT / out_path).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(_json_safe(report), indent=2, default=str), encoding="utf-8")
        print(f"\nFull audit report written to: {out_path}")

    # The script succeeds even when some datasets are skipped or parser-not-implemented,
    # because readiness auditing should be runnable incrementally during development.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
