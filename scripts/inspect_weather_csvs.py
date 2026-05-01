#!/usr/bin/env python3
"""Inspect weather CSV files (Open-Meteo / Meteostat style) for readiness auditing.

Supports:
- one or more CSV paths
- directory roots (recursively discovers *.csv)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.ingest.weather import load_weather_csv, load_weather_csvs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect weather CSVs (Open-Meteo/Meteostat style)")
    parser.add_argument("paths", nargs="+", help="One or more CSV paths and/or directories")
    parser.add_argument("--time-col", default="time")
    parser.add_argument("--head", type=int, default=5)
    parser.add_argument("--max-files", type=int, default=None, help="Limit discovered CSV files (after expansion)")
    parser.add_argument("--print-json", action="store_true", help="Print structured JSON summary")
    return parser.parse_args()


def _expand_paths(items: list[str]) -> list[Path]:
    out: list[Path] = []
    for item in items:
        path = Path(item)
        if not path.is_absolute():
            path = (REPO_ROOT / path).resolve()
        if not path.exists():
            print(f"WARN: missing path, skipping: {path}")
            continue
        if path.is_dir():
            out.extend(sorted(p for p in path.rglob("*.csv") if p.is_file()))
        else:
            out.append(path)

    seen: set[str] = set()
    deduped: list[Path] = []
    for p in out:
        key = str(p.resolve())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(p)
    return deduped


def _per_file_summary(df, *, time_col: str) -> dict:
    out = {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "null_counts": {str(k): int(v) for k, v in df.isna().sum().items()},
        "location_name": str(df["location_name"].iloc[0]) if "location_name" in df.columns and not df.empty else None,
        "source_file": str(df["source_file"].iloc[0]) if "source_file" in df.columns and not df.empty else None,
    }
    if time_col in df.columns:
        out["time_min"] = str(df[time_col].min()) if not df.empty else None
        out["time_max"] = str(df[time_col].max()) if not df.empty else None
        out["duplicate_timestamp_count"] = int(df.duplicated(subset=[time_col]).sum())
    if "pressure_msl" in df.columns:
        pressure = df["pressure_msl"]
        out["pressure_msl_stats"] = {
            "count": int(pressure.notna().sum()),
            "mean": float(pressure.mean()) if pressure.notna().any() else None,
            "min": float(pressure.min()) if pressure.notna().any() else None,
            "max": float(pressure.max()) if pressure.notna().any() else None,
        }
    return out


def main() -> int:
    args = parse_args()
    paths = _expand_paths(args.paths)
    if args.max_files is not None:
        paths = paths[: max(0, args.max_files)]

    if not paths:
        print("ERROR: no existing weather CSV paths provided")
        return 1

    per_file = []
    failures = []
    loaded_paths: list[Path] = []
    for path in paths:
        try:
            df_file = load_weather_csv(path, time_col=args.time_col)
            per_file.append(_per_file_summary(df_file, time_col=args.time_col))
            loaded_paths.append(path)
        except Exception as exc:  # noqa: BLE001
            failures.append({"file": str(path), "error": f"{type(exc).__name__}: {exc}"})

    if not loaded_paths:
        print("ERROR: failed to load all provided weather CSVs")
        if failures:
            print(json.dumps({"failures": failures}, indent=2))
        return 1

    df = load_weather_csvs(loaded_paths, time_col=args.time_col)

    duplicate_ts_overall = (
        int(df.duplicated(subset=["location_name", args.time_col]).sum())
        if args.time_col in df.columns and "location_name" in df.columns
        else None
    )

    by_location = {}
    if "location_name" in df.columns:
        for loc, g in df.groupby("location_name", dropna=False, sort=True):
            entry = {
                "rows": int(len(g)),
                "files": int(g["source_file"].nunique(dropna=True)) if "source_file" in g.columns else None,
                "time_min": str(g[args.time_col].min()) if args.time_col in g.columns else None,
                "time_max": str(g[args.time_col].max()) if args.time_col in g.columns else None,
                "duplicate_timestamp_count": int(g.duplicated(subset=[args.time_col]).sum()) if args.time_col in g.columns else None,
            }
            if "pressure_msl" in g.columns:
                pressure = g["pressure_msl"]
                entry["pressure_msl_stats"] = {
                    "count": int(pressure.notna().sum()),
                    "mean": float(pressure.mean()) if pressure.notna().any() else None,
                    "min": float(pressure.min()) if pressure.notna().any() else None,
                    "max": float(pressure.max()) if pressure.notna().any() else None,
                }
            by_location[str(loc)] = entry

    summary = {
        "file_count_input": int(len(paths)),
        "files_loaded_successfully": int(len(loaded_paths)),
        "files_failed": int(len(failures)),
        "failures": failures[:10],
        "rows_total": int(len(df)),
        "columns": list(df.columns),
        "null_counts": {str(k): int(v) for k, v in df.isna().sum().items()},
        "time_min": str(df[args.time_col].min()) if args.time_col in df.columns else None,
        "time_max": str(df[args.time_col].max()) if args.time_col in df.columns else None,
        "duplicate_timestamp_count_overall": duplicate_ts_overall,
        "locations": sorted(df["location_name"].astype(str).dropna().unique().tolist()) if "location_name" in df.columns else [],
        "per_location": by_location,
        "per_file_preview": per_file[:10],
    }

    if args.print_json:
        print(json.dumps(summary, indent=2))
        return 0

    print(f"file_count_input={summary['file_count_input']}")
    print(f"files_loaded_successfully={summary['files_loaded_successfully']}")
    print(f"files_failed={summary['files_failed']}")
    if failures:
        print("failures_preview:")
        print(json.dumps(failures[:5], indent=2))
    print(f"rows_total={summary['rows_total']} cols={len(summary['columns'])}")
    print("columns:", summary["columns"])
    print(f"time_min={summary['time_min']} time_max={summary['time_max']}")
    print(f"duplicate_timestamp_count_overall={summary['duplicate_timestamp_count_overall']}")
    print("locations:", summary["locations"])
    if "pressure_msl" in df.columns and df["pressure_msl"].notna().any():
        pressure = df["pressure_msl"]
        print(
            "pressure_msl_stats_overall:",
            {
                "count": int(pressure.notna().sum()),
                "mean": float(pressure.mean()),
                "min": float(pressure.min()),
                "max": float(pressure.max()),
            },
        )
    print("null_counts:")
    print(df.isna().sum().sort_values(ascending=False))
    print("\nper_location_summary:")
    print(json.dumps(by_location, indent=2))
    print("\nhead:")
    print(df.head(args.head))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
