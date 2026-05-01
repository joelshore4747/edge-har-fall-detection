#!/usr/bin/env python3
"""Inspect window metadata for a common-schema CSV after Chapter 3 preprocessing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.preprocess import PreprocessConfig, append_derived_channels, resample_dataframe, window_dataframe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect window metadata for a CSV file")
    parser.add_argument("path", nargs="?", default="tests/fixtures/timeseries_regular.csv", help="Path to common-schema CSV")
    parser.add_argument("--target-rate", type=float, default=50.0)
    parser.add_argument("--window-size", type=int, default=4, help="Window size in samples")
    parser.add_argument("--step-size", type=int, default=2, help="Step size in samples")
    parser.add_argument("--limit", type=int, default=5, help="Number of windows to print")
    return parser.parse_args()


def _to_printable(window: dict) -> dict:
    printable = dict(window)
    payload = printable.pop("sensor_payload", {})
    printable["sensor_payload_keys"] = sorted(payload.keys())
    printable["sensor_payload_lengths"] = {k: int(len(v)) for k, v in payload.items()}
    return printable


def main() -> int:
    args = parse_args()
    path = Path(args.path)
    if not path.exists():
        print(f"ERROR: file not found: {path}")
        return 1

    df = pd.read_csv(path)
    resampled = resample_dataframe(df, target_rate_hz=args.target_rate)
    derived = append_derived_channels(resampled)
    cfg = PreprocessConfig(target_sampling_rate_hz=args.target_rate)
    windows = window_dataframe(derived, window_size=args.window_size, step_size=args.step_size, config=cfg)

    print(f"windows_total={len(windows)}")
    for window in windows[: args.limit]:
        print(json.dumps(_to_printable(window), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
