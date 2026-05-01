#!/usr/bin/env python3
"""Chapter 3 preprocessing smoke test.

Runs fixture-friendly preprocessing steps:
1) (optional) Chapter 3 ingestion-schema validation
2) resampling
3) derived magnitudes
4) windowing
5) quality summary reporting
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.preprocess import (
    PreprocessConfig,
    append_derived_channels,
    estimate_sampling_rate,
    resample_dataframe,
    window_dataframe,
)
from pipeline.validation import validate_ingestion_dataframe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Chapter 3 preprocessing smoke test")
    parser.add_argument("paths", nargs="*", help="CSV files to process (common-schema style)")
    parser.add_argument("--use-fixtures", action="store_true", help="Use Chapter 3 fixture timeseries CSVs")
    parser.add_argument("--skip-lesson3-validation", action="store_true", help="Skip schema validation from Chapter 3")
    parser.add_argument("--target-rate", type=float, default=50.0, help="Target resampling rate (Hz)")
    parser.add_argument("--window-size-samples", type=int, default=None, help="Override window size in samples")
    parser.add_argument("--step-size-samples", type=int, default=None, help="Override step size in samples")
    parser.add_argument("--max-print-labels", type=int, default=10)
    return parser.parse_args()


def _default_fixture_paths() -> list[Path]:
    return [
        Path("tests/fixtures/timeseries_regular.csv"),
        Path("tests/fixtures/timeseries_irregular.csv"),
        Path("tests/fixtures/timeseries_missing_segments.csv"),
    ]


def _safe_window_sizes(n_rows: int, cfg: PreprocessConfig, override_window: int | None, override_step: int | None) -> tuple[int, int, str | None]:
    if override_window is not None:
        return override_window, override_step or max(1, override_window // 2), None

    if n_rows >= cfg.window_size_samples:
        return cfg.window_size_samples, (override_step or cfg.step_size_samples), None

    # Fixture-friendly fallback for short sample files.
    window = max(4, min(8, n_rows))
    step = override_step or max(1, window // 2)
    note = (
        f"using fixture-friendly window_size={window}, step_size={step} "
        f"(default would be {cfg.window_size_samples}/{cfg.step_size_samples})"
    )
    return window, step, note


def _label_counts(windows: list[dict], limit: int) -> list[tuple[str, int]]:
    counts: dict[str, int] = {}
    for w in windows:
        label = str(w.get("label_mapped_majority", "unknown"))
        counts[label] = counts.get(label, 0) + 1
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]


def process_file(path: Path, args: argparse.Namespace) -> bool:
    print(f"\n=== {path} ===")
    if not path.exists():
        print("WARN: file not found; skipping")
        return True

    df = pd.read_csv(path)

    if not args.skip_lesson3_validation:
        validation = validate_ingestion_dataframe(df)
        if not validation.is_valid:
            print("Chapter 3 validation: FAIL (continuing for preprocessing quality smoke test)")
            for err in validation.errors:
                print(f"  - {err}")
        else:
            print("Chapter 3 validation: PASS")
        for warning in validation.warnings:
            print(f"  warning: {warning}")

    est_rate = estimate_sampling_rate(df)
    print(f"estimated_sampling_rate_hz: {est_rate:.3f}" if est_rate is not None else "estimated_sampling_rate_hz: unknown")

    resampled = resample_dataframe(df, target_rate_hz=args.target_rate)
    print(f"rows_before: {len(df)}")
    print(f"rows_after_resample: {len(resampled)}")

    derived = append_derived_channels(resampled, include_acc=True, include_gyro=True)

    cfg = PreprocessConfig(target_sampling_rate_hz=args.target_rate)
    window_size, step_size, note = _safe_window_sizes(len(derived), cfg, args.window_size_samples, args.step_size_samples)
    if note:
        print(f"note: {note}")

    windows = window_dataframe(derived, window_size=window_size, step_size=step_size, config=cfg)
    accepted = sum(1 for w in windows if w["is_acceptable"])
    rejected = len(windows) - accepted

    print(f"windows_total: {len(windows)}")
    print(f"windows_accepted: {accepted}")
    print(f"windows_rejected: {rejected}")

    counts = _label_counts(windows, args.max_print_labels)
    print("window_majority_label_counts:")
    if not counts:
        print("  - none")
    else:
        for label, count in counts:
            print(f"  - {label}: {count}")
    return True


def main() -> int:
    args = parse_args()
    paths = [Path(p) for p in args.paths]
    if args.use_fixtures:
        paths = _default_fixture_paths()
    if not paths:
        paths = _default_fixture_paths()

    all_ok = True
    for path in paths:
        all_ok = process_file(path, args) and all_ok
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
