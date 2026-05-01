#!/usr/bin/env python3
"""Build cached fall feature tables for faster, reproducible model training."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_fall_artifact_train import (  # noqa: E402
    DEFAULT_STEP_SIZE,
    DEFAULT_TARGET_RATE_HZ,
    DEFAULT_WINDOW_SIZE,
    _json_safe,
    _load_and_feature_extract,
    _resolve_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mobifall-path", default="data/raw/MOBIACT_Dataset/MobiFall_Dataset_v2.0")
    parser.add_argument("--sisfall-path", default="data/raw/SISFALL_Dataset/SisFall_dataset")
    parser.add_argument("--target-rate", type=float, default=DEFAULT_TARGET_RATE_HZ)
    parser.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE)
    parser.add_argument("--step-size", type=int, default=DEFAULT_STEP_SIZE)
    parser.add_argument("--out-dir", default="data/processed/fall_features")
    parser.add_argument(
        "--format",
        choices=["parquet", "csv"],
        default="parquet",
        help="Parquet is preferred; use csv if pyarrow/fastparquet is unavailable.",
    )
    return parser.parse_args()


def _write_table(df: pd.DataFrame, path: Path, fmt: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "parquet":
        try:
            df.to_parquet(path, index=False)
        except ImportError as exc:
            raise RuntimeError(
                "Writing parquet requires pyarrow or fastparquet. "
                "Install one of them or rerun with --format csv."
            ) from exc
    else:
        df.to_csv(path, index=False)


def main() -> int:
    args = parse_args()
    mobifall_path = _resolve_path(args.mobifall_path)
    sisfall_path = _resolve_path(args.sisfall_path)
    out_dir = _resolve_path(args.out_dir)

    if not mobifall_path.exists():
        raise FileNotFoundError(f"MobiFall path not found: {mobifall_path}")
    if not sisfall_path.exists():
        raise FileNotFoundError(f"SisFall path not found: {sisfall_path}")

    suffix = "parquet" if args.format == "parquet" else "csv"
    stem = f"{int(args.target_rate)}hz_w{args.window_size}_s{args.step_size}"

    print("Building cached MobiFall fall feature table...")
    mobifall_df, mobifall_summary = _load_and_feature_extract(
        "mobifall",
        mobifall_path,
        target_rate=args.target_rate,
        window_size=args.window_size,
        step_size=args.step_size,
    )

    print("Building cached SisFall fall feature table...")
    sisfall_df, sisfall_summary = _load_and_feature_extract(
        "sisfall",
        sisfall_path,
        target_rate=args.target_rate,
        window_size=args.window_size,
        step_size=args.step_size,
    )

    combined_df = pd.concat([mobifall_df, sisfall_df], ignore_index=True)

    mobifall_out = out_dir / f"mobifall_{stem}.{suffix}"
    sisfall_out = out_dir / f"sisfall_{stem}.{suffix}"
    combined_out = out_dir / f"combined_{stem}.{suffix}"
    metadata_out = out_dir / "metadata.json"

    _write_table(mobifall_df, mobifall_out, args.format)
    _write_table(sisfall_df, sisfall_out, args.format)
    _write_table(combined_df, combined_out, args.format)

    metadata: dict[str, Any] = {
        "feature_table_id": f"fall_features_{stem}",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "format": args.format,
        "config": {
            "target_rate_hz": float(args.target_rate),
            "window_size": int(args.window_size),
            "step_size": int(args.step_size),
            "mobifall_path": str(mobifall_path),
            "sisfall_path": str(sisfall_path),
        },
        "outputs": {
            "mobifall": str(mobifall_out),
            "sisfall": str(sisfall_out),
            "combined": str(combined_out),
        },
        "summaries": {
            "mobifall": mobifall_summary,
            "sisfall": sisfall_summary,
            "combined_rows": int(len(combined_df)),
            "combined_label_counts": combined_df["true_label"].astype(str).value_counts(dropna=False).to_dict(),
        },
    }
    metadata_out.write_text(json.dumps(_json_safe(metadata), indent=2), encoding="utf-8")

    print(f"Saved MobiFall features -> {mobifall_out}")
    print(f"Saved SisFall features -> {sisfall_out}")
    print(f"Saved combined features -> {combined_out}")
    print(f"Saved metadata -> {metadata_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
