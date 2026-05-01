#!/usr/bin/env python3
"""End-to-end smoke test for the runtime inference API.

Loads a local phone-export folder or CSV, builds the request payload expected by
apps/api/main.py, sends it to the running FastAPI server, and prints the result.

Use this before building Flutter so the API contract is proven first.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from urllib import request, error

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.ingest.runtime_phone_csv import RuntimePhoneCsvConfig, load_runtime_phone_csv
from pipeline.ingest.runtime_phone_folder import RuntimePhoneFolderConfig, load_runtime_phone_folder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test the runtime inference API")
    parser.add_argument(
        "--input-source",
        choices=["phone_folder", "csv"],
        default="phone_folder",
        help="Source type to load locally before POSTing to the API",
    )
    parser.add_argument(
        "--input-path",
        default="./phone1",
        help="Path to phone folder or CSV",
    )
    parser.add_argument(
        "--api-url",
        default="http://127.0.0.1:8000/v1/infer/session",
        help="Runtime inference endpoint",
    )
    parser.add_argument(
        "--session-id",
        default="phone1_api_test",
        help="Session id to send in metadata",
    )
    parser.add_argument(
        "--subject-id",
        default="joel",
        help="Subject id to send in metadata",
    )
    parser.add_argument(
        "--placement",
        default="pocket",
        help="Placement metadata",
    )
    parser.add_argument(
        "--dataset-name",
        default="APP_RUNTIME_TEST",
        help="Dataset name metadata",
    )
    parser.add_argument(
        "--device-platform",
        default="ios",
        help="Device platform metadata",
    )
    parser.add_argument(
        "--device-model",
        default="iPhone",
        help="Device model metadata",
    )
    parser.add_argument(
        "--limit-samples",
        type=int,
        default=0,
        help="Optional cap on number of samples sent (0 = all samples)",
    )
    parser.add_argument(
        "--include-har-windows",
        action="store_true",
        help="Ask API to return HAR windows",
    )
    parser.add_argument(
        "--include-fall-windows",
        action="store_true",
        help="Ask API to return fall windows",
    )
    parser.add_argument(
        "--save-response-json",
        default="",
        help="Optional path to save full API response JSON",
    )
    return parser.parse_args()


def _resolve(path_str: str) -> Path:
    return Path(path_str).expanduser().resolve()


def _load_local_source(args: argparse.Namespace) -> pd.DataFrame:
    input_path = _resolve(args.input_path)

    if args.input_source == "phone_folder":
        cfg = RuntimePhoneFolderConfig(
            dataset_name=args.dataset_name,
            task_type="runtime",
            subject_id=args.subject_id,
            session_id=args.session_id,
            placement=args.placement,
        )
        return load_runtime_phone_folder(input_path, config=cfg)

    if args.input_source == "csv":
        cfg = RuntimePhoneCsvConfig(
            dataset_name=args.dataset_name,
            task_type="runtime",
            subject_id=args.subject_id,
            session_id=args.session_id,
            placement=args.placement,
        )
        return load_runtime_phone_csv(input_path, config=cfg)

    raise ValueError(f"Unsupported input_source: {args.input_source}")


def _build_payload(df: pd.DataFrame, args: argparse.Namespace) -> dict:
    working = df.copy()

    if args.limit_samples > 0:
        working = working.head(int(args.limit_samples)).reset_index(drop=True)

    for col in ["gx", "gy", "gz"]:
        if col not in working.columns:
            working[col] = pd.NA

    sampling_rate_hz = None
    if "sampling_rate_hz" in working.columns:
        s = pd.to_numeric(working["sampling_rate_hz"], errors="coerce").dropna()
        if not s.empty:
            sampling_rate_hz = float(s.iloc[0])

    samples = []
    for _, row in working.iterrows():
        samples.append(
            {
                "timestamp": float(row["timestamp"]),
                "ax": float(row["ax"]),
                "ay": float(row["ay"]),
                "az": float(row["az"]),
                "gx": None if pd.isna(row["gx"]) else float(row["gx"]),
                "gy": None if pd.isna(row["gy"]) else float(row["gy"]),
                "gz": None if pd.isna(row["gz"]) else float(row["gz"]),
            }
        )

    payload = {
        "metadata": {
            "session_id": args.session_id,
            "subject_id": args.subject_id,
            "placement": args.placement,
            "task_type": "runtime",
            "dataset_name": args.dataset_name,
            "source_type": "mobile_app",
            "device_platform": args.device_platform,
            "device_model": args.device_model,
            "sampling_rate_hz": sampling_rate_hz,
            "notes": "API smoke test from local phone export",
        },
        "samples": samples,
        "include_har_windows": bool(args.include_har_windows),
        "include_fall_windows": bool(args.include_fall_windows),
        "include_combined_timeline": True,
        "include_grouped_fall_events": True,
    }
    return payload


def _post_json(url: str, payload: dict) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=300) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body)
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Could not reach API at {url}: {exc}") from exc


def main() -> int:
    args = parse_args()

    print("Loading local source...")
    df = _load_local_source(args)
    print(f"Loaded rows: {len(df)}")

    payload = _build_payload(df, args)
    print(f"Sending samples: {len(payload['samples'])}")
    print(f"POST {args.api_url}")

    response = _post_json(args.api_url, payload)

    if args.save_response_json:
        out_path = _resolve(args.save_response_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(response, indent=2), encoding="utf-8")
        print(f"Saved full response JSON to: {out_path}")

    alert = response.get("alert_summary", {})
    print()
    print("Alert summary")
    print("-------------")
    print(f"warning_level: {alert.get('warning_level')}")
    print(f"likely_fall_detected: {alert.get('likely_fall_detected')}")
    print(f"top_har_label: {alert.get('top_har_label')}")
    print(f"top_har_fraction: {alert.get('top_har_fraction')}")
    print(f"grouped_fall_event_count: {alert.get('grouped_fall_event_count')}")
    print(f"top_fall_probability: {alert.get('top_fall_probability')}")
    print(f"recommended_message: {alert.get('recommended_message')}")

    grouped = response.get("grouped_fall_events", [])
    timeline = response.get("combined_timeline", [])
    print()
    print("Response sizes")
    print("--------------")
    print(f"grouped_fall_events: {len(grouped)}")
    print(f"combined_timeline: {len(timeline)}")
    print(f"har_windows: {len(response.get('har_windows', []))}")
    print(f"fall_windows: {len(response.get('fall_windows', []))}")

    if grouped:
        print()
        print("Top grouped event")
        print("-----------------")
        print(json.dumps(grouped[0], indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())