"""Package a trained run into a deployable, versioned runtime bundle.

The bundle layout is

    {
        "schema_version": "v1",
        "model": <CalibratedClassifierCV or RandomForest>,
        "feature_cols": [...],
        "labels": [...],
        "smoothing": {"mode": "rolling_mean", "window": 5},
        "fall_threshold": 0.6,
        "metadata": {
            "target_rate_hz": 50.0,
            "window_size": 100,
            "step_size": 50,
            "pocket_window_size": 200,
        },
        "trained_at_iso": "...",
        "git_sha": "...",
        "run_id": "...",
    }

It's loaded by ``services.runtime_model_loader.load_bundle`` and refused if
the schema version doesn't match. That tight coupling is intentional —
silently changing what's inside the joblib while keeping the same filename
is the easy way to ship a broken model.

The ``metadata`` block carries the training-time preprocess params so the
inference path (``services.runtime_inference._artifact_har_preprocess``)
windows the live signal at the same rate/size the model was trained on.
Without it, the legacy loader silently falls back to ``window_size=128``
and inference computes all 107 features over the wrong window length.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.lib.run_registry import resolve_current_run  # noqa: E402

logger = logging.getLogger("export_runtime_artifact")

SCHEMA_VERSION = "v1"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--run",
        default="current",
        help=(
            "Run id to package. ``current`` (default) follows the symlink. "
            "Pass an explicit id to package an older run."
        ),
    )
    p.add_argument(
        "--runs-root",
        type=Path,
        default=Path("artifacts/unifallmonitor/runs"),
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/unifallmonitor/deployable/runtime_v1.joblib"),
    )
    p.add_argument(
        "--fall-threshold",
        type=float,
        default=None,
        help=(
            "Operating threshold for the binary fall decision at inference "
            "time. When omitted, the F1-optimal threshold is read from the "
            "matching ``experiments/B_fall_threshold_sweep.csv`` (smoothed "
            "row only); falls back to 0.5."
        ),
    )
    p.add_argument(
        "--smoothing-mode",
        default="rolling_mean",
        choices=("none", "rolling_mean", "hmm"),
    )
    p.add_argument(
        "--smoothing-window",
        type=int,
        default=5,
    )
    return p.parse_args()


def _resolve_run_dir(*, run: str, runs_root: Path) -> Path:
    runs_root = runs_root.resolve()
    if run == "current":
        target = resolve_current_run(runs_root=runs_root)
        if target is None:
            raise SystemExit(
                f"No 'current' run resolved under {runs_root.parent}. "
                f"Train a run with --mark-current, or pass --run <id>."
            )
        return target
    candidate = runs_root / run
    if not candidate.exists():
        raise SystemExit(f"Run {run} not found under {runs_root}")
    return candidate.resolve()


def _build_bundle_metadata(train_args: dict[str, Any], *, run_dir: Path) -> dict[str, Any]:
    """Hoist preprocess params from a training run's metadata into a flat dict.

    ``services.runtime_inference._artifact_har_preprocess`` reads
    ``target_rate_hz`` / ``window_size`` / ``step_size`` as flat keys under
    the artifact's ``metadata``. The training run's own ``metadata.json``
    nests them under ``args``, so we copy them up to the level the inference
    path expects. Refuses to export if any required key is missing — better
    to fail at packaging time than to ship a bundle that silently falls back
    to defaults at inference (the regression that produced the runtime_v1
    incident on 2026-04-30).
    """
    required = ("target_rate_hz", "window_size", "step_size")
    missing = [k for k in required if train_args.get(k) is None]
    if missing:
        raise SystemExit(
            f"Training metadata at {run_dir / 'metadata.json'} is missing "
            f"required preprocess keys under args: {missing}. Re-run training "
            f"with explicit values, or pass them via the training CLI."
        )

    out: dict[str, Any] = {
        "target_rate_hz": float(train_args["target_rate_hz"]),
        "window_size": int(train_args["window_size"]),
        "step_size": int(train_args["step_size"]),
    }
    pocket = train_args.get("pocket_window_size")
    if pocket is not None:
        out["pocket_window_size"] = int(pocket)
    return out


def _read_optimal_threshold(run_dir: Path) -> float | None:
    sweep_csv = run_dir / "experiments" / "B_fall_threshold_sweep.csv"
    if not sweep_csv.exists():
        return None
    import csv

    best_f1 = -1.0
    best_thr: float | None = None
    with sweep_csv.open() as f:
        for row in csv.DictReader(f):
            if row.get("score") and row["score"] != "smoothed_max_p_fall":
                continue
            try:
                thr = float(row["threshold"])
                f1 = float(row["f1"]) if row.get("f1") not in (None, "", "None") else -1.0
            except (TypeError, ValueError):
                continue
            if f1 > best_f1:
                best_f1, best_thr = f1, thr
    return best_thr


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s :: %(message)s")
    args = parse_args()

    run_dir = _resolve_run_dir(run=args.run, runs_root=args.runs_root)
    logger.info("Packaging run %s", run_dir)

    model_path = run_dir / "model.joblib"
    metadata_path = run_dir / "metadata.json"
    if not model_path.exists():
        raise SystemExit(f"Missing {model_path}")
    if not metadata_path.exists():
        raise SystemExit(f"Missing {metadata_path}")

    import joblib

    raw_model_bundle = joblib.load(model_path)
    if not isinstance(raw_model_bundle, dict):
        raise SystemExit(f"{model_path}: expected a dict bundle, got {type(raw_model_bundle).__name__}")
    model: Any = raw_model_bundle["model"]
    feature_cols: list[str] = list(raw_model_bundle["feature_cols"])
    labels: list[str] = list(raw_model_bundle["labels"])

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    train_args = dict(metadata.get("args") or {})
    bundle_metadata = _build_bundle_metadata(train_args, run_dir=run_dir)

    threshold = args.fall_threshold
    if threshold is None:
        threshold = _read_optimal_threshold(run_dir)
    if threshold is None:
        threshold = 0.5
        logger.info("No threshold sweep found; using fall_threshold=0.5 fallback")
    else:
        logger.info("Selected fall_threshold=%.3f", threshold)

    bundle = {
        "schema_version": SCHEMA_VERSION,
        "model": model,
        "feature_cols": feature_cols,
        "labels": labels,
        "smoothing": {
            "mode": args.smoothing_mode,
            "window": int(args.smoothing_window),
        },
        "fall_threshold": float(threshold),
        "metadata": bundle_metadata,
        "trained_at_iso": str(metadata.get("run_id") or datetime.now(timezone.utc).isoformat()),
        "git_sha": str(metadata.get("git_sha", "")),
        "run_id": str(metadata.get("run_id", run_dir.name)),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.out)
    logger.info(
        "Wrote bundle %s (schema=%s, run=%s, threshold=%.3f)",
        args.out, SCHEMA_VERSION, bundle["run_id"], bundle["fall_threshold"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
