"""One-time migration: move ``har_classifier/`` into the ``runs/`` registry.

Before this change the trainer wrote a flat tree under
``artifacts/unifallmonitor/har_classifier/``. The new layout is

    artifacts/unifallmonitor/
      runs/<dated-id>/
        metadata.json, model.joblib, ...
      current -> runs/<dated-id>

This script preserves the prior outputs (so the baseline numbers in the
results writeup remain reproducible) by relocating them into a dated run
directory and pointing ``current`` at it. It is safe to run repeatedly —
already-migrated trees are detected and left alone.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.lib.run_registry import (  # noqa: E402
    register_run,
    update_current_symlink,
)

logger = logging.getLogger("migrate_unifallmonitor")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--root",
        type=Path,
        default=Path("artifacts/unifallmonitor"),
        help="Root of the unifallmonitor artifact tree",
    )
    p.add_argument(
        "--legacy-name",
        default="har_classifier",
        help="Legacy directory under --root to migrate from",
    )
    p.add_argument(
        "--also-experiments",
        action="store_true",
        help="Also relocate ``experiments/`` into the same run directory",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would happen but make no changes",
    )
    return p.parse_args()


def _resolve_run_id(metadata_path: Path) -> str:
    """Pick a deterministic run id for the legacy run.

    Use the metadata.json's `args.timestamp` if present, otherwise the file's
    mtime. Suffix with `_legacy` so the id is visually distinct from new
    runs created by the trainer post-migration.
    """
    if metadata_path.exists():
        try:
            data = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            data = {}
        if isinstance(data, dict) and isinstance(data.get("run_id"), str) and data["run_id"]:
            return data["run_id"]
    ts = datetime.fromtimestamp(metadata_path.stat().st_mtime, tz=timezone.utc) if metadata_path.exists() else datetime.now(timezone.utc)
    return ts.strftime("%Y-%m-%d_%H%M%S") + "_legacy"


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s :: %(message)s")
    root = args.root.resolve()
    legacy = root / args.legacy_name

    if not legacy.exists():
        logger.info("No legacy directory at %s — nothing to migrate.", legacy)
        return 0

    run_id = _resolve_run_id(legacy / "metadata.json")
    runs_root = root / "runs"
    target = runs_root / run_id

    if target.exists():
        logger.info("Target %s already exists; skipping move.", target)
    else:
        if args.dry_run:
            logger.info("[dry-run] Would move %s -> %s", legacy, target)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(legacy, target)
            logger.info("Copied %s -> %s", legacy, target)

    if args.also_experiments:
        legacy_exp = root / "experiments"
        if legacy_exp.exists():
            target_exp = target / "experiments"
            if target_exp.exists():
                logger.info("Target %s already exists; skipping experiments move.", target_exp)
            elif args.dry_run:
                logger.info("[dry-run] Would move %s -> %s", legacy_exp, target_exp)
            else:
                shutil.copytree(legacy_exp, target_exp)
                logger.info("Copied %s -> %s", legacy_exp, target_exp)

    if args.dry_run:
        logger.info("[dry-run] Would register run %s and update current symlink", run_id)
        return 0

    metrics: dict[str, float] = {}
    md_path = target / "metadata.json"
    if md_path.exists():
        try:
            md = json.loads(md_path.read_text(encoding="utf-8"))
            for key in ("accuracy", "macro_f1"):
                if isinstance(md.get(key), (int, float)):
                    metrics[key] = float(md[key])
        except json.JSONDecodeError:
            pass

    n_sessions = 0
    if md_path.exists():
        try:
            md = json.loads(md_path.read_text(encoding="utf-8"))
            n_sessions = int(md.get("session_count", 0) or 0)
        except json.JSONDecodeError:
            pass

    register_run(
        run_id=run_id,
        kind="train",
        out_dir=target,
        metrics=metrics,
        n_sessions=n_sessions,
        git_sha="",
        notes="migrated from har_classifier/",
        runs_root=runs_root,
    )

    link = update_current_symlink(runs_root=runs_root, run_id=run_id)
    logger.info("current -> %s", link)
    logger.info(
        "Migration complete. Legacy %s left in place; remove it manually once "
        "you've verified the new tree is good.", legacy,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
