"""Content-addressable run registry for trainer + experiments outputs.

Every training run and every experiments run gets a unique ``run_id`` derived
from the content of the run (args, data manifest, code git SHA), and is
recorded in ``artifacts/unifallmonitor/runs/index.csv`` so a flat ``ls`` shows
the full history with their headline metrics. The directory layout is

    artifacts/unifallmonitor/
      runs/
        2026-04-27_172401_a3f9c2e/
          metadata.json
          ...
        index.csv
      current -> runs/2026-04-27_172401_a3f9c2e

The ``current`` symlink is the read pointer for downstream consumers
(experiments, deployable export). It's only updated when the user passes
``--mark-current`` so a bad run can't silently overwrite the canonical
pointer.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import subprocess
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

logger = logging.getLogger(__name__)


# Columns written to the run index. Order matters — we append rows in this
# order; readers should use a DictReader.
INDEX_COLUMNS = (
    "run_id",
    "timestamp_utc",
    "git_sha",
    "kind",
    "n_sessions",
    "accuracy",
    "macro_f1",
    "fall_f1_at_0p5",
    "out_dir",
    "notes",
)


@dataclass(frozen=True)
class RunIdentity:
    run_id: str
    timestamp_utc: str
    git_sha: str
    short_sha: str


def compute_run_id(
    *,
    args: Mapping[str, Any],
    data_manifest: Iterable[str],
    git_sha: str | None = None,
) -> RunIdentity:
    """Hash args + sorted data manifest + git SHA into a stable run id.

    The id is ``YYYY-MM-DD_HHMMSS_<first 7 chars of git SHA>``. It's not
    purely content-addressable (the timestamp is included so successive runs
    with identical inputs sort chronologically), but the metadata.json
    written into the run directory captures the full hash for verification.
    """
    sha = git_sha or _resolve_git_sha()
    short = sha[:7] if sha else "nogit"
    now = datetime.now(timezone.utc)
    timestamp = now.strftime("%Y-%m-%d_%H%M%S")
    run_id = f"{timestamp}_{short}"
    return RunIdentity(
        run_id=run_id,
        timestamp_utc=now.isoformat(),
        git_sha=sha or "",
        short_sha=short,
    )


def content_hash(
    *,
    args: Mapping[str, Any],
    data_manifest: Iterable[str],
    git_sha: str = "",
) -> str:
    """SHA-256 over the canonical (args, data manifest, git SHA) tuple.

    Two runs that produced the same model from the same inputs yield the
    same content hash. The trainer writes this into ``metadata.json`` so a
    later reader can verify reproducibility without re-running.
    """
    payload = {
        "args": _stable_args(args),
        "data_manifest": sorted(str(s) for s in data_manifest),
        "git_sha": git_sha,
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def register_run(
    *,
    run_id: str,
    kind: str,
    out_dir: Path,
    metrics: Mapping[str, Any],
    n_sessions: int,
    git_sha: str,
    notes: str = "",
    runs_root: Path | None = None,
) -> Path:
    """Append one row to the runs index. Returns the index path.

    Idempotent on the (run_id, kind) tuple — re-registering the same run
    overwrites its prior row instead of appending a duplicate.
    """
    out_dir = Path(out_dir)
    runs_root = Path(runs_root) if runs_root else out_dir.parent
    runs_root.mkdir(parents=True, exist_ok=True)
    index_path = runs_root / "index.csv"

    row = {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha,
        "kind": kind,
        "n_sessions": str(int(n_sessions)),
        "accuracy": _fmt_metric(metrics.get("accuracy")),
        "macro_f1": _fmt_metric(metrics.get("macro_f1")),
        "fall_f1_at_0p5": _fmt_metric(metrics.get("fall_f1_at_0p5")),
        "out_dir": str(out_dir.resolve()),
        "notes": notes,
    }

    existing_rows: list[dict[str, str]] = []
    if index_path.exists():
        with index_path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                if r.get("run_id") == run_id and r.get("kind") == kind:
                    continue  # drop the prior row for this (run_id, kind)
                existing_rows.append(r)

    with index_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(INDEX_COLUMNS))
        writer.writeheader()
        for r in existing_rows:
            # Existing rows may have been written with an older column set;
            # fill missing keys with "" so the header stays canonical.
            writer.writerow({c: r.get(c, "") for c in INDEX_COLUMNS})
        writer.writerow(row)

    logger.info("Registered run %s (%s) → %s", run_id, kind, index_path)
    return index_path


def update_current_symlink(*, runs_root: Path, run_id: str) -> Path:
    """Point ``runs_root.parent / "current"`` at the given run directory.

    Writes a relative symlink so the tree stays portable across deploy
    locations. Falls back to writing a `.current` text file when the OS
    rejects symlinks (Windows without dev mode).
    """
    runs_root = Path(runs_root)
    target = runs_root / run_id
    if not target.exists():
        raise FileNotFoundError(f"Run directory {target} does not exist")
    parent = runs_root.parent
    link = parent / "current"
    rel_target = Path(os.path.relpath(target, parent))
    try:
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(rel_target, target_is_directory=True)
    except (OSError, NotImplementedError):
        marker = parent / "current.txt"
        marker.write_text(str(rel_target) + "\n", encoding="utf-8")
        return marker
    return link


def resolve_current_run(*, runs_root: Path) -> Path | None:
    """Return the run directory that ``current`` points at, if any."""
    runs_root = Path(runs_root)
    parent = runs_root.parent
    link = parent / "current"
    if link.is_symlink() or link.exists():
        return link.resolve()
    marker = parent / "current.txt"
    if marker.exists():
        rel = marker.read_text(encoding="utf-8").strip()
        if rel:
            candidate = (parent / rel).resolve()
            if candidate.exists():
                return candidate
    return None


def _resolve_git_sha() -> str:
    """Best-effort ``git rev-parse HEAD`` from the repo root."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def _stable_args(args: Mapping[str, Any]) -> dict[str, Any]:
    """Convert an argparse Namespace / mapping to a JSON-stable dict.

    Drops credentials so they never end up in the hash or on disk.
    """
    drop = {"username", "password", "auth_header"}
    out: dict[str, Any] = {}
    for k, v in dict(args).items():
        if k in drop:
            continue
        if isinstance(v, Path):
            out[k] = str(v)
        elif isinstance(v, (list, tuple)):
            out[k] = [str(x) for x in v]
        else:
            out[k] = v
    return out


def _fmt_metric(value: Any) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return str(value)
