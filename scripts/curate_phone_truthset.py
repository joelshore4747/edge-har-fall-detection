#!/usr/bin/env python3
"""Copy hand-picked pulled phone sessions into ``artifacts/phone_truthset/``.

The truthset is a small curated set the user trusts as ground truth. Once
populated, every diagnostic run + promotion gate can evaluate against it to
catch regressions that public-heldout metrics would miss.

Typical workflow after a retrain run:

    # Review the session list your retrain pulled
    ls artifacts/runtime_sessions_pulled/<subject>/
    # Pick the ones you want to trust as truth
    python scripts/curate_phone_truthset.py \\
        --pull-root artifacts/runtime_sessions_pulled \\
        --subject-id <subject> \\
        --sessions sess-abc.json sess-def.json \\
        --label-override sess-abc.json=fall
    # Later runs + promotions pick it up automatically.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


DEFAULT_PULL_ROOT = _REPO_ROOT / "artifacts" / "runtime_sessions_pulled"
DEFAULT_TRUTHSET = _REPO_ROOT / "artifacts" / "phone_truthset"


def _parse_overrides(overrides: list[str] | None) -> dict[str, str]:
    out: dict[str, str] = {}
    for entry in overrides or []:
        if "=" not in entry:
            raise SystemExit(f"--label-override entries must be key=label, got: {entry!r}")
        key, value = entry.split("=", 1)
        out[key.strip()] = value.strip().lower()
    return out


def _load_session(path: Path) -> dict[str, Any]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(doc, dict) and isinstance(doc.get("request"), dict):
        return doc["request"]
    return doc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pull-root", type=Path, default=DEFAULT_PULL_ROOT)
    parser.add_argument("--subject-id", required=True)
    parser.add_argument(
        "--sessions",
        nargs="+",
        required=True,
        help="Filenames under <pull_root>/<subject_id>/ to include in the truthset",
    )
    parser.add_argument(
        "--label-override",
        action="append",
        default=None,
        help="Rewrite activity_label, e.g. sess-abc.json=fall. Repeatable.",
    )
    parser.add_argument("--truthset-dir", type=Path, default=DEFAULT_TRUTHSET)
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Wipe the truthset dir first — use when re-curating from scratch.",
    )
    args = parser.parse_args(argv)

    overrides = _parse_overrides(args.label_override)
    source_dir = args.pull_root / args.subject_id
    if not source_dir.is_dir():
        raise SystemExit(f"Source dir not found: {source_dir}")

    truthset_dir = args.truthset_dir / args.subject_id
    if args.clear and truthset_dir.exists():
        shutil.rmtree(truthset_dir)
    truthset_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, Any]] = []
    for filename in args.sessions:
        src = source_dir / filename
        if not src.is_file():
            raise SystemExit(f"Session file not found: {src}")
        dst = truthset_dir / filename
        shutil.copy2(src, dst)

        payload = _load_session(src)
        meta = payload.get("metadata") or {}
        effective_label = overrides.get(filename, meta.get("activity_label") or "")
        manifest_rows.append(
            {
                "source": str(src),
                "filename": filename,
                "session_id": meta.get("session_id"),
                "subject_id": meta.get("subject_id"),
                "activity_label": str(effective_label).lower() if effective_label else None,
                "placement": meta.get("placement"),
                "sample_count": len(payload.get("samples") or []),
                "overridden": filename in overrides,
            }
        )

    manifest = {
        "curated_utc": datetime.now(timezone.utc).isoformat(),
        "subject_id": args.subject_id,
        "source_pull_root": str(args.pull_root),
        "sessions": manifest_rows,
    }
    manifest_path = truthset_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    print(f"Curated {len(manifest_rows)} session(s) into {truthset_dir}")
    for row in manifest_rows:
        note = " (overridden)" if row["overridden"] else ""
        print(f"  - {row['filename']}: label={row['activity_label']}{note}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())