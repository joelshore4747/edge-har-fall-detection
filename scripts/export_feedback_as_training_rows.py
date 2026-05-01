#!/usr/bin/env python3
"""Turn ``/v1/feedback`` records into additional training rows.

For each actionable feedback record (``confirmed_fall``, ``false_alarm``, or a
``corrected_label``), the script finds the corresponding pulled session file
under ``artifacts/runtime_sessions_pulled/<subject>/`` and writes a relabeled
copy to ``artifacts/runtime_sessions_pulled/<subject>/feedback/<id>.json``.

Because the feedback subfolder lives inside the pull root that
``retrain_from_phone.py`` already walks, the next retrain picks up the rows
with zero config change. A record that cannot be matched to a pulled session
is reported and skipped — it is not silently dropped.

Label mapping:
    - ``confirmed_fall``       → activity_label = "fall"
    - ``false_alarm``          → activity_label = "other"     (non_fall)
    - ``corrected_label=<x>``  → activity_label = <x> (normalised to lowercase)
    - ``uncertain``            → ignored
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


DEFAULT_FEEDBACK_JSONL = _REPO_ROOT / "artifacts" / "feedback" / "prediction_feedback.jsonl"
DEFAULT_PULL_ROOT = _REPO_ROOT / "artifacts" / "runtime_sessions_pulled"

ACTIONABLE = {"confirmed_fall", "false_alarm", "corrected_label"}


def _iter_feedback(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _resolve_target_label(record: dict[str, Any]) -> str | None:
    user_feedback = str(record.get("user_feedback") or "").strip().lower()
    if user_feedback == "confirmed_fall":
        return "fall"
    if user_feedback == "false_alarm":
        return "other"
    if user_feedback == "corrected_label":
        label = record.get("corrected_label")
        return str(label).strip().lower() if label else None
    return None


def _find_session_file(pull_root: Path, subject_id: str | None, session_id: str) -> Path | None:
    """Match by client-provided session_id, scanning known subject dirs."""
    candidates: list[Path] = []
    if subject_id:
        candidates.append(pull_root / subject_id)
    elif pull_root.is_dir():
        candidates.extend(p for p in pull_root.iterdir() if p.is_dir())

    for subject_dir in candidates:
        if not subject_dir.is_dir():
            continue
        for path in subject_dir.rglob("*.json"):
            if path.parent.name == "feedback":
                # Don't re-ingest feedback-derived files.
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            request = payload.get("request") if isinstance(payload, dict) else None
            metadata = (request or payload or {}).get("metadata") if isinstance(payload, dict) else None
            if not metadata:
                continue
            if str(metadata.get("session_id") or "") == session_id:
                return path
    return None


def export(
    *,
    feedback_jsonl: Path,
    pull_root: Path,
    dry_run: bool = False,
) -> dict[str, Any]:
    produced: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for record in _iter_feedback(feedback_jsonl):
        label = _resolve_target_label(record)
        session_id = str(record.get("session_id") or "").strip()
        if not label or not session_id:
            skipped.append({"reason": "non-actionable or missing session_id", "record": record})
            continue
        subject_hint = record.get("subject_id")
        source_path = _find_session_file(pull_root, subject_hint, session_id)
        if source_path is None:
            skipped.append(
                {
                    "reason": "no pulled session matched this session_id",
                    "session_id": session_id,
                    "subject_id": subject_hint,
                }
            )
            continue

        try:
            payload = json.loads(source_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            skipped.append({"reason": f"failed to parse source payload: {exc}", "source": str(source_path)})
            continue

        # Overwrite the activity_label inside the RuntimeSessionRequest wrapper.
        request_block = payload.get("request") if isinstance(payload, dict) and "request" in payload else payload
        if not isinstance(request_block, dict):
            skipped.append({"reason": "malformed payload", "source": str(source_path)})
            continue
        metadata = dict(request_block.get("metadata") or {})
        metadata["activity_label"] = label
        metadata["_feedback_source"] = {
            "user_feedback": record.get("user_feedback"),
            "original_label": (request_block.get("metadata") or {}).get("activity_label"),
            "feedback_request_id": str(record.get("request_id") or ""),
            "exported_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        request_block = dict(request_block)
        request_block["metadata"] = metadata
        new_payload = dict(payload) if isinstance(payload, dict) else {}
        if "request" in new_payload:
            new_payload["request"] = request_block
        else:
            new_payload = request_block

        # Destination: <pull_root>/<subject>/feedback/<feedback_id>.json
        subject_dir = source_path.parent
        feedback_dir = subject_dir / "feedback"
        feedback_id = str(record.get("request_id") or record.get("created_at") or datetime.now(timezone.utc).timestamp())
        dest = feedback_dir / f"{feedback_id}.json"

        if not dry_run:
            feedback_dir.mkdir(parents=True, exist_ok=True)
            dest.write_text(json.dumps(new_payload, default=str), encoding="utf-8")
        produced.append(
            {
                "feedback_request_id": feedback_id,
                "session_id": session_id,
                "source": str(source_path),
                "dest": str(dest),
                "label": label,
            }
        )

    return {"produced": produced, "skipped": skipped}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--feedback-jsonl", type=Path, default=DEFAULT_FEEDBACK_JSONL)
    p.add_argument("--pull-root", type=Path, default=DEFAULT_PULL_ROOT)
    p.add_argument("--dry-run", action="store_true", help="Report actions without writing files")
    args = p.parse_args(argv)

    summary = export(
        feedback_jsonl=args.feedback_jsonl,
        pull_root=args.pull_root,
        dry_run=args.dry_run,
    )

    print(f"Produced: {len(summary['produced'])} feedback-derived training rows")
    for item in summary["produced"]:
        print(f"  + {item['label']:<12s} from {item['source']} → {item['dest']}")
    if summary["skipped"]:
        print(f"Skipped: {len(summary['skipped'])}")
        for item in summary["skipped"]:
            print(f"  - {item.get('reason')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
