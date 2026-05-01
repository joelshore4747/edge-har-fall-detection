"""Feedback-to-training-rows export tests.

Tests that a synthetic ``prediction_feedback.jsonl`` plus a pulled session
produces a feedback-derived copy under ``<pull_root>/<subject>/feedback/``
with the activity_label overridden to match the feedback verdict.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "export_feedback_as_training_rows.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("export_feedback_as_training_rows", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    sys.modules[module.__name__] = module
    return module


def _write_pulled(pull_root: Path, subject: str, session_id: str, activity_label: str | None) -> Path:
    dest = pull_root / subject / f"{session_id}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "stored_at": "2026-04-24T00:00:00Z",
        "request": {
            "metadata": {
                "session_id": session_id,
                "subject_id": subject,
                "activity_label": activity_label,
                "placement": "pocket",
                "sampling_rate_hz": 50.0,
            },
            "samples": [
                {"timestamp": 0.0, "ax": 0.0, "ay": 0.0, "az": 9.8, "gx": 0.0, "gy": 0.0, "gz": 0.0},
                {"timestamp": 0.02, "ax": 0.0, "ay": 0.0, "az": 9.8, "gx": 0.0, "gy": 0.0, "gz": 0.0},
            ],
        },
    }
    dest.write_text(json.dumps(payload), encoding="utf-8")
    return dest


def _write_feedback(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def test_export_false_alarm_writes_feedback_row_with_overridden_label(tmp_path):
    module = _load_module()

    pull_root = tmp_path / "pulled"
    subject = "joe"
    _write_pulled(pull_root, subject, "sess-abc", activity_label="fall")

    feedback_path = tmp_path / "feedback.jsonl"
    _write_feedback(
        feedback_path,
        [
            {
                "session_id": "sess-abc",
                "subject_id": subject,
                "user_feedback": "false_alarm",
                "request_id": "fb-111",
            }
        ],
    )

    summary = module.export(feedback_jsonl=feedback_path, pull_root=pull_root, dry_run=False)
    assert len(summary["produced"]) == 1
    assert len(summary["skipped"]) == 0

    feedback_out = pull_root / subject / "feedback" / "fb-111.json"
    assert feedback_out.exists()
    payload = json.loads(feedback_out.read_text())
    assert payload["request"]["metadata"]["activity_label"] == "other"
    # Original label preserved in the audit trail.
    src = payload["request"]["metadata"]["_feedback_source"]
    assert src["original_label"] == "fall"
    assert src["user_feedback"] == "false_alarm"


def test_export_confirmed_fall_and_corrected_label(tmp_path):
    module = _load_module()

    pull_root = tmp_path / "pulled"
    subject = "joe"
    _write_pulled(pull_root, subject, "sess-conf", activity_label="other")
    _write_pulled(pull_root, subject, "sess-corr", activity_label="walking")

    feedback_path = tmp_path / "feedback.jsonl"
    _write_feedback(
        feedback_path,
        [
            {
                "session_id": "sess-conf",
                "subject_id": subject,
                "user_feedback": "confirmed_fall",
                "request_id": "fb-222",
            },
            {
                "session_id": "sess-corr",
                "subject_id": subject,
                "user_feedback": "corrected_label",
                "corrected_label": "stairs",
                "request_id": "fb-333",
            },
        ],
    )

    summary = module.export(feedback_jsonl=feedback_path, pull_root=pull_root, dry_run=False)
    assert len(summary["produced"]) == 2

    conf = json.loads((pull_root / subject / "feedback" / "fb-222.json").read_text())
    assert conf["request"]["metadata"]["activity_label"] == "fall"

    corr = json.loads((pull_root / subject / "feedback" / "fb-333.json").read_text())
    assert corr["request"]["metadata"]["activity_label"] == "stairs"


def test_export_skips_uncertain_and_missing_sessions(tmp_path):
    module = _load_module()
    pull_root = tmp_path / "pulled"
    _write_pulled(pull_root, "joe", "sess-real", activity_label="walking")

    feedback_path = tmp_path / "feedback.jsonl"
    _write_feedback(
        feedback_path,
        [
            {"session_id": "sess-real", "subject_id": "joe", "user_feedback": "uncertain", "request_id": "u1"},
            {"session_id": "sess-unknown", "subject_id": "joe", "user_feedback": "false_alarm", "request_id": "u2"},
        ],
    )

    summary = module.export(feedback_jsonl=feedback_path, pull_root=pull_root, dry_run=False)
    assert summary["produced"] == []
    assert len(summary["skipped"]) == 2


def test_export_ignores_existing_feedback_subdir_on_lookup(tmp_path):
    """If someone re-runs after a prior export, we should not treat feedback-derived
    files as source sessions (infinite loop risk)."""
    module = _load_module()
    pull_root = tmp_path / "pulled"
    subject = "joe"
    _write_pulled(pull_root, subject, "sess-loop", activity_label="fall")

    feedback_path = tmp_path / "feedback.jsonl"
    _write_feedback(
        feedback_path,
        [{"session_id": "sess-loop", "subject_id": subject, "user_feedback": "false_alarm", "request_id": "r1"}],
    )

    # First export creates feedback/r1.json
    module.export(feedback_jsonl=feedback_path, pull_root=pull_root)

    # Second run: the feedback file now exists as <subject>/feedback/r1.json.
    # It must not be treated as a source when looking up sess-loop the second time.
    summary_again = module.export(feedback_jsonl=feedback_path, pull_root=pull_root)
    assert len(summary_again["produced"]) == 1
    # Same source, not the feedback file.
    assert "/feedback/" not in summary_again["produced"][0]["source"]
