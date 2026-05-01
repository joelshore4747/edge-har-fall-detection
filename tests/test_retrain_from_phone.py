"""Smoke tests for ``scripts/retrain_from_phone.py``.

These tests exercise the orchestrator with HTTP + subprocess stubbed out so the
run can finish in milliseconds. They verify that the cache dir, the converted
phone-folder, and the annotation CSV are populated, and that per-label coverage
is threaded into the final report.
"""

from __future__ import annotations

import importlib.util
import io
import json
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "retrain_from_phone.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("retrain_from_phone", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def retrain_module():
    module = _load_module()
    yield module
    sys.modules.pop(module.__name__, None)


def _fake_session_payload(session_id: str, subject_id: str, label: str, *, n_samples: int = 300) -> dict[str, Any]:
    """Match the wrapper shape used by the storage layer: {stored_at, request}."""
    samples = []
    for i in range(n_samples):
        t = i * 0.02  # 50 Hz
        samples.append(
            {
                "timestamp": t,
                "ax": 0.01 * i,
                "ay": -0.01 * i,
                "az": 9.81,
                "gx": 0.001 * i,
                "gy": 0.0,
                "gz": 0.0,
            }
        )
    return {
        "stored_at": "2026-04-24T00:00:00Z",
        "request": {
            "metadata": {
                "session_id": session_id,
                "subject_id": subject_id,
                "activity_label": label,
                "placement": "pocket",
                "sampling_rate_hz": 50.0,
                "device_platform": "ios",
            },
            "samples": samples,
        },
    }


def test_retrain_orchestrator_pull_convert_annotate_report(
    tmp_path, monkeypatch, retrain_module, capsys
):
    """Happy path: pull → convert → annotate → report. Skip replay/build/train."""

    pull_root = tmp_path / "pulled"
    work_root = tmp_path / "work"
    report_root = tmp_path / "report"

    # Build two fake sessions with different labels.
    subject = "phoneuser1"
    sessions = [
        {
            "app_session_id": "sess-aaa",
            "subject_id": subject,
            "activity_label": "walking",
            "raw_payload_sha256": None,
        },
        {
            "app_session_id": "sess-bbb",
            "subject_id": subject,
            "activity_label": "stairs",
            "raw_payload_sha256": None,
        },
    ]
    payloads = {
        "sess-aaa": _fake_session_payload("session-aaa", subject, "walking"),
        "sess-bbb": _fake_session_payload("session-bbb", subject, "stairs"),
    }

    def fake_get_json(self, path, *, query=None):
        # First page returns the list, subsequent pages return empty.
        if path == "/v1/sessions":
            offset = int((query or {}).get("offset") or 0)
            if offset == 0:
                return {"sessions": sessions}
            return {"sessions": []}
        raise AssertionError(f"unexpected GET {path}")

    def fake_download(self, path, dest):
        # path = "/v1/sessions/{id}/raw"; pull out the ID.
        parts = path.strip("/").split("/")
        assert parts[0] == "v1" and parts[1] == "sessions" and parts[3] == "raw"
        sid = parts[2]
        data = json.dumps(payloads[sid]).encode("utf-8")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        return len(data)

    monkeypatch.setattr(retrain_module.HttpClient, "get_json", fake_get_json)
    monkeypatch.setattr(retrain_module.HttpClient, "download", fake_download)

    # No auth → bypass the env var requirement.
    monkeypatch.setattr(retrain_module, "_load_auth_token", lambda name: None)

    rc = retrain_module.main(
        [
            "--server-url",
            "http://fake-host:8000",
            "--subject-id",
            subject,
            "--pull-root",
            str(pull_root),
            "--work-root",
            str(work_root),
            "--report-root",
            str(report_root),
            "--stages",
            "pull,convert,annotate,report",
            "--auth-env",
            "UNUSED",
        ]
    )
    assert rc == 0

    # Pulled JSONs landed in the cache.
    pulled_files = sorted((pull_root / subject).glob("*.json"))
    assert len(pulled_files) == 2

    # Converted phone-folders exist.
    merged = work_root / "merged"
    assert (merged / "Accelerometer.csv").exists()
    assert (merged / "Gyroscope.csv").exists()
    # Per-session folder exists for each session.
    assert (work_root / "sessions" / "session-aaa" / "Accelerometer.csv").exists()
    assert (work_root / "sessions" / "session-bbb" / "Accelerometer.csv").exists()

    # Annotation CSV has a row per labelled session.
    annotations = (work_root / "annotations.csv").read_text().strip().splitlines()
    header = annotations[0].split(",")
    assert header[:4] == ["session_id", "start_ts", "end_ts", "final_label"]
    label_rows = [line for line in annotations[1:] if line]
    assert len(label_rows) == 2
    assert any("walking" in row for row in label_rows)
    assert any("stairs" in row for row in label_rows)

    # Report captures coverage + scarcity warnings.
    report_dirs = list(report_root.iterdir())
    assert len(report_dirs) == 1
    report = json.loads((report_dirs[0] / "report.json").read_text())
    assert report["sessions_pulled"] == 2
    assert set(report["session_label_coverage"].keys()) == {"walking", "stairs"}
    # Scarcity is expected at this tiny scale.
    assert report["scarcity_warnings"]
    # No labels dropped — every session has activity_label.
    assert not any("no activity_label" in w for w in report["scarcity_warnings"])


def test_retrain_orchestrator_resumes_using_sha(tmp_path, monkeypatch, retrain_module):
    """Second call should skip downloading when cached file hashes match."""
    import hashlib

    pull_root = tmp_path / "pulled"
    work_root = tmp_path / "work"
    report_root = tmp_path / "report"

    subject = "phoneuser1"
    payload = _fake_session_payload("session-aaa", subject, "walking")
    body = json.dumps(payload).encode("utf-8")
    sha = hashlib.sha256(body).hexdigest()

    sessions = [
        {
            "app_session_id": "sess-aaa",
            "subject_id": subject,
            "activity_label": "walking",
            "raw_payload_sha256": sha,
        },
    ]

    download_calls: list[str] = []

    def fake_get_json(self, path, *, query=None):
        if path == "/v1/sessions":
            offset = int((query or {}).get("offset") or 0)
            return {"sessions": sessions} if offset == 0 else {"sessions": []}
        raise AssertionError(path)

    def fake_download(self, path, dest):
        download_calls.append(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(body)
        return len(body)

    monkeypatch.setattr(retrain_module.HttpClient, "get_json", fake_get_json)
    monkeypatch.setattr(retrain_module.HttpClient, "download", fake_download)
    monkeypatch.setattr(retrain_module, "_load_auth_token", lambda name: None)

    common_args = [
        "--server-url",
        "http://fake-host:8000",
        "--subject-id",
        subject,
        "--pull-root",
        str(pull_root),
        "--work-root",
        str(work_root),
        "--report-root",
        str(report_root),
        "--stages",
        "pull",
        "--auth-env",
        "UNUSED",
    ]

    assert retrain_module.main(common_args) == 0
    assert download_calls == ["/v1/sessions/sess-aaa/raw"]

    # Second run — cached file matches SHA, so download must not fire again.
    download_calls.clear()
    assert retrain_module.main(common_args) == 0
    assert download_calls == []
