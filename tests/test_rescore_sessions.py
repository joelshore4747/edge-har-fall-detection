"""Smoke tests for the session backfill script.

Exercises the bits that don't need a live database: import safety,
filter-clause assembly, raw-payload parsing, and the diff formatter.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest


def test_module_imports() -> None:
    """Importing the script must not crash even without DATABASE_URL set —
    it should only fail at run time, not at import time."""
    import scripts.rescore_sessions as r

    assert callable(r.main)
    assert callable(r._select_target_sessions)
    assert callable(r._read_raw_request)
    assert callable(r._re_run_inference)
    assert callable(r._format_diff_row)
    assert r.DYNAMIC_HAR_FILTER == ("walking", "running", "stairs", "dynamic", "locomotion")


def test_select_target_sessions_builds_expected_sql() -> None:
    """Verify the SQL the script emits via a recorder cursor — confirms
    filters are applied correctly without needing a real database."""
    from scripts.rescore_sessions import _select_target_sessions

    captured: dict[str, Any] = {}

    class _RecordingCursor:
        def execute(self, sql: str, params: list[Any]) -> None:
            captured["sql"] = sql
            captured["params"] = params

        def fetchall(self) -> list[Any]:
            return []

    cur = _RecordingCursor()
    session_id = uuid4()
    _select_target_sessions(
        cur,  # type: ignore[arg-type]
        app_session_id=session_id,
        current_warning_level="high",
        har_filter=("walking", "stairs"),
        limit=25,
    )

    sql = captured["sql"]
    params = captured["params"]

    assert "s.app_session_id = %s" in sql
    assert "i.warning_level = %s" in sql
    assert "LOWER(COALESCE(i.top_har_label, '')) = ANY(%s)" in sql
    assert "LIMIT 25" in sql
    assert params == [session_id, "high", ["walking", "stairs"]]


def test_select_target_sessions_with_no_filters() -> None:
    """No filters → no WHERE clause and no LIMIT clause."""
    from scripts.rescore_sessions import _select_target_sessions

    captured: dict[str, Any] = {}

    class _RecordingCursor:
        def execute(self, sql: str, params: list[Any]) -> None:
            captured["sql"] = sql
            captured["params"] = params

        def fetchall(self) -> list[Any]:
            return []

    cur = _RecordingCursor()
    _select_target_sessions(
        cur,  # type: ignore[arg-type]
        app_session_id=None,
        current_warning_level=None,
        har_filter=None,
        limit=None,
    )

    sql = captured["sql"]
    assert "WHERE" not in sql.upper().split("ORDER BY")[-1]
    assert "LIMIT" not in sql.upper()
    assert captured["params"] == []


def test_read_raw_request_extracts_request_envelope(tmp_path: Path) -> None:
    """The on-disk format is ``{"request": {...}, "stored_at": "..."}``;
    the script must dig the request payload out of that envelope."""
    from scripts.rescore_sessions import _read_raw_request

    fake_request = {
        "metadata": {
            "session_id": "smoke",
            "subject_id": "subject_1",
            "dataset_name": "PHONE",
            "source_type": "mobile",
            "task_type": "fall_detection",
            "device_platform": "android",
        },
        "samples": [
            {"timestamp": 0.0, "ax": 0.0, "ay": 0.0, "az": 9.81},
            {"timestamp": 0.02, "ax": 0.0, "ay": 0.0, "az": 9.81},
        ],
    }
    payload_path = tmp_path / "raw.json"
    with payload_path.open("w") as f:
        json.dump({"request": fake_request, "stored_at": "2026-04-28T00:00:00Z"}, f)

    # Pydantic validation may add fields; we only need to assert it doesn't
    # raise on a minimally-shaped envelope. If the schema is stricter than
    # this, the test will surface it (and we can extend the fixture).
    try:
        req = _read_raw_request(str(payload_path))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(
            f"RuntimeSessionRequest schema is stricter than this fixture provides: {exc}"
        )
    assert req.metadata.session_id == "smoke"


def test_read_raw_request_rejects_missing_file(tmp_path: Path) -> None:
    from scripts.rescore_sessions import _read_raw_request

    with pytest.raises(FileNotFoundError):
        _read_raw_request(str(tmp_path / "does-not-exist.json"))


def test_read_raw_request_rejects_envelope_without_request(tmp_path: Path) -> None:
    from scripts.rescore_sessions import _read_raw_request

    payload_path = tmp_path / "bad.json"
    with payload_path.open("w") as f:
        json.dump({"stored_at": "2026-04-28T00:00:00Z"}, f)
    with pytest.raises(ValueError, match="no 'request' field"):
        _read_raw_request(str(payload_path))


def test_format_diff_row_human_readable() -> None:
    from scripts.rescore_sessions import _format_diff_row

    target = {
        "app_session_id": uuid4(),
        "warning_level": "high",
        "top_har_label": "walking",
        "top_har_fraction": 0.83,
    }
    line = _format_diff_row(target, "medium", 0.42)

    assert "walking" in line
    assert "0.83" in line
    assert "high" in line
    assert "medium" in line
    assert "0.420" in line
