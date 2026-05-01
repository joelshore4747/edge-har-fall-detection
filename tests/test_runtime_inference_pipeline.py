from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import apps.api.main as api_main
from apps.api.main import app
from apps.api.schemas import RuntimeSessionRequest
from fusion.fall_profiles import get_fall_event_thresholds
from fastapi.testclient import TestClient
from services.runtime_inference import (
    RuntimeArtifacts,
    RuntimeInferenceConfig,
    run_runtime_inference_from_dataframe,
)

DEMO_PAYLOAD_PATH = (
        PROJECT_ROOT / "apps" / "mobile" / "app_mobile" / "assets" / "demo_session_phone1.json"
)
HAR_ARTIFACT_PATH = PROJECT_ROOT / "artifacts" / "har" / "har_rf_ucihar.joblib"
FALL_ARTIFACT_PATH = (
        PROJECT_ROOT / "artifacts" / "fall" / "fall_meta_phone_negatives_v1" / "model.joblib"
)


def _require_demo_and_artifacts() -> None:
    missing: list[str] = []
    for path in [DEMO_PAYLOAD_PATH, HAR_ARTIFACT_PATH, FALL_ARTIFACT_PATH]:
        if not path.exists():
            missing.append(str(path))
    if missing:
        pytest.skip(f"Required test assets are missing: {missing}")


def _load_demo_payload() -> dict:
    _require_demo_and_artifacts()
    return json.loads(DEMO_PAYLOAD_PATH.read_text(encoding="utf-8"))


def _payload_to_dataframe(payload: dict) -> tuple[RuntimeSessionRequest, pd.DataFrame]:
    req = RuntimeSessionRequest.model_validate(payload)

    rows = []
    for s in req.samples:
        rows.append(
            {
                "timestamp": s.timestamp,
                "ax": s.ax,
                "ay": s.ay,
                "az": s.az,
                "gx": s.gx,
                "gy": s.gy,
                "gz": s.gz,
                "dataset_name": req.metadata.dataset_name,
                "subject_id": req.metadata.subject_id,
                "session_id": req.metadata.session_id,
                "task_type": req.metadata.task_type.value,
                "placement": req.metadata.placement.value,
                "sampling_rate_hz": req.metadata.sampling_rate_hz,
                "source_file": req.metadata.source_type.value,
            }
        )

    df = pd.DataFrame(rows)
    assert not df.empty, "Demo payload produced an empty dataframe"
    df = df.sort_values("timestamp", kind="stable").reset_index(drop=True)

    for col in ("gx", "gy", "gz"):
        if col not in df.columns:
            df[col] = pd.NA

    return req, df


def _default_artifacts() -> RuntimeArtifacts:
    return RuntimeArtifacts(
        har_artifact_path=HAR_ARTIFACT_PATH,
        fall_artifact_path=FALL_ARTIFACT_PATH,
    )


def _configure_runtime_artifact_env(monkeypatch: pytest.MonkeyPatch) -> None:
    _require_demo_and_artifacts()
    monkeypatch.setenv("HAR_ARTIFACT_PATH", str(HAR_ARTIFACT_PATH))
    monkeypatch.setenv("FALL_ARTIFACT_PATH", str(FALL_ARTIFACT_PATH))
    monkeypatch.setattr(api_main, "HAR_ARTIFACT_PATH", HAR_ARTIFACT_PATH)
    monkeypatch.setattr(api_main, "FALL_ARTIFACT_PATH", FALL_ARTIFACT_PATH)


def _make_sparse_runtime_payload(
    *,
    sample_count: int,
    duration_seconds: float,
) -> dict:
    if sample_count < 2:
        raise ValueError("sample_count must be at least 2")

    step = duration_seconds / float(sample_count - 1)
    samples = []
    for index in range(sample_count):
        timestamp = index * step
        samples.append(
            {
                "timestamp": timestamp,
                "ax": 0.1 * ((index % 5) - 2),
                "ay": 0.2 * ((index % 7) - 3),
                "az": 9.8 + (0.05 * ((index % 3) - 1)),
                "gx": None,
                "gy": None,
                "gz": None,
            }
        )

    return {
        "metadata": {
            "session_id": f"session_sparse_{sample_count}_{int(duration_seconds * 1000)}",
            "subject_id": "short_capture_user",
            "placement": "pocket",
            "task_type": "runtime",
            "dataset_name": "APP_RUNTIME",
            "source_type": "mobile_app",
            "device_platform": "ios",
            "recording_mode": "live_capture",
            "runtime_mode": "mobile_live",
        },
        "samples": samples,
        "include_har_windows": False,
        "include_fall_windows": False,
        "include_vulnerability_windows": False,
        "include_combined_timeline": True,
        "include_grouped_fall_events": True,
        "include_point_timeline": False,
        "include_timeline_events": True,
        "include_transition_events": True,
    }


def test_app_runtime_threshold_family_uses_phone_scale_cutoffs() -> None:
    runtime_thresholds = get_fall_event_thresholds("APP_RUNTIME")
    runtime_test_thresholds = get_fall_event_thresholds("APP_RUNTIME_TEST")
    generic_thresholds = get_fall_event_thresholds("UNRECOGNIZED_DATASET")

    assert runtime_thresholds.impact_threshold == pytest.approx(11.5)
    assert runtime_thresholds.strong_impact_threshold == pytest.approx(16.5)
    assert runtime_test_thresholds.impact_threshold == pytest.approx(
        runtime_thresholds.impact_threshold
    )
    assert runtime_test_thresholds.strong_impact_threshold == pytest.approx(
        runtime_thresholds.strong_impact_threshold
    )
    assert generic_thresholds.impact_threshold == pytest.approx(490.0)


@pytest.mark.integration
def test_runtime_inference_service_runs_on_demo_payload() -> None:
    payload = _load_demo_payload()
    req, df = _payload_to_dataframe(payload)

    result = run_runtime_inference_from_dataframe(
        df,
        artifacts=_default_artifacts(),
        config=RuntimeInferenceConfig(),
    )

    assert result.source_summary["rows_loaded"] == len(df)
    assert not result.har_windows.empty, "HAR windows should not be empty"
    assert not result.fall_windows.empty, "Fall windows should not be empty"
    assert not result.vulnerability_windows.empty, "Vulnerability windows should not be empty"
    assert not result.placement_windows.empty, "Placement windows should not be empty"

    assert not result.point_timeline.empty, "Point timeline should not be empty"
    assert not result.timeline_events.empty, "Timeline events should not be empty"
    assert result.session_summaries is not None
    assert not result.session_summaries.empty, "Session summaries should not be empty"

    # The display timeline should be more compressed than the point timeline.
    assert len(result.timeline_events) < len(result.point_timeline)

    # A narrative summary should exist and reference at least one session.
    assert isinstance(result.narrative_summary, dict)
    assert result.narrative_summary.get("session_count", 0) >= 1
    assert len(result.narrative_summary.get("sessions", [])) >= 1
    assert result.vulnerability_summary["window_count"] == len(result.vulnerability_windows)
    assert "vulnerability_level_counts" in result.vulnerability_summary
    assert "monitoring_state_counts" in result.vulnerability_summary
    assert result.vulnerability_summary["top_vulnerability_score"] is not None

    # Sanity checks on major columns expected by the UI.
    required_event_columns = {
        "event_id",
        "start_ts",
        "end_ts",
        "activity_label",
        "placement_label",
        "event_kind",
        "description",
    }
    assert required_event_columns.issubset(set(result.timeline_events.columns))

    # The session_id should propagate through the pipeline.
    assert str(req.metadata.session_id) in set(result.timeline_events["session_id"].astype(str))


@pytest.mark.integration
def test_runtime_api_infer_session_accepts_sparse_three_second_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_runtime_artifact_env(monkeypatch)
    monkeypatch.setenv("AUTH_REQUIRED", "false")
    payload = _make_sparse_runtime_payload(sample_count=10, duration_seconds=3.0)

    with TestClient(app) as client:
        response = client.post("/v1/infer/session", json=payload)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["session_id"] == payload["metadata"]["session_id"]
    assert "alert_summary" in body


@pytest.mark.integration
def test_runtime_api_infer_session_rejects_too_short_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_runtime_artifact_env(monkeypatch)
    monkeypatch.setenv("AUTH_REQUIRED", "false")
    payload = _make_sparse_runtime_payload(sample_count=78, duration_seconds=2.5)

    with TestClient(app) as client:
        response = client.post("/v1/infer/session", json=payload)

    assert response.status_code == 400, response.text
    body = response.json()
    assert "3.0 seconds" in body["message"]


@pytest.mark.integration
def test_runtime_inference_service_uses_balanced_app_runtime_thresholds() -> None:
    payload = _load_demo_payload()
    _, df = _payload_to_dataframe(payload)

    result = run_runtime_inference_from_dataframe(
        df,
        artifacts=_default_artifacts(),
        config=RuntimeInferenceConfig(),
    )

    early_windows = (
        result.vulnerability_windows.sort_values("window_id", kind="stable")
        .reset_index(drop=True)
        .head(8)
    )

    assert len(early_windows) == 8
    states = set(early_windows["fall_event_state"])
    assert states <= {"no_event", "impact_only", "possible_fall", "probable_fall"}
    assert states & {"possible_fall", "probable_fall"}


@pytest.mark.integration
def test_runtime_inference_service_produces_compressed_display_timeline() -> None:
    payload = _load_demo_payload()
    _, df = _payload_to_dataframe(payload)

    result = run_runtime_inference_from_dataframe(
        df,
        artifacts=_default_artifacts(),
        config=RuntimeInferenceConfig(),
    )

    point_count = len(result.point_timeline)
    event_count = len(result.timeline_events)
    transition_count = len(result.transition_events)

    assert point_count > 0
    assert event_count > 0
    assert event_count < point_count

    # Transitions should not exceed event_count - 1.
    assert transition_count <= max(0, event_count - 1)

    # Event durations should be non-negative.
    durations = pd.to_numeric(result.timeline_events["duration_seconds"], errors="coerce")
    assert durations.notna().all()
    assert (durations >= 0).all()

    # Timeline should be ordered.
    starts = pd.to_numeric(result.timeline_events["start_ts"], errors="coerce")
    assert starts.is_monotonic_increasing

    # Fall-like events should not exceed total timeline events.
    if "likely_fall" in result.timeline_events.columns:
        fall_like_count = int(result.timeline_events["likely_fall"].fillna(False).astype(bool).sum())
        assert fall_like_count <= event_count


@pytest.mark.integration
def test_runtime_api_infer_session_endpoint_returns_timeline_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_runtime_artifact_env(monkeypatch)
    monkeypatch.setenv("AUTH_REQUIRED", "false")
    payload = _load_demo_payload()
    payload["include_vulnerability_windows"] = True

    with TestClient(app) as client:
        response = client.post("/v1/infer/session", json=payload)

    assert response.status_code == 200, response.text

    body = response.json()

    assert "session_id" in body
    assert "alert_summary" in body
    assert "placement_summary" in body
    assert "har_summary" in body
    assert "fall_summary" in body
    assert "vulnerability_summary" in body

    assert "timeline_events" in body
    assert "transition_events" in body
    assert "session_narrative_summary" in body
    assert "vulnerability_windows" in body

    assert isinstance(body["timeline_events"], list)
    assert isinstance(body["transition_events"], list)
    assert isinstance(body["vulnerability_windows"], list)
    assert body["session_narrative_summary"] is None or isinstance(
        body["session_narrative_summary"], dict
    )

    assert len(body["timeline_events"]) > 0
    assert len(body["vulnerability_windows"]) > 0
    assert "top_vulnerability_score" in body["alert_summary"]
    assert "latest_vulnerability_level" in body["alert_summary"]
    assert "latest_monitoring_state" in body["alert_summary"]
    assert "top_vulnerability_score" in body["vulnerability_summary"]
    assert "latest_vulnerability_level" in body["vulnerability_summary"]

    first_event = body["timeline_events"][0]
    for key in (
            "event_id",
            "start_ts",
            "end_ts",
            "activity_label",
            "placement_label",
            "event_kind",
            "description",
    ):
        assert key in first_event, f"Missing key in first timeline event: {key}"

    first_vulnerability_window = body["vulnerability_windows"][0]
    for key in (
            "window_id",
            "fall_probability",
            "fall_event_state",
            "vulnerability_level",
            "vulnerability_score",
            "monitoring_state",
    ):
        assert key in first_vulnerability_window, f"Missing key in first vulnerability window: {key}"


@pytest.mark.integration
def test_runtime_api_health_endpoint_reports_ok(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AUTH_REQUIRED", "false")
    _require_demo_and_artifacts()

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200, response.text
    body = response.json()

    assert body["status"] == "ok"
    assert "service_name" in body
    assert "version" in body
    assert response.headers["X-Request-ID"]
