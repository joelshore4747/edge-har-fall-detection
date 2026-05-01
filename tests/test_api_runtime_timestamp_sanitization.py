from __future__ import annotations

import os
from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault(
    "HAR_ARTIFACT_PATH",
    str(PROJECT_ROOT / "artifacts" / "har" / "har_rf_ucihar.joblib"),
)
os.environ.setdefault(
    "FALL_ARTIFACT_PATH",
    str(
        PROJECT_ROOT / "artifacts" / "fall" / "fall_meta_phone_negatives_v1" / "model.joblib"
    ),
)
os.environ["LOG_FILE_PATH"] = ""

from apps.api import main
from apps.api.schemas import RuntimeSessionRequest


def _request_with_timestamps(timestamps: list[float]) -> RuntimeSessionRequest:
    return RuntimeSessionRequest.model_validate(
        {
            "metadata": {
                "session_id": "test_session",
                "subject_id": "test_subject",
                "placement": "pocket",
                "device_platform": "ios",
            },
            "samples": [
                {
                    "timestamp": ts,
                    "ax": 0.0,
                    "ay": 0.0,
                    "az": 9.81,
                    "gx": 0.0,
                    "gy": 0.0,
                    "gz": 0.0,
                }
                for ts in timestamps
            ],
        }
    )


def test_request_dataframe_drops_and_rebases_phantom_leading_sample() -> None:
    req = _request_with_timestamps([0.0, 83.5, 83.52, 83.54])

    df = main._request_to_dataframe(req)

    assert len(df) == 3
    assert df["timestamp"].tolist() == pytest.approx([0.0, 0.02, 0.04])
    assert main._runtime_session_duration_seconds(df) == pytest.approx(0.04)


def test_request_dataframe_rebases_clean_nonzero_runtime_start() -> None:
    req = _request_with_timestamps([10.0, 10.02, 10.04])

    df = main._request_to_dataframe(req)

    assert len(df) == 3
    assert df["timestamp"].tolist() == pytest.approx([0.0, 0.02, 0.04])
