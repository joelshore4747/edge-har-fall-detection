from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from apps.api.schemas import RuntimeSessionRequest
from services.runtime_inference import (
    RuntimeArtifacts,
    RuntimeInferenceConfig,
    run_runtime_inference_from_dataframe,
)


def load_demo_dataframe(project_root: Path) -> tuple[RuntimeSessionRequest, pd.DataFrame]:
    demo_path = project_root / "apps/mobile/app_mobile/assets/demo_session_phone1.json"
    payload = json.loads(demo_path.read_text(encoding="utf-8"))
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

    df = pd.DataFrame(rows).sort_values("timestamp", kind="stable").reset_index(drop=True)
    return req, df


def main() -> None:
    project_root = Path("/home/joels/PycharmProjects/vulnerability_dissertation_clean")

    req, df = load_demo_dataframe(project_root)

    artifacts = RuntimeArtifacts(
        har_artifact_path=project_root / "artifacts/har/har_rf_ucihar.joblib",
        fall_artifact_path=project_root / "artifacts/fall/fall_meta_phone_negatives_v1/model.joblib",
    )

    result = run_runtime_inference_from_dataframe(
        df,
        artifacts=artifacts,
        config=RuntimeInferenceConfig(),
    )

    print("\n=== SOURCE SUMMARY ===")
    print(result.source_summary)

    print("\n=== PLACEMENT SUMMARY ===")
    print(result.placement_summary)

    print("\n=== HAR SUMMARY ===")
    print(result.har_summary)

    print("\n=== FALL SUMMARY ===")
    print(result.fall_summary)

    print("\n=== COUNTS ===")
    print("point_timeline:", len(result.point_timeline))
    print("raw_timeline_events:", len(result.raw_timeline_events))
    print("timeline_events:", len(result.timeline_events))
    print("transition_events:", len(result.transition_events))

    print("\n=== HAR WINDOW LABEL COUNTS ===")
    if "predicted_label" in result.har_windows.columns:
        print(result.har_windows["predicted_label"].astype(str).value_counts(dropna=False))
    else:
        print("No HAR predicted_label column")

    print("\n=== PLACEMENT WINDOW LABEL COUNTS ===")
    if "placement_state_smoothed" in result.placement_windows.columns:
        print(result.placement_windows["placement_state_smoothed"].astype(str).value_counts(dropna=False))
    elif "placement_state" in result.placement_windows.columns:
        print(result.placement_windows["placement_state"].astype(str).value_counts(dropna=False))
    else:
        print("No placement label column")

    print("\n=== FALL WINDOW PROBABILITY SUMMARY ===")
    if "predicted_probability" in result.fall_windows.columns:
        probs = pd.to_numeric(result.fall_windows["predicted_probability"], errors="coerce")
        print(probs.describe())
        print(">= 0.50:", int((probs >= 0.50).sum()))
        print(">= 0.75:", int((probs >= 0.75).sum()))
        print(">= 0.90:", int((probs >= 0.90).sum()))
    else:
        print("No predicted_probability column")

    print("\n=== POINT TIMELINE HEAD ===")
    print(result.point_timeline.head(10).to_string(index=False))

    print("\n=== RAW TIMELINE EVENTS ===")
    if result.raw_timeline_events.empty:
        print("No raw_timeline_events")
    else:
        print(
            result.raw_timeline_events[
                [
                    "start_ts",
                    "end_ts",
                    "activity_label",
                    "placement_label",
                    "event_kind",
                    "likely_fall",
                    "description",
                ]
            ].to_string(index=False)
        )

    print("\n=== COMPRESSED TIMELINE EVENTS ===")
    if result.timeline_events.empty:
        print("No timeline_events")
    else:
        print(
            result.timeline_events[
                [
                    "start_ts",
                    "end_ts",
                    "activity_label",
                    "placement_label",
                    "event_kind",
                    "likely_fall",
                    "description",
                ]
            ].to_string(index=False)
        )

    print("\n=== TRANSITIONS ===")
    if result.transition_events.empty:
        print("No transitions")
    else:
        print(
            result.transition_events[
                [
                    "transition_ts",
                    "transition_kind",
                    "description",
                ]
            ].to_string(index=False)
        )

    print("\n=== SESSION SUMMARIES ===")
    if result.session_summaries.empty:
        print("No session summaries")
    else:
        print(result.session_summaries.to_string(index=False))

    print("\n=== NARRATIVE SUMMARY ===")
    print(json.dumps(result.narrative_summary, indent=2, default=str))


if __name__ == "__main__":
    main()
