from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import json
import numpy as np
import pandas as pd

from fusion.fall_event import FallEventInputs, FallEventResult, FallEventState, FallEventThresholds, classify_fall_event
from fusion.fall_profiles import get_fall_event_thresholds
from fusion.state_machine import StateMachineConfig, StateMachineInputs, StateMachineState, step_state_machine
from fusion.vulnerability_score import (
    VulnerabilityInputs,
    VulnerabilityLevel,
    VulnerabilityResult,
    get_vulnerability_profile,
    score_vulnerability,
)
from metrics.classification import compute_classification_metrics
from metrics.fall_metrics import compute_fall_detection_metrics
from models.fall.evaluate_threshold_fall import build_threshold_prediction_table
from models.fall.infer_fall import load_fall_model_artifact, predict_fall_from_artifact_path
from models.har.infer_har import predict_har_from_artifact_path
from models.har.train_har import load_har_model_artifact
from pipeline.fall.threshold_detector import default_fall_threshold_config
from pipeline.features import build_feature_table
from pipeline.ingest import load_mobifall, load_pamap2, load_sisfall, load_uci_har
from pipeline.ingest.runtime_phone_csv import RuntimePhoneCsvConfig, load_runtime_phone_csv
from pipeline.ingest.runtime_phone_folder import RuntimePhoneFolderConfig, load_runtime_phone_folder
from pipeline.preprocess import (
    PreprocessConfig,
    append_derived_channels,
    resample_dataframe,
    window_dataframe,
)
from pipeline.windowing import BranchWindowConfig, synchronize_windows
from services.placement_state import PlacementStateConfig, infer_placement_state_from_dataframe
from services.runtime_timeline import (
    RuntimeTimelineConfig,
    build_runtime_timeline_events,
)


@dataclass(slots=True)
class RuntimeArtifacts:
    har_artifact_path: str | Path
    fall_artifact_path: str | Path


@dataclass(slots=True)
class RuntimeInferenceConfig:
    threshold_mode: str = "shared"

    har_target_rate: float | None = None
    har_window_size: int | None = None
    har_step_size: int | None = None

    # Default None means "read from artifact metadata"
    # (training_config.target_rate_hz / window_size / step_size). Explicit
    # values override the artifact. Matches the HAR-branch semantics above.
    fall_target_rate: float | None = None
    fall_window_size: int | None = None
    fall_step_size: int | None = None

    timeline_tolerance_seconds: float = 1.0

    group_fall_events: bool = True
    event_probability_threshold: float | None = None
    # Optional placement-specific thresholds. When set, grouping picks the
    # threshold for a window's placement; falls back to
    # ``event_probability_threshold`` for any placement not listed. This is
    # populated from artifact metadata (``thresholds_by_placement``) so
    # retrained candidates can ship a tuned operating point per placement.
    event_thresholds_by_placement: dict[str, float] | None = None
    event_merge_gap_seconds: float = 0.25
    event_min_windows: int = 2
    event_max_duration_seconds: float = 4.0
    enable_vulnerability_scoring: bool = True
    event_profile: str = "balanced"
    vulnerability_profile: str = "balanced"
    state_machine_config: StateMachineConfig | None = None

    infer_placement_state: bool = True
    placement_config: PlacementStateConfig | None = None

    build_timeline_events: bool = True
    timeline_config: RuntimeTimelineConfig | None = None


@dataclass(slots=True)
class RuntimeInferenceResult:
    source_summary: dict[str, Any]
    har_summary: dict[str, Any]
    fall_summary: dict[str, Any]
    vulnerability_summary: dict[str, Any]
    report: dict[str, Any]
    har_windows: pd.DataFrame
    fall_windows: pd.DataFrame
    vulnerability_windows: pd.DataFrame
    combined_timeline: pd.DataFrame
    grouped_fall_events: pd.DataFrame
    placement_summary: dict[str, Any]
    placement_windows: pd.DataFrame
    point_timeline: pd.DataFrame
    raw_timeline_events: pd.DataFrame
    timeline_events: pd.DataFrame
    transition_events: pd.DataFrame
    session_summaries: pd.DataFrame
    narrative_summary: dict[str, Any]


def _resolve_path(path_like: str | Path) -> Path:
    return Path(path_like).expanduser().resolve()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except Exception:
        return default
    if not np.isfinite(result):
        return default
    return result


def _safe_optional_float(value: Any) -> float | None:
    try:
        result = float(value)
    except Exception:
        return None
    if not np.isfinite(result):
        return None
    return result


def _safe_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n", "", "nan", "none"}:
        return False
    return default


def _safe_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value)
    if text.lower() == "nan":
        return default
    return text


def _load_source_dataframe_from_path(
        *,
        input_source: str,
        input_path: str | Path,
        csv_config: RuntimePhoneCsvConfig | None = None,
        phone_folder_config: RuntimePhoneFolderConfig | None = None,
) -> pd.DataFrame:
    input_path = _resolve_path(input_path)

    if input_source == "csv":
        return load_runtime_phone_csv(input_path, config=csv_config or RuntimePhoneCsvConfig())
    if input_source == "phone_folder":
        return load_runtime_phone_folder(input_path, config=phone_folder_config or RuntimePhoneFolderConfig())
    if input_source == "ucihar":
        return load_uci_har(input_path)
    if input_source == "pamap2":
        return load_pamap2(input_path)
    if input_source == "mobifall":
        return load_mobifall(input_path)
    if input_source == "sisfall":
        return load_sisfall(input_path)

    raise ValueError(f"Unsupported input_source: {input_source}")


def _attach_session_vulnerability_aggregates(
        session_summaries: pd.DataFrame,
        vulnerability_df: pd.DataFrame,
) -> pd.DataFrame:
    """Add per-session vulnerability + HAR-attenuation aggregates.

    The session-level vulnerability cap applied by
    ``fusion.vulnerability_score._dynamic_har_attenuation`` (walking/stairs
    gating) is recorded per window in ``vulnerability_contributions
    ['dynamic_har_attenuation']``. The admin currently displays
    ``highest_fall_probability`` (raw model output) as the headline, which
    is *not* attenuated and so makes confident-walking sessions look like
    likely falls. Rolling the cap up to the session row lets the admin
    surface the attenuated ``vulnerability_score`` as the headline and
    badge the session as HAR-gated when appropriate.

    The returned frame is the original ``session_summaries`` left-joined
    against per-session aggregates from ``vulnerability_df``. If either
    side is empty the original frame is returned unchanged so callers and
    tests don't have to special-case the no-vulnerability path.
    """
    if session_summaries is None or session_summaries.empty:
        return session_summaries
    if vulnerability_df is None or vulnerability_df.empty:
        return session_summaries
    if "session_id" not in vulnerability_df.columns or "session_id" not in session_summaries.columns:
        return session_summaries

    vdf = vulnerability_df.copy()
    vdf["session_id"] = vdf["session_id"].astype(str)

    if "vulnerability_contributions" in vdf.columns:
        vdf["__attenuation"] = vdf["vulnerability_contributions"].apply(
            lambda c: _safe_float(
                c.get("dynamic_har_attenuation", 1.0) if isinstance(c, dict) else 1.0,
                default=1.0,
            )
        )
    else:
        vdf["__attenuation"] = 1.0
    vdf["__attenuated"] = vdf["__attenuation"] < 1.0

    aggregates: list[dict[str, Any]] = []
    for session_id, group in vdf.groupby("session_id", dropna=False, sort=False):
        scores = pd.to_numeric(group.get("vulnerability_score"), errors="coerce")
        levels = group.get("vulnerability_level")

        attenuated_mask = group["__attenuated"]
        attenuated_rows = group[attenuated_mask]

        if not attenuated_rows.empty and "har_label" in attenuated_rows.columns:
            label_series = attenuated_rows["har_label"].dropna().astype(str)
            label_series = label_series[label_series.str.strip() != ""]
            if not label_series.empty:
                har_attenuation_label = label_series.value_counts().idxmax()
            else:
                har_attenuation_label = None
        else:
            har_attenuation_label = None

        if not attenuated_rows.empty and "har_confidence" in attenuated_rows.columns:
            conf_series = pd.to_numeric(
                attenuated_rows["har_confidence"], errors="coerce"
            ).dropna()
            har_attenuation_confidence_mean = (
                float(conf_series.mean()) if not conf_series.empty else None
            )
        else:
            har_attenuation_confidence_mean = None

        if levels is not None and not levels.dropna().empty:
            dominant_level = str(
                levels.dropna().astype(str).value_counts().idxmax()
            )
        else:
            dominant_level = None

        aggregates.append(
            {
                "session_id": str(session_id),
                "peak_vulnerability_score": (
                    float(scores.max()) if scores.notna().any() else None
                ),
                "mean_vulnerability_score": (
                    float(scores.mean()) if scores.notna().any() else None
                ),
                "dominant_vulnerability_level": dominant_level,
                "har_attenuation_applied": bool(attenuated_mask.any()),
                "har_attenuation_window_count": int(attenuated_mask.sum()),
                "har_attenuation_label": har_attenuation_label,
                "har_attenuation_confidence_mean": har_attenuation_confidence_mean,
            }
        )

    if not aggregates:
        return session_summaries

    aggregates_df = pd.DataFrame(aggregates)
    enriched = session_summaries.copy()
    enriched["session_id"] = enriched["session_id"].astype(str)
    enriched = enriched.merge(aggregates_df, on="session_id", how="left")
    return enriched


def _restrict_sessions(df: pd.DataFrame, *, session_id: str | None, max_sessions: int | None) -> pd.DataFrame:
    if "session_id" not in df.columns:
        return df.copy().reset_index(drop=True)

    working = df.copy()
    working["session_id"] = working["session_id"].astype(str)

    if session_id:
        out = working[working["session_id"] == str(session_id)].copy()
        if out.empty:
            raise ValueError(f"Requested session_id not found: {session_id}")
        return out.reset_index(drop=True)

    if max_sessions is None or max_sessions <= 0:
        return working.reset_index(drop=True)

    keep_sessions = sorted(working["session_id"].dropna().astype(str).unique().tolist())[: max_sessions]
    out = working[working["session_id"].astype(str).isin(keep_sessions)].copy()
    return out.reset_index(drop=True)


def _has_ground_truth_labels(df: pd.DataFrame) -> bool:
    if "label_mapped" in df.columns and df["label_mapped"].dropna().astype(str).str.strip().ne("").any():
        return True
    if "label_raw" in df.columns and df["label_raw"].dropna().astype(str).str.strip().ne("").any():
        return True
    return False


def _make_detector_config(dataset_name: str | None, threshold_mode: str):
    if threshold_mode == "dataset_presets":
        return default_fall_threshold_config(dataset_name)
    return default_fall_threshold_config(None)


def _artifact_har_preprocess(
        har_artifact_path: Path,
        config: RuntimeInferenceConfig,
) -> dict[str, Any]:
    artifact = load_har_model_artifact(har_artifact_path)
    metadata = dict(artifact.get("metadata", {}))
    return {
        "target_rate_hz": float(
            config.har_target_rate if config.har_target_rate is not None else metadata.get("target_rate_hz", 50.0)
        ),
        "window_size": int(
            config.har_window_size if config.har_window_size is not None else metadata.get("window_size", 128)
        ),
        "step_size": int(
            config.har_step_size if config.har_step_size is not None else metadata.get("step_size", 64)
        ),
    }


def _artifact_fall_preprocess(
        fall_artifact_path: Path,
        config: RuntimeInferenceConfig,
) -> dict[str, Any]:
    """Resolve fall branch window parameters from explicit config or artifact metadata.

    Mirrors ``_artifact_har_preprocess``. The fall artifact stores its
    training-time window parameters under ``metadata.training_config`` (see
    ``scripts/run_fall_artifact_train.py``). If neither config nor artifact
    metadata supplies a value, falls back to the historical 100 Hz / 128-sample
    / 64-step defaults. Returns the loaded artifact dict alongside the
    resolved cfg so the caller does not need a second joblib load.
    """
    artifact = load_fall_model_artifact(fall_artifact_path)
    metadata = dict(artifact.get("metadata") or {})
    training_config = dict(metadata.get("training_config") or {})

    def _resolve(override: Any, *metadata_keys: str, fallback: Any) -> Any:
        if override is not None:
            return override
        for key in metadata_keys:
            if training_config.get(key) is not None:
                return training_config[key]
            if metadata.get(key) is not None:
                return metadata[key]
        return fallback

    raw_placement_thresholds = metadata.get("thresholds_by_placement")
    placement_thresholds: dict[str, float] | None = None
    if isinstance(raw_placement_thresholds, dict) and raw_placement_thresholds:
        placement_thresholds = {}
        for key, value in raw_placement_thresholds.items():
            try:
                placement_thresholds[str(key).lower()] = float(value)
            except (TypeError, ValueError):
                continue
        if not placement_thresholds:
            placement_thresholds = None

    return {
        "target_rate_hz": float(
            _resolve(config.fall_target_rate, "target_rate_hz", fallback=100.0)
        ),
        "window_size": int(
            _resolve(config.fall_window_size, "window_size", fallback=128)
        ),
        "step_size": int(
            _resolve(config.fall_step_size, "step_size", fallback=64)
        ),
        "thresholds_by_placement": placement_thresholds,
        "artifact": artifact,
    }


def _prepare_har_branch(
        source_df: pd.DataFrame,
        *,
        har_artifact_path: Path,
        config: RuntimeInferenceConfig,
        precomputed_resampled: pd.DataFrame | None = None,
        precomputed_windows: list[dict[str, Any]] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    har_cfg = _artifact_har_preprocess(har_artifact_path, config)

    if precomputed_resampled is not None and precomputed_windows is not None:
        resampled = precomputed_resampled
        windows = precomputed_windows
    else:
        resampled = resample_dataframe(source_df, target_rate_hz=har_cfg["target_rate_hz"])
        resampled = append_derived_channels(resampled)

        preprocess_cfg = PreprocessConfig(target_sampling_rate_hz=har_cfg["target_rate_hz"])
        windows = window_dataframe(
            resampled,
            window_size=har_cfg["window_size"],
            step_size=har_cfg["step_size"],
            config=preprocess_cfg,
        )

    has_labels = _has_ground_truth_labels(source_df)
    feature_df = build_feature_table(
        windows,
        filter_unacceptable=has_labels,
        default_sampling_rate_hz=har_cfg["target_rate_hz"],
    )
    if feature_df.empty:
        raise ValueError("HAR feature table is empty")

    preds = predict_har_from_artifact_path(feature_df, artifact_path=har_artifact_path)

    out = feature_df.copy()
    for col in preds.columns:
        out[col] = preds[col]

    if "start_ts" in out.columns and "end_ts" in out.columns:
        out["midpoint_ts"] = (
                                     pd.to_numeric(out["start_ts"], errors="coerce")
                                     + pd.to_numeric(out["end_ts"], errors="coerce")
                             ) / 2.0
    else:
        out["midpoint_ts"] = np.arange(len(out), dtype=float)

    summary = {
        "target_rate_hz": har_cfg["target_rate_hz"],
        "window_size": har_cfg["window_size"],
        "step_size": har_cfg["step_size"],
        "rows_after_resample": int(len(resampled)),
        "windows_total": int(len(windows)),
        "prediction_rows": int(len(out)),
        "used_label_based_filtering": bool(has_labels),
        "predicted_label_counts": out["predicted_label"].astype(str).value_counts(dropna=False).to_dict(),
    }

    if "label_mapped_majority" in out.columns:
        try:
            labels_used = sorted(set(out["label_mapped_majority"].astype(str)) | set(out["predicted_label"].astype(str)))
            summary["metrics"] = compute_classification_metrics(
                out["label_mapped_majority"].astype(str).tolist(),
                out["predicted_label"].astype(str).tolist(),
                labels=labels_used,
            )
        except Exception as exc:  # noqa: BLE001
            summary["metrics_error"] = str(exc)

    return out, summary


def _prepare_fall_branch(
        source_df: pd.DataFrame,
        *,
        fall_artifact_path: Path,
        config: RuntimeInferenceConfig,
        precomputed_resampled: pd.DataFrame | None = None,
        precomputed_windows: list[dict[str, Any]] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    fall_cfg = _artifact_fall_preprocess(fall_artifact_path, config)
    artifact = fall_cfg["artifact"]

    if precomputed_resampled is not None and precomputed_windows is not None:
        resampled = precomputed_resampled
        windows = precomputed_windows
    else:
        resampled = resample_dataframe(source_df, target_rate_hz=fall_cfg["target_rate_hz"])
        resampled = append_derived_channels(resampled)

        preprocess_cfg = PreprocessConfig(target_sampling_rate_hz=fall_cfg["target_rate_hz"])
        windows = window_dataframe(
            resampled,
            window_size=fall_cfg["window_size"],
            step_size=fall_cfg["step_size"],
            config=preprocess_cfg,
        )

    dataset_name = None
    if "dataset_name" in source_df.columns and not source_df["dataset_name"].dropna().empty:
        dataset_name = str(source_df["dataset_name"].dropna().astype(str).iloc[0])

    detector_config = _make_detector_config(dataset_name, config.threshold_mode)
    has_labels = _has_ground_truth_labels(source_df)

    threshold_df = build_threshold_prediction_table(
        windows,
        detector_config=detector_config,
        filter_unacceptable=has_labels,
        default_sampling_rate_hz=fall_cfg["target_rate_hz"],
    )
    if threshold_df.empty:
        raise ValueError("Fall threshold prediction table is empty")

    preds = predict_fall_from_artifact_path(threshold_df, artifact_path=fall_artifact_path)

    out = threshold_df.copy()
    for col in preds.columns:
        out[col] = preds[col]

    if "start_ts" in out.columns and "end_ts" in out.columns:
        out["midpoint_ts"] = (
                                     pd.to_numeric(out["start_ts"], errors="coerce")
                                     + pd.to_numeric(out["end_ts"], errors="coerce")
                             ) / 2.0
    else:
        out["midpoint_ts"] = np.arange(len(out), dtype=float)

    summary = {
        "target_rate_hz": fall_cfg["target_rate_hz"],
        "window_size": fall_cfg["window_size"],
        "step_size": fall_cfg["step_size"],
        "rows_after_resample": int(len(resampled)),
        "windows_total": int(len(windows)),
        "prediction_rows": int(len(out)),
        "used_label_based_filtering": bool(has_labels),
        "predicted_label_counts": out["predicted_label"].astype(str).value_counts(dropna=False).to_dict(),
        "probability_threshold_used": float(artifact["probability_threshold"]),
    }

    if "true_label" in out.columns:
        try:
            summary["metrics"] = compute_fall_detection_metrics(
                out["true_label"].astype(str).tolist(),
                out["predicted_label"].astype(str).tolist(),
                positive_label=str(artifact["positive_label"]),
                negative_label=str(artifact["negative_label"]),
            )
        except Exception as exc:  # noqa: BLE001
            summary["metrics_error"] = str(exc)

    return out, summary


def _timeline_subset_har(df: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "dataset_name",
        "subject_id",
        "session_id",
        "midpoint_ts",
        "window_id",
        "label_mapped_majority",
        "predicted_label",
        "predicted_confidence",
    ]
    out = df[[c for c in keep if c in df.columns]].copy()
    return out.rename(
        columns={
            "window_id": "har_window_id",
            "label_mapped_majority": "har_true_label",
            "predicted_label": "har_predicted_label",
            "predicted_confidence": "har_predicted_confidence",
        }
    )


def _timeline_subset_fall(df: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "dataset_name",
        "subject_id",
        "session_id",
        "midpoint_ts",
        "window_id",
        "true_label",
        "predicted_label",
        "predicted_probability",
        "predicted_is_fall",
    ]
    out = df[[c for c in keep if c in df.columns]].copy()
    return out.rename(
        columns={
            "window_id": "fall_window_id",
            "true_label": "fall_true_label",
            "predicted_label": "fall_predicted_label",
            "predicted_probability": "fall_predicted_probability",
            "predicted_is_fall": "fall_predicted_is_fall",
        }
    )


def _merge_timelines_by_session(
        har_df: pd.DataFrame,
        fall_df: pd.DataFrame,
        *,
        tolerance_seconds: float,
) -> pd.DataFrame:
    har_tl = _timeline_subset_har(har_df)
    fall_tl = _timeline_subset_fall(fall_df)

    if har_tl.empty and fall_tl.empty:
        return pd.DataFrame()
    if har_tl.empty:
        return fall_tl.copy()
    if fall_tl.empty:
        return har_tl.copy()

    if "session_id" not in har_tl.columns:
        har_tl["session_id"] = "unknown_session"
    if "session_id" not in fall_tl.columns:
        fall_tl["session_id"] = "unknown_session"

    session_ids = sorted(set(har_tl["session_id"].astype(str)) | set(fall_tl["session_id"].astype(str)))
    merged_parts: list[pd.DataFrame] = []

    for sid in session_ids:
        h = har_tl[har_tl["session_id"].astype(str) == sid].copy()
        f = fall_tl[fall_tl["session_id"].astype(str) == sid].copy()

        if h.empty:
            merged_parts.append(f)
            continue
        if f.empty:
            merged_parts.append(h)
            continue

        h = h.sort_values("midpoint_ts").reset_index(drop=True)
        f = f.sort_values("midpoint_ts").reset_index(drop=True)

        use_group_keys = {"dataset_name", "subject_id", "session_id"} <= (set(h.columns) & set(f.columns))
        merged = pd.merge_asof(
            h,
            f,
            on="midpoint_ts",
            by=["dataset_name", "subject_id", "session_id"] if use_group_keys else "session_id",
            direction="nearest",
            tolerance=float(tolerance_seconds),
        )
        merged_parts.append(merged)

    out = pd.concat(merged_parts, ignore_index=True, sort=False)
    out = out.sort_values(["session_id", "midpoint_ts"], kind="stable").reset_index(drop=True)
    return out


def _align_har_predictions_with_pairing(
        fall_df: pd.DataFrame,
        har_df: pd.DataFrame,
        pairing: pd.DataFrame,
) -> pd.DataFrame:
    """Join HAR predictions onto fall windows via a precomputed pairing.

    The pairing table is produced by
    :func:`pipeline.windowing.synchronize_windows` and keyed on
    ``fall_window_id`` + ``har_window_id``. Using it here avoids re-running
    :func:`pandas.merge_asof` after inference.
    """
    working = fall_df.copy().reset_index(drop=True)
    if working.empty:
        working["har_predicted_label"] = pd.Series(dtype="object")
        working["har_predicted_confidence"] = pd.Series(dtype="float64")
        return working

    if "window_id" not in working.columns or pairing.empty or har_df.empty:
        working["har_predicted_label"] = None
        working["har_predicted_confidence"] = np.nan
        return working

    pair = pairing[["fall_window_id", "har_window_id"]].copy()
    pair["fall_window_id"] = pd.to_numeric(pair["fall_window_id"], errors="coerce")
    pair["har_window_id"] = pd.to_numeric(pair["har_window_id"], errors="coerce")

    har_cols = {"window_id"}
    if "predicted_label" in har_df.columns:
        har_cols.add("predicted_label")
    if "predicted_confidence" in har_df.columns:
        har_cols.add("predicted_confidence")
    har_small = har_df[[c for c in har_cols if c in har_df.columns]].copy()
    har_small = har_small.rename(
        columns={
            "window_id": "har_window_id",
            "predicted_label": "har_predicted_label",
            "predicted_confidence": "har_predicted_confidence",
        }
    )
    har_small["har_window_id"] = pd.to_numeric(har_small["har_window_id"], errors="coerce")

    pair = pair.merge(har_small, on="har_window_id", how="left")
    working["fall_window_id"] = pd.to_numeric(working["window_id"], errors="coerce")
    out = working.merge(
        pair[["fall_window_id", "har_predicted_label", "har_predicted_confidence"]],
        on="fall_window_id",
        how="left",
    )
    out = out.drop(columns=["fall_window_id"], errors="ignore")
    if "har_predicted_label" not in out.columns:
        out["har_predicted_label"] = None
    if "har_predicted_confidence" not in out.columns:
        out["har_predicted_confidence"] = np.nan
    return out


def _align_har_predictions_to_fall_windows(
        fall_df: pd.DataFrame,
        har_df: pd.DataFrame,
        *,
        tolerance_seconds: float,
        pairing: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if pairing is not None:
        return _align_har_predictions_with_pairing(fall_df, har_df, pairing)

    working = fall_df.copy().reset_index(drop=True)
    working["_orig_order"] = np.arange(len(working))

    if working.empty:
        working["har_predicted_label"] = pd.Series(dtype="object")
        working["har_predicted_confidence"] = pd.Series(dtype="float64")
        return working.drop(columns=["_orig_order"], errors="ignore")

    if "midpoint_ts" not in working.columns:
        if {"start_ts", "end_ts"} <= set(working.columns):
            working["midpoint_ts"] = (
                    pd.to_numeric(working["start_ts"], errors="coerce")
                    + pd.to_numeric(working["end_ts"], errors="coerce")
            ) / 2.0
        else:
            working["midpoint_ts"] = np.arange(len(working), dtype=float)

    if har_df.empty:
        working["har_predicted_label"] = None
        working["har_predicted_confidence"] = np.nan
        return working.drop(columns=["_orig_order"], errors="ignore")

    har_working = har_df.copy().reset_index(drop=True)
    if "midpoint_ts" not in har_working.columns:
        if {"start_ts", "end_ts"} <= set(har_working.columns):
            har_working["midpoint_ts"] = (
                    pd.to_numeric(har_working["start_ts"], errors="coerce")
                    + pd.to_numeric(har_working["end_ts"], errors="coerce")
            ) / 2.0
        else:
            har_working["midpoint_ts"] = np.arange(len(har_working), dtype=float)

    if "session_id" not in working.columns:
        working["session_id"] = "unknown_session"
    if "session_id" not in har_working.columns:
        har_working["session_id"] = "unknown_session"

    keep_cols = [
        "session_id",
        "midpoint_ts",
        "predicted_label",
        "predicted_confidence",
    ]
    har_working = har_working[[c for c in keep_cols if c in har_working.columns]].copy()
    har_working = har_working.rename(
        columns={
            "predicted_label": "har_predicted_label",
            "predicted_confidence": "har_predicted_confidence",
        }
    )

    merged_parts: list[pd.DataFrame] = []
    for session_id, fall_group in working.groupby("session_id", dropna=False, sort=False):
        fall_group = fall_group.sort_values("midpoint_ts", kind="stable").reset_index(drop=True)
        har_group = har_working[har_working["session_id"].astype(str) == str(session_id)].copy()
        if har_group.empty:
            fall_group["har_predicted_label"] = None
            fall_group["har_predicted_confidence"] = np.nan
            merged_parts.append(fall_group)
            continue

        har_group = har_group.sort_values("midpoint_ts", kind="stable").reset_index(drop=True)
        merged = pd.merge_asof(
            fall_group,
            har_group,
            on="midpoint_ts",
            by="session_id",
            direction="nearest",
            tolerance=float(tolerance_seconds),
        )
        merged_parts.append(merged)

    out = pd.concat(merged_parts, ignore_index=True, sort=False)
    out = out.sort_values("_orig_order", kind="stable").reset_index(drop=True)
    return out.drop(columns=["_orig_order"], errors="ignore")


def _sequence_sort_columns(df: pd.DataFrame) -> list[str]:
    preferred = ["dataset_name", "subject_id", "session_id", "start_ts", "end_ts", "window_id"]
    return [c for c in preferred if c in df.columns]


def _vulnerability_group_columns(df: pd.DataFrame) -> list[str]:
    preferred = ["dataset_name", "subject_id", "session_id"]
    cols = [c for c in preferred if c in df.columns]
    return cols or ["__single_group__"]


def _infer_runtime_recovery_detected(row: pd.Series, thresholds: FallEventThresholds) -> bool:
    if "recovery_detected" in row.index:
        return _safe_bool(row.get("recovery_detected"))

    ratio = _safe_float(row.get("post_impact_motion_to_peak_ratio"), default=0.0)
    fall_probability = _safe_float(row.get("predicted_probability"), default=np.nan)

    if ratio >= thresholds.high_motion_ratio_threshold:
        if not np.isfinite(fall_probability):
            return True
        if fall_probability < 0.50:
            return True

    return False


def _infer_runtime_har_label(
        row: pd.Series,
        fall_event: FallEventResult,
        thresholds: FallEventThresholds,
) -> str | None:
    label = _safe_str(row.get("har_predicted_label"), default="")
    if label:
        return label

    available = _safe_bool(row.get("post_impact_available"), default=False)
    ratio = _safe_float(row.get("post_impact_motion_to_peak_ratio"), default=0.0)
    variance = _safe_float(row.get("post_impact_variance"), default=0.0)

    if not available:
        return None
    if ratio <= thresholds.low_motion_ratio_threshold and variance <= thresholds.low_variance_threshold:
        return "static"
    if ratio >= thresholds.high_motion_ratio_threshold:
        return "locomotion"
    if fall_event.state == FallEventState.IMPACT_ONLY:
        return "other"
    return None


def _infer_runtime_inactivity_seconds(row: pd.Series, thresholds: FallEventThresholds) -> float:
    available = _safe_bool(row.get("post_impact_available"), default=False)
    if not available:
        return 0.0

    ratio = _safe_float(row.get("post_impact_motion_to_peak_ratio"), default=0.0)
    variance = _safe_float(row.get("post_impact_variance"), default=0.0)

    if ratio <= thresholds.low_motion_ratio_threshold and variance <= thresholds.low_variance_threshold:
        return 20.0
    if ratio <= thresholds.medium_motion_ratio_threshold and variance <= thresholds.medium_variance_threshold:
        return 12.0
    if ratio <= thresholds.high_motion_ratio_threshold:
        return 6.0
    return 1.0


def _build_runtime_fall_event_inputs(
        row: pd.Series,
        thresholds: FallEventThresholds,
) -> FallEventInputs:
    predicted_probability = None
    if "predicted_probability" in row.index:
        predicted_probability = _safe_optional_float(row.get("predicted_probability"))

    return FallEventInputs(
        peak_acc=_safe_float(row.get("peak_acc"), default=0.0),
        stage_impact_pass=_safe_bool(row.get("stage_impact_pass"), default=False),
        stage_confirm_pass=_safe_bool(row.get("stage_confirm_pass"), default=False),
        stage_support_pass=_safe_bool(row.get("stage_support_pass"), default=False),
        post_impact_available=_safe_bool(row.get("post_impact_available"), default=False),
        post_impact_motion_to_peak_ratio=_safe_float(row.get("post_impact_motion_to_peak_ratio"), default=0.0),
        post_impact_variance=_safe_float(row.get("post_impact_variance"), default=0.0),
        post_impact_dyn_ratio_mean=_safe_float(row.get("post_impact_dyn_ratio_mean"), default=0.0),
        recovery_detected=_infer_runtime_recovery_detected(row, thresholds),
        meta_probability=predicted_probability,
        meta_predicted_is_fall=(
            _safe_bool(row.get("predicted_is_fall"), default=False)
            or _safe_str(row.get("predicted_label"), default="").strip().lower() == "fall"
        ),
    )


def _build_runtime_vulnerability_inputs(
        row: pd.Series,
        fall_event: FallEventResult,
        thresholds: FallEventThresholds,
) -> VulnerabilityInputs:
    recovery_detected = _infer_runtime_recovery_detected(row, thresholds)
    har_label = _infer_runtime_har_label(row, fall_event, thresholds)
    har_confidence = _safe_float(row.get("har_predicted_confidence"), default=0.0) if har_label else 0.0

    predicted_probability = _safe_optional_float(row.get("predicted_probability"))
    fall_probability = (
        predicted_probability
        if predicted_probability is not None
        else _safe_float(row.get("meta_probability"), default=fall_event.confidence)
    )

    return VulnerabilityInputs(
        fall_probability=fall_probability,
        impact_detected=_safe_bool(row.get("stage_impact_pass"), default=False)
        or fall_event.state != FallEventState.NO_EVENT,
        fall_event_state=fall_event.state.value,
        fall_event_confidence=fall_event.confidence,
        har_label=har_label,
        har_confidence=har_confidence,
        inactivity_seconds=_infer_runtime_inactivity_seconds(row, thresholds),
        recovery_detected=recovery_detected,
        post_event_motion_ratio=_safe_float(row.get("post_impact_motion_to_peak_ratio"), default=0.0),
        lying_detected=bool(har_label and har_label.strip().lower() in {"lying", "laying"}),
        hr_anomaly_score=0.0,
    )


def _build_runtime_vulnerability_branch(
        fall_df: pd.DataFrame,
        har_df: pd.DataFrame,
        *,
        config: RuntimeInferenceConfig,
        pairing: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    empty_summary = {
        "enabled": bool(config.enable_vulnerability_scoring),
        "event_profile": str(config.event_profile),
        "vulnerability_profile": str(config.vulnerability_profile),
        "window_count": 0,
        "session_count": 0,
        "alert_worthy_window_count": 0,
        "fall_event_state_counts": {},
        "vulnerability_level_counts": {},
        "monitoring_state_counts": {},
        "top_vulnerability_score": None,
        "mean_vulnerability_score": None,
        "latest_vulnerability_score": None,
        "latest_vulnerability_level": None,
        "latest_monitoring_state": None,
        "latest_fall_event_state": None,
        "top_vulnerability_reasons": [],
    }
    if not config.enable_vulnerability_scoring or fall_df.empty:
        return pd.DataFrame(), empty_summary

    aligned_df = _align_har_predictions_to_fall_windows(
        fall_df,
        har_df,
        tolerance_seconds=float(config.timeline_tolerance_seconds),
        pairing=pairing,
    )
    if "__single_group__" not in aligned_df.columns:
        aligned_df["__single_group__"] = "all"

    sort_cols = _sequence_sort_columns(aligned_df)
    if sort_cols:
        aligned_df = aligned_df.sort_values(sort_cols, kind="stable").reset_index(drop=True)
    else:
        aligned_df = aligned_df.reset_index(drop=True)

    rows: list[dict[str, Any]] = []
    group_cols = _vulnerability_group_columns(aligned_df)
    vuln_profile = get_vulnerability_profile(config.vulnerability_profile)
    sm_config = config.state_machine_config or StateMachineConfig()

    for _, group_df in aligned_df.groupby(group_cols, dropna=False, sort=False):
        machine_state = StateMachineState()
        for _, row in group_df.iterrows():
            dataset_name = _safe_str(row.get("dataset_name"), default="UNKNOWN")
            thresholds = get_fall_event_thresholds(dataset_name, profile=config.event_profile)

            fall_event_inputs = _build_runtime_fall_event_inputs(row, thresholds)
            fall_event = classify_fall_event(fall_event_inputs, thresholds=thresholds)

            vulnerability_inputs = _build_runtime_vulnerability_inputs(row, fall_event, thresholds)
            vulnerability = score_vulnerability(vulnerability_inputs, profile=vuln_profile)

            state_inputs = StateMachineInputs(
                fall_event_state=fall_event.state,
                fall_event_confidence=fall_event.confidence,
                vulnerability_level=vulnerability.level,
                vulnerability_score=vulnerability.score,
                har_label=vulnerability_inputs.har_label,
                recovery_detected=vulnerability_inputs.recovery_detected,
            )
            transition = step_state_machine(machine_state, state_inputs, config=sm_config)

            rows.append(
                {
                    "window_id": row.get("window_id"),
                    "dataset_name": row.get("dataset_name"),
                    "subject_id": row.get("subject_id"),
                    "session_id": row.get("session_id"),
                    "start_ts": _safe_optional_float(row.get("start_ts")),
                    "end_ts": _safe_optional_float(row.get("end_ts")),
                    "midpoint_ts": _safe_optional_float(row.get("midpoint_ts")),
                    "har_label": vulnerability_inputs.har_label,
                    "har_confidence": vulnerability_inputs.har_confidence if vulnerability_inputs.har_label else None,
                    "fall_probability": vulnerability_inputs.fall_probability,
                    "fall_predicted_label": row.get("predicted_label"),
                    "fall_predicted_is_fall": _safe_bool(row.get("predicted_is_fall"), default=False),
                    "fall_event_state": fall_event.state.value,
                    "fall_event_confidence": fall_event.confidence,
                    "fall_event_reasons": list(fall_event.reasons),
                    "fall_event_contributions": dict(fall_event.contributions),
                    "vulnerability_level": vulnerability.level.value,
                    "vulnerability_score": vulnerability.score,
                    "vulnerability_reasons": list(vulnerability.reasons),
                    "vulnerability_contributions": dict(vulnerability.contributions),
                    "monitoring_state": transition.current_state.value,
                    "escalated": transition.escalated,
                    "deescalated": transition.deescalated,
                    "state_machine_reasons": list(transition.reasons),
                    "state_machine_counters": dict(transition.counters),
                }
            )

    result_df = pd.DataFrame(rows)
    if result_df.empty:
        return result_df, empty_summary

    latest_row = result_df.iloc[-1]
    top_idx = pd.to_numeric(result_df["vulnerability_score"], errors="coerce").idxmax()
    top_row = result_df.loc[top_idx]

    summary = {
        "enabled": True,
        "event_profile": str(config.event_profile),
        "vulnerability_profile": vuln_profile.name,
        "window_count": int(len(result_df)),
        "session_count": int(result_df["session_id"].astype(str).nunique()) if "session_id" in result_df.columns else 1,
        "alert_worthy_window_count": int(
            result_df["vulnerability_level"].astype(str).isin(
                {VulnerabilityLevel.MEDIUM.value, VulnerabilityLevel.HIGH.value}
            ).sum()
        ),
        "fall_event_state_counts": result_df["fall_event_state"].astype(str).value_counts(dropna=False).to_dict(),
        "vulnerability_level_counts": result_df["vulnerability_level"].astype(str).value_counts(dropna=False).to_dict(),
        "monitoring_state_counts": result_df["monitoring_state"].astype(str).value_counts(dropna=False).to_dict(),
        "top_vulnerability_score": _safe_optional_float(top_row.get("vulnerability_score")),
        "mean_vulnerability_score": _safe_optional_float(
            pd.to_numeric(result_df["vulnerability_score"], errors="coerce").mean()
        ),
        "latest_vulnerability_score": _safe_optional_float(latest_row.get("vulnerability_score")),
        "latest_vulnerability_level": _safe_str(latest_row.get("vulnerability_level"), default="") or None,
        "latest_monitoring_state": _safe_str(latest_row.get("monitoring_state"), default="") or None,
        "latest_fall_event_state": _safe_str(latest_row.get("fall_event_state"), default="") or None,
        "top_vulnerability_reasons": list(top_row.get("vulnerability_reasons") or []),
    }
    return result_df, summary


def _resolve_threshold_series(
        fall_df: pd.DataFrame,
        *,
        probability_threshold: float | None,
        thresholds_by_placement: dict[str, float] | None,
) -> float | pd.Series | None:
    """Pick the threshold(s) used to mark positive windows.

    Returns a scalar when no per-placement overrides apply (matches existing
    behaviour), or a per-row Series when ``thresholds_by_placement`` is set
    and the dataframe carries a ``placement`` column. Rows with a placement
    not in the mapping fall back to ``probability_threshold``.
    """
    if not thresholds_by_placement or "placement" not in fall_df.columns:
        return probability_threshold
    fallback = float(probability_threshold) if probability_threshold is not None else 0.5
    placements = fall_df["placement"].astype(str).str.lower()
    lookup = {str(k).lower(): float(v) for k, v in thresholds_by_placement.items()}
    return placements.map(lookup).fillna(fallback).astype(float)


def _mark_positive_windows(
        fall_df: pd.DataFrame,
        probability_threshold: float | pd.Series | None,
) -> pd.Series:
    if probability_threshold is not None and "predicted_probability" in fall_df.columns:
        prob = pd.to_numeric(fall_df["predicted_probability"], errors="coerce").fillna(0.0)
        if isinstance(probability_threshold, pd.Series):
            return prob >= probability_threshold.reindex(prob.index).astype(float)
        return prob >= float(probability_threshold)

    if "predicted_is_fall" in fall_df.columns:
        if fall_df["predicted_is_fall"].dtype == bool:
            return fall_df["predicted_is_fall"].fillna(False)
        lowered = fall_df["predicted_is_fall"].astype(str).str.lower()
        if lowered.isin({"true", "false"}).all():
            return lowered.map({"true": True, "false": False}).fillna(False)
        return fall_df["predicted_is_fall"].astype(bool)

    if "predicted_label" in fall_df.columns:
        return fall_df["predicted_label"].astype(str).str.lower().eq("fall")

    if "predicted_probability" in fall_df.columns:
        prob = pd.to_numeric(fall_df["predicted_probability"], errors="coerce").fillna(0.0)
        return prob >= 0.5

    raise ValueError("Fall dataframe has no usable prediction columns for event grouping")


def _group_runtime_fall_events(
        fall_df: pd.DataFrame,
        *,
        probability_threshold: float | None,
        merge_gap_seconds: float,
        min_windows: int,
        max_event_duration_seconds: float,
        thresholds_by_placement: dict[str, float] | None = None,
) -> pd.DataFrame:
    if fall_df.empty:
        return pd.DataFrame()

    working = fall_df.copy()
    for col in ["start_ts", "end_ts", "midpoint_ts"]:
        if col in working.columns:
            working[col] = pd.to_numeric(working[col], errors="coerce")

    if "session_id" not in working.columns:
        working["session_id"] = "unknown_session"
    if "dataset_name" not in working.columns:
        working["dataset_name"] = "unknown_dataset"
    if "subject_id" not in working.columns:
        working["subject_id"] = "unknown_subject"

    resolved_threshold = _resolve_threshold_series(
        working,
        probability_threshold=probability_threshold,
        thresholds_by_placement=thresholds_by_placement,
    )
    working["is_positive_window"] = _mark_positive_windows(working, resolved_threshold)
    working = working[working["is_positive_window"]].copy()
    if working.empty:
        return pd.DataFrame(
            columns=[
                "event_id",
                "dataset_name",
                "subject_id",
                "session_id",
                "event_start_ts",
                "event_end_ts",
                "event_duration_seconds",
                "n_positive_windows",
                "peak_probability",
                "mean_probability",
                "median_probability",
                "first_midpoint_ts",
                "last_midpoint_ts",
            ]
        )

    working = working.sort_values(["session_id", "start_ts", "midpoint_ts"], kind="stable").reset_index(drop=True)

    events: list[dict[str, Any]] = []
    event_idx = 0

    def flush_chunk(session_id: str, rows: list[pd.Series]) -> None:
        nonlocal event_idx
        if len(rows) < int(min_windows):
            return

        chunk_df = pd.DataFrame(rows)
        event_start = float(pd.to_numeric(chunk_df["start_ts"], errors="coerce").min())
        event_end = float(pd.to_numeric(chunk_df["end_ts"], errors="coerce").max())
        prob = (
            pd.to_numeric(chunk_df["predicted_probability"], errors="coerce")
            if "predicted_probability" in chunk_df.columns
            else pd.Series([float("nan")] * len(chunk_df))
        )
        events.append(
            {
                "event_id": f"{session_id}_event_{event_idx:03d}",
                "dataset_name": str(chunk_df["dataset_name"].iloc[0]),
                "subject_id": str(chunk_df["subject_id"].iloc[0]),
                "session_id": str(session_id),
                "event_start_ts": event_start,
                "event_end_ts": event_end,
                "event_duration_seconds": float(event_end - event_start),
                "n_positive_windows": int(len(chunk_df)),
                "peak_probability": float(prob.max()) if prob.notna().any() else float("nan"),
                "mean_probability": float(prob.mean()) if prob.notna().any() else float("nan"),
                "median_probability": float(prob.median()) if prob.notna().any() else float("nan"),
                "first_midpoint_ts": float(pd.to_numeric(chunk_df["midpoint_ts"], errors="coerce").min()),
                "last_midpoint_ts": float(pd.to_numeric(chunk_df["midpoint_ts"], errors="coerce").max()),
            }
        )
        event_idx += 1

    for session_id, group in working.groupby("session_id", dropna=False, sort=False):
        group = group.sort_values("start_ts", kind="stable").reset_index(drop=True)
        current_rows: list[pd.Series] = []
        current_start = None
        current_end = None

        for _, row in group.iterrows():
            row_start = float(row["start_ts"])
            row_end = float(row["end_ts"])

            if not current_rows:
                current_rows.append(row)
                current_start = row_start
                current_end = row_end
                continue

            gap = row_start - float(current_end)
            proposed_end = max(float(current_end), row_end)
            proposed_duration = proposed_end - float(current_start)

            if gap > float(merge_gap_seconds) or proposed_duration > float(max_event_duration_seconds):
                flush_chunk(str(session_id), current_rows)
                current_rows = [row]
                current_start = row_start
                current_end = row_end
            else:
                current_rows.append(row)
                current_end = proposed_end

        flush_chunk(str(session_id), current_rows)

    out = pd.DataFrame(events)
    if out.empty:
        return out

    out = out.sort_values(
        ["peak_probability", "mean_probability", "event_duration_seconds", "event_start_ts"],
        ascending=[False, False, True, True],
        kind="stable",
    ).reset_index(drop=True)
    return out


def run_runtime_inference_from_dataframe(
        source_df: pd.DataFrame,
        *,
        artifacts: RuntimeArtifacts,
        config: RuntimeInferenceConfig | None = None,
) -> RuntimeInferenceResult:
    cfg = config or RuntimeInferenceConfig()
    har_artifact_path = _resolve_path(artifacts.har_artifact_path)
    fall_artifact_path = _resolve_path(artifacts.fall_artifact_path)

    if not har_artifact_path.exists():
        raise FileNotFoundError(f"HAR artifact not found: {har_artifact_path}")
    if not fall_artifact_path.exists():
        raise FileNotFoundError(f"Fall artifact not found: {fall_artifact_path}")

    source_df = source_df.copy().reset_index(drop=True)

    placement_windows = pd.DataFrame()
    placement_summary: dict[str, Any] = {
        "window_count": 0,
        "placement_state": "unknown",
        "placement_confidence": 0.0,
        "state_fraction": 0.0,
        "state_counts": {},
    }

    if cfg.infer_placement_state:
        placement_cfg = cfg.placement_config or PlacementStateConfig()
        placement_source = resample_dataframe(
            source_df,
            target_rate_hz=placement_cfg.target_rate_hz,
        )
        placement_windows, placement_summary = infer_placement_state_from_dataframe(
            placement_source,
            config=placement_cfg,
        )

    har_preprocess = _artifact_har_preprocess(har_artifact_path, cfg)
    fall_preprocess = _artifact_fall_preprocess(fall_artifact_path, cfg)
    sync = synchronize_windows(
        source_df,
        har_cfg=BranchWindowConfig(
            target_rate_hz=har_preprocess["target_rate_hz"],
            window_size=har_preprocess["window_size"],
            step_size=har_preprocess["step_size"],
        ),
        fall_cfg=BranchWindowConfig(
            target_rate_hz=fall_preprocess["target_rate_hz"],
            window_size=fall_preprocess["window_size"],
            step_size=fall_preprocess["step_size"],
        ),
        tolerance_seconds=float(cfg.timeline_tolerance_seconds),
    )

    har_df, har_summary = _prepare_har_branch(
        source_df,
        har_artifact_path=har_artifact_path,
        config=cfg,
        precomputed_resampled=sync.har_resampled,
        precomputed_windows=sync.har_windows,
    )
    fall_df, fall_summary = _prepare_fall_branch(
        source_df,
        fall_artifact_path=fall_artifact_path,
        config=cfg,
        precomputed_resampled=sync.fall_resampled,
        precomputed_windows=sync.fall_windows,
    )
    har_summary["synchronizer_stats"] = dict(sync.stats)
    fall_summary["synchronizer_stats"] = dict(sync.stats)
    vulnerability_df, vulnerability_summary = _build_runtime_vulnerability_branch(
        fall_df,
        har_df,
        config=cfg,
        pairing=sync.pairing,
    )
    combined_df = _merge_timelines_by_session(
        har_df,
        fall_df,
        tolerance_seconds=cfg.timeline_tolerance_seconds,
    )

    grouped_events_df = pd.DataFrame()
    if cfg.group_fall_events:
        # Config-supplied overrides win; else fall back to whatever the
        # artifact metadata shipped (populated by the operating-point tuner).
        placement_thresholds = (
            cfg.event_thresholds_by_placement
            if cfg.event_thresholds_by_placement is not None
            else fall_preprocess.get("thresholds_by_placement")
        )
        grouped_events_df = _group_runtime_fall_events(
            fall_df,
            probability_threshold=cfg.event_probability_threshold,
            merge_gap_seconds=cfg.event_merge_gap_seconds,
            min_windows=cfg.event_min_windows,
            max_event_duration_seconds=cfg.event_max_duration_seconds,
            thresholds_by_placement=placement_thresholds,
        )

    point_timeline = pd.DataFrame()
    raw_timeline_events = pd.DataFrame()
    timeline_events = pd.DataFrame()
    transition_events = pd.DataFrame()
    session_summaries = pd.DataFrame()
    narrative_summary: dict[str, Any] = {
        "session_count": 0,
        "total_event_count": 0,
        "total_transition_count": 0,
        "total_fall_event_count": 0,
        "sessions": [],
    }

    if cfg.build_timeline_events:
        timeline_result = build_runtime_timeline_events(
            har_windows=har_df,
            fall_windows=fall_df,
            placement_windows=placement_windows,
            grouped_fall_events=grouped_events_df,
            config=cfg.timeline_config or RuntimeTimelineConfig(),
        )
        point_timeline = timeline_result.point_timeline
        raw_timeline_events = timeline_result.raw_timeline_events
        timeline_events = timeline_result.timeline_events
        transition_events = timeline_result.transition_events
        session_summaries = timeline_result.session_summaries
        narrative_summary = timeline_result.narrative_summary

    session_summaries = _attach_session_vulnerability_aggregates(
        session_summaries, vulnerability_df
    )

    source_summary = {
        "rows_loaded": int(len(source_df)),
        "dataset_name_counts": source_df["dataset_name"].astype(str).value_counts(dropna=False).to_dict()
        if "dataset_name" in source_df.columns
        else {},
        "session_counts": source_df["session_id"].astype(str).value_counts(dropna=False).to_dict()
        if "session_id" in source_df.columns
        else {},
    }

    report = {
        "service": "runtime_inference",
        "source_summary": source_summary,
        "har_summary": har_summary,
        "fall_summary": fall_summary,
        "vulnerability_summary": vulnerability_summary,
        "placement_summary": placement_summary,
        "grouped_fall_events": {
            "enabled": bool(cfg.group_fall_events),
            "count": int(len(grouped_events_df)),
            "event_probability_threshold": cfg.event_probability_threshold,
            "event_merge_gap_seconds": float(cfg.event_merge_gap_seconds),
            "event_min_windows": int(cfg.event_min_windows),
            "event_max_duration_seconds": float(cfg.event_max_duration_seconds),
        },
        "combined_timeline_rows": int(len(combined_df)),
        "timeline_events": {
            "enabled": bool(cfg.build_timeline_events),
            "point_count": int(len(point_timeline)),
            "event_count": int(len(timeline_events)),
            "transition_count": int(len(transition_events)),
            "session_summary_count": int(len(session_summaries)),
            "narrative_summary": narrative_summary,
        },
        "artifacts": {
            "har_artifact_path": str(har_artifact_path),
            "fall_artifact_path": str(fall_artifact_path),
        },
        "config": asdict(cfg),
    }

    return RuntimeInferenceResult(
        source_summary=source_summary,
        placement_summary=placement_summary,
        har_summary=har_summary,
        fall_summary=fall_summary,
        vulnerability_summary=vulnerability_summary,
        report=report,
        placement_windows=placement_windows,
        har_windows=har_df,
        fall_windows=fall_df,
        vulnerability_windows=vulnerability_df,
        combined_timeline=combined_df,
        grouped_fall_events=grouped_events_df,
        point_timeline=point_timeline,
        raw_timeline_events=raw_timeline_events,
        timeline_events=timeline_events,
        transition_events=transition_events,
        session_summaries=session_summaries,
        narrative_summary=narrative_summary,
    )


def run_runtime_inference_from_path(
        *,
        input_source: str,
        input_path: str | Path,
        artifacts: RuntimeArtifacts,
        config: RuntimeInferenceConfig | None = None,
        csv_config: RuntimePhoneCsvConfig | None = None,
        phone_folder_config: RuntimePhoneFolderConfig | None = None,
        session_id: str | None = None,
        max_sessions: int | None = 1,
) -> RuntimeInferenceResult:
    source_df = _load_source_dataframe_from_path(
        input_source=input_source,
        input_path=input_path,
        csv_config=csv_config,
        phone_folder_config=phone_folder_config,
    )
    source_df = _restrict_sessions(source_df, session_id=session_id, max_sessions=max_sessions)
    return run_runtime_inference_from_dataframe(
        source_df,
        artifacts=artifacts,
        config=config,
    )


def save_runtime_inference_result(
        result: RuntimeInferenceResult,
        *,
        output_dir: str | Path,
        save_full_fall_windows: bool = True,
) -> dict[str, str]:
    out_dir = _resolve_path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    har_csv = out_dir / "har_windows.csv"
    fall_csv = out_dir / "fall_windows.csv"
    vulnerability_csv = out_dir / "vulnerability_windows.csv"
    timeline_csv = out_dir / "combined_timeline.csv"
    grouped_events_csv = out_dir / "grouped_fall_events.csv"
    point_timeline_csv = out_dir / "point_timeline.csv"
    timeline_events_csv = out_dir / "timeline_events.csv"
    transition_events_csv = out_dir / "transition_events.csv"
    session_summaries_csv = out_dir / "session_summaries.csv"
    report_json = out_dir / "runtime_inference_report.json"

    har_keep = [
        c
        for c in [
            "window_id",
            "dataset_name",
            "subject_id",
            "session_id",
            "start_ts",
            "end_ts",
            "midpoint_ts",
            "label_mapped_majority",
            "predicted_label",
            "predicted_confidence",
        ]
        if c in result.har_windows.columns
    ]
    result.har_windows[har_keep].to_csv(har_csv, index=False)

    if save_full_fall_windows:
        result.fall_windows.to_csv(fall_csv, index=False)
    else:
        fall_keep = [
            c
            for c in [
                "window_id",
                "dataset_name",
                "subject_id",
                "session_id",
                "start_ts",
                "end_ts",
                "midpoint_ts",
                "true_label",
                "predicted_label",
                "predicted_probability",
                "predicted_is_fall",
                "probability_threshold_used",
            ]
            if c in result.fall_windows.columns
        ]
        result.fall_windows[fall_keep].to_csv(fall_csv, index=False)

    result.vulnerability_windows.to_csv(vulnerability_csv, index=False)

    result.combined_timeline.to_csv(timeline_csv, index=False)
    result.grouped_fall_events.to_csv(grouped_events_csv, index=False)
    result.point_timeline.to_csv(point_timeline_csv, index=False)
    result.timeline_events.to_csv(timeline_events_csv, index=False)
    result.transition_events.to_csv(transition_events_csv, index=False)
    result.session_summaries.to_csv(session_summaries_csv, index=False)
    report_json.write_text(json.dumps(_json_safe(result.report), indent=2), encoding="utf-8")

    return {
        "har_csv": str(har_csv),
        "fall_csv": str(fall_csv),
        "vulnerability_csv": str(vulnerability_csv),
        "timeline_csv": str(timeline_csv),
        "grouped_events_csv": str(grouped_events_csv),
        "point_timeline_csv": str(point_timeline_csv),
        "timeline_events_csv": str(timeline_events_csv),
        "transition_events_csv": str(transition_events_csv),
        "session_summaries_csv": str(session_summaries_csv),
        "report_json": str(report_json),
    }
