from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import json
import numpy as np
import pandas as pd


@dataclass(slots=True)
class RuntimeAnnotationConfig:
    session_glob: str = "*.json"
    min_samples: int = 32

    window_size_samples: int = 128
    step_size_samples: int = 64

    require_segments: bool = False
    fallback_to_session_labels_for_unsegmented_sessions: bool = False

    segment_keys: tuple[str, ...] = ("annotation_segments", "segments")

    dataset_name: str = "APP_RUNTIME_ANNOTATED"


@dataclass(slots=True)
class RuntimeAnnotationDataset:
    sessions_df: pd.DataFrame
    samples_df: pd.DataFrame
    segments_df: pd.DataFrame
    labelled_samples_df: pd.DataFrame
    window_labels_df: pd.DataFrame
    summary: dict[str, Any]


def _resolve_path(path_like: str | Path) -> Path:
    return Path(path_like).expanduser().resolve()


def _clean_text(value: Any, *, default: str = "") -> str:
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except Exception:
        pass
    text = str(value).strip()
    return text if text else default


def _clean_label(value: Any, *, default: str = "unknown") -> str:
    text = _clean_text(value, default=default).lower()
    return text if text else default


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return {str(k): v for k, v in value.items()}
    return {}


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


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Session file is not a JSON object: {path}")
    return {str(k): v for k, v in payload.items()}


def _extract_session_record(
        payload: dict[str, Any],
        *,
        session_path: Path,
        sample_count: int,
        has_segments: bool,
) -> dict[str, Any]:
    inference_result = _as_dict(payload.get("inference_result"))
    feedback = _as_list(payload.get("feedback"))

    return {
        "session_path": str(session_path),
        "file_name": session_path.name,
        "session_id": _clean_text(payload.get("session_id"), default=session_path.stem),
        "subject_id": _clean_text(payload.get("subject_id"), default="anonymous_user"),
        "declared_placement": _clean_label(payload.get("placement"), default="unknown"),
        "session_activity_label": _clean_label(payload.get("activity_label"), default="unknown"),
        "session_placement_label": _clean_label(payload.get("placement_label"), default="unknown"),
        "notes": _clean_text(payload.get("notes"), default=""),
        "saved_at": _clean_text(payload.get("saved_at"), default=""),
        "updated_at": _clean_text(payload.get("updated_at"), default=""),
        "dataset_name": _clean_text(payload.get("dataset_name"), default="APP_RUNTIME"),
        "source_type": _clean_text(payload.get("source_type"), default="mobile_app"),
        "device_platform": _clean_text(payload.get("device_platform"), default="unknown"),
        "device_model": _clean_text(payload.get("device_model"), default=""),
        "recording_mode": _clean_text(payload.get("recording_mode"), default="unknown"),
        "runtime_mode": _clean_text(payload.get("runtime_mode"), default="unknown"),
        "sampling_rate_hz": _safe_float(payload.get("sampling_rate_hz")),
        "sample_count": int(sample_count),
        "has_segments": bool(has_segments),
        "feedback_count": int(len(feedback)),
        "has_inference_result": bool(inference_result),
    }


def _extract_samples_df(
        payload: dict[str, Any],
        *,
        session_path: Path,
) -> pd.DataFrame:
    samples = _as_list(payload.get("samples"))
    session_id = _clean_text(payload.get("session_id"), default=session_path.stem)
    subject_id = _clean_text(payload.get("subject_id"), default="anonymous_user")
    declared_placement = _clean_label(payload.get("placement"), default="unknown")
    dataset_name = _clean_text(payload.get("dataset_name"), default="APP_RUNTIME")

    rows: list[dict[str, Any]] = []
    for idx, raw in enumerate(samples):
        item = _as_dict(raw)
        ts = _safe_float(item.get("timestamp"))
        if ts is None:
            continue

        rows.append(
            {
                "session_path": str(session_path),
                "file_name": session_path.name,
                "session_id": session_id,
                "subject_id": subject_id,
                "dataset_name": dataset_name,
                "declared_placement": declared_placement,
                "sample_idx": int(idx),
                "timestamp": float(ts),
                "ax": _safe_float(item.get("ax")),
                "ay": _safe_float(item.get("ay")),
                "az": _safe_float(item.get("az")),
                "gx": _safe_float(item.get("gx")),
                "gy": _safe_float(item.get("gy")),
                "gz": _safe_float(item.get("gz")),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out = out.sort_values(["session_id", "timestamp", "sample_idx"], kind="stable").reset_index(drop=True)
    return out


def _extract_segments_df(
        payload: dict[str, Any],
        *,
        session_path: Path,
        config: RuntimeAnnotationConfig,
) -> pd.DataFrame:
    raw_segments: list[Any] = []
    for key in config.segment_keys:
        candidate = payload.get(key)
        if isinstance(candidate, list):
            raw_segments = candidate
            break

    if not raw_segments:
        return pd.DataFrame()

    session_id = _clean_text(payload.get("session_id"), default=session_path.stem)
    subject_id = _clean_text(payload.get("subject_id"), default="anonymous_user")
    dataset_name = _clean_text(payload.get("dataset_name"), default="APP_RUNTIME")

    rows: list[dict[str, Any]] = []
    for idx, raw in enumerate(raw_segments):
        item = _as_dict(raw)

        start_ts = _safe_float(item.get("start_ts"))
        end_ts = _safe_float(item.get("end_ts"))
        if start_ts is None or end_ts is None or end_ts < start_ts:
            continue

        rows.append(
            {
                "session_path": str(session_path),
                "file_name": session_path.name,
                "session_id": session_id,
                "subject_id": subject_id,
                "dataset_name": dataset_name,
                "segment_id": _clean_text(item.get("segment_id"), default=f"{session_id}_segment_{idx:03d}"),
                "start_ts": float(start_ts),
                "end_ts": float(end_ts),
                "duration_seconds": float(max(0.0, end_ts - start_ts)),
                "activity_label": _clean_label(item.get("activity_label"), default="unknown"),
                "placement_label": _clean_label(item.get("placement_label"), default="unknown"),
                "event_label": _clean_label(item.get("event_label"), default="none"),
                "notes": _clean_text(item.get("notes"), default=""),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    return out.sort_values(["session_id", "start_ts", "end_ts"], kind="stable").reset_index(drop=True)


def _label_samples_from_segments(
        samples_df: pd.DataFrame,
        segments_df: pd.DataFrame,
        sessions_df: pd.DataFrame,
        *,
        config: RuntimeAnnotationConfig,
) -> pd.DataFrame:
    if samples_df.empty:
        return pd.DataFrame()

    labelled_parts: list[pd.DataFrame] = []

    session_meta = sessions_df.set_index("session_id", drop=False) if not sessions_df.empty else pd.DataFrame()

    for session_id, sample_group in samples_df.groupby("session_id", dropna=False, sort=False):
        sample_group = sample_group.sort_values("timestamp", kind="stable").reset_index(drop=True).copy()

        if not segments_df.empty:
            seg_group = segments_df[segments_df["session_id"].astype(str) == str(session_id)].copy()
            seg_group = seg_group.sort_values("start_ts", kind="stable").reset_index(drop=True)
        else:
            seg_group = pd.DataFrame()

        sample_group["activity_label"] = None
        sample_group["placement_label"] = None
        sample_group["event_label"] = None
        sample_group["annotation_source"] = "unlabelled"
        sample_group["segment_id"] = None

        if not seg_group.empty:
            for _, seg in seg_group.iterrows():
                mask = (
                               pd.to_numeric(sample_group["timestamp"], errors="coerce") >= float(seg["start_ts"])
                       ) & (
                               pd.to_numeric(sample_group["timestamp"], errors="coerce") <= float(seg["end_ts"])
                       )
                if not mask.any():
                    continue

                sample_group.loc[mask, "activity_label"] = str(seg["activity_label"])
                sample_group.loc[mask, "placement_label"] = str(seg["placement_label"])
                sample_group.loc[mask, "event_label"] = str(seg["event_label"])
                sample_group.loc[mask, "annotation_source"] = "segment"
                sample_group.loc[mask, "segment_id"] = str(seg["segment_id"])

        has_segment_labels = sample_group["annotation_source"].eq("segment").any()

        if (
                not has_segment_labels
                and config.fallback_to_session_labels_for_unsegmented_sessions
                and session_id in session_meta.index
        ):
            session_row = session_meta.loc[session_id]
            session_activity = _clean_label(session_row.get("session_activity_label"), default="unknown")
            session_placement = _clean_label(session_row.get("session_placement_label"), default="unknown")

            sample_group["activity_label"] = session_activity
            sample_group["placement_label"] = session_placement
            sample_group["event_label"] = "none"
            sample_group["annotation_source"] = "session_fallback"
            sample_group["segment_id"] = None

        if config.require_segments and not sample_group["annotation_source"].eq("segment").any():
            continue

        labelled_parts.append(sample_group)

    if not labelled_parts:
        return pd.DataFrame()

    out = pd.concat(labelled_parts, ignore_index=True, sort=False)
    return out.sort_values(["session_id", "timestamp", "sample_idx"], kind="stable").reset_index(drop=True)


def _majority_label(series: pd.Series, *, default: str = "unknown") -> str:
    clean = series.dropna().astype(str).str.strip()
    clean = clean[clean.ne("")]
    if clean.empty:
        return default
    counts = clean.value_counts(dropna=False)
    return str(counts.index[0])


def _build_window_labels_df(
        labelled_samples_df: pd.DataFrame,
        *,
        config: RuntimeAnnotationConfig,
) -> pd.DataFrame:
    if labelled_samples_df.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []

    for session_id, group in labelled_samples_df.groupby("session_id", dropna=False, sort=False):
        group = group.sort_values("timestamp", kind="stable").reset_index(drop=True).copy()
        if len(group) < max(config.min_samples, config.window_size_samples):
            continue

        for start in range(0, len(group) - config.window_size_samples + 1, config.step_size_samples):
            window = group.iloc[start : start + config.window_size_samples].copy()
            if len(window) < config.window_size_samples:
                continue

            activity_majority = _majority_label(window["activity_label"], default="unknown")
            placement_majority = _majority_label(window["placement_label"], default="unknown")
            event_majority = _majority_label(window["event_label"], default="none")
            annotation_source_majority = _majority_label(window["annotation_source"], default="unlabelled")

            labelled_fraction = float(
                window["annotation_source"].astype(str).ne("unlabelled").mean()
            )
            segment_fraction = float(
                window["annotation_source"].astype(str).eq("segment").mean()
            )
            fall_fraction = float(
                window["event_label"].astype(str).str.lower().eq("fall").mean()
            )

            rows.append(
                {
                    "session_id": str(session_id),
                    "subject_id": str(window["subject_id"].iloc[0]),
                    "dataset_name": str(window["dataset_name"].iloc[0]),
                    "window_id": f"{session_id}_window_{start:06d}",
                    "start_sample_idx": int(window["sample_idx"].iloc[0]),
                    "end_sample_idx": int(window["sample_idx"].iloc[-1]),
                    "start_ts": float(window["timestamp"].iloc[0]),
                    "end_ts": float(window["timestamp"].iloc[-1]),
                    "midpoint_ts": float(window["timestamp"].mean()),
                    "window_size_samples": int(len(window)),
                    "activity_label_majority": activity_majority,
                    "placement_label_majority": placement_majority,
                    "event_label_majority": event_majority,
                    "annotation_source_majority": annotation_source_majority,
                    "labelled_fraction": labelled_fraction,
                    "segment_fraction": segment_fraction,
                    "fall_fraction": fall_fraction,
                    "contains_fall": bool(fall_fraction > 0.0),
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    return out.sort_values(["session_id", "start_ts"], kind="stable").reset_index(drop=True)


def load_runtime_annotation_dataset(
        sessions_dir: str | Path,
        *,
        config: RuntimeAnnotationConfig | None = None,
) -> RuntimeAnnotationDataset:
    cfg = config or RuntimeAnnotationConfig()
    sessions_dir = _resolve_path(sessions_dir)

    if not sessions_dir.exists():
        raise FileNotFoundError(f"Sessions directory not found: {sessions_dir}")
    if not sessions_dir.is_dir():
        raise NotADirectoryError(f"Expected a directory: {sessions_dir}")

    session_files = sorted(sessions_dir.glob(cfg.session_glob))
    if not session_files:
        raise FileNotFoundError(f"No session files matched {cfg.session_glob} in {sessions_dir}")

    session_rows: list[dict[str, Any]] = []
    sample_parts: list[pd.DataFrame] = []
    segment_parts: list[pd.DataFrame] = []

    for session_path in session_files:
        payload = _load_json(session_path)
        samples_df = _extract_samples_df(payload, session_path=session_path)
        segments_df = _extract_segments_df(payload, session_path=session_path, config=cfg)

        if len(samples_df) < cfg.min_samples:
            continue

        session_rows.append(
            _extract_session_record(
                payload,
                session_path=session_path,
                sample_count=len(samples_df),
                has_segments=not segments_df.empty,
            )
        )
        sample_parts.append(samples_df)
        if not segments_df.empty:
            segment_parts.append(segments_df)

    sessions_df = pd.DataFrame(session_rows)
    samples_df = (
        pd.concat(sample_parts, ignore_index=True, sort=False)
        if sample_parts
        else pd.DataFrame()
    )
    segments_df = (
        pd.concat(segment_parts, ignore_index=True, sort=False)
        if segment_parts
        else pd.DataFrame()
    )

    labelled_samples_df = _label_samples_from_segments(
        samples_df,
        segments_df,
        sessions_df,
        config=cfg,
    )
    window_labels_df = _build_window_labels_df(
        labelled_samples_df,
        config=cfg,
    )

    summary = {
        "config": asdict(cfg),
        "sessions_dir": str(sessions_dir),
        "n_sessions_total": int(len(sessions_df)),
        "n_sessions_with_segments": int(
            sessions_df["has_segments"].sum() if not sessions_df.empty and "has_segments" in sessions_df.columns else 0
        ),
        "n_sessions_with_labelled_samples": int(
            labelled_samples_df["session_id"].nunique() if not labelled_samples_df.empty else 0
        ),
        "n_sessions_with_window_labels": int(
            window_labels_df["session_id"].nunique() if not window_labels_df.empty else 0
        ),
        "total_samples": int(len(samples_df)),
        "total_labelled_samples": int(len(labelled_samples_df)),
        "total_segments": int(len(segments_df)),
        "total_windows": int(len(window_labels_df)),
        "activity_label_counts": (
            labelled_samples_df["activity_label"].astype(str).value_counts(dropna=False).to_dict()
            if not labelled_samples_df.empty and "activity_label" in labelled_samples_df.columns
            else {}
        ),
        "placement_label_counts": (
            labelled_samples_df["placement_label"].astype(str).value_counts(dropna=False).to_dict()
            if not labelled_samples_df.empty and "placement_label" in labelled_samples_df.columns
            else {}
        ),
        "event_label_counts": (
            labelled_samples_df["event_label"].astype(str).value_counts(dropna=False).to_dict()
            if not labelled_samples_df.empty and "event_label" in labelled_samples_df.columns
            else {}
        ),
    }

    return RuntimeAnnotationDataset(
        sessions_df=sessions_df,
        samples_df=samples_df,
        segments_df=segments_df,
        labelled_samples_df=labelled_samples_df,
        window_labels_df=window_labels_df,
        summary=summary,
    )


def save_runtime_annotation_dataset(
        dataset: RuntimeAnnotationDataset,
        *,
        output_dir: str | Path,
) -> dict[str, str]:
    out_dir = _resolve_path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sessions_csv = out_dir / "annotated_sessions.csv"
    samples_csv = out_dir / "annotated_samples.csv"
    segments_csv = out_dir / "annotation_segments.csv"
    labelled_samples_csv = out_dir / "labelled_samples.csv"
    window_labels_csv = out_dir / "window_labels.csv"
    summary_json = out_dir / "annotation_dataset_summary.json"

    dataset.sessions_df.to_csv(sessions_csv, index=False)
    dataset.samples_df.to_csv(samples_csv, index=False)
    dataset.segments_df.to_csv(segments_csv, index=False)
    dataset.labelled_samples_df.to_csv(labelled_samples_csv, index=False)
    dataset.window_labels_df.to_csv(window_labels_csv, index=False)
    summary_json.write_text(
        json.dumps(_json_safe(dataset.summary), indent=2),
        encoding="utf-8",
    )

    return {
        "sessions_csv": str(sessions_csv),
        "samples_csv": str(samples_csv),
        "segments_csv": str(segments_csv),
        "labelled_samples_csv": str(labelled_samples_csv),
        "window_labels_csv": str(window_labels_csv),
        "summary_json": str(summary_json),
    }