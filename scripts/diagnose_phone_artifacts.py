#!/usr/bin/env python3
"""Stratified diagnostic report for the current fall + HAR artifacts on phone data.

Reads the replay output CSVs (``har.csv`` / ``fall.csv``) and the
``sessions_manifest.json`` written by ``retrain_from_phone.py``'s ``convert``
stage, joins each window back to its real session (and thus placement +
activity_label), and emits:

- ``fall_diagnostic.json``  — threshold sweep + grouped-event FP-per-hour,
                              both overall and stratified by placement
                              and by underlying activity_label.
- ``har_diagnostic.json``   — per-class + per-placement confusion matrix
                              for the HAR labels observed on the phone.
- ``diagnostic_report.json`` — aggregate machine-readable summary that the
                              promotion gates read.
- ``diagnostic_report.md``  — human-readable top-N failure modes.

The script is invoked by the orchestrator's ``diagnose`` stage but also works
standalone when you want to re-evaluate without re-running replay.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from metrics.classification import compute_classification_metrics  # noqa: E402
from metrics.fall_metrics import compute_fall_detection_metrics  # noqa: E402
from services.runtime_inference import _group_runtime_fall_events  # noqa: E402

# Threshold sweep honoured by the fall diagnostic.
DEFAULT_THRESHOLDS = [round(0.30 + 0.025 * i, 3) for i in range(21)]  # 0.300 .. 0.800

# Grouped-event config: match the live RuntimeInferenceConfig defaults so the
# FP/hour number the report shows is what users would actually see at alert
# time. Keep this in sync with services/runtime_inference.py::RuntimeInferenceConfig.
GROUPED_PROBABILITY_THRESHOLD = 0.5
GROUPED_MERGE_GAP_SECONDS = 0.25
GROUPED_MIN_WINDOWS = 2
GROUPED_MAX_DURATION_SECONDS = 4.0
GROUPED_ALERT_PEAK_THRESHOLD = 0.75


# ------------------------------------------------------------------- joining --


def _load_manifest(path: Path) -> pd.DataFrame:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not raw:
        return pd.DataFrame(
            columns=[
                "session_id",
                "subject_id",
                "activity_label",
                "placement",
                "merged_start_ts",
                "merged_end_ts",
                "duration_seconds",
            ]
        )
    df = pd.DataFrame(raw)
    for col in ("merged_start_ts", "merged_end_ts", "duration_seconds"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["activity_label"] = df["activity_label"].fillna("_unlabeled").astype(str).str.lower()
    df["placement"] = df.get("placement", pd.Series(dtype="object")).fillna("unknown").astype(str).str.lower()
    return df


def _attach_session_context(df: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    """Stamp each window with the originating session's placement + activity."""
    if df.empty:
        out = df.copy()
        out["real_session_id"] = pd.Series(dtype=str)
        out["placement"] = pd.Series(dtype=str)
        out["activity_label"] = pd.Series(dtype=str)
        return out

    mid = pd.to_numeric(df.get("midpoint_ts"), errors="coerce")
    starts = manifest["merged_start_ts"].to_numpy()
    ends = manifest["merged_end_ts"].to_numpy()
    labels = manifest["activity_label"].to_numpy()
    placements = manifest["placement"].to_numpy()
    session_ids = manifest["session_id"].astype(str).to_numpy()

    real_session = []
    activity = []
    placement = []
    for ts in mid.fillna(-1.0).to_numpy():
        # Sessions are non-overlapping on the merged timeline by construction.
        matches = (ts >= starts) & (ts <= ends)
        if not matches.any():
            real_session.append(None)
            activity.append("_unlabeled")
            placement.append("unknown")
            continue
        idx = int(matches.argmax())
        real_session.append(session_ids[idx])
        activity.append(labels[idx])
        placement.append(placements[idx])

    out = df.copy()
    out["real_session_id"] = real_session
    out["activity_label"] = activity
    out["placement"] = placement
    return out


# ---------------------------------------------------------------- fall diag --


def _fall_metrics_at_threshold(
    df: pd.DataFrame,
    *,
    threshold: float,
) -> dict[str, Any]:
    probs = pd.to_numeric(df.get("predicted_probability"), errors="coerce").fillna(0.0)
    y_pred = ["fall" if p >= threshold else "non_fall" for p in probs]
    y_true = df["fall_true_label"].astype(str).str.lower().tolist()
    # Normalise truth labels to the binary axis the metric helper expects.
    y_true = ["fall" if v == "fall" else "non_fall" for v in y_true]
    base = compute_fall_detection_metrics(y_true, y_pred)
    base["threshold"] = float(threshold)
    return base


def _threshold_sweep(df: pd.DataFrame) -> list[dict[str, Any]]:
    return [_fall_metrics_at_threshold(df, threshold=t) for t in DEFAULT_THRESHOLDS]


def _stratified_fall_at_threshold(
    df: pd.DataFrame,
    *,
    threshold: float,
) -> dict[str, Any]:
    strat: dict[str, dict[str, Any]] = {"by_placement": {}, "by_activity_label": {}, "by_pair": {}}
    for placement, sub in df.groupby("placement", dropna=False):
        strat["by_placement"][str(placement)] = _fall_metrics_at_threshold(sub, threshold=threshold)
    for label, sub in df.groupby("activity_label", dropna=False):
        strat["by_activity_label"][str(label)] = _fall_metrics_at_threshold(sub, threshold=threshold)
    for (placement, label), sub in df.groupby(["placement", "activity_label"], dropna=False):
        key = f"{placement}|{label}"
        strat["by_pair"][key] = _fall_metrics_at_threshold(sub, threshold=threshold)
    return strat


def _grouped_event_fpr(df: pd.DataFrame, *, total_seconds: float) -> dict[str, Any]:
    """Compute false-alert rate per hour at the live grouping config."""
    if df.empty or total_seconds <= 0:
        return {"alerts": 0, "false_alerts": 0, "hours": 0.0, "alerts_per_hour": 0.0, "false_alerts_per_hour": 0.0}

    working = df.copy()
    # The grouping helper expects predicted_probability + start/end/midpoint_ts
    # columns; replay output has all of these under known names.
    if "start_ts" not in working.columns or "end_ts" not in working.columns:
        half_window = 1.28  # default 2.56s window ⇒ ±1.28s from midpoint.
        working["start_ts"] = pd.to_numeric(working["midpoint_ts"], errors="coerce") - half_window
        working["end_ts"] = pd.to_numeric(working["midpoint_ts"], errors="coerce") + half_window

    events = _group_runtime_fall_events(
        working,
        probability_threshold=GROUPED_PROBABILITY_THRESHOLD,
        merge_gap_seconds=GROUPED_MERGE_GAP_SECONDS,
        min_windows=GROUPED_MIN_WINDOWS,
        max_event_duration_seconds=GROUPED_MAX_DURATION_SECONDS,
    )

    alerts = 0
    false_alerts = 0
    if not events.empty:
        peak = pd.to_numeric(events["peak_probability"], errors="coerce").fillna(0.0)
        alert_mask = peak >= GROUPED_ALERT_PEAK_THRESHOLD
        alerts = int(alert_mask.sum())

        # A grouped event is a "false alert" when no fall window inside it is
        # truly labelled fall. We use the session's activity_label since that
        # is the ground truth the user provided at record time.
        if "activity_label" in working.columns and alerts:
            for _, ev in events[alert_mask].iterrows():
                overlap = working[
                    (working["session_id"] == ev["session_id"])
                    & (working["start_ts"] <= ev["event_end_ts"])
                    & (working["end_ts"] >= ev["event_start_ts"])
                ]
                if overlap.empty:
                    false_alerts += 1
                    continue
                real_labels = overlap["activity_label"].astype(str).str.lower().unique().tolist()
                if "fall" not in real_labels:
                    false_alerts += 1
        else:
            # No ground truth → can't distinguish TP vs FP; conservatively
            # report all alerts as false to avoid hiding the rate.
            false_alerts = alerts

    hours = total_seconds / 3600.0 if total_seconds else 0.0
    return {
        "alerts": alerts,
        "false_alerts": false_alerts,
        "hours": round(hours, 3),
        "alerts_per_hour": round(alerts / hours, 3) if hours else 0.0,
        "false_alerts_per_hour": round(false_alerts / hours, 3) if hours else 0.0,
    }


def _fall_diagnostic(
    fall_df: pd.DataFrame,
    *,
    current_threshold: float,
    total_seconds: float,
) -> dict[str, Any]:
    if fall_df.empty:
        return {"empty": True}

    # The replay output either uses `fall_true_label` (from the timeline subset)
    # or `true_label` (from the raw prediction CSV). Normalise.
    if "fall_true_label" not in fall_df.columns and "true_label" in fall_df.columns:
        fall_df = fall_df.rename(columns={"true_label": "fall_true_label"})
    if "fall_true_label" not in fall_df.columns:
        # Fall back to the session-level activity_label.
        fall_df = fall_df.copy()
        fall_df["fall_true_label"] = fall_df["activity_label"].apply(
            lambda v: "fall" if str(v).lower() == "fall" else "non_fall"
        )

    return {
        "windows": int(len(fall_df)),
        "positive_windows": int((fall_df["fall_true_label"].astype(str).str.lower() == "fall").sum()),
        "current_threshold": float(current_threshold),
        "at_current_threshold": {
            "overall": _fall_metrics_at_threshold(fall_df, threshold=current_threshold),
            "stratified": _stratified_fall_at_threshold(fall_df, threshold=current_threshold),
        },
        "threshold_sweep": _threshold_sweep(fall_df),
        "grouped_events": _grouped_event_fpr(fall_df, total_seconds=total_seconds),
    }


# ----------------------------------------------------------------- har diag --


def _har_metrics(y_true: Iterable[str], y_pred: Iterable[str]) -> dict[str, Any]:
    yt = [str(v).lower() for v in y_true]
    yp = [str(v).lower() for v in y_pred]
    if not yt:
        return {"empty": True}
    return compute_classification_metrics(yt, yp)


def _har_diagnostic(har_df: pd.DataFrame) -> dict[str, Any]:
    if har_df.empty:
        return {"empty": True}

    pred_col = "har_predicted_label" if "har_predicted_label" in har_df.columns else "predicted_label"
    true_col = (
        "har_true_label"
        if "har_true_label" in har_df.columns
        else "label_mapped_majority" if "label_mapped_majority" in har_df.columns
        else None
    )
    # The raw HAR replay CSV may not carry the session-label ground truth, so
    # fall back to the manifest-derived activity_label that we joined earlier.
    gt = har_df[true_col] if true_col else har_df["activity_label"]
    gt = gt.fillna("_unlabeled").astype(str).str.lower()

    out: dict[str, Any] = {
        "windows": int(len(har_df)),
        "overall": _har_metrics(gt, har_df[pred_col]),
        "by_placement": {},
    }

    for placement, sub in har_df.groupby("placement", dropna=False):
        sub_gt = sub[true_col] if true_col else sub["activity_label"]
        sub_gt = sub_gt.fillna("_unlabeled").astype(str).str.lower()
        out["by_placement"][str(placement)] = _har_metrics(sub_gt, sub[pred_col])

    return out


# ------------------------------------------------------------ failure modes --


def _top_failure_modes(
    fall_diag: dict[str, Any],
    har_diag: dict[str, Any],
    *,
    limit: int = 3,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []

    # Fall: rank (placement, activity_label) pairs by FP count at the current threshold.
    stratified = (fall_diag.get("at_current_threshold") or {}).get("stratified") or {}
    for key, metrics in (stratified.get("by_pair") or {}).items():
        fp = int(metrics.get("fp") or 0)
        support = int(metrics.get("support_total") or 0)
        if fp == 0 or support < 4:
            continue
        placement, activity = key.split("|", 1)
        candidates.append(
            {
                "kind": "fall_fp",
                "placement": placement,
                "activity_label": activity,
                "fp": fp,
                "support": support,
                "fpr": round(fp / support, 4) if support else 0.0,
                "detail": f"fall threshold={metrics.get('threshold')} produced {fp} FPs on {support} windows",
                "suggestion": (
                    f"record ≥10 more sessions where activity={activity} "
                    f"and placement={placement} to teach the negative class"
                ),
            }
        )

    # HAR: rank per-class cells with low recall.
    per_class = (har_diag.get("overall") or {}).get("per_class") or {}
    for label, metrics in per_class.items():
        recall = float(metrics.get("recall") or 0.0)
        support = int(metrics.get("support") or 0)
        if support < 10 or recall >= 0.7:
            continue
        candidates.append(
            {
                "kind": "har_recall",
                "activity_label": label,
                "recall": round(recall, 3),
                "support": support,
                "detail": f"HAR recall={recall:.3f} on {support} windows labelled {label}",
                "suggestion": (
                    f"record ≥5 more sessions labelled {label} across pocket + hand "
                    f"placements — the current HAR model confuses it"
                ),
            }
        )

    candidates.sort(
        key=lambda c: (c.get("fp", 0), -c.get("recall", 1.0)),
        reverse=True,
    )
    return candidates[:limit]


# ------------------------------------------------------------------- report --


def _render_markdown(
    *,
    fall_diag: dict[str, Any],
    har_diag: dict[str, Any],
    failure_modes: list[dict[str, Any]],
    current_threshold: float,
) -> str:
    lines: list[str] = ["# Phone diagnostic report", ""]
    if not failure_modes:
        lines.append("No top failure modes identified — model looks healthy against the phone set.")
    else:
        lines.append("## Top failure modes")
        for i, mode in enumerate(failure_modes, start=1):
            lines.append(f"{i}. **{mode['kind']}** — {mode['detail']}")
            lines.append(f"   - Suggestion: {mode['suggestion']}")
        lines.append("")

    lines.append("## Fall")
    if fall_diag.get("empty"):
        lines.append("_No fall windows found._")
    else:
        at_cur = fall_diag["at_current_threshold"]["overall"]
        lines.append(
            f"- Windows: {fall_diag['windows']} "
            f"(positive: {fall_diag['positive_windows']})"
        )
        lines.append(
            f"- At current threshold ({current_threshold}): "
            f"TP={at_cur['tp']} FP={at_cur['fp']} "
            f"FN={at_cur['fn']} TN={at_cur['tn']} "
            f"F1={at_cur['f1']:.3f}"
        )
        ev = fall_diag.get("grouped_events") or {}
        lines.append(
            f"- Grouped events over {ev.get('hours', 0)}h: "
            f"alerts={ev.get('alerts', 0)}, "
            f"false_alerts={ev.get('false_alerts', 0)} "
            f"({ev.get('false_alerts_per_hour', 0.0)}/hr)"
        )
    lines.append("")

    lines.append("## HAR")
    if har_diag.get("empty"):
        lines.append("_No HAR windows found._")
    else:
        overall = har_diag["overall"]
        lines.append(f"- Windows: {har_diag['windows']}")
        lines.append(
            f"- Overall accuracy={overall.get('accuracy'):.3f} "
            f"macro_f1={overall.get('macro_f1'):.3f}"
        )
        lines.append("- Per-class recall:")
        for label, metrics in (overall.get("per_class") or {}).items():
            lines.append(
                f"  - {label}: recall={metrics.get('recall', 0.0):.3f} "
                f"support={metrics.get('support', 0)}"
            )
    lines.append("")

    return "\n".join(lines) + "\n"


# -------------------------------------------------------------------- main --


def _read_csv_optional(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def run_diagnosis(
    *,
    fall_csv: Path,
    har_csv: Path,
    sessions_manifest: Path,
    current_threshold: float,
    out_dir: Path,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = _load_manifest(sessions_manifest)
    total_seconds = float(manifest["duration_seconds"].fillna(0.0).sum()) if not manifest.empty else 0.0

    fall_raw = _read_csv_optional(fall_csv)
    har_raw = _read_csv_optional(har_csv)

    fall_df = _attach_session_context(fall_raw, manifest) if not fall_raw.empty else fall_raw
    har_df = _attach_session_context(har_raw, manifest) if not har_raw.empty else har_raw

    fall_diag = _fall_diagnostic(fall_df, current_threshold=current_threshold, total_seconds=total_seconds)
    har_diag = _har_diagnostic(har_df)
    failure_modes = _top_failure_modes(fall_diag, har_diag, limit=3)

    (out_dir / "fall_diagnostic.json").write_text(json.dumps(fall_diag, indent=2, default=str), encoding="utf-8")
    (out_dir / "har_diagnostic.json").write_text(json.dumps(har_diag, indent=2, default=str), encoding="utf-8")

    aggregate = {
        "windows": {
            "fall": int(fall_diag.get("windows", 0)),
            "har": int(har_diag.get("windows", 0)),
        },
        "total_seconds": round(total_seconds, 3),
        "current_threshold": float(current_threshold),
        "top_failure_modes": failure_modes,
        "fall_grouped_events": fall_diag.get("grouped_events"),
        "fall_at_current_threshold_overall": (fall_diag.get("at_current_threshold") or {}).get("overall"),
        "har_overall": har_diag.get("overall"),
    }
    (out_dir / "diagnostic_report.json").write_text(
        json.dumps(aggregate, indent=2, default=str), encoding="utf-8"
    )
    (out_dir / "diagnostic_report.md").write_text(
        _render_markdown(
            fall_diag=fall_diag,
            har_diag=har_diag,
            failure_modes=failure_modes,
            current_threshold=current_threshold,
        ),
        encoding="utf-8",
    )
    return aggregate


def _resolve_current_threshold(cli_value: float | None) -> float:
    if cli_value is not None:
        return float(cli_value)
    try:
        from pipeline.artifacts import load_current_metadata
        meta = load_current_metadata("fall")
    except Exception:  # noqa: BLE001
        return 0.5
    probe = meta.get("threshold")
    if probe is None:
        probe = (meta.get("operating_point") or {}).get("threshold")
    try:
        return float(probe) if probe is not None else 0.5
    except (TypeError, ValueError):
        return 0.5


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fall-csv", type=Path, required=True, help="Replay output fall.csv")
    p.add_argument("--har-csv", type=Path, required=True, help="Replay output har.csv")
    p.add_argument(
        "--sessions-manifest",
        type=Path,
        required=True,
        help="sessions_manifest.json written by the convert stage",
    )
    p.add_argument(
        "--current-threshold",
        type=float,
        default=None,
        help="Override the fall threshold. Default reads from artifacts/fall/current/metadata.json",
    )
    p.add_argument("--out-dir", type=Path, required=True)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    current_threshold = _resolve_current_threshold(args.current_threshold)
    summary = run_diagnosis(
        fall_csv=args.fall_csv,
        har_csv=args.har_csv,
        sessions_manifest=args.sessions_manifest,
        current_threshold=current_threshold,
        out_dir=args.out_dir,
    )
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
