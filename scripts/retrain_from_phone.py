#!/usr/bin/env python3
"""Pull labelled phone recordings from the API and prepare / train local models.

Stages (run in order unless ``--stages`` is passed):

1. ``pull``        — list sessions via ``GET /v1/sessions`` and download each
                     raw payload via ``GET /v1/sessions/{id}/raw`` into
                     ``artifacts/runtime_sessions_pulled/<subject>/``. Uses the
                     SHA256 returned by the list response as a resume key.
2. ``convert``     — transform each session JSON into the canonical phone
                     sensor-export layout (``Accelerometer.csv`` +
                     ``Gyroscope.csv`` + ``Metadata.csv``). Per-session folders
                     live under ``sessions/<session_id>/``; a single merged
                     folder containing the concatenated stream lives under
                     ``merged/`` — that is what the existing HAR adaptation
                     and combined replay scripts ingest.
3. ``annotate``    — synthesize a builder-compatible annotation CSV
                     (``annotations.csv``) from the session-level
                     ``activity_label`` (the whole session becomes one labelled
                     interval). This matches the format expected by
                     ``build_phone_runtime_training_set.py`` and
                     ``run_phone_har_adaptation.py``.
4. ``replay``      — run ``run_combined_runtime_replay.py`` against the merged
                     folder using the current fall + HAR artifacts; emits
                     ``har.csv`` and ``fall.csv``.
5. ``build``       — run ``build_phone_runtime_training_set.py`` to attach the
                     annotation intervals to each window; emits
                     ``har_labeled_windows.csv`` and
                     ``fall_labeled_windows.csv``.
6. ``train-fall``  — (opt-in) run ``train_fall_with_phone_hard_negatives.py``
                     against the fall labelled windows.
7. ``train-har``   — (opt-in) run ``run_phone_har_adaptation.py`` against the
                     merged folder + annotation CSV.
8. ``report``      — write a ``report.json`` summarising pulled sessions,
                     exported windows, per-label counts, scarcity warnings, and
                     (when the train stages ran) baseline-vs-adapted metrics.

The default stage list is ``pull,convert,annotate,replay,build,report`` — the
train stages are opt-in because they are long-running and typically want
operator review of the report first.

Authentication:
    Pass ``--auth-env VAR`` where ``$VAR`` holds ``user:password`` in HTTP
    Basic form. The value is forwarded to every request against the API.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pipeline.artifacts import (  # noqa: E402
    load_current_metadata,
    resolve_current_artifact,
)

SCRIPT_DIR = _REPO_ROOT / "scripts"
DEFAULT_PULL_ROOT = _REPO_ROOT / "artifacts" / "runtime_sessions_pulled"
DEFAULT_WORK_ROOT = _REPO_ROOT / "artifacts" / "phone_training"
DEFAULT_REPORT_ROOT = _REPO_ROOT / "results" / "validation" / "phone_retrain"

DEFAULT_STAGES = ("pull", "convert", "annotate", "replay", "build", "diagnose", "report")
OPTIONAL_STAGES = ("train-fall", "train-har")
ALL_STAGES = DEFAULT_STAGES + OPTIONAL_STAGES

# Labels with fewer than these thresholds surface as warnings in the report.
MIN_SESSIONS_PER_LABEL = 5
MIN_SECONDS_PER_LABEL = 180.0

# --------------------------------------------------------------------- http ---


class HttpClient:
    def __init__(self, base_url: str, auth_token: str | None, *, timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.auth_token = auth_token
        self.timeout = timeout

    def _request(self, path: str, *, query: dict[str, Any] | None = None) -> urllib.request.Request:
        url = f"{self.base_url}{path}"
        if query:
            params = {k: v for k, v in query.items() if v is not None}
            if params:
                url = f"{url}?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(url)
        if self.auth_token:
            req.add_header("Authorization", f"Basic {self.auth_token}")
        req.add_header("Accept", "application/json")
        return req

    def get_json(self, path: str, *, query: dict[str, Any] | None = None) -> Any:
        req = self._request(path, query=query)
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            payload = resp.read()
        return json.loads(payload.decode("utf-8"))

    def download(self, path: str, dest: Path) -> int:
        req = self._request(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = resp.read()
        tmp.write_bytes(data)
        tmp.replace(dest)
        return len(data)


def _load_auth_token(auth_env: str | None) -> str | None:
    if not auth_env:
        return None
    raw = os.environ.get(auth_env)
    if not raw:
        raise SystemExit(f"Environment variable {auth_env!r} is empty or unset.")
    if ":" not in raw:
        raise SystemExit(f"Value in ${auth_env} must be in 'user:password' form.")
    return base64.b64encode(raw.encode("utf-8")).decode("ascii")


# --------------------------------------------------------------------- pull ---


def _iter_sessions(
    client: HttpClient,
    *,
    subject_id: str | None,
    labels: list[str] | None,
    since: str | None,
    page_size: int = 200,
) -> Iterable[dict[str, Any]]:
    offset = 0
    seen = 0
    while True:
        query = {
            "limit": page_size,
            "offset": offset,
            "subject_id": subject_id,
            "recorded_since": since,
        }
        page = client.get_json("/v1/sessions", query=query)
        sessions = page.get("sessions") or []
        if not sessions:
            return
        for item in sessions:
            sess = _normalise_session_list_item(item)
            if labels and str(sess.get("activity_label") or "").lower() not in labels:
                continue
            yield sess
            seen += 1
        if len(sessions) < page_size:
            return
        offset += len(sessions)


def _normalise_session_list_item(item: dict[str, Any]) -> dict[str, Any]:
    """Accept both flat legacy records and production list records.

    The production API returns list items as ``{"session": {...}, "latest_*": ...}``
    while older pulls returned the session fields at the top level. Downstream
    stages need the flat session metadata but keeping the latest inference
    summary is useful in the report/debug state.
    """

    if isinstance(item.get("session"), dict):
        out = dict(item["session"])
        for key, value in item.items():
            if key != "session" and key not in out:
                out[key] = value
        return out
    return dict(item)


def _pull_sessions(
    client: HttpClient,
    *,
    subject_id: str | None,
    labels: list[str] | None,
    since: str | None,
    pull_root: Path,
    limit: int | None,
) -> list[dict[str, Any]]:
    pulled: list[dict[str, Any]] = []
    for sess in _iter_sessions(
        client,
        subject_id=subject_id,
        labels=labels,
        since=since,
    ):
        if limit is not None and len(pulled) >= limit:
            break
        sess_id = str(sess["app_session_id"])
        subject = str(sess.get("subject_id") or "unknown_subject")
        dest = pull_root / subject / f"{sess_id}.json"
        expected_sha = (sess.get("raw_payload_sha256") or "").strip().lower() or None

        if dest.exists() and expected_sha:
            actual = hashlib.sha256(dest.read_bytes()).hexdigest().lower()
            if actual == expected_sha:
                sess["_local_path"] = str(dest)
                sess["_downloaded"] = False
                pulled.append(sess)
                continue

        try:
            size = client.download(f"/v1/sessions/{sess_id}/raw", dest)
        except urllib.error.HTTPError as exc:
            print(f"  ! skipping {sess_id}: HTTP {exc.code} — {exc.reason}", file=sys.stderr)
            continue
        sess["_local_path"] = str(dest)
        sess["_downloaded"] = True
        sess["_downloaded_bytes"] = size
        pulled.append(sess)
        print(f"  + {subject}/{sess_id} ({size} B, label={sess.get('activity_label')})")

    return pulled


# ------------------------------------------------------------------ convert ---


def _load_session_payload(path: Path) -> dict[str, Any]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    # Files are stored as ``{"stored_at": ..., "request": {...}}``; older dumps
    # may have been written without the wrapper, so accept both.
    if isinstance(doc, dict) and "request" in doc and isinstance(doc["request"], dict):
        return doc["request"]
    return doc


def _write_sensor_csv(rows: list[dict[str, float]], axes: tuple[str, str, str], dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    lines = ["seconds_elapsed,x,y,z"]
    for row in rows:
        t = float(row["timestamp"])
        x = row.get(axes[0])
        y = row.get(axes[1])
        z = row.get(axes[2])
        if x is None or y is None or z is None:
            continue
        lines.append(f"{t:.6f},{float(x):.6f},{float(y):.6f},{float(z):.6f}")
    dest.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_metadata_csv(meta: dict[str, Any], dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    sampling_rate = meta.get("sampling_rate_hz")
    sample_rate_ms = ""
    try:
        if sampling_rate is not None and float(sampling_rate) > 0:
            sample_rate_ms = str(round(1000.0 / float(sampling_rate), 2))
    except (TypeError, ValueError):
        pass
    rows = [
        ("subject_id", meta.get("subject_id", "")),
        ("session_id", meta.get("session_id", "")),
        ("activity_label", meta.get("activity_label") or ""),
        ("placement", meta.get("placement") or "pocket"),
        ("device_platform", meta.get("device_platform", "")),
        ("device_model", meta.get("device_model") or ""),
        ("recording_started_at", meta.get("recording_started_at") or ""),
        ("recording_ended_at", meta.get("recording_ended_at") or ""),
        ("sampleratems", sample_rate_ms),
    ]
    header = ",".join(k for k, _ in rows)
    values = ",".join(_csv_escape(v) for _, v in rows)
    dest.write_text(f"{header}\n{values}\n", encoding="utf-8")


def _safe_relative_to(path: Path, root: Path) -> Path:
    try:
        return path.relative_to(root)
    except ValueError:
        return path


def _resolve_repo_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (_REPO_ROOT / path).resolve()


def _csv_escape(value: Any) -> str:
    text = "" if value is None else str(value)
    if any(ch in text for ch in (",", '"', "\n")):
        return '"' + text.replace('"', '""') + '"'
    return text


def _drop_leading_timestamp_gap(
    samples: list[dict[str, Any]],
    *,
    threshold_seconds: float | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Drop a stale first sample when the live phone stream starts after a gap.

    Some mobile captures can contain a single bootstrap sample at ``t=0``
    followed by the real stream tens or thousands of seconds later. Keeping
    that sample makes a 30s capture look like a much longer session. This
    optional cleaning step removes only that leading isolated sample and
    rebases the remaining stream to zero.
    """

    diagnostics: dict[str, Any] = {
        "enabled": threshold_seconds is not None,
        "dropped_leading_samples": 0,
        "dropped_leading_gap_seconds": None,
        "original_duration_seconds": None,
        "cleaned_duration_seconds": None,
    }
    if len(samples) < 3:
        return samples, diagnostics

    try:
        timestamps = [float(row["timestamp"]) for row in samples]
    except (KeyError, TypeError, ValueError):
        return samples, diagnostics

    diagnostics["original_duration_seconds"] = max(0.0, timestamps[-1] - timestamps[0])
    if threshold_seconds is None:
        diagnostics["cleaned_duration_seconds"] = diagnostics["original_duration_seconds"]
        return samples, diagnostics

    first_gap = timestamps[1] - timestamps[0]
    if first_gap <= float(threshold_seconds):
        diagnostics["cleaned_duration_seconds"] = diagnostics["original_duration_seconds"]
        return samples, diagnostics

    base_ts = timestamps[1]
    cleaned: list[dict[str, Any]] = []
    for row in samples[1:]:
        next_row = dict(row)
        next_row["timestamp"] = float(next_row["timestamp"]) - base_ts
        cleaned.append(next_row)

    diagnostics.update(
        {
            "dropped_leading_samples": 1,
            "dropped_leading_gap_seconds": first_gap,
            "cleaned_duration_seconds": max(
                0.0,
                float(cleaned[-1]["timestamp"]) - float(cleaned[0]["timestamp"]),
            )
            if len(cleaned) >= 2
            else 0.0,
        }
    )
    return cleaned, diagnostics


def _convert_sessions(
    pulled: list[dict[str, Any]],
    *,
    work_root: Path,
    drop_leading_gap_seconds: float | None,
) -> dict[str, Any]:
    """Write per-session + merged phone-folder layouts.

    Returns a summary with per-session durations and the merged folder path.
    Sessions with ``activity_label is None`` are skipped — they cannot be used
    for supervised training without manual labelling.
    """
    sessions_dir = work_root / "sessions"
    merged_dir = work_root / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)

    merged_accel: list[dict[str, float]] = []
    merged_gyro: list[dict[str, float]] = []
    annotation_rows: list[dict[str, Any]] = []
    per_session: list[dict[str, Any]] = []

    time_offset = 0.0

    for sess in pulled:
        payload = _load_session_payload(Path(sess["_local_path"]))
        meta = payload.get("metadata") or {}
        samples = payload.get("samples") or []
        samples, timestamp_diagnostics = _drop_leading_timestamp_gap(
            samples,
            threshold_seconds=drop_leading_gap_seconds,
        )
        if not samples:
            continue

        session_id = str(meta.get("session_id") or sess["app_session_id"])
        activity_label = meta.get("activity_label")

        sess_dir = sessions_dir / session_id
        _write_sensor_csv(samples, ("ax", "ay", "az"), sess_dir / "Accelerometer.csv")
        _write_sensor_csv(samples, ("gx", "gy", "gz"), sess_dir / "Gyroscope.csv")
        _write_metadata_csv(meta, sess_dir / "Metadata.csv")

        start_ts = float(samples[0]["timestamp"])
        end_ts = float(samples[-1]["timestamp"])
        duration_s = max(0.0, end_ts - start_ts)

        # Offset per-session timestamps into the merged stream's time base.
        merged_start = time_offset
        merged_end = merged_start + duration_s
        for row in samples:
            t = float(row["timestamp"]) - start_ts + time_offset
            if row.get("ax") is not None and row.get("ay") is not None and row.get("az") is not None:
                merged_accel.append(
                    {
                        "timestamp": t,
                        "ax": float(row["ax"]),
                        "ay": float(row["ay"]),
                        "az": float(row["az"]),
                    }
                )
            if row.get("gx") is not None and row.get("gy") is not None and row.get("gz") is not None:
                merged_gyro.append(
                    {
                        "timestamp": t,
                        "gx": float(row["gx"]),
                        "gy": float(row["gy"]),
                        "gz": float(row["gz"]),
                    }
                )
        # Advance the merged clock by the full session duration plus a 1s
        # gap so the builder's overlap logic cannot leak across boundaries.
        time_offset = merged_end + 1.0

        per_session.append(
            {
                "session_id": session_id,
                "subject_id": meta.get("subject_id"),
                "activity_label": activity_label,
                "placement": meta.get("placement"),
                "sample_count": len(samples),
                "duration_seconds": duration_s,
                "merged_start_ts": merged_start,
                "merged_end_ts": merged_end,
                "session_dir": str(_safe_relative_to(sess_dir, _REPO_ROOT)),
                "timestamp_normalization": timestamp_diagnostics,
            }
        )

        if activity_label:
            annotation_rows.append(
                {
                    "session_id": session_id,
                    "start_ts": merged_start,
                    "end_ts": merged_end,
                    "final_label": str(activity_label).lower(),
                    "priority_rank": 1,
                    "row_type": "manual",
                    "notes": "session-level activity_label",
                }
            )

    # Write merged phone-folder.
    def _dump_merged(rows: list[dict[str, float]], axes: tuple[str, str, str], name: str) -> None:
        out = merged_dir / name
        lines = ["seconds_elapsed,x,y,z"]
        for row in rows:
            lines.append(
                f"{row['timestamp']:.6f},"
                f"{row[axes[0]]:.6f},{row[axes[1]]:.6f},{row[axes[2]]:.6f}"
            )
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    _dump_merged(merged_accel, ("ax", "ay", "az"), "Accelerometer.csv")
    _dump_merged(merged_gyro, ("gx", "gy", "gz"), "Gyroscope.csv")

    # Minimal merged metadata; the loader handles missing fields gracefully.
    (merged_dir / "Metadata.csv").write_text(
        "subject_id,session_id,placement\nphone_retrain,merged,pocket\n",
        encoding="utf-8",
    )

    # Manifest maps merged-stream midpoint_ts back to real session context.
    # The diagnose stage (and any downstream stratified analysis) consumes it.
    manifest_path = work_root / "sessions_manifest.json"
    manifest_path.write_text(
        json.dumps(per_session, indent=2, default=str),
        encoding="utf-8",
    )

    return {
        "sessions_dir": str(sessions_dir),
        "sessions_manifest": str(manifest_path),
        "merged_dir": str(merged_dir),
        "annotation_rows": annotation_rows,
        "per_session": per_session,
        "merged_samples": {"accel": len(merged_accel), "gyro": len(merged_gyro)},
    }


# ----------------------------------------------------------------- annotate ---


def _write_annotation_csv(rows: list[dict[str, Any]], dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    headers = ["session_id", "start_ts", "end_ts", "final_label", "priority_rank", "row_type", "notes"]
    lines = [",".join(headers)]
    for row in rows:
        lines.append(
            ",".join(
                _csv_escape(row.get(h, "")) for h in headers
            )
        )
    dest.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ------------------------------------------------------------------ replay ---


def _run_subscript(args: list[str], *, env: dict[str, str] | None = None) -> int:
    print(f"$ {' '.join(args)}")
    result = subprocess.run(args, cwd=str(_REPO_ROOT), env=env)
    return result.returncode


def _run_replay(
    *,
    merged_dir: Path,
    out_dir: Path,
    har_artifact: Path,
    fall_artifact: Path,
    threshold_mode: str,
    timeline_tolerance_seconds: float,
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    har_csv = out_dir / "har.csv"
    fall_csv = out_dir / "fall.csv"
    timeline_csv = out_dir / "timeline.csv"
    report_json = out_dir / "replay_report.json"

    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "run_combined_runtime_replay.py"),
        "--input-source",
        "phone_folder",
        "--input-path",
        str(merged_dir),
        "--har-artifact",
        str(har_artifact),
        "--fall-artifact",
        str(fall_artifact),
        "--threshold-mode",
        threshold_mode,
        "--timeline-tolerance-seconds",
        str(timeline_tolerance_seconds),
        "--har-out",
        str(har_csv),
        "--fall-out",
        str(fall_csv),
        "--combined-out",
        str(timeline_csv),
        "--report-out",
        str(report_json),
    ]
    rc = _run_subscript(cmd)
    if rc != 0:
        raise SystemExit(f"replay stage failed with exit {rc}")
    return {"har": har_csv, "fall": fall_csv, "timeline": timeline_csv, "report": report_json}


# ------------------------------------------------------------------- build ---


def _run_build(
    *,
    annotation_csv: Path,
    har_csv: Path,
    fall_csv: Path,
    out_dir: Path,
    min_overlap_fraction: float,
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    har_out = out_dir / "har_labeled_windows.csv"
    fall_out = out_dir / "fall_labeled_windows.csv"
    summary_out = out_dir / "build_summary.json"

    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "build_phone_runtime_training_set.py"),
        "--annotation-csv",
        str(annotation_csv),
        "--har-csv",
        str(har_csv),
        "--fall-csv",
        str(fall_csv),
        "--har-out",
        str(har_out),
        "--fall-out",
        str(fall_out),
        "--summary-out",
        str(summary_out),
        "--min-overlap-fraction",
        str(min_overlap_fraction),
    ]
    rc = _run_subscript(cmd)
    if rc != 0:
        raise SystemExit(f"build stage failed with exit {rc}")
    return {"har": har_out, "fall": fall_out, "summary": summary_out}


# -------------------------------------------------------------------- train --


def _run_train_fall(
    *,
    fall_labeled_csv: Path,
    out_dir: Path,
    public_train_source: str,
    public_eval_source: str,
    threshold_mode: str,
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "train_fall_summary.json"
    phone_preds = out_dir / "phone_predictions.csv"
    public_preds = out_dir / "public_predictions.csv"

    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "train_fall_with_phone_hard_negatives.py"),
        "--phone-fall-labeled-csv",
        str(fall_labeled_csv),
        "--public-train-source",
        public_train_source,
        "--public-eval-source",
        public_eval_source,
        "--threshold-mode",
        threshold_mode,
        "--out-json",
        str(out_json),
        "--phone-predictions-out",
        str(phone_preds),
        "--public-predictions-out",
        str(public_preds),
    ]
    rc = _run_subscript(cmd)
    if rc != 0:
        raise SystemExit(f"train-fall stage failed with exit {rc}")
    return {"summary": out_json, "phone_preds": phone_preds, "public_preds": public_preds}


def _run_train_har(
    *,
    merged_dir: Path,
    annotation_csv: Path,
    train_dataset: str,
    results_root: Path,
    run_id: str,
) -> dict[str, Path]:
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "run_phone_har_adaptation.py"),
        "--train-dataset",
        train_dataset,
        "--phone-folder",
        str(merged_dir),
        "--annotation-csv",
        str(annotation_csv),
        "--results-root",
        str(results_root),
        "--run-id",
        run_id,
        "--skip-plots",
    ]
    rc = _run_subscript(cmd)
    if rc != 0:
        raise SystemExit(f"train-har stage failed with exit {rc}")
    run_dir = results_root / run_id
    return {"run_dir": run_dir}


# ------------------------------------------------------------------ report ---


def _current_artifact_summary(overrides: dict[str, Path] | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for task in ("fall", "har"):
        override = (overrides or {}).get(task)
        if override is not None:
            out[task] = {
                "artifact_path": str(override),
                "source": "cli_override",
            }
            continue
        try:
            meta = load_current_metadata(task)
            out[task] = {
                "artifact_id": meta.get("artifact_id"),
                "model_kind": meta.get("model_kind"),
                "promoted_utc": meta.get("promoted_utc"),
                "heldout": meta.get("heldout"),
                "validation": meta.get("validation"),
            }
        except FileNotFoundError as exc:
            out[task] = {"error": str(exc)}
    return out


def _label_coverage(per_session: list[dict[str, Any]]) -> dict[str, Any]:
    coverage: dict[str, dict[str, Any]] = {}
    for row in per_session:
        label = (row.get("activity_label") or "_unlabeled").lower()
        entry = coverage.setdefault(label, {"sessions": 0, "seconds": 0.0, "samples": 0})
        entry["sessions"] += 1
        entry["seconds"] += float(row.get("duration_seconds") or 0.0)
        entry["samples"] += int(row.get("sample_count") or 0)
    return coverage


def _scarcity_warnings(coverage: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    for label, stats in coverage.items():
        if label == "_unlabeled":
            if stats["sessions"] > 0:
                warnings.append(
                    f"{stats['sessions']} sessions have no activity_label "
                    f"and will be dropped from supervised training"
                )
            continue
        if stats["sessions"] < MIN_SESSIONS_PER_LABEL:
            warnings.append(
                f"label {label!r}: only {stats['sessions']} sessions "
                f"(< {MIN_SESSIONS_PER_LABEL})"
            )
        if stats["seconds"] < MIN_SECONDS_PER_LABEL:
            warnings.append(
                f"label {label!r}: only {stats['seconds']:.1f}s of data "
                f"(< {MIN_SECONDS_PER_LABEL:.0f}s)"
            )
    return warnings


def _failure_driven_recommendations(
    diagnostic_summary: dict[str, Any] | None,
) -> list[str]:
    """Turn the diagnostic top-N failure modes into one-line actionable advice.

    Scarcity warnings tell the user what is *missing*; these tell the user what
    to record *next* based on which failure modes are actually hurting quality.
    """
    if not diagnostic_summary:
        return []
    out: list[str] = []
    for mode in diagnostic_summary.get("top_failure_modes") or []:
        out.append(f"{mode.get('detail', 'failure mode')} → {mode.get('suggestion', '')}".strip())
    return out


# -------------------------------------------------------------------- main ---


def _parse_stages(raw: str | None) -> list[str]:
    if not raw:
        return list(DEFAULT_STAGES)
    selected: list[str] = []
    for part in raw.split(","):
        name = part.strip().lower()
        if not name:
            continue
        if name not in ALL_STAGES:
            raise SystemExit(f"Unknown stage {name!r}. Valid: {', '.join(ALL_STAGES)}")
        selected.append(name)
    return selected


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--server-url", required=True, help="Base URL of the API, e.g. http://host:8000")
    parser.add_argument("--subject-id", default=None, help="Filter to one subject (default: all visible to caller)")
    parser.add_argument("--labels", default=None, help="Comma-separated activity_label filter")
    parser.add_argument("--since", default=None, help="Only pull sessions recorded after this ISO timestamp")
    parser.add_argument("--max-sessions", type=int, default=None, help="Stop after N sessions pulled")
    parser.add_argument("--auth-env", default="RUNTIME_BASIC_AUTH", help="Environment variable holding user:password")
    parser.add_argument("--pull-root", default=str(DEFAULT_PULL_ROOT))
    parser.add_argument(
        "--work-root",
        default=None,
        help="Output root for this run (default: artifacts/phone_training/<timestamp>)",
    )
    parser.add_argument(
        "--report-root",
        default=str(DEFAULT_REPORT_ROOT),
        help="Root directory for the final report",
    )
    parser.add_argument("--stages", default=None, help="Comma-separated stage list (default: all non-train stages)")
    parser.add_argument(
        "--replay-threshold-mode",
        default="shared",
        choices=["shared", "dataset_presets"],
    )
    parser.add_argument("--replay-tolerance-seconds", type=float, default=1.0)
    parser.add_argument(
        "--har-artifact",
        default=None,
        help="Optional HAR model artifact path. Defaults to artifacts/har/current/model.joblib.",
    )
    parser.add_argument(
        "--fall-artifact",
        default=None,
        help="Optional fall model artifact path. Defaults to artifacts/fall/current/model.joblib.",
    )
    parser.add_argument(
        "--drop-leading-gap-seconds",
        type=float,
        default=None,
        help=(
            "Optional phone-cleaning guard: when the first two samples are "
            "separated by more than this many seconds, drop the isolated first "
            "sample and rebase the remaining stream to zero."
        ),
    )
    parser.add_argument("--build-min-overlap", type=float, default=0.25)
    parser.add_argument(
        "--har-train-dataset",
        default="both",
        choices=["uci_har", "pamap2", "both"],
    )
    parser.add_argument(
        "--fall-public-train-source",
        default="combined",
        choices=["mobifall", "sisfall", "combined"],
    )
    parser.add_argument(
        "--fall-public-eval-source",
        default="sisfall",
        choices=["mobifall", "sisfall", "none"],
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    stages = _parse_stages(args.stages)
    run_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    work_root = Path(args.work_root) if args.work_root else DEFAULT_WORK_ROOT / run_ts
    pull_root = Path(args.pull_root)
    report_dir = Path(args.report_root) / run_ts
    work_root.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    artifact_overrides: dict[str, Path] = {}
    if args.har_artifact:
        artifact_overrides["har"] = _resolve_repo_path(args.har_artifact)
    if args.fall_artifact:
        artifact_overrides["fall"] = _resolve_repo_path(args.fall_artifact)

    labels_filter = None
    if args.labels:
        labels_filter = [x.strip().lower() for x in args.labels.split(",") if x.strip()]

    auth_token = _load_auth_token(args.auth_env)
    client = HttpClient(args.server_url, auth_token)

    state: dict[str, Any] = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "server_url": args.server_url,
        "subject_id": args.subject_id,
        "labels_filter": labels_filter,
        "since": args.since,
        "max_sessions": args.max_sessions,
        "work_root": str(work_root),
        "pull_root": str(pull_root),
        "stages_requested": stages,
        "stages_completed": [],
    }

    pulled: list[dict[str, Any]] = []
    conversion: dict[str, Any] = {}
    annotation_csv: Path | None = None
    replay_outputs: dict[str, Path] | None = None
    build_outputs: dict[str, Path] | None = None
    train_fall_outputs: dict[str, Path] | None = None
    train_har_outputs: dict[str, Path] | None = None

    if "pull" in stages:
        print(f"[pull] listing sessions from {args.server_url}")
        pulled = _pull_sessions(
            client,
            subject_id=args.subject_id,
            labels=labels_filter,
            since=args.since,
            pull_root=pull_root,
            limit=args.max_sessions,
        )
        print(f"[pull] {len(pulled)} session(s) resolved")
        state["pulled_count"] = len(pulled)
        state["stages_completed"].append("pull")
    else:
        # When skipping pull, assume prior-run cache is present. Walk the
        # subject dir recursively so feedback-derived rows under
        # ``<subject>/feedback/`` are picked up automatically — the export
        # script in Phase 4 is what populates that subfolder.
        if args.subject_id:
            subject_dir = pull_root / args.subject_id
            if subject_dir.exists():
                for path in sorted(subject_dir.rglob("*.json")):
                    pulled.append({
                        "app_session_id": path.stem,
                        "subject_id": args.subject_id,
                        "activity_label": None,  # inferred during convert
                        "_local_path": str(path),
                        "_downloaded": False,
                    })

    if "convert" in stages:
        if not pulled:
            raise SystemExit("convert requires pulled sessions; rerun with pull in --stages")
        print(f"[convert] writing {len(pulled)} session(s) to {work_root}")
        conversion = _convert_sessions(
            pulled,
            work_root=work_root,
            drop_leading_gap_seconds=args.drop_leading_gap_seconds,
        )
        print(
            f"[convert] merged accel samples={conversion['merged_samples']['accel']} "
            f"gyro samples={conversion['merged_samples']['gyro']}"
        )
        state["conversion"] = {
            "sessions_dir": conversion["sessions_dir"],
            "merged_dir": conversion["merged_dir"],
            "merged_samples": conversion["merged_samples"],
            "per_session_count": len(conversion["per_session"]),
        }
        state["stages_completed"].append("convert")

    if "annotate" in stages:
        if not conversion:
            raise SystemExit("annotate requires convert to have run first")
        annotation_csv = work_root / "annotations.csv"
        _write_annotation_csv(conversion["annotation_rows"], annotation_csv)
        print(f"[annotate] wrote {len(conversion['annotation_rows'])} rows to {annotation_csv}")
        state["annotation_csv"] = str(annotation_csv)
        state["stages_completed"].append("annotate")

    if "replay" in stages:
        if not conversion:
            raise SystemExit("replay requires convert to have run first")
        har_artifact = artifact_overrides.get("har") or resolve_current_artifact("har")
        fall_artifact = artifact_overrides.get("fall") or resolve_current_artifact("fall")
        if not har_artifact.is_file():
            raise SystemExit(f"HAR artifact not found: {har_artifact}")
        if not fall_artifact.is_file():
            raise SystemExit(f"fall artifact not found: {fall_artifact}")
        replay_dir = work_root / "replay"
        print(f"[replay] har={har_artifact.name} fall={fall_artifact.name}")
        replay_outputs = _run_replay(
            merged_dir=Path(conversion["merged_dir"]),
            out_dir=replay_dir,
            har_artifact=har_artifact,
            fall_artifact=fall_artifact,
            threshold_mode=args.replay_threshold_mode,
            timeline_tolerance_seconds=args.replay_tolerance_seconds,
        )
        state["replay"] = {k: str(v) for k, v in replay_outputs.items()}
        state["stages_completed"].append("replay")

    if "build" in stages:
        if annotation_csv is None or replay_outputs is None:
            raise SystemExit("build requires annotate + replay to have run first")
        build_dir = work_root / "build"
        print(f"[build] labelling windows via builder")
        build_outputs = _run_build(
            annotation_csv=annotation_csv,
            har_csv=replay_outputs["har"],
            fall_csv=replay_outputs["fall"],
            out_dir=build_dir,
            min_overlap_fraction=args.build_min_overlap,
        )
        state["build"] = {k: str(v) for k, v in build_outputs.items()}
        state["stages_completed"].append("build")

    diagnostic_summary: dict[str, Any] | None = None
    if "diagnose" in stages:
        if replay_outputs is None or not conversion:
            raise SystemExit("diagnose requires replay + convert to have run first")
        manifest_path = Path(conversion.get("sessions_manifest") or (work_root / "sessions_manifest.json"))
        diagnose_dir = work_root / "diagnose"
        diagnose_dir.mkdir(parents=True, exist_ok=True)
        print(f"[diagnose] stratifying replay outputs by placement + activity_label")
        from scripts.diagnose_phone_artifacts import (  # noqa: WPS433
            _resolve_current_threshold,
            run_diagnosis,
        )
        current_threshold = _resolve_current_threshold(None)
        diagnostic_summary = run_diagnosis(
            fall_csv=replay_outputs["fall"],
            har_csv=replay_outputs["har"],
            sessions_manifest=manifest_path,
            current_threshold=current_threshold,
            out_dir=diagnose_dir,
        )
        state["diagnose"] = {
            "out_dir": str(diagnose_dir),
            "top_failure_modes": diagnostic_summary.get("top_failure_modes", []),
        }
        state["stages_completed"].append("diagnose")

    if "train-fall" in stages:
        if build_outputs is None:
            raise SystemExit("train-fall requires build to have run first")
        train_fall_dir = work_root / "train_fall"
        print(f"[train-fall] training fall with phone hard negatives")
        train_fall_outputs = _run_train_fall(
            fall_labeled_csv=build_outputs["fall"],
            out_dir=train_fall_dir,
            public_train_source=args.fall_public_train_source,
            public_eval_source=args.fall_public_eval_source,
            threshold_mode=args.replay_threshold_mode,
        )
        state["train_fall"] = {k: str(v) for k, v in train_fall_outputs.items()}
        state["stages_completed"].append("train-fall")

    if "train-har" in stages:
        if annotation_csv is None or not conversion:
            raise SystemExit("train-har requires convert + annotate to have run first")
        run_id = f"phone_har_adapt__{run_ts}"
        har_results_root = work_root / "train_har"
        print(f"[train-har] HAR adaptation run_id={run_id}")
        train_har_outputs = _run_train_har(
            merged_dir=Path(conversion["merged_dir"]),
            annotation_csv=annotation_csv,
            train_dataset=args.har_train_dataset,
            results_root=har_results_root,
            run_id=run_id,
        )
        state["train_har"] = {"run_dir": str(train_har_outputs["run_dir"])}
        state["stages_completed"].append("train-har")

    if "report" in stages:
        coverage = _label_coverage(conversion.get("per_session", [])) if conversion else {}
        warnings = _scarcity_warnings(coverage)
        recommendations = _failure_driven_recommendations(diagnostic_summary)
        report = {
            "run_utc": run_ts,
            "server_url": args.server_url,
            "subject_id": args.subject_id,
            "stages_completed": state["stages_completed"],
            "sessions_pulled": len(pulled),
            "session_label_coverage": coverage,
            "scarcity_warnings": warnings,
            "failure_driven_recommendations": recommendations,
            "diagnostic_summary": diagnostic_summary,
            "current_artifacts": _current_artifact_summary(artifact_overrides),
            "paths": {
                "pull_root": str(pull_root),
                "work_root": str(work_root),
                "annotation_csv": str(annotation_csv) if annotation_csv else None,
                "replay": {k: str(v) for k, v in (replay_outputs or {}).items()} or None,
                "build": {k: str(v) for k, v in (build_outputs or {}).items()} or None,
                "diagnose": state.get("diagnose"),
                "train_fall": {k: str(v) for k, v in (train_fall_outputs or {}).items()} or None,
                "train_har": {"run_dir": str(train_har_outputs["run_dir"])} if train_har_outputs else None,
            },
        }

        # Attach end-to-end metric deltas when the train stages ran.
        if train_fall_outputs and train_fall_outputs.get("summary", Path()).exists():
            try:
                report["train_fall_summary"] = json.loads(train_fall_outputs["summary"].read_text())
            except (OSError, json.JSONDecodeError) as exc:
                report["train_fall_summary_error"] = str(exc)
        if train_har_outputs:
            summary_candidate = train_har_outputs["run_dir"] / "summary.json"
            if summary_candidate.exists():
                try:
                    report["train_har_summary"] = json.loads(summary_candidate.read_text())
                except (OSError, json.JSONDecodeError) as exc:
                    report["train_har_summary_error"] = str(exc)

        report_path = report_dir / "report.json"
        report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        print(f"[report] wrote {report_path}")

        print()
        print("Sessions pulled:", len(pulled))
        if coverage:
            print("Per-label coverage:")
            for label, stats in sorted(coverage.items()):
                print(
                    f"  {label:<16s} sessions={stats['sessions']:<3d} "
                    f"seconds={stats['seconds']:.1f} samples={stats['samples']}"
                )
        if warnings:
            print("Warnings:")
            for w in warnings:
                print(f"  ! {w}")
        if recommendations:
            print("Top failure modes to address next:")
            for rec in recommendations:
                print(f"  → {rec}")
        state["stages_completed"].append("report")

    # Suggest next-step commands if the train stages were skipped.
    if "train-fall" not in stages and build_outputs is not None:
        print()
        print("Suggested fall training command:")
        print(
            f"  python scripts/train_fall_with_phone_hard_negatives.py "
            f"--phone-fall-labeled-csv {build_outputs['fall']} "
            f"--public-train-source {args.fall_public_train_source} "
            f"--public-eval-source {args.fall_public_eval_source}"
        )
    if "train-har" not in stages and annotation_csv is not None and conversion:
        print("Suggested HAR adaptation command:")
        print(
            f"  python scripts/run_phone_har_adaptation.py "
            f"--train-dataset {args.har_train_dataset} "
            f"--phone-folder {conversion['merged_dir']} "
            f"--annotation-csv {annotation_csv}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
