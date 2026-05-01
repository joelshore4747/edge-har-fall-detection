"""Re-score persisted sessions through the current vulnerability scoring layer.

Usage (from repo root, with the same env vars the API uses set):

    # show what would change without writing anything (default)
    python scripts/rescore_sessions.py --current-warning-level high --limit 50

    # actually write the new inferences back
    python scripts/rescore_sessions.py --current-warning-level high --apply

    # one specific session
    python scripts/rescore_sessions.py --app-session-id <uuid> --apply

The script reads each target session's stored raw payload
(``app_sessions.raw_storage_uri``), re-runs the full inference pipeline
through the same code path the API uses, and inserts a fresh inference
row under the same ``app_session_id`` (the existing inference rows are
left in place — the latest one is what the admin UI surfaces).

Required env vars (same as the API): ``DATABASE_URL``, ``HAR_ARTIFACT_PATH``
or default, ``FALL_ARTIFACT_PATH`` or default.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import psycopg
from psycopg.rows import dict_row

# Allow running as a script from the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# apps.api.main hard-fails at import if ADMIN_SESSION_SECRET is unset or is
# the documented placeholder. This script never serves HTTP, so the secret
# value is never used — but we still need to clear the import-time guard.
# Use setdefault so a real value from the environment wins.
os.environ.setdefault(
    "ADMIN_SESSION_SECRET", "rescore-session-script-no-http-served"
)

from apps.api.main import (  # noqa: E402
    FALL_ARTIFACT_PATH,
    HAR_ARTIFACT_PATH,
    _build_debug_summary,
    _build_fall_summary,
    _build_har_summary,
    _build_model_info,
    _build_placement_summary,
    _build_session_narrative_summary,
    _build_source_summary,
    _build_vulnerability_summary,
    _derive_alert_summary,
    _request_id_from_runtime_request,
    _request_to_dataframe,
    _to_combined_timeline_models,
    _to_dataframe,
    _to_fall_models,
    _to_grouped_event_models,
    _to_har_models,
    _to_mapping,
    _to_point_timeline_models,
    _to_timeline_event_models,
    _to_transition_event_models,
    _to_vulnerability_models,
)
from apps.api.schemas import RuntimeSessionRequest, RuntimeSessionResponse  # noqa: E402
from services.runtime_inference import (  # noqa: E402
    RuntimeArtifacts,
    RuntimeInferenceConfig,
    run_runtime_inference_from_dataframe,
)
from services.runtime_persistence import persist_runtime_session  # noqa: E402

logger = logging.getLogger("rescore_sessions")

# HAR top labels typically dominated by the false-positive HIGH bug.
DYNAMIC_HAR_FILTER = ("walking", "running", "stairs", "dynamic", "locomotion")


def _select_target_sessions(
    cur: psycopg.Cursor,
    *,
    app_session_id: UUID | None,
    current_warning_level: str | None,
    har_filter: tuple[str, ...] | None,
    limit: int | None,
) -> list[dict[str, Any]]:
    """Return the most recent inference per session matching the filters."""
    where_clauses: list[str] = []
    params: list[Any] = []

    if app_session_id is not None:
        where_clauses.append("s.app_session_id = %s")
        params.append(app_session_id)
    if current_warning_level is not None:
        where_clauses.append("i.warning_level = %s")
        params.append(current_warning_level)
    if har_filter:
        where_clauses.append("LOWER(COALESCE(i.top_har_label, '')) = ANY(%s)")
        params.append(list(har_filter))

    where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
    limit_sql = f"LIMIT {int(limit)}" if limit is not None else ""

    cur.execute(
        f"""
        WITH latest AS (
            SELECT DISTINCT ON (app_session_id)
                inference_id,
                app_session_id,
                warning_level,
                top_har_label,
                top_har_fraction,
                created_at
            FROM app_session_inferences
            ORDER BY app_session_id, created_at DESC
        )
        SELECT
            i.inference_id,
            i.app_session_id,
            i.warning_level,
            i.top_har_label,
            i.top_har_fraction,
            s.client_session_id,
            s.raw_storage_uri,
            s.raw_storage_format
        FROM latest AS i
        JOIN app_sessions AS s ON s.app_session_id = i.app_session_id
        {where_sql}
        ORDER BY i.created_at DESC
        {limit_sql}
        """,
        params,
    )
    return [dict(row) for row in cur.fetchall()]


def _read_raw_request(raw_storage_uri: str) -> RuntimeSessionRequest:
    path = Path(raw_storage_uri)
    if not path.is_file():
        raise FileNotFoundError(f"raw payload missing on disk: {raw_storage_uri}")
    with path.open("r") as f:
        envelope = json.load(f)
    request_payload = envelope.get("request") if isinstance(envelope, dict) else None
    if not isinstance(request_payload, dict):
        raise ValueError(f"raw payload at {raw_storage_uri} has no 'request' field")
    return RuntimeSessionRequest.model_validate(request_payload)


def _re_run_inference(req: RuntimeSessionRequest) -> RuntimeSessionResponse:
    """Replicate the body of ``apps.api.main.infer_session`` (response-build
    + inference) without the HTTP/auth/logging layer. Persistence is the
    caller's responsibility."""
    input_df = _request_to_dataframe(req)
    artifacts = RuntimeArtifacts(
        har_artifact_path=HAR_ARTIFACT_PATH,
        fall_artifact_path=FALL_ARTIFACT_PATH,
    )
    config = RuntimeInferenceConfig()
    result = run_runtime_inference_from_dataframe(
        input_df, artifacts=artifacts, config=config
    )

    har_windows_df = _to_dataframe(getattr(result, "har_windows", None))
    fall_windows_df = _to_dataframe(getattr(result, "fall_windows", None))
    vulnerability_windows_df = _to_dataframe(getattr(result, "vulnerability_windows", None))
    grouped_events_df = _to_dataframe(getattr(result, "grouped_fall_events", None))
    combined_timeline_df = _to_dataframe(getattr(result, "combined_timeline", None))
    point_timeline_df = _to_dataframe(getattr(result, "point_timeline", None))
    timeline_events_df = _to_dataframe(getattr(result, "timeline_events", None))
    transition_events_df = _to_dataframe(getattr(result, "transition_events", None))

    source_summary = _build_source_summary(result, req, input_df)
    placement_summary = _build_placement_summary(result)
    har_summary = _build_har_summary(result)
    fall_summary = _build_fall_summary(result)
    vulnerability_summary = _build_vulnerability_summary(result)
    alert_summary = _derive_alert_summary(
        result, placement_summary, har_summary, fall_summary, vulnerability_summary
    )
    debug_summary = _build_debug_summary(result, req, input_df)
    model_info = _build_model_info()
    session_narrative_summary = _build_session_narrative_summary(result)
    narrative_summary = _to_mapping(getattr(result, "narrative_summary", None))

    # Mint a fresh request_id so persist_runtime_session inserts a new
    # inference row rather than dedup'ing against the original ingest's
    # request_id (UNIQUE constraint on app_session_inferences.request_id).
    return RuntimeSessionResponse(
        request_id=uuid4(),
        session_id=req.metadata.session_id,
        source_summary=source_summary,
        placement_summary=placement_summary,
        har_summary=har_summary,
        fall_summary=fall_summary,
        vulnerability_summary=vulnerability_summary,
        alert_summary=alert_summary,
        debug_summary=debug_summary,
        model_info=model_info,
        # The original ingestion may have had include_* flags off; for
        # backfill we always include the lot so the new inference row
        # carries full per-window provenance.
        grouped_fall_events=_to_grouped_event_models(grouped_events_df),
        har_windows=_to_har_models(har_windows_df),
        fall_windows=_to_fall_models(fall_windows_df),
        vulnerability_windows=_to_vulnerability_models(vulnerability_windows_df),
        combined_timeline=_to_combined_timeline_models(combined_timeline_df),
        point_timeline=_to_point_timeline_models(point_timeline_df),
        timeline_events=_to_timeline_event_models(timeline_events_df),
        transition_events=_to_transition_event_models(transition_events_df),
        session_narrative_summary=session_narrative_summary,
        narrative_summary=narrative_summary,
    )


def _format_diff_row(
    target: dict[str, Any], new_warning: str, new_top_score: float | None
) -> str:
    return (
        f"{str(target['app_session_id'])[:8]}…  "
        f"har={target.get('top_har_label') or '-':<10}  "
        f"frac={target.get('top_har_fraction') or 0.0:.2f}  "
        f"{target['warning_level']:>6} -> {new_warning:<6}  "
        f"top_score={new_top_score if new_top_score is not None else float('nan'):.3f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--app-session-id", type=UUID, default=None)
    parser.add_argument(
        "--current-warning-level",
        choices=("none", "low", "medium", "high"),
        default=None,
        help="Only re-score sessions whose latest stored warning_level matches",
    )
    parser.add_argument(
        "--only-walking-har",
        action="store_true",
        help="Restrict to sessions whose top_har_label is walking/stairs/etc.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually write the new inference rows. Default is dry-run.",
    )
    parser.add_argument("--db-url", default=None, help="Defaults to $DATABASE_URL")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    db_url = args.db_url or _resolve_db_url()
    har_filter = DYNAMIC_HAR_FILTER if args.only_walking_har else None

    with psycopg.connect(db_url, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            targets = _select_target_sessions(
                cur,
                app_session_id=args.app_session_id,
                current_warning_level=args.current_warning_level,
                har_filter=har_filter,
                limit=args.limit,
            )

    logger.info("matched %d candidate session(s)", len(targets))
    if not targets:
        return 0

    n_changed = 0
    n_unchanged = 0
    n_failed = 0

    for target in targets:
        app_session_id: UUID = target["app_session_id"]
        old_warning: str = target["warning_level"]
        raw_uri: str | None = target.get("raw_storage_uri")

        if not raw_uri:
            logger.warning("session %s has no raw_storage_uri; skipping", app_session_id)
            n_failed += 1
            continue

        try:
            req = _read_raw_request(raw_uri)
            response = _re_run_inference(req)
        except Exception as exc:  # noqa: BLE001
            logger.error("session %s: re-inference failed: %s", app_session_id, exc)
            n_failed += 1
            continue

        new_warning = response.alert_summary.warning_level.value
        new_top_score = response.alert_summary.top_vulnerability_score

        line = _format_diff_row(target, new_warning, new_top_score)
        if new_warning != old_warning:
            n_changed += 1
            print(("APPLY  " if args.apply else "DRY    ") + line)
        else:
            n_unchanged += 1
            if args.verbose:
                print("SAME   " + line)

        if args.apply and new_warning != old_warning:
            try:
                persist_runtime_session(req, response, db_url=db_url)
            except Exception as exc:  # noqa: BLE001
                logger.error("session %s: persist failed: %s", app_session_id, exc)
                n_failed += 1

    print()
    print(
        f"summary: {n_changed} changed, {n_unchanged} unchanged, {n_failed} failed "
        f"(dry-run={not args.apply})"
    )
    return 0 if n_failed == 0 else 1


def _resolve_db_url() -> str:
    import os

    url = os.environ.get("DATABASE_URL", "").strip()
    if not url:
        raise SystemExit(
            "DATABASE_URL is not set. Pass --db-url or export DATABASE_URL."
        )
    return url


if __name__ == "__main__":
    sys.exit(main())
