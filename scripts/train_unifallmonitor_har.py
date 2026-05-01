"""Pull labeled sessions from api.unifallmonitor.com, train/test an activity classifier.

Pipeline:

    list /v1/sessions  -> for each session, GET /v1/sessions/{id}/raw
    raw_payload (.request.metadata + .request.samples)
        -> per-sample DataFrame (timestamp, ax/ay/az, gx/gy/gz, label_mapped, ...)
        -> resample (50 Hz) + window (HAR-style, 2 s @ 50% overlap)
        -> build_feature_table  (existing pipeline.features)
        -> RandomForest classifier on canonical activity label
        -> classification_report + confusion matrix + joblib artifact

Canonical labels follow ``apps/api/schemas.CanonicalSessionLabel``:
``{fall, walking, stairs, static, other}``. With ``--collapse-static-to-other``
the four-class problem requested ("fall / walking / stairs / something else")
is produced.

Usage:

    python scripts/train_unifallmonitor_har.py \
        --base-url https://api.unifallmonitor.com \
        --username <user> --password <pwd> \
        --cache-dir artifacts/unifallmonitor/cache \
        --out artifacts/unifallmonitor/har_classifier
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from urllib import error as urlerror
from urllib import request as urlrequest
from urllib.parse import urlencode, urlsplit, urlunsplit

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from apps.api.schemas import normalise_canonical_session_label  # noqa: E402
from scripts.lib.run_registry import (  # noqa: E402
    compute_run_id,
    content_hash,
    register_run,
    update_current_symlink,
)
from models.har.train_har import (  # noqa: E402
    prepare_feature_matrices,
    select_feature_columns,
    subject_aware_group_split,
)
from pipeline.features import build_feature_table  # noqa: E402
from pipeline.preprocess import (  # noqa: E402
    PreprocessConfig,
    append_derived_channels,
    resample_dataframe,
    window_dataframe,
)


logger = logging.getLogger("train_unifallmonitor_har")


# ---- API client (stdlib only) -------------------------------------------------


@dataclass(frozen=True)
class RemoteClient:
    base_url: str
    auth_header: str | None
    timeout: float = 30.0

    def _open(self, path: str, query: dict[str, Any] | None = None) -> bytes:
        scheme, netloc, base_path, _, _ = urlsplit(self.base_url)
        full_path = base_path.rstrip("/") + ("" if path.startswith("/") else "/") + path
        encoded_query = urlencode({k: v for k, v in (query or {}).items() if v is not None})
        url = urlunsplit((scheme, netloc, full_path, encoded_query, ""))
        req = urlrequest.Request(url, headers={"Accept": "application/json"})
        if self.auth_header is not None:
            req.add_header("Authorization", self.auth_header)
        try:
            with urlrequest.urlopen(req, timeout=self.timeout) as resp:
                return resp.read()
        except urlerror.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code} for {url}: {body[:400]}") from exc

    def list_sessions(self, *, subject_id: str | None, limit: int, offset: int) -> dict[str, Any]:
        body = self._open(
            "/v1/sessions",
            query={"subject_id": subject_id, "limit": limit, "offset": offset},
        )
        return json.loads(body)

    def fetch_raw(self, app_session_id: str) -> dict[str, Any]:
        body = self._open(f"/v1/sessions/{app_session_id}/raw")
        return json.loads(body)


def make_basic_auth_header(username: str | None, password: str | None) -> str | None:
    if not username or password is None:
        return None
    token = base64.b64encode(f"{username}:{password}".encode("utf-8")).decode("ascii")
    return f"Basic {token}"


# ---- Session iteration --------------------------------------------------------


def iter_sessions(
    client: RemoteClient,
    *,
    subject_id: str | None,
    page_size: int,
    max_sessions: int | None,
) -> Iterable[dict[str, Any]]:
    fetched = 0
    offset = 0
    while True:
        page = client.list_sessions(subject_id=subject_id, limit=page_size, offset=offset)
        items = page.get("sessions") or []
        if not items:
            return
        for item in items:
            yield item
            fetched += 1
            if max_sessions is not None and fetched >= max_sessions:
                return
        if len(items) < page_size:
            return
        offset += page_size


def cache_path_for(cache_dir: Path | None, app_session_id: str) -> Path | None:
    if cache_dir is None:
        return None
    return cache_dir / f"{app_session_id}.json"


def load_or_fetch_raw(
    client: RemoteClient,
    *,
    app_session_id: str,
    cache_dir: Path | None,
    refresh: bool,
) -> dict[str, Any]:
    target = cache_path_for(cache_dir, app_session_id)
    if target is not None and target.is_file() and not refresh:
        return json.loads(target.read_text(encoding="utf-8"))

    payload = client.fetch_raw(app_session_id)
    if target is not None:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return payload


# ---- Label mapping ------------------------------------------------------------


CANONICAL_ACTIVITY_LABELS = ("fall", "walking", "stairs", "static", "other")


def canonicalize_activity_label(raw: str | None) -> str | None:
    """Normalise a free-form activity label using the API's canonical mapping.

    Returns the canonical string (``static`` / ``walking`` / ``stairs`` /
    ``fall`` / ``other``) or ``None`` for blank, ``unknown``, or unmappable
    inputs. Matches every alias the API accepts; mapping additions in
    ``apps/api/schemas`` automatically flow through here.
    """
    if raw is None:
        return None
    text = str(raw).strip().lower()
    if not text or text == "unknown":
        return None
    try:
        canonical = normalise_canonical_session_label(text, fallback=None)
    except ValueError:
        return None
    if canonical is None:
        return None
    return canonical.value


def collapse_to_four_class(label: str | None) -> str | None:
    if label is None:
        return None
    if label == "static":
        return "other"
    return label


# ---- Per-session DataFrame ----------------------------------------------------


TRIM_HEAD_SECONDS = 2.0
TRIM_TAIL_SECONDS = 2.0


def samples_to_dataframe(
    *,
    metadata: dict[str, Any],
    samples: list[dict[str, Any]],
    canonical_label: str,
    session_index: int,
) -> pd.DataFrame | None:
    if not samples:
        return None

    subject_id = str(metadata.get("subject_id") or "anonymous_user")
    session_id = str(metadata.get("session_id") or f"session_{session_index}")
    placement = str(metadata.get("placement") or "unknown")
    dataset_name = str(metadata.get("dataset_name") or "UNIFALLMONITOR")

    rows: list[dict[str, Any]] = []
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        try:
            ts = float(sample["timestamp"])
        except (KeyError, TypeError, ValueError):
            continue
        try:
            ax = float(sample.get("ax"))
            ay = float(sample.get("ay"))
            az = float(sample.get("az"))
        except (TypeError, ValueError):
            continue
        rows.append(
            {
                "timestamp": ts,
                "ax": ax,
                "ay": ay,
                "az": az,
                "gx": _opt_float(sample.get("gx")),
                "gy": _opt_float(sample.get("gy")),
                "gz": _opt_float(sample.get("gz")),
            }
        )

    if not rows:
        return None

    df = pd.DataFrame(rows).sort_values("timestamp", kind="stable").reset_index(drop=True)

    # Trim the first/last few seconds: those are the only places where the
    # phone's placement can change within a single recording (taking it out of
    # the pocket / putting it down). Keeping them confuses both the placement
    # and activity classifiers.
    start_ts = float(df["timestamp"].iloc[0])
    end_ts = float(df["timestamp"].iloc[-1])
    if (end_ts - start_ts) > (TRIM_HEAD_SECONDS + TRIM_TAIL_SECONDS + 1.0):
        df = df[
            (df["timestamp"] >= start_ts + TRIM_HEAD_SECONDS)
            & (df["timestamp"] <= end_ts - TRIM_TAIL_SECONDS)
        ].reset_index(drop=True)
        if df.empty:
            return None

    df["dataset_name"] = dataset_name
    df["subject_id"] = subject_id
    df["session_id"] = session_id
    df["source_file"] = f"unifallmonitor::{session_id}"
    df["task_type"] = "har"
    df["placement"] = placement
    df["label_raw"] = canonical_label
    df["label_mapped"] = canonical_label
    return df


def _opt_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


# ---- Windowing + feature pipeline --------------------------------------------


def build_features_for_dataframe(
    df: pd.DataFrame,
    *,
    target_rate_hz: float,
    window_size: int,
    step_size: int,
) -> pd.DataFrame:
    resampled = resample_dataframe(df, target_rate_hz=target_rate_hz)
    if resampled.empty:
        return pd.DataFrame()
    resampled = append_derived_channels(resampled)
    cfg = PreprocessConfig(target_sampling_rate_hz=target_rate_hz)
    windows = window_dataframe(
        resampled,
        window_size=window_size,
        step_size=step_size,
        config=cfg,
    )
    if not windows:
        return pd.DataFrame()
    feature_df = build_feature_table(
        windows,
        filter_unacceptable=True,
        default_sampling_rate_hz=target_rate_hz,
    )
    if feature_df.empty:
        return feature_df

    placement = df["placement"].iloc[0] if "placement" in df.columns else "unknown"
    feature_df["placement"] = placement
    return feature_df


# ---- Training / evaluation ----------------------------------------------------


def _stratified_split(
    feature_df: pd.DataFrame, *, test_size: float, random_state: int, label_col: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    from sklearn.model_selection import StratifiedShuffleSplit

    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    train_idx, test_idx = next(splitter.split(feature_df, feature_df[label_col]))
    return feature_df.iloc[train_idx].copy(), feature_df.iloc[test_idx].copy()


def split_train_test(
    feature_df: pd.DataFrame,
    *,
    test_size: float,
    random_state: int,
    label_col: str,
    mode: str = "auto",
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Return (train_df, test_df, mode_used).

    - ``mode="subject"`` forces leave-subjects-out via
      ``models.har.train_har.subject_aware_group_split``.
    - ``mode="stratified"`` forces a per-window stratified-by-label split.
    - ``mode="auto"`` (default) prefers a subject split, but falls back to
      stratified if the corpus is single-subject OR if the subject split would
      leave any train-set class unrepresented in test (which would make
      macro-F1 meaningless).
    """
    n_subjects = int(feature_df["subject_id"].nunique())

    if mode == "stratified" or (mode == "auto" and n_subjects < 2):
        train_df, test_df = _stratified_split(
            feature_df, test_size=test_size, random_state=random_state, label_col=label_col
        )
        return train_df, test_df, "stratified"

    train_df, test_df = subject_aware_group_split(
        feature_df,
        label_col=label_col,
        test_size=test_size,
        random_state=random_state,
    )
    if mode == "subject":
        return train_df, test_df, "subject"

    train_classes = set(train_df[label_col].unique())
    test_classes = set(test_df[label_col].unique())
    if train_classes - test_classes:
        logger.warning(
            "Subject split left classes %s out of the test set; falling back to "
            "stratified split. Re-run with --split-mode subject once more "
            "contributors have recorded labelled sessions.",
            sorted(train_classes - test_classes),
        )
        train_df, test_df = _stratified_split(
            feature_df, test_size=test_size, random_state=random_state, label_col=label_col
        )
        return train_df, test_df, "stratified_fallback"

    return train_df, test_df, "subject"


def train_and_evaluate(
    feature_df: pd.DataFrame,
    *,
    label_col: str,
    test_size: float,
    random_state: int,
    n_estimators: int,
    split_mode: str = "auto",
    calibrate: str = "none",
) -> dict[str, Any]:
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
    )

    feature_df = feature_df.dropna(subset=[label_col]).copy()
    if feature_df.empty:
        raise RuntimeError("After cleaning labels there are no rows left to train on.")

    train_df, test_df, split_used = split_train_test(
        feature_df,
        test_size=test_size,
        random_state=random_state,
        label_col=label_col,
        mode=split_mode,
    )

    # Drop feature columns that are entirely NaN in the training partition
    # (e.g. gyro features when gyro is absent across the training subjects).
    candidate_cols = select_feature_columns(train_df)
    all_nan_cols = [c for c in candidate_cols if train_df[c].isna().all()]
    if all_nan_cols:
        logger.info(
            "Dropping %d all-NaN feature columns: %s",
            len(all_nan_cols),
            all_nan_cols[:8],
        )
    surviving_cols = [c for c in candidate_cols if c not in all_nan_cols]
    if not surviving_cols:
        raise RuntimeError("Every feature column was NaN in train; cannot train.")

    # ``prepare_feature_matrices`` computes the median imputation on the
    # training split only and applies it to test — no leakage.
    X_train, X_test, y_train, y_test, feat_cols, _fill = prepare_feature_matrices(
        train_df, test_df, label_col=label_col, feature_cols=surviving_cols,
    )

    base_rf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        class_weight="balanced",
        n_jobs=-1,
    )

    calibrate = (calibrate or "none").lower()
    if calibrate not in {"none", "isotonic", "sigmoid"}:
        raise ValueError(f"Unknown calibrate={calibrate!r}")

    calibration_used = "none"
    if calibrate == "none" or len(X_train) < 50 or len(set(y_train)) < 2:
        # Fall back to no-calibration for tiny training sets where a 20%
        # holdout would leave too few rows for either fit. The metadata
        # records the actual policy applied.
        if calibrate != "none":
            logger.info(
                "Skipping calibration (%s): train size %d too small or only one class",
                calibrate, len(X_train),
            )
        base_rf.fit(X_train, y_train)
        model = base_rf
    else:
        # Subject-aware 80/20 calibration holdout. We avoid leaking the
        # held-out evaluation set (already removed via split_train_test),
        # but the calibration set must come from training subjects only —
        # otherwise calibration on test contaminates the test metrics.
        from sklearn.model_selection import StratifiedShuffleSplit

        calib_size = 0.2
        sss = StratifiedShuffleSplit(
            n_splits=1, test_size=calib_size, random_state=random_state
        )
        try:
            inner_idx, calib_idx = next(sss.split(X_train, y_train))
        except ValueError as exc:
            logger.info("StratifiedShuffleSplit failed (%s); training without calibration", exc)
            base_rf.fit(X_train, y_train)
            model = base_rf
        else:
            X_inner = X_train.iloc[inner_idx] if hasattr(X_train, "iloc") else X_train[inner_idx]
            y_inner = y_train.iloc[inner_idx] if hasattr(y_train, "iloc") else y_train[inner_idx]
            X_calib = X_train.iloc[calib_idx] if hasattr(X_train, "iloc") else X_train[calib_idx]
            y_calib = y_train.iloc[calib_idx] if hasattr(y_train, "iloc") else y_train[calib_idx]
            base_rf.fit(X_inner, y_inner)
            try:
                from sklearn.frozen import FrozenEstimator
                calibrated = CalibratedClassifierCV(
                    estimator=FrozenEstimator(base_rf), method=calibrate
                )
            except ImportError:
                calibrated = CalibratedClassifierCV(
                    estimator=base_rf, method=calibrate, cv="prefit"
                )
            calibrated.fit(X_calib, y_calib)
            model = calibrated
            calibration_used = calibrate
            logger.info(
                "Applied %s calibration on %d-row holdout (inner train=%d)",
                calibrate, len(X_calib), len(X_inner),
            )

    preds = model.predict(X_test)

    labels_present = sorted(set(y_train) | set(y_test))
    report = classification_report(
        y_test, preds, labels=labels_present, zero_division=0, output_dict=True
    )
    confusion = confusion_matrix(y_test, preds, labels=labels_present)

    return {
        "model": model,
        "feature_cols": feat_cols,
        "labels": labels_present,
        "split_mode": split_used,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_class_counts": y_train.value_counts().to_dict(),
        "test_class_counts": y_test.value_counts().to_dict(),
        "report": report,
        "confusion_matrix": confusion.tolist(),
        "accuracy": float(accuracy_score(y_test, preds)),
        "macro_f1": float(f1_score(y_test, preds, average="macro", zero_division=0)),
        "calibration": calibration_used,
    }


# ---- Main ---------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        default=os.environ.get("UNIFALL_BASE_URL", "https://api.unifallmonitor.com"),
        help="Backend base URL (env: UNIFALL_BASE_URL)",
    )
    parser.add_argument(
        "--username",
        default=os.environ.get("UNIFALL_USERNAME"),
        help="Basic-auth username (env: UNIFALL_USERNAME)",
    )
    parser.add_argument(
        "--password",
        default=os.environ.get("UNIFALL_PASSWORD"),
        help="Basic-auth password (env: UNIFALL_PASSWORD)",
    )
    parser.add_argument("--subject-id", default=None, help="Filter by subject_id")
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument("--max-sessions", type=int, default=None)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("artifacts/unifallmonitor/cache"),
        help="Cache directory for raw session payloads",
    )
    parser.add_argument(
        "--no-cache", action="store_true", help="Disable on-disk caching of raw payloads"
    )
    parser.add_argument(
        "--refresh-cache", action="store_true", help="Refetch sessions even if cached"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "Output directory for model + reports. When omitted, a fresh "
            "``artifacts/unifallmonitor/runs/<auto-id>/`` directory is "
            "created and used."
        ),
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("artifacts/unifallmonitor/runs"),
        help="Root directory for run-id-scoped output dirs (used when --out is omitted)",
    )
    parser.add_argument(
        "--mark-current",
        action="store_true",
        help=(
            "After training, point ``artifacts/unifallmonitor/current`` at "
            "this run so downstream consumers (experiments, deployable "
            "export) pick it up."
        ),
    )
    parser.add_argument("--target-rate-hz", type=float, default=50.0)
    parser.add_argument("--window-size", type=int, default=100, help="Samples per window (default placement)")
    parser.add_argument("--step-size", type=int, default=50, help="Stride between windows")
    parser.add_argument(
        "--pocket-window-size",
        type=int,
        default=200,
        help=(
            "Override window size (samples) for pocket-placement sessions. "
            "Default 200 (4 s @ 50 Hz) — pocket-walking gait has a longer "
            "characteristic period than hand-walking, so a 2 s window misses "
            "the full cycle and the classifier confuses it with stairs."
        ),
    )
    parser.add_argument(
        "--default-window-size",
        type=int,
        default=None,
        help="Override window size for non-pocket placements (defaults to --window-size)",
    )
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument(
        "--split-mode",
        choices=("auto", "subject", "stratified"),
        default="auto",
        help=(
            "Train/test split strategy. 'auto' (default) uses leave-subject-out "
            "but falls back to stratified-by-label if the subject split leaves "
            "classes unrepresented in test. 'subject' forces leave-subject-out. "
            "'stratified' splits at the window level."
        ),
    )
    parser.add_argument(
        "--collapse-static-to-other",
        action="store_true",
        help="Map 'static' to 'other' so the target is {fall, walking, stairs, other}",
    )
    parser.add_argument(
        "--drop-other",
        action="store_true",
        help=(
            "Exclude sessions whose canonical label is 'other' from training. "
            "'other' means we couldn't confidently assign one of the four "
            "target activities; including those sessions poisons the "
            "macro-F1 because a fitted model has nothing meaningful to learn "
            "about them. Note: this is applied AFTER --collapse-static-to-other, "
            "so static sessions are not dropped unless they were collapsed to "
            "other first."
        ),
    )
    parser.add_argument(
        "--calibrate",
        choices=("none", "isotonic", "sigmoid"),
        default="isotonic",
        help=(
            "Probability calibration applied to the trained classifier. "
            "Isotonic (default) holds out 20%% of the training set as a "
            "subject-aware calibration split, fits the RF on the remaining "
            "80%%, then wraps it in CalibratedClassifierCV(method='isotonic'). "
            "Necessary because RandomForest probabilities cluster near 0.5; "
            "without calibration the fall threshold sweep is uninformative."
        ),
    )
    parser.add_argument(
        "--per-placement",
        action="store_true",
        help="Also train one classifier per placement bucket",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only fetch + summarize sessions; do not train",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )


def collect_session_dataframes(
    client: RemoteClient,
    args: argparse.Namespace,
) -> tuple[list[pd.DataFrame], list[dict[str, Any]]]:
    cache_dir = None if args.no_cache else args.cache_dir
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    skipped_no_label = 0
    skipped_no_samples = 0
    skipped_unknown = 0

    started = time.monotonic()
    for index, item in enumerate(
        iter_sessions(
            client,
            subject_id=args.subject_id,
            page_size=args.page_size,
            max_sessions=args.max_sessions,
        )
    ):
        session = item.get("session") or {}
        app_session_id = session.get("app_session_id")
        if not app_session_id:
            continue

        raw_label = session.get("activity_label") or item.get("latest_annotation_label")
        canonical = canonicalize_activity_label(raw_label)
        if canonical is None:
            skipped_unknown += 1
            continue
        if args.collapse_static_to_other:
            canonical = collapse_to_four_class(canonical)
            if canonical is None:
                skipped_unknown += 1
                continue
        if getattr(args, "drop_other", False) and canonical == "other":
            skipped_unknown += 1
            continue

        try:
            payload = load_or_fetch_raw(
                client,
                app_session_id=str(app_session_id),
                cache_dir=cache_dir,
                refresh=args.refresh_cache,
            )
        except Exception as exc:  # noqa: BLE001 — log and skip per-session failures
            logger.warning("Failed to fetch raw payload for %s: %s", app_session_id, exc)
            continue

        request = payload.get("request") or {}
        metadata = request.get("metadata") or {}
        samples = request.get("samples") or []

        if not samples:
            skipped_no_samples += 1
            continue

        df = samples_to_dataframe(
            metadata=metadata,
            samples=samples,
            canonical_label=canonical,
            session_index=index,
        )
        if df is None or df.empty:
            skipped_no_samples += 1
            continue

        frames.append(df)
        summaries.append(
            {
                "app_session_id": str(app_session_id),
                "subject_id": df["subject_id"].iloc[0],
                "placement": df["placement"].iloc[0],
                "canonical_label": canonical,
                "raw_label": raw_label,
                "sample_count": int(len(df)),
                "duration_seconds": float(df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]),
            }
        )

    elapsed = time.monotonic() - started
    logger.info(
        "Collected %d sessions in %.1fs (skipped: no_label=%d, unknown=%d, no_samples=%d)",
        len(frames),
        elapsed,
        skipped_no_label,
        skipped_unknown,
        skipped_no_samples,
    )
    return frames, summaries


def write_outputs(
    *,
    out_dir: Path,
    feature_df: pd.DataFrame,
    summaries: list[dict[str, Any]],
    args: argparse.Namespace,
    overall_result: dict[str, Any],
    per_placement_results: dict[str, dict[str, Any]] | None,
    run_id: str = "",
    git_sha: str = "",
    content_sha: str = "",
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    feature_df.to_csv(out_dir / "feature_table.csv", index=False)
    pd.DataFrame(summaries).to_csv(out_dir / "session_summaries.csv", index=False)

    import joblib

    joblib.dump(
        {
            "model": overall_result["model"],
            "feature_cols": overall_result["feature_cols"],
            "labels": overall_result["labels"],
        },
        out_dir / "model.joblib",
    )

    confusion_df = pd.DataFrame(
        overall_result["confusion_matrix"],
        index=[f"true:{label}" for label in overall_result["labels"]],
        columns=[f"pred:{label}" for label in overall_result["labels"]],
    )
    confusion_df.to_csv(out_dir / "confusion_matrix.csv")

    metadata = {
        "run_id": run_id,
        "git_sha": git_sha,
        "content_sha": content_sha,
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()
                 if k not in {"username", "password"}},
        "data_manifest": sorted(s.get("app_session_id", "") for s in summaries),
        "session_count": len(summaries),
        "feature_row_count": int(len(feature_df)),
        "labels": overall_result["labels"],
        "feature_cols": overall_result["feature_cols"],
        "split_mode": overall_result["split_mode"],
        "calibration": overall_result.get("calibration", "none"),
        "train_rows": overall_result["train_rows"],
        "test_rows": overall_result["test_rows"],
        "train_class_counts": overall_result["train_class_counts"],
        "test_class_counts": overall_result["test_class_counts"],
        "accuracy": overall_result["accuracy"],
        "macro_f1": overall_result["macro_f1"],
        "report": overall_result["report"],
    }
    if per_placement_results is not None:
        metadata["per_placement"] = {
            placement: {
                "split_mode": result["split_mode"],
                "train_rows": result["train_rows"],
                "test_rows": result["test_rows"],
                "labels": result["labels"],
                "accuracy": result["accuracy"],
                "macro_f1": result["macro_f1"],
                "report": result["report"],
                "confusion_matrix": result["confusion_matrix"],
            }
            for placement, result in per_placement_results.items()
        }

    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)

    client = RemoteClient(
        base_url=args.base_url.rstrip("/"),
        auth_header=make_basic_auth_header(args.username, args.password),
    )

    # Reserve a run id up front. The data manifest hash is filled in below
    # once we know which sessions actually trained.
    run_identity = compute_run_id(args=vars(args), data_manifest=[])
    if args.out is None:
        args.out = args.runs_root / run_identity.run_id
    args.out.mkdir(parents=True, exist_ok=True)
    logger.info("Run id: %s (out: %s)", run_identity.run_id, args.out)

    frames, summaries = collect_session_dataframes(client, args)
    if not frames:
        logger.error("No labeled sessions were collected — nothing to train on.")
        return 2

    target_label = "label_mapped_majority"

    default_window = args.default_window_size or args.window_size

    def _window_size_for_placement(placement: str) -> int:
        # Pocket sessions are dominated by impact-style step transients and
        # the gait period is closer to 1.0 s than the hand-walking 0.5 s, so
        # a 2 s window can miss a full cycle. Hand/bag/desk keep the default.
        if str(placement).strip().lower() == "pocket":
            return int(args.pocket_window_size)
        return int(default_window)

    feature_chunks: list[pd.DataFrame] = []
    window_sizes_used: dict[str, int] = {}
    for df in frames:
        placement_for_session = (
            str(df["placement"].iloc[0]) if "placement" in df.columns else "unknown"
        )
        window_size = _window_size_for_placement(placement_for_session)
        # Adaptive fallback: short sessions can't fit the placement-default
        # window. Drop to the default size rather than producing no windows
        # — this keeps short pocket-fall recordings in the training set.
        n_samples = int(len(df))
        if n_samples < window_size:
            window_size = min(window_size, default_window, n_samples)
        if window_size < 16:
            logger.info(
                "Session %s too short (%d samples) for windowing; skipping",
                df["session_id"].iloc[0],
                n_samples,
            )
            continue
        # Step is half the window so 50% overlap is preserved across placements.
        step_size = max(1, window_size // 2)
        window_sizes_used[placement_for_session] = window_size
        try:
            chunk = build_features_for_dataframe(
                df,
                target_rate_hz=args.target_rate_hz,
                window_size=window_size,
                step_size=step_size,
            )
        except Exception as exc:  # noqa: BLE001 — keep going if one session fails
            logger.warning(
                "Feature extraction failed for session %s: %s",
                df["session_id"].iloc[0],
                exc,
            )
            continue
        if not chunk.empty:
            feature_chunks.append(chunk)

    if window_sizes_used:
        logger.info("Per-placement window sizes (samples): %s", window_sizes_used)

    if not feature_chunks:
        logger.error("No usable feature windows were produced — cannot train.")
        return 3

    feature_df = pd.concat(feature_chunks, ignore_index=True)
    logger.info(
        "Built feature table: %d windows, label distribution = %s",
        len(feature_df),
        feature_df[target_label].value_counts().to_dict(),
    )

    if args.dry_run:
        logger.info("--dry-run set; skipping training.")
        return 0

    overall = train_and_evaluate(
        feature_df,
        label_col=target_label,
        test_size=args.test_size,
        random_state=args.random_state,
        n_estimators=args.n_estimators,
        split_mode=args.split_mode,
        calibrate=args.calibrate,
    )
    logger.info(
        "Overall (%s split): accuracy=%.4f macro_f1=%.4f train=%d test=%d",
        overall["split_mode"],
        overall["accuracy"],
        overall["macro_f1"],
        overall["train_rows"],
        overall["test_rows"],
    )

    per_placement: dict[str, dict[str, Any]] | None = None
    if args.per_placement and "placement" in feature_df.columns:
        per_placement = {}
        for placement, group in feature_df.groupby("placement", dropna=False):
            placement_key = str(placement) if pd.notna(placement) else "unknown"
            if int(group[target_label].nunique()) < 2 or len(group) < 20:
                logger.info(
                    "Skipping per-placement training for %s (rows=%d, classes=%d)",
                    placement_key,
                    len(group),
                    int(group[target_label].nunique()),
                )
                continue
            try:
                per_placement[placement_key] = train_and_evaluate(
                    group.copy(),
                    label_col=target_label,
                    test_size=args.test_size,
                    random_state=args.random_state,
                    n_estimators=args.n_estimators,
                    split_mode=args.split_mode,
                    calibrate=args.calibrate,
                )
            except Exception as exc:  # noqa: BLE001 — placements with too little data
                logger.warning(
                    "Per-placement training failed for %s: %s", placement_key, exc
                )

    data_manifest = sorted(s.get("app_session_id", "") for s in summaries)
    content_sha = content_hash(
        args=vars(args), data_manifest=data_manifest, git_sha=run_identity.git_sha
    )

    write_outputs(
        out_dir=args.out,
        feature_df=feature_df,
        summaries=summaries,
        args=args,
        overall_result=overall,
        per_placement_results=per_placement,
        run_id=run_identity.run_id,
        git_sha=run_identity.git_sha,
        content_sha=content_sha,
    )

    register_run(
        run_id=run_identity.run_id,
        kind="train",
        out_dir=args.out,
        metrics={
            "accuracy": overall["accuracy"],
            "macro_f1": overall["macro_f1"],
        },
        n_sessions=len(summaries),
        git_sha=run_identity.git_sha,
        notes=f"calibrate={overall.get('calibration', 'none')} split={overall['split_mode']}",
        runs_root=args.runs_root,
    )

    if args.mark_current:
        link = update_current_symlink(runs_root=args.runs_root, run_id=run_identity.run_id)
        logger.info("Marked %s as current via %s", run_identity.run_id, link)

    logger.info("Wrote model + reports to %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
