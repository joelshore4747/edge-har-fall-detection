"""One-shot experiments on the cached unifallmonitor sessions.

Reads the same /v1/sessions/{id}/raw payloads cached by
``run_train_unifallmonitor_har.sh`` (under
``artifacts/unifallmonitor/cache``), reuses the existing windowing + feature
pipeline, and reports honest leave-one-session-out results for three different
framings of the same data:

    A) Placement classifier (pocket / hand / desk / bag)
    B) Fall *event* detection — flag a session as "fall" if any window has
       P(fall) > threshold (per-window peak, not session-majority vote).
    C) Five-class activity classifier (static kept separate from other).
    D) Synthetic placement-transition stitching probe.

Outputs are written into the same run directory as the matching train run
(``artifacts/unifallmonitor/runs/<run_id>/experiments/``) so a snapshot of
"what the trained model achieves on this dataset" is colocated. The
``--out-run`` flag picks a specific run id; by default the experiments use
the run that ``current`` points at.

Run after the training cache is populated:

    .venv/bin/python scripts/experiments_unifallmonitor.py
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.features import build_feature_table  # noqa: E402
from pipeline.preprocess import (  # noqa: E402
    PreprocessConfig,
    append_derived_channels,
    resample_dataframe,
    window_dataframe,
)

from models.har.train_har import select_feature_columns  # noqa: E402

from scripts.lib.run_registry import (  # noqa: E402
    compute_run_id,
    register_run,
    resolve_current_run,
)
from scripts.lib.smoothing import smooth_probs  # noqa: E402
from scripts.train_unifallmonitor_har import (  # noqa: E402
    canonicalize_activity_label,
    collapse_to_four_class,
    samples_to_dataframe,
)

logger = logging.getLogger("experiments_unifallmonitor")

DEFAULT_CACHE_DIR = REPO_ROOT / "artifacts" / "unifallmonitor" / "cache"
DEFAULT_RUNS_ROOT = REPO_ROOT / "artifacts" / "unifallmonitor" / "runs"


def canonical_activity(raw: str | None, *, collapse_static_to_other: bool) -> str | None:
    """Wrapper around the trainer's canonicalizer with optional static→other collapse."""
    label = canonicalize_activity_label(raw)
    if collapse_static_to_other:
        return collapse_to_four_class(label)
    return label


def load_session_frames(
    *,
    collapse_static: bool,
    cache_dir: Path,
    drop_other: bool = False,
) -> tuple[list[pd.DataFrame], list[dict[str, Any]]]:
    frames: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    for path in sorted(cache_dir.glob("*.json")):
        payload = json.loads(path.read_text())
        request = payload.get("request") or {}
        metadata = request.get("metadata") or {}
        samples = request.get("samples") or []
        canonical = canonical_activity(metadata.get("activity_label"), collapse_static_to_other=collapse_static)
        if canonical is None or not samples:
            continue
        if drop_other and canonical == "other":
            continue
        # ``samples_to_dataframe`` already trims the first/last few seconds of
        # each recording (the only places where the phone's placement can
        # change within a session), so we don't repeat that here.
        df = samples_to_dataframe(
            metadata=metadata, samples=samples, canonical_label=canonical, session_index=len(frames),
        )
        if df is None or df.empty:
            continue

        # Tag the session-level placement on the per-sample frame so it survives
        # into the feature table.
        df["session_placement"] = metadata.get("placement") or "unknown"
        df["session_name"] = metadata.get("session_name") or metadata.get("session_id")
        frames.append(df)
        summaries.append({
            "session_id": df["session_id"].iloc[0],
            "session_name": df["session_name"].iloc[0],
            "subject_id": df["subject_id"].iloc[0],
            "placement": df["session_placement"].iloc[0],
            "activity": canonical,
            "samples": int(len(df)),
        })
    return frames, summaries


def _window_size_for_placement(placement: str, *, default_window: int, pocket_window: int) -> int:
    if str(placement).strip().lower() == "pocket":
        return int(pocket_window)
    return int(default_window)


def build_features(
    frames: list[pd.DataFrame],
    *,
    default_window: int = 100,
    pocket_window: int = 200,
) -> pd.DataFrame:
    chunks: list[pd.DataFrame] = []
    for df in frames:
        resampled = resample_dataframe(df, target_rate_hz=50.0)
        if resampled.empty:
            continue
        resampled = append_derived_channels(resampled)
        cfg = PreprocessConfig(target_sampling_rate_hz=50.0)
        placement = (
            str(df["session_placement"].iloc[0])
            if "session_placement" in df.columns
            else "unknown"
        )
        win = _window_size_for_placement(
            placement, default_window=default_window, pocket_window=pocket_window
        )
        # Adaptive fallback: short sessions (e.g. pocket-fall recordings that
        # are 6 s total and become 2 s after trimming) cannot fit a 4 s
        # window. Falling back to the default lets us keep the session in
        # the evaluation rather than dropping it. Without this, peak fall-F1
        # regresses because the rare fall events disappear.
        if len(resampled) < win:
            win = min(win, default_window, len(resampled))
        if win < 16:
            # Not enough data to extract anything meaningful.
            continue
        step = max(1, win // 2)
        windows = window_dataframe(resampled, window_size=win, step_size=step, config=cfg)
        if not windows:
            continue
        chunk = build_feature_table(windows, filter_unacceptable=True, default_sampling_rate_hz=50.0)
        if chunk.empty:
            continue
        # propagate session-level fields onto every window row
        chunk["placement"] = placement
        chunk["session_name"] = df["session_name"].iloc[0]
        chunks.append(chunk)
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


def clean_features(feature_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    # ``select_feature_columns`` keeps numeric/bool columns only and excludes
    # the canonical metadata set, so string columns like ``session_name`` and
    # ``placement`` are dropped automatically. We still re-filter just in case
    # a future addition slips through.
    feat_cols = [c for c in select_feature_columns(feature_df) if c not in {"session_name"}]
    feature_df[feat_cols] = feature_df[feat_cols].apply(pd.to_numeric, errors="coerce")
    all_nan = [c for c in feat_cols if feature_df[c].isna().all()]
    feat_cols = [c for c in feat_cols if c not in all_nan]
    medians = feature_df[feat_cols].median(numeric_only=True)
    feature_df[feat_cols] = feature_df[feat_cols].fillna(medians).fillna(0.0)
    return feature_df, feat_cols


def loso_session_predictions(
    feature_df: pd.DataFrame,
    feat_cols: list[str],
    target: str,
    *,
    smoothing: str = "none",
    smoothing_window: int = 5,
) -> pd.DataFrame:
    """Train one RF per held-out session, predict that session's windows.

    Emits both the raw majority vote (``predicted``) and a smoothed
    majority vote (``predicted_smoothed``) computed by averaging per-window
    class probabilities over a temporal window before taking ``argmax``.
    A single noisy window can flip the raw majority on short sessions; the
    smoothed vote is robust to that.
    """
    from sklearn.ensemble import RandomForestClassifier

    sessions = list(feature_df["session_id"].dropna().unique())
    rows: list[dict[str, Any]] = []
    for sid in sessions:
        test_mask = feature_df["session_id"] == sid
        train = feature_df.loc[~test_mask]
        test = feature_df.loc[test_mask].sort_values("midpoint_ts")
        if test.empty or train[target].nunique() < 2:
            continue
        model = RandomForestClassifier(
            n_estimators=200, random_state=42, class_weight="balanced", n_jobs=-1,
        )
        model.fit(train[feat_cols], train[target])
        preds = model.predict(test[feat_cols])
        majority = pd.Series(preds).value_counts().idxmax()

        proba = model.predict_proba(test[feat_cols])
        smoothed = smooth_probs(proba, mode=smoothing, window=smoothing_window)
        smoothed_majority_label = model.classes_[int(smoothed.sum(axis=0).argmax())]

        actual = test[target].iloc[0]
        rows.append({
            "session_id": sid,
            "session_name": test["session_name"].iloc[0],
            "subject": test["subject_id"].iloc[0],
            "placement": test["placement"].iloc[0],
            "actual": actual,
            "predicted": majority,
            "predicted_smoothed": smoothed_majority_label,
            "windows": int(len(test)),
            "agree": int(majority == actual),
            "agree_smoothed": int(smoothed_majority_label == actual),
        })
    return pd.DataFrame(rows)


def loso_fall_event_detection(
    feature_df: pd.DataFrame,
    feat_cols: list[str],
    *,
    smoothing: str = "rolling_mean",
    smoothing_window: int = 5,
) -> pd.DataFrame:
    """For each held-out session, return the max P(fall) across its windows.

    Emits two scores per session:

    - ``max_p_fall`` / ``mean_p_fall`` — the raw per-window probability peak
      (and mean), computed directly from ``model.predict_proba`` without
      smoothing.
    - ``smoothed_max_p_fall`` / ``smoothed_mean_p_fall`` — the same after
      applying a centered rolling-mean (default) over ``smoothing_window``
      windows. This kills lone false-positive spikes — e.g. a single noisy
      window in a 2-minute walking session — that otherwise cap the
      precision-recall curve at the maximum non-fall peak.
    """
    from sklearn.ensemble import RandomForestClassifier

    fall_target = (feature_df["label_mapped_majority"] == "fall").astype(int)
    feature_df = feature_df.assign(_y_fall=fall_target)

    sessions = list(feature_df["session_id"].dropna().unique())
    rows: list[dict[str, Any]] = []
    for sid in sessions:
        test_mask = feature_df["session_id"] == sid
        train = feature_df.loc[~test_mask]
        test = feature_df.loc[test_mask].sort_values("midpoint_ts")
        if test.empty or train["_y_fall"].nunique() < 2:
            continue
        model = RandomForestClassifier(
            n_estimators=400, random_state=42, class_weight="balanced", n_jobs=-1,
        )
        model.fit(train[feat_cols], train["_y_fall"])
        proba = model.predict_proba(test[feat_cols])
        # Find the column for class 1
        if 1 in model.classes_:
            col = list(model.classes_).index(1)
            fall_probs = proba[:, col]
        else:
            fall_probs = np.zeros(len(test))
        smoothed = smooth_probs(fall_probs, mode=smoothing, window=smoothing_window)
        rows.append({
            "session_id": sid,
            "session_name": test["session_name"].iloc[0],
            "subject": test["subject_id"].iloc[0],
            "placement": test["placement"].iloc[0],
            "is_fall_session": int(test["label_mapped_majority"].iloc[0] == "fall"),
            "max_p_fall": float(fall_probs.max()),
            "mean_p_fall": float(fall_probs.mean()),
            "smoothed_max_p_fall": float(smoothed.max()),
            "smoothed_mean_p_fall": float(smoothed.mean()),
            "windows": int(len(test)),
        })
    return pd.DataFrame(rows)


def sweep_threshold(df: pd.DataFrame, score_col: str, label_col: str) -> pd.DataFrame:
    rows = []
    for thr in np.arange(0.05, 0.96, 0.05):
        pred = (df[score_col] >= thr).astype(int)
        tp = int(((pred == 1) & (df[label_col] == 1)).sum())
        fp = int(((pred == 1) & (df[label_col] == 0)).sum())
        fn = int(((pred == 0) & (df[label_col] == 1)).sum())
        tn = int(((pred == 0) & (df[label_col] == 0)).sum())
        precision = tp / (tp + fp) if (tp + fp) else float("nan")
        recall = tp / (tp + fn) if (tp + fn) else float("nan")
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision and recall and not np.isnan(precision) and not np.isnan(recall))
            else float("nan")
        )
        rows.append({
            "threshold": round(float(thr), 3),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(precision, 3) if not np.isnan(precision) else None,
            "recall": round(recall, 3) if not np.isnan(recall) else None,
            "f1": round(f1, 3) if not np.isnan(f1) else None,
        })
    return pd.DataFrame(rows)


def run_experiment(
    *,
    name: str,
    target_col: str,
    collapse_static: bool,
    out_prefix: str,
    out_dir: Path,
    cache_dir: Path,
    smoothing: str = "none",
    smoothing_window: int = 5,
    drop_other: bool = False,
    default_window: int = 100,
    pocket_window: int = 200,
) -> pd.DataFrame:
    print(f"\n=== Experiment {name} ===")
    frames, summaries = load_session_frames(
        collapse_static=collapse_static, cache_dir=cache_dir, drop_other=drop_other,
    )
    print(f"  sessions used: {len(frames)}")
    if not frames:
        print("  (no eligible sessions for this experiment)")
        return pd.DataFrame()

    feature_df = build_features(
        frames, default_window=default_window, pocket_window=pocket_window
    )
    if feature_df.empty:
        print("  (no acceptable feature windows)")
        return pd.DataFrame()

    feature_df, feat_cols = clean_features(feature_df)
    print(f"  windows: {len(feature_df)}  features: {len(feat_cols)}")
    print(f"  target distribution: {feature_df[target_col].value_counts().to_dict()}")

    pred_df = loso_session_predictions(
        feature_df, feat_cols, target_col,
        smoothing=smoothing, smoothing_window=smoothing_window,
    )
    out_path = out_dir / f"{out_prefix}_per_session.csv"
    pred_df.to_csv(out_path, index=False)

    if not pred_df.empty:
        acc_raw = pred_df["agree"].mean()
        print(f"  LOSO session-level accuracy (raw): {acc_raw:.3f}  ({pred_df['agree'].sum()}/{len(pred_df)})")
        if "agree_smoothed" in pred_df.columns:
            acc_sm = pred_df["agree_smoothed"].mean()
            print(f"  LOSO session-level accuracy (smoothed): {acc_sm:.3f}  ({pred_df['agree_smoothed'].sum()}/{len(pred_df)})")
        confusion = pd.crosstab(pred_df["actual"], pred_df["predicted"], dropna=False).fillna(0).astype(int)
        print("  per-session confusion (raw, actual vs predicted):")
        for line in confusion.to_string().splitlines():
            print(f"    {line}")
    print(f"  wrote {out_path.relative_to(REPO_ROOT)}")
    return pred_df


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(os.environ.get("UNIFALL_CACHE_DIR", str(DEFAULT_CACHE_DIR))),
    )
    p.add_argument(
        "--runs-root",
        type=Path,
        default=DEFAULT_RUNS_ROOT,
        help="Where run-scoped output dirs live",
    )
    p.add_argument(
        "--out-run",
        default=None,
        help=(
            "Run id whose ``experiments/`` subdir receives the outputs. "
            "Defaults to the run that ``current`` points at; if no current "
            "exists a fresh run id is minted."
        ),
    )
    p.add_argument(
        "--smoothing",
        choices=("none", "rolling_mean", "hmm"),
        default="rolling_mean",
        help=(
            "Per-window probability smoothing applied before per-session "
            "aggregation. The threshold sweep CSV emits both raw and smoothed "
            "columns regardless."
        ),
    )
    p.add_argument(
        "--smoothing-window",
        type=int,
        default=5,
        help="Smoothing window length in windows (default 5 = ~10 s @ 50% overlap of 2 s)",
    )
    p.add_argument(
        "--drop-other",
        action="store_true",
        help="Exclude sessions whose canonical activity is 'other' from the activity experiments",
    )
    p.add_argument(
        "--default-window-size",
        type=int,
        default=100,
        help="Window size (samples) for non-pocket placements",
    )
    p.add_argument(
        "--pocket-window-size",
        type=int,
        default=200,
        help="Window size (samples) for pocket-placement sessions",
    )
    return p.parse_args()


def _resolve_out_dir(args: argparse.Namespace) -> tuple[Path, str]:
    """Return ``(out_dir, run_id)`` where ``out_dir`` is run/<id>/experiments/."""
    runs_root = args.runs_root.resolve()
    run_id = args.out_run
    if not run_id:
        current = resolve_current_run(runs_root=runs_root)
        if current is not None:
            run_id = current.name
    if not run_id:
        # No current and no override — mint a fresh one. The trainer would
        # normally have done this; fall back so the experiments script still
        # produces output even without a prior train.
        run_id = compute_run_id(args=vars(args), data_manifest=[]).run_id
    out_dir = runs_root / run_id / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir, run_id


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s :: %(message)s")
    args = parse_args()

    if not any(args.cache_dir.glob("*.json")):
        print(f"No cached sessions found at {args.cache_dir}. Run scripts/run_train_unifallmonitor_har.sh first.")
        return 2

    out_dir, run_id = _resolve_out_dir(args)
    logger.info("Writing experiments outputs into %s (run_id=%s)", out_dir, run_id)

    # ---- Experiment A: placement classifier ---------------------------------
    run_experiment(
        name="A: placement classifier (LOSO)",
        target_col="placement",
        collapse_static=True,
        out_prefix="A_placement",
        out_dir=out_dir,
        cache_dir=args.cache_dir,
        smoothing=args.smoothing,
        smoothing_window=args.smoothing_window,
        default_window=args.default_window_size,
        pocket_window=args.pocket_window_size,
    )

    # ---- Experiment B: fall *event* detection -------------------------------
    print("\n=== Experiment B: fall-event detection (per-window peak) ===")
    frames, _ = load_session_frames(collapse_static=True, cache_dir=args.cache_dir)
    feature_df = build_features(
        frames,
        default_window=args.default_window_size,
        pocket_window=args.pocket_window_size,
    )
    feature_df, feat_cols = clean_features(feature_df)
    fall_df = loso_fall_event_detection(
        feature_df, feat_cols,
        smoothing=args.smoothing, smoothing_window=args.smoothing_window,
    )
    fall_df.to_csv(out_dir / "B_fall_event_per_session.csv", index=False)
    print(f"  sessions: {len(fall_df)}  fall sessions: {int(fall_df['is_fall_session'].sum())}")

    # Two threshold sweeps: raw and smoothed. The smoothed curve is the
    # one to report — raw is kept so the two are visibly comparable.
    sweep_raw = sweep_threshold(fall_df, score_col="max_p_fall", label_col="is_fall_session")
    sweep_raw.insert(0, "score", "max_p_fall")
    sweep_smoothed = sweep_threshold(fall_df, score_col="smoothed_max_p_fall", label_col="is_fall_session")
    sweep_smoothed.insert(0, "score", "smoothed_max_p_fall")
    sweep = pd.concat([sweep_raw, sweep_smoothed], ignore_index=True)
    sweep.to_csv(out_dir / "B_fall_threshold_sweep.csv", index=False)

    print("  threshold sweep (raw max-P(fall)):")
    print(sweep_raw.to_string(index=False))
    print("\n  threshold sweep (smoothed max-P(fall)):")
    print(sweep_smoothed.to_string(index=False))

    fall_f1_at_0p5 = float(sweep_smoothed.loc[sweep_smoothed["threshold"] == 0.5, "f1"].iloc[0]) \
        if not sweep_smoothed.empty and 0.5 in set(sweep_smoothed["threshold"]) else float("nan")

    # show the per-session ranked scores (smoothed)
    ranked = fall_df.sort_values("smoothed_max_p_fall", ascending=False)[
        ["session_name", "subject", "placement", "is_fall_session",
         "max_p_fall", "smoothed_max_p_fall", "windows"]
    ]
    print("\n  per-session smoothed max P(fall), highest first:")
    print(ranked.to_string(index=False))

    # ---- Experiment C: 5-class activity (static separate) -------------------
    run_experiment(
        name="C: 5-class activity (static separate)",
        target_col="label_mapped_majority",
        collapse_static=False,
        out_prefix="C_activity_5class",
        out_dir=out_dir,
        cache_dir=args.cache_dir,
        smoothing=args.smoothing,
        smoothing_window=args.smoothing_window,
        drop_other=args.drop_other,
        default_window=args.default_window_size,
        pocket_window=args.pocket_window_size,
    )

    # ---- Experiment D: synthetic placement-transition probe -----------------
    print("\n=== Experiment D: simulated placement transition ===")
    simulate_placement_transition(
        out_dir=out_dir,
        cache_dir=args.cache_dir,
        default_window=args.default_window_size,
        pocket_window=args.pocket_window_size,
    )

    register_run(
        run_id=run_id,
        kind="experiments",
        out_dir=out_dir.parent,  # the run dir, not experiments/
        metrics={
            "fall_f1_at_0p5": fall_f1_at_0p5,
        },
        n_sessions=int(len(fall_df)),
        git_sha="",
        notes=f"smoothing={args.smoothing}/w{args.smoothing_window} drop_other={args.drop_other}",
        runs_root=args.runs_root,
    )

    print(f"\nAll outputs under {out_dir.relative_to(REPO_ROOT)}")
    return 0


def simulate_placement_transition(
    *,
    out_dir: Path,
    cache_dir: Path,
    default_window: int = 100,
    pocket_window: int = 200,
) -> None:
    """Concatenate same-activity sessions with different placements and
    measure whether a placement classifier trained on the rest of the data
    flips its prediction at the join.

    Picks the most populous activity ('stairs') for joelshore12_9ikh4x with
    one pocket and one hand session, stitches them in time, then sweeps a
    sliding window through the per-window placement predictions to see how
    quickly the classifier transitions.
    """
    from sklearn.ensemble import RandomForestClassifier

    frames, summaries = load_session_frames(collapse_static=True, cache_dir=cache_dir)
    summary_df = pd.DataFrame(summaries)

    # Find two sessions: same subject, same activity, different placements,
    # both reasonably long.
    candidates = (
        summary_df[summary_df["subject_id"] == "joelshore12_9ikh4x"]
        .sort_values("samples", ascending=False)
    )
    pocket = candidates[(candidates["activity"] == "stairs") & (candidates["placement"] == "pocket")].head(1)
    hand = candidates[(candidates["activity"] == "stairs") & (candidates["placement"] == "hand")].head(1)
    if pocket.empty or hand.empty:
        print("  could not find a pocket+hand stairs pair for joelshore12_9ikh4x; skipping")
        return

    pocket_sid = pocket["session_id"].iloc[0]
    hand_sid = hand["session_id"].iloc[0]
    print(f"  stitching pocket session {pocket['session_name'].iloc[0]} → hand session {hand['session_name'].iloc[0]}")

    pocket_frame = next(f for f in frames if f["session_id"].iloc[0] == pocket_sid).copy()
    hand_frame = next(f for f in frames if f["session_id"].iloc[0] == hand_sid).copy()

    # Re-base hand timestamps to come immediately after pocket ones.
    pocket_end = pocket_frame["timestamp"].iloc[-1]
    hand_frame["timestamp"] = hand_frame["timestamp"] - hand_frame["timestamp"].iloc[0] + pocket_end + 0.02

    stitched = pd.concat([pocket_frame, hand_frame], ignore_index=True)
    stitched["session_id"] = "stitched_session"
    stitched["session_placement"] = "stitched"

    # Build features for the stitched session
    stitched_features = build_features(
        [stitched], default_window=default_window, pocket_window=pocket_window
    )
    if stitched_features.empty:
        print("  stitched feature table empty; skipping")
        return

    # Train a placement classifier on every OTHER session (exclude the two
    # constituent sessions to avoid leakage).
    train_frames = [
        f for f in frames
        if f["session_id"].iloc[0] not in {pocket_sid, hand_sid}
    ]
    train_features = build_features(
        train_frames, default_window=default_window, pocket_window=pocket_window
    )
    train_features, feat_cols = clean_features(train_features)

    # Drop placements with too little data (e.g. unknown).
    train_features = train_features[train_features["placement"].isin(["pocket", "hand", "desk"])]
    if train_features["placement"].nunique() < 2:
        print("  not enough placement diversity in training data; skipping")
        return

    model = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced", n_jobs=-1)
    model.fit(train_features[feat_cols], train_features["placement"])

    # Predict on the stitched session — order windows by midpoint timestamp.
    # `clean_features` will impute NaNs, but we need to mirror that for the test set.
    test_features = stitched_features.copy()
    test_features[feat_cols] = test_features[feat_cols].apply(pd.to_numeric, errors="coerce")
    medians = train_features[feat_cols].median(numeric_only=True)
    test_features[feat_cols] = test_features[feat_cols].fillna(medians).fillna(0.0)

    test_features = test_features.sort_values("midpoint_ts").reset_index(drop=True)
    test_features["pred"] = model.predict(test_features[feat_cols])

    # The join is at pocket_end seconds; mark windows with where they fall.
    test_features["true_placement"] = test_features["midpoint_ts"].apply(
        lambda ts: "pocket" if ts < pocket_end else "hand"
    )
    n_total = len(test_features)
    n_correct = int((test_features["pred"] == test_features["true_placement"]).sum())
    print(f"  stitched windows: {n_total}, correct: {n_correct} ({n_correct/n_total:.1%})")

    # Show every window in time order
    print("  window timeline (midpoint→true→pred):")
    for _, row in test_features.iterrows():
        marker = "✓" if row["pred"] == row["true_placement"] else "✗"
        print(f"    t={row['midpoint_ts']:7.2f}s   true={row['true_placement']:<6}  pred={row['pred']:<6}  {marker}")

    out_path = out_dir / "D_placement_transition.csv"
    test_features[["midpoint_ts","true_placement","pred"]].to_csv(out_path, index=False)
    print(f"  wrote {out_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    raise SystemExit(main())
