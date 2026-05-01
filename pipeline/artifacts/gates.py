"""Promotion gates for fall + HAR artifacts.

Three pure-function checks consumed by ``scripts/organize_{fall,har}_artifacts.py``
right before a candidate is copied into ``current/``. Each gate returns a
:class:`GateResult` — a boolean pass/fail plus a list of human-readable
reasons. The organizer aggregates them, refuses the promotion when any gate
fails, and stamps the aggregate into the promoted metadata for auditing.

The gates are deliberately tolerant of missing fields: a candidate that lacks
stratified metrics or a truthset is treated as a pass with a ``skipped``
reason, so older artifacts continue to promote without manual intervention.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class GateResult:
    name: str
    ok: bool
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "ok": self.ok, "reasons": list(self.reasons)}


def _f2_from_metrics(metrics: dict[str, Any] | None) -> float | None:
    """Return F2 when the metrics dict has enough fields, else ``None``."""
    if not metrics:
        return None
    if "f2" in metrics and metrics["f2"] is not None:
        try:
            return float(metrics["f2"])
        except (TypeError, ValueError):
            return None
    # Fall-detection metrics sometimes store only counts; derive F2 from them.
    tp = metrics.get("tp")
    fp = metrics.get("fp")
    fn = metrics.get("fn")
    try:
        tp_f = float(tp) if tp is not None else None
        fp_f = float(fp) if fp is not None else None
        fn_f = float(fn) if fn is not None else None
    except (TypeError, ValueError):
        return None
    if None in (tp_f, fp_f, fn_f):
        return None
    denom = 5.0 * tp_f + 4.0 * fn_f + fp_f
    if denom <= 0:
        return None
    return (5.0 * tp_f) / denom


def _lookup_primary_metric(block: dict[str, Any] | None) -> float | None:
    """Return the most important scalar metric found in a metrics block.

    Order of preference: ``f2`` → ``f1`` → ``macro_f1`` → ``accuracy``. The
    first one that is present and coercible wins.
    """
    if not block:
        return None
    f2 = _f2_from_metrics(block)
    if f2 is not None:
        return f2
    for key in ("f1", "macro_f1", "accuracy"):
        val = block.get(key)
        if val is None:
            continue
        try:
            return float(val)
        except (TypeError, ValueError):
            continue
    return None


def heldout_no_regression(
    candidate_meta: dict[str, Any],
    current_meta: dict[str, Any] | None,
    *,
    tolerance: float = 0.01,
) -> GateResult:
    """Reject if the candidate's public-heldout primary metric drops past tolerance."""
    if not current_meta:
        return GateResult("heldout_no_regression", True, ["no current artifact to compare against — pass"])
    cand = _lookup_primary_metric(candidate_meta.get("heldout"))
    curr = _lookup_primary_metric(current_meta.get("heldout"))
    if cand is None or curr is None:
        return GateResult(
            "heldout_no_regression",
            True,
            [f"missing heldout metric (candidate={cand}, current={curr}) — skipped"],
        )
    delta = cand - curr
    ok = delta >= -tolerance
    reasons = [
        f"candidate heldout primary metric = {cand:.4f}",
        f"current heldout primary metric   = {curr:.4f}",
        f"delta = {delta:+.4f} (tolerance = {tolerance:.4f})",
    ]
    return GateResult("heldout_no_regression", ok, reasons)


def no_placement_regression(
    candidate_meta: dict[str, Any],
    current_meta: dict[str, Any] | None,
    *,
    tolerance: float = 0.02,
) -> GateResult:
    """Reject if any stratified-placement primary metric drops past tolerance."""
    if not current_meta:
        return GateResult("no_placement_regression", True, ["no current artifact — pass"])
    cand_by = (candidate_meta.get("heldout") or {}).get("by_placement") or {}
    curr_by = (current_meta.get("heldout") or {}).get("by_placement") or {}
    if not cand_by and not curr_by:
        return GateResult(
            "no_placement_regression",
            True,
            ["neither artifact has by_placement metrics — skipped"],
        )
    reasons: list[str] = []
    ok = True
    for placement in sorted(set(cand_by) & set(curr_by)):
        cand = _lookup_primary_metric(cand_by[placement])
        curr = _lookup_primary_metric(curr_by[placement])
        if cand is None or curr is None:
            reasons.append(f"{placement}: missing metric — skipped")
            continue
        delta = cand - curr
        if delta < -tolerance:
            ok = False
            reasons.append(f"{placement}: {delta:+.4f} (candidate={cand:.4f}, current={curr:.4f}) — REGRESSION")
        else:
            reasons.append(f"{placement}: {delta:+.4f} (candidate={cand:.4f}, current={curr:.4f})")
    return GateResult("no_placement_regression", ok, reasons)


def phone_truthset_not_worse(
    candidate_meta: dict[str, Any],
    current_meta: dict[str, Any] | None,
    *,
    min_f2: float | None = None,
    max_fp_per_hour: float | None = None,
    tolerance: float = 0.01,
) -> GateResult:
    """Reject if the candidate regresses on the curated phone truthset.

    The actual truthset eval is produced upstream (by the diagnose stage or
    by the organizer invoking the diagnostic helper against
    ``artifacts/phone_truthset/``) and must be stamped into the candidate /
    current metadata under ``phone_truthset``. This gate only compares the
    two blobs — it does not re-run evaluation.
    """
    cand_set = candidate_meta.get("phone_truthset")
    if not cand_set:
        return GateResult(
            "phone_truthset_not_worse",
            True,
            ["candidate has no phone_truthset metrics — skipped"],
        )
    reasons: list[str] = []
    ok = True
    cand_f2 = _lookup_primary_metric(cand_set)
    if min_f2 is not None and cand_f2 is not None and cand_f2 < min_f2:
        ok = False
        reasons.append(f"candidate F2 on truthset = {cand_f2:.4f} < floor {min_f2:.4f}")

    cand_fph = cand_set.get("false_alerts_per_hour")
    if max_fp_per_hour is not None and cand_fph is not None:
        try:
            if float(cand_fph) > max_fp_per_hour:
                ok = False
                reasons.append(
                    f"candidate false_alerts_per_hour = {float(cand_fph):.3f} > ceiling {max_fp_per_hour:.3f}"
                )
        except (TypeError, ValueError):
            pass

    curr_set = (current_meta or {}).get("phone_truthset")
    curr_f2 = _lookup_primary_metric(curr_set)
    if cand_f2 is not None and curr_f2 is not None:
        delta = cand_f2 - curr_f2
        if delta < -tolerance:
            ok = False
            reasons.append(
                f"truthset F2 regressed: candidate={cand_f2:.4f} current={curr_f2:.4f} delta={delta:+.4f}"
            )
        else:
            reasons.append(
                f"truthset F2: candidate={cand_f2:.4f} current={curr_f2:.4f} delta={delta:+.4f}"
            )

    if not reasons:
        reasons.append("no numeric regressions against truthset")
    return GateResult("phone_truthset_not_worse", ok, reasons)


def run_gates(
    candidate_meta: dict[str, Any],
    current_meta: dict[str, Any] | None,
    *,
    heldout_tolerance: float = 0.01,
    placement_tolerance: float = 0.02,
    truthset_min_f2: float | None = None,
    truthset_max_fp_per_hour: float | None = None,
    truthset_tolerance: float = 0.01,
) -> list[GateResult]:
    """Run every promotion gate and return their individual results."""
    return [
        heldout_no_regression(candidate_meta, current_meta, tolerance=heldout_tolerance),
        no_placement_regression(candidate_meta, current_meta, tolerance=placement_tolerance),
        phone_truthset_not_worse(
            candidate_meta,
            current_meta,
            min_f2=truthset_min_f2,
            max_fp_per_hour=truthset_max_fp_per_hour,
            tolerance=truthset_tolerance,
        ),
    ]


def gates_passed(results: list[GateResult]) -> bool:
    return all(r.ok for r in results)