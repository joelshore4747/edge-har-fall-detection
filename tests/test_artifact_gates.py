"""Unit tests for promotion gates.

Each test builds a candidate + current metadata pair by hand and asserts the
gate returns the expected ok/fail plus a reason string the operator can act
on. Gates that can't make a judgement must return ``ok=True`` with a
``skipped`` note — older artifacts that lack stratification should continue to
promote.
"""

from __future__ import annotations

from pipeline.artifacts.gates import (
    GateResult,
    gates_passed,
    heldout_no_regression,
    no_placement_regression,
    phone_truthset_not_worse,
    run_gates,
)


def test_heldout_no_regression_passes_on_identical_metrics():
    cand = {"heldout": {"f2": 0.75}}
    curr = {"heldout": {"f2": 0.75}}
    result = heldout_no_regression(cand, curr)
    assert result.ok
    assert result.name == "heldout_no_regression"


def test_heldout_no_regression_fails_on_big_drop():
    cand = {"heldout": {"f2": 0.60}}
    curr = {"heldout": {"f2": 0.75}}
    result = heldout_no_regression(cand, curr, tolerance=0.01)
    assert not result.ok
    assert any("delta" in r for r in result.reasons)


def test_heldout_no_regression_accepts_within_tolerance():
    cand = {"heldout": {"f2": 0.745}}
    curr = {"heldout": {"f2": 0.750}}
    result = heldout_no_regression(cand, curr, tolerance=0.01)
    assert result.ok


def test_heldout_no_regression_handles_missing_current():
    cand = {"heldout": {"f2": 0.60}}
    result = heldout_no_regression(cand, None)
    assert result.ok
    assert any("no current artifact" in r for r in result.reasons)


def test_heldout_no_regression_handles_missing_metrics():
    cand = {"heldout": {}}
    curr = {"heldout": {}}
    result = heldout_no_regression(cand, curr)
    assert result.ok
    assert any("missing heldout metric" in r or "skipped" in r for r in result.reasons)


def test_heldout_no_regression_derives_f2_from_counts():
    cand = {"heldout": {"tp": 70, "fp": 10, "fn": 30}}
    curr = {"heldout": {"tp": 75, "fp": 5, "fn": 25}}
    result = heldout_no_regression(cand, curr, tolerance=0.01)
    # Candidate F2 = 5*70 / (5*70 + 4*30 + 10) = 350/480 = 0.729
    # Current F2   = 5*75 / (5*75 + 4*25 + 5)  = 375/480 = 0.781
    assert not result.ok  # 0.729 < 0.781 - 0.01


def test_no_placement_regression_fails_when_any_placement_drops():
    cand = {"heldout": {"by_placement": {"pocket": {"f2": 0.70}, "hand": {"f2": 0.60}}}}
    curr = {"heldout": {"by_placement": {"pocket": {"f2": 0.70}, "hand": {"f2": 0.75}}}}
    result = no_placement_regression(cand, curr, tolerance=0.02)
    assert not result.ok
    assert any("hand" in r and "REGRESSION" in r for r in result.reasons)


def test_no_placement_regression_passes_when_everything_holds():
    cand = {"heldout": {"by_placement": {"pocket": {"f2": 0.73}, "hand": {"f2": 0.72}}}}
    curr = {"heldout": {"by_placement": {"pocket": {"f2": 0.72}, "hand": {"f2": 0.70}}}}
    result = no_placement_regression(cand, curr, tolerance=0.02)
    assert result.ok


def test_no_placement_regression_skipped_when_no_stratification():
    cand = {"heldout": {"f2": 0.70}}
    curr = {"heldout": {"f2": 0.70}}
    result = no_placement_regression(cand, curr)
    assert result.ok
    assert any("by_placement" in r for r in result.reasons)


def test_phone_truthset_not_worse_skipped_when_candidate_has_no_truthset():
    cand = {"heldout": {"f2": 0.7}}
    curr = {"heldout": {"f2": 0.7}}
    result = phone_truthset_not_worse(cand, curr)
    assert result.ok
    assert any("no phone_truthset" in r for r in result.reasons)


def test_phone_truthset_not_worse_enforces_floor():
    cand = {"phone_truthset": {"f2": 0.40, "false_alerts_per_hour": 2.0}}
    result = phone_truthset_not_worse(cand, None, min_f2=0.6)
    assert not result.ok
    assert any("floor" in r for r in result.reasons)


def test_phone_truthset_not_worse_enforces_ceiling_on_fp_rate():
    cand = {"phone_truthset": {"f2": 0.80, "false_alerts_per_hour": 10.0}}
    result = phone_truthset_not_worse(cand, None, max_fp_per_hour=3.0)
    assert not result.ok
    assert any("ceiling" in r for r in result.reasons)


def test_phone_truthset_not_worse_compares_against_current():
    cand = {"phone_truthset": {"f2": 0.60}}
    curr = {"phone_truthset": {"f2": 0.80}}
    result = phone_truthset_not_worse(cand, curr, tolerance=0.01)
    assert not result.ok
    assert any("regressed" in r for r in result.reasons)


def test_run_gates_and_gates_passed_aggregate():
    cand = {"heldout": {"f2": 0.60}}
    curr = {"heldout": {"f2": 0.75}}
    results = run_gates(cand, curr)
    assert len(results) == 3
    assert all(isinstance(r, GateResult) for r in results)
    assert not gates_passed(results)  # heldout regressed

    cand_ok = {"heldout": {"f2": 0.75}}
    results_ok = run_gates(cand_ok, curr)
    assert gates_passed(results_ok)
