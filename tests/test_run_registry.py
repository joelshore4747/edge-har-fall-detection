"""Tests for ``scripts.lib.run_registry``."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from scripts.lib.run_registry import (
    compute_run_id,
    content_hash,
    register_run,
    resolve_current_run,
    update_current_symlink,
)


def test_content_hash_is_order_invariant_in_data_manifest():
    a = content_hash(args={"x": 1}, data_manifest=["s2", "s1"], git_sha="abc")
    b = content_hash(args={"x": 1}, data_manifest=["s1", "s2"], git_sha="abc")
    assert a == b


def test_content_hash_changes_with_args():
    a = content_hash(args={"x": 1}, data_manifest=["s1"], git_sha="abc")
    b = content_hash(args={"x": 2}, data_manifest=["s1"], git_sha="abc")
    assert a != b


def test_content_hash_strips_credentials():
    # Username/password must not influence the hash, so the hash is the
    # same identity regardless of who ran the train.
    a = content_hash(args={"x": 1, "username": "alice"}, data_manifest=["s1"], git_sha="abc")
    b = content_hash(args={"x": 1, "username": "bob"}, data_manifest=["s1"], git_sha="abc")
    assert a == b


def test_register_run_idempotent_on_same_kind(tmp_path: Path):
    runs = tmp_path / "runs"
    out = runs / "abc"
    out.mkdir(parents=True)
    register_run(
        run_id="abc", kind="train", out_dir=out,
        metrics={"accuracy": 0.9, "macro_f1": 0.8},
        n_sessions=10, git_sha="g",
    )
    register_run(
        run_id="abc", kind="train", out_dir=out,
        metrics={"accuracy": 0.95, "macro_f1": 0.85},
        n_sessions=10, git_sha="g",
    )
    rows = list(csv.DictReader((runs / "index.csv").open()))
    assert len(rows) == 1
    assert rows[0]["accuracy"] == "0.950000"


def test_register_run_keeps_different_kinds(tmp_path: Path):
    runs = tmp_path / "runs"
    out = runs / "abc"
    out.mkdir(parents=True)
    register_run(run_id="abc", kind="train", out_dir=out, metrics={}, n_sessions=10, git_sha="g")
    register_run(run_id="abc", kind="experiments", out_dir=out, metrics={"fall_f1_at_0p5": 0.7}, n_sessions=10, git_sha="g")
    rows = list(csv.DictReader((runs / "index.csv").open()))
    assert sorted(r["kind"] for r in rows) == ["experiments", "train"]


def test_current_symlink_round_trip(tmp_path: Path):
    runs = tmp_path / "runs"
    out = runs / "abc"
    out.mkdir(parents=True)
    update_current_symlink(runs_root=runs, run_id="abc")
    resolved = resolve_current_run(runs_root=runs)
    assert resolved == out.resolve()


def test_current_symlink_target_must_exist(tmp_path: Path):
    runs = tmp_path / "runs"
    runs.mkdir()
    with pytest.raises(FileNotFoundError):
        update_current_symlink(runs_root=runs, run_id="missing")


def test_compute_run_id_has_short_sha_prefix():
    ri = compute_run_id(args={}, data_manifest=[])
    # Format is "YYYY-MM-DD_HHMMSS_<short>" — check structure.
    parts = ri.run_id.split("_")
    assert len(parts) == 3
    assert len(parts[2]) >= 5  # at least 5 chars of short sha or "nogit"
