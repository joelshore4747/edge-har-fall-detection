"""Tests for the canonical artifact registry."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline.artifacts import (
    CandidateDescriptor,
    list_candidates,
    load_current_metadata,
    resolve_candidate_artifact,
    resolve_current_artifact,
)


def _make_fake_task_tree(
    root: Path,
    *,
    task: str,
    current_variant: str,
    variants: list[str],
) -> None:
    """Lay down a synthetic artifacts/<task>/{current,candidates,archive}/ tree."""
    task_root = root / task
    (task_root / "archive").mkdir(parents=True, exist_ok=True)

    for name in variants:
        cand_dir = task_root / "candidates" / name
        cand_dir.mkdir(parents=True, exist_ok=True)
        (cand_dir / "model.joblib").write_bytes(b"stub")
        (cand_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "artifact_id": f"{task}_{name}",
                    "model_kind": "rf" if task == "har" else "xgb",
                    "status": "current" if name == current_variant else "candidate",
                }
            ),
            encoding="utf-8",
        )

    current_dir = task_root / "current"
    current_dir.mkdir(parents=True, exist_ok=True)
    (current_dir / "model.joblib").write_bytes(b"stub")
    (current_dir / "metadata.json").write_text(
        json.dumps(
            {
                "artifact_id": f"{task}_{current_variant}",
                "model_kind": "rf" if task == "har" else "xgb",
                "status": "current",
                "promoted_utc": "2026-04-22T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )


@pytest.fixture
def fake_root(tmp_path: Path) -> Path:
    root = tmp_path / "artifacts"
    root.mkdir()
    _make_fake_task_tree(root, task="fall", current_variant="xgb", variants=["hgb", "xgb"])
    _make_fake_task_tree(
        root,
        task="har",
        current_variant="movement_v2",
        variants=[
            "movement_v2",
            "pamap2_shared_rf_balanced",
            "pamap2_shared_rf_balanced_reg",
        ],
    )
    return root


def test_resolve_current_artifact_for_each_task(fake_root: Path) -> None:
    fall_path = resolve_current_artifact("fall", root=fake_root)
    har_path = resolve_current_artifact("har", root=fake_root)
    assert fall_path == fake_root / "fall" / "current" / "model.joblib"
    assert har_path == fake_root / "har" / "current" / "model.joblib"
    assert fall_path.is_file()
    assert har_path.is_file()


def test_load_current_metadata_roundtrip(fake_root: Path) -> None:
    meta = load_current_metadata("har", root=fake_root)
    assert meta["artifact_id"] == "har_movement_v2"
    assert meta["status"] == "current"
    assert meta["promoted_utc"] == "2026-04-22T00:00:00+00:00"


def test_resolve_current_artifact_raises_when_layout_missing(tmp_path: Path) -> None:
    empty_root = tmp_path / "nowhere"
    empty_root.mkdir()
    with pytest.raises(FileNotFoundError) as exc:
        resolve_current_artifact("fall", root=empty_root)
    assert "organize_fall_artifacts.py" in str(exc.value)


def test_resolve_current_artifact_raises_when_model_file_missing(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    (root / "fall" / "current").mkdir(parents=True)
    # metadata but no model.joblib
    (root / "fall" / "current" / "metadata.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileNotFoundError) as exc:
        resolve_current_artifact("fall", root=root)
    assert "model.joblib" in str(exc.value)


def test_list_candidates_returns_all_variants_with_current_flag(fake_root: Path) -> None:
    har_candidates = list_candidates("har", root=fake_root)
    names = [c.name for c in har_candidates]
    assert names == sorted(
        [
            "movement_v2",
            "pamap2_shared_rf_balanced",
            "pamap2_shared_rf_balanced_reg",
        ]
    )
    current = [c for c in har_candidates if c.is_current]
    assert len(current) == 1
    assert current[0].name == "movement_v2"
    assert isinstance(current[0], CandidateDescriptor)
    assert current[0].metadata["status"] == "current"


def test_list_candidates_empty_when_candidates_dir_missing(tmp_path: Path) -> None:
    # Only current/ exists, no candidates/.
    root = tmp_path / "artifacts"
    (root / "har" / "current").mkdir(parents=True)
    (root / "har" / "current" / "model.joblib").write_bytes(b"")
    (root / "har" / "current" / "metadata.json").write_text("{}", encoding="utf-8")
    assert list_candidates("har", root=root) == []


def test_resolve_candidate_artifact_by_name(fake_root: Path) -> None:
    path = resolve_candidate_artifact("har", "pamap2_shared_rf_balanced_reg", root=fake_root)
    assert path == fake_root / "har" / "candidates" / "pamap2_shared_rf_balanced_reg" / "model.joblib"
    assert path.is_file()


def test_resolve_candidate_artifact_rejects_unknown_name(fake_root: Path) -> None:
    with pytest.raises(FileNotFoundError):
        resolve_candidate_artifact("har", "does_not_exist", root=fake_root)


def test_unknown_task_raises_value_error(fake_root: Path) -> None:
    with pytest.raises(ValueError):
        resolve_current_artifact("ambient", root=fake_root)  # type: ignore[arg-type]