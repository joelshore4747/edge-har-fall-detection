"""Tests for scripts/organize_har_artifacts.py.

The organizer discovers a fixed set of variant directories under the
artifact root (``KNOWN_VARIANTS`` in the script), copies their model +
metadata into ``candidates/<variant>/``, and promotes one to ``current/``.
The test writes a minimal subset of that layout into a tmp_path, runs the
script against it, then checks the emitted canonical layout.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "organize_har_artifacts.py"

# Must match one of the variants hardcoded in scripts/organize_har_artifacts.KNOWN_VARIANTS.
VARIANT = "movement_v2"
VARIANT_MODEL_FILENAME = "model.joblib"
REPORT_FILENAME = "har_movement_v2_combined.json"


def _make_minimal_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Lay down artifact-root + legacy-report-dir + canonical-result-dir."""
    artifact_root = tmp_path / "artifacts" / "har"
    report_dir = tmp_path / "results" / "validation"
    canonical_result_dir = tmp_path / "results" / "validation" / "har"

    variant_dir = artifact_root / VARIANT
    variant_dir.mkdir(parents=True, exist_ok=True)
    (variant_dir / VARIANT_MODEL_FILENAME).write_bytes(b"stub-model-bytes")
    (variant_dir / "metadata.json").write_text(
        json.dumps(
            {
                "target_rate_hz": 50.0,
                "window_size": 128,
                "step_size": 64,
                "n_features": 103,
                "restrict_to_shared_labels": False,
                "holdout_mode": "internal_group_split",
                "holdout_source": "none",
                "train_source": "combined_all",
                "train_dataset_names": ["PAMAP2", "UCIHAR", "WISDM"],
                "allowed_labels": ["static", "locomotion", "stairs", "other"],
            }
        ),
        encoding="utf-8",
    )

    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / REPORT_FILENAME).write_text(
        json.dumps(
            {
                "artifact_version": "v2_movement",
                "metrics": {
                    "labels": ["locomotion", "static"],
                    "accuracy": 0.91,
                    "macro_f1": 0.88,
                    "support_total": 100,
                    "per_class": {
                        "locomotion": {"precision": 0.9, "recall": 0.85, "f1": 0.87, "support": 60},
                        "static": {"precision": 0.92, "recall": 0.93, "f1": 0.92, "support": 40},
                    },
                },
                "train": {"artifact_label_order": ["locomotion", "static"]},
                "per_dataset_metrics": {},
            }
        ),
        encoding="utf-8",
    )

    return artifact_root, report_dir, canonical_result_dir


@pytest.fixture
def fixture_layout(tmp_path: Path) -> tuple[Path, Path, Path]:
    return _make_minimal_fixture(tmp_path)


def _run_organizer(artifact_root: Path, report_dir: Path, canonical_result_dir: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--current-variant",
            VARIANT,
            "--artifact-root",
            str(artifact_root),
            "--legacy-report-dir",
            str(report_dir),
            "--canonical-result-dir",
            str(canonical_result_dir),
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"organizer failed:\nstdout={result.stdout}\nstderr={result.stderr}"


def test_promotion_populates_current_directory(fixture_layout: tuple[Path, Path, Path]) -> None:
    artifact_root, report_dir, canonical_result_dir = fixture_layout
    _run_organizer(artifact_root, report_dir, canonical_result_dir)

    current_model = artifact_root / "current" / "model.joblib"
    current_metadata = artifact_root / "current" / "metadata.json"
    assert current_model.is_file()
    assert current_metadata.is_file()
    assert current_model.read_bytes() == b"stub-model-bytes"


def test_current_metadata_contains_required_keys(fixture_layout: tuple[Path, Path, Path]) -> None:
    artifact_root, report_dir, canonical_result_dir = fixture_layout
    _run_organizer(artifact_root, report_dir, canonical_result_dir)

    metadata = json.loads((artifact_root / "current" / "metadata.json").read_text())
    required = {
        "artifact_id",
        "model_kind",
        "status",
        "created_utc",
        "source_artifact_path",
        "artifact_path",
        "selection_metric",
        "target_rate_hz",
        "window_size",
        "step_size",
        "validation",
        "heldout",
        "promoted_utc",
    }
    missing = required - set(metadata.keys())
    assert not missing, f"metadata is missing required keys: {sorted(missing)}"
    assert metadata["status"] == "current"
    assert metadata["selection_metric"] == "manual_current_baseline"
    assert metadata["promoted_utc"]  # non-empty ISO8601 string


def test_promotion_populates_candidates_and_archive(fixture_layout: tuple[Path, Path, Path]) -> None:
    artifact_root, report_dir, canonical_result_dir = fixture_layout
    _run_organizer(artifact_root, report_dir, canonical_result_dir)

    candidate_dir = artifact_root / "candidates" / VARIANT
    assert (candidate_dir / "model.joblib").is_file()
    assert (candidate_dir / "metadata.json").is_file()
    assert (artifact_root / "archive").is_dir()


def test_comparison_report_is_emitted(fixture_layout: tuple[Path, Path, Path]) -> None:
    artifact_root, report_dir, canonical_result_dir = fixture_layout
    _run_organizer(artifact_root, report_dir, canonical_result_dir)

    comp_dir = canonical_result_dir / "comparison"
    assert (comp_dir / "har_variant_comparison.json").is_file()
    assert (comp_dir / "har_variant_comparison.csv").is_file()
    assert (comp_dir / "har_variant_comparison.md").is_file()

    payload = json.loads((comp_dir / "har_variant_comparison.json").read_text())
    assert payload["candidates"], "comparison report should list at least one candidate"
    assert payload["candidates"][0]["variant"] == VARIANT
