"""Canonical artifact-layout resolver.

The fall and HAR promotion scripts
(``scripts/organize_fall_artifacts.py``, ``scripts/organize_har_artifacts.py``)
populate a shared on-disk layout::

    artifacts/
      fall/
        current/{model.joblib, metadata.json}
        candidates/<kind>/{model.joblib, metadata.json}
        archive/...
      har/
        current/{model.joblib, metadata.json}
        candidates/<variant>/{model.joblib, metadata.json}
        archive/...

This module is the single read-side entry point for that layout. Callers never
hardcode ``artifacts/<task>/<variant>/...`` — they ask the registry for
``current`` and get back a path plus metadata dict.

The registry is *resolution only*: it does not load joblib models. The
existing low-level loaders
``models.fall.infer_fall.load_fall_model_artifact`` and
``models.har.train_har.load_har_model_artifact`` remain the right tools for
that.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

ArtifactTask = Literal["fall", "har"]

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_ROOT = _REPO_ROOT / "artifacts"
_VALID_TASKS: tuple[ArtifactTask, ...] = ("fall", "har")
_MODEL_FILENAME = "model.joblib"
_METADATA_FILENAME = "metadata.json"


@dataclass(frozen=True)
class CandidateDescriptor:
    """A single promotable candidate under ``candidates/<name>/``."""

    task: ArtifactTask
    name: str
    artifact_path: Path
    metadata_path: Path
    metadata: dict[str, Any]

    @property
    def is_current(self) -> bool:
        return (self.metadata.get("status") or "").lower() == "current"


def _validate_task(task: ArtifactTask) -> None:
    if task not in _VALID_TASKS:
        raise ValueError(f"Unknown artifact task {task!r}; expected one of {_VALID_TASKS}")


def resolve_artifact_root(root: Path | None = None) -> Path:
    """Return the configured artifact root (defaults to ``<repo>/artifacts``)."""
    return Path(root) if root is not None else _DEFAULT_ROOT


def _task_root(task: ArtifactTask, root: Path | None) -> Path:
    _validate_task(task)
    return resolve_artifact_root(root) / task


def resolve_current_artifact(task: ArtifactTask, root: Path | None = None) -> Path:
    """Return the path to ``artifacts/<task>/current/model.joblib``.

    Raises ``FileNotFoundError`` with a clear message if the canonical
    ``current/`` directory or model file is missing — callers (notably the
    API startup probe) should let this propagate so misconfiguration fails
    loudly rather than silently falling back to a stale path.
    """
    task_root = _task_root(task, root)
    current_dir = task_root / "current"
    model_path = current_dir / _MODEL_FILENAME
    if not current_dir.is_dir():
        raise FileNotFoundError(
            f"No canonical current/ directory for task {task!r} at {current_dir}. "
            f"Run scripts/organize_{task}_artifacts.py to populate it."
        )
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Missing {_MODEL_FILENAME} under {current_dir}. "
            f"Run scripts/organize_{task}_artifacts.py to promote a candidate."
        )
    return model_path


def load_current_metadata(task: ArtifactTask, root: Path | None = None) -> dict[str, Any]:
    """Return the metadata dict for ``artifacts/<task>/current/metadata.json``.

    Raises ``FileNotFoundError`` if the metadata is missing.
    """
    task_root = _task_root(task, root)
    metadata_path = task_root / "current" / _METADATA_FILENAME
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Missing {_METADATA_FILENAME} under {task_root / 'current'}. "
            f"Run scripts/organize_{task}_artifacts.py to populate it."
        )
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def resolve_candidate_artifact(
    task: ArtifactTask,
    name: str,
    root: Path | None = None,
) -> Path:
    """Return the model path for a named candidate.

    Raises ``FileNotFoundError`` if the candidate directory or model is
    missing.
    """
    task_root = _task_root(task, root)
    candidate_dir = task_root / "candidates" / name
    model_path = candidate_dir / _MODEL_FILENAME
    if not candidate_dir.is_dir():
        raise FileNotFoundError(
            f"Unknown {task} candidate {name!r}: no directory at {candidate_dir}."
        )
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Missing {_MODEL_FILENAME} under {candidate_dir}."
        )
    return model_path


def list_candidates(task: ArtifactTask, root: Path | None = None) -> list[CandidateDescriptor]:
    """Enumerate all candidates for a task.

    Returns an empty list (not an error) if ``candidates/`` does not exist.
    Each descriptor carries the parsed metadata dict when present; if a
    candidate lacks ``metadata.json`` the descriptor's ``metadata`` is an
    empty dict.
    """
    task_root = _task_root(task, root)
    candidates_dir = task_root / "candidates"
    if not candidates_dir.is_dir():
        return []

    descriptors: list[CandidateDescriptor] = []
    for entry in sorted(candidates_dir.iterdir()):
        if not entry.is_dir():
            continue
        model_path = entry / _MODEL_FILENAME
        metadata_path = entry / _METADATA_FILENAME
        metadata: dict[str, Any] = {}
        if metadata_path.is_file():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                metadata = {}
        descriptors.append(
            CandidateDescriptor(
                task=task,
                name=entry.name,
                artifact_path=model_path,
                metadata_path=metadata_path,
                metadata=metadata,
            )
        )
    return descriptors