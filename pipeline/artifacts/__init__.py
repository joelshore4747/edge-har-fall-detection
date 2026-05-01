"""Canonical artifact layout and resolution.

Exposes a thin registry over ``artifacts/{task}/{current,candidates,archive}/``
so callers (API, services, orchestration scripts) share a single source of
truth for the active fall / HAR model paths and metadata.
"""

from pipeline.artifacts.registry import (
    ArtifactTask,
    CandidateDescriptor,
    list_candidates,
    load_current_metadata,
    resolve_artifact_root,
    resolve_candidate_artifact,
    resolve_current_artifact,
)

__all__ = [
    "ArtifactTask",
    "CandidateDescriptor",
    "list_candidates",
    "load_current_metadata",
    "resolve_artifact_root",
    "resolve_candidate_artifact",
    "resolve_current_artifact",
]
