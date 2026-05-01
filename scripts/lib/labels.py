"""Single source of truth for canonical activity-label normalisation.

The trainer and the experiments scripts must agree on how raw label strings
map to canonical labels (`fall` / `walking` / `stairs` / `static` / `other`).
Previously each script re-derived this and they drifted (the trainer collapsed
`static -> other` while experiments kept them split, which scrambled
genuinely-other sessions in the 5-class evaluation).

All consumers should call :func:`canonicalize` and, when the
four-class problem is intended, :func:`collapse_static_to_other`.
"""

from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from apps.api.schemas import normalise_canonical_session_label  # noqa: E402

CANONICAL_ACTIVITY_LABELS = ("fall", "walking", "stairs", "static", "other")
FOUR_CLASS_LABELS = ("fall", "walking", "stairs", "other")


def canonicalize(raw: str | None) -> str | None:
    """Return the canonical activity label, or ``None`` for blank/unknown.

    Wraps :func:`normalise_canonical_session_label` from the API schema so
    every alias the API accepts is recognised here too.
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


def collapse_static_to_other(label: str | None) -> str | None:
    """Fold `static` into `other` for the four-class problem."""
    if label is None:
        return None
    return "other" if label == "static" else label


def is_droppable_other(label: str | None) -> bool:
    """True when this label should be excluded under ``--drop-other``.

    `other` after canonicalisation means "we can't confidently assign one of
    the four target activities", which poisons the activity classifier. When
    `--drop-other` is set, those sessions are excluded from training and from
    activity-only experiments. Note that `static` is *not* droppable — it's a
    well-defined canonical label and only becomes `other` if the caller
    explicitly applies :func:`collapse_static_to_other` first.
    """
    return label == "other"
