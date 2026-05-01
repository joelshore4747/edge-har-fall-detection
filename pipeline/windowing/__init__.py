"""Shared windowing helpers for multi-branch runtime inference.

The runtime pipeline runs fall detection (100 Hz) and HAR (50 Hz) as two
independent resample+window passes on the same source dataframe. This
package bundles the two passes behind a single call that also produces a
window-level pairing table up front, so downstream code never has to
re-derive the alignment from predictions after the fact.
"""

from pipeline.windowing.synchronizer import (
    BranchWindowConfig,
    SynchronizedWindows,
    synchronize_windows,
)

__all__ = [
    "BranchWindowConfig",
    "SynchronizedWindows",
    "synchronize_windows",
]