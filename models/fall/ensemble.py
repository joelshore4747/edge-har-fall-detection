"""Soft-voting ensemble wrapper for the fall v3 artifact.

Lives in `models.fall.ensemble` (not a script module) so joblib-pickled
artifacts remain loadable from the runtime without a script-package import.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


class SoftVotingFallEnsemble:
    """Averages class-1 probabilities across a dict of base fall models.

    Exposes `predict_proba(X)` so it satisfies the runtime contract in
    `models/fall/infer_fall.py`. `classes_` is `[0, 1]` so `_positive_proba`
    pulls the correct column.
    """

    def __init__(self, kind_models: dict[str, Any]):
        self.kind_models = dict(kind_models)
        self.classes_ = np.array([0, 1])

    def _positive_prob(self, model: Any, X: pd.DataFrame) -> np.ndarray:
        proba = model.predict_proba(X)
        classes = [int(c) for c in getattr(model, "classes_", [0, 1])]
        if 1 not in classes:
            raise ValueError(
                f"Base model {type(model).__name__} missing class 1"
            )
        return proba[:, classes.index(1)]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        probs = [self._positive_prob(m, X) for m in self.kind_models.values()]
        mean_pos = np.mean(np.stack(probs, axis=0), axis=0)
        out = np.zeros((len(mean_pos), 2), dtype=float)
        out[:, 0] = 1.0 - mean_pos
        out[:, 1] = mean_pos
        return out

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)