"""Platt-scaling probability calibrator (monotone, so it does not change ranking metrics)."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


class PlattCalibrator:
    """Maps raw model scores to calibrated probabilities with a 1-D logistic regression."""

    def __init__(self) -> None:
        self._lr = LogisticRegression(C=1e6, max_iter=1000)
        self.fitted = False

    def fit(self, scores: np.ndarray, y: np.ndarray) -> PlattCalibrator:
        y = np.asarray(y)
        if len(np.unique(y)) < 2:
            self.fitted = False  # cannot calibrate with a single class; fall back to identity
            return self
        self._lr.fit(_logit(scores), y)
        self.fitted = True
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(scores, dtype=float)
        if not self.fitted:
            return np.clip(scores, 0.0, 1.0)
        return self._lr.predict_proba(_logit(scores))[:, 1]
