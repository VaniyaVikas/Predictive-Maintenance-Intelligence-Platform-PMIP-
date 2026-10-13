"""Unsupervised anomaly detection on sensor behaviour (Isolation Forest).

IMPORTANT: "anomalous" means *statistically unusual relative to normal training operation*.
It is NOT the same as "machine failure": a failure can occur with ordinary-looking sensors, and unusual
readings can be harmless. The AI4I data has no ground-truth anomaly labels, so the failure label is
used only as a weak sanity check, never as proof of detector quality.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from .features import ALL_NUMERIC, SensorFeatureEngineer


class SensorAnomalyDetector:
    """Isolation Forest fitted on normal rows; scores are percentile ranks vs. normal training scores.

    ``score`` is in [0, 1]: the fraction of normal training rows that look *less* unusual.
    A row is flagged when ``score >= flag_percentile`` (default 0.99, i.e. more unusual than 99% of
    normal training rows, so roughly 1% of normal operation is flagged by construction).
    """

    def __init__(self, n_estimators: int = 200, flag_percentile: float = 0.99, seed: int = 42) -> None:
        self.flag_percentile = flag_percentile
        self._fe = SensorFeatureEngineer()
        self._model = IsolationForest(n_estimators=n_estimators, contamination="auto", random_state=seed, n_jobs=-1)
        self._ref_scores: np.ndarray | None = None

    def _matrix(self, X: pd.DataFrame) -> np.ndarray:
        return self._fe.transform(X)[ALL_NUMERIC].to_numpy()

    def fit(self, X_normal: pd.DataFrame) -> SensorAnomalyDetector:
        m = self._matrix(X_normal)
        self._model.fit(m)
        self._ref_scores = np.sort(-self._model.score_samples(m))
        return self

    def raw_score(self, X: pd.DataFrame) -> np.ndarray:
        return -self._model.score_samples(self._matrix(X))

    def score(self, X: pd.DataFrame) -> np.ndarray:
        if self._ref_scores is None:
            raise RuntimeError("Detector is not fitted.")
        raw = self.raw_score(X)
        return np.searchsorted(self._ref_scores, raw, side="right") / len(self._ref_scores)

    def is_anomalous(self, X: pd.DataFrame) -> np.ndarray:
        return self.score(X) >= self.flag_percentile
