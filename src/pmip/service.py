"""Prediction service shared by the REST API and the dashboard."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .config import Settings
from .monitoring import MonitoringStore
from .registry import ModelBundle
from .schema import validate_features


def risk_level(prob: float, medium: float, high: float) -> str:
    return "HIGH" if prob >= high else "MEDIUM" if prob >= medium else "LOW"


class PredictionService:
    def __init__(self, bundle: ModelBundle, settings: Settings, store: MonitoringStore | None = None) -> None:
        self.bundle = bundle
        self.settings = settings
        self.store = store

    def predict(self, df: pd.DataFrame, explain: bool = False, top_k: int = 5, log: bool = True, source: str = "api") -> list[dict]:
        t0 = time.perf_counter()
        X = validate_features(df)
        b, risk = self.bundle, self.settings.risk
        prob = b.predict_proba(X)
        anomaly = b.anomaly_detector.score(X)
        flagged = anomaly >= b.anomaly_detector.flag_percentile
        modes = None
        if b.failure_type_model is not None:
            modes = b.failure_type_model.predict_proba(X)
            classes = list(b.failure_type_model.classes_)
        contribs = b.explainer.local(X, top_k) if explain and b.explainer.supported else None

        results = []
        for i, p in enumerate(prob):
            r = {
                "failure_probability": float(p),
                "predicted_failure": bool(p >= b.threshold),
                "risk_level": risk_level(float(p), risk.medium, risk.high),
                "decision_threshold": float(b.threshold),
                "likely_failure_mode": None,
                "failure_mode_probabilities": None,
                "anomaly_score": float(anomaly[i]),
                "is_anomalous": bool(flagged[i]),
                "top_contributions": contribs[i] if contribs else None,
            }
            if modes is not None and p >= risk.medium:  # only meaningful when a failure is plausible
                probs = {c: float(v) for c, v in zip(classes, modes[i], strict=True)}
                r["failure_mode_probabilities"] = dict(sorted(probs.items(), key=lambda kv: -kv[1]))
                r["likely_failure_mode"] = max(probs, key=probs.get)
            results.append(r)
        if log and self.store is not None:
            self.store.log_predictions(X, results, b.version, (time.perf_counter() - t0) * 1000, source=source)
        return results

    def anomaly(self, df: pd.DataFrame) -> list[dict]:
        X = validate_features(df)
        det = self.bundle.anomaly_detector
        s = det.score(X)
        return [
            {"anomaly_score": float(v), "is_anomalous": bool(v >= det.flag_percentile), "flag_threshold": float(det.flag_percentile)}
            for v in np.asarray(s)
        ]
