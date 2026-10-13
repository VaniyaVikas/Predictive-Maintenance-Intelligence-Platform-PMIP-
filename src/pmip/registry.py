"""Versioned model bundles on disk + a loader with clear failure modes.

Security note: bundles are joblib (pickle) files. Only load bundles you produced yourself.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from .anomaly import SensorAnomalyDetector
from .calibration import PlattCalibrator
from .explain import ModelExplainer


class ModelNotAvailableError(RuntimeError):
    """Raised when no usable model bundle can be loaded."""


@dataclass
class ModelBundle:
    version: str
    pipeline: Pipeline
    calibrator: PlattCalibrator
    threshold: float
    anomaly_detector: SensorAnomalyDetector
    failure_type_model: Pipeline | None
    reference: dict
    metadata: dict = field(default_factory=dict)
    _explainer: ModelExplainer | None = field(default=None, repr=False, compare=False)

    def raw_scores(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(X)[:, 1]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Calibrated failure probability."""
        return self.calibrator.transform(self.raw_scores(X))

    @property
    def explainer(self) -> ModelExplainer:
        if self._explainer is None:
            self._explainer = ModelExplainer(self.pipeline)
        return self._explainer


def new_version() -> str:
    return datetime.now(UTC).strftime("v%Y%m%d-%H%M%S")


def save_bundle(bundle: ModelBundle, models_dir: Path) -> Path:
    """Persist a bundle under ``models_dir/<version>/`` and point ``latest.json`` at it."""
    models_dir = Path(models_dir)
    out = models_dir / bundle.version
    out.mkdir(parents=True, exist_ok=True)
    payload = {k: v for k, v in bundle.__dict__.items() if k != "_explainer"}
    joblib.dump(payload, out / "bundle.joblib", compress=3)
    (out / "metadata.json").write_text(json.dumps(bundle.metadata, indent=2, default=str))
    (models_dir / "latest.json").write_text(json.dumps({"version": bundle.version, "path": bundle.version}))
    return out


def list_versions(models_dir: Path) -> list[str]:
    models_dir = Path(models_dir)
    if not models_dir.exists():
        return []
    return sorted(p.name for p in models_dir.iterdir() if p.is_dir() and (p / "bundle.joblib").exists())


def load_bundle(models_dir: Path, version: str | None = None) -> ModelBundle:
    """Load ``version`` (default: the one referenced by ``latest.json``)."""
    models_dir = Path(models_dir)
    if not models_dir.exists():
        raise ModelNotAvailableError(f"Model directory '{models_dir}' does not exist. Train a model first: `make train`.")
    if version is None:
        pointer = models_dir / "latest.json"
        if pointer.exists():
            version = json.loads(pointer.read_text()).get("version")
        else:
            versions = list_versions(models_dir)
            if not versions:
                raise ModelNotAvailableError(f"No trained model found in '{models_dir}'. Train one with `make train`.")
            version = versions[-1]
    path = models_dir / version / "bundle.joblib"
    if not path.exists():
        raise ModelNotAvailableError(f"Model artefact '{path}' not found (available: {list_versions(models_dir)}).")
    try:
        payload = joblib.load(path)
        return ModelBundle(**payload)
    except Exception as exc:  # corrupt file, incompatible library versions, ...
        raise ModelNotAvailableError(
            f"Model artefact '{path}' could not be loaded ({type(exc).__name__}). "
            "It may be corrupted or built with an incompatible library version; retrain it."
        ) from exc
