"""Thin MLflow wrapper; becomes a no-op when tracking is disabled (e.g. in unit tests)."""
from __future__ import annotations

import logging
import math
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)


class Tracker:
    def __init__(self, enabled: bool, uri: str, experiment: str = "pmip") -> None:
        self.enabled = enabled
        self._mlflow = None
        if enabled:
            import mlflow

            mlflow.set_tracking_uri(uri)
            mlflow.set_experiment(experiment)
            self._mlflow = mlflow

    @contextmanager
    def run(self, name: str, nested: bool = False):
        if not self.enabled:
            yield None
            return
        with self._mlflow.start_run(run_name=name, nested=nested) as r:
            yield r

    def params(self, d: dict) -> None:
        if self.enabled:
            self._mlflow.log_params({k: str(v)[:250] for k, v in d.items()})

    def tags(self, d: dict) -> None:
        if self.enabled:
            self._mlflow.set_tags({k: str(v)[:250] for k, v in d.items()})

    def metrics(self, d: dict, prefix: str = "") -> None:
        if self.enabled:
            clean = {f"{prefix}{k}": float(v) for k, v in d.items()
                     if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)}
            self._mlflow.log_metrics(clean)

    def artifact(self, path: Path, folder: str | None = None) -> None:
        if self.enabled and Path(path).exists():
            self._mlflow.log_artifact(str(path), artifact_path=folder)

    def model(self, pipeline, registered_name: str) -> None:
        if not self.enabled:
            return
        kwargs = {"registered_model_name": registered_name, "serialization_format": "cloudpickle"}
        try:
            try:
                self._mlflow.sklearn.log_model(pipeline, name="model", **kwargs)
            except TypeError:  # older MLflow uses artifact_path
                self._mlflow.sklearn.log_model(pipeline, artifact_path="model", **kwargs)
        except Exception as exc:  # tracking problems must never break training
            logger.warning("MLflow model logging failed: %s: %s", type(exc).__name__, str(exc)[:200])
