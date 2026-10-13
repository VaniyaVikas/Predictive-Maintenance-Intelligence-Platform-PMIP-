"""Centralised, typed configuration loaded from YAML with environment overrides."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, model_validator

PROJECT_ROOT = Path(os.environ.get("PMIP_HOME", Path(__file__).resolve().parents[2]))


class Paths(BaseModel):
    raw_data: Path = Path("data/raw/ai4i2020.csv")
    models_dir: Path = Path("models")
    reports_dir: Path = Path("reports")
    monitoring_db: Path = Path("monitoring/monitoring.db")
    mlflow_uri: str = "sqlite:///mlflow.db"


class Split(BaseModel):
    test_size: float = Field(0.15, gt=0, lt=0.5)
    val_size: float = Field(0.15, gt=0, lt=0.5)


class Training(BaseModel):
    models: list[str] = ["dummy", "logistic_regression", "random_forest", "xgboost", "lightgbm"]
    tune_models: list[str] = ["xgboost", "lightgbm"]
    tuning_enabled: bool = True
    tuning_n_iter: int = Field(20, ge=1)
    cv_folds: int = Field(5, ge=2)
    selection_metric: str = "average_precision"
    threshold_beta: float = Field(2.0, gt=0)
    threshold_min_precision: float = Field(0.0, ge=0, le=1)
    bootstrap_samples: int = Field(300, ge=0)


class Anomaly(BaseModel):
    n_estimators: int = 200
    flag_percentile: float = Field(0.99, gt=0.5, lt=1)


class Risk(BaseModel):
    medium: float = Field(0.20, gt=0, lt=1)
    high: float = Field(0.50, gt=0, lt=1)

    @model_validator(mode="after")
    def _ordered(self) -> Risk:
        if self.medium >= self.high:
            raise ValueError("risk.medium must be smaller than risk.high")
        return self


class MonitoringCfg(BaseModel):
    psi_warning: float = 0.10
    psi_alert: float = 0.25
    ks_alpha: float = 0.01
    min_samples: int = 30


class ApiCfg(BaseModel):
    max_batch_size: int = 1000


class Settings(BaseModel):
    seed: int = 42
    paths: Paths = Paths()
    split: Split = Split()
    training: Training = Training()
    anomaly: Anomaly = Anomaly()
    risk: Risk = Risk()
    monitoring: MonitoringCfg = MonitoringCfg()
    api: ApiCfg = ApiCfg()

    def resolve(self, p: Path) -> Path:
        """Resolve a configured path relative to the project root."""
        return p if p.is_absolute() else PROJECT_ROOT / p


def _apply_env(data: dict) -> dict:
    env = os.environ
    data.setdefault("paths", {})
    data.setdefault("risk", {})
    if env.get("PMIP_MODELS_DIR"):
        data["paths"]["models_dir"] = env["PMIP_MODELS_DIR"]
    if env.get("MLFLOW_TRACKING_URI"):
        data["paths"]["mlflow_uri"] = env["MLFLOW_TRACKING_URI"]
    if env.get("PMIP_RISK_MEDIUM"):
        data["risk"]["medium"] = float(env["PMIP_RISK_MEDIUM"])
    if env.get("PMIP_RISK_HIGH"):
        data["risk"]["high"] = float(env["PMIP_RISK_HIGH"])
    return data


@lru_cache(maxsize=1)
def load_settings() -> Settings:
    """Load settings from ``PMIP_CONFIG`` (default ``configs/config.yaml``)."""
    cfg_path = Path(os.environ.get("PMIP_CONFIG", "configs/config.yaml"))
    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / cfg_path
    data = yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}
    return Settings(**_apply_env(data or {}))
