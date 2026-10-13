"""Shared fixtures. All data here is SYNTHETIC (see pmip.synthetic) — never real factory data."""
from __future__ import annotations

import pytest

from pmip.config import Settings
from pmip.registry import load_bundle
from pmip.schema import standardize_columns
from pmip.synthetic import generate_synthetic_ai4i
from pmip.train import train_pipeline

VALID = {
    "machine_type": "M", "air_temperature_k": 300.1, "process_temperature_k": 310.2,
    "rotational_speed_rpm": 1500, "torque_nm": 42.5, "tool_wear_min": 120,
}


@pytest.fixture(scope="session")
def raw_df():
    return generate_synthetic_ai4i(5000, seed=3)


@pytest.fixture(scope="session")
def df(raw_df):
    return standardize_columns(raw_df)


@pytest.fixture(scope="session")
def settings(tmp_path_factory):
    root = tmp_path_factory.mktemp("pmip")
    s = Settings()
    s.paths.models_dir = root / "models"
    s.paths.reports_dir = root / "reports"
    s.paths.monitoring_db = root / "monitoring.db"
    s.training.models = ["dummy", "logistic_regression", "random_forest", "xgboost"]
    s.training.tune_models = ["xgboost"]
    s.training.tuning_enabled = True
    s.training.tuning_n_iter = 2
    s.training.cv_folds = 2
    s.training.bootstrap_samples = 20
    return s


@pytest.fixture(scope="session")
def trained(settings, df):
    return train_pipeline(settings, df=df.copy(), use_mlflow=False)


@pytest.fixture(scope="session")
def bundle(trained, settings):
    return load_bundle(settings.paths.models_dir)


@pytest.fixture()
def valid_payload():
    return dict(VALID)
