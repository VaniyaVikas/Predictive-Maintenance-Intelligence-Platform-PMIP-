import numpy as np
import pandas as pd
import pytest

from pmip.registry import ModelNotAvailableError, load_bundle, save_bundle
from pmip.schema import INPUT_FEATURES
from pmip.service import PredictionService, risk_level
from pmip.train import train_pipeline

EXPECTED_KEYS = {"failure_probability", "predicted_failure", "risk_level", "decision_threshold", "likely_failure_mode",
                 "failure_mode_probabilities", "anomaly_score", "is_anomalous", "top_contributions"}


def test_bundle_artifacts_exist_and_load(trained, settings):
    assert (trained.model_dir / "bundle.joblib").exists() and (trained.model_dir / "metadata.json").exists()
    b = load_bundle(settings.paths.models_dir)
    assert b.version == trained.version and 0 < b.threshold < 1


def test_selected_model_is_not_dummy_and_beats_baseline(trained):
    m = trained.metrics
    assert m["model"] != "dummy"
    assert m["test_metrics"]["average_precision"] > m["baseline_dummy_test_metrics"]["average_precision"]
    assert "test_metrics_ci95" in m and m["limitations"]


def test_comparison_contains_all_models_and_tuned_variant(trained):
    assert {"dummy", "logistic_regression", "random_forest", "xgboost"} <= set(trained.comparison["model"])
    assert "tuned" in set(trained.comparison["variant"])


def test_prediction_output_schema(bundle, settings, df):
    svc = PredictionService(bundle, settings, store=None)
    out = svc.predict(df[INPUT_FEATURES].head(10), explain=True, log=False)
    assert len(out) == 10
    for r in out:
        assert EXPECTED_KEYS <= set(r)
        assert 0 <= r["failure_probability"] <= 1 and r["risk_level"] in {"LOW", "MEDIUM", "HIGH"}
        assert 0 <= r["anomaly_score"] <= 1
        assert r["top_contributions"] is None or len(r["top_contributions"]) <= 5


def test_failure_type_only_for_elevated_risk(bundle, settings, df):
    svc = PredictionService(bundle, settings, store=None)
    out = svc.predict(df[INPUT_FEATURES].head(300), log=False)
    for r in out:
        if r["risk_level"] == "LOW":
            assert r["likely_failure_mode"] is None


def test_risk_levels_configurable():
    assert risk_level(0.1, 0.2, 0.5) == "LOW" and risk_level(0.3, 0.2, 0.5) == "MEDIUM" and risk_level(0.9, 0.2, 0.5) == "HIGH"


def test_training_is_reproducible(settings, df):
    a = train_pipeline(settings, df=df.copy(), use_mlflow=False, save=False, write_reports=False)
    b = train_pipeline(settings, df=df.copy(), use_mlflow=False, save=False, write_reports=False)
    assert a.metrics["test_metrics"]["average_precision"] == pytest.approx(b.metrics["test_metrics"]["average_precision"])
    assert a.metrics["decision_threshold"] == pytest.approx(b.metrics["decision_threshold"])


def test_shap_matches_deployed_pipeline(bundle, df):
    if not bundle.explainer.supported:
        pytest.skip("selected model is not tree-based")
    X = df[INPUT_FEATURES].head(20)
    vals, Xt = bundle.explainer.shap_values(X)
    assert vals.shape == Xt.shape and np.isfinite(vals).all()


def test_missing_model_dir_gives_clear_error(tmp_path):
    with pytest.raises(ModelNotAvailableError, match="Train"):
        load_bundle(tmp_path / "missing")
    (tmp_path / "empty").mkdir()
    with pytest.raises(ModelNotAvailableError, match="No trained model"):
        load_bundle(tmp_path / "empty")


def test_corrupt_artifact_reports_clear_error(bundle, tmp_path):
    save_bundle(bundle, tmp_path)
    (tmp_path / bundle.version / "bundle.joblib").write_bytes(b"not a model")
    with pytest.raises(ModelNotAvailableError, match="could not be loaded"):
        load_bundle(tmp_path)


def test_invalid_input_raises(bundle, settings):
    from pmip.schema import SchemaValidationError

    with pytest.raises(SchemaValidationError):
        PredictionService(bundle, settings).predict(pd.DataFrame([{"machine_type": "L"}]), log=False)
