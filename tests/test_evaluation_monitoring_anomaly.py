import numpy as np
import pandas as pd
import pytest

from pmip.anomaly import SensorAnomalyDetector
from pmip.calibration import PlattCalibrator
from pmip.config import MonitoringCfg
from pmip.evaluation import compute_metrics, select_threshold
from pmip.monitoring import MonitoringStore, build_reference, compute_drift, psi
from pmip.schema import INPUT_FEATURES


def test_select_threshold_prefers_recall_with_beta_2():
    y = np.array([0] * 90 + [1] * 10)
    p = np.concatenate([np.linspace(0, 0.6, 90), np.linspace(0.3, 0.95, 10)])
    t2, t05 = select_threshold(y, p, beta=2), select_threshold(y, p, beta=0.5)
    assert t2 <= t05
    assert compute_metrics(y, p, t2)["recall"] >= compute_metrics(y, p, t05)["recall"]


def test_metrics_confusion_counts():
    m = compute_metrics([0, 0, 1, 1], [0.1, 0.9, 0.8, 0.2], 0.5)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (1, 1, 1, 1)


def test_calibrator_outputs_probabilities_and_is_monotone():
    rng = np.random.default_rng(0)
    s = rng.random(500)
    y = (rng.random(500) < s).astype(int)
    c = PlattCalibrator().fit(s, y)
    out = c.transform(np.array([0.1, 0.5, 0.9]))
    assert np.all((out >= 0) & (out <= 1)) and out[0] < out[1] < out[2]


def test_anomaly_scores_in_range_and_extremes_rank_higher(df):
    X = df[INPUT_FEATURES]
    det = SensorAnomalyDetector(50, 0.99, 0).fit(X[df["machine_failure"] == 0])
    sc = det.score(X.head(200))
    assert sc.min() >= 0 and sc.max() <= 1
    odd = X.head(5).copy()
    odd["torque_nm"], odd["rotational_speed_rpm"], odd["tool_wear_min"] = 120.0, 3500.0, 600.0
    assert det.score(odd).mean() > sc.mean()
    assert det.is_anomalous(odd).all()


def test_psi_zero_for_same_and_large_for_shift(df):
    X = df[INPUT_FEATURES]
    ref = build_reference(X.iloc[:3000])
    cfg = MonitoringCfg()
    same = compute_drift(ref, X.iloc[3000:], cfg)
    assert same["psi"].max() < 0.1
    shifted = X.iloc[3000:].copy()
    shifted["torque_nm"] += 15
    d = compute_drift(ref, shifted, cfg).set_index("feature")
    assert d.loc["torque_nm", "status"] == "alert" and d.loc["torque_nm", "psi"] > 0.25
    assert psi(np.array([0.5, 0.5]), np.array([0.5, 0.5])) == pytest.approx(0)


def test_drift_requires_minimum_samples(df):
    ref = build_reference(df[INPUT_FEATURES].iloc[:1000])
    out = compute_drift(ref, df[INPUT_FEATURES].iloc[:5], MonitoringCfg(min_samples=30))
    assert set(out["status"]) == {"insufficient_data"}


def test_monitoring_store_roundtrip(tmp_path):
    store = MonitoringStore(tmp_path / "m.db")
    X = pd.DataFrame([{"machine_type": "L", "air_temperature_k": 300, "process_temperature_k": 310,
                       "rotational_speed_rpm": 1500, "torque_nm": 40, "tool_wear_min": 10}])
    res = [{"failure_probability": 0.3, "predicted_failure": True, "risk_level": "MEDIUM", "anomaly_score": 0.5}]
    store.log_predictions(X, res, "v1", 5.0)
    store.log_error("/predict", "schema")
    s = store.summary()
    assert s["total_predictions"] == 1 and s["predicted_failures"] == 1 and s["logged_errors"] == 1
    assert len(store.recent()) == 1
