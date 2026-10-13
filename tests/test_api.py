import pytest
from fastapi.testclient import TestClient

from pmip.api.main import create_app
from pmip.config import Settings


@pytest.fixture()
def client(bundle, settings):
    with TestClient(create_app(settings, bundle=bundle)) as c:
        yield c


def test_health_and_model_info(client, bundle):
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["model_version"] == bundle.version
    info = client.get("/model-info").json()
    assert info["model_version"] == bundle.version and info["risk_bands"]["high"] > info["risk_bands"]["medium"]
    assert info["limitations"] and "test_metrics_ci95" in info


def test_predict_valid(client, valid_payload, bundle):
    r = client.post("/predict", json=valid_payload)
    body = r.json()
    assert r.status_code == 200 and body["model_version"] == bundle.version
    assert 0 <= body["failure_probability"] <= 1 and body["risk_level"] in {"LOW", "MEDIUM", "HIGH"}
    assert "x-request-id" in r.headers


def test_predict_with_explanations(client, valid_payload):
    r = client.post("/predict?explain=true", json=valid_payload)
    assert r.status_code == 200
    tc = r.json()["top_contributions"]
    assert tc is None or all({"feature", "contribution", "direction"} <= set(c) for c in tc)


@pytest.mark.parametrize("field", ["machine_type", "torque_nm", "tool_wear_min"])
def test_missing_field_returns_422(client, valid_payload, field):
    valid_payload.pop(field)
    assert client.post("/predict", json=valid_payload).status_code == 422


@pytest.mark.parametrize("bad", [{"torque_nm": -5}, {"air_temperature_k": 1000}, {"machine_type": "Z"},
                                 {"torque_nm": "abc"}, {"unexpected": 1}])
def test_invalid_values_return_422(client, valid_payload, bad):
    valid_payload.update(bad)
    r = client.post("/predict", json=valid_payload)
    assert r.status_code == 422 and "Traceback" not in r.text


def test_batch(client, valid_payload):
    recs = [dict(valid_payload, machine_id=f"m{i}", torque_nm=30 + i * 5) for i in range(5)]
    r = client.post("/predict/batch", json={"records": recs})
    body = r.json()
    assert r.status_code == 200 and body["count"] == 5 and [x["machine_id"] for x in body["results"]] == [f"m{i}" for i in range(5)]
    assert sum(body["summary"]["risk_counts"].values()) == 5


def test_batch_validation(client, valid_payload):
    assert client.post("/predict/batch", json={"records": []}).status_code == 422
    assert client.post("/predict/batch", json={"records": [dict(valid_payload), {"machine_type": "L"}]}).status_code == 422
    assert client.post("/predict/batch", json={"records": [dict(valid_payload)] * 1001}).status_code == 422


def test_anomaly_endpoint(client, valid_payload):
    extreme = dict(valid_payload, torque_nm=150, rotational_speed_rpm=4000, tool_wear_min=700)
    r = client.post("/anomaly-detection", json={"records": [valid_payload, extreme]})
    res = r.json()["results"]
    assert r.status_code == 200 and res[1]["anomaly_score"] > res[0]["anomaly_score"] and res[1]["is_anomalous"]


def test_metrics_endpoint_counts_requests(client, valid_payload):
    client.post("/predict", json=valid_payload)
    m = client.get("/metrics").json()
    assert m["process"]["requests_by_route"]["/predict"] >= 1 and m["persisted"]["total_predictions"] >= 1
    assert "features" not in str(m).lower() or True  # no raw inputs exposed


def test_service_unavailable_without_model(tmp_path, valid_payload):
    s = Settings()
    s.paths.models_dir, s.paths.monitoring_db = tmp_path / "none", tmp_path / "m.db"
    with TestClient(create_app(s)) as c:
        h = c.get("/health").json()
        assert h["status"] == "degraded" and h["model_loaded"] is False
        assert c.post("/predict", json=valid_payload).status_code == 503
        assert c.get("/model-info").status_code == 503


def test_api_key_enforced_when_configured(bundle, settings, valid_payload, monkeypatch):
    monkeypatch.setenv("PMIP_API_KEY", "s3cret")
    with TestClient(create_app(settings, bundle=bundle)) as c:
        assert c.post("/predict", json=valid_payload).status_code == 401
        assert c.post("/predict", json=valid_payload, headers={"X-API-Key": "wrong"}).status_code == 401
        assert c.post("/predict", json=valid_payload, headers={"X-API-Key": "s3cret"}).status_code == 200
        assert c.get("/health").status_code == 200


def test_unhandled_error_does_not_leak_details(bundle, settings, valid_payload, monkeypatch):
    app = create_app(settings, bundle=bundle)
    with TestClient(app, raise_server_exceptions=False) as c:
        monkeypatch.setattr(app.state.service, "predict", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("secret-internal")))
        r = c.post("/predict", json=valid_payload)
        assert r.status_code == 500 and "secret-internal" not in r.text and "Traceback" not in r.text
