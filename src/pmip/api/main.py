"""FastAPI application factory for PMIP."""
from __future__ import annotations

import logging
import os
import secrets
import threading
import time
import uuid
from collections import Counter, deque
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import JSONResponse, RedirectResponse

from .. import __version__
from ..config import Settings, load_settings
from ..logging_utils import configure_logging, log_event
from ..monitoring import MonitoringStore
from ..registry import ModelBundle, ModelNotAvailableError, load_bundle
from ..schema import SchemaValidationError
from ..service import PredictionService
from .schemas import (
    AnomalyResponse,
    AnomalyResult,
    BatchRequest,
    BatchResponse,
    BatchSummary,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
    PredictionResult,
    SensorReading,
)

logger = logging.getLogger("pmip.api")


class OperationalMetrics:
    """Thread-safe in-memory counters (reset on restart). Persistent history lives in the SQLite store."""

    def __init__(self) -> None:
        self.started = time.time()
        self._lock = threading.Lock()
        self.requests: Counter = Counter()
        self.errors: Counter = Counter()
        self.latencies: deque = deque(maxlen=2000)

    def record(self, route: str, status: int, ms: float) -> None:
        with self._lock:
            self.requests[route] += 1
            self.latencies.append(ms)
            if status >= 400:
                self.errors[f"{route}:{status}"] += 1

    def snapshot(self) -> dict:
        with self._lock:
            lat = list(self.latencies)
            return {
                "uptime_seconds": round(time.time() - self.started, 1),
                "requests_by_route": dict(self.requests),
                "errors_by_route_status": dict(self.errors),
                "latency_ms": {"avg": float(np.mean(lat)) if lat else None, "p95": float(np.percentile(lat, 95)) if lat else None},
            }


def create_app(settings: Settings | None = None, bundle: ModelBundle | None = None) -> FastAPI:
    """Build the app. ``bundle`` can be injected (tests); otherwise the latest model is loaded at startup."""
    configure_logging()
    settings = settings or load_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.service = None
        app.state.load_error = None
        app.state.store = MonitoringStore(settings.resolve(settings.paths.monitoring_db))
        try:
            b = bundle or load_bundle(settings.resolve(settings.paths.models_dir))
            app.state.service = PredictionService(b, settings, app.state.store)
            log_event(logger, "model loaded", version=b.version)
        except ModelNotAvailableError as exc:
            app.state.load_error = str(exc)
            logger.error("model unavailable: %s", exc)
        yield

    app = FastAPI(title="Predictive Maintenance Intelligence Platform", version=__version__, lifespan=lifespan,
                  description="Failure-risk scoring, anomaly detection and explanations for industrial sensor data.")
    app.state.metrics = OperationalMetrics()
    app.state.settings = settings

    def get_service(request: Request) -> PredictionService:
        svc = request.app.state.service
        if svc is None:
            raise HTTPException(503, "Model not available. Train and deploy a model, then restart the service.")
        return svc

    def require_key(x_api_key: str | None = Header(None)) -> None:
        expected = os.environ.get("PMIP_API_KEY")
        if expected and not (x_api_key and secrets.compare_digest(x_api_key, expected)):
            raise HTTPException(401, "Missing or invalid API key.")

    @app.middleware("http")
    async def observe(request: Request, call_next):
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        t0 = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            logger.exception("unhandled error", extra={"extra_fields": {"request_id": rid, "path": request.url.path}})
            store = getattr(request.app.state, "store", None)
            if store:
                store.log_error(request.url.path, "unhandled")
            response = JSONResponse({"detail": "Internal server error.", "request_id": rid}, status_code=500)
        ms = (time.perf_counter() - t0) * 1000
        route = request.scope.get("route")
        request.app.state.metrics.record(getattr(route, "path", request.url.path), response.status_code, ms)
        response.headers["X-Request-ID"] = rid
        log_event(logger, "request", request_id=rid, method=request.method, path=request.url.path,
                  status=response.status_code, latency_ms=round(ms, 2))
        return response

    @app.exception_handler(SchemaValidationError)
    async def schema_error(request: Request, exc: SchemaValidationError):
        request.app.state.store.log_error(request.url.path, "schema")
        return JSONResponse({"detail": exc.issues}, status_code=422)

    def to_df(records: list[SensorReading]) -> pd.DataFrame:
        return pd.DataFrame([r.model_dump(exclude={"machine_id"}) for r in records])

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        """Send browsers that open the bare URL to the interactive API docs."""
        return RedirectResponse("/docs")

    @app.get("/health", response_model=HealthResponse, tags=["system"])
    def health(request: Request) -> HealthResponse:
        svc = request.app.state.service
        if svc is None:
            return HealthResponse(status="degraded", model_loaded=False, detail="No model loaded.")
        return HealthResponse(status="ok", model_loaded=True, model_version=svc.bundle.version)

    @app.get("/model-info", response_model=ModelInfoResponse, tags=["system"])
    def model_info(svc: PredictionService = Depends(get_service)) -> ModelInfoResponse:
        m, b = svc.bundle.metadata, svc.bundle
        return ModelInfoResponse(
            model_version=b.version, model=m.get("model", "?"), variant=m.get("variant", "?"), created_utc=m.get("created_utc"),
            decision_threshold=b.threshold, threshold_objective=m.get("threshold_objective", ""),
            calibration=m.get("calibration", ""), risk_bands={"medium": svc.settings.risk.medium, "high": svc.settings.risk.high},
            features=m.get("features", {}).get("inputs", []), dataset=m.get("dataset", {}),
            validation_metrics=m.get("validation_metrics", {}), test_metrics=m.get("test_metrics", {}),
            test_metrics_ci95=m.get("test_metrics_ci95", {}), shap_supported=b.explainer.supported,
            limitations=m.get("limitations", []),
        )

    @app.post("/predict", response_model=PredictionResponse, tags=["prediction"], dependencies=[Depends(require_key)])
    def predict(reading: SensorReading, explain: bool = Query(False, description="Add SHAP top contributions."),
                svc: PredictionService = Depends(get_service)) -> PredictionResponse:
        res = svc.predict(to_df([reading]), explain=explain)[0]
        return PredictionResponse(**res, machine_id=reading.machine_id, model_version=svc.bundle.version)

    @app.post("/predict/batch", response_model=BatchResponse, tags=["prediction"], dependencies=[Depends(require_key)])
    def predict_batch(req: BatchRequest, svc: PredictionService = Depends(get_service)) -> BatchResponse:
        if len(req.records) > svc.settings.api.max_batch_size:
            raise HTTPException(413, f"Batch too large (max {svc.settings.api.max_batch_size} records).")
        results = svc.predict(to_df(req.records))
        out = [PredictionResult(**r, machine_id=rec.machine_id) for r, rec in zip(results, req.records, strict=True)]
        return BatchResponse(
            model_version=svc.bundle.version, count=len(out), results=out,
            summary=BatchSummary(
                predicted_failures=sum(r.predicted_failure for r in out), anomalous_rows=sum(r.is_anomalous for r in out),
                risk_counts=dict(Counter(r.risk_level for r in out))),
        )

    @app.post("/anomaly-detection", response_model=AnomalyResponse, tags=["prediction"], dependencies=[Depends(require_key)])
    def anomaly(req: BatchRequest, svc: PredictionService = Depends(get_service)) -> AnomalyResponse:
        res = svc.anomaly(to_df(req.records))
        return AnomalyResponse(model_version=svc.bundle.version, count=len(res),
                               results=[AnomalyResult(**r, machine_id=rec.machine_id) for r, rec in zip(res, req.records, strict=True)])

    @app.get("/metrics", tags=["system"])
    def metrics(request: Request) -> dict:
        """Operational metrics only (counts and latencies); no inputs or secrets are exposed."""
        return {"process": request.app.state.metrics.snapshot(), "persisted": request.app.state.store.summary()}

    return app


app = create_app()