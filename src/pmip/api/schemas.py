"""Pydantic request/response models for the REST API."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class SensorReading(BaseModel):
    """One machine's sensor snapshot. Unknown fields are rejected."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"example": {
            "machine_id": "demo-001", "machine_type": "M", "air_temperature_k": 300.1,
            "process_temperature_k": 310.2, "rotational_speed_rpm": 1500, "torque_nm": 42.5, "tool_wear_min": 120}},
    )
    machine_id: str | None = Field(None, max_length=64, description="Optional label echoed back; never used by the model.")
    machine_type: Literal["L", "M", "H"] = Field(..., description="Product quality variant: L, M or H.")
    air_temperature_k: float = Field(..., ge=250, le=350)
    process_temperature_k: float = Field(..., ge=250, le=360)
    rotational_speed_rpm: float = Field(..., gt=0, le=10000)
    torque_nm: float = Field(..., ge=0, le=500)
    tool_wear_min: float = Field(..., ge=0, le=1000)


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    records: list[SensorReading] = Field(..., min_length=1, max_length=1000)


class FeatureContribution(BaseModel):
    feature: str
    value: float
    contribution: float
    direction: Literal["increases_risk", "decreases_risk"]


class PredictionResult(BaseModel):
    machine_id: str | None = None
    failure_probability: float = Field(..., ge=0, le=1, description="Platt-calibrated on the validation set; see /model-info.")
    predicted_failure: bool
    risk_level: Literal["LOW", "MEDIUM", "HIGH"]
    decision_threshold: float
    likely_failure_mode: str | None = Field(None, description="Conditional on a failure; only given for MEDIUM+ risk.")
    failure_mode_probabilities: dict[str, float] | None = None
    anomaly_score: float = Field(..., ge=0, le=1)
    is_anomalous: bool
    top_contributions: list[FeatureContribution] | None = Field(None, description="SHAP associations, not causal proof.")


class PredictionResponse(PredictionResult):
    model_version: str
    probability_calibrated: bool = True


class BatchSummary(BaseModel):
    predicted_failures: int
    anomalous_rows: int
    risk_counts: dict[str, int]


class BatchResponse(BaseModel):
    model_version: str
    count: int
    summary: BatchSummary
    results: list[PredictionResult]


class AnomalyResult(BaseModel):
    machine_id: str | None = None
    anomaly_score: float
    is_anomalous: bool
    flag_threshold: float


class AnomalyResponse(BaseModel):
    model_version: str
    count: int
    results: list[AnomalyResult]
    note: str = "Anomalous = unusual vs. normal training operation. It is not a confirmed failure."


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_version: str | None = None
    detail: str | None = None


class ModelInfoResponse(BaseModel):
    model_version: str
    model: str
    variant: str
    created_utc: str | None
    decision_threshold: float
    threshold_objective: str
    calibration: str
    risk_bands: dict[str, float]
    features: list[str]
    dataset: dict
    validation_metrics: dict
    test_metrics: dict
    test_metrics_ci95: dict
    shap_supported: bool
    limitations: list[str]
