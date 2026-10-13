"""Input-data drift monitoring and operational logging.

Drift here means a change in the *input feature distribution* versus the training reference
(PSI + Kolmogorov-Smirnov). It is NOT proof of concept drift or of model degradation: those require
ground-truth failure labels arriving later. A drift warning means "investigate", never "retrain automatically".
"""
from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from .config import MonitoringCfg
from .features import ALL_NUMERIC, SensorFeatureEngineer
from .schema import INPUT_FEATURES, MACHINE_TYPES

EPS = 1e-4
N_BINS = 10
KS_SAMPLE = 2000


def build_reference(X_train: pd.DataFrame, seed: int = 42) -> dict:
    """Store the training feature distribution (decile bin proportions + a sample for KS tests)."""
    feats = SensorFeatureEngineer().transform(X_train)
    rng = np.random.default_rng(seed)
    ref: dict = {"n_rows": int(len(feats)), "numeric": {}, "categorical": {}}
    for col in ALL_NUMERIC:
        v = feats[col].to_numpy(dtype=float)
        edges = np.unique(np.quantile(v, np.linspace(0, 1, N_BINS + 1)))
        inner = edges[1:-1]
        props = np.bincount(np.searchsorted(inner, v, side="right"), minlength=len(inner) + 1) / len(v)
        sample = rng.choice(v, size=min(KS_SAMPLE, len(v)), replace=False)
        ref["numeric"][col] = {
            "edges": inner.tolist(), "proportions": props.tolist(),
            "mean": float(v.mean()), "std": float(v.std()), "sample": sample.tolist(),
        }
    counts = feats["machine_type"].value_counts(normalize=True)
    ref["categorical"]["machine_type"] = {k: float(counts.get(k, 0.0)) for k in MACHINE_TYPES}
    return ref


def psi(expected: np.ndarray, actual: np.ndarray) -> float:
    """Population Stability Index between two proportion vectors."""
    e = np.clip(np.asarray(expected, dtype=float), EPS, None)
    a = np.clip(np.asarray(actual, dtype=float), EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def _status(value: float, warn: float, alert: float) -> str:
    return "alert" if value >= alert else "warning" if value >= warn else "ok"


def compute_drift(reference: dict, current: pd.DataFrame, cfg: MonitoringCfg) -> pd.DataFrame:
    """Per-feature PSI (+ KS for numerics). Requires ``cfg.min_samples`` rows, otherwise status is 'insufficient_data'."""
    feats = SensorFeatureEngineer().transform(current)
    n = len(feats)
    rows = []
    for col, ref in reference["numeric"].items():
        if n < cfg.min_samples:
            rows.append({"feature": col, "psi": np.nan, "ks_stat": np.nan, "ks_pvalue": np.nan, "status": "insufficient_data"})
            continue
        v = feats[col].to_numpy(dtype=float)
        props = np.bincount(np.searchsorted(np.asarray(ref["edges"]), v, side="right"), minlength=len(ref["edges"]) + 1) / n
        p = psi(np.asarray(ref["proportions"]), props)
        ks = ks_2samp(np.asarray(ref["sample"]), v)
        status = _status(p, cfg.psi_warning, cfg.psi_alert)
        if status == "ok" and ks.pvalue < cfg.ks_alpha:
            status = "warning"  # statistically detectable shift even though PSI is small
        rows.append({"feature": col, "psi": p, "ks_stat": float(ks.statistic), "ks_pvalue": float(ks.pvalue), "status": status})
    cat_ref = reference["categorical"]["machine_type"]
    if n < cfg.min_samples:
        rows.append({"feature": "machine_type", "psi": np.nan, "ks_stat": np.nan, "ks_pvalue": np.nan, "status": "insufficient_data"})
    else:
        cur = feats["machine_type"].value_counts(normalize=True)
        p = psi(np.array([cat_ref[k] for k in MACHINE_TYPES]), np.array([cur.get(k, 0.0) for k in MACHINE_TYPES]))
        rows.append({"feature": "machine_type", "psi": p, "ks_stat": np.nan, "ks_pvalue": np.nan,
                     "status": _status(p, cfg.psi_warning, cfg.psi_alert)})
    return pd.DataFrame(rows)


class MonitoringStore:
    """SQLite log of served predictions (features + outputs only; no secrets, no free text)."""

    def __init__(self, db_path: Path) -> None:
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.execute(
                """CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, model_version TEXT, source TEXT,
                machine_type TEXT, air_temperature_k REAL, process_temperature_k REAL,
                rotational_speed_rpm REAL, torque_nm REAL, tool_wear_min REAL,
                failure_probability REAL, predicted_failure INTEGER, risk_level TEXT,
                anomaly_score REAL, latency_ms REAL)"""
            )
            c.execute("CREATE TABLE IF NOT EXISTS errors (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, endpoint TEXT, kind TEXT)")

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=10)

    def log_predictions(self, X: pd.DataFrame, results: list[dict], model_version: str, latency_ms: float, source: str = "api") -> None:
        ts = datetime.now(UTC).isoformat(timespec="seconds")
        per_row = latency_ms / max(1, len(results))
        rows = [
            (ts, model_version, source, r_in["machine_type"], r_in["air_temperature_k"], r_in["process_temperature_k"],
             r_in["rotational_speed_rpm"], r_in["torque_nm"], r_in["tool_wear_min"], res["failure_probability"],
             int(res["predicted_failure"]), res["risk_level"], res["anomaly_score"], per_row)
            for r_in, res in zip(X.to_dict("records"), results, strict=True)
        ]
        with self._conn() as c:
            c.executemany("INSERT INTO predictions (ts,model_version,source,machine_type,air_temperature_k,process_temperature_k,"
                          "rotational_speed_rpm,torque_nm,tool_wear_min,failure_probability,predicted_failure,risk_level,"
                          "anomaly_score,latency_ms) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)

    def log_error(self, endpoint: str, kind: str) -> None:
        with self._conn() as c:
            c.execute("INSERT INTO errors (ts, endpoint, kind) VALUES (?,?,?)", (datetime.now(UTC).isoformat(timespec="seconds"), endpoint, kind))

    def recent(self, limit: int = 5000) -> pd.DataFrame:
        with self._conn() as c:
            return pd.read_sql_query("SELECT * FROM predictions ORDER BY id DESC LIMIT ?", c, params=(limit,))

    def summary(self) -> dict:
        with self._conn() as c:
            n, fails, avg_lat = c.execute("SELECT COUNT(*), COALESCE(SUM(predicted_failure),0), AVG(latency_ms) FROM predictions").fetchone()
            n_err = c.execute("SELECT COUNT(*) FROM errors").fetchone()[0]
            lat = [r[0] for r in c.execute("SELECT latency_ms FROM predictions ORDER BY id DESC LIMIT 2000")]
        return {
            "total_predictions": int(n), "predicted_failures": int(fails),
            "predicted_failure_rate": float(fails / n) if n else None,
            "logged_errors": int(n_err), "avg_latency_ms": float(avg_lat) if avg_lat is not None else None,
            "p95_latency_ms": float(np.percentile(lat, 95)) if lat else None,
        }


def save_reference(reference: dict, path: Path) -> None:
    Path(path).write_text(json.dumps(reference))


def load_reference(path: Path) -> dict:
    return json.loads(Path(path).read_text())


__all__ = ["INPUT_FEATURES", "MonitoringStore", "build_reference", "compute_drift", "psi"]
