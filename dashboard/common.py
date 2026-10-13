
from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from pmip.config import load_settings
from pmip.data import derive_failure_type, load_raw_dataset
from pmip.monitoring import MonitoringStore
from pmip.registry import ModelNotAvailableError, load_bundle
from pmip.schema import FEATURE_RANGES, INPUT_FEATURES
from pmip.service import PredictionService

DEMO_BANNER = ("Demo / historical benchmark data. This dashboard is **not connected to physical machines**; "
               "the UCI AI4I 2020 dataset is itself synthetic.")


def settings():
    return load_settings()


@st.cache_resource(show_spinner="Loading model...")
def service() -> PredictionService | None:
    s = settings()
    try:
        bundle = load_bundle(s.resolve(s.paths.models_dir))
    except ModelNotAvailableError as exc:
        st.session_state["load_error"] = str(exc)
        return None
    return PredictionService(bundle, s, MonitoringStore(s.resolve(s.paths.monitoring_db)))


def require_service() -> PredictionService:
    svc = service()
    if svc is None:
        st.error(st.session_state.get("load_error", "No model available."))
        st.info("Train one with `make train` (needs the dataset: `make data`).")
        st.stop()
    return svc


@st.cache_data(show_spinner="Loading dataset...")
def dataset() -> pd.DataFrame | None:
    s = settings()
    try:
        df = load_raw_dataset(s.resolve(s.paths.raw_data))
    except FileNotFoundError:
        return None
    df["failure_type"] = derive_failure_type(df)
    return df


def require_dataset() -> pd.DataFrame:
    df = dataset()
    if df is None:
        st.warning("Dataset not found. Run `make data` to download the UCI AI4I 2020 dataset.")
        st.stop()
    return df


def reports_dir():
    s = settings()
    return s.resolve(s.paths.reports_dir)


def read_csv_report(name: str) -> pd.DataFrame | None:
    p = reports_dir() / name
    return pd.read_csv(p) if p.exists() else None


def read_json_report(name: str) -> dict | None:
    p = reports_dir() / name
    return json.loads(p.read_text()) if p.exists() else None


def show_figure(name: str, caption: str = "") -> None:
    p = reports_dir() / "figures" / name
    if p.exists():
        st.image(str(p), caption=caption, width="stretch")
    else:
        st.caption(f"Figure `{name}` not generated yet (run `make train` / `make eda`).")


__all__ = ["DEMO_BANNER", "FEATURE_RANGES", "INPUT_FEATURES"]
