import streamlit as st

from common import require_service
from pmip.features import FEATURE_RATIONALE
from pmip.registry import list_versions


def render() -> None:
    svc = require_service()
    m = svc.bundle.metadata
    st.subheader("Model")
    st.json({k: m.get(k) for k in ["version", "created_utc", "model", "variant", "selection_metric", "decision_threshold",
                                   "threshold_objective", "calibration", "seed"]})
    st.write("Available versions:", list_versions(svc.settings.resolve(svc.settings.paths.models_dir)))
    st.subheader("Test metrics with 95% bootstrap intervals")
    st.json({"point_estimates": m["test_metrics"], "ci95": m.get("test_metrics_ci95")})
    st.subheader("Failure-mode model (conditional on failure)")
    st.json(m.get("failure_type_model"))
    st.subheader("Engineered features")
    st.table({"feature": list(FEATURE_RATIONALE), "why": list(FEATURE_RATIONALE.values())})
    st.subheader("Dataset")
    st.json(m.get("dataset"))
    st.markdown("Source: UCI AI4I 2020 Predictive Maintenance Dataset (CC BY 4.0), "
                "https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset")
    st.subheader("Limitations")
    for item in m.get("limitations", []):
        st.markdown(f"- {item}")
