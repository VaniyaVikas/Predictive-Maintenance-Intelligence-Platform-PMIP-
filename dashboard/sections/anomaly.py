import plotly.express as px
import streamlit as st

from common import read_csv_report, require_service


def render() -> None:
    svc = require_service()
    st.warning("**Anomalous is not the same as failed.** The detector flags sensor combinations that are unusual compared "
               "with normal training operation. The dataset has no true anomaly labels, so quality is only sanity-checked "
               "against failure labels.")
    det = svc.bundle.anomaly_detector
    st.write(f"Flag rule: score >= **{det.flag_percentile:.2f}** (more unusual than {det.flag_percentile:.0%} of normal training rows).")
    t = read_csv_report("test_predictions.csv")
    if t is not None:
        t["status"] = t["machine_failure"].map({0: "no failure", 1: "failure"})
        st.plotly_chart(px.histogram(t, x="anomaly_score", color="status", barmode="overlay", histnorm="probability density", nbins=40,
                                     opacity=0.7), width="stretch")
        st.plotly_chart(px.scatter(t, x="rotational_speed_rpm", y="torque_nm", color="anomaly_score", symbol="status",
                                   color_continuous_scale="Turbo", opacity=0.7), width="stretch")
        st.json(svc.bundle.metadata.get("anomaly_detector", {}))
