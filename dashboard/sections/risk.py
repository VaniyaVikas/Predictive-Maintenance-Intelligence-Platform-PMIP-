import plotly.express as px
import streamlit as st

from common import read_csv_report, require_service
from pmip.evaluation import compute_metrics, threshold_curve


def render() -> None:
    svc = require_service()
    t = read_csv_report("test_predictions.csv")
    if t is None:
        st.warning("No test predictions found. Run `make train`.")
        return
    st.caption("Based on the held-out TEST split of the benchmark dataset (historical, not live data).")
    risk = svc.settings.risk
    t["risk_band"] = t["failure_probability"].apply(lambda p: "HIGH" if p >= risk.high else "MEDIUM" if p >= risk.medium else "LOW")
    g = t.groupby("risk_band")["machine_failure"].agg(rows="count", failures="sum", observed_failure_rate="mean").reindex(["LOW", "MEDIUM", "HIGH"])
    st.subheader("Observed failure rate per risk band")
    st.dataframe(g)
    st.subheader("Detection vs. false alarms")
    thr = st.slider("Decision threshold (exploration only)", 0.01, 0.99, float(svc.bundle.threshold), 0.01)
    m = compute_metrics(t["machine_failure"], t["failure_probability"], thr)
    c = st.columns(5)
    for col, (k, lab) in zip(c, [("recall", "Recall"), ("precision", "Precision"), ("fn", "Missed failures"), ("fp", "False alarms"), ("f2", "F2")], strict=True):
        col.metric(lab, f"{m[k]:.3f}" if isinstance(m[k], float) else m[k])
    st.caption(f"The deployed threshold ({svc.bundle.threshold:.3f}) was chosen on validation data. Moving it here and "
               "reading test results is for understanding the trade-off only - do not re-tune on the test set.")
    cur = threshold_curve(t["machine_failure"], t["failure_probability"])
    st.plotly_chart(px.line(cur, x="threshold", y=["precision", "recall", "f2"]), width="stretch")
    st.subheader("Missed failures by type")
    miss = t[(t["machine_failure"] == 1) & (t["failure_probability"] < thr)]["failure_type"].value_counts()
    st.bar_chart(miss) if len(miss) else st.success("No missed failures at this threshold.")
