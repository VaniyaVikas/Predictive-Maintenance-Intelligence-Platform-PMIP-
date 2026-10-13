import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import require_service


def render() -> None:
    svc = require_service()
    with st.form("single"):
        c = st.columns(3)
        mtype = c[0].selectbox("Machine type", ["L", "M", "H"], index=1)
        air = c[0].slider("Air temperature [K]", 290.0, 310.0, 300.0, 0.1)
        proc = c[1].slider("Process temperature [K]", 300.0, 320.0, 310.0, 0.1)
        rpm = c[1].slider("Rotational speed [rpm]", 1100, 3000, 1500, 10)
        torque = c[2].slider("Torque [Nm]", 3.0, 80.0, 40.0, 0.5)
        wear = c[2].slider("Tool wear [min]", 0, 260, 100)
        go_btn = st.form_submit_button("Predict", type="primary")
    if not go_btn:
        st.caption("Adjust the sensor values and press **Predict**.")
        return
    row = pd.DataFrame([{"machine_type": mtype, "air_temperature_k": air, "process_temperature_k": proc,
                         "rotational_speed_rpm": rpm, "torque_nm": torque, "tool_wear_min": wear}])
    with st.spinner("Scoring..."):
        try:
            r = svc.predict(row, explain=True, log=False, source="dashboard")[0]
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            return
    risk = svc.settings.risk
    fig = go.Figure(go.Indicator(mode="gauge+number", value=r["failure_probability"] * 100, number={"suffix": "%"},
        title={"text": f"Failure probability - {r['risk_level']} risk"},
        gauge={"axis": {"range": [0, 100]}, "steps": [
            {"range": [0, risk.medium * 100], "color": "#c8e6c9"},
            {"range": [risk.medium * 100, risk.high * 100], "color": "#fff3b0"},
            {"range": [risk.high * 100, 100], "color": "#ffcdd2"}],
            "threshold": {"line": {"color": "black", "width": 3}, "value": svc.bundle.threshold * 100}}))
    fig.update_layout(height=300, margin=dict(t=60, b=10))
    a, b = st.columns([1, 1])
    a.plotly_chart(fig, width="stretch")
    with b:
        st.metric("Predicted class", "FAILURE LIKELY" if r["predicted_failure"] else "No failure predicted")
        st.metric("Anomaly score", f"{r['anomaly_score']:.2f}", "unusual sensor pattern" if r["is_anomalous"] else "normal pattern", delta_color="inverse" if r["is_anomalous"] else "off")
        if r["likely_failure_mode"]:
            st.write(f"Most likely mode *if* a failure occurs: **{r['likely_failure_mode']}**")
    if r["top_contributions"]:
        st.subheader("Why? Top feature contributions (SHAP)")
        d = pd.DataFrame(r["top_contributions"])
        st.bar_chart(d.set_index("feature")["contribution"])
        st.caption(f"Units: {svc.bundle.explainer.units}. Positive = pushes toward failure. These are model "
                   "associations, not proof of physical cause. Explains the raw model score before calibration.")
    st.caption("Probability is calibrated on benchmark validation data (~3% base failure rate); it is not a guaranteed real-world probability.")
