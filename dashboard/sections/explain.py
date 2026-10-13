import pandas as pd
import streamlit as st

from common import read_csv_report, require_service, show_figure


def render() -> None:
    svc = require_service()
    ex = svc.bundle.explainer
    if not ex.supported:
        st.warning("The selected model is not tree-based, so SHAP is unavailable.")
        return
    st.caption("SHAP shows which inputs the deployed model relies on. It reveals associations learned from data, not physical causes.")
    a, b = st.columns(2)
    with a:
        show_figure("shap_summary.png", "Global: SHAP summary (validation sample)")
    with b:
        show_figure("shap_importance.png", "Global: mean |SHAP|")
    t = read_csv_report("test_predictions.csv")
    if t is None:
        return
    st.subheader("Local explanation")
    top = t.sort_values("failure_probability", ascending=False).head(50).reset_index()
    idx = st.selectbox("Pick a test-set machine (highest-risk first)", top["index"].tolist(),
                       format_func=lambda i: f"row {i} - p={t.loc[i, 'failure_probability']:.2f}, actual failure={t.loc[i, 'machine_failure']}")
    row = t.loc[[idx], ["machine_type", "air_temperature_k", "process_temperature_k", "rotational_speed_rpm", "torque_nm", "tool_wear_min"]]
    st.dataframe(row)
    d = pd.DataFrame(ex.local(row, top_k=8)[0])
    st.bar_chart(d.set_index("feature")["contribution"])
    st.caption(f"Units: {ex.units}; positive pushes toward failure.")
