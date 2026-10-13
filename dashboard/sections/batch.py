import pandas as pd
import plotly.express as px
import streamlit as st

from common import require_service
from pmip.schema import SchemaValidationError, standardize_columns


def render() -> None:
    svc = require_service()
    st.write("Upload a CSV with columns `Type`/`machine_type`, `Air temperature [K]`, `Process temperature [K]`, "
             "`Rotational speed [rpm]`, `Torque [Nm]`, `Tool wear [min]` (raw UCI or snake_case names).")
    limit = svc.settings.api.max_batch_size * 20
    up = st.file_uploader("CSV file", type="csv")
    if up is None:
        return
    try:
        df = standardize_columns(pd.read_csv(up))
    except Exception:
        st.error("Could not read the file as CSV.")
        return
    if len(df) > limit:
        st.error(f"File too large for the dashboard ({len(df):,} rows; max {limit:,}).")
        return
    with st.spinner(f"Scoring {len(df):,} rows..."):
        try:
            res = pd.DataFrame(svc.predict(df, log=False, source="dashboard")).drop(columns=["top_contributions", "failure_mode_probabilities"])
        except SchemaValidationError as exc:
            st.error("Invalid input:")
            for i in exc.issues:
                st.write(f"- {i}")
            return
    out = pd.concat([df.reset_index(drop=True), res], axis=1)
    c = st.columns(3)
    c[0].metric("Rows", f"{len(out):,}")
    c[1].metric("Predicted failures", int(out["predicted_failure"].sum()))
    c[2].metric("Anomalous rows", int(out["is_anomalous"].sum()))
    st.plotly_chart(px.histogram(out, x="failure_probability", color="risk_level", nbins=40,
                                 color_discrete_map={"LOW": "#66bb6a", "MEDIUM": "#ffca28", "HIGH": "#ef5350"}), width="stretch")
    st.dataframe(out.sort_values("failure_probability", ascending=False), width="stretch")
    st.download_button("Download results (CSV)", out.to_csv(index=False).encode(), "pmip_predictions.csv", "text/csv")
