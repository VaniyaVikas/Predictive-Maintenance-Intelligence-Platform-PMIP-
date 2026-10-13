import plotly.express as px
import streamlit as st

from common import dataset, require_service
from pmip.monitoring import compute_drift
from pmip.schema import INPUT_FEATURES, standardize_columns


def render() -> None:
    svc = require_service()
    cfg = svc.settings.monitoring
    st.info("Data drift = the *input* distribution moved away from the training reference (PSI + KS test). It is **not** proof "
            "of concept drift or accuracy loss: that needs ground-truth failure labels. Drift only triggers a human review; "
            "models are never retrained automatically.")
    st.write(f"PSI thresholds: warning >= {cfg.psi_warning}, alert >= {cfg.psi_alert}; minimum {cfg.min_samples} rows.")
    src = st.radio("Current data source", ["Logged API predictions", "Upload CSV", "Simulated shift on demo data"], horizontal=True)
    cur = None
    if src == "Logged API predictions":
        logged = svc.store.recent() if svc.store else None
        if logged is None or logged.empty:
            st.info("No predictions logged yet. Call the API's /predict endpoints first.")
        else:
            cur = logged[INPUT_FEATURES]
            st.caption(f"{len(cur):,} logged requests")
    elif src == "Upload CSV":
        up = st.file_uploader("CSV", type="csv")
        if up:
            cur = standardize_columns(__import__("pandas").read_csv(up))
    else:
        df = dataset()
        if df is None:
            st.warning("Dataset not found (`make data`).")
        else:
            st.caption("SIMULATION: a sample of benchmark data with an artificial offset - not real drift.")
            col = st.selectbox("Feature to shift", ["torque_nm", "air_temperature_k", "tool_wear_min", "rotational_speed_rpm"])
            pct = st.slider("Shift (% of feature std)", 0, 300, 100)
            cur = df[INPUT_FEATURES].sample(min(1000, len(df)), random_state=1).copy()
            cur[col] = cur[col] + cur[col].std() * pct / 100
    if cur is None:
        return
    try:
        res = compute_drift(svc.bundle.reference, cur, cfg)
    except Exception as exc:
        st.error(f"Could not compute drift: {exc}")
        return
    st.dataframe(res.round(4), width="stretch")
    st.plotly_chart(px.bar(res, x="feature", y="psi", color="status", color_discrete_map={"ok": "#66bb6a", "warning": "#ffca28", "alert": "#ef5350", "insufficient_data": "#bdbdbd"}), width="stretch")
    worst = res["status"].tolist()
    if "alert" in worst:
        st.error("Drift ALERT on at least one feature. Investigate sensors/process; consider validating and re-training with approval.")
    elif "warning" in worst:
        st.warning("Drift warning on at least one feature.")
    elif "insufficient_data" in worst:
        st.info("Not enough rows for a reliable drift estimate.")
    else:
        st.success("No input drift detected.")
