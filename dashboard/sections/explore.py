import plotly.express as px
import streamlit as st

from common import require_dataset
from pmip.features import ALL_NUMERIC, SensorFeatureEngineer


def render() -> None:
    df = require_dataset()
    feats = SensorFeatureEngineer().transform(df)
    feats["machine_failure"] = df["machine_failure"].astype(str).to_numpy()
    feats["failure_type"] = df["failure_type"].to_numpy()
    c = st.columns(3)
    c[0].metric("Rows", f"{len(df):,}")
    c[1].metric("Failures", f"{int(df['machine_failure'].sum()):,}")
    c[2].metric("Failure rate", f"{df['machine_failure'].mean():.2%}")
    col = st.selectbox("Feature", ALL_NUMERIC)
    st.plotly_chart(px.histogram(feats, x=col, color="machine_failure", barmode="overlay", histnorm="probability density",
                                 opacity=0.65, marginal="box"), width="stretch")
    x, y = st.columns(2)
    xa = x.selectbox("X axis", ALL_NUMERIC, index=2)
    ya = y.selectbox("Y axis", ALL_NUMERIC, index=3)
    st.plotly_chart(px.scatter(feats.sample(min(4000, len(feats)), random_state=0), x=xa, y=ya, color="machine_failure", opacity=0.6),
                    width="stretch")
    st.plotly_chart(px.imshow(feats[ALL_NUMERIC].corr(), text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                              title="Correlation (engineered features included)"), width="stretch")
    st.subheader("Failure-type distribution")
    st.bar_chart(df.loc[df["machine_failure"] == 1, "failure_type"].value_counts())
    with st.expander("Raw sample"):
        st.dataframe(df.head(200))
