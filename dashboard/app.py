
import streamlit as st

from common import DEMO_BANNER
from sections import anomaly, batch, drift, explain, explore, info, overview, performance, risk, single

st.set_page_config(page_title="PMIP | Predictive Maintenance", page_icon="⚙️", layout="wide")
st.markdown("<style>.block-container{padding-top:1.5rem}[data-testid=stMetricValue]{font-size:1.6rem}</style>", unsafe_allow_html=True)

PAGES = {
    "1. Overview": overview, "2. Sensor exploration": explore, "3. Single machine prediction": single,
    "4. Batch CSV prediction": batch, "5. Failure-risk analysis": risk, "6. Anomaly detection": anomaly,
    "7. Model performance": performance, "8. SHAP explainability": explain, "9. Drift monitoring": drift,
    "10. Model & dataset info": info,
}
with st.sidebar:
    st.title("⚙️ PMIP")
    st.caption("Predictive Maintenance Intelligence Platform")
    choice = st.radio("Navigate", list(PAGES), label_visibility="collapsed")
    st.divider()
    st.caption(DEMO_BANNER)
st.header(choice.split(". ", 1)[1])
PAGES[choice].render()
