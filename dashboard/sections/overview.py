import streamlit as st

from common import DEMO_BANNER, dataset, service


def render() -> None:
    st.info(DEMO_BANNER)
    svc, df = service(), dataset()
    c = st.columns(4)
    c[0].metric("Model", "loaded" if svc else "missing")
    c[1].metric("Version", svc.bundle.version if svc else "-")
    c[2].metric("Dataset", f"{len(df):,} rows" if df is not None else "not found")
    c[3].metric("Decision threshold", f"{svc.bundle.threshold:.3f}" if svc else "-")
    if not svc:
        st.error(st.session_state.get("load_error", "No model available. Run `make train`."))
        return
    m = svc.bundle.metadata["test_metrics"]
    st.subheader("Held-out test performance (selected model)")
    c = st.columns(5)
    for col, k in zip(c, ["precision", "recall", "f1", "average_precision", "roc_auc"], strict=True):
        col.metric(k.replace("_", " ").title(), f"{m[k]:.3f}")
    st.caption("Test set was used once, after model, calibration and threshold were fixed on validation data. "
               "See the confidence intervals on the Model info page: the test set contains few failures.")
    st.subheader("Serving activity (from the API/monitoring log)")
    st.json(svc.store.summary() if svc.store else {})
    st.markdown("**Scope:** failure classification, failure-mode ranking (conditional on failure) and anomaly "
                "detection. **Remaining useful life is not estimated** - the dataset has no run-to-failure sequences.")
