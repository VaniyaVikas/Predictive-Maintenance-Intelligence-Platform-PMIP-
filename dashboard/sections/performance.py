import plotly.express as px
import streamlit as st

from common import read_csv_report, show_figure


def render() -> None:
    comp = read_csv_report("model_comparison.csv")
    if comp is None:
        st.warning("No comparison found. Run `make train`.")
        return
    st.caption("All rows are VALIDATION-split results (the test split is used once, for the selected model only). "
               "Selection metric: average precision (PR-AUC). Accuracy is deliberately not shown.")
    comp["label"] = comp["model"] + " (" + comp["variant"] + ")"
    st.dataframe(comp.drop(columns=["label"]).round(3), width="stretch")
    st.plotly_chart(px.bar(comp, x="label", y=["average_precision", "recall", "precision"], barmode="group"), width="stretch")
    c = st.columns(2)
    with c[0]:
        show_figure("pr_roc_test.png", "Selected model - test set")
        show_figure("threshold_tradeoff_val.png", "Threshold choice (validation)")
    with c[1]:
        show_figure("confusion_matrix_test.png", "Confusion matrix - test set")
        show_figure("calibration_test.png", "Calibration - test set")
