"""SHAP explanations for the *deployed* classifier (after the same preprocessing).

SHAP values describe how the model's output changes with each input relative to a background;
they show what the model relies on, NOT physical causes of failure.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

SUPPORTED = ("XGBClassifier", "LGBMClassifier", "RandomForestClassifier")


class ModelExplainer:
    def __init__(self, pipeline: Pipeline) -> None:
        self._pre = pipeline[:-1]
        self._clf = pipeline[-1]
        self.supported = type(self._clf).__name__ in SUPPORTED
        self._explainer = None
        if self.supported:
            import shap

            self._explainer = shap.TreeExplainer(self._clf)

    @property
    def units(self) -> str:
        return "probability" if type(self._clf).__name__ == "RandomForestClassifier" else "log-odds (raw model margin)"

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self._pre.transform(X)

    def shap_values(self, X: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
        if not self.supported:
            raise NotImplementedError("SHAP explanations are only available for tree-based models.")
        Xt = self.transform(X)
        vals = self._explainer.shap_values(Xt)
        if isinstance(vals, list):
            vals = vals[-1]
        vals = np.asarray(vals)
        if vals.ndim == 3:
            vals = vals[:, :, -1]
        return vals, Xt

    def local(self, X: pd.DataFrame, top_k: int = 5) -> list[list[dict]]:
        """Top-k feature contributions per row (positive => pushes toward 'failure')."""
        vals, Xt = self.shap_values(X)
        out = []
        for i in range(len(Xt)):
            order = np.argsort(-np.abs(vals[i]))[:top_k]
            out.append(
                [
                    {
                        "feature": str(Xt.columns[j]),
                        "value": float(Xt.iloc[i, j]),
                        "contribution": float(vals[i, j]),
                        "direction": "increases_risk" if vals[i, j] > 0 else "decreases_risk",
                    }
                    for j in order
                ]
            )
        return out

    def global_importance(self, X: pd.DataFrame) -> pd.DataFrame:
        vals, Xt = self.shap_values(X)
        return (
            pd.DataFrame({"feature": Xt.columns, "mean_abs_shap": np.abs(vals).mean(axis=0)})
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True)
        )
