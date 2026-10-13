"""Reusable feature-engineering and preprocessing (identical at training and inference time)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .schema import MACHINE_TYPES, NUMERIC_FEATURES, validate_features

ENGINEERED_FEATURES = ["temp_diff_k", "mechanical_power_w", "torque_wear_product"]
ALL_NUMERIC = NUMERIC_FEATURES + ENGINEERED_FEATURES

FEATURE_RATIONALE = {
    "temp_diff_k": "process - air temperature: small values mean poor heat dissipation (heat-dissipation failures).",
    "mechanical_power_w": "torque x angular speed (2*pi*rpm/60), in watts: very low/high power relates to power failures.",
    "torque_wear_product": "torque x tool wear: accumulated mechanical strain (overstrain failures).",
}


class SensorFeatureEngineer(BaseEstimator, TransformerMixin):
    """Validate raw sensor inputs and append physically-motivated derived features.

    Stateless (nothing is learned from data), so there is no train/test leakage.
    """

    def fit(self, X: pd.DataFrame, y=None) -> SensorFeatureEngineer:
        validate_features(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = validate_features(X)
        df["temp_diff_k"] = df["process_temperature_k"] - df["air_temperature_k"]
        df["mechanical_power_w"] = df["torque_nm"] * df["rotational_speed_rpm"] * 2 * np.pi / 60.0
        df["torque_wear_product"] = df["torque_nm"] * df["tool_wear_min"]
        return df[ALL_NUMERIC + ["machine_type"]]

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.array(ALL_NUMERIC + ["machine_type"])


def build_preprocessor(scale: bool) -> ColumnTransformer:
    """One-hot encode machine type; scale numerics only for scale-sensitive models."""
    numeric = StandardScaler() if scale else "passthrough"
    ct = ColumnTransformer(
        [
            ("num", numeric, ALL_NUMERIC),
            ("cat", OneHotEncoder(categories=[list(MACHINE_TYPES)], sparse_output=False), ["machine_type"]),
        ],
        verbose_feature_names_out=False,
    )
    return ct.set_output(transform="pandas")
