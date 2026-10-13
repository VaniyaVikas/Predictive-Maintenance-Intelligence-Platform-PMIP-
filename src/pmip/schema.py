"""Dataset schema, column names and input validation (shared by training and inference)."""
from __future__ import annotations

import numpy as np
import pandas as pd

RAW_COLUMN_MAP = {
    "UDI": "udi",
    "Product ID": "product_id",
    "Type": "machine_type",
    "Air temperature [K]": "air_temperature_k",
    "Process temperature [K]": "process_temperature_k",
    "Rotational speed [rpm]": "rotational_speed_rpm",
    "Torque [Nm]": "torque_nm",
    "Tool wear [min]": "tool_wear_min",
    "Machine failure": "machine_failure",
}
FAILURE_MODES = ["TWF", "HDF", "PWF", "OSF", "RNF"]
TARGET = "machine_failure"
IDENTIFIER_COLUMNS = ["udi", "product_id"]

NUMERIC_FEATURES = [
    "air_temperature_k",
    "process_temperature_k",
    "rotational_speed_rpm",
    "torque_nm",
    "tool_wear_min",
]
CATEGORICAL_FEATURES = ["machine_type"]
INPUT_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
MACHINE_TYPES = ("H", "L", "M")

# Generous physical plausibility ranges (reject sensor glitches, unit mistakes, etc.)
FEATURE_RANGES: dict[str, tuple[float, float]] = {
    "air_temperature_k": (250.0, 350.0),
    "process_temperature_k": (250.0, 360.0),
    "rotational_speed_rpm": (1.0, 10000.0),
    "torque_nm": (0.0, 500.0),
    "tool_wear_min": (0.0, 1000.0),
}


class SchemaValidationError(ValueError):
    """Raised when input data does not satisfy the model's input contract."""

    def __init__(self, issues: list[str]):
        self.issues = issues
        super().__init__("; ".join(issues))


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename raw UCI column names (e.g. ``Torque [Nm]``) to internal snake_case names."""
    return df.rename(columns={k: v for k, v in RAW_COLUMN_MAP.items() if k in df.columns})


def validate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalise model inputs. Returns a clean copy with ``INPUT_FEATURES`` only."""
    if not isinstance(df, pd.DataFrame):
        raise SchemaValidationError(["input must be a pandas DataFrame"])
    df = standardize_columns(df)
    missing = [c for c in INPUT_FEATURES if c not in df.columns]
    if missing:
        raise SchemaValidationError([f"missing required columns: {missing}"])
    if len(df) == 0:
        raise SchemaValidationError(["input contains no rows"])

    out = df[INPUT_FEATURES].copy()
    issues: list[str] = []
    for col in NUMERIC_FEATURES:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype(float)
        bad = ~np.isfinite(out[col].to_numpy())
        if bad.any():
            issues.append(f"{col}: {int(bad.sum())} missing/non-numeric/non-finite value(s)")
            continue
        lo, hi = FEATURE_RANGES[col]
        oor = (out[col] < lo) | (out[col] > hi)
        if oor.any():
            issues.append(f"{col}: {int(oor.sum())} value(s) outside plausible range [{lo}, {hi}]")
    out["machine_type"] = out["machine_type"].astype(str).str.strip().str.upper()
    bad_type = ~out["machine_type"].isin(MACHINE_TYPES)
    if bad_type.any():
        issues.append(f"machine_type: values must be one of {list(MACHINE_TYPES)}")
    if issues:
        raise SchemaValidationError(issues)
    return out
