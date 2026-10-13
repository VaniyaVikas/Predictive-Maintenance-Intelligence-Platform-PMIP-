"""Dataset loading, quality checks, label construction and splitting."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import Settings
from .schema import FAILURE_MODES, INPUT_FEATURES, NUMERIC_FEATURES, TARGET, standardize_columns

UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip"


def load_raw_dataset(path: Path) -> pd.DataFrame:
    """Load the AI4I 2020 CSV and standardise column names."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run `python -m pmip.cli download` "
            "(or `make data`) to fetch the UCI AI4I 2020 dataset."
        )
    df = standardize_columns(pd.read_csv(path))
    required = INPUT_FEATURES + [TARGET] + FAILURE_MODES
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {missing}")
    return df


def derive_failure_type(df: pd.DataFrame) -> pd.Series:
    """Build a single failure-type label.

    ``NONE`` (no failure), a single mode (``TWF``/``HDF``/``PWF``/``OSF``/``RNF``), ``MULTIPLE`` when
    several mode flags are set, and ``UNSPECIFIED`` when ``machine_failure==1`` but no mode flag is set
    (a documented inconsistency in the source data).
    """
    modes = df[FAILURE_MODES].astype(int)
    n_modes = modes.sum(axis=1)
    label = pd.Series("NONE", index=df.index, dtype="object")
    single = n_modes == 1
    label[single] = modes[single].idxmax(axis=1)
    label[n_modes > 1] = "MULTIPLE"
    label[(df[TARGET] == 1) & (n_modes == 0)] = "UNSPECIFIED"
    return label


def quality_report(df: pd.DataFrame) -> dict:
    """Missing values, duplicates, invalid values, class balance and label inconsistencies."""
    from .schema import FEATURE_RANGES

    n = len(df)
    report: dict = {
        "n_rows": n,
        "n_columns": df.shape[1],
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "missing_values": {c: int(v) for c, v in df.isna().sum().items() if v},
        "duplicate_rows": int(df.duplicated().sum()),
    }
    if "udi" in df.columns:
        report["duplicate_ids"] = int(df["udi"].duplicated().sum())
    report["out_of_range_values"] = {
        c: int(((df[c] < lo) | (df[c] > hi)).sum()) for c, (lo, hi) in FEATURE_RANGES.items() if c in df
    }
    fail = int(df[TARGET].sum())
    report["class_balance"] = {
        "failures": fail,
        "non_failures": n - fail,
        "failure_rate": fail / n if n else None,
        "imbalance_ratio": (n - fail) / fail if fail else None,
    }
    if all(m in df.columns for m in FAILURE_MODES):
        any_mode = df[FAILURE_MODES].sum(axis=1) > 0
        report["failure_mode_counts"] = {m: int(df[m].sum()) for m in FAILURE_MODES}
        report["label_inconsistencies"] = {
            "failure_flag_without_any_mode": int(((df[TARGET] == 1) & ~any_mode).sum()),
            "mode_flag_without_failure_flag": int(((df[TARGET] == 0) & any_mode).sum()),
        }
    return report


def dataset_fingerprint(df: pd.DataFrame) -> str:
    """Stable SHA-256 fingerprint of the dataframe contents (for experiment tracking)."""
    h = hashlib.sha256(pd.util.hash_pandas_object(df, index=False).values.tobytes())
    h.update(",".join(df.columns).encode())
    return h.hexdigest()


def split_data(df: pd.DataFrame, settings: Settings) -> dict[str, pd.DataFrame]:
    """Stratified train/validation/test split on the failure label.

    The AI4I data has no timestamp or machine identifier (``UDI`` is just a row index and
    ``Product ID`` is a quality-variant + serial number), so a chronological or grouped split
    is not possible; see docs/DATA.md for the consequences.
    """
    seed = settings.seed
    trainval, test = train_test_split(df, test_size=settings.split.test_size, stratify=df[TARGET], random_state=seed)
    rel_val = settings.split.val_size / (1 - settings.split.test_size)
    train, val = train_test_split(trainval, test_size=rel_val, stratify=trainval[TARGET], random_state=seed)
    return {"train": train, "val": val, "test": test}


def feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Model inputs only: no identifiers, no failure-mode flags (they define the target => leakage)."""
    return df[INPUT_FEATURES].copy()


__all__ = [
    "NUMERIC_FEATURES",
    "UCI_ZIP_URL",
    "dataset_fingerprint",
    "derive_failure_type",
    "feature_frame",
    "load_raw_dataset",
    "quality_report",
    "split_data",
]
