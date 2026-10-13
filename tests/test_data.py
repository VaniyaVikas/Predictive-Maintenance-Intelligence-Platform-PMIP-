import pandas as pd
import pytest

from pmip.config import Settings
from pmip.data import derive_failure_type, load_raw_dataset, quality_report, split_data


def test_failure_type_derivation():
    d = pd.DataFrame({"machine_failure": [0, 1, 1, 1, 1], "TWF": [0, 1, 1, 0, 0], "HDF": [0, 0, 1, 0, 0],
                      "PWF": [0, 0, 0, 0, 0], "OSF": [0, 0, 0, 0, 0], "RNF": [0, 0, 0, 0, 0]})
    assert derive_failure_type(d).tolist() == ["NONE", "TWF", "MULTIPLE", "UNSPECIFIED", "UNSPECIFIED"]


def test_split_is_stratified_disjoint_and_complete(df):
    s = split_data(df, Settings())
    idx = [set(p.index) for p in s.values()]
    assert not (idx[0] & idx[1]) and not (idx[0] & idx[2]) and not (idx[1] & idx[2])
    assert sum(len(p) for p in s.values()) == len(df)
    rates = [p["machine_failure"].mean() for p in s.values()]
    assert max(rates) - min(rates) < 0.01


def test_split_reproducible(df):
    a, b = split_data(df, Settings()), split_data(df, Settings())
    assert a["test"].index.equals(b["test"].index)


def test_quality_report(df):
    r = quality_report(df)
    assert r["duplicate_rows"] == 0 and r["class_balance"]["failures"] == int(df["machine_failure"].sum())
    assert "label_inconsistencies" in r


def test_missing_dataset_has_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="download"):
        load_raw_dataset(tmp_path / "nope.csv")
