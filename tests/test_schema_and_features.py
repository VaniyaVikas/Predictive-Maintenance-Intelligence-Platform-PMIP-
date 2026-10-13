import numpy as np
import pandas as pd
import pytest

from pmip.features import ALL_NUMERIC, ENGINEERED_FEATURES, SensorFeatureEngineer, build_preprocessor
from pmip.schema import FAILURE_MODES, INPUT_FEATURES, SchemaValidationError, validate_features


def frame(**over):
    row = {"machine_type": "L", "air_temperature_k": 300.0, "process_temperature_k": 310.0,
           "rotational_speed_rpm": 1500.0, "torque_nm": 40.0, "tool_wear_min": 100.0}
    row.update(over)
    return pd.DataFrame([row])


def test_validate_accepts_valid_and_normalises_type():
    out = validate_features(frame(machine_type=" m "))
    assert list(out.columns) == INPUT_FEATURES and out.loc[0, "machine_type"] == "M"


def test_validate_accepts_raw_uci_column_names(raw_df):
    assert len(validate_features(raw_df.head(5))) == 5


@pytest.mark.parametrize("bad", [{"torque_nm": np.nan}, {"torque_nm": -1}, {"air_temperature_k": 900},
                                 {"machine_type": "X"}, {"tool_wear_min": "abc"}])
def test_validate_rejects_bad_values(bad):
    with pytest.raises(SchemaValidationError):
        validate_features(frame(**bad))


def test_validate_rejects_missing_column_and_empty():
    with pytest.raises(SchemaValidationError, match="missing"):
        validate_features(frame().drop(columns=["torque_nm"]))
    with pytest.raises(SchemaValidationError):
        validate_features(frame().iloc[0:0])


def test_engineered_features_values_and_no_mutation():
    X = frame()
    before = X.copy()
    out = SensorFeatureEngineer().fit(X).transform(X)
    pd.testing.assert_frame_equal(X, before)
    assert out.loc[0, "temp_diff_k"] == pytest.approx(10.0)
    assert out.loc[0, "mechanical_power_w"] == pytest.approx(40 * 1500 * 2 * np.pi / 60)
    assert out.loc[0, "torque_wear_product"] == pytest.approx(4000.0)
    assert set(ENGINEERED_FEATURES) <= set(out.columns)


def test_no_target_leakage_columns_in_features():
    assert not set(FAILURE_MODES) & set(INPUT_FEATURES + ALL_NUMERIC)
    assert "machine_failure" not in INPUT_FEATURES


def test_preprocessor_scaling_fit_on_train_only(df):
    X = df[INPUT_FEATURES]
    fe = SensorFeatureEngineer()
    train, other = fe.transform(X.iloc[:2000]), fe.transform(X.iloc[2000:])
    pre = build_preprocessor(scale=True).fit(train)
    assert pre.transform(train)["torque_nm"].mean() == pytest.approx(0.0, abs=1e-8)
    assert abs(pre.transform(other)["torque_nm"].mean()) > 0  # held-out data is not re-centred
    assert {"machine_type_H", "machine_type_L", "machine_type_M"} <= set(pre.get_feature_names_out())
