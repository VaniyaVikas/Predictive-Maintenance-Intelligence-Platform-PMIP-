# EDA summary (auto-generated from the loaded data)

- **Shape:** 10,000 rows x 15 columns; missing values: 0; duplicate rows: 0.
- **Class imbalance:** 339 failures (3.39%); roughly 28:1 non-failure to failure. Accuracy is therefore a misleading metric (always predicting 'no failure' scores high); PR-AUC, recall and precision are used instead.
- **Failure types:** {'TWF': 46, 'HDF': 115, 'PWF': 95, 'OSF': 98, 'RNF': 19}. Several modes are rare; failure-type classification has very small support for some classes.
- **Label inconsistencies:** {'failure_flag_without_any_mode': 9, 'mode_flag_without_failure_flag': 18} (documented in the source data; such rows are labelled `UNSPECIFIED`/kept as in the source).
- **Out-of-range values:** {'air_temperature_k': 0, 'process_temperature_k': 0, 'rotational_speed_rpm': 0, 'torque_nm': 0, 'tool_wear_min': 0}.
- **Strongest linear associations with failure:** `torque_nm` (+0.19), `torque_wear_product` (+0.19), `mechanical_power_w` (+0.18), `temp_diff_k` (-0.11). Linear correlation understates threshold-style failure rules, so tree models are expected to do better.
- **Outliers (IQR rule, count):** `air_temperature_k`: 0, `process_temperature_k`: 0, `rotational_speed_rpm`: 418, `torque_nm`: 69, `tool_wear_min`: 0. Outliers in torque/speed are partly the failure signal itself, so they are NOT removed.

## Descriptive statistics
```
                         count     mean     std     min     25%     50%     75%     max
air_temperature_k      10000.0   300.00    2.00   295.3   298.3   300.1   301.5   304.5
process_temperature_k  10000.0   310.01    1.48   305.7   308.8   310.1   311.1   313.8
rotational_speed_rpm   10000.0  1538.78  179.28  1168.0  1423.0  1503.0  1612.0  2886.0
torque_nm              10000.0    39.99    9.97     3.8    33.2    40.1    46.8    76.6
tool_wear_min          10000.0   107.95   63.65     0.0    53.0   108.0   162.0   253.0
```

## Caveats
- The UCI AI4I 2020 dataset is itself a **synthetic** dataset created to mimic industrial data. Conclusions are about this benchmark, not about any real plant.
- No timestamps or machine identifiers exist, so time-based/grouped validation is impossible and no remaining-useful-life (RUL) target exists. This project therefore does NOT estimate RUL.
- Failure-mode flags (TWF, HDF, PWF, OSF, RNF) define the failure label and are excluded from features (leakage).