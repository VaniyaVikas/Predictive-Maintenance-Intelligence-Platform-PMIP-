"""Synthetic AI4I-*like* data generator used ONLY for tests, CI and smoke runs.

This is NOT the UCI dataset and NOT real factory data. It reproduces the published failure rules
of AI4I so that the whole pipeline can be exercised offline. Never report metrics from it as real.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def generate_synthetic_ai4i(n: int = 6000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    mtype = rng.choice(["L", "M", "H"], size=n, p=[0.6, 0.3, 0.1])
    air = rng.normal(300.0, 2.0, n)
    proc = air + 10.0 + rng.normal(0, 1.0, n)
    rpm = 1250 + rng.gamma(3.0, 80.0, n)
    torque = np.clip(rng.normal(40.0, 5.5, n), 3.0, None)
    wear = rng.integers(0, 254, n).astype(float)
    power = torque * rpm * 2 * np.pi / 60

    twf = (wear > 215) & (rng.random(n) < 0.08)
    hdf = ((proc - air) < 8.6) & (rpm < 1380)
    pwf = (power < 3500) | (power > 9000)
    thr = np.select([mtype == "L", mtype == "M"], [11000, 12000], default=13000)
    osf = wear * torque > thr
    rnf = rng.random(n) < 0.001
    fail = (twf | hdf | pwf | osf | rnf).astype(int)
    return pd.DataFrame(
        {
            "UDI": np.arange(1, n + 1),
            "Product ID": [f"{t}{i + 10000}" for i, t in enumerate(mtype)],
            "Type": mtype,
            "Air temperature [K]": air.round(1),
            "Process temperature [K]": proc.round(1),
            "Rotational speed [rpm]": rpm.round(0),
            "Torque [Nm]": torque.round(1),
            "Tool wear [min]": wear,
            "Machine failure": fail,
            "TWF": twf.astype(int),
            "HDF": hdf.astype(int),
            "PWF": pwf.astype(int),
            "OSF": osf.astype(int),
            "RNF": rnf.astype(int),
        }
    )
