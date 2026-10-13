"""Metrics, threshold selection, bootstrap intervals and error analysis."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(y_true, proba, threshold: float) -> dict:
    """Threshold-dependent and threshold-free metrics. Accuracy is reported but never used for selection."""
    y = np.asarray(y_true).astype(int)
    p = np.asarray(proba, dtype=float)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    two_classes = len(np.unique(y)) == 2
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(fbeta_score(y, pred, beta=1, zero_division=0)),
        "f2": float(fbeta_score(y, pred, beta=2, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, p)) if two_classes else float("nan"),
        "average_precision": float(average_precision_score(y, p)) if two_classes else float("nan"),
        "brier": float(brier_score_loss(y, np.clip(p, 0, 1))),
        "accuracy": float(accuracy_score(y, pred)),
        "false_alarm_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def select_threshold(y_true, proba, beta: float = 2.0, min_precision: float = 0.0) -> float:
    """Pick the probability threshold maximising F-beta (optionally subject to a precision floor)."""
    y = np.asarray(y_true).astype(int)
    prec, rec, thr = precision_recall_curve(y, proba)
    prec, rec = prec[:-1], rec[:-1]
    b2 = beta**2
    with np.errstate(divide="ignore", invalid="ignore"):
        fb = np.where((b2 * prec + rec) > 0, (1 + b2) * prec * rec / (b2 * prec + rec), 0.0)
    ok = prec >= min_precision
    if not ok.any():
        ok = np.ones_like(ok, dtype=bool)
    fb = np.where(ok, fb, -1.0)
    if len(thr) == 0:
        return 0.5
    best = np.flatnonzero(fb == fb.max())
    return float(thr[best[-1]])  # highest threshold among ties => fewer false alarms


def bootstrap_ci(y_true, proba, threshold: float, n: int = 300, seed: int = 0) -> dict:
    """Percentile bootstrap 95% CIs (matters because the test set holds only a few dozen failures)."""
    if n <= 0:
        return {}
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y_true).astype(int), np.asarray(proba, dtype=float)
    rows = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].sum() == 0 or y[idx].sum() == len(idx):
            continue
        m = compute_metrics(y[idx], p[idx], threshold)
        rows.append([m["precision"], m["recall"], m["f1"], m["average_precision"]])
    if not rows:
        return {}
    arr = np.array(rows)
    names = ["precision", "recall", "f1", "average_precision"]
    return {k: [float(np.percentile(arr[:, i], 2.5)), float(np.percentile(arr[:, i], 97.5))] for i, k in enumerate(names)}


def threshold_curve(y_true, proba) -> pd.DataFrame:
    """Precision/recall/false-alarm trade-off across thresholds (used for plots and the dashboard)."""
    y, p = np.asarray(y_true).astype(int), np.asarray(proba, dtype=float)
    grid = np.unique(np.clip(np.concatenate([np.linspace(0.01, 0.99, 99), np.quantile(p, [0.9, 0.95, 0.99])]), 0, 1))
    rows = [compute_metrics(y, p, t) for t in grid]
    return pd.DataFrame(rows)[["threshold", "precision", "recall", "f1", "f2", "false_alarm_rate", "fp", "fn"]]


def error_analysis(frame: pd.DataFrame, y_true, proba, threshold: float) -> dict:
    """Break down false negatives / false positives (frame must contain ``failure_type``)."""
    y = np.asarray(y_true).astype(int)
    pred = (np.asarray(proba) >= threshold).astype(int)
    fn_mask, fp_mask = (y == 1) & (pred == 0), (y == 0) & (pred == 1)
    out = {"false_negatives": int(fn_mask.sum()), "false_positives": int(fp_mask.sum())}
    if "failure_type" in frame:
        ft = frame["failure_type"].to_numpy()
        out["missed_failures_by_type"] = pd.Series(ft[fn_mask]).value_counts().to_dict()
        out["detected_failures_by_type"] = pd.Series(ft[(y == 1) & (pred == 1)]).value_counts().to_dict()
    num = [c for c in ["air_temperature_k", "process_temperature_k", "rotational_speed_rpm", "torque_nm", "tool_wear_min"] if c in frame]
    if fn_mask.any() and num:
        out["false_negative_feature_means"] = frame.loc[fn_mask, num].mean().round(3).to_dict()
    if fp_mask.any() and num:
        out["false_positive_feature_means"] = frame.loc[fp_mask, num].mean().round(3).to_dict()
    return out
