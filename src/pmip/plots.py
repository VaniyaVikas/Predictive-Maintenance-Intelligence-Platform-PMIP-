"""Matplotlib/Seaborn figure helpers (all functions save to disk and close the figure)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import precision_recall_curve, roc_curve  # noqa: E402

sns.set_theme(style="whitegrid", context="notebook")


def _save(fig, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_pr_roc(y, proba, path: Path, title: str = "") -> Path:
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    p, r, _ = precision_recall_curve(y, proba)
    ax[0].plot(r, p)
    ax[0].axhline(np.mean(y), ls="--", c="grey", label=f"no-skill ({np.mean(y):.3f})")
    ax[0].set(xlabel="Recall", ylabel="Precision", title=f"Precision-Recall {title}")
    ax[0].legend()
    fpr, tpr, _ = roc_curve(y, proba)
    ax[1].plot(fpr, tpr)
    ax[1].plot([0, 1], [0, 1], "--", c="grey")
    ax[1].set(xlabel="False positive rate", ylabel="True positive rate", title=f"ROC {title}")
    return _save(fig, path)


def plot_confusion(cm: np.ndarray, labels: list[str], path: Path, title: str = "Confusion matrix") -> Path:
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax, cbar=False)
    ax.set(xlabel="Predicted", ylabel="Actual", title=title)
    return _save(fig, path)


def plot_calibration(y, raw, calibrated, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    for name, p in [("raw score", raw), ("calibrated", calibrated)]:
        try:
            frac, mean = calibration_curve(y, p, n_bins=6, strategy="quantile")
            ax.plot(mean, frac, marker="o", label=name)
        except ValueError:
            pass
    ax.plot([0, 1], [0, 1], "--", c="grey")
    ax.set(xlabel="Mean predicted probability", ylabel="Observed failure rate", title="Reliability (test set)")
    ax.legend()
    return _save(fig, path)


def plot_threshold_tradeoff(curve, chosen: float, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(curve["threshold"], curve["precision"], label="precision")
    ax.plot(curve["threshold"], curve["recall"], label="recall")
    ax.plot(curve["threshold"], curve["f2"], label="F2", ls=":")
    ax.axvline(chosen, c="red", ls="--", label=f"chosen = {chosen:.3f}")
    ax.set(xlabel="Decision threshold", ylabel="Score", title="Detection vs. false-alarm trade-off (validation)")
    ax.legend()
    return _save(fig, path)


def plot_anomaly_hist(normal, failure, flag: float, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(normal, bins=30, alpha=0.6, density=True, label="no failure")
    if len(failure):
        ax.hist(failure, bins=30, alpha=0.6, density=True, label="failure")
    ax.axvline(flag, c="red", ls="--", label=f"flag >= {flag:.2f}")
    ax.set(xlabel="Anomaly score (percentile vs. normal training rows)", ylabel="Density", title="Anomaly scores (test set)")
    ax.legend()
    return _save(fig, path)


def plot_shap(explainer, X, summary_path: Path, bar_path: Path, max_rows: int = 1500) -> None:
    import shap

    X = X.iloc[:max_rows]
    vals, Xt = explainer.shap_values(X)
    plt.figure(figsize=(8, 5))
    shap.summary_plot(vals, Xt, show=False)
    _save(plt.gcf(), summary_path)
    plt.figure(figsize=(7, 4))
    shap.summary_plot(vals, Xt, plot_type="bar", show=False)
    _save(plt.gcf(), bar_path)
