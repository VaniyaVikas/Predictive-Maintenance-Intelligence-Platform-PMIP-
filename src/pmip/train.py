"""End-to-end training: compare models, tune, calibrate, pick a threshold, evaluate once on test, save."""
from __future__ import annotations

import json
import logging
import platform
import subprocess
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from sklearn.base import clone
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from . import __version__, plots
from .anomaly import SensorAnomalyDetector
from .calibration import PlattCalibrator
from .config import Settings
from .data import (
    dataset_fingerprint,
    derive_failure_type,
    feature_frame,
    load_raw_dataset,
    split_data,
)
from .evaluation import bootstrap_ci, compute_metrics, error_analysis, select_threshold, threshold_curve
from .features import ENGINEERED_FEATURES, FEATURE_RATIONALE
from .models import build_pipeline, lightgbm_available, search_space
from .monitoring import build_reference
from .registry import ModelBundle, new_version, save_bundle
from .schema import TARGET
from .tracking import Tracker

logger = logging.getLogger(__name__)
LIMITATIONS = [
    "Trained on the UCI AI4I 2020 benchmark, which is synthetic; performance does not transfer to real plants without re-validation.",
    "Stratified random split (no timestamps/machine IDs available): optimistic compared with deployment on future data.",
    "Test set has only a few dozen failures; see bootstrap confidence intervals.",
    "Probabilities are Platt-calibrated on a small validation set and reflect the training failure rate (~3%), not a plant's real base rate.",
    "No remaining-useful-life model: the dataset has no run-to-failure sequences.",
    "Anomaly scores flag unusual sensor combinations; they are not failure predictions.",
]


@dataclass
class TrainingResult:
    version: str
    model_dir: Path
    metrics: dict
    comparison: pd.DataFrame


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "unknown"


def _fit_eval(name: str, variant: str, pipe, X_tr, y_tr, X_val, y_val, s: Settings) -> dict:
    t0 = time.perf_counter()
    pipe.fit(X_tr, y_tr)
    val_p = pipe.predict_proba(X_val)[:, 1]
    thr = select_threshold(y_val, val_p, s.training.threshold_beta, s.training.threshold_min_precision)
    m = compute_metrics(y_val, val_p, thr)
    m.update(model=name, variant=variant, fit_seconds=round(time.perf_counter() - t0, 2))
    return {"pipe": pipe, "val_proba": val_p, "metrics": m}


def _train_failure_type_model(train: pd.DataFrame, test: pd.DataFrame, s: Settings):
    """Failure-mode classifier trained on FAILURE rows only (answers: *given* a failure, which mode?)."""
    fails = train[train[TARGET] == 1]
    if len(fails) < 20:
        return None, {"skipped": "fewer than 20 failure rows"}
    counts = fails["failure_type"].value_counts()
    rare = set(counts[counts < 3].index)
    relabel = lambda col: col.where(~col.isin(rare), "OTHER_RARE")  # noqa: E731
    pipe = build_pipeline("random_forest", s.seed)
    pipe.fit(feature_frame(fails), relabel(fails["failure_type"]))
    tf = test[test[TARGET] == 1]
    if tf.empty:
        return pipe, {"note": "no failures in test set"}
    known = set(pipe.classes_)
    y_true = relabel(tf["failure_type"]).where(lambda c: c.isin(known), "OTHER_RARE")
    y_pred = pipe.predict(feature_frame(tf))
    rep = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return pipe, {
        "evaluated_on": "test-set rows with an actual failure",
        "n_eval_rows": int(len(tf)),
        "accuracy": float(rep["accuracy"]),
        "macro_f1": float(rep["macro avg"]["f1-score"]),
        "per_class": {k: v for k, v in rep.items() if k not in ("accuracy", "macro avg", "weighted avg")},
        "note": "Small support per class: treat as indicative, not conclusive.",
    }


def train_pipeline(settings: Settings, df: pd.DataFrame | None = None, use_mlflow: bool = True,
                   save: bool = True, write_reports: bool = True) -> TrainingResult:
    s = settings
    df = (df if df is not None else load_raw_dataset(s.resolve(s.paths.raw_data))).reset_index(drop=True)
    df["failure_type"] = derive_failure_type(df)
    fingerprint = dataset_fingerprint(df.drop(columns=["failure_type"]))
    splits = split_data(df, s)
    tr, va, te = splits["train"], splits["val"], splits["test"]
    X_tr, y_tr, X_va, y_va, X_te, y_te = feature_frame(tr), tr[TARGET], feature_frame(va), va[TARGET], feature_frame(te), te[TARGET]
    spw = float((y_tr == 0).sum() / max(1, (y_tr == 1).sum()))
    reports = s.resolve(s.paths.reports_dir)
    fig_dir = reports / "figures"
    version = new_version()
    tracker = Tracker(use_mlflow, s.paths.mlflow_uri)

    names = [n for n in s.training.models if n != "lightgbm" or lightgbm_available()]
    with tracker.run(f"train-{version}") as _parent:
        tracker.tags({"version": version, "git_commit": _git_commit(), "dataset_sha256": fingerprint[:16]})
        tracker.params({
            "seed": s.seed, "n_rows": len(df), "n_train": len(tr), "n_val": len(va), "n_test": len(te),
            "split": s.split.model_dump(), "training": s.training.model_dump(), "dataset_sha256": fingerprint,
            "engineered_features": ENGINEERED_FEATURES, "pmip_version": __version__,
        })
        results: list[dict] = []
        for name in names:
            with tracker.run(name, nested=True):
                r = _fit_eval(name, "untuned", build_pipeline(name, s.seed, spw), X_tr, y_tr, X_va, y_va, s)
                tracker.params({"model": name, "variant": "untuned"})
                tracker.metrics(r["metrics"], "val_")
                results.append(r)
                logger.info("%s untuned val AP=%.3f", name, r["metrics"]["average_precision"])

        if s.training.tuning_enabled:
            cv = StratifiedKFold(s.training.cv_folds, shuffle=True, random_state=s.seed)
            for name in [n for n in s.training.tune_models if n in names]:
                with tracker.run(f"{name}-tuned", nested=True):
                    search = RandomizedSearchCV(
                        build_pipeline(name, s.seed, spw), search_space(name), n_iter=s.training.tuning_n_iter,
                        scoring="average_precision", cv=cv, random_state=s.seed, n_jobs=1, refit=True,
                    )
                    t0 = time.perf_counter()
                    search.fit(X_tr, y_tr)  # training split only; validation stays untouched by the search
                    val_p = search.best_estimator_.predict_proba(X_va)[:, 1]
                    thr = select_threshold(y_va, val_p, s.training.threshold_beta, s.training.threshold_min_precision)
                    m = compute_metrics(y_va, val_p, thr)
                    m.update(model=name, variant="tuned", cv_average_precision=float(search.best_score_),
                             fit_seconds=round(time.perf_counter() - t0, 2))
                    tracker.params({"model": name, "variant": "tuned", **{k: round(v, 5) if isinstance(v, float) else v for k, v in search.best_params_.items()}})
                    tracker.metrics(m, "val_")
                    results.append({"pipe": search.best_estimator_, "val_proba": val_p, "metrics": m})

        comparison = pd.DataFrame([r["metrics"] for r in results])
        cols = ["model", "variant", "average_precision", "roc_auc", "precision", "recall", "f1", "f2", "threshold",
                "false_alarm_rate", "fn", "fp", "cv_average_precision", "fit_seconds"]
        comparison = comparison[[c for c in cols if c in comparison.columns]]

        # --- model selection on VALIDATION average precision (dummy baseline is never eligible)
        metric = s.training.selection_metric
        eligible = [r for r in results if r["metrics"]["model"] != "dummy"] or results
        best = max(eligible, key=lambda r: r["metrics"][metric])
        pipe, name, variant = best["pipe"], best["metrics"]["model"], best["metrics"]["variant"]

        # --- calibration + operating threshold (validation split)
        val_raw = best["val_proba"]
        calibrator = PlattCalibrator().fit(val_raw, y_va)
        val_cal = calibrator.transform(val_raw)
        threshold = select_threshold(y_va, val_cal, s.training.threshold_beta, s.training.threshold_min_precision)
        val_metrics = compute_metrics(y_va, val_cal, threshold)

        # --- final, single evaluation on the untouched TEST split
        te_raw = pipe.predict_proba(X_te)[:, 1]
        te_cal = calibrator.transform(te_raw)
        test_metrics = compute_metrics(y_te, te_cal, threshold)
        ci = bootstrap_ci(y_te, te_cal, threshold, s.training.bootstrap_samples, s.seed)
        dummy = build_pipeline("dummy", s.seed).fit(X_tr, y_tr)
        base_metrics = compute_metrics(y_te, dummy.predict_proba(X_te)[:, 1], threshold=0.5)
        errors = error_analysis(te.reset_index(drop=True), y_te, te_cal, threshold)

        # --- failure-type model, anomaly detector, drift reference
        ft_model, ft_metrics = _train_failure_type_model(tr, te, s)
        det = SensorAnomalyDetector(s.anomaly.n_estimators, s.anomaly.flag_percentile, s.seed).fit(X_tr[y_tr == 0])
        te_anom = det.score(X_te)
        from sklearn.metrics import average_precision_score, roc_auc_score
        anomaly_metrics = {
            "flag_percentile": s.anomaly.flag_percentile,
            "share_flagged_among_non_failures": float((te_anom[y_te == 0] >= det.flag_percentile).mean()),
            "share_flagged_among_failures": float((te_anom[y_te == 1] >= det.flag_percentile).mean()) if (y_te == 1).any() else None,
            "roc_auc_vs_failure_label": float(roc_auc_score(y_te, te_anom)) if y_te.nunique() > 1 else None,
            "average_precision_vs_failure_label": float(average_precision_score(y_te, te_anom)) if y_te.nunique() > 1 else None,
            "caveat": "No ground-truth anomaly labels exist; the failure label is only a weak sanity check.",
        }
        reference = build_reference(X_tr, s.seed)

        metadata = {
            "version": version, "created_utc": datetime.now(UTC).isoformat(timespec="seconds"), "pmip_version": __version__,
            "model": name, "variant": variant, "selection_metric": f"validation {metric}",
            "decision_threshold": threshold, "threshold_objective": f"max F{s.training.threshold_beta:g} on validation (calibrated probabilities)",
            "calibration": "Platt scaling fitted on the validation split", "seed": s.seed,
            "dataset": {"rows": len(df), "sha256": fingerprint, "train": len(tr), "val": len(va), "test": len(te),
                        "failure_rate": float(df[TARGET].mean()), "source": "UCI AI4I 2020 (CC BY 4.0) — or synthetic stand-in if flagged"},
            "classifier_params": {k: v for k, v in pipe[-1].get_params().items() if isinstance(v, (int, float, str, bool, type(None)))},
            "features": {"inputs": list(X_tr.columns), "engineered": FEATURE_RATIONALE},
            "validation_metrics": val_metrics, "test_metrics": test_metrics, "test_metrics_ci95": ci,
            "baseline_dummy_test_metrics": base_metrics, "error_analysis": errors,
            "failure_type_model": ft_metrics, "anomaly_detector": anomaly_metrics,
            "limitations": LIMITATIONS,
            "environment": {"python": platform.python_version(), "git_commit": _git_commit()},
        }
        bundle = ModelBundle(version, pipe, calibrator, threshold, det, ft_model, reference, metadata)

        model_dir = s.resolve(s.paths.models_dir) / version
        if save:
            model_dir = save_bundle(bundle, s.resolve(s.paths.models_dir))
        if write_reports:
            _write_reports(s, reports, fig_dir, bundle, comparison, y_va, val_cal, y_te, te_raw, te_cal, te, X_va, threshold, te_anom, det)
        tracker.params({"selected_model": name, "selected_variant": variant, "decision_threshold": round(threshold, 4)})
        tracker.metrics(val_metrics, "final_val_")
        tracker.metrics(test_metrics, "test_")
        if write_reports:
            for f in [reports / "metrics.json", reports / "model_comparison.csv", *sorted(fig_dir.glob("*.png"))]:
                tracker.artifact(f, "reports")
        if save:
            tracker.artifact(model_dir / "metadata.json", "model_bundle")
        tracker.model(pipe, "pmip-failure-classifier")
    return TrainingResult(version, model_dir, metadata, comparison)


def _write_reports(s, reports, fig_dir, bundle, comparison, y_va, val_cal, y_te, te_raw, te_cal, te, X_va, threshold, te_anom, det):
    reports.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(reports / "model_comparison.csv", index=False)
    (reports / "metrics.json").write_text(json.dumps(bundle.metadata, indent=2, default=str))
    plots.plot_pr_roc(y_te, te_cal, fig_dir / "pr_roc_test.png", "(test)")
    cm = confusion_matrix(y_te, (te_cal >= threshold).astype(int), labels=[0, 1])
    plots.plot_confusion(cm, ["no failure", "failure"], fig_dir / "confusion_matrix_test.png", "Confusion matrix (test)")
    plots.plot_calibration(y_te, te_raw, te_cal, fig_dir / "calibration_test.png")
    plots.plot_threshold_tradeoff(threshold_curve(y_va, val_cal), threshold, fig_dir / "threshold_tradeoff_val.png")
    plots.plot_anomaly_hist(te_anom[y_te.to_numpy() == 0], te_anom[y_te.to_numpy() == 1], det.flag_percentile, fig_dir / "anomaly_scores_test.png")
    if bundle.explainer.supported:
        plots.plot_shap(bundle.explainer, X_va, fig_dir / "shap_summary.png", fig_dir / "shap_importance.png")
    out = feature_frame(te).copy()
    out["machine_failure"] = te[TARGET].to_numpy()
    out["failure_type"] = te["failure_type"].to_numpy()
    out["raw_score"] = te_raw
    out["failure_probability"] = te_cal
    out["predicted_failure"] = (te_cal >= threshold).astype(int)
    out["anomaly_score"] = te_anom
    out.to_csv(reports / "test_predictions.csv", index=False)
    _ = clone  # keep import used for downstream extensions
