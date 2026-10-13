"""Model factory, pipelines and hyper-parameter search spaces."""
from __future__ import annotations

import importlib.util

from scipy.stats import loguniform, randint, uniform
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .features import SensorFeatureEngineer, build_preprocessor

SCALE_REQUIRED = {"logistic_regression"}
TREE_MODELS = {"random_forest", "xgboost", "lightgbm"}


def lightgbm_available() -> bool:
    return importlib.util.find_spec("lightgbm") is not None


def make_estimator(name: str, seed: int, scale_pos_weight: float = 1.0):
    if name == "dummy":
        return DummyClassifier(strategy="prior")
    if name == "logistic_regression":
        return LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, class_weight="balanced_subsample", n_jobs=-1, random_state=seed
        )
    if name == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight, eval_metric="aucpr", tree_method="hist",
            n_jobs=-1, random_state=seed,
        )
    if name == "lightgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            n_estimators=300, learning_rate=0.05, num_leaves=15, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, class_weight="balanced", n_jobs=-1, random_state=seed, verbose=-1,
        )
    raise ValueError(f"Unknown model '{name}'")


def build_pipeline(name: str, seed: int = 42, scale_pos_weight: float = 1.0) -> Pipeline:
    """features (validation + derived) -> preprocessing -> classifier. Fit on training data only."""
    return Pipeline(
        [
            ("features", SensorFeatureEngineer()),
            ("preprocess", build_preprocessor(scale=name in SCALE_REQUIRED)),
            ("clf", make_estimator(name, seed, scale_pos_weight)),
        ]
    )


def search_space(name: str) -> dict:
    """Deliberately small spaces: the dataset is small and most gains come from the first few parameters."""
    if name == "xgboost":
        return {
            "clf__n_estimators": randint(150, 500),
            "clf__max_depth": randint(2, 7),
            "clf__learning_rate": loguniform(0.02, 0.2),
            "clf__subsample": uniform(0.6, 0.4),
            "clf__colsample_bytree": uniform(0.6, 0.4),
            "clf__min_child_weight": randint(1, 8),
            "clf__reg_lambda": loguniform(0.5, 10),
        }
    if name == "lightgbm":
        return {
            "clf__n_estimators": randint(150, 500),
            "clf__num_leaves": randint(7, 40),
            "clf__learning_rate": loguniform(0.02, 0.2),
            "clf__min_child_samples": randint(5, 40),
            "clf__colsample_bytree": uniform(0.6, 0.4),
            "clf__reg_lambda": loguniform(0.5, 10),
        }
    if name == "random_forest":
        return {
            "clf__n_estimators": randint(150, 500),
            "clf__max_depth": randint(4, 20),
            "clf__min_samples_leaf": randint(1, 10),
            "clf__max_features": uniform(0.4, 0.6),
        }
    raise ValueError(f"No search space defined for '{name}'")
