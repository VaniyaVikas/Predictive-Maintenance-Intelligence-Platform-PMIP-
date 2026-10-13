"""Exploratory data analysis: figures, quality report and a data-driven written summary."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from .data import derive_failure_type, quality_report  # noqa: E402
from .features import ALL_NUMERIC, SensorFeatureEngineer  # noqa: E402
from .plots import _save  # noqa: E402
from .schema import FAILURE_MODES, NUMERIC_FEATURES, TARGET  # noqa: E402


def _iqr_outliers(s: pd.Series) -> int:
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    return int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())


def run_eda(df: pd.DataFrame, out_dir: Path) -> dict:
    """Run the full EDA, writing figures to ``out_dir/figures`` and ``eda_summary.md`` to ``out_dir``."""
    out_dir = Path(out_dir)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    df = df.copy()
    df["failure_type"] = derive_failure_type(df)
    feats = SensorFeatureEngineer().transform(df)
    feats[TARGET] = df[TARGET].to_numpy()
    report = quality_report(df)
    (out_dir / "data_quality.json").write_text(json.dumps(report, indent=2))
    desc = df[NUMERIC_FEATURES].describe().T

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    sns.countplot(x=TARGET, data=df, ax=ax[0])
    ax[0].set(title="Machine failure (0 = no, 1 = yes)")
    order = df.loc[df[TARGET] == 1, "failure_type"].value_counts()
    ax[1].bar(order.index, order.values)
    ax[1].set(title="Failure types (failures only)")
    _save(fig, fig_dir / "eda_target_distribution.png")

    fig, axes = plt.subplots(2, 4, figsize=(17, 7))
    for ax, col in zip(axes.ravel(), ALL_NUMERIC, strict=False):
        sns.histplot(data=feats, x=col, hue=TARGET, stat="density", common_norm=False, bins=30, ax=ax, element="step")
    axes.ravel()[-1].axis("off")
    _save(fig, fig_dir / "eda_distributions_by_failure.png")

    fig, axes = plt.subplots(1, 5, figsize=(17, 3.8))
    for ax, col in zip(axes, NUMERIC_FEATURES, strict=True):
        sns.boxplot(data=feats, x=TARGET, y=col, ax=ax)
    _save(fig, fig_dir / "eda_boxplots_by_failure.png")

    corr = feats[ALL_NUMERIC + [TARGET]].corr()
    fig, ax = plt.subplots(figsize=(8.5, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation matrix (Pearson)")
    _save(fig, fig_dir / "eda_correlation.png")

    wear_bins = pd.cut(df["tool_wear_min"], bins=[-1, 50, 100, 150, 200, 1000])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    df.groupby(wear_bins, observed=True)[TARGET].mean().plot.bar(ax=ax[0], title="Failure rate by tool-wear bin")
    df.groupby("machine_type")[TARGET].mean().plot.bar(ax=ax[1], title="Failure rate by machine type")
    _save(fig, fig_dir / "eda_failure_rate_groups.png")

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(data=df, x="rotational_speed_rpm", y="torque_nm", hue=TARGET, alpha=0.5, s=14, ax=ax)
    ax.set_title("Torque vs. rotational speed")
    _save(fig, fig_dir / "eda_torque_vs_speed.png")

    corr_t = corr[TARGET].drop(TARGET).sort_values(key=np.abs, ascending=False)
    cb = report["class_balance"]
    lines = [
        "# EDA summary (auto-generated from the loaded data)",
        "",
        f"- **Shape:** {report['n_rows']:,} rows x {report['n_columns']} columns; "
        f"missing values: {sum(report['missing_values'].values())}; duplicate rows: {report['duplicate_rows']}.",
        f"- **Class imbalance:** {cb['failures']:,} failures ({cb['failure_rate']:.2%}); roughly "
        f"{cb['imbalance_ratio']:.0f}:1 non-failure to failure. Accuracy is therefore a misleading metric "
        "(always predicting 'no failure' scores high); PR-AUC, recall and precision are used instead.",
        f"- **Failure types:** {report.get('failure_mode_counts')}. Several modes are rare; failure-type "
        "classification has very small support for some classes.",
        f"- **Label inconsistencies:** {report.get('label_inconsistencies')} (documented in the source data; "
        "such rows are labelled `UNSPECIFIED`/kept as in the source).",
        f"- **Out-of-range values:** {report['out_of_range_values']}.",
        "- **Strongest linear associations with failure:** "
        + ", ".join(f"`{k}` ({v:+.2f})" for k, v in corr_t.head(4).items())
        + ". Linear correlation understates threshold-style failure rules, so tree models are expected to do better.",
        "- **Outliers (IQR rule, count):** " + ", ".join(f"`{c}`: {_iqr_outliers(df[c])}" for c in NUMERIC_FEATURES)
        + ". Outliers in torque/speed are partly the failure signal itself, so they are NOT removed.",
        "",
        "## Descriptive statistics",
        desc.round(2).to_markdown() if _has_tabulate() else "```\n" + desc.round(2).to_string() + "\n```",
        "",
        "## Caveats",
        "- The UCI AI4I 2020 dataset is itself a **synthetic** dataset created to mimic industrial data. "
        "Conclusions are about this benchmark, not about any real plant.",
        "- No timestamps or machine identifiers exist, so time-based/grouped validation is impossible and "
        "no remaining-useful-life (RUL) target exists. This project therefore does NOT estimate RUL.",
        f"- Failure-mode flags ({', '.join(FAILURE_MODES)}) define the failure label and are excluded from features (leakage).",
    ]
    (out_dir / "eda_summary.md").write_text("\n".join(lines))
    return report


def _has_tabulate() -> bool:
    try:
        import tabulate  # noqa: F401

        return True
    except ImportError:
        return False
