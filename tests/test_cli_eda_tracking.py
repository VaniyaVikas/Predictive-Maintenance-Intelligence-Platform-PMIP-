import io
import zipfile

import mlflow
import pandas as pd
import pytest
import yaml

from pmip import cli, download
from pmip.config import load_settings
from pmip.eda import run_eda
from pmip.tracking import Tracker


@pytest.fixture()
def cli_env(tmp_path, monkeypatch):
    cfg = {"paths": {"raw_data": str(tmp_path / "raw.csv"), "models_dir": str(tmp_path / "models"),
                     "reports_dir": str(tmp_path / "reports"), "monitoring_db": str(tmp_path / "m.db"),
                     "mlflow_uri": f"sqlite:///{tmp_path}/mlflow.db"},
           "training": {"models": ["dummy", "random_forest", "xgboost"], "tuning_enabled": False, "bootstrap_samples": 10}}
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(cfg))
    monkeypatch.setenv("PMIP_CONFIG", str(p))
    load_settings.cache_clear()
    yield tmp_path
    load_settings.cache_clear()


def test_cli_synthetic_eda_train_with_mlflow(cli_env, capsys):
    assert cli.main(["synthetic", "--rows", "3000"]) == 0
    assert "SYNTHETIC" in capsys.readouterr().out
    assert cli.main(["eda"]) == 0
    assert (cli_env / "reports" / "eda_summary.md").exists()
    assert cli.main(["train"]) == 0  # MLflow enabled against a temp sqlite store
    assert (cli_env / "models" / "latest.json").exists()
    assert (cli_env / "reports" / "figures" / "pr_roc_test.png").exists()
    mlflow.set_tracking_uri(f"sqlite:///{cli_env}/mlflow.db")
    runs = mlflow.search_runs(experiment_names=["pmip"])
    assert len(runs) >= 4  # parent + one nested run per model
    assert any(m.name == "pmip-failure-classifier" for m in mlflow.MlflowClient().search_registered_models())


def test_eda_outputs(df, tmp_path):
    report = run_eda(df, tmp_path)
    text = (tmp_path / "eda_summary.md").read_text()
    assert "Class imbalance" in text and "synthetic" in text.lower() and report["n_rows"] == len(df)
    assert (tmp_path / "figures" / "eda_correlation.png").exists() and (tmp_path / "data_quality.json").exists()


def test_download_extracts_csv(tmp_path, monkeypatch, raw_df):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("ai4i2020.csv", raw_df.head(20).to_csv(index=False))

    class Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(download.urllib.request, "urlopen", lambda *a, **k: Resp(buf.getvalue()))
    out = download.download_ai4i(tmp_path / "x" / "ai4i2020.csv")
    assert len(pd.read_csv(out)) == 20


def test_download_without_csv_fails(tmp_path, monkeypatch):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("readme.txt", "x")

    class Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(download.urllib.request, "urlopen", lambda *a, **k: Resp(buf.getvalue()))
    with pytest.raises(RuntimeError):
        download.download_ai4i(tmp_path / "a.csv")


def test_disabled_tracker_is_noop(tmp_path):
    t = Tracker(False, "")
    with t.run("x"):
        t.params({"a": 1})
        t.metrics({"b": 1.0})
        t.tags({"c": 1})
        t.artifact(tmp_path / "none")
        t.model(object(), "m")
