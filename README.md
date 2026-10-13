# Predictive Maintenance Intelligence Platform (PMIP)

End-to-end ML system on the UCI **AI4I 2020** dataset: failure-risk classification, failure-mode ranking, anomaly detection, SHAP explanations, MLflow tracking, FastAPI service, Streamlit dashboard, drift monitoring, tests, Docker and CI.

## What it does — and does not do
| Task | Status | Notes |
|---|---|---|
| Failure classification (binary) | | 5 models compared, tuned with `RandomizedSearchCV` (PR-AUC), Platt-calibrated, F2-optimal threshold chosen on validation |
| Failure-**type** classification | (limited) | RandomForest trained on failure rows only → "if a failure occurs, which mode?"; tiny support per class |
| Anomaly detection |  | Isolation Forest on normal rows; percentile score; **unusual ≠ failed**; no true anomaly labels exist |
| Remaining useful life |  | Dataset has no run-to-failure sequences, so it is *not* estimated |
| Concept drift |  not confirmed | Only **input data drift** (PSI/KS) is monitored; concept drift needs labels |

The dataset is synthetic and has no timestamps → stratified split, results are optimistic and not transferable to a real plant. See `docs/DATA.md`.

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate
make install
make data            # downloads the real dataset to data/raw/ai4i2020.csv
make eda             # reports/eda_summary.md + figures
make train           # tuning + MLflow; writes models/<version>/ and reports/
make api             # http://localhost:8000/docs
make dashboard       # http://localhost:8501
make mlflow          # http://localhost:5000
make test            # pytest, 57 tests, ~96% line coverage (synthetic data, no download needed)
```
No internet/dataset yet? `make synthetic && make train-fast` runs the toolchain on **synthetic stand-in data** (never quote those metrics).

## Evaluation policy
Split 70/15/15 stratified. Models and thresholds are chosen on **validation** (average precision; threshold = max F2, i.e. recall-weighted because a missed failure usually costs more than a false alarm; configurable in `configs/config.yaml`). The **test set is evaluated once**, for the selected model and a dummy baseline, with bootstrap 95% CIs (only a few dozen failures exist). Accuracy is reported but never used for selection. Scores in `reports/metrics.json` come from your own run — none are hard-coded here.

## API
`GET /health`, `GET /model-info`, `POST /predict[?explain=true]`, `POST /predict/batch`, `POST /anomaly-detection`, `GET /metrics`.
```bash
curl -X POST "localhost:8000/predict?explain=true" -H 'Content-Type: application/json' \
 -d '{"machine_type":"M","air_temperature_k":300.1,"process_temperature_k":310.2,"rotational_speed_rpm":1500,"torque_nm":42.5,"tool_wear_min":120}'
```
Validation errors → 422, missing model → 503, unexpected errors → generic 500 (details only in server logs). Optional `PMIP_API_KEY` enables `X-API-Key` auth. Risk bands (`risk.medium`/`risk.high`, or `PMIP_RISK_*`) apply to the calibrated probability; the *predicted class* uses the tuned decision threshold, and the two are independent. Probabilities are Platt-calibrated on a small validation set and reflect the benchmark's ~3% base rate.

## Monitoring
Reference decile bins + sample are stored with each model. Served requests are logged to SQLite (`monitoring/`). Drift = PSI (warn 0.10, alert 0.25) plus KS test, shown on dashboard page 9. Drift raises a flag for human review; **no automatic retraining**.

## Layout
```
configs/config.yaml    central settings         src/pmip/        library (data, features, models, train, anomaly,
dashboard/             Streamlit (10 pages)                      explain, monitoring, registry, service, api/)
tests/                 pytest suite             docs/DATA.md     dataset notes & limitations
```

## Docker
```bash
make train                         # or: docker compose --profile train run --rm trainer
docker compose up --build          # api :8000, dashboard :8501 (models/ is mounted)
```

## Deployment (suggested)
- **Render / Railway / Fly.io:** two web services from the same Dockerfile. API start command: default `CMD`; dashboard: `streamlit run dashboard/app.py --server.port $PORT --server.address 0.0.0.0`. Health check path `/health` (API) and `/_stcore/health` (dashboard).
- Models are git-ignored. For a demo deploy, train locally, then either remove `models/*` from `.gitignore` for the (small) bundle, attach a persistent disk, or publish it as a GitHub Release asset and download it in the start command. Also ship `reports/` for the dashboard's comparison/SHAP pages and `data/raw/` for the exploration page (or accept that those pages show a "run make data" hint).



## Extending
NASA C-MAPSS / bearing run-to-failure data would enable a genuine RUL model with time-aware, engine-grouped validation — add it as a separate module; it is intentionally not faked here.
