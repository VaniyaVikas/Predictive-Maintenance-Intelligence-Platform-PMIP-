.PHONY: install data synthetic eda train train-fast api dashboard mlflow test lint docker-up
export PYTHONPATH := src:dashboard

install:        ## install dependencies
	pip install -r requirements-dev.txt
data:           ## download the real UCI AI4I 2020 dataset
	python -m pmip.cli download
synthetic:      ## SYNTHETIC stand-in data for offline smoke tests only
	python -m pmip.cli synthetic
eda:
	python -m pmip.cli eda
train:          ## full run incl. hyper-parameter search + MLflow
	python -m pmip.cli train
train-fast:
	python -m pmip.cli train --fast
api:
	uvicorn pmip.api.main:app --reload --port 8000
dashboard:
	streamlit run dashboard/app.py
mlflow:         ## browse experiments at http://localhost:5000
	mlflow ui --backend-store-uri sqlite:///mlflow.db
test:
	pytest --cov=pmip --cov-report=term-missing
lint:
	ruff check src tests dashboard
docker-up:
	docker compose up --build
