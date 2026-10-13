FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app/src:/app/dashboard PMIP_HOME=/app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY configs ./configs
COPY src ./src
COPY dashboard ./dashboard
RUN useradd -m app && mkdir -p models reports/figures monitoring data/raw && chown -R app /app
USER app
EXPOSE 8000 8501
# default = API; docker-compose overrides the command for the dashboard
CMD ["uvicorn", "pmip.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
