# Deployment Guide

This guide covers deploying the Fair Credit Score Prediction system from local development through production deployment.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Local Development](#local-development)
- [Docker Deployment](#docker-deployment)
- [API Server](#api-server)
- [Cloud Deployment](#cloud-deployment)
- [CI/CD Pipeline](#cicd-pipeline)
- [Monitoring & Alerting](#monitoring--alerting)
- [Model Registry](#model-registry)
- [Security Considerations](#security-considerations)
- [Rollback Procedures](#rollback-procedures)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

- Python 3.10+
- Docker & Docker Compose (for containerized deployment)
- AWS CLI / GCP CLI / Azure CLI (for cloud deployment)
- Git

---

## Local Development

### Quick Setup

```bash
# Clone and setup
git clone <repo-url>
cd credprediction-opus45

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e ".[dev]"  # Install with dev dependencies

# Run tests to verify setup
pytest -v

# Run the pipeline
python -m src.main run --data-path data/credit.csv
```

### Environment Variables

Create a `.env` file for local development:

```bash
# .env
ENVIRONMENT=development
LOG_LEVEL=DEBUG
DATA_PATH=./data
MODEL_PATH=./models
API_HOST=0.0.0.0
API_PORT=8000

# Optional: Cloud storage
AWS_S3_BUCKET=your-bucket-name
GCP_PROJECT_ID=your-project-id
```

---

## Docker Deployment

### Dockerfile

Create `Dockerfile` in the project root:

```dockerfile
# Dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (cache layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY config/ ./config/
COPY src/ ./src/
COPY pyproject.toml .

# Create necessary directories
RUN mkdir -p /app/data /app/models /app/explanations /app/reports /app/logs

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import src; print('healthy')" || exit 1

# Default command (can be overridden)
CMD ["python", "-m", "src.main", "run", "--help"]
```

### Docker Compose

Create `docker-compose.yml`:

```yaml
# docker-compose.yml
version: '3.8'

services:
  # Main prediction service
  credit-prediction:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: fair-credit-prediction
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - ./explanations:/app/explanations
      - ./reports:/app/reports
      - ./logs:/app/logs
    environment:
      - ENVIRONMENT=production
      - LOG_LEVEL=INFO
    ports:
      - "8000:8000"
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  # API service (when running FastAPI)
  api:
    build:
      context: .
      dockerfile: Dockerfile.api
    container_name: fair-credit-api
    volumes:
      - ./models:/app/models:ro
    environment:
      - ENVIRONMENT=production
      - MODEL_PATH=/app/models/credit_model.pkl
    ports:
      - "8000:8000"
    restart: unless-stopped
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G

  # Monitoring (optional)
  prometheus:
    image: prom/prometheus:latest
    container_name: prometheus
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
    ports:
      - "9090:9090"
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    container_name: grafana
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-data:/var/lib/grafana
    ports:
      - "3000:3000"
    restart: unless-stopped

volumes:
  grafana-data:
```

### Build and Run

```bash
# Build the image
docker build -t fair-credit-prediction:latest .

# Run training pipeline
docker run -v $(pwd)/data:/app/data \
           -v $(pwd)/models:/app/models \
           fair-credit-prediction:latest \
           python -m src.main run --data-path /app/data/credit.csv

# Run with docker-compose
docker-compose up -d

# View logs
docker-compose logs -f credit-prediction

# Stop services
docker-compose down
```

---

## API Server

### FastAPI Implementation

Create `src/api.py`:

```python
"""FastAPI server for credit prediction."""

import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd

from config.config import load_config
from src.model import CreditModel
from src.preprocessing import Preprocessor
from src.fairness import FairnessAnalyzer
from src.explainability import Explainer
from src.utils import setup_logging, get_logger

# Initialize
app = FastAPI(
    title="Fair Credit Score Prediction API",
    description="Equitable, explainable credit scoring",
    version="0.1.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
config = None
model = None
explainer = None
logger = None


class PredictionRequest(BaseModel):
    """Single prediction request."""
    Income: float = Field(..., ge=0)
    Debt: float = Field(..., ge=0)
    Loan_Amount: float = Field(..., ge=0)
    Loan_Term: int = Field(..., ge=1)
    Num_Credit_Cards: int = Field(..., ge=0)
    Gender: str
    Education: str
    Payment_History: str
    Employment_Status: str
    Residence_Type: str
    Marital_Status: str


class PredictionResponse(BaseModel):
    """Prediction response."""
    creditworthy: bool
    probability: float
    confidence: str
    explanation: dict[str, Any] | None = None


class BatchPredictionRequest(BaseModel):
    """Batch prediction request."""
    records: list[PredictionRequest]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    version: str


@app.on_event("startup")
async def startup():
    """Load model and configuration on startup."""
    global config, model, explainer, logger
    
    config_path = os.getenv("CONFIG_PATH")
    config = load_config(config_path)
    logger = setup_logging(config.logging)
    
    model_path = os.getenv("MODEL_PATH", "models/credit_model.pkl")
    
    if Path(model_path).exists():
        model = CreditModel(config)
        model.load(model_path)
        
        explainer = Explainer(config)
        explainer.setup_shap_explainer(model)
        
        logger.info(f"Model loaded from {model_path}")
    else:
        logger.warning(f"Model not found at {model_path}")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        version="0.1.0",
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest, include_explanation: bool = False):
    """Generate a single prediction."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Convert to DataFrame
    df = pd.DataFrame([request.dict()])
    
    # Preprocess
    preprocessor = Preprocessor(config)
    df = preprocessor.convert_numeric_columns(df)
    df = preprocessor.create_derived_features(df)
    df = preprocessor.calculate_cdi(df)
    df = preprocessor.encode_categorical(df)
    
    # Predict
    prob = model.predict_proba(df)[0]
    pred = int(prob >= config.fairness.threshold_privileged)
    
    # Confidence level
    if prob > 0.8 or prob < 0.2:
        confidence = "high"
    elif prob > 0.6 or prob < 0.4:
        confidence = "medium"
    else:
        confidence = "low"
    
    response = PredictionResponse(
        creditworthy=bool(pred),
        probability=float(prob),
        confidence=confidence,
    )
    
    # Add explanation if requested
    if include_explanation and explainer:
        explanation = explainer.explain_individual(df, 0, pred)
        response.explanation = {
            "top_positive_factors": explanation["top_positive_factors"][:3],
            "top_negative_factors": explanation["top_negative_factors"][:3],
        }
    
    return response


@app.post("/batch_predict")
async def batch_predict(request: BatchPredictionRequest):
    """Generate batch predictions."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    results = []
    for record in request.records:
        result = await predict(record, include_explanation=False)
        results.append(result.dict())
    
    return {"predictions": results, "count": len(results)}


@app.get("/fairness_metrics")
async def get_fairness_metrics():
    """Get current fairness configuration and targets."""
    return {
        "thresholds": {
            "privileged": config.fairness.threshold_privileged,
            "unprivileged": config.fairness.threshold_unprivileged,
        },
        "targets": {
            "disparate_impact_ratio": config.fairness.target_disparate_impact_ratio,
            "statistical_parity_diff": config.fairness.target_statistical_parity_diff,
            "equalized_odds_diff": config.fairness.target_equalized_odds_diff,
        },
        "cdi_factors": config.cdi.factors,
        "cdi_threshold": config.cdi.threshold,
    }


@app.get("/model_info")
async def get_model_info():
    """Get model metadata."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    metadata = model.get_training_metadata()
    return {
        "feature_count": metadata.get("num_features"),
        "features": metadata.get("feature_names"),
        "used_fairness_weights": metadata.get("used_sample_weights"),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### API Dockerfile

Create `Dockerfile.api`:

```dockerfile
# Dockerfile.api
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir fastapi uvicorn

COPY config/ ./config/
COPY src/ ./src/

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Running the API

```bash
# Local development
uvicorn src.api:app --reload --host 0.0.0.0 --port 8000

# Docker
docker build -f Dockerfile.api -t fair-credit-api:latest .
docker run -p 8000:8000 -v $(pwd)/models:/app/models fair-credit-api:latest

# Test endpoints
curl http://localhost:8000/health
curl http://localhost:8000/fairness_metrics

# Make a prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Income": 75000,
    "Debt": 15000,
    "Loan_Amount": 25000,
    "Loan_Term": 36,
    "Num_Credit_Cards": 3,
    "Gender": "Female",
    "Education": "Bachelor\'s",
    "Payment_History": "Good",
    "Employment_Status": "Employed",
    "Residence_Type": "Rented",
    "Marital_Status": "Single"
  }'
```

---

## Cloud Deployment

### AWS Deployment

#### Option 1: AWS ECS (Elastic Container Service)

```bash
# 1. Create ECR repository
aws ecr create-repository --repository-name fair-credit-prediction

# 2. Build and push image
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com

docker build -f Dockerfile.api -t fair-credit-api .
docker tag fair-credit-api:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/fair-credit-prediction:latest
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/fair-credit-prediction:latest

# 3. Create ECS task definition (task-definition.json)
# 4. Create ECS service
```

**ECS Task Definition** (`ecs-task-definition.json`):

```json
{
  "family": "fair-credit-prediction",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "executionRoleArn": "arn:aws:iam::ACCOUNT_ID:role/ecsTaskExecutionRole",
  "containerDefinitions": [
    {
      "name": "fair-credit-api",
      "image": "ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/fair-credit-prediction:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {"name": "ENVIRONMENT", "value": "production"},
        {"name": "MODEL_PATH", "value": "/app/models/credit_model.pkl"}
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/fair-credit-prediction",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "healthCheck": {
        "command": ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"],
        "interval": 30,
        "timeout": 5,
        "retries": 3
      }
    }
  ]
}
```

#### Option 2: AWS Lambda + API Gateway

For serverless deployment, see `scripts/deploy_lambda.py` (create as needed).

### GCP Deployment

#### Cloud Run

```bash
# 1. Configure gcloud
gcloud config set project YOUR_PROJECT_ID

# 2. Build with Cloud Build
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/fair-credit-api

# 3. Deploy to Cloud Run
gcloud run deploy fair-credit-api \
  --image gcr.io/YOUR_PROJECT_ID/fair-credit-api \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --min-instances 1 \
  --max-instances 10 \
  --set-env-vars ENVIRONMENT=production
```

### Azure Deployment

#### Azure Container Instances

```bash
# 1. Create resource group
az group create --name fair-credit-rg --location eastus

# 2. Create container registry
az acr create --resource-group fair-credit-rg --name faircreditacr --sku Basic

# 3. Build and push
az acr build --registry faircreditacr --image fair-credit-api:latest .

# 4. Deploy container instance
az container create \
  --resource-group fair-credit-rg \
  --name fair-credit-api \
  --image faircreditacr.azurecr.io/fair-credit-api:latest \
  --cpu 2 \
  --memory 4 \
  --ports 8000 \
  --dns-name-label fair-credit-api
```

---

## CI/CD Pipeline

### GitHub Actions

Create `.github/workflows/ci.yml`:

```yaml
name: CI/CD Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

env:
  PYTHON_VERSION: "3.11"
  DOCKER_IMAGE: fair-credit-prediction

jobs:
  # ============================================
  # Code Quality & Tests
  # ============================================
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
          cache: 'pip'

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install -e ".[dev]"

      - name: Lint with flake8
        run: |
          flake8 src/ tests/ --count --select=E9,F63,F7,F82 --show-source --statistics
          flake8 src/ tests/ --count --exit-zero --max-complexity=10 --max-line-length=88 --statistics

      - name: Format check with black
        run: |
          black --check src/ tests/

      - name: Type check with mypy
        run: |
          mypy src/ --ignore-missing-imports

      - name: Run tests with pytest
        run: |
          pytest --cov=src --cov-report=xml --cov-report=html -v

      - name: Upload coverage reports
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml
          fail_ci_if_error: false

  # ============================================
  # Security Scan
  # ============================================
  security:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Run Snyk to check for vulnerabilities
        uses: snyk/actions/python@master
        continue-on-error: true
        env:
          SNYK_TOKEN: ${{ secrets.SNYK_TOKEN }}

  # ============================================
  # Build Docker Image
  # ============================================
  build:
    needs: test
    runs-on: ubuntu-latest
    if: github.event_name == 'push'
    steps:
      - uses: actions/checkout@v4

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Login to Container Registry
        uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ghcr.io/${{ github.repository }}
          tags: |
            type=sha
            type=ref,event=branch
            type=semver,pattern={{version}}

      - name: Build and push
        uses: docker/build-push-action@v5
        with:
          context: .
          file: ./Dockerfile.api
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max

  # ============================================
  # Deploy to Staging
  # ============================================
  deploy-staging:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/develop'
    environment: staging
    steps:
      - name: Deploy to staging
        run: |
          echo "Deploying to staging environment..."
          # Add your staging deployment commands here
          # e.g., kubectl apply, aws ecs update-service, etc.

  # ============================================
  # Deploy to Production
  # ============================================
  deploy-production:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    environment: production
    steps:
      - name: Deploy to production
        run: |
          echo "Deploying to production environment..."
          # Add your production deployment commands here

  # ============================================
  # Fairness Audit (Weekly)
  # ============================================
  fairness-audit:
    runs-on: ubuntu-latest
    if: github.event_name == 'schedule'
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}

      - name: Run fairness audit
        run: |
          pip install -r requirements.txt
          python -m src.main run --data-path data/audit_sample.csv
          # Check fairness metrics and alert if thresholds breached

      - name: Upload audit report
        uses: actions/upload-artifact@v3
        with:
          name: fairness-audit-report
          path: outputs/metrics.json
```

### Scheduled Fairness Audits

Add to `.github/workflows/ci.yml` triggers:

```yaml
on:
  schedule:
    - cron: '0 0 * * 0'  # Weekly on Sunday at midnight
```

---

## Monitoring & Alerting

### Prometheus Metrics

Create `monitoring/prometheus.yml`:

```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

alerting:
  alertmanagers:
    - static_configs:
        - targets: []

rule_files:
  - "alerts.yml"

scrape_configs:
  - job_name: 'fair-credit-api'
    static_configs:
      - targets: ['api:8000']
    metrics_path: /metrics
```

### Alert Rules

Create `monitoring/alerts.yml`:

```yaml
# alerts.yml
groups:
  - name: fairness_alerts
    rules:
      - alert: DisparateImpactBelowThreshold
        expr: fairness_disparate_impact < 0.80
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Disparate Impact below legal threshold"
          description: "Disparate Impact ratio is {{ $value }}, below the 0.80 threshold"

      - alert: StatisticalParityViolation
        expr: abs(fairness_statistical_parity_diff) > 0.05
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Statistical Parity difference exceeds threshold"

      - alert: HighLatency
        expr: histogram_quantile(0.95, rate(prediction_latency_seconds_bucket[5m])) > 0.5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "95th percentile latency exceeds 500ms"

      - alert: HighErrorRate
        expr: rate(prediction_errors_total[5m]) > 0.01
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Prediction error rate exceeds 1%"
```

### Grafana Dashboard

Create `monitoring/grafana-dashboard.json` with panels for:
- Prediction request rate
- Latency percentiles (p50, p95, p99)
- Fairness metrics over time
- Error rates by type
- Model version tracking

### Application Metrics

Add to `src/api.py`:

```python
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from starlette.responses import Response

# Metrics
PREDICTION_COUNTER = Counter(
    'predictions_total', 
    'Total predictions made',
    ['outcome', 'group']
)
PREDICTION_LATENCY = Histogram(
    'prediction_latency_seconds',
    'Prediction latency in seconds'
)
FAIRNESS_GAUGE = Gauge(
    'fairness_disparate_impact',
    'Current disparate impact ratio'
)

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type="text/plain")
```

---

## Model Registry

### MLflow Integration

```python
# scripts/register_model.py
import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://mlflow-server:5000")

def register_model(model_path: str, metrics: dict, run_name: str = "credit_model"):
    with mlflow.start_run(run_name=run_name):
        # Log parameters
        mlflow.log_params({
            "cdi_threshold": 2,
            "fairness_threshold_privileged": 0.5,
            "fairness_threshold_unprivileged": 0.4,
        })
        
        # Log metrics
        mlflow.log_metrics(metrics)
        
        # Log model
        mlflow.xgboost.log_model(
            xgb_model=model,
            artifact_path="model",
            registered_model_name="fair-credit-model"
        )
        
        # Log fairness report
        mlflow.log_artifact("outputs/metrics.json", "fairness")
```

### Model Versioning Strategy

```
models/
├── credit_model_v1.0.0.pkl      # Initial production model
├── credit_model_v1.1.0.pkl      # Improved fairness
├── credit_model_v1.2.0.pkl      # Added features
└── credit_model_latest.pkl      # Symlink to current production
```

---

## Security Considerations

### Data Security

1. **Encryption at Rest**: All model files and data encrypted using AES-256
2. **Encryption in Transit**: TLS 1.3 for all API communications
3. **Access Control**: IAM roles with least-privilege access
4. **Audit Logging**: All predictions logged with timestamp, user, and outcome

### API Security

```python
# Add to src/api.py
from fastapi import Depends, HTTPException, Security
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key")

async def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != os.getenv("API_KEY"):
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key

@app.post("/predict", dependencies=[Depends(verify_api_key)])
async def predict(...):
    ...
```

### Secrets Management

```bash
# AWS Secrets Manager
aws secretsmanager create-secret \
    --name fair-credit/api-keys \
    --secret-string '{"api_key":"your-secret-key"}'

# Kubernetes Secrets
kubectl create secret generic fair-credit-secrets \
    --from-literal=api-key=your-secret-key
```

---

## Rollback Procedures

### Automatic Rollback Triggers

1. **Fairness Violation**: Disparate Impact < 0.80 for 5 minutes
2. **Error Rate**: > 1% errors for 5 minutes
3. **Latency**: p95 > 1 second for 5 minutes
4. **Health Check Failures**: 3 consecutive failures

### Manual Rollback

```bash
# Docker/Kubernetes
kubectl rollout undo deployment/fair-credit-api

# AWS ECS
aws ecs update-service \
    --cluster fair-credit-cluster \
    --service fair-credit-api \
    --task-definition fair-credit-api:PREVIOUS_VERSION

# Switch model version
export MODEL_PATH=/app/models/credit_model_v1.0.0.pkl
docker-compose restart api
```

### Blue-Green Deployment

```yaml
# kubernetes/blue-green.yml
apiVersion: v1
kind: Service
metadata:
  name: fair-credit-api
spec:
  selector:
    app: fair-credit-api
    version: green  # Switch between blue/green
  ports:
    - port: 8000
```

---

## Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| Model not loading | Missing file | Check `MODEL_PATH` environment variable |
| High latency | Cold start | Increase min instances, add warmup |
| Fairness violation | Data drift | Trigger retraining pipeline |
| OOM errors | Large batch | Reduce batch size, increase memory |

### Debug Commands

```bash
# Check container logs
docker logs fair-credit-api --tail 100

# Check health
curl -v http://localhost:8000/health

# Test prediction endpoint
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @test_request.json

# Check metrics
curl http://localhost:8000/metrics

# Shell into container
docker exec -it fair-credit-api /bin/bash
```

### Performance Tuning

```bash
# Increase workers
uvicorn src.api:app --workers 4

# Enable response caching
# Add Redis for caching frequent predictions

# Optimize model loading
# Use ONNX runtime for faster inference
```

---

## Checklist

### Pre-Deployment

- [ ] All tests passing (`pytest -v`)
- [ ] Code quality checks pass (`black`, `flake8`, `mypy`)
- [ ] Fairness metrics within targets
- [ ] Security scan clean
- [ ] Documentation updated
- [ ] Model registered in MLflow

### Post-Deployment

- [ ] Health check passing
- [ ] Smoke tests passing
- [ ] Monitoring dashboards active
- [ ] Alerts configured
- [ ] Rollback tested

### Quarterly Audit

- [ ] Fairness audit completed
- [ ] Model performance reviewed
- [ ] Data drift analysis
- [ ] Security patches applied
- [ ] Documentation current
