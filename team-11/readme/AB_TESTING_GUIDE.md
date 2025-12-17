# A/B Testing Guide

This guide explains how to use the A/B testing infrastructure to deploy and compare multiple model versions in production.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Model Registry](#model-registry)
- [Traffic Routing](#traffic-routing)
- [API Endpoints](#api-endpoints)
- [Monitoring](#monitoring)
- [Best Practices](#best-practices)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)

## Overview

The A/B testing infrastructure allows you to:

- **Deploy multiple model versions** simultaneously
- **Split traffic** between model versions
- **Compare performance** in real-time
- **Gradual rollouts** with adjustable traffic weights
- **Quick rollbacks** if issues are detected
- **Track metrics** per model version

### Key Features

- ✅ **Model Registry**: Centralized version management
- ✅ **Traffic Splitting**: Configurable weights for each version
- ✅ **Consistent Routing**: Same user gets same model version
- ✅ **Hot Swapping**: Load/unload models without downtime
- ✅ **Prometheus Metrics**: Per-version performance tracking
- ✅ **Audit Logging**: Complete history of model deployments

## Architecture

```
┌─────────────┐
│   Request   │
└──────┬──────┘
       │
       ▼
┌─────────────────────┐
│  Model Registry     │◄── Manages versions
│  (Traffic Router)   │◄── Routes by hash
└──────┬──────┬───────┘
       │      │
       ▼      ▼
   ┌─────┐ ┌─────┐
   │ v1  │ │ v2  │  ← Loaded models
   │ 80% │ │ 20% │  ← Traffic weights
   └─────┘ └─────┘
```

### Components

1. **Model Registry** (`src/model_registry.py`):
   - Stores model metadata
   - Manages loaded models
   - Routes traffic based on weights

2. **API Integration** (`src/api.py`):
   - Registry endpoints for management
   - A/B testing endpoints for configuration
   - Metrics collection per version

3. **Prometheus Metrics**:
   - `ab_test_requests_total` - Requests by version
   - `model_version_predictions_total` - Predictions by version/outcome
   - `active_models_count` - Number of active models

## Quick Start

### 1. Enable A/B Testing

Set environment variable:
```bash
export AB_TESTING_ENABLED=true
export MODEL_REGISTRY_PATH=models/registry.json  # Optional, defaults to this
```

Or in `.env` file:
```bash
AB_TESTING_ENABLED=true
MODEL_REGISTRY_PATH=models/registry.json
```

### 2. Register Model Versions

```bash
# Register version 1
curl -X POST "http://localhost:8000/registry/models" \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "version": "v1.0.0",
    "name": "Production Model",
    "model_path": "models/credit_model_v1.pkl",
    "preprocessor_path": "models/preprocessor_v1.pkl",
    "description": "Current production model",
    "metrics": {
      "accuracy": 0.8542,
      "brier_score": 0.1256
    },
    "tags": ["production", "baseline"]
  }'

# Register version 2
curl -X POST "http://localhost:8000/registry/models" \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type": application/json" \
  -d '{
    "version": "v1.1.0",
    "name": "Experimental Model",
    "model_path": "models/credit_model_v1_1.pkl",
    "preprocessor_path": "models/preprocessor_v1_1.pkl",
    "description": "New model with improved calibration",
    "metrics": {
      "accuracy": 0.8601,
      "brier_score": 0.1104
    },
    "tags": ["testing", "improved-calibration"]
  }'
```

### 3. Load Model Versions

```bash
# Load both models into memory
curl -X POST "http://localhost:8000/registry/models/v1.0.0/load" \
  -H "X-API-Key: your-api-key"

curl -X POST "http://localhost:8000/registry/models/v1.1.0/load" \
  -H "X-API-Key: your-api-key"
```

### 4. Set Traffic Weights

```bash
# 80% to v1.0.0, 20% to v1.1.0
curl -X POST "http://localhost:8000/registry/ab-test/weights" \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "v1.0.0": 0.8,
    "v1.1.0": 0.2
  }'
```

### 5. Monitor Performance

```bash
# Check current weights
curl "http://localhost:8000/registry/ab-test/weights" \
  -H "X-API-Key: your-api-key"

# Compare versions
curl "http://localhost:8000/registry/ab-test/compare/v1.0.0/v1.1.0" \
  -H "X-API-Key: your-api-key"
```

## Model Registry

### Registering Models

Models must be registered before they can be used in A/B tests.

**Required Fields:**
- `version`: Unique version identifier (e.g., "v1.0.0", "2024-12-08-prod")
- `name`: Human-readable name
- `model_path`: Path to saved model file

**Optional Fields:**
- `description`: Model description
- `preprocessor_path`: Path to saved preprocessor
- `metrics`: Performance metrics dictionary
- `tags`: List of tags for categorization

**Model Status:**
- `active`: Currently in production use
- `testing`: Being tested with limited traffic
- `inactive`: Registered but not deployed
- `deprecated`: Old version, scheduled for removal

### Listing Models

```bash
# List all models
curl "http://localhost:8000/registry/models" \
  -H "X-API-Key: your-api-key"

# Filter by status
curl "http://localhost:8000/registry/models?status=active" \
  -H "X-API-Key: your-api-key"
```

### Model Metadata

Get detailed information about a specific version:

```bash
curl "http://localhost:8000/registry/models/v1.0.0" \
  -H "X-API-Key: your-api-key"
```

Response:
```json
{
  "version": "v1.0.0",
  "name": "Production Model",
  "description": "Current production model",
  "created_at": "2024-12-08T10:30:00",
  "updated_at": "2024-12-08T10:30:00",
  "status": "active",
  "model_path": "models/credit_model_v1.pkl",
  "preprocessor_path": "models/preprocessor_v1.pkl",
  "metrics": {
    "accuracy": 0.8542,
    "brier_score": 0.1256
  },
  "tags": ["production", "baseline"],
  "traffic_weight": 0.8
}
```

## Traffic Routing

### How Traffic Splitting Works

1. **Consistent Hashing**: Each request generates a hash based on request data
2. **Deterministic Routing**: Same request always goes to same model version
3. **Weight-Based**: Traffic split according to configured weights

**Example:**
```
Weights: v1=0.7, v2=0.3
100 requests → 70 to v1, 30 to v2 (approximately)
Same user always gets same version
```

### Setting Traffic Weights

Weights must be between 0.0 and 1.0 and sum to 1.0:

```bash
# Gradual rollout scenarios

# 1. Initial testing (5% traffic to new version)
curl -X POST "http://localhost:8000/registry/ab-test/weights" \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"v1.0.0": 0.95, "v1.1.0": 0.05}'

# 2. Expand testing (20% traffic)
curl -X POST "http://localhost:8000/registry/ab-test/weights" \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"v1.0.0": 0.80, "v1.1.0": 0.20}'

# 3. Even split (A/B test)
curl -X POST "http://localhost:8000/registry/ab-test/weights" \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"v1.0.0": 0.50, "v1.1.0": 0.50}'

# 4. Majority new version (80% traffic)
curl -X POST "http://localhost:8000/registry/ab-test/weights" \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"v1.0.0": 0.20, "v1.1.0": 0.80}'

# 5. Full rollout (100% new version)
curl -X POST "http://localhost:8000/registry/ab-test/weights" \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"v1.1.0": 1.0}'
```

### Emergency Rollback

If issues are detected:

```bash
# Immediately route 100% traffic back to stable version
curl -X POST "http://localhost:8000/registry/ab-test/weights" \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"v1.0.0": 1.0}'
```

## API Endpoints

### Model Registry

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/registry/models` | GET | List all registered models |
| `/registry/models` | POST | Register new model version |
| `/registry/models/{version}` | GET | Get model metadata |
| `/registry/models/{version}/load` | POST | Load model into memory |
| `/registry/models/{version}/unload` | POST | Unload model from memory |

### A/B Testing

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/registry/ab-test/weights` | GET | Get current traffic weights |
| `/registry/ab-test/weights` | POST | Set traffic weights |
| `/registry/ab-test/compare/{v1}/{v2}` | GET | Compare two model versions |

## Monitoring

### Prometheus Metrics

**Per-Version Metrics:**
```promql
# Requests by model version
ab_test_requests_total{model_version="v1.0.0"}

# Predictions by version and outcome
model_version_predictions_total{version="v1.0.0", outcome="approved"}

# Active models count
active_models_count
```

**Comparison Queries:**
```promql
# Approval rate by version
sum(rate(model_version_predictions_total{outcome="approved"}[5m])) by (version)
/
sum(rate(model_version_predictions_total[5m])) by (version)

# Error rate by version
sum(rate(prediction_errors_total[5m])) by (model_version)
```

### Grafana Dashboard

Create a dashboard to monitor A/B test performance:

```json
{
  "panels": [
    {
      "title": "Traffic Split",
      "targets": [
        {"expr": "sum(rate(ab_test_requests_total[5m])) by (model_version)"}
      ]
    },
    {
      "title": "Approval Rate by Version",
      "targets": [
        {"expr": "sum(rate(model_version_predictions_total{outcome='approved'}[5m])) by (version) / sum(rate(model_version_predictions_total[5m])) by (version)"}
      ]
    }
  ]
}
```

## Best Practices

### 1. Gradual Rollout

**Don't:**
- ❌ Go from 0% → 100% immediately
- ❌ Run A/B test with equal split right away

**Do:**
- ✅ Start with 5-10% traffic to new version
- ✅ Monitor for 24-48 hours
- ✅ Gradually increase: 5% → 20% → 50% → 80% → 100%

### 2. Set Clear Success Criteria

Before starting A/B test, define:
- **Primary metric**: e.g., Brier score < 0.15
- **Secondary metrics**: Accuracy, fairness metrics
- **Guardrails**: Error rate < 1%, latency p95 < 500ms
- **Sample size**: Minimum requests needed for significance

### 3. Monitor Closely

During A/B test, monitor:
- **Performance metrics**: Per-version accuracy, calibration
- **Fairness metrics**: Ensure new model maintains fairness
- **System metrics**: Latency, error rates, memory usage
- **Business metrics**: Approval rates, risk distribution

### 4. Version Naming

Use semantic versioning:
- **v1.0.0**: Major.Minor.Patch
- **2024-12-08-hotfix**: Date-based for quick fixes
- **exp-calibration-v2**: Descriptive for experiments

### 5. Documentation

For each model version, document:
- Training data date range
- Hyperparameters used
- Performance metrics
- Known issues or limitations
- Rollback procedure

## Examples

### Example 1: Canary Deployment

Deploy new model to 10% of traffic:

```bash
# 1. Register new model
curl -X POST "http://localhost:8000/registry/models" \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "version": "v1.2.0-canary",
    "name": "Canary Test",
    "model_path": "models/credit_model_v1_2.pkl",
    "tags": ["canary"]
  }'

# 2. Load model
curl -X POST "http://localhost:8000/registry/models/v1.2.0-canary/load" \
  -H "X-API-Key: your-api-key"

# 3. Route 10% traffic
curl -X POST "http://localhost:8000/registry/ab-test/weights" \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"v1.0.0": 0.90, "v1.2.0-canary": 0.10}'

# 4. Monitor for 24 hours
# (Check Grafana dashboard, Prometheus alerts)

# 5a. If successful, increase traffic
curl -X POST "http://localhost:8000/registry/ab-test/weights" \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"v1.0.0": 0.50, "v1.2.0-canary": 0.50}'

# 5b. If issues detected, rollback
curl -X POST "http://localhost:8000/registry/ab-test/weights" \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"v1.0.0": 1.0}'
```

### Example 2: Champion/Challenger Test

Compare two models with 50/50 split:

```bash
# Set equal weights
curl -X POST "http://localhost:8000/registry/ab-test/weights" \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"champion-v1.0": 0.50, "challenger-v2.0": 0.50}'

# Run for 1 week, then compare
curl "http://localhost:8000/registry/ab-test/compare/champion-v1.0/challenger-v2.0" \
  -H "X-API-Key: your-api-key"

# Response shows metric differences
{
  "metric_differences": {
    "accuracy": {
      "value_a": 0.8542,
      "value_b": 0.8601,
      "absolute_difference": 0.0059,
      "percent_change": 0.69
    },
    "brier_score": {
      "value_a": 0.1256,
      "value_b": 0.1104,
      "absolute_difference": -0.0152,
      "percent_change": -12.10
    }
  }
}
```

### Example 3: Multi-Version Testing

Test 3 models simultaneously:

```bash
curl -X POST "http://localhost:8000/registry/ab-test/weights" \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "v1.0.0-baseline": 0.70,
    "v1.1.0-calibrated": 0.15,
    "v1.2.0-fairness": 0.15
  }'
```

## Troubleshooting

### Weights don't sum to 1.0

**Error:**
```
Traffic weights must sum to 1.0, got 0.95
```

**Solution:**
```bash
# Ensure weights sum exactly to 1.0
curl -X POST "http://localhost:8000/registry/ab-test/weights" \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"v1.0.0": 0.80, "v1.1.0": 0.20}'  # 0.80 + 0.20 = 1.0
```

### Model not found

**Error:**
```
Model version 'v1.1.0' not found in registry
```

**Solution:**
1. Check registered models:
   ```bash
   curl "http://localhost:8000/registry/models" -H "X-API-Key: your-api-key"
   ```
2. Register the model if missing (see Quick Start)

### A/B testing not enabled

**Error:**
```
A/B testing not enabled. Set AB_TESTING_ENABLED=true
```

**Solution:**
```bash
# Set environment variable
export AB_TESTING_ENABLED=true

# Restart API
uvicorn src.api:app --reload
```

### Model file not found

**Error:**
```
Model file not found: models/credit_model_v1_1.pkl
```

**Solution:**
1. Verify file exists:
   ```bash
   ls -la models/credit_model_v1_1.pkl
   ```
2. Use correct absolute or relative path
3. Ensure API has read permissions

## See Also

- [PRODUCTION_INFERENCE.md](PRODUCTION_INFERENCE.md) - Production deployment guide
- [CALIBRATION_GUIDE.md](CALIBRATION_GUIDE.md) - Model calibration
- [Grafana dashboards](grafana/README.md) - Monitoring setup
- [API documentation](http://localhost:8000/docs) - Full API reference
