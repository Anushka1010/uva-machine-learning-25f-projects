# Production Inference Guide

This guide covers best practices for deploying and using the Fair Credit Prediction model in production environments.

## Table of Contents

- [Overview](#overview)
- [Model Artifacts](#model-artifacts)
- [Preprocessor State](#preprocessor-state)
- [Handling Unseen Categories](#handling-unseen-categories)
- [Feature Alignment](#feature-alignment)
- [API Deployment](#api-deployment)
- [Monitoring](#monitoring)
- [Error Handling](#error-handling)
- [Performance Optimization](#performance-optimization)
- [Data Drift Detection](#data-drift-detection)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

## Overview

Production inference requires careful handling of preprocessing, feature engineering, and model serving to ensure consistent, reliable predictions.

### Key Challenges in Production

1. **Feature consistency**: Production features must match training features exactly
2. **Unseen categories**: New categorical values not seen during training
3. **Missing values**: Handling incomplete data gracefully
4. **Data drift**: Distribution changes over time
5. **Latency**: Meeting real-time prediction requirements
6. **Monitoring**: Detecting degradation and errors

## Model Artifacts

### Required Artifacts

For production deployment, save and load these artifacts:

```python
# After training
preprocessor.save("models/preprocessor.pkl")  # ← Essential!
model.save("models/credit_model.pkl")         # ← Essential!

# Optional but recommended
explainer.save("models/explainer.pkl")
fairness_config.save("models/fairness_config.pkl")
```

### What Gets Saved

**Preprocessor (`preprocessor.pkl`):**
- Fitted encoders for categorical variables
- Feature names and order
- Scaling parameters (if used)
- CDI calculation logic
- Category mappings

**Model (`credit_model.pkl`):**
- Trained XGBoost model
- Calibration model (if calibrated)
- Feature importance
- Training metadata
- Model version

### Loading in Production

```python
from src.preprocessing import Preprocessor
from src.model import CreditModel
from config.config import load_config

# Load configuration
config = load_config("config/production_config.yaml")

# Load preprocessor (critical for consistent features!)
preprocessor = Preprocessor(config)
preprocessor.load("models/preprocessor.pkl")

# Load model
model = CreditModel(config)
model.load("models/credit_model.pkl")

# Now ready for predictions
```

## Preprocessor State

### Why Preprocessor State Matters

**Without saved preprocessor:**
```python
# ❌ WRONG: Will create different features
preprocessor_new = Preprocessor(config)
X = preprocessor_new.prepare_inference_features(data)  # Different encoding!
model.predict(X)  # Unpredictable results
```

**With saved preprocessor:**
```python
# ✓ CORRECT: Uses same encoding as training
preprocessor.load("models/preprocessor.pkl")
X = preprocessor.prepare_inference_features(data)  # Same encoding!
model.predict(X)  # Reliable results
```

### What Changes Without Saved State

| Component | Without State | With State |
|-----------|---------------|------------|
| **Categorical encoding** | Random order | Fixed, consistent |
| **One-hot columns** | May differ | Exact match |
| **Feature order** | Unpredictable | Guaranteed match |
| **Missing categories** | Error or wrong encoding | Handled gracefully |
| **Scaling parameters** | Recomputed (wrong) | Preserved (correct) |

### Inference Pipeline

```python
def make_prediction(raw_data: pd.DataFrame) -> np.ndarray:
    """Production-ready prediction pipeline."""

    # Step 1: Prepare features using saved preprocessor
    X = preprocessor.prepare_inference_features(raw_data)

    # Step 2: Validate features match training
    assert X.shape[1] == len(preprocessor.get_feature_names()) + 1  # +1 for proxy

    # Step 3: Generate predictions
    y_prob = model.predict_proba(X)

    # Step 4: Apply fairness-aware thresholds
    protected = X[config.cdi.proxy_column].values
    y_pred = fairness_analyzer.apply_threshold_adjustment(y_prob, protected)

    return y_pred, y_prob
```

## Handling Unseen Categories

### The Problem

```python
# Training data had: Education = ["High School", "Bachelor's", "Master's"]
# Production data has: Education = "Trade School"  # ← Unseen!
```

Without proper handling, this causes:
- ❌ Encoding errors
- ❌ Missing feature columns
- ❌ Model prediction failures

### The Solution

The preprocessor's `prepare_inference_features()` method handles this:

```python
# Automatic handling of unseen categories
X = preprocessor.prepare_inference_features(new_data)

# Unseen categories are:
# 1. Detected
# 2. Mapped to special "unknown" encoding
# 3. Logged for monitoring
# 4. Prediction proceeds safely
```

### Manual Handling (if needed)

```python
from src.preprocessing import Preprocessor

preprocessor = Preprocessor(config)
preprocessor.load("models/preprocessor.pkl")

# Explicitly handle unseen categories
X_safe = preprocessor.handle_unseen_categories(raw_data)

# Check for unseen categories
known_cats = preprocessor._category_mappings  # Stored during fit
for col in categorical_columns:
    seen = set(known_cats.get(col, []))
    current = set(raw_data[col].unique())
    unseen = current - seen

    if unseen:
        logger.warning(f"Unseen categories in {col}: {unseen}")
        # Trigger monitoring alert
```

### Monitoring Unseen Categories

```python
# In your prediction endpoint
unseen_count = 0
for col in categorical_columns:
    unseen = detect_unseen(data[col], known_categories[col])
    unseen_count += len(unseen)

    # Update Prometheus metrics
    if unseen:
        for value in unseen:
            DATA_UNSEEN_CATEGORIES.labels(column=col).inc()

# Alert if too many unseen categories
if unseen_count > threshold:
    alert_team(f"High unseen category rate: {unseen_count}")
```

## Feature Alignment

### Ensuring Feature Consistency

Production features must match training features exactly:

```python
# Get expected features from preprocessor
expected_features = preprocessor.get_feature_names()

# Align inference data to expected features
X_aligned = preprocessor.align_to_feature_space(X_raw)

# Verify alignment
assert list(X_aligned.columns) == expected_features
assert X_aligned.shape[1] == len(expected_features)
```

### Common Alignment Issues

**Problem 1: Missing Features**
```python
# Training: 50 features
# Inference: 48 features (missing 2)

# Solution: Add missing features with default values
for col in expected_features:
    if col not in X_inference.columns:
        X_inference[col] = 0.0  # or appropriate default
```

**Problem 2: Extra Features**
```python
# Training: 50 features
# Inference: 52 features (2 extra)

# Solution: Drop extra features
X_inference = X_inference[expected_features]
```

**Problem 3: Wrong Order**
```python
# Training: [feature_1, feature_2, ...]
# Inference: [feature_2, feature_1, ...]  # Wrong order!

# Solution: Reorder to match training
X_inference = X_inference[expected_features]
```

### Automated Alignment

The `align_to_feature_space()` method handles all of these:

```python
# Handles missing, extra, and reordering automatically
X_aligned = preprocessor.align_to_feature_space(X_raw)
```

## API Deployment

### Production API Setup

**1. Environment Variables**

```bash
# .env file
MODEL_PATH=models/credit_model.pkl
PREPROCESSOR_PATH=models/preprocessor.pkl
CONFIG_PATH=config/production_config.yaml
API_KEY=your-secret-key-here
CORS_ORIGINS=https://yourdomain.com
```

**2. Start API Server**

```bash
# Production mode
gunicorn src.api:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --timeout 60 \
  --access-logfile - \
  --error-logfile -

# Or with Docker
docker-compose up -d
```

**3. Health Checks**

```python
# Configure load balancer health check
GET /health

# Response:
{
  "status": "healthy",
  "model_loaded": true,
  "version": "0.1.0",
  "uptime_seconds": 86400
}
```

### Making Predictions

**Single Prediction:**

```python
import requests

response = requests.post(
    "https://api.example.com/predict",
    headers={"X-API-Key": "your-key"},
    json={
        "Income": 75000,
        "Debt": 15000,
        "Loan_Amount": 25000,
        "Loan_Term": 36,
        "Num_Credit_Cards": 3,
        "Gender": "Female",
        "Education": "Bachelor's",
        "Payment_History": "Good",
        "Employment_Status": "Employed",
        "Residence_Type": "Rented",
        "Marital_Status": "Single"
    }
)

prediction = response.json()
# {
#   "creditworthy": true,
#   "probability": 0.78,
#   "confidence": "high",
#   "cdi_score": 2,
#   "group": "unprivileged"
# }
```

**Batch Predictions:**

```python
response = requests.post(
    "https://api.example.com/batch_predict",
    headers={"X-API-Key": "your-key"},
    json={"records": [record1, record2, ...]}  # Up to 1000
)

results = response.json()
# {
#   "predictions": [...],
#   "count": 100,
#   "processing_time_seconds": 0.234
# }
```

### API Rate Limiting

```python
# Add rate limiting middleware
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/predict")
@limiter.limit("100/minute")
async def predict(...):
    ...
```

## Monitoring

### Key Metrics to Track

**1. Model Performance Metrics**

```python
# Track via Prometheus
model_accuracy = Gauge('model_accuracy', 'Model accuracy on recent data')
model_auc = Gauge('model_auc', 'Model AUC on recent data')
brier_score = Gauge('brier_score', 'Calibration brier score')

# Update periodically
def update_model_metrics():
    recent_preds = get_recent_predictions(hours=24)
    if len(recent_preds) > 100:
        accuracy = calculate_accuracy(recent_preds)
        model_accuracy.set(accuracy)
```

**2. Fairness Metrics**

```python
# Monitor fairness over time
FAIRNESS_DISPARATE_IMPACT.set(current_di)
FAIRNESS_STATISTICAL_PARITY.set(current_spd)

# Alert if fairness degrades
if current_di < 0.80:
    alert_team("Fairness violation: DI < 0.80")
```

**3. Data Quality Metrics**

```python
# Track data quality issues
DATA_MISSING_VALUES.set(missing_count)
DATA_UNSEEN_CATEGORIES.labels(column=col).inc()
DATA_QUALITY_SCORE.set(quality_score)
```

**4. Operational Metrics**

```python
# Prediction latency
PREDICTION_LATENCY.observe(duration)

# Throughput
PREDICTIONS_TOTAL.labels(outcome=outcome, group=group).inc()

# Error rates
PREDICTION_ERRORS.labels(error_type="timeout").inc()
```

### Grafana Dashboards

Create dashboards to visualize:

1. **Model Performance Dashboard**
   - Accuracy, precision, recall trends
   - AUC over time
   - Brier score (calibration quality)

2. **Fairness Dashboard**
   - Disparate impact ratio
   - Statistical parity difference
   - Group-wise approval rates

3. **Data Quality Dashboard**
   - Missing value rates
   - Unseen category frequency
   - Data drift indicators

4. **Operational Dashboard**
   - Request rate
   - Latency percentiles (p50, p95, p99)
   - Error rates
   - System resources

### Alerting Rules

```yaml
# Prometheus alerting rules
groups:
  - name: model_alerts
    rules:
      - alert: ModelAccuracyDegradation
        expr: model_accuracy < 0.80
        for: 1h
        annotations:
          summary: "Model accuracy below threshold"

      - alert: FairnessViolation
        expr: fairness_disparate_impact < 0.80
        for: 30m
        annotations:
          summary: "Fairness constraint violated"

      - alert: HighErrorRate
        expr: rate(prediction_errors_total[5m]) > 0.05
        for: 5m
        annotations:
          summary: "High prediction error rate"

      - alert: DataQualityIssue
        expr: data_quality_score < 70
        for: 15m
        annotations:
          summary: "Data quality degraded"
```

## Error Handling

### Graceful Degradation

```python
@app.post("/predict")
async def predict(request: PredictionRequest):
    try:
        # Attempt prediction
        result = model.predict(preprocess(request))
        return result

    except UnseeCategory Error as e:
        # Log and handle gracefully
        logger.warning(f"Unseen category: {e}")
        PREDICTION_ERRORS.labels(error_type="unseen_category").inc()

        # Return with warning
        return {
            "creditworthy": False,  # Conservative default
            "probability": 0.5,
            "confidence": "low",
            "warning": "Data quality issue detected"
        }

    except Exception as e:
        # Critical error
        logger.error(f"Prediction failed: {e}")
        PREDICTION_ERRORS.labels(error_type="critical").inc()
        raise HTTPException(status_code=500, detail="Prediction service unavailable")
```

### Input Validation

```python
from pydantic import BaseModel, Field, validator

class PredictionRequest(BaseModel):
    Income: float = Field(..., ge=0, le=1000000)
    Debt: float = Field(..., ge=0, le=1000000)
    Loan_Amount: float = Field(..., ge=0, le=1000000)
    Loan_Term: int = Field(..., ge=1, le=360)

    @validator('Income')
    def validate_income(cls, v):
        if v <= 0:
            raise ValueError('Income must be positive')
        if v > 1000000:
            raise ValueError('Income suspiciously high')
        return v
```

### Retry Logic

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10)
)
def make_prediction_with_retry(data):
    return model.predict(data)
```

## Performance Optimization

### Batch Processing

```python
# Instead of 1000 individual requests
for record in records:
    predict(record)  # Slow!

# Process in batch
predict_batch(records)  # Much faster!
```

**Performance comparison:**
- Individual: 1000 predictions × 20ms = 20 seconds
- Batch: 1000 predictions in batch = 0.5 seconds (40x faster!)

### Caching

```python
from functools import lru_cache
import hashlib

@lru_cache(maxsize=1000)
def get_prediction(request_hash):
    """Cache predictions for identical requests."""
    return model.predict(request)

def predict_with_cache(request):
    # Hash request to use as cache key
    request_hash = hashlib.md5(
        str(request.dict()).encode()
    ).hexdigest()

    return get_prediction(request_hash)
```

### Model Serving Optimization

```python
# Preload model at startup (not per request)
@app.on_event("startup")
async def load_model():
    global model, preprocessor
    model = load_model_from_disk()
    preprocessor = load_preprocessor_from_disk()

# Use async for I/O-bound operations
@app.post("/predict")
async def predict(...):
    # Model inference is CPU-bound, run in thread pool
    result = await asyncio.to_thread(model.predict, X)
    return result
```

### Reduce Latency

**Optimization checklist:**
- ✅ Use batch predictions when possible
- ✅ Cache frequent predictions
- ✅ Precompute static features
- ✅ Use efficient data structures (numpy > pandas for inference)
- ✅ Profile and optimize bottlenecks
- ✅ Use async I/O for database/API calls
- ✅ Consider model quantization
- ✅ Deploy closer to users (CDN/edge)

## Data Drift Detection

### Types of Drift

**1. Feature Drift (Covariate Shift)**
```python
# Distribution of input features changes
# Training: Mean income = $60K
# Production: Mean income = $75K  # Drift!
```

**2. Label Drift (Prior Probability Shift)**
```python
# Distribution of outcomes changes
# Training: 30% default rate
# Production: 45% default rate  # Drift!
```

**3. Concept Drift**
```python
# Relationship between features and target changes
# Previously: High income → Low default risk
# Now: High income → Medium default risk  # Economic shift
```

### Detecting Drift

**Statistical Tests:**

```python
from scipy.stats import ks_2samp

def detect_feature_drift(train_data, production_data, threshold=0.05):
    """Detect drift using Kolmogorov-Smirnov test."""

    drift_detected = {}

    for column in train_data.columns:
        statistic, p_value = ks_2samp(
            train_data[column],
            production_data[column]
        )

        drift_detected[column] = p_value < threshold

    return drift_detected
```

**Monitoring Drift:**

```python
# Periodically check for drift
def monitor_drift():
    recent_data = get_production_data(days=7)
    training_data = load_training_data()

    drift = detect_feature_drift(training_data, recent_data)

    drifted_features = [col for col, drifted in drift.items() if drifted]

    if drifted_features:
        logger.warning(f"Drift detected in: {drifted_features}")
        alert_team(f"Data drift detected: {len(drifted_features)} features")

        # Consider retraining
        if len(drifted_features) > 5:
            trigger_retraining_pipeline()
```

### Handling Drift

**Options:**
1. **Retrain model** on recent data
2. **Online learning** (continuous updates)
3. **Ensemble** old and new models
4. **Feature engineering** to account for changes
5. **Alert stakeholders** if drift is expected (business changes)

## Best Practices

### 1. Version Everything

```
models/
├── v1/
│   ├── credit_model.pkl
│   ├── preprocessor.pkl
│   ├── config.yaml
│   └── training_metadata.json
├── v2/
│   ├── credit_model.pkl
│   ├── preprocessor.pkl
│   ├── config.yaml
│   └── training_metadata.json
└── current -> v2/  # Symlink to current version
```

### 2. Maintain Audit Logs

```python
# Log every prediction
prediction_log = {
    "timestamp": datetime.now(),
    "model_version": "v2.1",
    "input_data": request.dict(),
    "prediction": result,
    "probability": prob,
    "group": group,
    "request_id": request_id
}

save_to_audit_log(prediction_log)
```

### 3. A/B Testing

```python
# Split traffic between model versions
if user_id % 100 < 10:  # 10% traffic
    prediction = model_v2.predict(X)
    model_version = "v2"
else:  # 90% traffic
    prediction = model_v1.predict(X)
    model_version = "v1"

# Track performance by version
track_prediction(prediction, model_version)
```

### 4. Canary Deployments

```python
# Gradually roll out new model
traffic_split = {
    "v1": 0.90,  # 90% on old model
    "v2": 0.10   # 10% on new model
}

# Monitor for issues
if model_v2_error_rate > threshold:
    rollback_to_v1()
else:
    increase_v2_traffic()
```

### 5. Regular Retraining

```python
# Retrain schedule
if months_since_last_training >= 3:
    new_model = retrain_on_recent_data()
    validate_model(new_model)
    if performance_acceptable(new_model):
        deploy_model(new_model)
```

## Troubleshooting

### Problem: Predictions Different from Training

**Diagnosis:**
```python
# Check feature alignment
print("Training features:", len(train_features))
print("Inference features:", len(inference_features))
print("Match:", set(train_features) == set(inference_features))
```

**Common causes:**
- Preprocessor not saved/loaded
- Different encoding for categories
- Wrong feature order
- Missing features

**Solution:**
```python
# Always use saved preprocessor
preprocessor.load("models/preprocessor.pkl")
X = preprocessor.prepare_inference_features(data)
```

### Problem: High Latency in Production

**Diagnosis:**
```python
import cProfile

cProfile.run('make_predictions(data)')
# Identify bottleneck: preprocessing, model, post-processing?
```

**Solutions:**
- Use batch predictions
- Cache frequent requests
- Optimize preprocessing
- Use faster serialization (joblib > pickle)
- Consider model compression

### Problem: Memory Issues with Large Batches

**Solution:**
```python
# Process in chunks
def predict_large_batch(data, chunk_size=1000):
    results = []
    for i in range(0, len(data), chunk_size):
        chunk = data[i:i+chunk_size]
        chunk_results = model.predict(chunk)
        results.extend(chunk_results)
    return results
```

## See Also

- [TUNING_GUIDE.md](TUNING_GUIDE.md) - Hyperparameter tuning
- [CALIBRATION_GUIDE.md](CALIBRATION_GUIDE.md) - Model calibration
- [DEPLOYMENT.md](DEPLOYMENT.md) - Deployment guide
- [README.md](README.md) - Project overview
