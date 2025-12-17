# Fair Credit Score Prediction Model

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-122%20passed-brightgreen.svg)]()
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Production-ready credit scoring with fairness, explainability, and monitoring built into every layer.**

Made by team 11: Pierce Brookins and Aleya Banthavong

---

## 🎯 Why This Project Exists

Traditional credit scoring models often perpetuate historical biases. This project demonstrates how to build a **fair, transparent, and auditable** ML system for credit decisions:

- **Fairness First** – Mitigate discriminatory lending using CDI-based proxy groups with pre/in/post-processing fairness techniques
- **Full Transparency** – SHAP, LIME, and natural-language explanations for every decision
- **Production Ready** – FastAPI service, Docker images, Prometheus/Grafana monitoring, audit logging
- **Reproducible** – Typed configuration, deterministic seeds, serialized pipelines, comprehensive tests

---

## 🚀 Quick Start

```bash
# Clone and setup
git clone <repo-url>
cd credprediction

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install fastapi uvicorn  # For API

# Run tests (122 tests)
python -m pytest

# Train a model
python scripts/run_local.py --train

# Start API server
python scripts/run_local.py --api
# Visit http://127.0.0.1:8000/docs for Swagger UI
```

---

## 📦 What's Included

### Core ML Pipeline
| Feature | Description |
|---------|-------------|
| **XGBoost Classifier** | High-performance gradient boosting |
| **Hyperparameter Tuning** | Grid search and random search with cross-validation |
| **Model Calibration** | Isotonic and sigmoid calibration for accurate probabilities |
| **Cross-Validation** | Stratified K-fold with multiple metrics |
| **Model Versioning** | Built-in model registry with metadata tracking |
| **Model Persistence** | Pickle-based save/load with version tracking |

### Fairness Toolkit
| Feature | Description |
|---------|-------------|
| **Composite Disadvantage Index (CDI)** | Proxy-based fairness without protected attributes |
| **Reweighing** | Pre-processing to balance training data |
| **Threshold Adjustment** | Post-processing with group-specific thresholds |
| **Disparate Impact Analysis** | Ratio of positive rates between groups |
| **Statistical Parity** | Difference in positive rates |
| **Equalized Odds** | TPR/FPR parity across groups |
| **AIF360 Integration** | Industry-standard fairness toolkit |

### Explainability
| Feature | Description |
|---------|-------------|
| **SHAP Values** | Global and local feature importance |
| **LIME Explanations** | Local interpretable model-agnostic explanations |
| **Natural Language** | Human-readable decision explanations |
| **Waterfall Plots** | Visual feature contribution breakdown |
| **Force Plots** | Interactive SHAP visualizations |
| **Summary Plots** | Dataset-wide feature importance |

### Production Infrastructure
| Feature | Description |
|---------|-------------|
| **FastAPI Server** | High-performance async API with OpenAPI docs |
| **Batch Predictions** | Process multiple records efficiently |
| **Prometheus Metrics** | Request counts, latencies, fairness gauges |
| **Grafana Dashboards** | Pre-built monitoring dashboards |
| **Audit Logging** | JSON-formatted audit trail |
| **Docker Support** | Multi-stage builds for training and API |
| **Rate Limiting** | Optional request throttling |

---

## 📁 Project Structure

```
credprediction/
├── config/
│   ├── config.py              # Dataclass-based configuration
│   └── default_config.yaml    # All settings in one place
├── src/
│   ├── api.py                 # FastAPI server (45+ endpoints)
│   ├── main.py                # CLI entry point
│   ├── model.py               # CreditModel class
│   ├── preprocessing.py       # Data transformations
│   ├── fairness.py            # FairnessAnalyzer class
│   ├── explainability.py      # Explainer class (SHAP/LIME)
│   ├── model_registry.py      # Model versioning
│   ├── validators.py          # Input validation decorators
│   ├── protected_attribute.py # CDI utilities
│   ├── protocols.py           # Type protocols
│   └── utils.py               # Logging, metrics helpers
├── scripts/
│   ├── run_local.py           # Dev runner (train/api/test)
│   ├── generate_sample_data.py # Synthetic data generator
│   ├── benchmark_models.py    # Model comparison
│   ├── tune_hyperparameters.py # Hyperparameter search
│   ├── analyze_data_quality.py # Data profiling
│   └── compare_model_versions.py # A/B testing
├── tests/                     # 122 pytest tests
│   ├── test_model.py
│   ├── test_preprocessing.py
│   ├── test_fairness.py
│   ├── test_explainability.py
│   ├── test_integration.py
│   ├── test_validators.py
│   ├── test_protected_attribute.py
│   └── test_config.py
├── grafana/                   # Monitoring stack
│   ├── dashboards/            # Pre-built dashboards
│   ├── prometheus/            # Alert rules
│   └── docker-compose.yml     # Monitoring stack
├── models/                    # Saved models
├── outputs/                   # Metrics JSON
├── reports/figures/           # Generated plots
├── data/                      # Dataset directory
├── logs/                      # Application logs
├── Dockerfile                 # Training container
├── Dockerfile.api             # API container
├── docker-compose.yml         # Full stack
├── requirements.txt           # Dependencies
├── pyproject.toml             # Project metadata
└── RUN.md                     # Quick start guide
```

---

## 🛠️ Usage

### CLI Commands

```bash
# Full pipeline (generate data → train → evaluate)
python scripts/run_local.py

# Individual steps
python scripts/run_local.py --generate-data  # Create synthetic data
python scripts/run_local.py --train          # Train model
python scripts/run_local.py --api            # Start API server
python scripts/run_local.py --test           # Run pytest
python scripts/run_local.py --test-api       # Test API endpoints

# Using main module directly
python -m src.main run --data-path data/sample_credit_data.csv
python -m src.main run --data-path data/credit.csv --config config/custom.yaml
```

### Python API

```python
from src.model import CreditModel
from src.preprocessing import Preprocessor
from src.fairness import FairnessAnalyzer
from src.explainability import Explainer
from config.config import load_config

# Load config
config = load_config("config/default_config.yaml")

# Preprocess data
preprocessor = Preprocessor(config)
X_train, X_test, y_train, y_test, sample_weights = preprocessor.fit_transform(df)

# Train model with fairness weights
model = CreditModel(config)
model.train(X_train, y_train, sample_weights=sample_weights)

# Evaluate
metrics = model.evaluate(X_test, y_test)
print(f"Accuracy: {metrics['accuracy']:.3f}")
print(f"ROC-AUC: {metrics['roc_auc']:.3f}")

# Fairness analysis
fairness = FairnessAnalyzer(config)
y_pred = model.predict(X_test)
fairness_metrics = fairness.calculate_metrics(X_test, y_test, y_pred)
print(f"Disparate Impact: {fairness_metrics.disparate_impact:.3f}")

# Explanations
explainer = Explainer(model, X_train, config)
explainer.setup_shap_explainer()
explanation = explainer.explain_individual(X_test, idx=0, prediction=1)
print(explanation["natural_language"])
```

### Hyperparameter Tuning

```python
model = CreditModel(config)

param_grid = {
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.1, 0.2],
    "n_estimators": [50, 100, 200],
}

results = model.tune_hyperparameters(
    X_train, y_train,
    param_grid=param_grid,
    search_type="random",  # or "grid"
    cv=5,
    n_iter=20,
)

print(f"Best params: {results['best_params']}")
print(f"Best CV score: {results['best_score']:.3f}")
```

### Model Calibration

```python
# Train base model
model.train(X_train, y_train)

# Calibrate probabilities
model.calibrate_model(X_calib, y_calib, method="isotonic")

# Get calibration curve
calib_data = model.get_calibration_curve(X_test, y_test)
print(f"Brier Score: {calib_data['brier_score']:.3f}")
```

---

## 🌐 API Endpoints

### Core Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/ready` | Readiness probe |
| `POST` | `/predict` | Single prediction |
| `POST` | `/batch_predict` | Batch predictions |
| `GET` | `/explain/{idx}` | Get explanation for sample |

### Fairness Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/fairness_metrics` | Current fairness metrics |
| `GET` | `/fairness_config` | Fairness configuration |
| `GET` | `/fairness_report` | Detailed fairness report |

### Model Info

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/model_info` | Model metadata |
| `GET` | `/feature_importance` | Feature importance scores |
| `GET` | `/calibration_curve` | Calibration data |

### Monitoring

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/metrics` | Prometheus metrics |
| `GET` | `/docs` | Swagger UI |
| `GET` | `/redoc` | ReDoc documentation |

### Example Request

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Income": 75000,
    "Debt": 15000,
    "Loan_Amount": 25000,
    "Loan_Term": 36,
    "Num_Credit_Cards": 3,
    "Gender": "Female",
    "Education": "Bachelors",
    "Payment_History": "Good",
    "Employment_Status": "Employed",
    "Residence_Type": "Rented",
    "Marital_Status": "Single"
  }'
```

### Example Response

```json
{
  "creditworthy": true,
  "probability": 0.73,
  "confidence": "high",
  "cdi_score": 2,
  "group": "unprivileged",
  "explanation": {
    "decision": "APPROVED",
    "top_positive_factors": [
      {"feature": "Income", "impact": "strong positive"},
      {"feature": "Payment_History_Good", "impact": "moderate positive"}
    ],
    "top_negative_factors": [
      {"feature": "Loan_Amount", "impact": "weak negative"}
    ]
  }
}
```

---

## ⚖️ Fairness Methodology

### The Problem

Traditional credit models often discriminate against protected groups (race, gender, age). However, we can't simply remove these features because:
1. They may not be in the data
2. Other features can proxy for them
3. Legal requirements vary by jurisdiction

### Our Solution: Composite Disadvantage Index (CDI)

We create a **proxy disadvantage score** based on features correlated with historical discrimination:

```yaml
cdi:
  factors:
    Education: "High School"      # +1 if education is high school only
    Residence_Type: "Rented"      # +1 if renting
    Marital_Status: "Single"      # +1 if single
    Gender: "Female"              # +1 if female
  threshold: 2                    # CDI >= 2 = "unprivileged" group
```

### Fairness Pipeline

```
1. PRE-PROCESSING (Reweighing)
   └─ Adjust sample weights to balance groups
   
2. IN-PROCESSING (Training)
   └─ Train on reweighted data
   
3. POST-PROCESSING (Thresholds)
   └─ Apply group-specific decision thresholds:
      - Privileged: 0.50 (standard threshold)
      - Unprivileged: 0.40 (lower threshold = more approvals)
```

### Metrics We Track

| Metric | Formula | Target | Meaning |
|--------|---------|--------|----------|
| **Disparate Impact** | P(Y=1|unpriv) / P(Y=1|priv) | ≥ 0.80 | 80% rule |
| **Statistical Parity** | P(Y=1|unpriv) - P(Y=1|priv) | < 0.05 | Equal positive rates |
| **Equalized Odds** | max(ΔTPR, ΔFPR) | < 0.10 | Equal error rates |
| **PPV Parity** | PPV_unpriv - PPV_priv | ~ 0 | Equal precision |

---

## 🔍 Explainability

### SHAP (SHapley Additive exPlanations)

```python
explainer = Explainer(model, X_train)
explainer.setup_shap_explainer()

# Global feature importance
explainer.generate_global_importance_plot(X_test, "reports/figures/")

# Individual explanation
explanation = explainer.explain_individual(X_test, idx=0, prediction=1)
```

### Generated Visualizations

- `feature_importance_shap.png` - Bar chart of mean |SHAP| values
- `feature_importance_detailed.png` - Beeswarm plot showing value distributions
- `waterfall_plot_*.png` - Individual prediction breakdowns
- `force_plot_*.html` - Interactive force plots

### Natural Language Explanations

```
Prediction: APPROVED (probability: 0.78)

✅ Factors supporting approval:
  • Income ($85,000) had a STRONG POSITIVE impact
  • Low Debt-to-Income ratio (0.15) had a MODERATE POSITIVE impact
  • Good Payment History had a MODERATE POSITIVE impact

❌ Factors against approval:
  • High Loan Amount ($50,000) had a WEAK NEGATIVE impact
  • Number of Credit Cards (5) had a WEAK NEGATIVE impact
```

---

## ⚙️ Configuration

All settings are centralized in `config/default_config.yaml`:

```yaml
# Model settings
model:
  random_state: 42
  test_size: 0.2
  xgb_params:
    n_estimators: 100
    max_depth: 6
    learning_rate: 0.1
    eval_metric: "logloss"

# Fairness settings
fairness:
  threshold_privileged: 0.5
  threshold_unprivileged: 0.4
  target_disparate_impact_ratio: 0.80
  target_statistical_parity_diff: 0.05
  apply_reweighting: true

# CDI settings
cdi:
  proxy_column: "Proxy_Disadvantaged"
  threshold: 2
  factors:
    Education: "High School"
    Residence_Type: "Rented"
    Marital_Status: "Single"
    Gender: "Female"

# Explainability
explainability:
  num_samples_to_explain: 5
  use_shap: true
  use_lime: true
  lime_num_features: 10

# Hyperparameter tuning
tuning:
  search_type: "random"
  cv_folds: 5
  n_iter: 50

# Model calibration
calibration:
  enabled: true
  method: "isotonic"
```

---

## 📊 Monitoring

### Prometheus Metrics

The API exposes metrics at `/metrics`:

```
# Predictions
prediction_requests_total{outcome="approved",group="privileged"}
prediction_latency_seconds_bucket{le="0.1"}
prediction_errors_total{error_type="model_not_loaded"}

# Fairness
fairness_disparate_impact_ratio
fairness_statistical_parity_difference
fairness_equalized_odds_difference

# Data Quality
data_quality_validation_issues_total{issue_type="missing_values"}
data_quality_unseen_categories_total{column="Education"}

# Model
model_cv_score_mean
model_feature_drift{feature="Income"}
```

### Grafana Dashboards

Pre-built dashboards in `grafana/dashboards/`:

| Dashboard | Metrics |
|-----------|--------|
| **Model Performance** | Accuracy, AUC, latency, error rates |
| **Fairness Metrics** | DI ratio, stat parity, equalized odds over time |
| **Data Quality** | Missing values, drift detection, validation issues |
| **Calibration** | Calibration curves, Brier score |

### Start Monitoring Stack

```bash
cd grafana
docker-compose up -d

# Grafana: http://localhost:3000 (admin/admin)
# Prometheus: http://localhost:9090
```

### Alerts

```yaml
# grafana/prometheus/alert_rules.yml
groups:
  - name: fairness_alerts
    rules:
      - alert: FairnessViolation
        expr: fairness_disparate_impact_ratio < 0.8
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Disparate impact below legal threshold"
```

---

## 🧪 Testing

### Run All Tests

```bash
python -m pytest                    # All 122 tests
python -m pytest -v                 # Verbose output
python -m pytest --cov=src          # With coverage
python -m pytest -x                 # Stop on first failure
```

### Run Specific Tests

```bash
python -m pytest tests/test_model.py
python -m pytest tests/test_fairness.py::TestFairnessAnalyzer
python -m pytest -k "calibration"   # Tests matching pattern
```

### Test Categories

| File | Tests | Coverage |
|------|-------|----------|
| `test_model.py` | 17 | Model training, prediction, tuning, calibration |
| `test_preprocessing.py` | 19 | Data transforms, encoding, scaling |
| `test_fairness.py` | 14 | Reweighing, thresholds, metrics |
| `test_explainability.py` | 15 | SHAP, LIME, plots |
| `test_integration.py` | 13 | End-to-end pipelines |
| `test_validators.py` | 17 | Input validation |
| `test_protected_attribute.py` | 14 | CDI utilities |
| `test_config.py` | 11 | Configuration loading |

---

## 🐳 Docker

### Build Images

```bash
# Training image
docker build -t credit-prediction .

# API image
docker build -f Dockerfile.api -t credit-prediction-api .
```

### Run Containers

```bash
# Training
docker run -v $(pwd)/data:/app/data credit-prediction

# API
docker run -p 8000:8000 credit-prediction-api
```

### Docker Compose (Full Stack)

```bash
docker-compose up

# Services:
# - API: http://localhost:8000
# - Prometheus: http://localhost:9090
# - Grafana: http://localhost:3000
```

---

## 📚 Additional Documentation

| Document | Description |
|----------|-------------|
| [RUN.md](RUN.md) | Quick start commands |
| [LOCAL_DEVELOPMENT.md](LOCAL_DEVELOPMENT.md) | Detailed dev setup |
| [DEPLOYMENT.md](DEPLOYMENT.md) | Production deployment guide |
| [TUNING_GUIDE.md](TUNING_GUIDE.md) | Hyperparameter tuning |
| [CALIBRATION_GUIDE.md](CALIBRATION_GUIDE.md) | Model calibration |
| [AB_TESTING_GUIDE.md](AB_TESTING_GUIDE.md) | A/B testing models |
| [grafana/README.md](grafana/README.md) | Monitoring setup |

---

## 📈 Data Schema

### Input Features

| Feature | Type | Description |
|---------|------|-------------|
| `Income` | numeric | Annual income in dollars |
| `Debt` | numeric | Total debt in dollars |
| `Loan_Amount` | numeric | Requested loan amount |
| `Loan_Term` | numeric | Loan term in months |
| `Num_Credit_Cards` | numeric | Number of credit cards |
| `Gender` | categorical | Male/Female |
| `Education` | categorical | High School/Bachelor's/Master's/PhD |
| `Payment_History` | categorical | Poor/Fair/Good |
| `Employment_Status` | categorical | Employed/Self-Employed/Unemployed |
| `Residence_Type` | categorical | Owned/Rented/Other |
| `Marital_Status` | categorical | Single/Married/Divorced |

### Derived Features (Auto-generated)

| Feature | Formula |
|---------|---------|
| `Debt_to_Income` | Debt / Income |
| `Loan_to_Income` | Loan_Amount / Income |
| `CDI` | Sum of disadvantage factors |
| `Proxy_Disadvantaged` | 1 if CDI >= threshold, else 0 |

### Target

| Feature | Type | Values |
|---------|------|--------|
| `Creditworthiness` | binary | 0 = Denied, 1 = Approved |

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Write tests first
4. Implement your changes
5. Run the test suite (`python -m pytest`)
6. Format code (`black .`)
7. Submit a pull request

### Code Quality Standards

- **Black** for formatting
- **Flake8** for linting
- **MyPy** for type checking
- **Pytest** for testing
- Files under 600 lines (split if larger)
- DRY, YAGNI, SOLID principles

---

## 📝 License

MIT License - See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- [AIF360](https://github.com/Trusted-AI/AIF360) - IBM's AI Fairness toolkit
- [Fairlearn](https://github.com/fairlearn/fairlearn) - Microsoft's fairness toolkit
- [SHAP](https://github.com/slundberg/shap) - SHapley Additive exPlanations
- [LIME](https://github.com/marcotcr/lime) - Local Interpretable Model-agnostic Explanations
- [XGBoost](https://github.com/dmlc/xgboost) - Gradient boosting framework

---

**Built with ❤️ for fair and transparent machine learning**
