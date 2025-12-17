# Local Development Guide

Get the Fair Credit Score Prediction system running on your local machine in minutes.

## Table of Contents

- [Quick Start](#quick-start)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Generate Sample Data](#generate-sample-data)
- [Run the Training Pipeline](#run-the-training-pipeline)
- [Start the API Server](#start-the-api-server)
- [Test the API](#test-the-api)
- [Run Tests](#run-tests)
- [Common Issues](#common-issues)

---

## Quick Start

```bash
# 1. Setup environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Generate sample data
python scripts/generate_sample_data.py

# 3. Train the model
python -m src.main run --data-path data/sample_credit_data.csv

# 4. Start the API
uvicorn src.api:app --reload --host 127.0.0.1 --port 8000

# 5. Test it!
curl http://127.0.0.1:8000/health
```

---

## Prerequisites

- **Python 3.10+** (check with `python --version`)
- **pip** (comes with Python)
- **Git** (for cloning)

### Check Your Python Version

```bash
python --version
# Should output: Python 3.10.x or higher
```

If you have multiple Python versions:
```bash
python3.11 --version
# Use python3.11 instead of python in commands below
```

---

## Installation

### Step 1: Clone the Repository

```bash
git clone <repo-url>
cd credprediction-opus45
```

### Step 2: Create Virtual Environment

```bash
# Create venv
python -m venv venv

# Activate it
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows CMD
venv\Scripts\Activate.ps1       # Windows PowerShell
```

### Step 3: Install Dependencies

```bash
# Install all requirements
pip install -r requirements.txt

# Install dev dependencies (for testing)
pip install pytest pytest-cov black flake8

# Install API dependencies
pip install fastapi uvicorn
```

### Step 4: Create Required Directories

```bash
mkdir -p data models explanations reports/figures logs outputs
```

---

## Generate Sample Data

We provide a script to generate synthetic credit data for testing:

```bash
python scripts/generate_sample_data.py
```

This creates `data/sample_credit_data.csv` with 1000 realistic synthetic records.

### Manual Data Generation

Alternatively, you can generate data in Python:

```python
import pandas as pd
import numpy as np

np.random.seed(42)
n = 1000

data = pd.DataFrame({
    'Income': np.random.randint(25000, 200000, n),
    'Debt': np.random.randint(0, 80000, n),
    'Loan_Amount': np.random.randint(5000, 150000, n),
    'Loan_Term': np.random.choice([12, 24, 36, 48, 60], n),
    'Num_Credit_Cards': np.random.randint(0, 12, n),
    'Credit_Score': np.random.randint(300, 850, n),
    'Gender': np.random.choice(['Male', 'Female'], n),
    'Education': np.random.choice(['High School', "Bachelor's", "Master's", 'PhD'], n),
    'Payment_History': np.random.choice(['Excellent', 'Good', 'Fair', 'Poor'], n),
    'Employment_Status': np.random.choice(['Employed', 'Self-Employed', 'Unemployed'], n),
    'Residence_Type': np.random.choice(['Owned', 'Rented', 'Mortgage'], n),
    'Marital_Status': np.random.choice(['Single', 'Married', 'Divorced'], n),
    'Creditworthiness': np.random.choice([0, 1], n, p=[0.35, 0.65]),
})

data.to_csv('data/sample_credit_data.csv', index=False)
print(f"Created data/sample_credit_data.csv with {len(data)} records")
```

---

## Run the Training Pipeline

### Basic Run

```bash
python -m src.main run --data-path data/sample_credit_data.csv
```

### With Custom Config

```bash
python -m src.main run --data-path data/sample_credit_data.csv --config config/default_config.yaml
```

### What Happens:

1. 📂 **Data Loading** - Loads and validates CSV
2. 🔧 **Preprocessing** - Converts types, creates features, calculates CDI
3. ⚖️ **Fairness Pre-processing** - Applies reweighing
4. 🤖 **Model Training** - Trains XGBoost classifier
5. 📊 **Evaluation** - Calculates accuracy, precision, recall, ROC AUC
6. 🎯 **Fairness Post-processing** - Applies threshold adjustment
7. 📋 **Fairness Analysis** - Calculates disparate impact, statistical parity, equalized odds
8. 💡 **Explainability** - Generates SHAP plots and natural language explanations
9. 💾 **Save Artifacts** - Saves model, metrics, explanations

### Expected Output:

```
============================================================
Fair Credit Score Prediction Pipeline
============================================================

[STEP 1] Loading and validating data...
Loaded 1000 rows, 13 columns

[STEP 2] Preprocessing data...
Training set: 800 samples
Test set: 200 samples

[STEP 3] Applying fairness pre-processing...
Reweighing applied to training data

[STEP 4] Training model...
Model trained on 800 samples, 15 features

[STEP 5] Generating predictions and evaluating...
Evaluation metrics:
  accuracy: 0.6850
  precision: 0.7234
  recall: 0.7619
  f1: 0.7421

[STEP 6] Applying fairness post-processing...
Threshold adjustment applied: privileged=0.5, unprivileged=0.4

[STEP 7] Calculating fairness metrics...
============================================================
Fairness Metrics (After Adjustment)
============================================================
Disparate Impact Ratio: 0.9824 (target >= 0.80) [✓ PASS]
Statistical Parity Diff: -0.0163 (target < 0.05) [✓ PASS]
Equalized Odds Diff: 0.0357 (target < 0.10) [✓ PASS]
============================================================

[STEP 8] Generating explanations...
Generated 5 explanations in explanations/

[STEP 9] Saving artifacts...
Model saved to models/credit_model.pkl

============================================================
PIPELINE COMPLETE
============================================================
Model Accuracy: 0.6850
Disparate Impact: 0.9824
Statistical Parity Diff: -0.0163
Equalized Odds Diff: 0.0357
Explanations generated: 5
============================================================
```

### Check Generated Files:

```bash
ls -la models/           # credit_model.pkl
ls -la explanations/     # person_X_explanation.txt files
ls -la reports/figures/  # SHAP plots
ls -la outputs/          # metrics.json
```

---

## Start the API Server

### Development Mode (with auto-reload)

```bash
uvicorn src.api:app --reload --host 127.0.0.1 --port 8000
```

### Production Mode

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000 --workers 4
```

### With Environment Variables

```bash
MODEL_PATH=models/credit_model.pkl \
LOG_LEVEL=DEBUG \
uvicorn src.api:app --reload --host 127.0.0.1 --port 8000
```

### API Server Output:

```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Started reloader process [12345]
INFO:     Started server process [12346]
INFO:     Waiting for application startup.
INFO:     Model loaded from models/credit_model.pkl
INFO:     Application startup complete.
```

---

## Test the API

### Interactive Documentation

Open in your browser:
- **Swagger UI**: http://127.0.0.1:8000/docs
- **ReDoc**: http://127.0.0.1:8000/redoc

### Health Check

```bash
curl http://127.0.0.1:8000/health
```

Response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "version": "0.1.0",
  "uptime_seconds": 42.5
}
```

### Get Fairness Configuration

```bash
curl http://127.0.0.1:8000/fairness_metrics
```

Response:
```json
{
  "thresholds": {
    "privileged": 0.5,
    "unprivileged": 0.4
  },
  "targets": {
    "disparate_impact_ratio": 0.8,
    "statistical_parity_diff": 0.05,
    "equalized_odds_diff": 0.1
  },
  "cdi_config": {
    "factors": {
      "Education": "High School",
      "Residence_Type": "Rented",
      "Marital_Status": "Single",
      "Gender": "Female"
    },
    "threshold": 2
  }
}
```

### Make a Prediction

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
    "Education": "High School",
    "Payment_History": "Good",
    "Employment_Status": "Employed",
    "Residence_Type": "Rented",
    "Marital_Status": "Single"
  }'
```

Response:
```json
{
  "creditworthy": true,
  "probability": 0.6823,
  "confidence": "medium",
  "cdi_score": 4,
  "group": "unprivileged",
  "explanation": null
}
```

### Prediction with Explanation

```bash
curl -X POST "http://127.0.0.1:8000/predict?include_explanation=true" \
  -H "Content-Type: application/json" \
  -d '{
    "Income": 75000,
    "Debt": 15000,
    "Loan_Amount": 25000,
    "Loan_Term": 36,
    "Num_Credit_Cards": 3,
    "Gender": "Male",
    "Education": "Bachelor'\''s",
    "Payment_History": "Excellent",
    "Employment_Status": "Employed",
    "Residence_Type": "Owned",
    "Marital_Status": "Married"
  }'
```

Response with explanation:
```json
{
  "creditworthy": true,
  "probability": 0.7845,
  "confidence": "medium",
  "cdi_score": 0,
  "group": "privileged",
  "explanation": {
    "decision": "approved for credit",
    "top_positive_factors": [
      {"feature": "Income", "impact": 0.234},
      {"feature": "Payment_History_Excellent", "impact": 0.156},
      {"feature": "Debt_to_Income", "impact": 0.098}
    ],
    "top_negative_factors": [
      {"feature": "Num_Credit_Cards", "impact": 0.045}
    ]
  }
}
```

### Batch Predictions

```bash
curl -X POST http://127.0.0.1:8000/batch_predict \
  -H "Content-Type: application/json" \
  -d '{
    "records": [
      {
        "Income": 50000,
        "Debt": 20000,
        "Loan_Amount": 15000,
        "Loan_Term": 24,
        "Num_Credit_Cards": 5,
        "Gender": "Male",
        "Education": "High School",
        "Payment_History": "Fair",
        "Employment_Status": "Employed",
        "Residence_Type": "Rented",
        "Marital_Status": "Single"
      },
      {
        "Income": 120000,
        "Debt": 10000,
        "Loan_Amount": 50000,
        "Loan_Term": 60,
        "Num_Credit_Cards": 2,
        "Gender": "Female",
        "Education": "Master'\''s",
        "Payment_History": "Excellent",
        "Employment_Status": "Employed",
        "Residence_Type": "Owned",
        "Marital_Status": "Married"
      }
    ]
  }'
```

### Get Model Info

```bash
curl http://127.0.0.1:8000/model_info
```

---

## Run Tests

### Run All Tests

```bash
pytest -v
```

### Run with Coverage

```bash
pytest --cov=src --cov-report=term-missing -v
```

### Run Specific Test File

```bash
pytest tests/test_fairness.py -v
```

### Run Specific Test

```bash
pytest tests/test_fairness.py::TestFairnessMetrics::test_passes_thresholds_all_pass -v
```

---

## Using the Convenience Scripts

### Quick Run Script

```bash
# Make executable
chmod +x scripts/run_local.sh

# Run everything
./scripts/run_local.sh
```

### Python Runner

```bash
python scripts/run_local.py
```

---

## Project Structure

```
credprediction-opus45/
├── config/                    # Configuration files
│   ├── config.py              # Config dataclasses
│   └── default_config.yaml    # Default settings
├── src/                       # Source code
│   ├── api.py                 # FastAPI server
│   ├── data_loader.py         # Data loading
│   ├── preprocessing.py       # Feature engineering
│   ├── model.py               # Model training
│   ├── fairness.py            # Fairness analysis
│   ├── explainability.py      # SHAP/LIME explanations
│   ├── utils.py               # Utilities
│   └── main.py                # CLI entry point
├── tests/                     # Test suite
├── scripts/                   # Helper scripts
├── data/                      # Data files (gitignored)
├── models/                    # Trained models
├── explanations/              # Generated explanations
├── reports/                   # Reports and figures
└── outputs/                   # Pipeline outputs
```

---

## Common Issues

### Issue: "Module not found" errors

**Solution**: Make sure you're in the project root and venv is activated:

```bash
cd credprediction-opus45
source venv/bin/activate
export PYTHONPATH=$(pwd)
```

### Issue: "Model not loaded" API error

**Solution**: Train the model first:

```bash
python -m src.main run --data-path data/sample_credit_data.csv
```

Then check the model exists:

```bash
ls -la models/credit_model.pkl
```

### Issue: Port 8000 already in use

**Solution**: Use a different port:

```bash
uvicorn src.api:app --reload --host 127.0.0.1 --port 8001
```

Or kill the existing process:

```bash
lsof -i :8000
kill -9 <PID>
```

### Issue: aif360 or fairlearn import errors

**Solution**: These packages can be tricky. Try:

```bash
pip install --upgrade aif360 fairlearn
pip install 'aif360[all]'
```

### Issue: SHAP plots not showing

**Solution**: For headless environments:

```bash
export MPLBACKEND=Agg
```

Plots are saved to `reports/figures/` instead of displayed.

### Issue: Memory errors with large datasets

**Solution**: Process in batches or reduce data size:

```python
# In config/default_config.yaml
explainability:
  num_samples_to_explain: 3  # Reduce from 5
```

### Issue: Tests failing

**Solution**: Check dependencies and run with verbose output:

```bash
pip install -r requirements.txt
pytest -v --tb=long
```

---

## Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `models/credit_model.pkl` | Path to trained model |
| `CONFIG_PATH` | `None` | Path to config YAML |
| `LOG_LEVEL` | `INFO` | Logging level |
| `API_HOST` | `0.0.0.0` | API bind host |
| `API_PORT` | `8000` | API port |
| `API_KEY` | `None` | API key for auth (optional) |
| `CORS_ORIGINS` | `*` | Allowed CORS origins |

---

## New Features & Advanced Usage

The modules have been enhanced with powerful new capabilities:

### 🔧 Preprocessing Enhancements

```python
from src.preprocessing import Preprocessor

preprocessor = Preprocessor()

# Validate data quality before processing
validation_report = preprocessor.validate_data(df)
print(f"Missing values: {validation_report['missing_values']}")
print(f"Duplicate rows: {validation_report['duplicate_rows']}")

# Apply feature scaling (optional)
df_scaled = preprocessor.apply_scaling(df, method='standard')  # or 'robust'

# Save preprocessor state for production inference
preprocessor.save('models/preprocessor.pkl')

# Load and use for new data
preprocessor.load('models/preprocessor.pkl')
new_features = preprocessor.prepare_inference_features(new_data)
```

### 🤖 Model Enhancements

```python
from src.model import CreditModel

model = CreditModel()

# Cross-validation with multiple metrics
cv_results = model.cross_validate(
    X_train, y_train,
    cv=5,
    scoring=['roc_auc', 'f1', 'accuracy']
)
print(f"CV ROC-AUC: {cv_results['roc_auc']['mean']:.4f} ± {cv_results['roc_auc']['std']:.4f}")

# Hyperparameter tuning
tuning_results = model.tune_hyperparameters(
    X_train, y_train,
    search_type='random',  # or 'grid'
    cv=5,
    n_iter=20
)
print(f"Best params: {tuning_results['best_params']}")
print(f"Best CV score: {tuning_results['best_score']:.4f}")

# Model calibration for better probability estimates
model.calibrate_model(X_calib, y_calib, method='isotonic')  # or 'sigmoid'
calibration_data = model.get_calibration_curve(X_test, y_test)
print(f"Brier score: {calibration_data['brier_score']:.4f}")

# Get all feature importance types
all_importances = model.get_all_feature_importances()
print(all_importances['gain'].head(10))  # weight, gain, or cover

# Model versioning is automatic
print(f"Model version: {model.get_version()}")  # timestamp-based
```

### ⚖️ Fairness Enhancements

```python
from src.fairness import FairnessAnalyzer

analyzer = FairnessAnalyzer()

# Calculate comprehensive fairness metrics
metrics = analyzer.calculate_metrics(X_test, y_test, y_pred)
print(f"PPV Parity: {metrics.ppv_parity:.4f}")
print(f"FNR Parity: {metrics.fnr_parity:.4f}")
print(f"FPR Parity: {metrics.fpr_parity:.4f}")
print(f"TPR Privileged: {metrics.tpr_privileged:.4f}")
print(f"TPR Unprivileged: {metrics.tpr_unprivileged:.4f}")

# Generate fairness visualization
plot_path = analyzer.plot_fairness_metrics(metrics)
print(f"Fairness plot saved to: {plot_path}")

# Generate comprehensive fairness report
report_path = analyzer.generate_fairness_report(metrics)
print(f"Fairness report saved to: {report_path}")
```

### 💡 Explainability Enhancements

```python
from src.explainability import Explainer

explainer = Explainer()
explainer.setup_shap_explainer(model)

# Generate waterfall plot for individual prediction
waterfall_path = explainer.generate_waterfall_plot(X_test, index=0)
print(f"Waterfall plot: {waterfall_path}")

# Generate interactive force plot (HTML)
force_path = explainer.generate_force_plot(X_test, index=0)
print(f"Force plot: {force_path}")

# Compare multiple predictions
comparison = explainer.compare_explanations(
    X_test,
    indices=[0, 1, 2, 3, 4],
    predictions=y_pred,
    output_path='reports/comparison.json'
)
print(f"Common positive features: {comparison['common_positive_features']}")
print(f"Common negative features: {comparison['common_negative_features']}")

# Generate interactive HTML report for all samples
report_path = explainer.generate_interactive_report(X_test, y_pred)
print(f"Interactive report: {report_path}")

# Get top features summary
top_features = explainer.get_top_features_summary(X_test, top_n=10)
print(top_features)
```

### 📊 Complete Example with New Features

```python
from src.data_loader import load_data
from src.preprocessing import Preprocessor
from src.model import CreditModel
from src.fairness import FairnessAnalyzer
from src.explainability import Explainer

# 1. Load and validate data
df = load_data('data/credit.csv')
preprocessor = Preprocessor()
validation = preprocessor.validate_data(df)

# 2. Preprocess with saving
X_train, X_test, y_train, y_test, _ = preprocessor.fit_transform(df)
preprocessor.save('models/preprocessor.pkl')

# 3. Train with hyperparameter tuning
model = CreditModel()
tuning_results = model.tune_hyperparameters(
    X_train, y_train, search_type='random', n_iter=30
)

# 4. Cross-validate
cv_results = model.cross_validate(
    X_train, y_train, cv=5, scoring=['roc_auc', 'f1']
)

# 5. Apply fairness preprocessing
analyzer = FairnessAnalyzer()
sample_weights = analyzer.apply_reweighing(X_train, y_train)

# 6. Train final model
model.train(X_train, y_train, sample_weights=sample_weights)

# 7. Calibrate model
split_idx = int(len(X_test) * 0.5)
X_calib, X_eval = X_test[:split_idx], X_test[split_idx:]
y_calib, y_eval = y_test[:split_idx], y_test[split_idx:]
model.calibrate_model(X_calib, y_calib, method='isotonic')

# 8. Evaluate
y_pred = model.predict(X_eval)
y_prob = model.predict_proba(X_eval, use_calibrated=True)

# 9. Fairness analysis with visualization
metrics = analyzer.calculate_metrics(X_eval, y_eval, y_pred)
analyzer.plot_fairness_metrics(metrics)
analyzer.generate_fairness_report(metrics)

# 10. Explainability with advanced plots
explainer = Explainer()
explainer.setup_shap_explainer(model)
explainer.generate_waterfall_plot(X_eval, index=0)
explainer.generate_interactive_report(X_eval, y_pred)
top_features = explainer.get_top_features_summary(X_eval)

# 11. Save everything
model.save('models/credit_model.pkl')
```

### 📈 Generated Outputs

After running with new features, you'll find:

```bash
models/
├── credit_model.pkl           # Model with calibration & metadata
└── preprocessor.pkl           # Preprocessor state for inference

reports/
├── fairness_report.txt        # Comprehensive fairness analysis
├── explainability_report.html # Interactive SHAP visualizations
└── figures/
    ├── fairness_metrics.png   # Fairness comparison plots
    ├── feature_importance_shap.png
    └── feature_importance_detailed.png

explanations/
├── person_0_explanation.txt
├── waterfall_person_0.png     # Individual prediction breakdown
├── force_plot_person_0.html   # Interactive force plot
└── comparison.json            # Multi-sample comparison
```

---

## Next Steps

Once you have everything running locally:

1. 🧪 **Run the test suite**: `pytest -v`
2. 📊 **Experiment with config**: Edit `config/default_config.yaml`
3. 🔍 **Explore the API docs**: http://127.0.0.1:8000/docs
4. 🎯 **Try new features**: Cross-validation, hyperparameter tuning, fairness plots
5. 🚀 **Deploy**: See [DEPLOYMENT.md](DEPLOYMENT.md) for production deployment

---

## Quick Reference

```bash
# Activate environment
source venv/bin/activate

# Train model
python -m src.main run --data-path data/sample_credit_data.csv

# Start API
uvicorn src.api:app --reload --host 127.0.0.1 --port 8000

# Test API
curl http://127.0.0.1:8000/health

# Run tests
pytest -v

# Deactivate environment
deactivate
```
