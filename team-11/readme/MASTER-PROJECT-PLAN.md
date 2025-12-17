# 🎯 MASTER PROJECT PLAN: Fair Credit Score Prediction Model

> **The Single Source of Truth** - Consolidated from version-a, version-b, and version-c roadmaps
>
> **Authors:** Pierce Brookins and Aleya Banthavong
>
> *Woof! Let's build something fair and awesome! 🐶*

---

## 📋 Executive Summary

**Mission:** Build a fair, transparent, and accurate credit scoring model that reduces systematic bias and promotes transparency in lending decisions.

**Current State:** A Colab notebook (`creditprediction.py`) proof-of-concept that:
- Uses XGBoost for credit prediction
- Implements fairness via AIF360 (reweighing) and Fairlearn
- Uses a Composite Disadvantage Index (CDI) as a proxy for protected groups
- Generates SHAP explanations for model decisions

**Goal:** Transform this PoC into a production-ready, fair lending system that can:
- Reduce discriminatory lending practices by 75%
- Increase approvals for qualified minority applicants by 15,000+
- Provide explainable decisions to 100% of applicants

---

## 🚨 PHASE 0: Critical Fixes (Do These FIRST!)

> **Timeline: 1-2 days** | **Priority: P0 - BLOCKING**

These issues prevent the code from running outside Colab.

### 0.1 Remove Colab Dependencies
- [ ] Remove `from google.colab import files` import
- [ ] Remove `files.upload()` call
- [ ] Replace with CLI argument or config-based file path

### 0.2 Quick Wins
- [ ] Create `requirements.txt` with pinned versions:
  - pandas, numpy, scikit-learn, xgboost
  - fairlearn, aif360
  - shap, lime
  - matplotlib, seaborn, plotly
- [ ] Seed `random.sample()` for reproducible explanations
- [ ] Either USE the LIME import or REMOVE it (YAGNI!)
- [ ] Add basic error handling for file not found

### 0.3 Minimal README
- [ ] Add project description
- [ ] Add installation instructions
- [ ] Add basic usage example

---

## 🏗️ PHASE 1: Code Refactoring & Architecture

> **Timeline: 2-3 weeks** | **Priority: P0**

### 1.1 Project Structure Setup

Create proper Python package structure:

```
credprediction/
├── src/
│   ├── __init__.py
│   ├── config.py           # Constants, thresholds, hyperparameters
│   ├── data_loader.py      # Data loading, validation, schema checking
│   ├── preprocessing.py    # Feature engineering, CDI calculation, encoding
│   ├── fairness.py         # Reweighing, metrics, threshold optimization
│   ├── model.py            # Training, prediction, evaluation
│   ├── explainability.py   # SHAP/LIME explanation generation
│   └── utils.py            # Logging, helpers, visualization
├── tests/
│   ├── __init__.py
│   ├── test_preprocessing.py
│   ├── test_fairness.py
│   ├── test_model.py
│   └── test_explainability.py
├── data/                   # .gitignored - for local datasets
├── models/                 # Saved model artifacts
├── explanations/           # Generated explanations
├── reports/
│   └── figures/            # Reproducible plots
├── notebooks/              # Exploratory analysis
├── config/
│   ├── config.yaml
│   └── fairness_config.yaml
├── main.py                 # CLI entry point
├── requirements.txt
├── requirements-dev.txt
├── pyproject.toml
├── .gitignore
└── README.md
```

### 1.2 Module Breakdown

#### `config.py`
- [ ] Model hyperparameters (random_state, n_estimators, etc.)
- [ ] Fairness thresholds (privileged: 0.5, unprivileged: 0.4)
- [ ] CDI formula components (configurable disadvantage factors)
- [ ] Train/test split ratio
- [ ] File paths and output directories

#### `data_loader.py`
- [ ] Replace Colab upload with configurable data loading
- [ ] Support CSV, Parquet, and database sources
- [ ] Schema validation (expected columns, types)
- [ ] Handle coercion failures from `pd.to_numeric()` properly
- [ ] Data versioning tracking
- [ ] Logging for validation issues

#### `preprocessing.py`
- [ ] `CompositeDisadvantageIndexBuilder` class
- [ ] `FeatureEngineer` class for derived features (Debt_to_Income, Loan_to_Income)
- [ ] Categorical encoding pipeline
- [ ] sklearn Pipeline integration
- [ ] Persist intermediate artifacts (cleaned data, features, CDI scores)

#### `fairness.py`
- [ ] `FairnessPreprocessor` class (Reweighing wrapper)
- [ ] `FairnessEvaluator` class for metrics calculation
- [ ] `ThresholdOptimizer` class for post-processing
- [ ] Support multiple fairness definitions
- [ ] Automated fairness reporting with thresholds/alerting

#### `model.py`
- [ ] `FairCreditModel` wrapper class
- [ ] Support multiple model types (XGBoost, Random Forest, LightGBM, Logistic Regression)
- [ ] Model serialization (save/load)
- [ ] Hyperparameter tuning with fairness constraints
- [ ] Cross-validation with fairness metrics

#### `explainability.py`
- [ ] `ExplainabilityEngine` class
- [ ] SHAP explanations (already working)
- [ ] LIME explanations (currently unused!)
- [ ] `to_plain_english()` helper function
- [ ] Batch explanation support
- [ ] HTML report generation

#### `utils.py`
- [ ] Logging utilities (replace print statements)
- [ ] `pathlib.Path` for all file operations
- [ ] Visualization helpers
- [ ] Metrics calculation helpers

### 1.3 CLI Interface

Using `argparse` or `click`:

```bash
python main.py train --data-path ./data/credit.csv --output-dir ./models
python main.py predict --model-path ./models/model.pkl --data-path ./data/new.csv
python main.py explain --model-path ./models/model.pkl --sample-id 42
python main.py evaluate --model-path ./models/model.pkl --data-path ./data/test.csv
```

Arguments:
- [ ] `--data-path` - Path to input CSV
- [ ] `--output-dir` - Where to save results
- [ ] `--model-path` - Path to saved model
- [ ] `--threshold-priv` - Threshold for privileged group
- [ ] `--threshold-unpriv` - Threshold for unprivileged group
- [ ] `--test-size` - Train/test split ratio
- [ ] `--random-seed` - For reproducibility
- [ ] `--config` - Path to config file

### 1.4 Fix Data Handling Issues
- [ ] Fix `Proxy_Disadvantaged` being added AFTER split (potential data leakage)
- [ ] Create clean pipeline that handles protected attribute properly
- [ ] Ensure `Proxy_Disadvantaged` handling matches between training and scoring

---

## 🧪 PHASE 2: Testing & Validation

> **Timeline: 2 weeks** | **Priority: P0**

### 2.1 Unit Tests

- [ ] **Data Loading Tests**
  - [ ] Valid CSV loading
  - [ ] Missing columns handling
  - [ ] Invalid data type handling
  - [ ] Schema validation

- [ ] **Preprocessing Tests**
  - [ ] CDI calculation with known inputs
  - [ ] Feature engineering (Debt_to_Income, Loan_to_Income)
  - [ ] Categorical encoding
  - [ ] Missing value handling

- [ ] **Fairness Tests**
  - [ ] Disparate impact ratio calculation
  - [ ] Statistical parity measurement
  - [ ] Equalized odds measurement
  - [ ] Reweighing sample weights
  - [ ] Threshold adjustment logic

- [ ] **Model Tests**
  - [ ] Training completes without error
  - [ ] Predictions are in expected range
  - [ ] Model serialization/deserialization

- [ ] **Explainability Tests**
  - [ ] `to_plain_english()` function
  - [ ] SHAP values generated correctly
  - [ ] Explanation files created

### 2.2 Integration Tests

- [ ] End-to-end pipeline: data → model → predictions → explanations
- [ ] Fairness pipeline: preprocessing → training → evaluation
- [ ] Different data formats and sources
- [ ] CLI commands work correctly

### 2.3 Test Data

- [ ] Create small synthetic test dataset (~100 rows)
- [ ] Document expected outputs for deterministic testing
- [ ] Include edge cases (missing values, outliers)

### 2.4 CI/CD Setup

- [ ] GitHub Actions workflow:
  - [ ] Run pytest on push/PR
  - [ ] Linting with ruff
  - [ ] Type checking with mypy
  - [ ] Code formatting check with black
- [ ] Pre-commit hooks configuration
- [ ] Code coverage reporting (target: >80%)

---

## ⚖️ PHASE 3: Enhanced Fairness Implementation

> **Timeline: 2-3 weeks** | **Priority: P1**

### 3.1 Additional Fairness Metrics

- [ ] **Calibration Metrics**
  - [ ] Equal calibration across groups
  - [ ] Calibration curves visualization

- [ ] **Predictive Parity**
  - [ ] Equal positive predictive value across groups

- [ ] **Equal Opportunity**
  - [ ] Equal true positive rates

- [ ] **False Discovery Rate Parity**

- [ ] **Individual Fairness**
  - [ ] Distance metric definition
  - [ ] Similar individuals → similar outcomes

### 3.2 Advanced Bias Mitigation

- [ ] **Explore Fairlearn In-Processing** (already imported!)
  - [ ] `ExponentiatedGradient` with `DemographicParity`
  - [ ] `ExponentiatedGradient` with `EqualizedOdds`
  - [ ] Compare against current reweighing approach

- [ ] **Threshold Optimization**
  - [ ] Replace hardcoded 0.5/0.4 thresholds
  - [ ] Use Fairlearn `ThresholdOptimizer`
  - [ ] Document rationale for any manual overrides

- [ ] **Adversarial Debiasing** (future)
  - [ ] Train adversarial network
  - [ ] Remove protected attribute predictability

### 3.3 CDI Improvements

- [ ] **Document CDI Methodology**
  - [ ] Justify proxy group definition
  - [ ] Ethical considerations (Gender as disadvantage factor?)
  
- [ ] **Sensitivity Analysis**
  - [ ] Compare alternative CDI definitions
  - [ ] Test with different threshold (currently >=2)
  - [ ] Explore additional features if available

- [ ] **Parameterize CDI**
  - [ ] Make components configurable
  - [ ] Allow experimentation with different formulas

### 3.4 Fairness-Accuracy Tradeoff

- [ ] Visualize tradeoff curves
- [ ] Pareto frontier analysis
- [ ] Stakeholder preference elicitation

---

## 🔍 PHASE 4: Explainability Enhancements

> **Timeline: 2 weeks** | **Priority: P1**

### 4.1 SHAP Improvements

- [ ] Stabilize computations (sampling, memory)
- [ ] Handle high cardinality dummies
- [ ] Global feature importance plot (already working)
- [ ] Individual waterfall plots
- [ ] Dependence plots for key features

### 4.2 LIME Implementation

- [ ] Actually USE the lime import!
- [ ] Generate LIME explanations for representative samples
- [ ] Compare SHAP vs LIME explanations
- [ ] Document when to use each

### 4.3 Report Generation

- [ ] HTML reports instead of text files
- [ ] Include visualizations
- [ ] Summary statistics
- [ ] Embed in slide deck / README

### 4.4 Fairness-Explainability Integration

- [ ] Document how CDI features influence SHAP values
- [ ] Narrative on fairness constraints impact
- [ ] Stakeholder-friendly explanations

### 4.5 Counterfactual Explanations (Future)

- [ ] "What if" scenarios
- [ ] Actionable recommendations
- [ ] Validate recommendations are fair

---

## 📊 PHASE 5: Alternative Data Integration

> **Timeline: 3-4 weeks** | **Priority: P2**

### 5.1 Research Data Sources

- [ ] Rent payment history APIs
- [ ] Utility payment history APIs
- [ ] Telecom payment data
- [ ] Bank account transaction patterns
- [ ] Employment verification data

### 5.2 Data Integration

- [ ] Create ingestion pipelines
- [ ] Data quality checks
- [ ] Handle missing alternative data gracefully
- [ ] Privacy and compliance review

### 5.3 Feature Engineering

- [ ] Features from rent payments
- [ ] Features from utility payments
- [ ] Features from transactions
- [ ] Validate predictive power
- [ ] Monitor for new bias sources

### 5.4 Regional Adjustments

- [ ] Integrate cost-of-living data
- [ ] Create adjustment factors
- [ ] Apply to income/debt features
- [ ] Validate fairness impact

---

## 📈 PHASE 6: Monitoring Dashboard

> **Timeline: 3-4 weeks** | **Priority: P1**

### 6.1 Dashboard Backend

- [ ] Choose framework (Streamlit recommended for MVP)
- [ ] FastAPI backend for metrics API
- [ ] Real-time metrics calculation
- [ ] Database for predictions and metrics

### 6.2 Dashboard Frontend

- [ ] **Fairness Metrics Visualization**
  - [ ] Disparate impact ratio over time
  - [ ] Statistical parity trends
  - [ ] Equalized odds tracking
  - [ ] Group-wise confusion matrices

- [ ] **Model Performance**
  - [ ] Accuracy, precision, recall over time
  - [ ] ROC curves per group
  - [ ] Calibration plots

- [ ] **Data Drift Detection**
  - [ ] Feature distribution changes
  - [ ] Label distribution changes
  - [ ] Alerts for significant drift

- [ ] **Explainability Dashboard**
  - [ ] Global feature importance
  - [ ] Sample explanations viewer

### 6.3 Alerting System

- [ ] Define fairness metric thresholds
- [ ] Alert triggers
- [ ] Email/Slack notifications
- [ ] Incident response procedures

---

## 🚀 PHASE 7: Production Deployment

> **Timeline: 4-6 weeks** | **Priority: P1**

### 7.1 API Development

- [ ] FastAPI REST API
  - [ ] `POST /predict` - Single prediction
  - [ ] `POST /batch_predict` - Bulk predictions
  - [ ] `GET /explain/{id}` - Get explanation
  - [ ] `GET /fairness_metrics` - Current metrics
  - [ ] `GET /health` - Health check
- [ ] Input validation (Pydantic models)
- [ ] Rate limiting and authentication
- [ ] OpenAPI/Swagger documentation
- [ ] API versioning

### 7.2 Infrastructure

- [ ] Dockerize application
- [ ] docker-compose for local dev
- [ ] Model registry (MLflow)
- [ ] Model versioning
- [ ] Blue-green deployment

### 7.3 Scalability

- [ ] Batch prediction optimization
- [ ] Caching for frequent predictions
- [ ] Load balancing
- [ ] Auto-scaling

### 7.4 Security & Compliance

- [ ] Encryption at rest and in transit
- [ ] Audit logging for all predictions
- [ ] GDPR compliance
- [ ] ECOA (Equal Credit Opportunity Act) compliance
- [ ] FCRA (Fair Credit Reporting Act) compliance
- [ ] Security scanning
- [ ] Penetration testing

---

## 🔄 PHASE 8: Continuous Improvement

> **Timeline: Ongoing** | **Priority: P1**

### 8.1 Model Retraining Pipeline

- [ ] Automated retraining pipeline
- [ ] Triggers:
  - [ ] Scheduled (monthly/quarterly)
  - [ ] Performance degradation
  - [ ] Fairness metric violations
- [ ] Fairness validation before deployment
- [ ] Staged rollout

### 8.2 Data Pipeline

- [ ] Automated data collection/validation
- [ ] Data quality monitoring
- [ ] Data versioning (DVC)
- [ ] Feature store

### 8.3 Feedback Loop

- [ ] Collect prediction outcomes
- [ ] Analyze actual vs predicted
- [ ] Identify production fairness issues
- [ ] Regular fairness audits (quarterly)

---

## 📝 PHASE 9: Documentation & Compliance

> **Timeline: 2-3 weeks** | **Priority: P0**

### 9.1 Technical Documentation

- [ ] Architecture documentation
- [ ] API documentation
- [ ] **Model Card**
  - [ ] Intended use
  - [ ] Training data description
  - [ ] Fairness metrics
  - [ ] Limitations and biases
  - [ ] Monitoring plans
- [ ] Deployment guide
- [ ] Operations runbook

### 9.2 User Documentation

- [ ] User guide for understanding decisions
- [ ] Fairness approach explanation
- [ ] FAQ for applicants
- [ ] Transparency report

### 9.3 Compliance Documentation

- [ ] Fair lending compliance docs
- [ ] Adverse action notice templates
- [ ] Regulatory reporting templates
- [ ] Third-party audit preparation
- [ ] Risk assessment

---

## 🐛 Known Issues & Technical Debt

### Code Smells

1. **Warnings Suppressed Globally** - `warnings.filterwarnings('ignore')` hides important warnings
2. **Magic Numbers** - Thresholds hardcoded without explanation
3. **No Error Handling** - Script crashes ungracefully on bad input
4. **Unseeded Randomness** - `random.sample()` not seeded, explanations not reproducible
5. **Memory Inefficiency** - Loading dataset as strings then converting
6. **Unused Import** - `lime` imported but never used
7. **Ethical Concern** - Using `Gender == "Female"` as disadvantage factor needs review

### Technical Debt Backlog

- [ ] Replace print() with proper logging
- [ ] Add type hints throughout
- [ ] Use pathlib.Path consistently
- [ ] Add proper exception handling
- [ ] Follow PEP 8 naming conventions
- [ ] Regular dependency updates
- [ ] Security patches

---

## 📊 Success Metrics

### Technical Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Test Accuracy | ~67.75% | >65% ✅ |
| Disparate Impact Ratio | ~0.98 | >0.80 (ideal >0.95) ✅ |
| Statistical Parity Diff | ~-0.016 | <0.05 (ideal <0.02) ✅ |
| Equal Odds Difference | ~0.036 | <0.10 (ideal <0.05) ✅ |
| Code Coverage | 0% | >80% |
| API Latency | N/A | <500ms |
| Uptime | N/A | 99.9% |

### Business Impact Metrics

- [ ] Reduce discriminatory lending practices by 75%
- [ ] Increase approvals for qualified minority applicants by 15,000+
- [ ] Provide explainable decisions to 100% of applicants
- [ ] Zero regulatory violations related to fair lending

---

## ⏱️ Timeline Summary

| Phase | Duration | Priority |
|-------|----------|----------|
| Phase 0: Critical Fixes | 1-2 days | P0 |
| Phase 1: Refactoring | 2-3 weeks | P0 |
| Phase 2: Testing | 2 weeks | P0 |
| Phase 3: Enhanced Fairness | 2-3 weeks | P1 |
| Phase 4: Explainability | 2 weeks | P1 |
| Phase 5: Alternative Data | 3-4 weeks | P2 |
| Phase 6: Dashboard | 3-4 weeks | P1 |
| Phase 7: Production | 4-6 weeks | P1 |
| Phase 8: Continuous Improvement | Ongoing | P1 |
| Phase 9: Documentation | 2-3 weeks | P0 |

**Total: ~5-6 months for core implementation**

---

## 🎯 Priority Matrix

### P0 - Critical (Do First)
- Phase 0: Remove Colab dependencies
- Phase 1: Code refactoring
- Phase 2: Testing
- Phase 9: Documentation

### P1 - High Priority
- Phase 3: Enhanced fairness
- Phase 4: Explainability
- Phase 6: Dashboard
- Phase 7: Production deployment
- Phase 8: Continuous improvement

### P2 - Medium Priority
- Phase 5: Alternative data integration
- Advanced interpretability features

### P3 - Nice to Have
- Multi-stakeholder fairness optimization
- Fairness-aware AutoML
- Marketing and communication

---

## 👥 Resources Needed

### Team
- ML Engineers (2-3)
- Backend Engineers (1-2)
- Frontend Engineer (1) - for dashboard
- Data Engineer (1)
- Legal/Compliance Advisor (1)
- QA Engineer (1)

### Technical Stack
- Python 3.8+
- Cloud infrastructure (AWS/GCP/Azure)
- CI/CD (GitHub Actions)
- Monitoring (Prometheus, Grafana)
- MLOps (MLflow)

---

## ⚠️ Risk Management

| Risk | Impact | Mitigation |
|------|--------|------------|
| Fairness-Accuracy Tradeoff | High | Careful threshold tuning, alternative data |
| Data Quality Issues | High | Robust validation, monitoring |
| Regulatory Changes | Medium | Regular compliance reviews, flexible architecture |
| Proxy Variable Leakage | High | Thorough fairness testing for new features |
| Model Drift | Medium | Continuous monitoring, automated retraining |

---

## 🚀 Next Immediate Steps

1. **TODAY**: Remove Colab dependencies, add CLI file path arg
2. **TODAY**: Create `requirements.txt`
3. **THIS WEEK**: Set up project structure
4. **THIS WEEK**: Write real README
5. **NEXT WEEK**: Start modular refactoring
6. **NEXT WEEK**: Write first unit tests

---

*Woof! This is a big project, but we've got a solid plan. Let's fetch some code! 🐕*

---

**Document Version:** 1.0
**Last Updated:** $(date)
**Consolidated from:** version-a-project-roadmap.md, version-b-todo.md, version-c-project-plan.md
