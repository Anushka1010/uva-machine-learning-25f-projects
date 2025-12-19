# Version B - Fair Credit Score Prediction Model TODO 🐶

> Created by PiercePuppy after sniffing through all the code!

---

## 📋 Project Overview

**What is this?** A machine learning model for predicting creditworthiness with fairness constraints. The goal is to prevent algorithmic discrimination against disadvantaged groups while maintaining predictive accuracy.

**Current State:** A Colab notebook (`creditprediction.py`) that:
- Uses XGBoost for credit prediction
- Implements fairness via AIF360 (reweighing) and Fairlearn
- Uses a Composite Disadvantage Index (CDI) as a proxy for protected groups
- Generates SHAP explanations for model decisions

---

## 🚨 Critical Issues to Fix

### 1. Colab Dependencies - MUST REMOVE
- [ ] Remove `from google.colab import files` - this only works in Colab!
- [ ] Replace `files.upload()` with proper file path argument or CLI input
- [ ] Make the script runnable locally without Colab

### 2. Hardcoded Values
- [ ] Move threshold values (`0.5`, `0.4`) to configuration/constants
- [ ] Move `random_state=42` to a config constant
- [ ] Make `test_size=0.2` configurable
- [ ] Externalize the CDI formula components (make them configurable)

### 3. Data Handling Issues
- [ ] `Proxy_Disadvantaged` is added to X_train/X_test AFTER the split - potential data leakage concern
- [ ] The column is dropped during prediction but re-added - this is confusing and error-prone
- [ ] Need proper pipeline to handle this cleanly

---

## 🏗️ Architecture Refactoring

### Split the Monolithic Script (~250 lines → multiple files)

Currently everything is in one file. Break it up into:

- [ ] `config.py` - All constants, thresholds, hyperparameters
- [ ] `data_loader.py` - Data loading and initial validation
- [ ] `preprocessing.py` - Feature engineering, CDI calculation, encoding
- [ ] `fairness.py` - All fairness-related logic (reweighing, metrics, thresholds)
- [ ] `model.py` - Model training, prediction, evaluation
- [ ] `explainability.py` - SHAP/LIME explanation generation
- [ ] `utils.py` - Helper functions like `to_plain_english()`
- [ ] `main.py` - Orchestration script that ties it all together

### Add Proper CLI Interface
- [ ] Use `argparse` or `click` for command-line arguments
- [ ] Arguments needed:
  - `--data-path` - Path to input CSV
  - `--output-dir` - Where to save explanations and results
  - `--threshold-priv` - Threshold for privileged group
  - `--threshold-unpriv` - Threshold for unprivileged group
  - `--test-size` - Train/test split ratio
  - `--random-seed` - For reproducibility

---

## 🧪 Testing Requirements

### Unit Tests Needed
- [ ] Test CDI calculation with known inputs
- [ ] Test `to_plain_english()` function
- [ ] Test threshold adjustment logic
- [ ] Test data preprocessing pipeline
- [ ] Test fairness metric calculations

### Integration Tests
- [ ] End-to-end test with sample data
- [ ] Test that explanations are generated correctly
- [ ] Test model serialization/deserialization

### Test Data
- [ ] Create a small synthetic test dataset for reproducible testing
- [ ] Document expected outputs for the test dataset

---

## 📊 Feature Additions

### Model Improvements
- [ ] Add hyperparameter tuning (GridSearchCV or Optuna)
- [ ] Add cross-validation for more robust evaluation
- [ ] Consider ensemble methods or model comparison
- [ ] Add ROC-AUC curves per group
- [ ] Implement model persistence (save/load trained models)

### Fairness Enhancements
- [ ] Add more fairness metrics:
  - [ ] Predictive parity
  - [ ] Calibration across groups
  - [ ] False discovery rate parity
- [ ] Implement threshold optimization (find optimal thresholds automatically)
- [ ] Add fairness-accuracy tradeoff visualization
- [ ] Consider alternative in-processing methods (adversarial debiasing)

### Explainability Enhancements
- [ ] Actually USE the LIME import (it's imported but never used!)
- [ ] Add LIME explanations alongside SHAP
- [ ] Generate HTML reports instead of just text files
- [ ] Add visualizations for individual predictions
- [ ] Create a summary dashboard

### Data Validation
- [ ] Add schema validation for input data
- [ ] Add missing value handling strategy
- [ ] Add outlier detection/handling
- [ ] Validate expected columns exist

---

## 📝 Documentation Needed

### README.md (Currently Empty!)
- [ ] Project description and motivation
- [ ] Installation instructions
- [ ] Usage examples
- [ ] Dataset requirements/schema
- [ ] Fairness methodology explanation
- [ ] Results interpretation guide

### Code Documentation
- [ ] Add docstrings to all functions
- [ ] Add type hints throughout
- [ ] Document the CDI methodology and rationale
- [ ] Document threshold selection rationale

### Additional Docs
- [ ] `CONTRIBUTING.md` - How to contribute
- [ ] `METHODOLOGY.md` - Detailed explanation of fairness approach
- [ ] API documentation if exposing as a service

---

## 🔧 Code Quality Improvements

### Python Best Practices
- [ ] Add type hints to all functions
- [ ] Add proper logging instead of `print()` statements
- [ ] Use `pathlib.Path` instead of string paths
- [ ] Add proper exception handling
- [ ] Remove unused imports (`lime` is imported but never used)
- [ ] Follow PEP 8 naming conventions consistently

### Dependencies
- [ ] Create `requirements.txt` with pinned versions
- [ ] Create `pyproject.toml` for modern Python packaging
- [ ] Consider using `poetry` or `uv` for dependency management
- [ ] Add development dependencies (pytest, black, ruff, mypy)

### CI/CD
- [ ] Add GitHub Actions workflow for:
  - [ ] Running tests
  - [ ] Linting (ruff/flake8)
  - [ ] Type checking (mypy)
  - [ ] Code formatting (black)
- [ ] Add pre-commit hooks

---

## 🎯 Deployment Considerations

### Model Serving (Future)
- [ ] Create FastAPI/Flask endpoint for predictions
- [ ] Add input validation for API
- [ ] Add explanation endpoint
- [ ] Containerize with Docker

### Monitoring (Future)
- [ ] Track model drift over time
- [ ] Monitor fairness metrics in production
- [ ] Set up alerts for fairness degradation

---

## 🐛 Known Bugs / Code Smells

1. **Warnings Suppressed Globally** - `warnings.filterwarnings('ignore')` hides potentially important warnings
2. **Magic Numbers** - Thresholds and other numbers are hardcoded without explanation
3. **No Error Handling** - Script will crash ungracefully on bad input
4. **Random Sample Selection** - `random.sample()` isn't seeded, so explanations aren't reproducible
5. **Potential Memory Issues** - Loading entire dataset as strings then converting is inefficient
6. **CDI Logic** - Using `Gender == "Female"` as a disadvantage factor is... questionable ethically

---

## 📅 Suggested Implementation Order

### Phase 1: Make It Work Locally
1. Remove Colab dependencies
2. Add CLI arguments for file paths
3. Create `requirements.txt`
4. Update README with basic usage

### Phase 2: Refactor for Quality
1. Split into multiple modules
2. Add type hints and docstrings
3. Add logging
4. Create config module

### Phase 3: Testing
1. Create synthetic test data
2. Write unit tests
3. Write integration tests
4. Set up CI/CD

### Phase 4: Enhance Features
1. Add LIME explanations
2. Add more fairness metrics
3. Implement hyperparameter tuning
4. Create HTML reports

### Phase 5: Production Ready
1. Add API endpoint
2. Dockerize
3. Add monitoring hooks

---

## 💡 Quick Wins (Do These First!)

1. ✨ Remove `google.colab` import and add file path CLI arg
2. ✨ Create `requirements.txt`
3. ✨ Write a real README
4. ✨ Seed the random sample selection for reproducibility
5. ✨ Use the LIME import or remove it (YAGNI!)

---

*Woof! That's a lot of work, but we got this! 🐕*
