# Model Calibration Guide

This guide explains probability calibration for the credit prediction model and how to use it to improve prediction reliability.

## Table of Contents

- [What is Calibration?](#what-is-calibration)
- [When to Use Calibration](#when-to-use-calibration)
- [Calibration Methods](#calibration-methods)
- [Usage Examples](#usage-examples)
- [Interpreting Results](#interpreting-results)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)
- [FAQ](#faq)

## What is Calibration?

**Probability calibration** is the process of adjusting a model's predicted probabilities to match the true probability of the outcome. A well-calibrated model's probabilities reflect reality: if the model predicts 70% probability of creditworthiness for 100 applicants, approximately 70 of them should actually be creditworthy.

### Why It Matters

**Uncalibrated models** might be accurate (good at ranking/classification) but have unreliable probabilities:
- Model says 90% probability → actual outcome is only 70%
- Cannot trust probability values for decision-making
- Problematic for risk assessment and regulatory compliance

**Calibrated models** provide trustworthy probabilities:
- Model says 70% probability → actual outcome is ~70%
- Probabilities can be used for risk-based decisions
- Better for threshold-based decision making
- More interpretable for stakeholders

### Real-World Example

**Uncalibrated Model:**
```
Predicted 80% creditworthy → Actually 65% creditworthy
Predicted 60% creditworthy → Actually 50% creditworthy
Predicted 40% creditworthy → Actually 35% creditworthy
```

**Calibrated Model:**
```
Predicted 80% creditworthy → Actually 79% creditworthy
Predicted 60% creditworthy → Actually 61% creditworthy
Predicted 40% creditworthy → Actually 39% creditworthy
```

## When to Use Calibration

### Use Calibration When:

✅ **Making probability-based decisions**
- Risk assessment based on probability thresholds
- Expected value calculations
- Confidence intervals for decision-making

✅ **Regulatory requirements**
- Need explainable, trustworthy probabilities
- Compliance with fair lending regulations
- Audit trails require probability justification

✅ **Threshold optimization**
- Setting custom thresholds for different groups
- Balancing precision and recall
- Fairness-aware threshold adjustment

✅ **Model ensembling**
- Combining multiple models
- Averaging probabilities
- Stacking/blending approaches

✅ **Cost-sensitive decisions**
- Different costs for false positives vs false negatives
- Expected profit/loss calculations
- Portfolio optimization

### Skip Calibration If:

❌ **Only need rankings**
- Top-K recommendations
- Relative ordering matters, not absolute probability
- AUC is the primary metric

❌ **Simple classification**
- Binary yes/no decisions at fixed threshold (0.5)
- Don't use probabilities for decisions
- Only care about accuracy

❌ **Small dataset**
- Not enough data for reliable calibration
- Risk of overfitting calibration curve
- Better to use uncalibrated probabilities

❌ **Computational constraints**
- Real-time predictions with strict latency requirements
- Calibration adds overhead
- Batch processing is not an option

## Calibration Methods

### Platt Scaling (Sigmoid Calibration)

**How it works:**
- Fits a logistic regression on top of model outputs
- Assumes probabilities need monotonic sigmoid transformation
- Parametric approach (only 2 parameters)

**Pros:**
- Fast and simple
- Works well with small calibration sets
- Good for binary classification
- Preserves ranking order

**Cons:**
- Assumes specific shape (sigmoid)
- May underfit complex calibration curves
- Not as flexible as isotonic regression

**When to use:**
- Small calibration dataset (<1000 samples)
- Probabilities are roughly sigmoid-shaped
- Need simple, fast calibration
- Working with well-behaved models

**Configuration:**
```yaml
calibration:
  enabled: true
  method: sigmoid
```

### Isotonic Regression

**How it works:**
- Learns a piecewise constant, monotonic function
- Non-parametric approach
- Fits step function to calibration data

**Pros:**
- Very flexible
- Can fit complex calibration patterns
- Often more accurate than Platt scaling
- Default recommended method

**Cons:**
- Needs more calibration data (>1000 samples recommended)
- Can overfit with small datasets
- Slightly slower than Platt scaling
- May create discrete jumps in probabilities

**When to use:**
- Large calibration dataset (>1000 samples)
- Complex calibration patterns
- Need highest accuracy
- XGBoost or other tree-based models (default choice)

**Configuration:**
```yaml
calibration:
  enabled: true
  method: isotonic
```

### Choosing Between Methods

| Factor | Platt Scaling | Isotonic Regression |
|--------|---------------|---------------------|
| **Dataset size** | <1000 samples | >1000 samples |
| **Model type** | Logistic Regression, Naive Bayes | XGBoost, Random Forest, SVM |
| **Accuracy** | Good | Excellent |
| **Flexibility** | Low | High |
| **Speed** | Fastest | Fast |
| **Overfitting risk** | Low | Medium |
| **Default choice** | For small data | For XGBoost (our use case) |

**Rule of thumb for credit prediction:**
- Use **isotonic regression** (default) for production
- Use **sigmoid** only if calibration data is limited

## Usage Examples

### CLI Calibration

```bash
# Basic calibration (isotonic)
python -m src.main calibrate \
  --model-path models/credit_model.pkl \
  --data-path data/calibration_data.csv

# Sigmoid calibration
python -m src.main calibrate \
  --model-path models/credit_model.pkl \
  --data-path data/calibration_data.csv \
  --method sigmoid

# Save to specific location
python -m src.main calibrate \
  --model-path models/credit_model.pkl \
  --data-path data/calibration_data.csv \
  --output models/calibrated_model.pkl
```

### Calibration During Training

```bash
# Train with calibration enabled
python -m src.main run \
  --data-path data/credit.csv \
  --calibrate

# Train with tuning and calibration
python -m src.main run \
  --data-path data/credit.csv \
  --tune \
  --calibrate \
  --cv
```

### Python API

```python
from src.model import CreditModel
from config.config import load_config

# Load and train model
config = load_config()
model = CreditModel(config)
model.train(X_train, y_train)

# Calibrate model
model.calibrate_model(X_cal, y_cal, method='isotonic')

# Get calibration metrics
prob_true, prob_pred, brier_score = model.get_calibration_curve(X_test, y_test)
print(f"Brier score: {brier_score:.4f}")  # Lower is better

# Make calibrated predictions
y_prob_calibrated = model.predict_proba(X_test)
```

### Local Development

```bash
# Run with calibration
python scripts/run_local.py --train --calibrate

# All enhancements
python scripts/run_local.py --train --tune --calibrate --cv --validate-data
```

## Interpreting Results

### Calibration Curve

The calibration curve plots predicted probabilities vs actual outcomes:

```
Perfect Calibration (diagonal line):
Predicted 0.2 → Actual 0.2
Predicted 0.5 → Actual 0.5
Predicted 0.8 → Actual 0.8

Uncalibrated (deviates from diagonal):
Predicted 0.2 → Actual 0.3 (underconfident)
Predicted 0.8 → Actual 0.6 (overconfident)
```

**Reading the plot:**
- **On the diagonal** = well-calibrated
- **Above diagonal** = underconfident (predicted probabilities too low)
- **Below diagonal** = overconfident (predicted probabilities too high)

### Brier Score

Brier score measures the mean squared error between predicted probabilities and actual outcomes.

**Formula:** `Brier = (1/N) * Σ(predicted_prob - actual)²`

**Interpretation:**
- **0.0** = perfect predictions
- **0.25** = random predictions (for binary)
- **Lower is better**

**Typical values for credit prediction:**
- **Excellent**: < 0.10
- **Good**: 0.10 - 0.15
- **Acceptable**: 0.15 - 0.20
- **Poor**: > 0.20

**Example output:**
```
Before calibration: Brier score = 0.1842
After calibration:  Brier score = 0.1256
Improvement:        -0.0586 (31.8% better)
```

### Model Comparison

```python
# Compare uncalibrated vs calibrated
uncalibrated_brier = 0.1842
calibrated_brier = 0.1256

improvement = ((uncalibrated_brier - calibrated_brier) /
               uncalibrated_brier * 100)
print(f"Improvement: {improvement:.1f}%")  # 31.8%
```

**What the improvement means:**
- **5-10%**: Minor improvement, calibration might not be necessary
- **10-20%**: Moderate improvement, calibration recommended
- **20-30%**: Significant improvement, calibration highly recommended
- **>30%**: Major improvement, model was poorly calibrated

## Best Practices

### 1. Use Separate Calibration Data

**DON'T:**
```python
# Bad: calibrating on training data
model.train(X_train, y_train)
model.calibrate_model(X_train, y_train)  # ❌ Wrong!
```

**DO:**
```python
# Good: separate calibration set
model.train(X_train, y_train)
model.calibrate_model(X_cal, y_cal)  # ✓ Correct

# Or use test set if no separate calibration set
model.calibrate_model(X_test, y_test)  # ✓ Acceptable
```

### 2. Calibration Set Size

**Minimum sizes:**
- **Sigmoid calibration**: 200-500 samples minimum
- **Isotonic regression**: 1000+ samples recommended

**Splitting strategy:**
```
Total data: 10,000 samples
├── Train: 7,000 (70%)
├── Calibration: 1,500 (15%)
└── Test: 1,500 (15%)
```

### 3. Check Calibration Regularly

**Monitor calibration in production:**
```python
# Periodic calibration check
if months_since_last_calibration > 3:
    prob_true, prob_pred, brier = model.get_calibration_curve(X_recent, y_recent)
    if brier > threshold:
        recalibrate_model()
```

### 4. Preserve Model Performance

**Before calibration:**
```
Accuracy: 0.8523
AUC: 0.9012
Brier: 0.1842
```

**After calibration:**
```
Accuracy: 0.8523  # Should be same
AUC: 0.9012       # Should be same
Brier: 0.1256     # Should improve
```

**Calibration should:**
- ✓ Keep accuracy/AUC the same (±0.001)
- ✓ Improve Brier score
- ✓ Preserve ranking order

### 5. Fairness Considerations

Calibration affects fairness metrics:

```python
# Calibrate then apply fairness thresholds
model.train(X_train, y_train)
model.calibrate_model(X_cal, y_cal)

# Fairness-aware thresholding with calibrated probabilities
y_prob = model.predict_proba(X_test)
y_pred = fairness_analyzer.apply_threshold_adjustment(y_prob, protected_attr)
```

**Calibration + Fairness workflow:**
1. Train model
2. Calibrate probabilities
3. Apply fairness-aware thresholds
4. Verify both calibration and fairness

### 6. Version Control Calibrated Models

```bash
models/
├── credit_model_v1.pkl              # Uncalibrated
├── credit_model_v1_calibrated.pkl   # Calibrated
├── calibration_curve_v1.png         # Diagnostic plot
└── calibration_report_v1.json       # Metrics
```

## Troubleshooting

### Problem: Calibration Makes Model Worse

**Symptoms:**
- Brier score increases after calibration
- Calibration curve is worse

**Causes:**
1. **Too little calibration data**
   - Solution: Use more data or switch to sigmoid method

2. **Data distribution mismatch**
   - Solution: Ensure calibration data is representative

3. **Already well-calibrated**
   - Solution: Skip calibration, use uncalibrated model

### Problem: Calibration Curve is Noisy

**Symptoms:**
- Calibration curve has many wiggles
- Unstable across different calibration sets

**Causes:**
1. **Small calibration set**
   - Solution: Get more calibration data

2. **Isotonic regression overfitting**
   - Solution: Switch to sigmoid calibration

3. **Too many bins in calibration plot**
   - Solution: Use fewer bins (5-10 instead of 20)

### Problem: Different Groups Have Different Calibration

**Symptoms:**
- Model is calibrated overall but not within groups
- Privileged group well-calibrated, unprivileged group not

**Solutions:**
1. **Group-specific calibration:**
```python
# Calibrate separately for each group
model_priv.calibrate_model(X_cal[protected == 0], y_cal[protected == 0])
model_unpriv.calibrate_model(X_cal[protected == 1], y_cal[protected == 1])
```

2. **Stratified calibration:**
```python
# Ensure calibration set has balanced groups
X_cal, y_cal = stratified_split(X, y, protected_attr)
```

### Problem: Production Calibration Drift

**Symptoms:**
- Model was calibrated but probabilities drift over time
- Brier score degrades in production

**Solutions:**
1. **Regular recalibration:**
```python
# Recalibrate monthly on recent data
if time.now() - last_calibration > timedelta(months=1):
    recalibrate_on_recent_data()
```

2. **Monitor calibration metrics:**
```python
# Alert if Brier score > threshold
if brier_score > 0.20:
    alert_team("Calibration drift detected")
```

3. **Online calibration:**
```python
# Update calibration continuously
calibrator.partial_fit(X_batch, y_batch)
```

## FAQ

**Q: Does calibration improve accuracy?**
A: No, calibration doesn't change classification accuracy or ranking (AUC). It only improves probability estimates. Accuracy before/after should be identical.

**Q: Should I calibrate before or after fairness adjustments?**
A: Calibrate first, then apply fairness thresholds. Calibration improves probability quality, then fairness adjustments set appropriate thresholds.

**Q: Can I use cross-validation for calibration?**
A: Yes, but it's complex. Use `cv='prefit'` for already-trained models, or implement custom CV with calibration in each fold.

**Q: How often should I recalibrate in production?**
A: Monitor Brier score monthly. Recalibrate if:
- Brier score degrades by >10%
- Data distribution changes significantly
- Every 3-6 months as a precaution

**Q: Is calibration necessary for XGBoost?**
A: XGBoost often produces poorly calibrated probabilities (tends to be overconfident). Calibration is highly recommended for XGBoost, especially for probability-based decisions.

**Q: What if I don't have enough data for calibration?**
A: Options:
1. Use sigmoid calibration (needs less data)
2. Use cross-validation on training set
3. Skip calibration and acknowledge probability uncertainty
4. Collect more data before production deployment

**Q: Does calibration work with class imbalance?**
A: Yes, calibration works with imbalanced data. In fact, it's even more important because models tend to be poorly calibrated with imbalanced classes.

**Q: Can I calibrate a calibrated model?**
A: Not recommended. Calibrate once on good data. Multiple calibrations can compound errors.

**Q: How do I explain calibration to stakeholders?**
A: "Calibration ensures that when the model says '70% likely to default,' it means 70 out of 100 similar applicants will actually default. This makes the probabilities trustworthy for risk assessment."

## See Also

- [TUNING_GUIDE.md](TUNING_GUIDE.md) - Hyperparameter tuning
- [PRODUCTION_INFERENCE.md](PRODUCTION_INFERENCE.md) - Production deployment
- [LOCAL_DEVELOPMENT.md](LOCAL_DEVELOPMENT.md) - Local development
- [README.md](README.md) - Project overview

## References

- [Predicting Good Probabilities With Supervised Learning (Niculescu-Mizil & Caruana, 2005)](https://www.cs.cornell.edu/~alexn/papers/calibration.icml05.crc.rev3.pdf)
- [Scikit-learn Calibration Guide](https://scikit-learn.org/stable/modules/calibration.html)
- [Trustworthy Online Controlled Experiments: A Practical Guide to A/B Testing](https://www.cambridge.org/core/books/trustworthy-online-controlled-experiments/D97B26382EB0EB2DC2019A7A7B518F59)
