# Hyperparameter Tuning Guide

This guide explains how to use hyperparameter tuning to optimize your credit prediction model's performance.

## Table of Contents

- [Overview](#overview)
- [When to Use Tuning](#when-to-use-tuning)
- [Search Strategies](#search-strategies)
- [Parameter Grid](#parameter-grid)
- [Usage Examples](#usage-examples)
- [Interpreting Results](#interpreting-results)
- [Best Practices](#best-practices)
- [FAQ](#faq)

## Overview

Hyperparameter tuning is the process of finding the optimal configuration of model parameters that maximizes performance on your specific dataset. Unlike model parameters (which are learned during training), hyperparameters must be set before training begins.

For XGBoost models, key hyperparameters include:
- `max_depth`: Maximum tree depth
- `learning_rate`: Step size shrinkage
- `n_estimators`: Number of boosting rounds
- `min_child_weight`: Minimum sum of instance weight in a child
- `subsample`: Subsample ratio of training instances
- `colsample_bytree`: Subsample ratio of features

## When to Use Tuning

### Use Hyperparameter Tuning When:

- **Initial model development**: Finding a good starting configuration
- **Performance is suboptimal**: Model accuracy is below requirements
- **New dataset**: Working with significantly different data
- **Production deployment**: Squeezing out the last bit of performance
- **Fairness-performance tradeoff**: Balancing accuracy with fairness metrics

### Skip Tuning If:

- **Prototyping**: Quick iteration is more important than optimal performance
- **Resource constraints**: Limited time or computational resources
- **Default parameters work**: Model already meets requirements
- **Small dataset**: Risk of overfitting to validation set

## Search Strategies

### Random Search

**Pros:**
- Faster for large parameter spaces
- More efficient exploration
- Can be stopped early
- Good coverage of parameter space

**Cons:**
- May miss optimal configuration
- Less systematic than grid search

**When to use:** Large parameter spaces, limited time/resources

```bash
python -m src.main tune --data-path data/credit.csv --search-type random --n-iter 50
```

### Grid Search

**Pros:**
- Exhaustive search
- Guaranteed to find best in grid
- Systematic exploration
- Reproducible results

**Cons:**
- Computationally expensive
- Exponential growth with parameters
- Slower than random search

**When to use:** Small parameter space, need certainty, enough compute resources

```bash
python -m src.main tune --data-path data/credit.csv --search-type grid
```

## Parameter Grid

### Default Parameter Grid

The default configuration (`config/default_config.yaml`) includes:

```yaml
tuning:
  search_type: "random"
  cv_folds: 5
  n_iter: 50
  param_grid:
    max_depth: [3, 5, 7, 9]
    learning_rate: [0.01, 0.05, 0.1, 0.2]
    n_estimators: [50, 100, 200]
    min_child_weight: [1, 3, 5]
    subsample: [0.8, 0.9, 1.0]
    colsample_bytree: [0.8, 0.9, 1.0]
```

### Parameter Recommendations

#### `max_depth`
- **Range:** 3-10
- **Default:** 6
- **Tuning tips:**
  - Start with [3, 5, 7, 9]
  - Higher values = more complex models (risk overfitting)
  - Lower values = simpler models (may underfit)
  - For credit data: 5-7 usually works well

#### `learning_rate`
- **Range:** 0.01-0.3
- **Default:** 0.1
- **Tuning tips:**
  - Lower = slower learning, needs more n_estimators
  - Higher = faster learning, risk of overshooting
  - Common values: [0.01, 0.05, 0.1, 0.2]
  - Often pair low learning_rate with high n_estimators

#### `n_estimators`
- **Range:** 50-500
- **Default:** 100
- **Tuning tips:**
  - More trees = better performance (diminishing returns)
  - Watch for overfitting with too many trees
  - Consider early stopping instead of fixed number
  - For credit data: 100-200 usually sufficient

#### `min_child_weight`
- **Range:** 1-10
- **Default:** 1
- **Tuning tips:**
  - Higher values = more conservative (prevents overfitting)
  - Lower values = more aggressive (better fit)
  - For imbalanced data: try higher values [3, 5, 7]

#### `subsample`
- **Range:** 0.5-1.0
- **Default:** 1.0
- **Tuning tips:**
  - Controls row sampling
  - Lower values add randomness (regularization)
  - Common range: [0.7, 0.8, 0.9, 1.0]
  - Values below 0.7 rarely improve performance

#### `colsample_bytree`
- **Range:** 0.5-1.0
- **Default:** 1.0
- **Tuning tips:**
  - Controls feature sampling
  - Lower values add randomness
  - Good for datasets with many features
  - Try [0.7, 0.8, 0.9, 1.0]

### Custom Parameter Grids

#### Quick Tuning (5-10 minutes)
```yaml
param_grid:
  max_depth: [5, 7]
  learning_rate: [0.05, 0.1]
  n_estimators: [100, 200]
```

#### Thorough Tuning (30-60 minutes)
```yaml
param_grid:
  max_depth: [3, 5, 7, 9]
  learning_rate: [0.01, 0.05, 0.1, 0.2]
  n_estimators: [50, 100, 200, 300]
  min_child_weight: [1, 3, 5]
  subsample: [0.7, 0.8, 0.9, 1.0]
  colsample_bytree: [0.7, 0.8, 0.9, 1.0]
```

#### Production-Grade Tuning (2-4 hours)
```yaml
param_grid:
  max_depth: [3, 4, 5, 6, 7, 8, 9]
  learning_rate: [0.01, 0.03, 0.05, 0.1, 0.15, 0.2]
  n_estimators: [50, 100, 150, 200, 300]
  min_child_weight: [1, 2, 3, 5, 7]
  subsample: [0.6, 0.7, 0.8, 0.9, 1.0]
  colsample_bytree: [0.6, 0.7, 0.8, 0.9, 1.0]
  gamma: [0, 0.1, 0.2, 0.3]
  reg_alpha: [0, 0.1, 1]
  reg_lambda: [1, 1.5, 2]
```

## Usage Examples

### CLI Tuning

```bash
# Basic random search
python -m src.main tune \
  --data-path data/credit.csv \
  --search-type random \
  --n-iter 50

# Grid search with custom folds
python -m src.main tune \
  --data-path data/credit.csv \
  --search-type grid \
  --cv-folds 10

# Save results to specific location
python -m src.main tune \
  --data-path data/credit.csv \
  --search-type random \
  --output results/tuning_results.json

# Save best parameters to config
python -m src.main tune \
  --data-path data/credit.csv \
  --search-type random \
  --save-to-config \
  --config-output config/tuned_config.yaml
```

### Using Tuning Script

```bash
# Automated tuning with config output
python scripts/tune_hyperparameters.py \
  --data-path data/credit.csv \
  --search-type random \
  --n-iter 100 \
  --save-to-config

# Then train with tuned parameters
python -m src.main run \
  --data-path data/credit.csv \
  --config config/tuned_config.yaml
```

### Inline Tuning During Training

```bash
# Run full pipeline with tuning
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

config = load_config()
model = CreditModel(config)

# Define parameter grid
param_grid = {
    "max_depth": [3, 5, 7],
    "learning_rate": [0.05, 0.1, 0.2],
    "n_estimators": [100, 200]
}

# Run tuning
best_params, results = model.tune_hyperparameters(
    X_train, y_train,
    search_type="random",
    param_grid=param_grid,
    cv=5,
    n_iter=20
)

print(f"Best parameters: {best_params}")
print(f"Best CV score: {results['best_score']:.4f}")
```

## Interpreting Results

### Understanding Output

```
Best CV Score: 0.8542
Best Parameters:
  max_depth:           7
  learning_rate:       0.05
  n_estimators:        200
  min_child_weight:    3
  subsample:           0.9
  colsample_bytree:    0.8
```

**What this tells you:**
- Model achieves 85.42% accuracy with cross-validation
- Optimal tree depth is 7 levels
- Slower learning rate (0.05) with more trees (200) works best
- Moderate regularization (subsample=0.9, min_child_weight=3)

### Comparing to Baseline

```python
baseline_score = 0.8245
tuned_score = 0.8542
improvement = tuned_score - baseline_score  # +0.0297 or +2.97%
```

### Statistical Significance

Check if improvement is meaningful:
- **Improvement > 1%**: Likely significant
- **Improvement > 3%**: Definitely significant
- **Improvement < 0.5%**: May not be worth the complexity

### Cross-Validation Scores

```
Fold 1: 0.8523
Fold 2: 0.8631
Fold 3: 0.8489
Fold 4: 0.8502
Fold 5: 0.8565
Mean:   0.8542 (+/- 0.0052)
```

**Low standard deviation** (<0.01) = stable, consistent model
**High standard deviation** (>0.03) = inconsistent, may overfit

## Best Practices

### 1. Start Simple
- Begin with small parameter grid
- Add complexity gradually
- Monitor improvement vs. compute time

### 2. Use Appropriate CV Folds
- Small datasets (<1000): 5-10 folds
- Medium datasets (1000-10000): 5 folds
- Large datasets (>10000): 3-5 folds

### 3. Balance Speed vs. Thoroughness
- Development: Random search, 20-50 iterations
- Production: Random search, 100-200 iterations or grid search
- Emergency: Grid search on small grid

### 4. Monitor for Overfitting
- Check train vs. validation scores
- If train score >> validation score = overfitting
- Increase regularization (higher min_child_weight, lower subsample)

### 5. Consider Fairness
- Tune for both accuracy AND fairness
- Create custom scoring function:

```python
def fairness_aware_score(estimator, X, y):
    from sklearn.metrics import accuracy_score
    # Calculate both accuracy and fairness
    y_pred = estimator.predict(X)
    accuracy = accuracy_score(y, y_pred)
    # Add fairness penalty (implement your metric)
    fairness_score = calculate_fairness(X, y, y_pred)
    return 0.7 * accuracy + 0.3 * fairness_score
```

### 6. Save Everything
- Save tuning results
- Save best parameters
- Document why certain parameters work
- Version control config files

### 7. Validate on Hold-Out Set
- Don't make decisions based only on CV scores
- Always test final model on separate test set
- Beware of overfitting to validation set

## FAQ

**Q: How long does tuning take?**
A: Depends on:
- Dataset size: 1000 rows × 50 features ≈ 10-30 minutes for random search (50 iterations)
- Parameter grid size: Grid search can take hours with large grids
- CV folds: More folds = longer time
- Hardware: CPU cores matter

**Q: Can I run tuning in parallel?**
A: Yes! XGBoost uses all CPU cores automatically for training. Multiple CV folds also run in parallel.

**Q: Should I tune separately for each dataset?**
A: Generally yes. However, parameters often transfer well between similar datasets. Start with tuned parameters from similar data.

**Q: What if tuning doesn't improve performance?**
A: Common reasons:
1. Default parameters are already good
2. Need more/different features
3. Data quality issues
4. Wrong metric optimization
5. Insufficient parameter space explored

**Q: How do I tune for fairness AND accuracy?**
A: Use multi-objective optimization or weighted scoring. Currently, tune for accuracy first, then adjust thresholds for fairness post-processing.

**Q: Can I resume interrupted tuning?**
A: Random search: No, but you can combine results from multiple runs
Grid search: Technically yes, but not currently implemented

**Q: Should I re-tune after getting new data?**
A: Only if:
- Significant data distribution changes
- Performance degrades
- Different data characteristics
Otherwise, reuse existing parameters.

## See Also

- [CALIBRATION_GUIDE.md](CALIBRATION_GUIDE.md) - Model calibration
- [PRODUCTION_INFERENCE.md](PRODUCTION_INFERENCE.md) - Production deployment
- [LOCAL_DEVELOPMENT.md](LOCAL_DEVELOPMENT.md) - Local development guide
- [README.md](README.md) - Project overview
