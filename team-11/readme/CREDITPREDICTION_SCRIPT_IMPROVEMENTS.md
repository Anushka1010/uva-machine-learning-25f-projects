# `creditprediction.py` Improvement Backlog

> The legacy Colab-exported script still lives at `creditprediction.py`. It works for ad-hoc experiments, but each issue below must be addressed before the script could be considered production-ready.

## Environment & I/O
- **Introduce proper logging** – replace `print` statements with `logging` configured for console/file output and log levels.

## Data Validation & Preprocessing

- **Centralize schema validation** – before coercion, assert required numeric/categorical columns are present and report missing/invalid values.
- **Handle coercion errors** – after `pd.to_numeric(..., errors="coerce")`, check the resulting NaNs and either impute or fail fast; currently silent failures propagate.
- **Avoid leakage on splits** – `Proxy_Disadvantaged` is reattached to `X_train`/`X_test` by looking up indices after a split that dropped the column. Instead, compute CDI/proxy values before the split and keep them as dedicated columns to prevent index mismatches.
- **Use preprocessing pipelines** – encode categoricals via `ColumnTransformer`/`OneHotEncoder` and persist fitted transformers for inference parity.

## Modeling & Evaluation

- **Parameterize XGBoost** – expose hyperparameters, seeds, and calibration options; today the classifier is instantiated with hardcoded defaults.
- **Add cross-validation & tuning** – single train/test split can lead to noisy metrics; integrate CV, search strategies, and reproducible seeds.
- **Calibrate probabilities** – current model reports raw XGBoost probabilities; consider Platt scaling/isotonic regression for calibrated outputs.

## Fairness Workflow

- **Abstract fairness utilities** – fairness logic (reweighing, thresholds, metrics) is inline, which makes experimentation brittle. Wrap AIF360/Fairlearn usage in reusable classes that enforce consistent group definitions.
- **Track fairness KPIs** – store disparate impact, stat parity, equalized odds, and confusion matrices for both groups; emit structured reports or JSON for monitoring.
- **Automate threshold tuning** – instead of hardcoding 0.5/0.4, search for thresholds that satisfy target fairness constraints while maximizing utility.

## Explainability

- **Persist explainers** – SHAP/LIME objects should be initialized with training data and saved alongside the model for API reuse.
- **Generate human-readable artifacts** – export feature importance charts, per-sample narratives, and aggregated reports rather than only running SHAP inline.
- **Control randomness** – seed SHAP/LIME sampling to avoid drift between runs.

## Reproducibility & Ops

- **Configuration management** – move all constants (paths, thresholds, CDI factors) into YAML or environment-driven configs.
- **Testing** – there is no unit/integration coverage. Add pytest suites for CDI calculation, preprocessing, protected attribute handling, and fairness math.
- **Packaging & Automation** – wrap the workflow in a CLI (`python -m src.main`) or script (`scripts/run_local.py`), add requirements/pyproject metadata, linting (`black`, `flake8`, `mypy`), and Dockerfiles for consistent execution environments.
- **GUI/UX surface** – if the legacy script must stay usable by non-engineers, build a lightweight GUI (Streamlit/Gradio or a small web front-end) that leverages the refactored modules so people aren’t depending on Colab notebooks.
