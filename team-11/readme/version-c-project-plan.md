# version-c Credit Prediction Work Plan

This document captures the outstanding work needed to turn the current Colab-style prototype (`creditprediction.py`) and supporting demo decks into a production-ready, reproducible fair-credit modeling project.

## Environment & Repository Hygiene
- [ ] Replace the Colab-only upload block (lines 12-21 in `creditprediction.py`) with a configurable data-loading helper that reads from a local path or cloud bucket and can be driven via CLI/params.
- [ ] Capture all Python dependencies (pandas, numpy, scikit-learn, xgboost, fairlearn, aif360, shap, lime, matplotlib, seaborn, plotly, etc.) in a `requirements.txt` with pinned versions and add environment setup instructions to `README.md`.
- [ ] Convert the notebook-export script into a package/module structure with clear entry points (e.g., `src/`, `main.py`) and move constants + thresholds into a config file.
- [ ] Add `.gitignore`, linting/formatting config, and optionally a lightweight Makefile or task runner for `install`, `train`, `evaluate`, and `explain` targets.

## Data Management
- [ ] Document the dataset schema expected by the pipeline (column names, units, categorical domains) and provide guidance on sourcing/refresh cadence.
- [ ] Add validation/cleaning steps for coercion failures introduced by `pd.to_numeric(..., errors="coerce")` to avoid silently propagating NaNs (e.g., imputation or row filtering with logging).
- [ ] Persist intermediate artifacts (cleaned dataset, engineered features, CDI scores) so they can be inspected outside the notebook runtime.
- [ ] Parameterize the Composite Disadvantage Index (CDI) thresholds so alternate definitions of the proxy disadvantaged group can be experimented with quickly.

## Modeling & Evaluation
- [ ] Implement reproducible data splits (store RNG seeds, optionally stratify on the protected attribute) and log/train metadata for auditability.
- [ ] Extend beyond a single default `XGBClassifier`: add baseline models (logistic regression, random forest) and hyper-parameter search to quantify lift.
- [ ] Persist trained model artifacts, evaluation metrics (accuracy, ROC AUC, confusion matrices), and fairness metrics to disk (JSON/CSV) for traceability.
- [ ] Build a CLI or notebook cells for batch scoring + evaluation on new datasets, ensuring `Proxy_Disadvantaged` handling matches training.

## Fairness Mitigation
- [ ] Justify and document the CDI-based proxy group, including sensitivity analyses comparing alternative definitions or additional features (e.g., zip code if available).
- [ ] Automate fairness evaluation by wrapping `ClassificationMetric` outputs (disparate impact, statistical parity difference, equalized odds) into a reporting helper with thresholds and alerting.
- [ ] Explore in-processing techniques already imported (Fairlearn `ExponentiatedGradient`, `EqualizedOdds`) and compare against the current preprocessing reweighing approach to quantify trade-offs.
- [ ] Investigate post-processing threshold optimization methods (e.g., Fairlearn ThresholdOptimizer) rather than hard-coded `0.5 / 0.4` splits, and document rationale for any overrides.

## Explainability & Transparency
- [ ] Stabilize SHAP computations (sampling strategy, feature naming, handling of high cardinality dummies) to avoid runtime/memory spikes on larger datasets.
- [ ] Add LIME explanations (imports exist but are unused) for a few representative applicants and integrate the outputs into the `explanations/` folder.
- [ ] Convert text explanations into visuals or summary tables that can be embedded in the slide deck or README, highlighting key drivers of approvals/denials.
- [ ] Provide a narrative on how fairness constraints interact with feature importance (e.g., how CDI-related features influence SHAP values) for stakeholder review.

## Visualization & Demo Assets
- [ ] Produce reproducible plots (accuracy vs. fairness trade-offs, group-wise confusion matrices) saved under a `reports/figures` directory and referenced from the deck.
- [ ] Update the PowerPoint/PDF artifacts with refreshed metrics, fairness discussion, and explainability screenshots generated from the latest pipeline run.
- [ ] Create a lightweight dashboard or notebook section using Plotly/Seaborn for interactive exploration during demos.

## Documentation, Testing & Compliance
- [ ] Expand `README.md` with project overview, ethical considerations, setup instructions, sample commands, and troubleshooting tips.
- [ ] Add unit tests for data preprocessing (CDI calculation, feature engineering), fairness metric wrappers, and explanation generation; wire them into CI.
- [ ] Include a model card or responsible AI statement documenting intended use, limitations, and monitoring plans.
- [ ] Define governance steps for refreshing the model (data drift checks, fairness regression tests) and outline how to version datasets/models going forward.
