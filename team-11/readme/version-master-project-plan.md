# Master Roadmap: Fair Credit Score Prediction Model

This master document fuses the content from `version-a-project-roadmap.md`, `version-b-todo.md`, and `version-c-project-plan.md` into a single source of truth for building a fair, transparent, production-ready credit scoring system.

## Project Snapshot
- **Goal:** Deliver an equitable, explainable creditworthiness model that mitigates discriminatory lending while maintaining strong predictive performance.
- **Current State:** A Colab-exported script (`creditprediction.py`) plus demo decks that showcase CDI-based fairness proxies, AIF360 reweighing, XGBoost classification, and ad-hoc SHAP explanations.
- **Owners:** Pierce Brookins, Aleya Banthavong, and collaborators; requires a cross-functional team (ML, data, backend, frontend, QA, legal/compliance).

## Current State & Gaps
### Completed to Date
- Proof-of-concept notebook with CDI proxy, reweighing, XGBoost, dual-threshold post-processing, and SHAP explanations.
- Initial fairness metrics (disparate impact, statistical parity, equalized odds) and demo collateral (PDF/PPTX).

### Limitations & Critical Issues
- Colab-specific upload path, hardcoded thresholds/seeds, and monolithic script without package structure or configuration.
- No documented dataset schema, validation, or handling for `pd.to_numeric(..., errors="coerce")` NaNs.
- Limited testing, logging, docs, or CI; README is effectively empty.
- Random explainability sampling is non-deterministic; LIME imports unused; fairness monitoring is manual.
- No API, deployment path, governance plan, or alternative data integration.

## Guiding Principles
- **Fairness First:** Make CDI definitions configurable, measure multiple fairness notions, and track trade-offs explicitly.
- **Reproducibility & Transparency:** Version datasets/models, pin dependencies, and keep explanations user-friendly.
- **Modular Architecture:** Separate data, preprocessing, modeling, fairness, explainability, and orchestration layers with configs + CLI.
- **Governance & Compliance:** Document limitations, monitoring, and regulatory alignment (ECOA, FCRA, GDPR).

## Workstreams & Key Deliverables
### 1. Environment & Architecture
- [ ] Replace Colab upload flow with a local/cloud data-loader (CLI/args) and convert notebook export into a modular package (`src/`, `config/`, `scripts/`, etc.).
- [ ] Introduce configs (`config.py`, `config.yaml`) capturing thresholds, seeds, CDI factors, paths, and fairness parameters.
- [ ] Capture dependencies via `requirements.txt` + dev extras, add `.gitignore`, lint/format configs, and optionally a Makefile/task runner.
- [ ] Implement logging utilities, structured exception handling, and unify helpers (`utils.py`) for shared logic.

### 2. Data Management & Governance
- [ ] Document the expected dataset schema, units, categorical domains, and sourcing cadence in README/docs.
- [ ] Build validation pipelines for coercion failures, missing data, outliers, and column presence; log or persist rejected rows.
- [ ] Parameterize CDI construction so analysts can experiment with definitions and thresholds; persist CDI scores + engineered features.
- [ ] Version raw/processed datasets (e.g., via `data/`, DVC) and retain derived artifacts for audits.

### 3. Modeling & Fairness
- [ ] Support multiple models (logistic regression, random forest, gradient boosting) with hyper-parameter search and fairness-aware cross-validation.
- [ ] Ensure reproducible splits stratified on target/protected attributes; persist train/test metadata and model artifacts.
- [ ] Automate fairness pipelines: preprocessing (reweighing), in-processing (ExponentiatedGradient, EqualizedOdds), and post-processing (threshold optimizers) with standardized reporting.
- [ ] Store metrics (accuracy, ROC AUC, group confusion matrices, fairness deltas) as JSON/CSV for downstream dashboards.
- [ ] Explore alternative data sources while guarding against proxy leakage and documenting ethical considerations.

### 4. Explainability & Transparency
- [ ] Solidify SHAP workflows (sampling, feature naming, resource management) and add LIME analyses plus HTML/visual summaries.
- [ ] Convert natural-language explanations into reusable templates and include fairness context (e.g., how CDI-driven features influence outcomes).
- [ ] Save outputs under `explanations/` and `reports/figures`, then surface highlights in README/decks.
- [ ] Plan future enhancements such as counterfactuals, feature interaction analysis, ICE/PDP/Anchors visualizations.

### 5. Testing & Quality Assurance
- [ ] Establish unit tests for CDI calculations, preprocessing pipelines, thresholding logic, fairness metrics, explainability helpers, and CLI/config parsing (target >80% coverage).
- [ ] Build integration tests covering end-to-end flows (data ➜ model ➜ predictions ➜ fairness ➜ explanations) plus serialization/deserialization.
- [ ] Create a synthetic regression test dataset with expected outcomes; add continuous testing via GitHub Actions (lint, type-check, pytest).
- [ ] Seed randomness (e.g., SHAP sampling) and track warnings instead of suppressing them globally.

### 6. Deployment, Monitoring & Operations
- [ ] Develop FastAPI/Flask services with `/predict`, `/batch_predict`, `/explain`, `/fairness_metrics`, and `/health` endpoints; containerize with Docker and plan blue-green/A/B deployments.
- [ ] Implement performance/scalability work (caching, batching, load balancing, auto-scaling) plus model registry + versioning (MLflow or similar).
- [ ] Build monitoring dashboards (backend services + frontend) covering fairness trends, performance metrics, data drift, and explainability snapshots.
- [ ] Define alert thresholds, notification channels, incident response, and automated rollback triggers for fairness violations.
- [ ] Stand up continuous retraining/data pipelines with triggers (schedule, drift, fairness failures) and staged rollouts.

### 7. Documentation, Compliance & Stakeholder Materials
- [ ] Expand README with description, setup, usage, dataset schema, fairness methodology, troubleshooting.
- [ ] Author additional docs: `CONTRIBUTING.md`, `METHODOLOGY.md`, model cards, API docs, runbooks, transparency reports, user guides, and regulatory templates (adverse action notices, audit packs).
- [ ] Update slide deck / PDF assets with refreshed metrics, fairness narratives, and explanation visuals; optionally add interactive demo dashboards.
- [ ] Outline governance policy for periodic audits, data/model versioning, and responsible AI disclosures.

### 8. Advanced Capabilities & Business Enablement
- [ ] Add monitoring dashboards (Plotly, React, or BI) with fairness-performance trade-off visualizations and data drift insights.
- [ ] Implement counterfactual explanation tooling, fairness-aware AutoML/feature selection, and multi-stakeholder fairness optimization.
- [ ] Prepare cost-benefit analyses, stakeholder engagement plans, marketing/comms packages, and training for loan officers.
- [ ] Track ongoing technical debt (dependency bumps, security patches, performance tuning) and schedule quarterly fairness audits.

## Quick Wins (Do First)
1. Remove `google.colab` dependencies, add CLI args for input/output paths, and seed randomness for reproducible explanations.
2. Create `requirements.txt`, `.gitignore`, and populate README with basic overview + usage.
3. Break out `config.py`, `data_loader.py`, `preprocessing.py`, `model.py`, `fairness.py`, `explainability.py`, and `main.py`.
4. Parameterize thresholds (privileged/unprivileged, CDI) and move them into config to eliminate magic numbers.
5. Use or remove unused imports (e.g., implement LIME) and replace `print` statements with structured logging.

## Timeline & Priority Guidance
- **Estimated schedule:** Phases 1-2 (Refactor & Testing) 4–6 weeks; Phases 3-4 (Enhanced Fairness + Alternative Data) 3–4 weeks; Phase 5 (Monitoring) 3–4 weeks; Phase 6 (Deployment) 4–6 weeks; Phase 7 (Continuous Improvement) 2–3 weeks; Phase 8 (Documentation) 2–3 weeks; Phase 9 (Advanced Features) 4–6 weeks; Phase 10 (Business Operations) ongoing. Total ≈5–6 months for core delivery.
- **Priority Matrix:**  
  - **P0:** Architecture refactor, testing foundation, security/compliance, documentation.  
  - **P1:** Enhanced fairness methods, monitoring dashboard, API + deployment, continuous improvement loops.  
  - **P2:** Alternative data, advanced interpretability, business enablement.  
  - **P3:** Marketing/comms extras and experimental fairness research.

## Success Metrics
### Technical
- Test accuracy ≥65%, ROC AUC tracked per group, Disparate Impact Ratio >0.80 (target >0.95), Statistical Parity Difference <0.05 (target <0.02), Equalized Odds Difference <0.10 (target <0.05).
- Code coverage >80%, API latency <500 ms for single predictions, uptime 99.9%.

### Business
- Reduce discriminatory lending by 75%, increase approvals for qualified minority applicants by 15K+, deliver explanations to 100% of applicants, maintain zero regulatory violations.

### Operational
- Zero critical fairness breaches, <1 hour MTTR, quarterly fairness audits, every prediction paired with stored explanations.

## Next Immediate Actions
1. Stand up the project skeleton with configs, dependency files, and CLI-driven local execution.
2. Draft the enriched README plus schema documentation and capture all dependencies.
3. Prioritize unit/integration tests alongside data validation + CDI parameterization to lock in reproducibility.
4. Define fairness monitoring thresholds/reporting so future phases (API, dashboard, retraining) have clear targets.
