# When Does Machine Learning Fail?
### A Broad Stress Test of Common Models

## Team ID and Members
**Team ID:** team-29
**Members:** 
- Max Grant

## Overview

Most machine learning projects focus on maximizing accuracy under ideal assumptions.
However, real-world deployments often involve noisy features, imperfect labels, missing data,
and distribution shifts that standard evaluation does not capture.

This project investigates **when and how machine learning models fail** by systematically
stress-testing three common model families:

- Logistic Regression
- Random Forest
- Multi-Layer Perceptron (Neural Network)

Using the UCI Adult Income dataset, we evaluate model robustness under multiple realistic
failure scenarios, including feature noise, label corruption, distribution shift, and
confidence behavior. The goal is not to identify a single “best” model, but to understand
how different models degrade under stress and what risks they pose in deployment.

---

## Run the Notebook
Launch Jupyter and open the main notebook:
- jupyter notebook src/stress_test.ipynb
Run the notebook cells top to bottom to reproduce all experiments and figures.

---

## Core Results
The notebook produces the following key analyses:
- Baseline performance comparison across models
- Accuracy degradation under feature noise
- Robustness under label corruption
- Train–test gaps under distribution shift
- Model confidence behavior under clean, shifted, and noisy data
All plots and metrics used in the final presentation are generated directly in the notebook.
