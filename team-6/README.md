# uva-machine-learning-25f-projects team 6 (Dylan Wang uyh9nu, Eddie Xiao pym4ns)
# Diabetes Prediction with Machine Learning

**Team ID:** Team 6  
**Team Members:**  
- Dylan Yilin Wang (uyh9nu)  
- Eddie Xiao (pym4ns)

## Overview
This project applies machine learning to predict diabetes based on health and lifestyle indicators. We implement, compare, and tune multiple models—including Logistic Regression, Random Forest, XGBoost, LightGBM, and a soft-voting ensemble—to achieve a balance of high accuracy, robust generalization, and interpretability.

## Usage
To train the models and reproduce the core results (including performance metrics and visualizations), run:

```bash
python src/ensemble.py
```

This script will:
1. Load and preprocess the dataset (`data/diabetes_dataset.csv`).
2. Train all five models.
3. Evaluate them on the test set and output the performance metrics to the terminal.
4. Generate ROC curves, Precision-Recall curves, confusion matrices, and feature importance plots.

## Setup
If you need to set up the environment, install the required packages from `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Video
A brief demo video walking through the project setup, execution, and key results can be found [here](https://youtu.be/4sVgWTkxrcI). The video demonstrates running the main training script and interpreting the output metrics and visualizations.