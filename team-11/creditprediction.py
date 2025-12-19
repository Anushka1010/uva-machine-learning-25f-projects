# -*- coding: utf-8 -*-
"""CreditPrediction.ipynb

⚠️  DEPRECATED - This file is the original Colab notebook export.

This file is kept for historical reference only. The production-ready code
is in the `src/` directory. Use the proper modules instead:

    from src.main import run_pipeline
    from src.model import CreditModel
    from src.preprocessing import Preprocessor
    from src.fairness import FairnessAnalyzer
    from src.explainability import Explainer

Or use the CLI:
    python -m src.main run --data-path data/credit.csv

Original file is located at:
    https://colab.research.google.com/drive/1Dc8lPew0bgpowX8KC7NqRJdCOgut__ZK

⚠️  DO NOT USE FOR PRODUCTION - Contains Colab-specific imports and hardcoded values!
"""

import warnings

warnings.warn(
    "creditprediction.py is DEPRECATED! Use the modules in src/ instead. "
    "See: python -m src.main run --data-path <path>",
    DeprecationWarning,
    stacklevel=2,
)

# upload dataset into Colab

from google.colab import files
uploaded = files.upload()

# ML & data
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import xgboost as xgb
from xgboost import XGBClassifier

# fairness
from fairlearn.reductions import ExponentiatedGradient, DemographicParity, EqualizedOdds
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from aif360.algorithms.preprocessing import Reweighing

# explainability
import shap
import lime
import lime.lime_tabular

# visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# setup
import os
import warnings
warnings.filterwarnings('ignore')

# load dataset (replace with your CSV)
csv_filename = list(uploaded.keys())[0]
print("Loaded file:", csv_filename)
data = pd.read_csv(csv_filename, dtype=str)  # read as string to inspect messy encodings
data.head()

# preprocessing, feature engineering, and proxy disadvantaged group
# hard to account for systemic racism directly without zipcode or race column, which are already not
# accounted for in tradtional credit score prediction models

# core question: how can we define a disadvantaged group in a fairness study when the same features
# used to define disadvantage are also used to assess credit risk?

# define proxy disadvantaged group using a composite disadvantage index (CDI) based on structural
# demographic factors (education, residence type, gender, marital status) these variables reflect long-term
# socioeconomic barriers without overlapping with core credit-risk features

# convert columns to numeric values
numeric_cols = ['Income', 'Debt', 'Loan_Amount', 'Loan_Term', 'Num_Credit_Cards', 'Credit_Score', 'Creditworthiness']
for col in numeric_cols:
    # coerce converts invalid parsing results to NaN instead of raising an error
    data[col] = pd.to_numeric(data[col], errors='coerce')

# derived features from the dataset
data['Debt_to_Income'] = data['Debt'] / data['Income']
data['Loan_to_Income'] = data['Loan_Amount'] / data['Income']

# create composite disadvantage index (CDI)
data["CDI"] = (
    (data["Education"] == "High School").astype(int) +
    (data["Residence_Type"] == "Rented").astype(int) +
    (data["Marital_Status"] == "Single").astype(int) +
    (data["Gender"] == "Female").astype(int)
)

# define proxy disadvantaged group
data["Proxy_Disadvantaged"] = (data["CDI"] >= 2).astype(int)

# encode categorical variables
print(data.columns.tolist())
categorical_cols = ['Gender', 'Education', 'Payment_History', 'Employment_Status', 'Residence_Type', 'Marital_Status']
for col in categorical_cols:
    if col in data.columns:
        data[col] = data[col].astype(str)  # ensure all categorical columns are strings
data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)
print(data.columns.tolist())

# split features and target
X = data.drop(columns=['Creditworthiness', 'Proxy_Disadvantaged', 'Credit_Score'])
y = data['Creditworthiness'].astype(float)  # ensure numeric for AIF360

# split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ensure all train features are numeric
X_train = X_train.apply(pd.to_numeric, errors='coerce')
X_test = X_test.apply(pd.to_numeric, errors='coerce')

# make sure protected attribute is numeric
X_train['Proxy_Disadvantaged'] = data.loc[X_train.index, 'Proxy_Disadvantaged'].astype(float)
X_test['Proxy_Disadvantaged'] = data.loc[X_test.index, 'Proxy_Disadvantaged'].astype(float)

# convert to AIF360 dataset
train_bld = BinaryLabelDataset(
    df=pd.concat([X_train, y_train], axis=1),
    label_names=['Creditworthiness'],
    protected_attribute_names=['Proxy_Disadvantaged']
)

# reweighing for fairness
RW = Reweighing(
    unprivileged_groups=[{'Proxy_Disadvantaged': 1}],
    privileged_groups=[{'Proxy_Disadvantaged': 0}]
)
train_bld_transf = RW.fit_transform(train_bld)

# access sample weights for model training
sample_weights = train_bld_transf.instance_weights
print("Preprocessing complete. Sample weights ready for fairness-aware training.")

# in-processing -- model training

model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    use_label_encoder=False,
    random_state=42
)

model.fit(
    X_train.drop(columns=['Proxy_Disadvantaged']),
    y_train,
    sample_weight=sample_weights
)

# predictions
y_pred = model.predict(X_test.drop(columns=['Proxy_Disadvantaged']))
y_pred_prob = model.predict_proba(X_test.drop(columns=['Proxy_Disadvantaged']))[:,1]

# accuracy
acc = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {acc:.4f}")

# TN (True Negative): Model predicted 0, actual is 0 → correctly identified as not creditworthy.
# FP (False Positive): Model predicted 1, actual is 0 → incorrectly approved someone who is not creditworthy.
# FN (False Negative): Model predicted 0, actual is 1 → incorrectly denied someone who is creditworthy.
# TP (True Positive): Model predicted 1, actual is 1 → correctly approved a creditworthy person.

# separate test set by group
privileged_idx = X_test['Proxy_Disadvantaged'] == 0
unprivileged_idx = X_test['Proxy_Disadvantaged'] == 1

# true labels and predictions
y_test_priv = y_test[privileged_idx]
y_pred_priv = y_pred[privileged_idx]

y_test_unpriv = y_test[unprivileged_idx]
y_pred_unpriv = y_pred[unprivileged_idx]

# confusion matrices
cm_priv = confusion_matrix(y_test_priv, y_pred_priv)
cm_unpriv = confusion_matrix(y_test_unpriv, y_pred_unpriv)

print("Privileged group confusion matrix:\n", cm_priv)
print("Unprivileged group confusion matrix:\n", cm_unpriv)

# post-processing

# define thresholds
threshold_priv = 0.5   # privileged group
threshold_unpriv = 0.4 # unprivileged group (approve more applicants)

# adjust predictions per group
y_pred_adj = []
for i in range(len(X_test)):
    if X_test['Proxy_Disadvantaged'].iloc[i] == 0:  # privileged
        y_pred_adj.append(int(y_pred_prob[i] >= threshold_priv))
    else:  # unprivileged
        y_pred_adj.append(int(y_pred_prob[i] >= threshold_unpriv))

y_pred_adj = np.array(y_pred_adj)

# privileged group
priv_idx = X_test['Proxy_Disadvantaged'] == 0
cm_priv_adj = confusion_matrix(y_test[priv_idx], y_pred_adj[priv_idx])
print("Privileged group confusion matrix (adjusted):\n", cm_priv_adj)

# unprivileged group
unpriv_idx = X_test['Proxy_Disadvantaged'] == 1
cm_unpriv_adj = confusion_matrix(y_test[unpriv_idx], y_pred_adj[unpriv_idx])
print("Unprivileged group confusion matrix (adjusted):\n", cm_unpriv_adj)

test_bld = BinaryLabelDataset(
    df=pd.concat([X_test, y_test], axis=1),
    label_names=['Creditworthiness'],
    protected_attribute_names=['Proxy_Disadvantaged']
)

# add predictions to dataset
test_bld_pred = test_bld.copy()
test_bld_pred.labels = y_pred.reshape(-1,1)

# fairness metrics
metric_test = ClassificationMetric(
    test_bld,
    test_bld_pred,
    unprivileged_groups=[{'Proxy_Disadvantaged': 1}],
    privileged_groups=[{'Proxy_Disadvantaged': 0}]
)

print("Fairness Metrics\n")
# disparate impact ratio: measures how often the unprivileged group (Proxy_Disadvantaged = 1)
# is approved compared to the privileged group (0)
# target is >0.80; actual value is ~0.9824
print(f"Disparate Impact Ratio: {metric_test.disparate_impact():.4f}")
# demographic parity: equal approval rates across groups.
# ideal = 0; actual value is ~ -0.0163 meaning slightly lower approval for the unprivileged group
print(f"Statistical Parity Difference: {metric_test.statistical_parity_difference():.4f}")
# equalized odds: equal false positive/negative rates
# ideal = 0; actual value is ~ 0.0357
print(f"Equal Odds Difference: {metric_test.equalized_odds_difference():.4f}\n")

# new overall accuracy
acc_adj = accuracy_score(y_test, y_pred_adj)
print(f"Adjusted Test Accuracy: {acc_adj:.4f}")
# small increase from before (0.6775), thus improving FN for unprivileged group did not hurt overall accuracy

os.makedirs("explanations", exist_ok=True)

# prepare data for explanation (drop sensitive label)
X_explain = X_test.drop(columns=["Proxy_Disadvantaged"])

# create SHAP explainer for the creditworthiness model
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_explain)

# global feature importance
plt.figure()
shap.summary_plot(shap_values, X_explain, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig("explanations/feature_importance_simple.png")
plt.close()

# individual decision summaries
def to_plain_english(feature, value, shap_val):
    """Turn SHAP output into a friendly explanation."""
    direction = "helped" if shap_val > 0 else "hurt"
    strength = abs(shap_val)

    if strength > 0.5:
        level = "a strong amount"
    elif strength > 0.1:
        level = "some"
    else:
        level = "a small amount"

    return f"- {feature} (value: {value}) {direction} the decision by {level}."

import random

# choose 5 random indices from X_explain
random_indices = random.sample(range(len(X_explain)), 5)

for i in random_indices:  # explain 5 random samples
    person = X_explain.iloc[i]
    shap_vals = shap_values[i]

    # model predictions
    creditworthiness_pred = y_pred_adj[i]  # 0 = denied, 1 = approved
    creditworthiness_text = "approved for credit" if creditworthiness_pred == 1 else "denied credit"

    summary_lines = [
        f"Explanation for Person #{i}:",
        f"- Model decision: {creditworthiness_text}",
        "Factors influencing this decision:"
    ]

    # rank features by importance for this person
    ranked = sorted(
        zip(X_explain.columns, person.values, shap_vals),
        key=lambda x: abs(x[2]),
        reverse=True
    )

    for feature, value, sv in ranked:
        summary_lines.append(to_plain_english(feature, value, sv))

    # save explanation
    with open(f"explanations/person_{i}_explanation.txt", "w") as f:
        f.write("\n".join(summary_lines))

print("Accessible explainability completed. Check the 'explanations' folder.")