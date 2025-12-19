# Credit Card Fraud Detection Analysis
## Team 16: Suhani Malhotra, Samriddhi Kumar, Altaf Syed, Nishitha Khasnavis

Overview:
This project implements supervised learning models to detect fraudulent credit card transactions within the Kaggle Credit Card Fraud dataset.

Two models were evaluated to handle the highly imbalanced data:
* Logistic Regression: Achieved high recall (99.20%) but poor precision (16.54%), resulting in a high number of false positives (444)
* Neural Network: Outperformed the baseline with a significantly higher F1-score (80.83%) and high precision, reducing false positives to only 17.

Usage:
1. Download the 'creditcard.csv' dataset from Kaggle: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
2. Place the csv file in the project directory.
3. Run the 'CreditCardFraudDetection.ipynb' notebook to execute the training and evaluation pipeline.

Setup:
*Ensure that ‘creditcard.csv’ dataset is in same directory as ‘CreditCardFraudDetection.ipynb’

Video:
youtube.com/watch?v=O41fVjY96qk&feature=youtu.be