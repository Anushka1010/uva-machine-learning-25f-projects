# Project Title: 
Sentiment Analysis of Major News
# Team ID: 
12
# Members: 
Elijah Kim (cys8qu)
# Overview:
This project builds a 3-class sentiment classifier for Reddit text (**-1 = negative, 0 = neutral, 1 = positive**) and then applies it to multiple subreddit datasets to study how sentiment changes around high-engagement “event weeks.”

There are two main stages:
**Model training (model_training.ipynb):** Train and tune two pipelines (Word TF-IDF + LinearSVC, Character TF-IDF + LinearSVC) using grid search and macro-F1, evaluate on a held-out test set, and save the best model to `sentiment_best_model.joblib`.
**Event + sentiment analysis (analysis.ipynb):** Load subreddit CSVs, clean and standardize them, score sentiment using the saved model (with optional transformer fallback on low-confidence cases), detect engagement spikes using weekly comment volume, build event windows (pre/during/post), extract event keywords, generate plots, and export slide-ready CSV outputs.

# Usage:
From the project root, run:
  pip install -r requirements.txt


- Train and save the sentiment model (core ML results)
Run the training notebook/script (the file that trains both Word TF-IDF + LinearSVC and Char TF-IDF + LinearSVC, then chooses the best one).
- Run the subreddit analysis pipeline (core event results)
Run the analysis notebook/script (the file that loads multiple subreddit CSVs, scores sentiment, and detects event spikes).


Core outputs you should see:
  - Cleaning diagnostics (rows dropped, date range, etc.)
  - Sentiment scoring summaries:
      - sentiment class distribution
      - SVM margin summary (confidence proxy)
  - Weekly engagement + spike detection results
  - Event tables:
      - selected spike weeks per subreddit
      - event catalog with top keywords and representative titles
  - Plots:
      - spike score distribution histogram
      - pre/during/post sentiment shift lines for events
      - engagement shift plots for events