# MarketMinds – Headlines to Returns

## Team
- **Team ID:** 10
- **Members:** Varun Togaru, Ronit Malhotra, Visvajit Murali

## Overview
MarketMinds builds an end-to-end ML pipeline that takes daily news headlines, aligns them with DJIA market data, and predicts **next-day direction** (up vs. down). It compares:
- **Market-only** features (momentum/volatility baselines)
- **TF-IDF** text features
- **FinBERT** embeddings (finance transformer features)
across multiple models (Logistic Regression, SVM, XGBoost), using **rolling time splits** to avoid look-ahead bias. It also produces a SHAP-based interpretability output and a simple ablation study.

## Usage (Core Results)
1. Open `FinalRunnable.ipynb` in Google Colab.
2. Enable GPU: **Runtime → Change runtime type → T4 GPU**
3. Run everything: **Runtime → Run all**
4. When finished, download:
   - `reports/`
   - `data/features/`

### Dataset location
The notebook expects the headlines CSV at:
- `data/raw/DailyNews/Combined_News_DJIA.csv`

## Outputs
Running the notebook generates these key artifacts:
- `data/raw/headlines_long.parquet` (reshaped headlines)
- `data/raw/djia_prices.parquet` (downloaded DJIA prices)
- `data/processed/model_table.parquet` (aligned + labeled data)
- `data/processed/model_table_clean.parquet` (cleaned text)
- `data/features/finbert_day_embeddings.parquet` (FinBERT embeddings)
- `reports/all_model_results.csv` (all folds, all models)
- `reports/model_comparison_summary.csv` (aggregated comparison table)
- `reports/shap_feature_importance.csv` (interpretability output)
- `reports/ablation_study.csv` (feature-set comparison)

## Video Link
https://youtu.be/iN-jLGnTcNQ
