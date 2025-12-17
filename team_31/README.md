# BundleBuddy

A machine learning application that predicts outfit categories based on user weather preferences and past data.

## Project Overview

BundleBuddy uses past weather preference data to recommend outfit categories for users. The model learns from historical data where users indicated which outfits they preferred for different weather conditions.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Launch Jupyter Notebook:
```bash
jupyter notebook
```

3. Open `notebooks/bundlebuddy_ml.ipynb` to start working

## Dataset

The dataset should be a CSV file placed in the `data/` folder.

### Expected CSV Columns:
- **temperature**: Temperature in Fahrenheit or Celsius
- **humidity**: Humidity percentage (0-100)
- **precipitation**: Precipitation level (0-100 or categorical)
- **wind_speed**: Wind speed
- **outfit_category**: The outfit category the user preferred (target variable)

## Project Structure


## Usage

1. Create your dataset CSV file in the `data/` folder
2. Open the Jupyter notebook
3. Update the file path to your dataset
4. Run all cells to train the model and make predictions

