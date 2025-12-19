# DJ Mixing Recommendation System - Quick Start Guide

## Prerequisites

1. **Python 3.7+** installed on your system
2. **Dataset file** at `data/dataset.csv`

## Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   Or install individually:
   ```bash
   pip install pandas numpy xgboost scikit-learn
   ```

## Running the System

### Basic Usage

The main script is `src/main.py`. Run from the project root directory:

### 1. Search by Track ID (Exact Match)
```bash
python src/main.py --track_id 5SuOikwiRyPMVoIQDJUgSV
```

### 2. Search by Song Name and Artist (Recommended)
```bash
python src/main.py --song "Strobe" --artist "deadmau5"
```

### 3. Search by Song Name Only (Fuzzy Matching)
```bash
python src/main.py --song "Strobe"
```
If multiple matches are found, you'll be prompted to select one.

### 4. Search by Artist Only
```bash
python src/main.py --artist "deadmau5"
```
This will show a list of songs by that artist for you to choose from.

### 5. Search by Dataset Index (For Testing)
```bash
python src/main.py --index 0
```

## Command-Line Options

```
--track_id <id>        Spotify track ID (exact match)
--song <name>          Track name (fuzzy matching)
--artist <name>         Artist name (fuzzy matching)
--index <number>        Dataset row index
--data <path>           Path to dataset CSV (default: data/dataset.csv)
--model_path <path>     Path to hybrid ML model (default: hybrid_model.pkl)
--train_model           Force retrain the hybrid ML model
--no_eval              Skip evaluation metrics
```

## First Run

On the first run, the system will:
1. Load and preprocess the dataset (~114,000 tracks)
2. **Train the hybrid ML model** (takes 2-5 minutes)
3. Save the model to `hybrid_model.pkl` for future use

Subsequent runs will load the pre-trained model (much faster).

## Example Output

```
================================================================================
DJ Mixing Recommendation System
================================================================================

[1/5] Loading and preprocessing dataset...
Dataset loaded in 2.34 seconds

[2/5] Preparing hybrid ML model...
Training hybrid ML model (this may take a few minutes)...
Generating 10000 training samples...
Training XGBoost model...
Model trained and saved to hybrid_model.pkl

[3/5] Finding current song...
Current Song: Strobe by deadmau5 | BPM: 128.0 | Key: 8A | Energy: 0.85

[4/5] Generating recommendations...

[5/5] Results:
================================================================================
Rule-Based Recommendations (Top 10)
================================================================================
1. Song Name
   Artist: Artist Name
   BPM: 128.0 | Key: 8A | Energy: 0.85 | Score: 0.9500
...

Evaluation Metrics
...
Performance Summary
Total Recommendation Time: 0.234 seconds
Target Met: ✓
```

## Troubleshooting

### "No module named 'xgboost'"
Install dependencies: `pip install -r requirements.txt`

### "FileNotFoundError: data/dataset.csv"
Make sure the dataset file exists at `data/dataset.csv`

### Model training takes too long
- The first run trains the model (2-5 minutes)
- Subsequent runs use the saved model (much faster)
- You can reduce training samples in `src/model_hybrid_ml.py` (line with `n_samples=10000`)

### No recommendations found
- Try a different song
- Check if the song exists in the dataset
- Rule-based model requires BPM ±6 and compatible keys, so fewer results

## Performance

- **First run**: ~2-5 minutes (includes model training)
- **Subsequent runs**: < 1 second (uses saved model)
- **Target response time**: < 1 second ✓

## Tips

1. **Use exact artist + song name** for fastest results
2. **Fuzzy matching** works well for partial names or typos
3. **Rule-based model** is strictest (BPM ±6, key compatible)
4. **Audio similarity** ignores BPM/key, finds similar-sounding tracks
5. **Hybrid ML** balances rules with audio features
