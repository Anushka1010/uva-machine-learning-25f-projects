# Project Structure

This document describes the organization of the DJ Mixing Recommendation System project.

## Directory Structure

```
team-x/
├── src/                          # All source code
│   ├── __init__.py              # Package initialization
│   ├── main.py                  # Main execution script
│   ├── data_preprocessing.py    # Data loading and preprocessing
│   ├── model_rule_based.py      # Rule-based recommendation model
│   ├── model_audio_similarity.py # Audio similarity baseline model
│   ├── model_hybrid_ml.py        # Hybrid ML model (XGBoost)
│   ├── evaluation.py             # Evaluation metrics and comparison
│   ├── utils.py                  # Utility functions
│   └── visualize_results.py     # Visualization script
├── doc/                          # Documentation
│   ├── QUICKSTART.md            # Quick start guide
│   └── PROJECT_STRUCTURE.md     # This file
├── data/                         # Dataset
│   └── dataset.csv              # Spotify dataset (114,000+ tracks)
├── plots/                        # Generated visualizations (created at runtime)
├── requirements.txt              # Python dependencies
├── hybrid_model.pkl              # Trained ML model (generated)
└── README.md                     # Project overview and usage

```

## File Descriptions

### Source Code (`src/`)

- **main.py**: Main entry point. Handles command-line arguments, loads data, runs all three models, and displays results.
- **data_preprocessing.py**: Loads dataset, converts keys to Camelot notation, normalizes features.
- **model_rule_based.py**: Rule-based recommendation using BPM ±6, key compatibility, and energy flow.
- **model_audio_similarity.py**: Baseline model using cosine similarity on audio features.
- **model_hybrid_ml.py**: XGBoost model combining DJ rules with audio features.
- **evaluation.py**: Calculates and displays evaluation metrics for all models.
- **utils.py**: Helper functions for Camelot conversion, key compatibility, and song search.
- **visualize_results.py**: Generates descriptive graphs for data exploration and results.

### Documentation (`doc/`)

- **QUICKSTART.md**: Detailed usage instructions, examples, and troubleshooting.
- **PROJECT_STRUCTURE.md**: This file - describes project organization.

### Data (`data/`)

- **dataset.csv**: Spotify dataset with ~114,000 tracks containing:
  - BPM (tempo)
  - Musical key and mode
  - Audio features (energy, valence, danceability, etc.)
  - Track metadata (name, artist, genre)

## Running the System

All commands should be run from the project root directory:

```bash
# Main script
python src/main.py --song "Strobe" --artist "deadmau5"

# Generate visualizations
python src/visualize_results.py
```

## Path Handling

All file paths are handled relative to the project root:
- Dataset: `data/dataset.csv`
- Model: `hybrid_model.pkl` (in project root)
- Plots: `plots/` (created at runtime)

The code automatically resolves paths correctly whether run from project root or src/ directory.
