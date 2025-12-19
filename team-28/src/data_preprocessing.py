"""
Data preprocessing for DJ Mixing Recommendation System.
Loads dataset, converts keys to Camelot notation, and prepares features.
"""

import pandas as pd
import numpy as np
from utils import key_to_camelot, get_compatible_keys


def load_dataset(filepath=None):
    """
    Load the Spotify dataset from CSV file.
    
    Args:
        filepath: Path to the dataset CSV file (default: ../data/dataset.csv)
    
    Returns:
        DataFrame with loaded data
    """
    if filepath is None:
        from pathlib import Path
        base_path = Path(__file__).parent.parent
        filepath = str(base_path / 'data' / 'dataset.csv')
    print(f"Loading dataset from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} tracks")
    return df


def convert_to_camelot(dataset):
    """
    Convert Spotify key and mode to Camelot Wheel notation.
    
    Args:
        dataset: DataFrame with 'key' and 'mode' columns
    
    Returns:
        DataFrame with added 'camelot_key' column
    """
    print("Converting keys to Camelot Wheel notation...")
    
    # Handle missing values
    dataset['key'] = pd.to_numeric(dataset['key'], errors='coerce').fillna(0).astype(int)
    dataset['mode'] = pd.to_numeric(dataset['mode'], errors='coerce').fillna(0).astype(int)
    
    # Convert to Camelot notation
    dataset['camelot_key'] = dataset.apply(
        lambda row: key_to_camelot(row['key'], row['mode']),
        axis=1
    )
    
    print(f"Converted {len(dataset[dataset['camelot_key'].notna()])} keys to Camelot notation")
    return dataset


def add_compatible_keys(dataset):
    """
    Add compatible keys list for each track.
    
    Args:
        dataset: DataFrame with 'camelot_key' column
    
    Returns:
        DataFrame with added 'compatible_keys' column (list of compatible Camelot keys)
    """
    print("Calculating compatible keys...")
    
    dataset['compatible_keys'] = dataset['camelot_key'].apply(
        lambda x: get_compatible_keys(x) if pd.notna(x) else []
    )
    
    return dataset


def normalize_audio_features(dataset):
    """
    Normalize audio features for similarity calculations.
    Store original values and add normalized versions.
    
    Args:
        dataset: DataFrame with audio features
    
    Returns:
        DataFrame with normalized features
    """
    print("Normalizing audio features...")
    
    # Features to normalize
    audio_features = ['energy', 'valence', 'danceability', 'acousticness', 
                     'instrumentalness', 'loudness', 'speechiness', 'liveness']
    
    # Normalize loudness (typically negative values, normalize to 0-1 range)
    if 'loudness' in dataset.columns:
        # Loudness is typically in range -60 to 0, normalize to 0-1
        min_loudness = dataset['loudness'].min()
        max_loudness = dataset['loudness'].max()
        if max_loudness > min_loudness:
            dataset['loudness_normalized'] = (dataset['loudness'] - min_loudness) / (max_loudness - min_loudness)
        else:
            dataset['loudness_normalized'] = 0.5
    
    # Other features are already in 0-1 range, but ensure they're numeric
    for feature in audio_features:
        if feature in dataset.columns and feature != 'loudness':
            dataset[feature] = pd.to_numeric(dataset[feature], errors='coerce').fillna(0)
    
    return dataset


def preprocess_dataset(filepath=None):
    """
    Complete preprocessing pipeline: load, convert keys, add compatible keys, normalize.
    
    Args:
        filepath: Path to the dataset CSV file (default: ../data/dataset.csv)
    
    Returns:
        Preprocessed DataFrame ready for recommendation models
    """
    if filepath is None:
        from pathlib import Path
        base_path = Path(__file__).parent.parent
        filepath = str(base_path / 'data' / 'dataset.csv')
    # Load dataset
    df = load_dataset(filepath)
    
    # Convert to Camelot notation
    df = convert_to_camelot(df)
    
    # Add compatible keys
    df = add_compatible_keys(df)
    
    # Normalize audio features
    df = normalize_audio_features(df)
    
    # Ensure tempo is numeric
    df['tempo'] = pd.to_numeric(df['tempo'], errors='coerce')
    df['energy'] = pd.to_numeric(df['energy'], errors='coerce').fillna(0)
    
    print("Preprocessing complete!")
    return df


if __name__ == "__main__":
    # Test preprocessing
    df = preprocess_dataset()
    print(f"\nSample data:")
    print(df[['track_name', 'artists', 'tempo', 'key', 'mode', 'camelot_key', 'energy']].head())
    print(f"\nCompatible keys example:")
    print(df[['track_name', 'camelot_key', 'compatible_keys']].head())
