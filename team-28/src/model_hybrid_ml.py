"""
Hybrid ML Recommendation System using XGBoost.
Combines DJ mixing rules with machine learning for intelligent recommendations.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from utils import get_compatible_keys
import pickle
import os


def generate_training_data(dataset, n_samples=10000, random_state=42):
    """
    Generate synthetic training data based on DJ mixing rules.
    
    Positive examples (label=1): BPM within ±6 AND key compatible AND energy difference < 0.3
    Negative examples (label=0): BPM > ±6 OR key incompatible OR energy difference > 0.5
    
    Args:
        dataset: DataFrame with preprocessed song data
        n_samples: Number of training samples to generate
        random_state: Random seed for reproducibility
    
    Returns:
        DataFrame with columns: [features..., 'label']
    """
    np.random.seed(random_state)
    
    print(f"Generating {n_samples} training samples...")
    
    training_pairs = []
    n_songs = len(dataset)
    
    # Sample pairs
    for _ in range(n_samples):
        # Randomly select two different songs
        idx1, idx2 = np.random.choice(n_songs, size=2, replace=False)
        song1 = dataset.iloc[idx1]
        song2 = dataset.iloc[idx2]
        
        # Extract features
        features = extract_pair_features(song1, song2)
        
        # Determine label based on DJ mixing rules
        label = determine_label(song1, song2)
        
        features['label'] = label
        training_pairs.append(features)
    
    training_df = pd.DataFrame(training_pairs)
    
    # Balance the dataset (approximately 50/50)
    positive_count = training_df['label'].sum()
    negative_count = len(training_df) - positive_count
    
    print(f"Generated {positive_count} positive examples, {negative_count} negative examples")
    
    return training_df


def extract_pair_features(song1, song2):
    """
    Extract features for a song pair to use in ML model.
    
    Args:
        song1: First song (Series)
        song2: Second song (Series)
    
    Returns:
        Dictionary of features
    """
    # BPM distance (40% weight in scoring)
    bpm_distance = abs(song1.get('tempo', 0) - song2.get('tempo', 0))
    
    # Key compatibility (30% weight in scoring)
    key1 = song1.get('camelot_key', '')
    key2 = song2.get('camelot_key', '')
    key_compatible = 1 if (pd.notna(key1) and pd.notna(key2) and 
                          key2 in get_compatible_keys(key1)) else 0
    
    # Energy difference (30% weight in scoring)
    energy_diff = abs(song1.get('energy', 0) - song2.get('energy', 0))
    
    # Additional audio feature differences
    valence_diff = abs(song1.get('valence', 0) - song2.get('valence', 0))
    danceability_diff = abs(song1.get('danceability', 0) - song2.get('danceability', 0))
    acousticness_diff = abs(song1.get('acousticness', 0) - song2.get('acousticness', 0))
    instrumentalness_diff = abs(song1.get('instrumentalness', 0) - song2.get('instrumentalness', 0))
    
    # Loudness difference (normalized)
    loudness1 = song1.get('loudness', 0)
    loudness2 = song2.get('loudness', 0)
    loudness_diff = abs(loudness1 - loudness2) / 60.0  # Normalize to 0-1 range
    
    # Genre compatibility (1 if same genre, 0 if different)
    genre1 = song1.get('track_genre', '')
    genre2 = song2.get('track_genre', '')
    genre_compatible = 1 if (pd.notna(genre1) and pd.notna(genre2) and 
                            str(genre1).lower().strip() == str(genre2).lower().strip()) else 0
    
    features = {
        'bpm_distance': bpm_distance,
        'key_compatible': key_compatible,
        'energy_diff': energy_diff,
        'valence_diff': valence_diff,
        'danceability_diff': danceability_diff,
        'acousticness_diff': acousticness_diff,
        'instrumentalness_diff': instrumentalness_diff,
        'loudness_diff': loudness_diff,
        'genre_compatible': genre_compatible,
    }
    
    return features


def determine_label(song1, song2):
    """
    Determine label (1=good transition, 0=bad) based on DJ mixing rules.
    
    Args:
        song1: First song (Series)
        song2: Second song (Series)
    
    Returns:
        1 if good transition, 0 if bad
    """
    bpm1 = song1.get('tempo', 0)
    bpm2 = song2.get('tempo', 0)
    bpm_diff = abs(bpm1 - bpm2)
    
    key1 = song1.get('camelot_key', '')
    key2 = song2.get('camelot_key', '')
    key_compatible = (pd.notna(key1) and pd.notna(key2) and 
                     key2 in get_compatible_keys(key1))
    
    energy1 = song1.get('energy', 0)
    energy2 = song2.get('energy', 0)
    energy_diff = abs(energy1 - energy2)
    
    # Positive: BPM within ±6 AND key compatible AND energy difference < 0.3
    if bpm_diff <= 6 and key_compatible and energy_diff < 0.3:
        return 1
    
    # Negative: BPM > ±6 OR key incompatible OR energy difference > 0.5
    if bpm_diff > 6 or not key_compatible or energy_diff > 0.5:
        return 0
    
    # Borderline cases: default to negative
    return 0


def train_hybrid_model(training_data, model_path=None):
    """
    Train XGBoost model on synthetic training data.
    
    Args:
        training_data: DataFrame with features and labels
        model_path: Path to save the trained model (default: ../hybrid_model.pkl)
    
    Returns:
        Trained XGBoost model
    """
    if model_path is None:
        from pathlib import Path
        base_path = Path(__file__).parent.parent
        model_path = str(base_path / 'hybrid_model.pkl')
    print("Training XGBoost model...")
    
    # Separate features and labels
    feature_cols = [col for col in training_data.columns if col != 'label']
    X = training_data[feature_cols]
    y = training_data['label']
    
    # Create XGBoost classifier
    # Use sample_weight to emphasize BPM (40%), Key (30%), Energy (30%)
    # We'll weight samples based on feature importance
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric='logloss'
    )
    
    # Train model
    model.fit(X, y)
    
    # Save model
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Model trained and saved to {model_path}")
    print(f"Feature importance:")
    for i, feature in enumerate(feature_cols):
        print(f"  {feature}: {model.feature_importances_[i]:.4f}")
    
    return model, feature_cols


def load_model(model_path=None):
    """
    Load a pre-trained model.
    
    Args:
        model_path: Path to saved model (default: ../hybrid_model.pkl)
    
    Returns:
        Tuple of (model, feature_columns) or (None, None) if not found
    """
    if model_path is None:
        from pathlib import Path
        base_path = Path(__file__).parent.parent
        model_path = str(base_path / 'hybrid_model.pkl')
    
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        # Feature columns should match the training data
        feature_cols = ['bpm_distance', 'key_compatible', 'energy_diff', 
                       'valence_diff', 'danceability_diff', 'acousticness_diff',
                       'instrumentalness_diff', 'loudness_diff', 'genre_compatible']
        return model, feature_cols
    return None, None


def predict_compatibility_score(model, feature_cols, current_song, candidate_song):
    """
    Predict compatibility score for a song pair.
    
    Args:
        model: Trained XGBoost model
        feature_cols: List of feature column names
        current_song: Series with current song data
        candidate_song: Series with candidate song data
    
    Returns:
        Compatibility score (probability of good transition, 0-1)
    """
    features = extract_pair_features(current_song, candidate_song)
    
    # Handle models trained without genre_compatible feature
    feature_vector = []
    for col in feature_cols:
        if col in features:
            feature_vector.append(features[col])
        else:
            # Default value if feature missing (for backward compatibility)
            feature_vector.append(0.0)
    
    feature_vector = np.array([feature_vector])
    
    # Predict probability of positive class (good transition)
    score = model.predict_proba(feature_vector)[0][1]
    
    return score


def recommend_hybrid_ml(current_song, dataset, model, feature_cols, top_k=10):
    """
    Hybrid ML recommendation: use trained model to predict compatibility scores.
    
    Args:
        current_song: Series with current song data
        dataset: DataFrame with all candidate songs
        model: Trained XGBoost model
        feature_cols: List of feature column names
        top_k: Number of recommendations to return
    
    Returns:
        DataFrame with top_k recommendations, sorted by compatibility score
    """
    current_track_id = current_song.get('track_id', '')
    
    # Exclude current song
    candidates = dataset[dataset['track_id'] != current_track_id].copy()
    
    if len(candidates) == 0:
        return pd.DataFrame()
    
    # Calculate compatibility scores for all candidates
    scores = []
    for _, candidate in candidates.iterrows():
        score = predict_compatibility_score(model, feature_cols, current_song, candidate)
        scores.append(score)
    
    candidates['compatibility_score'] = scores
    
    # Heavily prioritize BPM, Key, and Energy with strong bonuses
    current_bpm = current_song.get('tempo', 0)
    current_camelot = current_song.get('camelot_key', '')
    current_energy = current_song.get('energy', 0)
    
    # BPM bonus: Strong boost for songs within ±6 BPM
    bpm_diff = abs(candidates['tempo'] - current_bpm)
    bpm_bonus = np.where(bpm_diff <= 6, 0.2, np.where(bpm_diff <= 12, 0.05, -0.15))
    candidates['compatibility_score'] += bpm_bonus
    
    # Key bonus: Strong boost for compatible keys
    if pd.notna(current_camelot):
        compatible_keys = get_compatible_keys(current_camelot)
        key_match = candidates['camelot_key'].isin(compatible_keys)
        candidates.loc[key_match, 'compatibility_score'] += 0.15
        candidates.loc[~key_match, 'compatibility_score'] -= 0.1
    
    # Energy bonus: Boost for similar energy levels
    energy_diff = abs(candidates['energy'] - current_energy)
    energy_bonus = np.where(energy_diff < 0.2, 0.1, np.where(energy_diff < 0.4, 0.03, -0.05))
    candidates['compatibility_score'] += energy_bonus
    
    # Add genre bonus: small boost for same genre
    current_genre = current_song.get('track_genre', '')
    if pd.notna(current_genre):
        current_genre_str = str(current_genre).lower().strip()
        genre_match = candidates['track_genre'].astype(str).str.lower().str.strip() == current_genre_str
        # Small bonus to same-genre songs
        candidates.loc[genre_match, 'compatibility_score'] += 0.02
    
    # Remove duplicates: keep highest scoring version of each unique song
    # First deduplicate by track_id
    candidates = candidates.sort_values('compatibility_score', ascending=False)
    candidates = candidates.drop_duplicates(subset=['track_id'], keep='first')
    
    # Also deduplicate by track_name + artists (in case track_id has duplicates)
    candidates['_unique_key'] = candidates['track_name'].astype(str) + '|' + candidates['artists'].astype(str)
    candidates = candidates.drop_duplicates(subset=['_unique_key'], keep='first')
    
    # Exclude current song by track_name + artists (in case track_id doesn't match)
    current_track_name = str(current_song.get('track_name', '')).lower().strip()
    current_artists = str(current_song.get('artists', '')).lower().strip()
    current_unique_key = f"{current_track_name}|{current_artists}"
    candidates = candidates[candidates['_unique_key'].str.lower() != current_unique_key]
    candidates = candidates.drop('_unique_key', axis=1)
    
    # Sort by score and return top_k
    recommendations = candidates.nlargest(top_k, 'compatibility_score')
    
    return recommendations[['track_id', 'track_name', 'artists', 'tempo', 'camelot_key',
                           'energy', 'compatibility_score']]


if __name__ == "__main__":
    # Test with sample data
    from data_preprocessing import preprocess_dataset
    
    df = preprocess_dataset()
    if len(df) > 0:
        print("Generating training data...")
        training_data = generate_training_data(df, n_samples=5000)
        
        print("\nTraining model...")
        model, feature_cols = train_hybrid_model(training_data)
        
        print("\nTesting recommendations...")
        current = df.iloc[0]
        print(f"Current song: {current['track_name']} by {current['artists']}")
        
        recommendations = recommend_hybrid_ml(current, df, model, feature_cols, top_k=10)
        print(f"\nFound {len(recommendations)} recommendations:")
        print(recommendations[['track_name', 'artists', 'tempo', 'camelot_key', 'compatibility_score']].head(10))
