"""
Audio Similarity Baseline Recommendation System.
Uses cosine similarity on audio features, ignoring BPM and key constraints.
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from utils import get_compatible_keys


def extract_audio_features(song):
    """
    Extract audio feature vector for similarity calculation.
    
    Args:
        song: Series with song data
    
    Returns:
        NumPy array of audio features
    """
    features = [
        'energy',
        'valence',
        'danceability',
        'acousticness',
        'instrumentalness',
        'speechiness',
        'liveness'
    ]
    
    # Use normalized loudness if available, otherwise use original
    if 'loudness_normalized' in song.index:
        features.append('loudness_normalized')
    elif 'loudness' in song.index:
        # Normalize loudness on the fly (rough approximation)
        loudness = song.get('loudness', 0)
        normalized_loudness = (loudness + 60) / 60  # Rough normalization
        features.append('loudness')
        # We'll handle this in the vector extraction
    
    feature_vector = []
    for feature in features:
        if feature == 'loudness' and 'loudness_normalized' not in song.index:
            # Normalize loudness on the fly
            loudness = song.get('loudness', 0)
            normalized = max(0, min(1, (loudness + 60) / 60))
            feature_vector.append(normalized)
        else:
            value = song.get(feature, 0)
            feature_vector.append(float(value) if pd.notna(value) else 0.0)
    
    return np.array(feature_vector)


def calculate_audio_similarity(song1, song2):
    """
    Calculate cosine similarity between two songs' audio features.
    
    Args:
        song1: Series with first song data
        song2: Series with second song data
    
    Returns:
        Cosine similarity score (0-1)
    """
    features1 = extract_audio_features(song1).reshape(1, -1)
    features2 = extract_audio_features(song2).reshape(1, -1)
    
    similarity = cosine_similarity(features1, features2)[0][0]
    return similarity


def recommend_audio_similarity(current_song, dataset, top_k=10):
    """
    Audio similarity recommendation: rank by cosine similarity on audio features.
    
    Args:
        current_song: Series with current song data
        dataset: DataFrame with all candidate songs
        top_k: Number of recommendations to return
    
    Returns:
        DataFrame with top_k recommendations, sorted by similarity score
    """
    current_track_id = current_song.get('track_id', '')
    
    # Exclude current song
    candidates = dataset[dataset['track_id'] != current_track_id].copy()
    
    if len(candidates) == 0:
        return pd.DataFrame()
    
    # Extract current song features
    current_features = extract_audio_features(current_song).reshape(1, -1)
    
    # Extract candidate features
    candidate_features = np.array([
        extract_audio_features(row) for _, row in candidates.iterrows()
    ])
    
    # Calculate cosine similarities
    similarities = cosine_similarity(current_features, candidate_features)[0]
    
    # Add similarity scores
    candidates['similarity_score'] = similarities
    
    # Heavily prioritize BPM, Key, and Energy with strong bonuses
    current_bpm = current_song.get('tempo', 0)
    current_camelot = current_song.get('camelot_key', '')
    current_energy = current_song.get('energy', 0)
    
    # BPM bonus: Strong boost for songs within ±6 BPM
    bpm_diff = abs(candidates['tempo'] - current_bpm)
    bpm_bonus = np.where(bpm_diff <= 6, 0.3, np.where(bpm_diff <= 12, 0.1, -0.2))
    candidates['similarity_score'] += bpm_bonus
    
    # Key bonus: Strong boost for compatible keys
    if pd.notna(current_camelot):
        compatible_keys = get_compatible_keys(current_camelot)
        key_match = candidates['camelot_key'].isin(compatible_keys)
        candidates.loc[key_match, 'similarity_score'] += 0.25
        candidates.loc[~key_match, 'similarity_score'] -= 0.15
    
    # Energy bonus: Boost for similar energy levels
    energy_diff = abs(candidates['energy'] - current_energy)
    energy_bonus = np.where(energy_diff < 0.2, 0.15, np.where(energy_diff < 0.4, 0.05, -0.1))
    candidates['similarity_score'] += energy_bonus
    
    # Add genre bonus: small boost for same genre
    current_genre = current_song.get('track_genre', '')
    if pd.notna(current_genre):
        current_genre_str = str(current_genre).lower().strip()
        genre_match = candidates['track_genre'].astype(str).str.lower().str.strip() == current_genre_str
        # Small bonus to same-genre songs
        candidates.loc[genre_match, 'similarity_score'] += 0.05
    
    # Remove duplicates: keep highest scoring version of each unique song
    # First deduplicate by track_id
    candidates = candidates.sort_values('similarity_score', ascending=False)
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
    
    # Sort by similarity and return top_k
    recommendations = candidates.nlargest(top_k, 'similarity_score')
    
    return recommendations[['track_id', 'track_name', 'artists', 'tempo', 'camelot_key',
                            'energy', 'similarity_score']]


if __name__ == "__main__":
    # Test with sample data
    from data_preprocessing import preprocess_dataset
    
    df = preprocess_dataset()
    if len(df) > 0:
        current = df.iloc[0]
        print(f"Current song: {current['track_name']} by {current['artists']}")
        
        recommendations = recommend_audio_similarity(current, df, top_k=10)
        print(f"\nFound {len(recommendations)} recommendations:")
        print(recommendations[['track_name', 'artists', 'tempo', 'similarity_score']].head(10))
