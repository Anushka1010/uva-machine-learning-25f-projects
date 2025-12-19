"""
Rule-Based DJ Recommendation System.
Uses hard constraints: BPM ±6, key compatibility, and energy flow.
"""

import pandas as pd
import numpy as np
from utils import get_compatible_keys


def calculate_bpm_score(current_bpm, candidate_bpm, max_diff=6):
    """
    Calculate BPM compatibility score.
    Closer BPM = higher score.
    
    Args:
        current_bpm: Current song BPM
        candidate_bpm: Candidate song BPM
        max_diff: Maximum allowed BPM difference (default: 6)
    
    Returns:
        Score between 0 and 1 (1.0 = same BPM, 0.0 = max_diff away)
    """
    bpm_diff = abs(current_bpm - candidate_bpm)
    
    if bpm_diff > max_diff:
        return 0.0
    
    # Score decreases linearly from 1.0 (same BPM) to 0.0 (max_diff away)
    return 1.0 - (bpm_diff / max_diff)


def calculate_key_score(current_camelot, candidate_camelot):
    """
    Calculate key compatibility score.
    
    Args:
        current_camelot: Current song Camelot key (e.g., "8A")
        candidate_camelot: Candidate song Camelot key
    
    Returns:
        Score: 1.0 (same key), 0.8 (±1 key), 0.0 (incompatible)
    """
    if pd.isna(current_camelot) or pd.isna(candidate_camelot):
        return 0.0
    
    if current_camelot == candidate_camelot:
        return 1.0
    
    compatible_keys = get_compatible_keys(current_camelot)
    if candidate_camelot in compatible_keys:
        return 0.8
    
    return 0.0


def calculate_energy_flow_score(current_energy, candidate_energy):
    """
    Calculate energy flow score for smooth transitions.
    Small energy differences are preferred.
    
    Args:
        current_energy: Current song energy (0-1)
        candidate_energy: Candidate song energy (0-1)
    
    Returns:
        Score between 0 and 1 (1.0 = same energy, decreases with difference)
    """
    energy_diff = abs(current_energy - candidate_energy)
    
    # Score decreases with energy difference
    # Small differences (< 0.2) get high scores, large differences (> 0.5) get low scores
    if energy_diff < 0.2:
        return 1.0 - (energy_diff / 0.2) * 0.2  # 1.0 to 0.8
    elif energy_diff < 0.5:
        return 0.8 - ((energy_diff - 0.2) / 0.3) * 0.5  # 0.8 to 0.3
    else:
        return max(0.0, 0.3 - (energy_diff - 0.5) * 0.6)  # 0.3 to 0.0


def calculate_genre_score(current_genre, candidate_genre):
    """
    Calculate genre compatibility score.
    Same genre gets full score, different genres get partial score.
    
    Args:
        current_genre: Current song genre
        candidate_genre: Candidate song genre
    
    Returns:
        Score: 1.0 (same genre), 0.7 (different genre)
    """
    if pd.isna(current_genre) or pd.isna(candidate_genre):
        return 0.7  # Neutral score if genre info missing
    
    current_genre_str = str(current_genre).lower().strip()
    candidate_genre_str = str(candidate_genre).lower().strip()
    
    if current_genre_str == candidate_genre_str:
        return 1.0
    
    return 0.7  # Different genre, but still acceptable


def calculate_mixing_score(current_song, candidate_song):
    """
    Calculate combined mixing score for rule-based ranking.
    Heavily prioritizes BPM, Key, and Energy.
    
    Args:
        current_song: Series with current song data
        candidate_song: Series with candidate song data
    
    Returns:
        Combined score (weighted: BPM 40%, Key 35%, Energy 20%, Genre 5%)
    """
    current_bpm = current_song.get('tempo', 0)
    candidate_bpm = candidate_song.get('tempo', 0)
    bpm_score = calculate_bpm_score(current_bpm, candidate_bpm)
    
    current_key = current_song.get('camelot_key', '')
    candidate_key = candidate_song.get('camelot_key', '')
    key_score = calculate_key_score(current_key, candidate_key)
    
    current_energy = current_song.get('energy', 0)
    candidate_energy = candidate_song.get('energy', 0)
    energy_score = calculate_energy_flow_score(current_energy, candidate_energy)
    
    current_genre = current_song.get('track_genre', '')
    candidate_genre = candidate_song.get('track_genre', '')
    genre_score = calculate_genre_score(current_genre, candidate_genre)
    
    # Weighted combination: 40% BPM, 35% Key, 20% Energy, 5% Genre
    # Heavily prioritizes BPM, Key, and Energy
    combined_score = (0.40 * bpm_score) + (0.35 * key_score) + (0.20 * energy_score) + (0.05 * genre_score)
    
    return combined_score


def recommend_rule_based(current_song, dataset, top_k=10):
    """
    Rule-based recommendation: filter by BPM ±6 and key compatibility, then rank.
    
    Args:
        current_song: Series with current song data
        dataset: DataFrame with all candidate songs
        top_k: Number of recommendations to return
    
    Returns:
        DataFrame with top_k recommendations, sorted by mixing score
    """
    current_bpm = current_song.get('tempo', 0)
    current_camelot = current_song.get('camelot_key', '')
    current_track_id = current_song.get('track_id', '')
    
    # Filter: BPM within ±6
    bpm_filter = abs(dataset['tempo'] - current_bpm) <= 6
    
    # Filter: Key compatible
    if pd.notna(current_camelot):
        compatible_keys = get_compatible_keys(current_camelot)
        key_filter = dataset['camelot_key'].isin(compatible_keys)
    else:
        key_filter = pd.Series([True] * len(dataset), index=dataset.index)
    
    # Exclude current song
    not_current = dataset['track_id'] != current_track_id
    
    # Apply filters
    filtered = dataset[bpm_filter & key_filter & not_current].copy()
    
    if len(filtered) == 0:
        return pd.DataFrame()
    
    # Calculate mixing scores
    filtered['mixing_score'] = filtered.apply(
        lambda row: calculate_mixing_score(current_song, row),
        axis=1
    )
    
    # Remove duplicates: keep highest scoring version of each unique song
    # First deduplicate by track_id
    filtered = filtered.sort_values('mixing_score', ascending=False)
    filtered = filtered.drop_duplicates(subset=['track_id'], keep='first')
    
    # Also deduplicate by track_name + artists (in case track_id has duplicates)
    filtered['_unique_key'] = filtered['track_name'].astype(str) + '|' + filtered['artists'].astype(str)
    filtered = filtered.drop_duplicates(subset=['_unique_key'], keep='first')
    
    # Exclude current song by track_name + artists (in case track_id doesn't match)
    current_track_name = str(current_song.get('track_name', '')).lower().strip()
    current_artists = str(current_song.get('artists', '')).lower().strip()
    current_unique_key = f"{current_track_name}|{current_artists}"
    filtered = filtered[filtered['_unique_key'].str.lower() != current_unique_key]
    filtered = filtered.drop('_unique_key', axis=1)
    
    # Sort by score and return top_k
    recommendations = filtered.nlargest(top_k, 'mixing_score')
    
    return recommendations[['track_id', 'track_name', 'artists', 'tempo', 'camelot_key', 
                           'energy', 'mixing_score']]


if __name__ == "__main__":
    # Test with sample data
    from data_preprocessing import preprocess_dataset
    
    df = preprocess_dataset()
    if len(df) > 0:
        current = df.iloc[0]
        print(f"Current song: {current['track_name']} by {current['artists']}")
        print(f"BPM: {current['tempo']:.1f}, Key: {current['camelot_key']}")
        
        recommendations = recommend_rule_based(current, df, top_k=10)
        print(f"\nFound {len(recommendations)} recommendations:")
        print(recommendations[['track_name', 'artists', 'tempo', 'camelot_key', 'mixing_score']].head(10))
