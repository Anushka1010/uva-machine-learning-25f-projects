"""
Evaluation framework for DJ Mixing Recommendation System.
Calculates metrics: BPM compatibility, key compatibility, energy flow, response time.
"""

import pandas as pd
import numpy as np
from utils import get_compatible_keys
import time


def evaluate_bpm_compatibility(current_song, recommendations, tolerance=6):
    """
    Calculate BPM compatibility: % of recommendations within ±tolerance BPM.
    Target: 100%
    
    Args:
        current_song: Series with current song data
        recommendations: DataFrame with recommendations
        tolerance: BPM tolerance (default: 6)
    
    Returns:
        Dictionary with BPM compatibility metrics
    """
    if len(recommendations) == 0:
        return {
            'bpm_compatibility_pct': 0.0,
            'bpm_within_tolerance': 0,
            'total_recommendations': 0,
            'avg_bpm_diff': 0.0
        }
    
    current_bpm = current_song.get('tempo', 0)
    bpm_diffs = abs(recommendations['tempo'] - current_bpm)
    
    within_tolerance = (bpm_diffs <= tolerance).sum()
    total = len(recommendations)
    compatibility_pct = (within_tolerance / total) * 100 if total > 0 else 0.0
    avg_bpm_diff = bpm_diffs.mean()
    
    return {
        'bpm_compatibility_pct': compatibility_pct,
        'bpm_within_tolerance': within_tolerance,
        'total_recommendations': total,
        'avg_bpm_diff': avg_bpm_diff,
        'target_met': compatibility_pct == 100.0
    }


def evaluate_key_compatibility(current_song, recommendations):
    """
    Calculate key compatibility: % of recommendations with compatible keys.
    Target: 80%+
    
    Args:
        current_song: Series with current song data
        recommendations: DataFrame with recommendations
    
    Returns:
        Dictionary with key compatibility metrics
    """
    if len(recommendations) == 0:
        return {
            'key_compatibility_pct': 0.0,
            'key_compatible_count': 0,
            'total_recommendations': 0,
            'target_met': False
        }
    
    current_camelot = current_song.get('camelot_key', '')
    
    if pd.isna(current_camelot):
        return {
            'key_compatibility_pct': 0.0,
            'key_compatible_count': 0,
            'total_recommendations': len(recommendations),
            'target_met': False
        }
    
    compatible_keys = get_compatible_keys(current_camelot)
    
    compatible_count = recommendations['camelot_key'].isin(compatible_keys).sum()
    total = len(recommendations)
    compatibility_pct = (compatible_count / total) * 100 if total > 0 else 0.0
    
    return {
        'key_compatibility_pct': compatibility_pct,
        'key_compatible_count': compatible_count,
        'total_recommendations': total,
        'target_met': compatibility_pct >= 80.0
    }


def evaluate_energy_flow(current_song, recommendations):
    """
    Calculate energy flow metrics: average energy difference for smooth transitions.
    
    Args:
        current_song: Series with current song data
        recommendations: DataFrame with recommendations
    
    Returns:
        Dictionary with energy flow metrics
    """
    if len(recommendations) == 0:
        return {
            'avg_energy_diff': 0.0,
            'max_energy_diff': 0.0,
            'min_energy_diff': 0.0,
            'smooth_transitions_pct': 0.0
        }
    
    current_energy = current_song.get('energy', 0)
    energy_diffs = abs(recommendations['energy'] - current_energy)
    
    # Smooth transitions: energy difference < 0.3
    smooth_count = (energy_diffs < 0.3).sum()
    smooth_pct = (smooth_count / len(recommendations)) * 100 if len(recommendations) > 0 else 0.0
    
    return {
        'avg_energy_diff': energy_diffs.mean(),
        'max_energy_diff': energy_diffs.max(),
        'min_energy_diff': energy_diffs.min(),
        'smooth_transitions_pct': smooth_pct,
        'smooth_transitions_count': smooth_count
    }


def evaluate_recommendations(current_song, recommendations, model_name="Model"):
    """
    Comprehensive evaluation of recommendations.
    
    Args:
        current_song: Series with current song data
        recommendations: DataFrame with recommendations
        model_name: Name of the model being evaluated
    
    Returns:
        Dictionary with all evaluation metrics
    """
    bpm_metrics = evaluate_bpm_compatibility(current_song, recommendations)
    key_metrics = evaluate_key_compatibility(current_song, recommendations)
    energy_metrics = evaluate_energy_flow(current_song, recommendations)
    
    results = {
        'model_name': model_name,
        'num_recommendations': len(recommendations),
        **bpm_metrics,
        **key_metrics,
        **energy_metrics
    }
    
    return results


def compare_models(current_song, dataset, rule_based_recs, audio_sim_recs, hybrid_ml_recs):
    """
    Compare all three models and return evaluation results.
    
    Args:
        current_song: Series with current song data
        dataset: Full dataset
        rule_based_recs: Recommendations from rule-based model
        audio_sim_recs: Recommendations from audio similarity model
        hybrid_ml_recs: Recommendations from hybrid ML model
    
    Returns:
        DataFrame with comparison results
    """
    results = []
    
    # Evaluate each model
    if len(rule_based_recs) > 0:
        results.append(evaluate_recommendations(current_song, rule_based_recs, "Rule-Based"))
    
    if len(audio_sim_recs) > 0:
        results.append(evaluate_recommendations(current_song, audio_sim_recs, "Audio Similarity"))
    
    if len(hybrid_ml_recs) > 0:
        results.append(evaluate_recommendations(current_song, hybrid_ml_recs, "Hybrid ML"))
    
    comparison_df = pd.DataFrame(results)
    return comparison_df


def print_evaluation_results(results):
    """
    Print evaluation results in a formatted way.
    
    Args:
        results: Dictionary or DataFrame with evaluation results
    """
    if isinstance(results, dict):
        results = pd.DataFrame([results])
    
    for _, row in results.iterrows():
        print(f"\n{'='*60}")
        print(f"Model: {row['model_name']}")
        print(f"{'='*60}")
        print(f"Number of Recommendations: {row['num_recommendations']}")
        print(f"\nBPM Compatibility:")
        print(f"  Within ±6 BPM: {row['bpm_within_tolerance']}/{row['total_recommendations']} ({row['bpm_compatibility_pct']:.1f}%)")
        print(f"  Average BPM Difference: {row['avg_bpm_diff']:.2f}")
        print(f"  Target Met (100%): {'✓' if row.get('target_met', False) else '✗'}")
        print(f"\nKey Compatibility:")
        print(f"  Compatible Keys: {row['key_compatible_count']}/{row['total_recommendations']} ({row['key_compatibility_pct']:.1f}%)")
        print(f"  Target Met (80%+): {'✓' if row.get('target_met', False) else '✗'}")
        print(f"\nEnergy Flow:")
        print(f"  Average Energy Difference: {row['avg_energy_diff']:.3f}")
        print(f"  Smooth Transitions (<0.3 diff): {row.get('smooth_transitions_count', 0)} ({row.get('smooth_transitions_pct', 0):.1f}%)")
        print(f"{'='*60}\n")


def measure_response_time(func, *args, **kwargs):
    """
    Measure the execution time of a function.
    
    Args:
        func: Function to measure
        *args: Positional arguments for function
        **kwargs: Keyword arguments for function
    
    Returns:
        Tuple of (result, execution_time_seconds)
    """
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    execution_time = end_time - start_time
    
    return result, execution_time


if __name__ == "__main__":
    # Test evaluation
    from data_preprocessing import preprocess_dataset
    from model_rule_based import recommend_rule_based
    
    df = preprocess_dataset()
    if len(df) > 0:
        current = df.iloc[0]
        print(f"Current song: {current['track_name']} by {current['artists']}")
        print(f"BPM: {current['tempo']:.1f}, Key: {current['camelot_key']}")
        
        recommendations = recommend_rule_based(current, df, top_k=10)
        results = evaluate_recommendations(current, recommendations, "Rule-Based")
        print_evaluation_results(results)
