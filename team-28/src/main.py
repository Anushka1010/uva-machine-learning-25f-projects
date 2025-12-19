"""
Main script for DJ Mixing Recommendation System.
Loads data, trains models, accepts song input, and generates recommendations.
"""

import argparse
import pandas as pd
import time
import os
import sys
from pathlib import Path

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from data_preprocessing import preprocess_dataset
from model_rule_based import recommend_rule_based
from model_audio_similarity import recommend_audio_similarity
from model_hybrid_ml import (
    generate_training_data, train_hybrid_model, load_model,
    recommend_hybrid_ml
)
from evaluation import (
    evaluate_recommendations, compare_models, print_evaluation_results,
    measure_response_time
)
from utils import (
    search_by_track_id, search_by_track_name, search_by_artist,
    search_by_artist_and_track, search_by_index, format_song_info
)


def find_song(dataset, track_id=None, song=None, artist=None, index=None):
    """
    Find a song using various search methods.
    
    Args:
        dataset: DataFrame with song data
        track_id: Spotify track ID
        song: Track name
        artist: Artist name
        index: Dataset row index
    
    Returns:
        Series with song data or None if not found
    """
    if track_id:
        result = search_by_track_id(dataset, track_id)
        if result is not None:
            return result
        print(f"Track ID '{track_id}' not found.")
        return None
    
    if index is not None:
        result = search_by_index(dataset, index)
        if result is not None:
            return result
        print(f"Index {index} out of range.")
        return None
    
    if song and artist:
        results = search_by_artist_and_track(dataset, artist, song, fuzzy=True)
        if len(results) > 0:
            if len(results) == 1:
                return results.iloc[0]
            else:
                print(f"\nFound {len(results)} matches. Please select one:")
                for i, (_, row) in enumerate(results.head(10).iterrows()):
                    print(f"{i+1}. {format_song_info(row)}")
                try:
                    choice = int(input("\nEnter number (1-{}): ".format(min(len(results), 10)))) - 1
                    if 0 <= choice < len(results):
                        return results.iloc[choice]
                except (ValueError, IndexError):
                    print("Invalid choice.")
                return None
    
    if song:
        results = search_by_track_name(dataset, song, fuzzy=True)
        if len(results) > 0:
            if len(results) == 1:
                return results.iloc[0]
            else:
                print(f"\nFound {len(results)} matches. Please select one:")
                for i, (_, row) in enumerate(results.head(10).iterrows()):
                    print(f"{i+1}. {format_song_info(row)}")
                try:
                    choice = int(input("\nEnter number (1-{}): ".format(min(len(results), 10)))) - 1
                    if 0 <= choice < len(results):
                        return results.iloc[choice]
                except (ValueError, IndexError):
                    print("Invalid choice.")
                return None
    
    if artist:
        results = search_by_artist(dataset, artist, fuzzy=True)
        if len(results) > 0:
            print(f"\nFound {len(results)} songs by '{artist}'. Please select one:")
            for i, (_, row) in enumerate(results.head(20).iterrows()):
                print(f"{i+1}. {format_song_info(row)}")
            try:
                choice = int(input("\nEnter number (1-{}): ".format(min(len(results), 20)))) - 1
                if 0 <= choice < len(results):
                    return results.iloc[choice]
            except (ValueError, IndexError):
                print("Invalid choice.")
            return None
    
    return None


def display_recommendations(recommendations, model_name, score_column='score'):
    """
    Display recommendations in a formatted way.
    
    Args:
        recommendations: DataFrame with recommendations
        model_name: Name of the model
        score_column: Name of the score column
    """
    print(f"\n{'='*80}")
    print(f"{model_name} Recommendations (Top 10)")
    print(f"{'='*80}")
    
    if len(recommendations) == 0:
        print("No recommendations found.")
        return
    
    for i, (_, rec) in enumerate(recommendations.iterrows(), 1):
        track_name = rec.get('track_name', 'Unknown')
        artists = rec.get('artists', 'Unknown')
        bpm = rec.get('tempo', 0)
        key = rec.get('camelot_key', 'N/A')
        energy = rec.get('energy', 0)
        score = rec.get(score_column, 0)
        
        print(f"\n{i}. {track_name}")
        print(f"   Artist: {artists}")
        print(f"   BPM: {bpm:.1f} | Key: {key} | Energy: {energy:.2f} | Score: {score:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description='DJ Mixing Recommendation System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --track_id 5SuOikwiRyPMVoIQDJUgSV
  python main.py --song "Strobe" --artist "deadmau5"
  python main.py --song "Strobe"
  python main.py --artist "deadmau5"
  python main.py --index 0
        """
    )
    
    parser.add_argument('--track_id', type=str, help='Spotify track ID')
    parser.add_argument('--song', type=str, help='Track name')
    parser.add_argument('--artist', type=str, help='Artist name')
    parser.add_argument('--index', type=int, help='Dataset row index')
    # Determine base path (project root)
    base_path = Path(__file__).parent.parent
    parser.add_argument('--data', type=str, default=str(base_path / 'data' / 'dataset.csv'),
                       help='Path to dataset CSV file')
    parser.add_argument('--model_path', type=str, default=str(base_path / 'hybrid_model.pkl'),
                       help='Path to save/load hybrid ML model')
    parser.add_argument('--train_model', action='store_true',
                       help='Force retrain hybrid ML model')
    parser.add_argument('--no_eval', action='store_true',
                       help='Skip evaluation metrics')
    
    args = parser.parse_args()
    
    # Check if at least one search method is provided
    if not any([args.track_id, args.song, args.artist, args.index is not None]):
        parser.print_help()
        return
    
    print("="*80)
    print("DJ Mixing Recommendation System")
    print("="*80)
    
    # Load and preprocess dataset
    print("\n[1/5] Loading and preprocessing dataset...")
    start_time = time.time()
    dataset = preprocess_dataset(args.data)
    load_time = time.time() - start_time
    print(f"Dataset loaded in {load_time:.2f} seconds")
    
    # Load or train hybrid ML model
    print("\n[2/5] Preparing hybrid ML model...")
    model, feature_cols = load_model(args.model_path)
    
    if model is None or args.train_model:
        print("Training hybrid ML model (this may take a few minutes)...")
        training_data = generate_training_data(dataset, n_samples=10000)
        model, feature_cols = train_hybrid_model(training_data, args.model_path)
    else:
        print(f"Loaded pre-trained model from {args.model_path}")
    
    # Find current song
    print("\n[3/5] Finding current song...")
    current_song = find_song(
        dataset,
        track_id=args.track_id,
        song=args.song,
        artist=args.artist,
        index=args.index
    )
    
    if current_song is None:
        print("Could not find the specified song. Please try again with different search criteria.")
        return
    
    print(f"\nCurrent Song: {format_song_info(current_song)}")
    
    # Generate recommendations from all three models
    print("\n[4/5] Generating recommendations...")
    total_start_time = time.time()
    
    # Rule-based recommendations
    rule_based_recs, rule_time = measure_response_time(
        recommend_rule_based, current_song, dataset, top_k=10
    )
    rule_based_recs['score'] = rule_based_recs.get('mixing_score', 0)
    
    # Audio similarity recommendations
    audio_sim_recs, audio_time = measure_response_time(
        recommend_audio_similarity, current_song, dataset, top_k=10
    )
    audio_sim_recs['score'] = audio_sim_recs.get('similarity_score', 0)
    
    # Hybrid ML recommendations
    hybrid_ml_recs, hybrid_time = measure_response_time(
        recommend_hybrid_ml, current_song, dataset, model, feature_cols, top_k=10
    )
    hybrid_ml_recs['score'] = hybrid_ml_recs.get('compatibility_score', 0)
    
    total_time = time.time() - total_start_time
    
    # Display recommendations
    print("\n[5/5] Results:")
    display_recommendations(rule_based_recs, "Rule-Based", 'mixing_score')
    display_recommendations(audio_sim_recs, "Audio Similarity", 'similarity_score')
    display_recommendations(hybrid_ml_recs, "Hybrid ML", 'compatibility_score')
    
    # Evaluation metrics
    if not args.no_eval:
        print("\n" + "="*80)
        print("Evaluation Metrics")
        print("="*80)
        
        comparison = compare_models(
            current_song, dataset,
            rule_based_recs, audio_sim_recs, hybrid_ml_recs
        )
        print_evaluation_results(comparison)
    
    # Performance summary
    print("\n" + "="*80)
    print("Performance Summary")
    print("="*80)
    print(f"Rule-Based Model: {rule_time:.3f} seconds")
    print(f"Audio Similarity Model: {audio_time:.3f} seconds")
    print(f"Hybrid ML Model: {hybrid_time:.3f} seconds")
    print(f"Total Recommendation Time: {total_time:.3f} seconds")
    print(f"Target: < 1.0 second")
    print(f"Target Met: {'✓' if total_time < 1.0 else '✗'}")
    print("="*80)


if __name__ == "__main__":
    main()
