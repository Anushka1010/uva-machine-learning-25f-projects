"""
Visualization script for DJ Mixing Recommendation System.
Creates descriptive graphs for data exploration and model results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def load_data(filepath=None):
    """Load and preprocess dataset."""
    print("Loading dataset...")
    import sys
    from pathlib import Path
    
    if filepath is None:
        base_path = Path(__file__).parent.parent
        filepath = str(base_path / 'data' / 'dataset.csv')
    
    sys.path.insert(0, str(Path(__file__).parent))
    from data_preprocessing import preprocess_dataset
    df = preprocess_dataset(filepath)
    return df


def plot_bpm_distribution(df, save_path=None):
    if save_path is None:
        from pathlib import Path
        src_path = Path(__file__).parent
        save_path = str(src_path / 'plots' / 'bpm_distribution.png')
    """Plot BPM distribution across the dataset."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    axes[0].hist(df['tempo'].dropna(), bins=50, color='#8B5CF6', edgecolor='black', alpha=0.7)
    axes[0].axvline(df['tempo'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["tempo"].mean():.1f} BPM')
    axes[0].axvline(df['tempo'].median(), color='green', linestyle='--', linewidth=2, label=f'Median: {df["tempo"].median():.1f} BPM')
    axes[0].set_xlabel('BPM (Beats Per Minute)', fontsize=12)
    axes[0].set_ylabel('Number of Songs', fontsize=12)
    axes[0].set_title('BPM Distribution Across Dataset', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Box plot by genre (if available)
    if 'track_genre' in df.columns:
        top_genres = df['track_genre'].value_counts().head(10).index
        genre_df = df[df['track_genre'].isin(top_genres)]
        genre_df.boxplot(column='tempo', by='track_genre', ax=axes[1], rot=45)
        axes[1].set_title('BPM Distribution by Genre (Top 10)', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Genre', fontsize=12)
        axes[1].set_ylabel('BPM', fontsize=12)
        plt.suptitle('')  # Remove default title
    else:
        axes[1].text(0.5, 0.5, 'Genre data not available', 
                    ha='center', va='center', fontsize=12, transform=axes[1].transAxes)
        axes[1].set_title('BPM by Genre', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_key_distribution(df, save_path=None):
    if save_path is None:
        from pathlib import Path
        src_path = Path(__file__).parent
        save_path = str(src_path / 'plots' / 'key_distribution.png')
    """Plot key distribution (Camelot Wheel)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Camelot key distribution
    if 'camelot_key' in df.columns:
        camelot_counts = df['camelot_key'].value_counts().sort_index()
        colors = plt.cm.viridis(np.linspace(0, 1, len(camelot_counts)))
        
        axes[0].bar(camelot_counts.index, camelot_counts.values, color=colors, edgecolor='black')
        axes[0].set_xlabel('Camelot Key', fontsize=12)
        axes[0].set_ylabel('Number of Songs', fontsize=12)
        axes[0].set_title('Camelot Key Distribution', fontsize=14, fontweight='bold')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(True, alpha=0.3, axis='y')
    
    # Major vs Minor distribution
    if 'mode' in df.columns:
        mode_counts = df['mode'].value_counts()
        labels = ['Minor (B)', 'Major (A)']
        colors = ['#FF6B6B', '#4ECDC4']
        axes[1].pie(mode_counts.values, labels=labels, autopct='%1.1f%%', 
                   colors=colors, startangle=90, textprops={'fontsize': 12})
        axes[1].set_title('Major vs Minor Distribution', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_energy_distribution(df, save_path=None):
    if save_path is None:
        from pathlib import Path
        src_path = Path(__file__).parent
        save_path = str(src_path / 'plots' / 'energy_distribution.png')
    """Plot energy distribution and correlation with BPM."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Energy histogram
    axes[0].hist(df['energy'].dropna(), bins=30, color='#FF6B6B', edgecolor='black', alpha=0.7)
    axes[0].axvline(df['energy'].mean(), color='red', linestyle='--', linewidth=2, 
                    label=f'Mean: {df["energy"].mean():.2f}')
    axes[0].set_xlabel('Energy Level', fontsize=12)
    axes[0].set_ylabel('Number of Songs', fontsize=12)
    axes[0].set_title('Energy Distribution', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Energy vs BPM scatter
    sample_df = df.sample(min(5000, len(df)))  # Sample for performance
    scatter = axes[1].scatter(sample_df['tempo'], sample_df['energy'], 
                             alpha=0.3, c=sample_df['energy'], cmap='viridis', s=10)
    axes[1].set_xlabel('BPM', fontsize=12)
    axes[1].set_ylabel('Energy', fontsize=12)
    axes[1].set_title('Energy vs BPM Relationship', fontsize=14, fontweight='bold')
    plt.colorbar(scatter, ax=axes[1], label='Energy Level')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_audio_features_correlation(df, save_path=None):
    if save_path is None:
        from pathlib import Path
        src_path = Path(__file__).parent
        save_path = str(src_path / 'plots' / 'audio_features_correlation.png')
    """Plot correlation matrix of audio features."""
    audio_features = ['energy', 'valence', 'danceability', 'acousticness', 
                     'instrumentalness', 'loudness', 'speechiness', 'liveness', 'tempo']
    
    available_features = [f for f in audio_features if f in df.columns]
    
    if len(available_features) < 2:
        print("Not enough audio features for correlation plot")
        return
    
    corr_matrix = df[available_features].corr()
    
    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', 
               center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Audio Features Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_model_comparison(results_dict, save_path=None):
    if save_path is None:
        from pathlib import Path
        src_path = Path(__file__).parent
        save_path = str(src_path / 'plots' / 'model_comparison.png')
    """Plot comparison of different models' performance."""
    if not results_dict:
        print("No results provided for model comparison")
        return
    
    models = list(results_dict.keys())
    metrics = ['bpm_compatibility', 'key_compatibility', 'energy_flow']
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    for i, metric in enumerate(metrics):
        values = [results_dict[model].get(metric, 0) * 100 for model in models]
        colors = ['#8B5CF6', '#4ECDC4', '#FF6B6B']
        
        bars = axes[i].bar(models, values, color=colors[:len(models)], edgecolor='black', alpha=0.7)
        axes[i].set_ylabel('Percentage (%)', fontsize=12)
        axes[i].set_title(f'{metric.replace("_", " ").title()}', fontsize=14, fontweight='bold')
        axes[i].set_ylim(0, 105)
        axes[i].grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            axes[i].text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%', ha='center', va='bottom', fontsize=10)
    
    plt.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_feature_importance(feature_importance_dict, save_path=None):
    if save_path is None:
        from pathlib import Path
        src_path = Path(__file__).parent
        save_path = str(src_path / 'plots' / 'feature_importance.png')
    """Plot feature importance for hybrid ML model."""
    if not feature_importance_dict:
        print("No feature importance data provided")
        return
    
    features = list(feature_importance_dict.keys())
    importance = list(feature_importance_dict.values())
    
    # Sort by importance
    sorted_data = sorted(zip(features, importance), key=lambda x: x[1], reverse=True)
    features, importance = zip(*sorted_data)
    
    plt.figure(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(features)))
    bars = plt.barh(features, importance, color=colors, edgecolor='black', alpha=0.7)
    
    plt.xlabel('Importance Score', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.title('Feature Importance in Hybrid ML Model', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, importance)):
        plt.text(val, i, f' {val:.4f}', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_recommendation_example(current_song, recommendations, model_name, save_path=None):
    if save_path is None:
        from pathlib import Path
        src_path = Path(__file__).parent
        save_path = str(src_path / 'plots' / 'recommendation_example.png')
    """Plot example recommendations with BPM, Key, and Energy."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Extract data
    rec_bpm = recommendations['tempo'].values
    rec_keys = recommendations['camelot_key'].values
    rec_energy = recommendations['energy'].values
    rec_names = recommendations['track_name'].values[:10]  # Limit to 10 for readability
    
    current_bpm = current_song.get('tempo', 0)
    current_key = current_song.get('camelot_key', 'Unknown')
    current_energy = current_song.get('energy', 0)
    
    # BPM comparison
    axes[0, 0].barh(range(len(rec_bpm)), rec_bpm, color='#8B5CF6', alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(current_bpm, color='red', linestyle='--', linewidth=2, label=f'Current: {current_bpm:.1f} BPM')
    axes[0, 0].axvline(current_bpm - 6, color='orange', linestyle=':', linewidth=1, label='±6 BPM range')
    axes[0, 0].axvline(current_bpm + 6, color='orange', linestyle=':', linewidth=1)
    axes[0, 0].set_yticks(range(len(rec_bpm)))
    axes[0, 0].set_yticklabels([f"Song {i+1}" for i in range(len(rec_bpm))])
    axes[0, 0].set_xlabel('BPM', fontsize=12)
    axes[0, 0].set_title('BPM Comparison', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3, axis='x')
    
    # Key compatibility
    key_counts = pd.Series(rec_keys).value_counts()
    colors = plt.cm.Set3(np.linspace(0, 1, len(key_counts)))
    axes[0, 1].pie(key_counts.values, labels=key_counts.index, autopct='%1.1f%%', 
                   colors=colors, startangle=90)
    axes[0, 1].set_title(f'Key Distribution (Current: {current_key})', fontsize=14, fontweight='bold')
    
    # Energy comparison
    axes[1, 0].barh(range(len(rec_energy)), rec_energy, color='#FF6B6B', alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(current_energy, color='red', linestyle='--', linewidth=2, 
                       label=f'Current: {current_energy:.2f}')
    axes[1, 0].set_yticks(range(len(rec_energy)))
    axes[1, 0].set_yticklabels([f"Song {i+1}" for i in range(len(rec_energy))])
    axes[1, 0].set_xlabel('Energy Level', fontsize=12)
    axes[1, 0].set_title('Energy Comparison', fontsize=14, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='x')
    
    # 3D scatter: BPM vs Energy vs Key compatibility
    key_compat = [1 if k == current_key or (pd.notna(current_key) and k in 
                ['7A', '9A', '8B'] if current_key == '8A' else []) else 0 
                for k in rec_keys]  # Simplified compatibility check
    
    scatter = axes[1, 1].scatter(rec_bpm, rec_energy, c=key_compat, 
                                cmap='RdYlGn', s=100, alpha=0.6, edgecolors='black')
    axes[1, 1].scatter([current_bpm], [current_energy], color='red', s=200, 
                     marker='*', label='Current Song', edgecolors='black', linewidth=2)
    axes[1, 1].set_xlabel('BPM', fontsize=12)
    axes[1, 1].set_ylabel('Energy', fontsize=12)
    axes[1, 1].set_title('BPM vs Energy (Color = Key Compatible)', fontsize=14, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[1, 1], label='Key Compatible')
    
    plt.suptitle(f'{model_name} Recommendations Example', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_bpm_tolerance_analysis(df, save_path=None):
    if save_path is None:
        from pathlib import Path
        src_path = Path(__file__).parent
        save_path = str(src_path / 'plots' / 'bpm_tolerance_analysis.png')
    """Analyze how BPM tolerance affects number of compatible songs."""
    # Sample a few songs for analysis
    sample_songs = df.sample(min(100, len(df)))
    
    tolerances = [2, 4, 6, 8, 10, 12]
    avg_compatible = []
    
    for tolerance in tolerances:
        compatible_counts = []
        for _, song in sample_songs.iterrows():
            bpm = song['tempo']
            compatible = len(df[(abs(df['tempo'] - bpm) <= tolerance) & 
                              (df.index != song.name)])
            compatible_counts.append(compatible)
        avg_compatible.append(np.mean(compatible_counts))
    
    plt.figure(figsize=(10, 6))
    plt.plot(tolerances, avg_compatible, marker='o', linewidth=2, markersize=8, color='#8B5CF6')
    plt.fill_between(tolerances, avg_compatible, alpha=0.3, color='#8B5CF6')
    plt.xlabel('BPM Tolerance (±)', fontsize=12)
    plt.ylabel('Average Number of Compatible Songs', fontsize=12)
    plt.title('Effect of BPM Tolerance on Compatible Song Count', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.axvline(6, color='red', linestyle='--', linewidth=2, label='Standard DJ Rule (±6)')
    plt.legend()
    
    # Add value labels
    for tol, count in zip(tolerances, avg_compatible):
        plt.text(tol, count, f'{int(count)}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def generate_all_visualizations(df, results_dict=None, feature_importance=None):
    """Generate all visualizations."""
    print("\n" + "="*60)
    print("Generating Visualizations")
    print("="*60)
    
    # Create plots directory inside src/
    from pathlib import Path
    src_path = Path(__file__).parent
    plots_dir = src_path / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Data exploration plots
    print("\n1. Creating data exploration plots...")
    plot_bpm_distribution(df)
    plot_key_distribution(df)
    plot_energy_distribution(df)
    plot_audio_features_correlation(df)
    plot_bpm_tolerance_analysis(df)
    
    # Model results plots
    if results_dict:
        print("\n2. Creating model comparison plots...")
        plot_model_comparison(results_dict)
    
    if feature_importance:
        print("\n3. Creating feature importance plot...")
        plot_feature_importance(feature_importance)
    
    print("\n" + "="*60)
    print(f"All visualizations saved to 'src/plots/' directory")
    print("="*60)


def main():
    """Main function to generate all visualizations."""
    # Load data
    try:
        df = load_data()
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Please ensure data/dataset.csv exists and data_preprocessing.py works correctly")
        return
    
    # Example model results (you can replace with actual results)
    example_results = {
        'Rule-Based': {
            'bpm_compatibility': 1.0,
            'key_compatibility': 1.0,
            'energy_flow': 0.95
        },
        'Audio Similarity': {
            'bpm_compatibility': 0.3,
            'key_compatibility': 0.25,
            'energy_flow': 0.85
        },
        'Hybrid ML': {
            'bpm_compatibility': 0.95,
            'key_compatibility': 0.90,
            'energy_flow': 0.92
        }
    }
    
    # Example feature importance (you can replace with actual from model)
    example_feature_importance = {
        'bpm_distance': 0.1518,
        'key_compatible': 0.6740,
        'energy_diff': 0.1678,
        'valence_diff': 0.0000,
        'danceability_diff': 0.0014,
        'acousticness_diff': 0.0020,
        'instrumentalness_diff': 0.0029,
        'loudness_diff': 0.0001,
        'genre_compatible': 0.0000
    }
    
    # Generate all visualizations
    generate_all_visualizations(df, example_results, example_feature_importance)
    
    print("\n✅ Visualization generation complete!")


if __name__ == "__main__":
    main()
