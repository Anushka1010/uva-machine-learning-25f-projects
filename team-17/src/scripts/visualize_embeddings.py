"""Visualize course embeddings in 2D to see clustering."""

import sys
sys.path.insert(0, ".")

import chromadb
from chromadb.config import Settings as ChromaSettings
from pathlib import Path
import numpy as np

def main():
    print("Loading embeddings from ChromaDB...")
    
    # Connect directly to ChromaDB
    persist_dir = Path("data/chroma")
    client = chromadb.PersistentClient(
        path=str(persist_dir),
        settings=ChromaSettings(anonymized_telemetry=False),
    )
    collection = client.get_collection("courses")
    
    # Get all documents with embeddings
    results = collection.get(
        include=["embeddings", "metadatas", "documents"]
    )
    
    if not results["ids"]:
        print("No documents found! Run indexing first.")
        return
    
    embeddings = np.array(results["embeddings"])
    metadatas = results["metadatas"]
    
    print(f"Loaded {len(embeddings)} embeddings with {embeddings.shape[1]} dimensions")
    
    # Get subject labels for coloring
    subjects = [m.get("subject", "Unknown") for m in metadatas]
    unique_subjects = list(set(subjects))
    print(f"Subjects: {unique_subjects}")
    
    # Reduce to 2D using PCA first (fast), then UMAP if available
    print("\nReducing dimensions...")
    
    try:
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        
        # First reduce with PCA to 50 dims (faster for t-SNE)
        if embeddings.shape[1] > 50:
            print("  PCA: 3072 -> 50 dimensions...")
            pca = PCA(n_components=50)
            embeddings_pca = pca.fit_transform(embeddings)
        else:
            embeddings_pca = embeddings
        
        # Then t-SNE to 2D
        print("  t-SNE: 50 -> 2 dimensions...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
        embeddings_2d = tsne.fit_transform(embeddings_pca)
        
    except ImportError:
        print("sklearn not found. Install with: pip install scikit-learn")
        return
    
    # Create visualization
    try:
        import matplotlib.pyplot as plt
        
        print("\nCreating visualization...")
        
        # Color map for subjects
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_subjects)))
        color_map = {subj: colors[i] for i, subj in enumerate(unique_subjects)}
        point_colors = [color_map[s] for s in subjects]
        
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Plot each subject separately for legend
        for subj in unique_subjects:
            mask = [s == subj for s in subjects]
            xs = embeddings_2d[mask, 0]
            ys = embeddings_2d[mask, 1]
            ax.scatter(xs, ys, c=[color_map[subj]], label=subj, alpha=0.6, s=50)
        
        # Add course labels for a sample
        for i, meta in enumerate(metadatas):
            if i % 10 == 0:  # Label every 10th point to avoid clutter
                label = f"{meta.get('subject', '')} {meta.get('catalog_number', '')}"
                ax.annotate(label, (embeddings_2d[i, 0], embeddings_2d[i, 1]), 
                           fontsize=6, alpha=0.7)
        
        ax.set_title("Course Embeddings Visualization (t-SNE)", fontsize=14)
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
        ax.legend(loc='upper right', title="Subject")
        
        # Save figure
        output_path = "data/embeddings_visualization.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved visualization to: {output_path}")
        
        # Also show interactively if possible
        plt.show()
        
    except ImportError:
        print("matplotlib not found. Install with: pip install matplotlib")
        return
    
    # Print some similarity stats
    print("\n" + "="*50)
    print("SIMILARITY ANALYSIS")
    print("="*50)
    
    # Find most similar course pairs
    from sklearn.metrics.pairwise import cosine_similarity
    
    sim_matrix = cosine_similarity(embeddings)
    np.fill_diagonal(sim_matrix, 0)  # Ignore self-similarity
    
    # Find top similar pairs
    print("\nTop 10 most similar course pairs:")
    flat_indices = np.argsort(sim_matrix.ravel())[-10:][::-1]
    for idx in flat_indices:
        i, j = np.unravel_index(idx, sim_matrix.shape)
        m1, m2 = metadatas[i], metadatas[j]
        c1 = f"{m1.get('subject', '')} {m1.get('catalog_number', '')} - {m1.get('title', '')}"
        c2 = f"{m2.get('subject', '')} {m2.get('catalog_number', '')} - {m2.get('title', '')}"
        print(f"  {sim_matrix[i,j]:.3f}: {c1} <-> {c2}")
    
    # Find least similar pairs among same subject
    print("\nPotential outliers (low similarity within same subject):")
    for subj in ["CS", "MATH", "STAT"]:
        subj_mask = [m.get("subject") == subj for m in metadatas]
        subj_indices = [i for i, m in enumerate(metadatas) if m.get("subject") == subj]
        
        if len(subj_indices) < 2:
            continue
            
        # Get avg similarity for each course to others in same subject
        avg_sims = []
        for i in subj_indices:
            sims = [sim_matrix[i, j] for j in subj_indices if i != j]
            avg_sims.append((i, np.mean(sims) if sims else 0))
        
        avg_sims.sort(key=lambda x: x[1])
        
        print(f"\n  {subj} courses with lowest avg similarity to other {subj} courses:")
        for i, avg in avg_sims[:3]:
            m = metadatas[i]
            print(f"    {avg:.3f}: {m.get('subject', '')} {m.get('catalog_number', '')} - {m.get('title', '')}")


if __name__ == "__main__":
    main()

