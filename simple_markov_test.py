#!/usr/bin/env python3
"""
Simplified Markov Chain Anomaly Detection Demo

This script demonstrates the core concepts of Markov chain-based anomaly detection
without requiring the full framework setup.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.manifold import TSNE
import umap
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

def generate_synthetic_timeseries(n_points: int = 1000, anomaly_ratio: float = 0.05) -> pd.DataFrame:
    """Generate synthetic time series data with known anomalies."""
    # Generate base time series with trend and seasonality
    dates = pd.date_range(start='2024-01-01', periods=n_points, freq='1H')
    
    # Base signal: trend + seasonality + noise
    trend = np.linspace(0, 10, n_points)
    seasonality = 5 * np.sin(2 * np.pi * np.arange(n_points) / 24)  # Daily seasonality
    weekly_seasonality = 2 * np.sin(2 * np.pi * np.arange(n_points) / (24 * 7))  # Weekly seasonality
    noise = np.random.normal(0, 1, n_points)
    
    base_signal = trend + seasonality + weekly_seasonality + noise
    
    # Add anomalies
    n_anomalies = int(n_points * anomaly_ratio)
    anomaly_indices = np.random.choice(n_points, n_anomalies, replace=False)
    
    # Different types of anomalies
    for idx in anomaly_indices:
        anomaly_type = np.random.choice(['spike', 'drop', 'level_shift'])
        
        if anomaly_type == 'spike':
            base_signal[idx] += np.random.uniform(5, 15)
        elif anomaly_type == 'drop':
            base_signal[idx] -= np.random.uniform(5, 15)
        else:  # level_shift
            shift_duration = np.random.randint(5, 20)
            shift_magnitude = np.random.uniform(3, 8)
            end_idx = min(idx + shift_duration, n_points)
            base_signal[idx:end_idx] += shift_magnitude
    
    # Create DataFrame
    df = pd.DataFrame({
        'ds': dates,
        'y': base_signal
    })
    
    return df, anomaly_indices

def create_event_sequences(df: pd.DataFrame, interval_minutes: int, n_states: int = 10) -> list:
    """Create event sequences from time series data at specified interval."""
    # Ensure datetime column
    df['ds'] = pd.to_datetime(df['ds'])
    
    # Discretize values into states
    values = df['y'].values
    states = pd.cut(values, bins=n_states, labels=False, duplicates='drop')
    
    # Group by time intervals
    interval_td = timedelta(minutes=interval_minutes)
    sequences = []
    
    current_sequence = []
    current_time = df['ds'].iloc[0]
    
    for idx, row in df.iterrows():
        time_diff = row['ds'] - current_time
        
        if time_diff >= interval_td:
            # Start new sequence
            if current_sequence:
                sequences.append(current_sequence)
            current_sequence = [states[idx]]
            current_time = row['ds']
        else:
            # Add to current sequence
            current_sequence.append(states[idx])
    
    # Add final sequence
    if current_sequence:
        sequences.append(current_sequence)
    
    return sequences

def build_markov_chain(sequence: list, n_states: int = 10, smoothing_factor: float = 0.01) -> np.ndarray:
    """Build Markov chain transition matrix from sequence."""
    # Initialize transition matrix
    transition_matrix = np.zeros((n_states, n_states))
    
    # Count transitions
    for i in range(len(sequence) - 1):
        current_state = sequence[i]
        next_state = sequence[i + 1]
        
        if current_state < n_states and next_state < n_states:
            transition_matrix[current_state, next_state] += 1
    
    # Apply Laplace smoothing
    transition_matrix += smoothing_factor
    
    # Normalize to get probabilities
    row_sums = transition_matrix.sum(axis=1)
    transition_matrix = transition_matrix / row_sums[:, np.newaxis]
    
    return transition_matrix

def compute_similarity_matrix(markov_chains: list, method: str = "kl_divergence") -> np.ndarray:
    """Compute similarity matrix between Markov chains."""
    n_chains = len(markov_chains)
    similarity_matrix = np.zeros((n_chains, n_chains))
    
    for i in range(n_chains):
        for j in range(i + 1, n_chains):
            if method == "kl_divergence":
                similarity = kl_divergence_similarity(markov_chains[i], markov_chains[j])
            elif method == "cosine":
                similarity = cosine_similarity(markov_chains[i], markov_chains[j])
            elif method == "euclidean":
                similarity = euclidean_similarity(markov_chains[i], markov_chains[j])
            else:
                similarity = kl_divergence_similarity(markov_chains[i], markov_chains[j])
            
            similarity_matrix[i, j] = similarity
            similarity_matrix[j, i] = similarity
    
    # Set diagonal to 1
    np.fill_diagonal(similarity_matrix, 1.0)
    
    return similarity_matrix

def kl_divergence_similarity(matrix1: np.ndarray, matrix2: np.ndarray) -> float:
    """Compute similarity using KL divergence."""
    try:
        # Flatten matrices and add small epsilon to avoid log(0)
        p = matrix1.flatten() + 1e-10
        q = matrix2.flatten() + 1e-10
        
        # Normalize
        p = p / p.sum()
        q = q / q.sum()
        
        # Compute KL divergence
        kl_div = np.sum(p * np.log(p / q))
        
        # Convert to similarity (0 to 1)
        similarity = np.exp(-kl_div)
        return min(max(similarity, 0.0), 1.0)
    except:
        return 0.0

def cosine_similarity(matrix1: np.ndarray, matrix2: np.ndarray) -> float:
    """Compute cosine similarity between matrices."""
    try:
        # Flatten matrices
        vec1 = matrix1.flatten()
        vec2 = matrix2.flatten()
        
        # Compute cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        return min(max(similarity, 0.0), 1.0)
    except:
        return 0.0

def euclidean_similarity(matrix1: np.ndarray, matrix2: np.ndarray) -> float:
    """Compute similarity using Euclidean distance."""
    try:
        # Flatten matrices
        vec1 = matrix1.flatten()
        vec2 = matrix2.flatten()
        
        # Compute Euclidean distance
        distance = np.linalg.norm(vec1 - vec2)
        
        # Convert to similarity (0 to 1)
        similarity = 1.0 / (1.0 + distance)
        return min(max(similarity, 0.0), 1.0)
    except:
        return 0.0

def reduce_dimensions(similarity_matrix: np.ndarray, method: str = "umap") -> np.ndarray:
    """Perform non-linear dimension reduction on similarity matrix."""
    if method == "umap":
        reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, metric="cosine")
    elif method == "tsne":
        reducer = TSNE(n_components=2, perplexity=30, n_iter=1000, metric="cosine")
    else:
        raise ValueError(f"Unsupported reduction method: {method}")
    
    embeddings = reducer.fit_transform(similarity_matrix)
    return embeddings

def detect_anomalies_from_embeddings(embeddings: np.ndarray, threshold: float = 0.95) -> list:
    """Detect anomalies from low-dimensional embeddings."""
    # Compute distances from centroid
    centroid = np.mean(embeddings, axis=0)
    distances = np.linalg.norm(embeddings - centroid, axis=1)
    
    # Find anomalies based on distance threshold
    threshold_distance = np.percentile(distances, threshold * 100)
    anomaly_indices = np.where(distances > threshold_distance)[0]
    
    anomalies = []
    for idx in anomaly_indices:
        anomalies.append({
            "sequence_index": int(idx),
            "distance": float(distances[idx]),
            "threshold_distance": float(threshold_distance),
            "embedding_coordinates": embeddings[idx].tolist()
        })
    
    return anomalies

def plot_results(df: pd.DataFrame, results: dict, anomaly_indices: np.ndarray, interval: int):
    """Plot the results of Markov chain anomaly detection."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Original time series with known anomalies
    axes[0, 0].plot(df['ds'], df['y'], alpha=0.7, label='Time Series')
    axes[0, 0].scatter(df.iloc[anomaly_indices]['ds'], df.iloc[anomaly_indices]['y'], 
                       color='red', s=50, alpha=0.8, label='Known Anomalies')
    axes[0, 0].set_title(f'Original Time Series with Known Anomalies')
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('Value')
    axes[0, 0].legend()
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 2. Similarity matrix heatmap
    if 'similarity_matrix' in results:
        sns.heatmap(results['similarity_matrix'], ax=axes[0, 1], cmap='viridis')
        axes[0, 1].set_title(f'Similarity Matrix - {interval}min intervals')
        axes[0, 1].set_xlabel('Sequence Index')
        axes[0, 1].set_ylabel('Sequence Index')
    
    # 3. Embedding scatter plot
    if 'embeddings' in results:
        embeddings = results['embeddings']
        axes[1, 0].scatter(embeddings[:, 0], embeddings[:, 1], alpha=0.6, s=50)
        
        # Highlight anomalies
        anomalies = results.get('anomalies', [])
        if anomalies:
            anomaly_indices = [a['sequence_index'] for a in anomalies]
            axes[1, 0].scatter(
                embeddings[anomaly_indices, 0],
                embeddings[anomaly_indices, 1],
                color='red', s=100, alpha=0.8, label='Anomalies'
            )
        
        axes[1, 0].set_title(f'Embedding Space - {interval}min intervals')
        axes[1, 0].set_xlabel('Component 1')
        axes[1, 0].set_ylabel('Component 2')
        axes[1, 0].legend()
    
    # 4. Statistics
    stats_data = {
        'Metric': ['Total Sequences', 'Detected Anomalies', 'Anomaly Ratio', 'Avg Similarity'],
        'Value': [
            results.get('n_sequences', 0),
            len(results.get('anomalies', [])),
            f"{len(results.get('anomalies', [])) / max(results.get('n_sequences', 1), 1):.3f}",
            f"{results.get('avg_similarity', 0):.3f}"
        ]
    }
    
    stats_df = pd.DataFrame(stats_data)
    axes[1, 1].axis('tight')
    axes[1, 1].axis('off')
    table = axes[1, 1].table(cellText=stats_df.values, colLabels=stats_df.columns, 
                             cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    axes[1, 1].set_title('Detection Statistics')
    
    plt.tight_layout()
    plt.savefig(f'markov_anomaly_detection_{interval}min.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main demonstration function."""
    print("🔍 Markov Chain-based Anomaly Detection Demo")
    print("=" * 60)
    
    # 1. Generate synthetic data
    print("📊 Generating synthetic time series data...")
    df, known_anomalies = generate_synthetic_timeseries(n_points=1000, anomaly_ratio=0.05)
    print(f"   Generated {len(df)} data points with {len(known_anomalies)} known anomalies")
    
    # 2. Test different time intervals
    time_intervals = [30, 60, 720]  # 30min, 1hr, 12hr
    similarity_methods = ["kl_divergence", "cosine", "euclidean"]
    
    for interval in time_intervals:
        print(f"\n🔬 Processing {interval}min intervals...")
        
        # Create event sequences
        sequences = create_event_sequences(df, interval, n_states=10)
        print(f"   Created {len(sequences)} sequences")
        
        # Build Markov chains
        markov_chains = []
        for seq in sequences:
            if len(seq) > 1:  # Need at least 2 points for transitions
                markov_chain = build_markov_chain(seq, n_states=10)
                markov_chains.append(markov_chain)
        
        print(f"   Built {len(markov_chains)} Markov chains")
        
        if len(markov_chains) < 2:
            print(f"   Skipping {interval}min - insufficient sequences")
            continue
        
        # Test different similarity methods
        for method in similarity_methods:
            print(f"   Testing {method} similarity...")
            
            # Compute similarity matrix
            similarity_matrix = compute_similarity_matrix(markov_chains, method)
            
            # Reduce dimensions
            embeddings = reduce_dimensions(similarity_matrix, method="umap")
            
            # Detect anomalies
            anomalies = detect_anomalies_from_embeddings(embeddings, threshold=0.95)
            
            # Calculate statistics
            avg_similarity = np.mean(similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)])
            
            results = {
                "n_sequences": len(markov_chains),
                "similarity_matrix": similarity_matrix,
                "embeddings": embeddings,
                "anomalies": anomalies,
                "avg_similarity": avg_similarity,
                "method": method
            }
            
            print(f"     ✅ {method}: {len(anomalies)} anomalies, "
                  f"avg_similarity={avg_similarity:.3f}")
            
            # Plot results for the first method
            if method == "kl_divergence":
                plot_results(df, results, known_anomalies, interval)
    
    print("\n🎉 Markov chain anomaly detection demo completed!")
    print("\n📊 Summary:")
    print("   - Event sequences created at multiple time intervals")
    print("   - Markov chain transition matrices computed")
    print("   - Similarity matrices calculated using multiple metrics")
    print("   - Dimension reduction applied using UMAP")
    print("   - Anomalies detected based on embedding distances")
    print("   - Visualizations generated for analysis")

if __name__ == "__main__":
    main() 