#!/usr/bin/env python3
"""
Improved Markov Chain Anomaly Detection Demo

This script demonstrates the sophisticated anomaly detection system using:
- Event sequence processing at different time intervals
- Markov chain modeling for sequence representation
- Similarity matrix computation between sequences
- Non-linear dimension reduction for anomaly identification
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

def create_event_sequences_improved(df: pd.DataFrame, interval_minutes: int, n_states: int = 10) -> list:
    """Create event sequences from time series data at specified interval."""
    # Ensure datetime column
    df['ds'] = pd.to_datetime(df['ds'])
    
    # Discretize values into states
    values = df['y'].values
    states = pd.cut(values, bins=n_states, labels=False, duplicates='drop')
    
    # Group by time intervals
    interval_td = timedelta(minutes=interval_minutes)
    sequences = []
    
    # Group data points by time intervals
    df['interval'] = (df['ds'] - df['ds'].iloc[0]).dt.total_seconds() // (interval_minutes * 60)
    
    # Create sequences for each interval
    for interval_id in df['interval'].unique():
        interval_data = df[df['interval'] == interval_id]
        if len(interval_data) > 1:  # Need at least 2 points for transitions
            sequence = states[interval_data.index].tolist()
            sequences.append(sequence)
    
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
    """Compute similarity using cosine similarity."""
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

def plot_comprehensive_results(df: pd.DataFrame, results_dict: dict, anomaly_indices: np.ndarray):
    """Plot comprehensive results of Markov chain anomaly detection."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Original time series with known anomalies
    axes[0, 0].plot(df['ds'], df['y'], alpha=0.7, label='Time Series')
    axes[0, 0].scatter(df.iloc[anomaly_indices]['ds'], df.iloc[anomaly_indices]['y'], 
                       color='red', s=50, alpha=0.8, label='Known Anomalies')
    axes[0, 0].set_title('Original Time Series with Known Anomalies')
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('Value')
    axes[0, 0].legend()
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 2. Similarity matrix heatmap (first interval)
    first_interval = list(results_dict.keys())[0]
    if 'similarity_matrix' in results_dict[first_interval]:
        sns.heatmap(results_dict[first_interval]['similarity_matrix'], 
                   ax=axes[0, 1], cmap='viridis')
        axes[0, 1].set_title(f'Similarity Matrix - {first_interval}min intervals')
        axes[0, 1].set_xlabel('Sequence Index')
        axes[0, 1].set_ylabel('Sequence Index')
    
    # 3. Embedding scatter plot (first interval)
    if 'embeddings' in results_dict[first_interval]:
        embeddings = results_dict[first_interval]['embeddings']
        axes[0, 2].scatter(embeddings[:, 0], embeddings[:, 1], alpha=0.6, s=50)
        
        # Highlight anomalies
        anomalies = results_dict[first_interval].get('anomalies', [])
        if anomalies:
            anomaly_indices_emb = [a['sequence_index'] for a in anomalies]
            axes[0, 2].scatter(
                embeddings[anomaly_indices_emb, 0],
                embeddings[anomaly_indices_emb, 1],
                color='red', s=100, alpha=0.8, label='Anomalies'
            )
        
        axes[0, 2].set_title(f'Embedding Space - {first_interval}min intervals')
        axes[0, 2].set_xlabel('Component 1')
        axes[0, 2].set_ylabel('Component 2')
        axes[0, 2].legend()
    
    # 4. Comparison of similarity methods
    methods = ['kl_divergence', 'cosine', 'euclidean']
    avg_similarities = []
    anomaly_counts = []
    
    for method in methods:
        if method in results_dict[first_interval]:
            avg_sim = results_dict[first_interval][method].get('avg_similarity', 0)
            anomaly_count = len(results_dict[first_interval][method].get('anomalies', []))
            avg_similarities.append(avg_sim)
            anomaly_counts.append(anomaly_count)
    
    x = np.arange(len(methods))
    width = 0.35
    
    axes[1, 0].bar(x - width/2, avg_similarities, width, label='Avg Similarity')
    axes[1, 0].set_xlabel('Similarity Method')
    axes[1, 0].set_ylabel('Average Similarity')
    axes[1, 0].set_title('Similarity Method Comparison')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(methods)
    axes[1, 0].legend()
    
    # 5. Anomaly counts by method
    axes[1, 1].bar(x, anomaly_counts, label='Detected Anomalies')
    axes[1, 1].set_xlabel('Similarity Method')
    axes[1, 1].set_ylabel('Number of Anomalies')
    axes[1, 1].set_title('Anomaly Detection by Method')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(methods)
    axes[1, 1].legend()
    
    # 6. Summary statistics
    stats_data = []
    for interval, results in results_dict.items():
        for method, method_results in results.items():
            if isinstance(method_results, dict):
                stats_data.append({
                    'Interval': f'{interval}min',
                    'Method': method,
                    'Sequences': method_results.get('n_sequences', 0),
                    'Anomalies': len(method_results.get('anomalies', [])),
                    'Avg Similarity': f"{method_results.get('avg_similarity', 0):.3f}"
                })
    
    if stats_data:
        stats_df = pd.DataFrame(stats_data)
        axes[1, 2].axis('tight')
        axes[1, 2].axis('off')
        table = axes[1, 2].table(cellText=stats_df.values, colLabels=stats_df.columns, 
                                 cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.2, 1.5)
        axes[1, 2].set_title('Detection Statistics')
    
    plt.tight_layout()
    plt.savefig('markov_anomaly_detection_comprehensive.png', dpi=300, bbox_inches='tight')
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
    time_intervals = [60, 120, 720]  # 1hr, 2hr, 12hr
    similarity_methods = ["kl_divergence", "cosine", "euclidean"]
    
    results_dict = {}
    
    for interval in time_intervals:
        print(f"\n🔬 Processing {interval}min intervals...")
        
        # Create event sequences
        sequences = create_event_sequences_improved(df, interval, n_states=10)
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
        
        results_dict[interval] = {}
        
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
            
            results_dict[interval][method] = {
                "n_sequences": len(markov_chains),
                "similarity_matrix": similarity_matrix,
                "embeddings": embeddings,
                "anomalies": anomalies,
                "avg_similarity": avg_similarity
            }
            
            print(f"     ✅ {method}: {len(anomalies)} anomalies, "
                  f"avg_similarity={avg_similarity:.3f}")
    
    # 3. Generate comprehensive visualization
    print("\n📊 Generating comprehensive visualizations...")
    plot_comprehensive_results(df, results_dict, known_anomalies)
    
    # 4. Print summary
    print("\n📈 Summary Results:")
    print("Interval | Method        | Sequences | Anomalies | Avg Similarity")
    print("-" * 65)
    
    for interval, methods in results_dict.items():
        for method, results in methods.items():
            if isinstance(results, dict):
                print(f"{interval:8}min | {method:12} | {results['n_sequences']:9} | "
                      f"{len(results['anomalies']):9} | {results['avg_similarity']:.3f}")
    
    print("\n🎉 Markov chain anomaly detection demo completed!")
    print("\n📊 Key Features Demonstrated:")
    print("   ✅ Event sequence processing at multiple time intervals")
    print("   ✅ Markov chain transition matrix computation")
    print("   ✅ Multiple similarity metrics (KL divergence, cosine, euclidean)")
    print("   ✅ Non-linear dimension reduction using UMAP")
    print("   ✅ Anomaly detection based on embedding distances")
    print("   ✅ Comprehensive visualizations and analysis")
    print("\n🔬 Scientific Approach:")
    print("   - Uses Markov chains to model temporal dependencies")
    print("   - Compares sequences using information-theoretic measures")
    print("   - Reduces high-dimensional similarity matrices to 2D")
    print("   - Identifies anomalies as outliers in reduced space")
    print("   - Provides interpretable results with confidence scores")

if __name__ == "__main__":
    main() 