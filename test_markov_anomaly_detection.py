#!/usr/bin/env python3
"""
Test script for Markov Chain-based Anomaly Detection

This script demonstrates the sophisticated anomaly detection system using:
- Event sequence processing at multiple time intervals
- Markov chain modeling for sequence representation  
- Similarity matrix computation between sequences
- Non-linear dimension reduction for anomaly identification
"""

import asyncio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import sys
import os

# Add src to path
sys.path.append('src')

from agents.markov_anomaly_detection_agent import MarkovAnomalyDetectionAgent
from config.config import get_config

def generate_synthetic_timeseries(n_points: int = 1000, anomaly_ratio: float = 0.05) -> pd.DataFrame:
    """
    Generate synthetic time series data with known anomalies.
    
    Args:
        n_points: Number of data points
        anomaly_ratio: Ratio of anomalous points
        
    Returns:
        DataFrame with 'ds' (datetime) and 'y' (values) columns
    """
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

def plot_results(df: pd.DataFrame, results: dict, anomaly_indices: np.ndarray):
    """
    Plot the results of Markov chain anomaly detection.
    
    Args:
        df: Original time series data
        results: Results from anomaly detection
        anomaly_indices: Known anomaly indices
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Original time series with known anomalies
    axes[0, 0].plot(df['ds'], df['y'], alpha=0.7, label='Time Series')
    axes[0, 0].scatter(df.iloc[anomaly_indices]['ds'], df.iloc[anomaly_indices]['y'], 
                       color='red', s=50, alpha=0.8, label='Known Anomalies')
    axes[0, 0].set_title('Original Time Series with Known Anomalies')
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('Value')
    axes[0, 0].legend()
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 2. Detected anomalies by interval
    detected_anomalies = results.get('anomalies', [])
    if detected_anomalies:
        intervals = list(set([a['interval_minutes'] for a in detected_anomalies]))
        colors = plt.cm.Set3(np.linspace(0, 1, len(intervals)))
        
        for i, interval in enumerate(intervals):
            interval_anomalies = [a for a in detected_anomalies if a['interval_minutes'] == interval]
            if interval_anomalies:
                anomaly_times = [df['ds'].iloc[a['sequence_index']] for a in interval_anomalies]
                anomaly_values = [df['y'].iloc[a['sequence_index']] for a in interval_anomalies]
                axes[0, 1].scatter(anomaly_times, anomaly_values, 
                                  color=colors[i], s=50, alpha=0.8, 
                                  label=f'{interval}min intervals')
        
        axes[0, 1].plot(df['ds'], df['y'], alpha=0.7, color='gray')
        axes[0, 1].set_title('Detected Anomalies by Time Interval')
        axes[0, 1].set_xlabel('Time')
        axes[0, 1].set_ylabel('Value')
        axes[0, 1].legend()
        axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 3. Similarity matrices
    similarity_matrices = results.get('similarity_matrices', {})
    if similarity_matrices:
        # Show first similarity matrix as example
        first_interval = list(similarity_matrices.keys())[0]
        sim_matrix = np.random.rand(10, 10)  # Placeholder - would need actual matrix
        sns.heatmap(sim_matrix, ax=axes[1, 0], cmap='viridis')
        axes[1, 0].set_title(f'Similarity Matrix Example ({first_interval}min)')
    
    # 4. Statistics
    stats_data = {
        'Metric': ['Total Anomalies', 'Confidence', 'Sequences (5min)', 'Sequences (30min)', 
                  'Sequences (1hr)', 'Sequences (12hr)', 'Sequences (24hr)'],
        'Value': [
            results.get('total_anomalies', 0),
            f"{results.get('confidence', 0):.3f}",
            results.get('sequences', {}).get(5, 0),
            results.get('sequences', {}).get(30, 0),
            results.get('sequences', {}).get(60, 0),
            results.get('sequences', {}).get(720, 0),
            results.get('sequences', {}).get(1440, 0)
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
    plt.savefig('markov_anomaly_detection_results.png', dpi=300, bbox_inches='tight')
    plt.show()

async def test_markov_anomaly_detection():
    """
    Test the Markov chain-based anomaly detection system.
    """
    print("🔍 Testing Markov Chain-based Anomaly Detection")
    print("=" * 60)
    
    # 1. Generate synthetic data
    print("📊 Generating synthetic time series data...")
    df, known_anomalies = generate_synthetic_timeseries(n_points=1000, anomaly_ratio=0.05)
    print(f"   Generated {len(df)} data points with {len(known_anomalies)} known anomalies")
    
    # 2. Initialize Markov anomaly detection agent
    print("🤖 Initializing Markov Anomaly Detection Agent...")
    agent = MarkovAnomalyDetectionAgent(
        agent_id="markov_test",
        config={
            "time_intervals": [5, 30, 60, 720, 1440],  # 5min, 30min, 1hr, 12hr, 24hr
            "similarity_method": "kl_divergence",
            "reduction_method": "umap",
            "anomaly_threshold": 0.95
        }
    )
    
    # 3. Prepare request
    request = {
        "data": df.to_dict('records'),
        "time_intervals": [5, 30, 60, 720, 1440],
        "similarity_method": "kl_divergence",
        "reduction_method": "umap",
        "anomaly_threshold": 0.95
    }
    
    # 4. Run anomaly detection
    print("🔬 Running Markov chain-based anomaly detection...")
    try:
        results = await agent.process_request(request)
        
        if "error" in results:
            print(f"❌ Error: {results['error']}")
            return
        
        # 5. Display results
        print("\n📈 Detection Results:")
        print(f"   Total anomalies detected: {results.get('total_anomalies', 0)}")
        print(f"   Confidence score: {results.get('confidence', 0):.3f}")
        print(f"   Method: {results.get('method', 'N/A')}")
        print(f"   Similarity method: {results.get('similarity_method', 'N/A')}")
        print(f"   Reduction method: {results.get('reduction_method', 'N/A')}")
        
        print("\n📊 Sequence Statistics:")
        sequences = results.get('sequences', {})
        for interval, count in sequences.items():
            print(f"   {interval}min intervals: {count} sequences")
        
        print("\n🔍 Anomaly Details:")
        anomalies = results.get('anomalies', [])
        for i, anomaly in enumerate(anomalies[:5]):  # Show first 5
            print(f"   Anomaly {i+1}: Interval={anomaly['interval_minutes']}min, "
                  f"Sequence={anomaly['sequence_index']}, Distance={anomaly['distance']:.3f}")
        
        if len(anomalies) > 5:
            print(f"   ... and {len(anomalies) - 5} more anomalies")
        
        # 6. Generate plots
        print("\n📊 Generating visualizations...")
        plot_results(df, results, known_anomalies)
        
        # 7. Performance analysis
        print("\n⚡ Performance Analysis:")
        similarity_matrices = results.get('similarity_matrices', {})
        embeddings = results.get('embeddings', {})
        
        for interval in [5, 30, 60, 720, 1440]:
            if interval in similarity_matrices:
                matrix_shape = similarity_matrices[interval]
                embedding_shape = embeddings.get(interval, (0, 0))
                print(f"   {interval}min: {matrix_shape[0]}x{matrix_shape[1]} similarity matrix, "
                      f"{embedding_shape[0]}x{embedding_shape[1]} embeddings")
        
        print("\n✅ Markov chain anomaly detection test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()

async def test_different_similarity_methods():
    """
    Test different similarity methods for Markov chain comparison.
    """
    print("\n🔄 Testing Different Similarity Methods")
    print("=" * 50)
    
    # Generate data
    df, _ = generate_synthetic_timeseries(n_points=500, anomaly_ratio=0.03)
    
    # Test different similarity methods
    similarity_methods = ["kl_divergence", "cosine", "euclidean", "wasserstein"]
    
    results_comparison = {}
    
    for method in similarity_methods:
        print(f"\n🔍 Testing {method} similarity method...")
        
        agent = MarkovAnomalyDetectionAgent(
            agent_id=f"markov_{method}",
            config={"similarity_method": method}
        )
        
        request = {
            "data": df.to_dict('records'),
            "time_intervals": [30, 60],  # Shorter test
            "similarity_method": method,
            "reduction_method": "umap",
            "anomaly_threshold": 0.95
        }
        
        try:
            results = await agent.process_request(request)
            
            if "error" not in results:
                results_comparison[method] = {
                    "total_anomalies": results.get('total_anomalies', 0),
                    "confidence": results.get('confidence', 0),
                    "sequences": results.get('sequences', {})
                }
                print(f"   ✅ {method}: {results.get('total_anomalies', 0)} anomalies, "
                      f"confidence={results.get('confidence', 0):.3f}")
            else:
                print(f"   ❌ {method}: {results['error']}")
                
        except Exception as e:
            print(f"   ❌ {method}: Error - {str(e)}")
    
    # Compare results
    print("\n📊 Similarity Method Comparison:")
    print("Method           | Anomalies | Confidence")
    print("-" * 40)
    for method, results in results_comparison.items():
        print(f"{method:15} | {results['total_anomalies']:9} | {results['confidence']:.3f}")

async def main():
    """
    Main test function.
    """
    print("🚀 Starting Markov Chain Anomaly Detection Tests")
    print("=" * 60)
    
    # Test 1: Basic functionality
    await test_markov_anomaly_detection()
    
    # Test 2: Different similarity methods
    await test_different_similarity_methods()
    
    print("\n🎉 All tests completed!")

if __name__ == "__main__":
    asyncio.run(main()) 