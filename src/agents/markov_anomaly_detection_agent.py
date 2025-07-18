"""
Markov Chain-based Anomaly Detection Agent for Time Series RAG Framework

This agent implements sophisticated anomaly detection using:
- Event sequence processing at different time intervals
- Markov chain modeling for sequence representation
- Similarity matrix computation between sequences
- Non-linear dimension reduction for anomaly identification
"""
from typing import Dict, Any, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.manifold import TSNE
import umap
from sklearn.preprocessing import StandardScaler
import warnings
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import networkx as nx
import os

from .base_agent import BaseAgent, AgentResponse
from ..config.config import get_config
from ..utils.logger import get_logger

class MarkovAnomalyDetectionAgent(BaseAgent):
    """
    Advanced anomaly detection agent using Markov chains and sequence similarity.
    
    Features:
    - Event sequence processing at multiple time intervals
    - Markov chain transition matrix computation
    - Sequence similarity calculation using Markov properties
    - Non-linear dimension reduction (UMAP/t-SNE)
    - Anomaly detection based on similarity patterns
    """
    
    def __init__(self, agent_id: str, model_name: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        super().__init__(
            agent_id=agent_id,
            agent_type="markov_anomaly_detection",
            model_name=model_name or get_config().model.anomaly_agent_model,
            config=config
        )
        self.logger = get_logger(f"markov_anomaly_detection_agent_{agent_id}")
        
        # Default time intervals (in minutes)
        self.time_intervals = [5, 30, 60, 720, 1440]  # 5min, 30min, 1hr, 12hr, 24hr
        
        # Markov chain parameters
        self.n_states = 10  # Number of states for discretization
        self.smoothing_factor = 0.01  # Laplace smoothing
        
        # Similarity parameters
        self.similarity_methods = {
            "kl_divergence": self._kl_divergence_similarity,
            "cosine": self._cosine_similarity,
            "euclidean": self._euclidean_similarity,
            "wasserstein": self._wasserstein_similarity
        }
        
        # Dimension reduction parameters
        self.umap_params = {
            "n_components": 2,
            "n_neighbors": 15,
            "min_dist": 0.1,
            "metric": "cosine"
        }
        
        self.tsne_params = {
            "n_components": 2,
            "perplexity": 30,
            "n_iter": 1000,
            "metric": "cosine"
        }
    
    async def _process_request_internal(
        self,
        request: Dict[str, Any],
        context: Optional[Dict[str, Any]],
        relevant_prompts: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Process a Markov chain-based anomaly detection request.
        
        Args:
            request: Dict with 'data' (time series) and detection parameters
            context: Optional context
            relevant_prompts: List of retrieved prompts
            
        Returns:
            Dict with anomaly detection results
        """
        # Extract data
        data = request.get('data')
        if data is None:
            return {"error": "No data provided"}
        
        # Convert to DataFrame
        try:
            df = pd.DataFrame(data)
            if 'ds' not in df.columns or 'y' not in df.columns:
                return {"error": "Data must contain 'ds' (date) and 'y' (value) columns"}
        except Exception as e:
            return {"error": f"Invalid data format: {str(e)}"}
        
        # Get detection parameters
        time_intervals = request.get('time_intervals', self.time_intervals)
        similarity_method = request.get('similarity_method', 'kl_divergence')
        reduction_method = request.get('reduction_method', 'umap')
        anomaly_threshold = request.get('anomaly_threshold', 0.95)
        
        try:
            # Perform Markov chain-based anomaly detection
            results = await self._detect_anomalies_markov(
                df, time_intervals, similarity_method, reduction_method, anomaly_threshold
            )
            
            # Add context information
            results.update({
                "method": "markov_chain",
                "similarity_method": similarity_method,
                "reduction_method": reduction_method,
                "time_intervals": time_intervals,
                "used_prompts": relevant_prompts[:3]
            })
            
            return results
            
        except Exception as e:
            self.logger.error(f"Markov anomaly detection failed: {str(e)}")
            return {"error": f"Anomaly detection failed: {str(e)}"}
    
    async def _detect_anomalies_markov(
        self,
        df: pd.DataFrame,
        time_intervals: List[int],
        similarity_method: str,
        reduction_method: str,
        anomaly_threshold: float
    ) -> Dict[str, Any]:
        """
        Main anomaly detection method using Markov chains.
        
        Args:
            df: Time series DataFrame
            time_intervals: List of time intervals in minutes
            similarity_method: Method for computing similarity
            reduction_method: Dimension reduction method ('umap' or 'tsne')
            anomaly_threshold: Threshold for anomaly detection
            
        Returns:
            Dict with anomaly detection results
        """
        results = {}
        
        # 1. Process event sequences at different time intervals
        sequences = {}
        for interval in time_intervals:
            sequences[interval] = await self._create_event_sequences(df, interval)
        
        # 2. Build Markov chains for each sequence
        markov_chains = {}
        for interval, seq_list in sequences.items():
            markov_chains[interval] = []
            for seq in seq_list:
                markov_chain = await self._build_markov_chain(seq)
                markov_chains[interval].append(markov_chain)
        
        # 3. Compute similarity matrices
        similarity_matrices = {}
        for interval in time_intervals:
            similarity_matrices[interval] = await self._compute_similarity_matrix(
                markov_chains[interval], similarity_method
            )
        
        # 4. Perform dimension reduction
        embeddings = {}
        for interval in time_intervals:
            embeddings[interval] = await self._reduce_dimensions(
                similarity_matrices[interval], reduction_method
            )
        
        # 5. Detect anomalies
        anomalies = await self._detect_anomalies_from_embeddings(
            embeddings, anomaly_threshold
        )
        
        # 6. Generate visualizations
        plots = await self._generate_visualizations(
            sequences, markov_chains, similarity_matrices, embeddings, anomalies
        )
        
        results.update({
            "anomalies": anomalies,
            "sequences": {k: len(v) for k, v in sequences.items()},
            "similarity_matrices": {k: v.shape for k, v in similarity_matrices.items()},
            "embeddings": {k: v.shape for k, v in embeddings.items()},
            "plots": plots,
            "total_anomalies": len(anomalies),
            "confidence": self._calculate_markov_confidence(anomalies, similarity_matrices)
        })
        
        return results
    
    async def _create_event_sequences(
        self, df: pd.DataFrame, interval_minutes: int
    ) -> List[List[int]]:
        """
        Create event sequences from time series data at specified interval.
        
        Args:
            df: Time series DataFrame
            interval_minutes: Time interval in minutes
            
        Returns:
            List of event sequences
        """
        # Ensure datetime column
        df['ds'] = pd.to_datetime(df['ds'])
        
        # Discretize values into states
        values = df['y'].values
        states = pd.cut(values, bins=self.n_states, labels=False, duplicates='drop')
        
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
    
    async def _build_markov_chain(self, sequence: List[int]) -> np.ndarray:
        """
        Build Markov chain transition matrix from sequence.
        
        Args:
            sequence: List of state indices
            
        Returns:
            Transition matrix
        """
        # Initialize transition matrix
        transition_matrix = np.zeros((self.n_states, self.n_states))
        
        # Count transitions
        for i in range(len(sequence) - 1):
            current_state = sequence[i]
            next_state = sequence[i + 1]
            
            if current_state < self.n_states and next_state < self.n_states:
                transition_matrix[current_state, next_state] += 1
        
        # Apply Laplace smoothing
        transition_matrix += self.smoothing_factor
        
        # Normalize to get probabilities
        row_sums = transition_matrix.sum(axis=1)
        transition_matrix = transition_matrix / row_sums[:, np.newaxis]
        
        return transition_matrix
    
    async def _compute_similarity_matrix(
        self, markov_chains: List[np.ndarray], method: str
    ) -> np.ndarray:
        """
        Compute similarity matrix between Markov chains.
        
        Args:
            markov_chains: List of transition matrices
            method: Similarity computation method
            
        Returns:
            Similarity matrix
        """
        n_chains = len(markov_chains)
        similarity_matrix = np.zeros((n_chains, n_chains))
        
        similarity_func = self.similarity_methods.get(method, self._kl_divergence_similarity)
        
        for i in range(n_chains):
            for j in range(i + 1, n_chains):
                similarity = similarity_func(markov_chains[i], markov_chains[j])
                similarity_matrix[i, j] = similarity
                similarity_matrix[j, i] = similarity
        
        # Set diagonal to 1
        np.fill_diagonal(similarity_matrix, 1.0)
        
        return similarity_matrix
    
    def _kl_divergence_similarity(self, matrix1: np.ndarray, matrix2: np.ndarray) -> float:
        """Compute similarity using KL divergence"""
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
    
    def _cosine_similarity(self, matrix1: np.ndarray, matrix2: np.ndarray) -> float:
        """Compute cosine similarity between matrices"""
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
    
    def _euclidean_similarity(self, matrix1: np.ndarray, matrix2: np.ndarray) -> float:
        """Compute similarity using Euclidean distance"""
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
    
    def _wasserstein_similarity(self, matrix1: np.ndarray, matrix2: np.ndarray) -> float:
        """Compute similarity using Wasserstein distance approximation"""
        try:
            # Flatten matrices
            vec1 = matrix1.flatten()
            vec2 = matrix2.flatten()
            
            # Sort for Wasserstein distance approximation
            vec1_sorted = np.sort(vec1)
            vec2_sorted = np.sort(vec2)
            
            # Compute Wasserstein distance
            wasserstein_dist = np.mean(np.abs(vec1_sorted - vec2_sorted))
            
            # Convert to similarity (0 to 1)
            similarity = 1.0 / (1.0 + wasserstein_dist)
            return min(max(similarity, 0.0), 1.0)
        except:
            return 0.0
    
    async def _reduce_dimensions(
        self, similarity_matrix: np.ndarray, method: str
    ) -> np.ndarray:
        """
        Perform non-linear dimension reduction on similarity matrix.
        
        Args:
            similarity_matrix: Similarity matrix
            method: Reduction method ('umap' or 'tsne')
            
        Returns:
            Low-dimensional embeddings
        """
        if method == 'umap':
            reducer = umap.UMAP(**self.umap_params)
        elif method == 'tsne':
            reducer = TSNE(**self.tsne_params)
        else:
            raise ValueError(f"Unsupported reduction method: {method}")
        
        embeddings = reducer.fit_transform(similarity_matrix)
        return embeddings
    
    async def _detect_anomalies_from_embeddings(
        self, embeddings: Dict[int, np.ndarray], threshold: float
    ) -> List[Dict[str, Any]]:
        """
        Detect anomalies from low-dimensional embeddings.
        
        Args:
            embeddings: Dict of embeddings for each time interval
            threshold: Anomaly detection threshold
            
        Returns:
            List of detected anomalies
        """
        anomalies = []
        
        for interval, embedding in embeddings.items():
            # Compute distances from centroid
            centroid = np.mean(embedding, axis=0)
            distances = np.linalg.norm(embedding - centroid, axis=1)
            
            # Find anomalies based on distance threshold
            threshold_distance = np.percentile(distances, threshold * 100)
            anomaly_indices = np.where(distances > threshold_distance)[0]
            
            for idx in anomaly_indices:
                anomalies.append({
                    "interval_minutes": interval,
                    "sequence_index": int(idx),
                    "distance": float(distances[idx]),
                    "threshold_distance": float(threshold_distance),
                    "embedding_coordinates": embedding[idx].tolist(),
                    "method": "markov_embedding"
                })
        
        return anomalies
    
    async def _generate_visualizations(
        self,
        sequences: Dict[int, List[List[int]]],
        markov_chains: Dict[int, List[np.ndarray]],
        similarity_matrices: Dict[int, np.ndarray],
        embeddings: Dict[int, np.ndarray],
        anomalies: List[Dict[str, Any]]
    ) -> Dict[str, str]:
        """
        Generate visualization plots for analysis.
        
        Args:
            sequences: Event sequences for each interval
            markov_chains: Markov chains for each interval
            similarity_matrices: Similarity matrices
            embeddings: Low-dimensional embeddings
            anomalies: Detected anomalies
            
        Returns:
            Dict with plot file paths
        """
        plots = {}
        
        try:
            # Create plots directory
            plots_dir = "models/plots"
            os.makedirs(plots_dir, exist_ok=True)
            
            # 1. Similarity matrix heatmap
            for interval, sim_matrix in similarity_matrices.items():
                plt.figure(figsize=(10, 8))
                sns.heatmap(sim_matrix, cmap='viridis', annot=False)
                plt.title(f'Similarity Matrix - {interval}min intervals')
                plt.xlabel('Sequence Index')
                plt.ylabel('Sequence Index')
                
                plot_path = f"{plots_dir}/similarity_matrix_{interval}min.png"
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                plots[f"similarity_matrix_{interval}min"] = plot_path
            
            # 2. Embedding scatter plot
            for interval, embedding in embeddings.items():
                plt.figure(figsize=(10, 8))
                
                # Plot all points
                plt.scatter(embedding[:, 0], embedding[:, 1], alpha=0.6, s=50)
                
                # Highlight anomalies
                anomaly_indices = [a['sequence_index'] for a in anomalies if a['interval_minutes'] == interval]
                if anomaly_indices:
                    plt.scatter(
                        embedding[anomaly_indices, 0],
                        embedding[anomaly_indices, 1],
                        color='red', s=100, alpha=0.8, label='Anomalies'
                    )
                
                plt.title(f'Embedding Space - {interval}min intervals')
                plt.xlabel('Component 1')
                plt.ylabel('Component 2')
                plt.legend()
                
                plot_path = f"{plots_dir}/embedding_{interval}min.png"
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                plots[f"embedding_{interval}min"] = plot_path
            
            # 3. Sequence length distribution
            plt.figure(figsize=(12, 8))
            for interval, seq_list in sequences.items():
                lengths = [len(seq) for seq in seq_list]
                plt.hist(lengths, alpha=0.7, label=f'{interval}min', bins=20)
            
            plt.title('Sequence Length Distribution')
            plt.xlabel('Sequence Length')
            plt.ylabel('Frequency')
            plt.legend()
            
            plot_path = f"{plots_dir}/sequence_length_distribution.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            plots["sequence_length_distribution"] = plot_path
            
        except Exception as e:
            self.logger.error(f"Visualization generation failed: {str(e)}")
        
        return plots
    
    def _calculate_markov_confidence(
        self, anomalies: List[Dict[str, Any]], similarity_matrices: Dict[int, np.ndarray]
    ) -> float:
        """
        Calculate confidence score for anomaly detection.
        
        Args:
            anomalies: Detected anomalies
            similarity_matrices: Similarity matrices
            
        Returns:
            Confidence score (0 to 1)
        """
        if not anomalies:
            return 0.0
        
        # Calculate average similarity for each interval
        avg_similarities = {}
        for interval, sim_matrix in similarity_matrices.items():
            # Exclude diagonal elements
            mask = ~np.eye(sim_matrix.shape[0], dtype=bool)
            avg_similarities[interval] = np.mean(sim_matrix[mask])
        
        # Calculate confidence based on similarity consistency
        avg_similarity = np.mean(list(avg_similarities.values()))
        
        # Higher similarity = higher confidence
        confidence = min(avg_similarity, 1.0)
        
        return confidence 