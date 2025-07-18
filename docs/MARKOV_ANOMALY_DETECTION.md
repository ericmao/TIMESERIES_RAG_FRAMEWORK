# Markov Chain-based Anomaly Detection

## Overview

The Markov Chain-based Anomaly Detection system is a sophisticated approach to identifying anomalies in time series data using event sequences, Markov chain modeling, and non-linear dimension reduction. This system is designed to detect anomalies at multiple time scales and provides rich visualizations and analysis capabilities.

## Concept

### Core Idea

The system works by:

1. **Event Sequence Processing**: Converting time series data into event sequences at different time intervals (5min, 30min, 60min, 12hr, 24hr)
2. **Markov Chain Modeling**: Building transition matrices that describe the probability of moving between states in each sequence
3. **Similarity Calculation**: Computing similarity between sequences using various distance metrics
4. **Dimension Reduction**: Using UMAP or t-SNE to reduce high-dimensional similarity matrices to 2D
5. **Anomaly Detection**: Identifying anomalous time intervals based on their position in the reduced space

### Mathematical Foundation

#### Markov Chain Transition Matrix

For each sequence `S = [s₁, s₂, ..., sₙ]`, we build a transition matrix `T` where:

```
T[i,j] = P(sₜ₊₁ = j | sₜ = i)
```

The transition matrix captures the probability of moving from state `i` to state `j` in the sequence.

#### Similarity Metrics

We support four similarity methods:

1. **KL Divergence**: Measures the difference between two probability distributions
   ```
   D_KL(P||Q) = Σ P(i) log(P(i)/Q(i))
   Similarity = exp(-D_KL)
   ```

2. **Cosine Similarity**: Measures the angle between two vectors
   ```
   cos(θ) = (A·B) / (||A|| ||B||)
   ```

3. **Euclidean Distance**: Standard L2 distance
   ```
   d(A,B) = √(Σ(Aᵢ - Bᵢ)²)
   Similarity = 1 / (1 + d)
   ```

4. **Wasserstein Distance**: Earth mover's distance approximation
   ```
   W(A,B) = Σ|A_sorted[i] - B_sorted[i]|
   Similarity = 1 / (1 + W)
   ```

## Architecture

### Components

1. **MarkovAnomalyDetectionAgent**: Main agent class
2. **Event Sequence Processor**: Converts time series to sequences
3. **Markov Chain Builder**: Creates transition matrices
4. **Similarity Calculator**: Computes similarity matrices
5. **Dimension Reducer**: UMAP/t-SNE for visualization
6. **Anomaly Detector**: Identifies anomalies in reduced space
7. **Visualization Generator**: Creates plots and charts

### Data Flow

```
Time Series Data
       ↓
Event Sequences (multiple intervals)
       ↓
Markov Chain Transition Matrices
       ↓
Similarity Matrices
       ↓
Dimension Reduction (UMAP/t-SNE)
       ↓
Anomaly Detection
       ↓
Results + Visualizations
```

## Implementation Details

### Configuration

The system is highly configurable through the `TimeSeriesConfig`:

```python
# Time intervals for sequence processing
markov_time_intervals = [5, 30, 60, 720, 1440]  # minutes

# Markov chain parameters
markov_n_states = 10  # Number of discretization states
markov_smoothing_factor = 0.01  # Laplace smoothing

# Similarity and reduction methods
markov_similarity_method = "kl_divergence"  # or cosine, euclidean, wasserstein
markov_reduction_method = "umap"  # or tsne
```

### Key Methods

#### Event Sequence Creation

```python
async def _create_event_sequences(self, df: pd.DataFrame, interval_minutes: int) -> List[List[int]]:
    """
    Convert time series to event sequences at specified interval.
    
    Steps:
    1. Discretize values into states (0-9)
    2. Group by time intervals
    3. Create sequences of state transitions
    """
```

#### Markov Chain Building

```python
async def _build_markov_chain(self, sequence: List[int]) -> np.ndarray:
    """
    Build transition matrix from sequence.
    
    Steps:
    1. Count state transitions
    2. Apply Laplace smoothing
    3. Normalize to probabilities
    """
```

#### Similarity Computation

```python
async def _compute_similarity_matrix(self, markov_chains: List[np.ndarray], method: str) -> np.ndarray:
    """
    Compute similarity matrix between all pairs of Markov chains.
    
    Returns: NxN similarity matrix where N = number of sequences
    """
```

## Usage

### Basic Usage

```python
from agents.markov_anomaly_detection_agent import MarkovAnomalyDetectionAgent

# Initialize agent
agent = MarkovAnomalyDetectionAgent(
    agent_id="markov_detector",
    config={
        "time_intervals": [5, 30, 60, 720, 1440],
        "similarity_method": "kl_divergence",
        "reduction_method": "umap",
        "anomaly_threshold": 0.95
    }
)

# Prepare data
data = {
    "data": df.to_dict('records'),  # DataFrame with 'ds' and 'y' columns
    "time_intervals": [5, 30, 60, 720, 1440],
    "similarity_method": "kl_divergence",
    "reduction_method": "umap",
    "anomaly_threshold": 0.95
}

# Run detection
results = await agent.process_request(data)
```

### Advanced Configuration

```python
# Custom configuration
config = {
    "time_intervals": [5, 15, 30, 60, 120, 720, 1440],  # More granular intervals
    "similarity_method": "wasserstein",  # Different similarity metric
    "reduction_method": "tsne",  # Use t-SNE instead of UMAP
    "anomaly_threshold": 0.99,  # More strict threshold
    "markov_n_states": 15,  # More states for finer granularity
    "markov_smoothing_factor": 0.001  # Less smoothing
}

agent = MarkovAnomalyDetectionAgent(
    agent_id="custom_markov",
    config=config
)
```

## Results Interpretation

### Output Structure

```python
{
    "anomalies": [
        {
            "interval_minutes": 30,
            "sequence_index": 5,
            "distance": 2.34,
            "threshold_distance": 1.89,
            "embedding_coordinates": [0.45, -0.23],
            "method": "markov_embedding"
        }
    ],
    "sequences": {
        5: 120,    # 5min intervals: 120 sequences
        30: 24,    # 30min intervals: 24 sequences
        60: 12,    # 1hr intervals: 12 sequences
        720: 1,    # 12hr intervals: 1 sequence
        1440: 1    # 24hr intervals: 1 sequence
    },
    "similarity_matrices": {
        5: (120, 120),    # 120x120 similarity matrix
        30: (24, 24),     # 24x24 similarity matrix
        # ... etc
    },
    "embeddings": {
        5: (120, 2),      # 120 sequences reduced to 2D
        30: (24, 2),      # 24 sequences reduced to 2D
        # ... etc
    },
    "plots": {
        "similarity_matrix_5min": "path/to/plot.png",
        "embedding_5min": "path/to/plot.png",
        "sequence_length_distribution": "path/to/plot.png"
    },
    "total_anomalies": 15,
    "confidence": 0.87,
    "method": "markov_chain",
    "similarity_method": "kl_divergence",
    "reduction_method": "umap"
}
```

### Understanding Results

1. **Anomalies**: List of detected anomalous sequences with details
2. **Sequences**: Number of sequences created for each time interval
3. **Similarity Matrices**: Shape of similarity matrices (NxN where N = number of sequences)
4. **Embeddings**: Shape of reduced embeddings (Nx2 for 2D visualization)
5. **Plots**: Generated visualization files
6. **Confidence**: Overall confidence score based on similarity consistency

## Visualizations

### Generated Plots

1. **Similarity Matrix Heatmaps**: Show similarity between all sequence pairs
2. **Embedding Scatter Plots**: 2D visualization of sequences with anomalies highlighted
3. **Sequence Length Distribution**: Histogram of sequence lengths across intervals

### Interpreting Visualizations

- **Similarity Matrix**: Brighter colors = higher similarity
- **Embedding Plot**: Anomalies appear as outliers from the main cluster
- **Length Distribution**: Shows the distribution of sequence lengths

## Performance Considerations

### Computational Complexity

- **Time Complexity**: O(N²) for similarity matrix computation
- **Space Complexity**: O(N²) for storing similarity matrices
- **Memory Usage**: Scales with number of sequences and time intervals

### Optimization Strategies

1. **Parallel Processing**: Compute similarity matrices in parallel
2. **Sparse Matrices**: Use sparse matrices for large datasets
3. **Approximate Methods**: Use approximate similarity methods for speed
4. **Subsampling**: Process subsets for very large datasets

### Scalability

- **Small datasets** (< 1000 points): Real-time processing
- **Medium datasets** (1000-10000 points): Minutes to process
- **Large datasets** (> 10000 points): May require optimization

## Comparison with Other Methods

### Advantages

1. **Multi-scale Analysis**: Detects anomalies at multiple time scales
2. **Sequence-aware**: Considers temporal dependencies
3. **Robust**: Less sensitive to noise than point-based methods
4. **Interpretable**: Provides rich visualizations and explanations
5. **Flexible**: Multiple similarity and reduction methods

### Limitations

1. **Computational Cost**: Higher than simple statistical methods
2. **Parameter Sensitivity**: Requires tuning of multiple parameters
3. **Discretization**: Loss of information through state discretization
4. **Interpretation**: Complex results may be harder to interpret

### When to Use

**Use Markov Chain Anomaly Detection when:**
- You have time series with complex temporal patterns
- You need multi-scale anomaly detection
- You want rich visualizations and explanations
- You have sufficient computational resources

**Consider alternatives when:**
- You need real-time processing of large datasets
- You have simple, stationary time series
- You need very fast processing
- You have limited computational resources

## Best Practices

### Parameter Tuning

1. **Time Intervals**: Choose based on your data characteristics
   - High-frequency data: [1, 5, 15, 30] minutes
   - Daily data: [30, 60, 720, 1440] minutes
   - Weekly data: [1440, 10080] minutes

2. **Number of States**: Balance granularity vs. noise
   - Too few: May miss subtle patterns
   - Too many: May capture noise
   - Recommended: 8-15 states

3. **Similarity Method**: Choose based on your data
   - KL Divergence: Good for probability distributions
   - Cosine: Good for direction-based similarity
   - Euclidean: Good for magnitude-based similarity
   - Wasserstein: Good for distribution shape

### Data Preparation

1. **Clean Data**: Remove obvious outliers before processing
2. **Normalize**: Consider normalizing if scales vary greatly
3. **Handle Missing Data**: Interpolate or remove missing values
4. **Ensure Regular Sampling**: Resample to regular intervals if needed

### Interpretation

1. **Check Multiple Intervals**: Anomalies at multiple scales are more reliable
2. **Examine Visualizations**: Use plots to understand the patterns
3. **Validate Results**: Compare with domain knowledge
4. **Iterate**: Adjust parameters based on results

## Integration with RAG Framework

The Markov Chain Anomaly Detection agent integrates seamlessly with the Time Series RAG Framework:

1. **Agent Registration**: Automatically registered as `markov_anomaly_detection`
2. **Prompt Integration**: Uses relevant prompts from the RAG system
3. **Master Agent Coordination**: Can be orchestrated by the master agent
4. **API Integration**: Available through the FastAPI endpoints

### Example Integration

```python
from agents.master_agent import MasterAgent

# Master agent can coordinate multiple detection methods
master = MasterAgent(agent_id="master")

# Request anomaly detection with multiple methods
request = {
    "task": "anomaly_detection",
    "data": time_series_data,
    "methods": ["markov_chain", "isolation_forest", "zscore"],
    "parameters": {
        "markov_chain": {
            "time_intervals": [5, 30, 60, 720, 1440],
            "similarity_method": "kl_divergence"
        }
    }
}

results = await master.process_request(request)
```

## Future Enhancements

### Planned Features

1. **Adaptive Time Intervals**: Automatically determine optimal intervals
2. **Online Learning**: Update models with new data
3. **Ensemble Methods**: Combine multiple similarity metrics
4. **Causal Analysis**: Identify causal relationships in anomalies
5. **Real-time Processing**: Stream processing capabilities

### Research Directions

1. **Deep Markov Models**: Use neural networks for transition modeling
2. **Attention Mechanisms**: Focus on relevant time periods
3. **Graph Neural Networks**: Model complex temporal relationships
4. **Multi-modal Integration**: Combine with other data sources

## Conclusion

The Markov Chain-based Anomaly Detection system provides a sophisticated approach to identifying anomalies in time series data. By leveraging event sequences, Markov chain modeling, and non-linear dimension reduction, it can detect complex temporal patterns that traditional methods might miss.

The system is particularly valuable for:
- Multi-scale time series analysis
- Complex temporal pattern detection
- Rich visualization and interpretation
- Integration with the broader RAG framework

With proper parameter tuning and interpretation, this system can provide powerful insights into time series anomalies across multiple temporal scales. 