# AgenticRAG with Markov Chain Anomaly Detection for CERT Dataset

## Overview

This guide demonstrates how to use AgenticRAG (Retrieval-Augmented Generation) with Markov chain anomaly detection to analyze the CERT insider threat dataset. The integration combines the power of language models with sophisticated time series analysis for detecting insider threats.

## 🎯 Key Features

### AgenticRAG Components
- **Language Model Integration**: Uses Breeze-2 models for natural language understanding
- **RAG Pipeline**: Retrieval-augmented generation for context-aware analysis
- **Markov Chain Analysis**: Advanced sequence modeling for anomaly detection
- **Multi-dimensional Analysis**: Multiple feature sequences and similarity methods
- **Visualization**: Comprehensive plots and reports

### Markov Chain Anomaly Detection
- **Event Sequence Processing**: Analyzes user behavior patterns over time
- **Transition Probability Modeling**: Computes Markov chain transition matrices
- **Similarity Analysis**: Multiple similarity methods (KL divergence, cosine, etc.)
- **Dimension Reduction**: UMAP/t-SNE for anomaly identification
- **Threshold Optimization**: Dynamic threshold selection

## 📊 Dataset Preparation

### CERT Dataset Structure
```python
# Required columns for AgenticRAG
{
    'ds': '2024-01-01 00:00:00',  # Timestamp
    'y': 0.75,                    # Activity score
    'is_malicious': False,         # Ground truth label
    'user_id': 'user_123',        # User identifier
    'activity_level': 0.8,        # Activity level (0-1)
    'file_accesses': 5,           # Number of file accesses
    'network_connections': 3,      # Network connections
    'data_transfer_mb': 10.5,     # Data transfer in MB
    'login_events': 2,            # Login events
    'privilege_escalation': 0      # Privilege escalation attempts
}
```

### Data Preprocessing Steps
1. **Time-based Features**: Hour, day of week, work hours indicators
2. **Derived Features**: Activity rates, user behavior deviations
3. **Rolling Statistics**: Moving averages, standard deviations
4. **Change Detection**: First/second derivatives of time series

## 🚀 Usage Examples

### Basic AgenticRAG Setup

```python
from src.agents.markov_anomaly_detection_agent import MarkovAnomalyDetectionAgent

# Initialize AgenticRAG agent
agent = MarkovAnomalyDetectionAgent(
    agent_id="cert_analysis",
    model_name="microsoft/DialoGPT-medium"
)

# Initialize the agent
await agent.initialize()
```

### Data Preparation

```python
# Prepare data for AgenticRAG
def prepare_cert_data(df, time_interval=60):
    # Resample to specified interval
    df_resampled = df.set_index('timestamp').resample(f'{time_interval}T').mean()
    
    # Create activity score
    df_resampled['activity_score'] = (
        df_resampled['activity_level'] * 0.3 +
        df_resampled['file_accesses'] * 0.2 +
        df_resampled['network_connections'] * 0.2 +
        df_resampled['data_transfer_mb'] * 0.15 +
        df_resampled['login_events'] * 0.1 +
        df_resampled['privilege_escalation'] * 0.05
    )
    
    # Format for AgenticRAG
    agentic_data = []
    for idx, row in df_resampled.iterrows():
        agentic_data.append({
            'ds': idx.strftime('%Y-%m-%d %H:%M:%S'),
            'y': row['activity_score'],
            'is_malicious': row['is_malicious'],
            'user_id': 'aggregated'
        })
    
    return agentic_data
```

### Running Analysis

```python
# Create request for AgenticRAG
request = {
    'data': agentic_data,
    'time_intervals': [60, 120, 720],  # 1hr, 2hr, 12hr
    'similarity_method': 'kl_divergence',
    'reduction_method': 'umap',
    'anomaly_threshold': 0.95,
    'analysis_type': 'insider_threat_detection'
}

# Process request
response = await agent.process_request(request)

if response.success:
    results = response.data
    print("Analysis completed successfully!")
else:
    print(f"Analysis failed: {response.message}")
```

## 📈 Analysis Results

### Output Structure
```python
{
    "anomalies": [
        {
            "start_time": "2024-01-15 14:00:00",
            "end_time": "2024-01-15 16:00:00",
            "confidence": 0.95,
            "method": "agentic_rag_markov"
        }
    ],
    "sequences": {
        "60": 25,    # Number of sequences at 60min intervals
        "120": 15,   # Number of sequences at 120min intervals
        "720": 5     # Number of sequences at 720min intervals
    },
    "similarity_matrices": {
        "60": [25, 25],   # Matrix shape
        "120": [15, 15],
        "720": [5, 5]
    },
    "method": "markov_chain",
    "similarity_method": "kl_divergence",
    "reduction_method": "umap",
    "time_intervals": [60, 120, 720],
    "used_prompts": [...]  # Retrieved prompts from RAG
}
```

### Performance Metrics
- **Precision**: Accuracy of detected anomalies
- **Recall**: Coverage of actual malicious activities
- **F1-Score**: Harmonic mean of precision and recall
- **AUC**: Area under ROC curve
- **Confidence**: Anomaly detection confidence scores

## 🔧 Configuration Options

### Markov Chain Parameters
```python
# Time intervals for analysis
time_intervals = [5, 30, 60, 720, 1440]  # 5min to 24hr

# Number of states for discretization
n_states = 10

# Smoothing factor for transition probabilities
smoothing_factor = 0.01
```

### Similarity Methods
```python
similarity_methods = {
    "kl_divergence": "Kullback-Leibler divergence",
    "cosine": "Cosine similarity",
    "euclidean": "Euclidean distance",
    "wasserstein": "Wasserstein distance"
}
```

### Dimension Reduction
```python
# UMAP parameters
umap_params = {
    "n_components": 2,
    "n_neighbors": 15,
    "min_dist": 0.1,
    "metric": "cosine"
}

# t-SNE parameters
tsne_params = {
    "n_components": 2,
    "perplexity": 30,
    "n_iter": 1000,
    "metric": "cosine"
}
```

## 📊 Visualization

### Generated Plots
1. **Sequence Analysis**: Markov chain transition matrices
2. **Similarity Heatmaps**: Pairwise sequence similarities
3. **Dimension Reduction**: UMAP/t-SNE embeddings
4. **Anomaly Detection**: Anomaly scores and thresholds
5. **Performance Metrics**: Precision, recall, F1-score comparisons

### Example Visualization Code
```python
import matplotlib.pyplot as plt
import seaborn as sns

# Plot similarity matrix
plt.figure(figsize=(10, 8))
sns.heatmap(similarity_matrix, annot=True, cmap='viridis')
plt.title('Markov Chain Similarity Matrix')
plt.show()

# Plot anomaly scores
plt.figure(figsize=(12, 6))
plt.plot(anomaly_scores)
plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
plt.title('Anomaly Scores Over Time')
plt.legend()
plt.show()
```

## 🎯 Best Practices

### Data Quality
1. **Clean Data**: Remove outliers and handle missing values
2. **Feature Engineering**: Create domain-specific features
3. **Temporal Alignment**: Ensure proper time windowing
4. **User Segmentation**: Analyze individual user patterns

### Model Configuration
1. **Time Intervals**: Choose appropriate intervals for your use case
2. **Similarity Method**: Select based on data characteristics
3. **Threshold Tuning**: Optimize for precision vs recall trade-off
4. **Feature Selection**: Use relevant features for your domain

### Performance Optimization
1. **Parallel Processing**: Use multiple time intervals simultaneously
2. **Caching**: Cache frequently computed features
3. **Incremental Learning**: Update models without full retraining
4. **Memory Management**: Process data in chunks for large datasets

## 🔍 Troubleshooting

### Common Issues

#### 1. Low Performance
- **Issue**: Poor precision/recall scores
- **Solution**: 
  - Adjust time intervals
  - Try different similarity methods
  - Optimize feature engineering
  - Tune anomaly thresholds

#### 2. Memory Issues
- **Issue**: Out of memory errors
- **Solution**:
  - Reduce dataset size
  - Use smaller time intervals
  - Process data in chunks
  - Optimize feature selection

#### 3. Model Initialization Failures
- **Issue**: Agent fails to initialize
- **Solution**:
  - Check model dependencies
  - Verify data format
  - Ensure sufficient memory
  - Check configuration parameters

### Debugging Tips
```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check agent status
status = agent.get_status()
print(f"Agent status: {status}")

# Validate data format
def validate_data(data):
    required_fields = ['ds', 'y', 'is_malicious']
    for record in data:
        for field in required_fields:
            if field not in record:
                raise ValueError(f"Missing required field: {field}")
```

## 📚 Advanced Usage

### Custom Similarity Methods
```python
def custom_similarity(matrix1, matrix2):
    """Custom similarity function"""
    # Implement your similarity metric
    return similarity_score

# Add to agent
agent.similarity_methods['custom'] = custom_similarity
```

### Ensemble Methods
```python
# Combine multiple similarity methods
results = {}
for method in ['kl_divergence', 'cosine', 'euclidean']:
    request['similarity_method'] = method
    response = await agent.process_request(request)
    results[method] = response.data

# Ensemble the results
ensemble_anomalies = combine_results(results)
```

### Real-time Processing
```python
# Process streaming data
async def process_streaming_data(data_stream):
    for batch in data_stream:
        request = {'data': batch}
        response = await agent.process_request(request)
        yield response.data
```

## 🎉 Success Metrics

### Expected Performance
- **Precision**: 70-85% for well-tuned models
- **Recall**: 80-95% for comprehensive detection
- **F1-Score**: 75-90% for balanced performance
- **AUC**: 85-95% for robust anomaly detection

### Key Success Factors
1. **Quality Data**: Clean, well-structured time series data
2. **Feature Engineering**: Domain-specific feature creation
3. **Parameter Tuning**: Optimized thresholds and intervals
4. **Validation**: Proper train/test splits and cross-validation

## 📁 File Structure

```
timeseries_rag_framework/
├── src/
│   ├── agents/
│   │   ├── markov_anomaly_detection_agent.py
│   │   └── base_agent.py
│   └── config/
├── data/
│   └── cert_insider_threat/
├── tests/
├── agentic_rag_cert_test.py
├── simple_agentic_rag_test.py
└── agentic_rag_integration_test.py
```

## 🚀 Getting Started

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare Data**:
   ```python
   # Create or load CERT dataset
   df = create_cert_dataset()
   ```

3. **Run Analysis**:
   ```python
   # Run AgenticRAG test
   python simple_agentic_rag_test.py
   ```

4. **Review Results**:
   - Check generated JSON files for detailed results
   - Review CSV summaries for performance metrics
   - Examine visualizations for insights

## 📞 Support

For questions and issues:
1. Check the troubleshooting section
2. Review the example scripts
3. Examine the generated logs
4. Consult the performance enhancement guide

The AgenticRAG framework provides a powerful combination of language model capabilities with sophisticated time series analysis, making it ideal for detecting complex insider threats in the CERT dataset. 