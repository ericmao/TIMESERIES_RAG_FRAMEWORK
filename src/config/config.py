"""
Configuration settings for the Time Series RAG Framework
"""
import os
from typing import Dict, List, Optional
from dataclasses import dataclass
from pathlib import Path

@dataclass
class ModelConfig:
    """Configuration for language models and embeddings"""
    # Base models
    base_lm_model: str = "MediaTek-Research/Breeze-2-8B"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Fine-tuning parameters
    learning_rate: float = 2e-5
    batch_size: int = 8
    max_length: int = 512
    num_epochs: int = 3
    
    # Model sizes for different agents
    master_agent_model: str = "microsoft/DialoGPT-large"
    forecasting_agent_model: str = "microsoft/DialoGPT-medium"
    anomaly_agent_model: str = "microsoft/DialoGPT-medium"
    classification_agent_model: str = "microsoft/DialoGPT-medium"

@dataclass
class RAGConfig:
    """Configuration for RAG components"""
    # Vector database
    vector_db_path: str = "models/vector_db"
    collection_name: str = "timeseries_prompts"
    
    # Retrieval parameters
    top_k: int = 5
    similarity_threshold: float = 0.7
    
    # Prompt pool settings
    prompt_pool_path: str = "models/prompts"
    max_prompt_length: int = 1000
    
    # Knowledge distillation
    distillation_temperature: float = 0.7
    distillation_alpha: float = 0.5

@dataclass
class TimeSeriesConfig:
    """Configuration for time series specific settings"""
    # Data preprocessing
    min_sequence_length: int = 50
    max_sequence_length: int = 1000
    forecast_horizon: int = 24
    
    # Feature engineering
    use_trend_features: bool = True
    use_seasonal_features: bool = True
    use_lag_features: bool = True
    max_lag: int = 12
    
    # Anomaly detection
    anomaly_threshold: float = 0.95
    window_size: int = 10
    
    # Markov chain anomaly detection
    markov_time_intervals: List[int] = None  # [5, 30, 60, 720, 1440] minutes
    markov_n_states: int = 10
    markov_smoothing_factor: float = 0.01
    markov_similarity_method: str = "kl_divergence"  # kl_divergence, cosine, euclidean, wasserstein
    markov_reduction_method: str = "umap"  # umap, tsne
    markov_umap_n_neighbors: int = 15
    markov_umap_min_dist: float = 0.1
    markov_tsne_perplexity: int = 30
    markov_tsne_n_iter: int = 1000
    
    # Classification
    num_classes: int = 5
    classification_threshold: float = 0.5
    
    def __post_init__(self):
        if self.markov_time_intervals is None:
            self.markov_time_intervals = [5, 30, 60, 720, 1440]  # 5min, 30min, 1hr, 12hr, 24hr

@dataclass
class AgentConfig:
    """Configuration for multi-agent system"""
    # Agent types
    agent_types: List[str] = None
    
    # Communication
    max_conversation_turns: int = 10
    timeout_seconds: int = 30
    
    # Orchestration
    enable_hierarchical: bool = True
    enable_parallel: bool = True
    
    def __post_init__(self):
        if self.agent_types is None:
            self.agent_types = [
                "forecasting",
                "anomaly_detection",
                "markov_anomaly_detection",
                "hmm_anomaly_detection",
                "crf_anomaly_detection",
                "classification",
                "trend_analysis",
                "seasonality_analysis"
            ]

@dataclass
class DataConfig:
    """Configuration for data handling"""
    # Data paths
    raw_data_path: str = "data/raw"
    processed_data_path: str = "data/processed"
    
    # Data formats
    supported_formats: List[str] = None
    
    # Data validation
    min_data_points: int = 100
    max_missing_ratio: float = 0.1
    
    def __post_init__(self):
        if self.supported_formats is None:
            self.supported_formats = [".csv", ".json", ".parquet", ".xlsx"]

@dataclass
class TrainingConfig:
    """Configuration for training and evaluation"""
    # Training
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    
    # Evaluation metrics
    forecasting_metrics: List[str] = None
    classification_metrics: List[str] = None
    anomaly_metrics: List[str] = None
    
    # Logging
    log_level: str = "INFO"
    save_checkpoints: bool = True
    checkpoint_interval: int = 1000
    
    def __post_init__(self):
        if self.forecasting_metrics is None:
            self.forecasting_metrics = ["mse", "mae", "rmse", "mape"]
        if self.classification_metrics is None:
            self.classification_metrics = ["accuracy", "f1", "precision", "recall"]
        if self.anomaly_metrics is None:
            self.anomaly_metrics = ["f1", "precision", "recall", "auc"]

@dataclass
class Config:
    """Main configuration class"""
    model: ModelConfig = None
    rag: RAGConfig = None
    timeseries: TimeSeriesConfig = None
    agent: AgentConfig = None
    data: DataConfig = None
    training: TrainingConfig = None
    
    # General settings
    project_name: str = "TimeSeriesRAGFramework"
    version: str = "1.0.0"
    random_seed: int = 42
    
    def __post_init__(self):
        if self.model is None:
            self.model = ModelConfig()
        if self.rag is None:
            self.rag = RAGConfig()
        if self.timeseries is None:
            self.timeseries = TimeSeriesConfig()
        if self.agent is None:
            self.agent = AgentConfig()
        if self.data is None:
            self.data = DataConfig()
        if self.training is None:
            self.training = TrainingConfig()

# Global configuration instance
config = Config()

def get_config() -> Config:
    """Get the global configuration instance"""
    return config

def update_config(**kwargs) -> None:
    """Update configuration with new values"""
    global config
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            raise ValueError(f"Unknown configuration key: {key}") 