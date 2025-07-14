# Time Series RAG Framework

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Framework](https://img.shields.io/badge/Framework-Next.js%20%7C%20FastAPI%20%7C%20PostgreSQL-blue.svg)]()

A comprehensive intelligent agent framework designed for time series analysis, combining RAG (Retrieval-Augmented Generation) technology with specialized AI agents to provide comprehensive time series analysis solutions.

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Core Features](#core-features)
- [Architecture Design](#architecture-design)
- [Quick Start](#quick-start)
- [Installation Guide](#installation-guide)
- [Usage Examples](#usage-examples)
- [API Documentation](#api-documentation)
- [Configuration](#configuration)
- [Model Comparison](#model-comparison)
- [Deployment Guide](#deployment-guide)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Project Overview

This framework is specifically designed for time series analysis, integrating multiple AI technologies:

- **Intelligent Agent System**: Multiple specialized AI agents working collaboratively
- **RAG Technology**: Enhancing analysis accuracy and interpretability
- **Multi-Model Support**: Supporting various HuggingFace models
- **Modular Design**: Easy to extend and maintain
- **Automated Testing**: Complete model comparison and evaluation

### Use Cases

- **Financial Analysis**: Stock price prediction, risk assessment
- **Industrial Monitoring**: Equipment anomaly detection, predictive maintenance
- **E-commerce Analysis**: Sales forecasting, demand planning
- **Medical Monitoring**: Patient data analysis
- **Weather Forecasting**: Weather pattern analysis
- **Energy Management**: Power load forecasting

## 🚀 Core Features

### 1. Time Series Forecasting
- Using Prophet model for forecasting
- Supporting multiple forecast horizons
- Providing forecast confidence assessment
- Automatic seasonal decomposition

### 2. Anomaly Detection
- **Z-score Method**: Statistical-based anomaly detection
- **IQR Method**: Interquartile range anomaly detection
- **Isolation Forest**: Machine learning anomaly detection
- **Rolling Statistics**: Dynamic anomaly detection
- **Combined Methods**: Multi-method fusion for improved accuracy

### 3. Pattern Classification
- **Trend Classification**: Upward/downward/stable trend identification
- **Seasonality Classification**: Seasonal pattern detection
- **Behavior Classification**: Stable/volatile behavior analysis
- **Comprehensive Patterns**: Multi-dimensional pattern recognition

### 4. Model Comparison
- Support for multiple AI model comparison
- Automated performance evaluation
- Detailed comparison reports generation
- Visualization of results

## 🏗️ Architecture Design

### System Architecture
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Master Agent  │    │  Forecasting    │    │  Anomaly       │
│   (Coordinator) │◄──►│  Agent          │    │  Detection      │
│                 │    │                 │    │  Agent          │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Classification  │    │  RAG System     │    │  Model Pool     │
│ Agent           │    │  (Vector DB)    │    │  (Multi-Model)  │
│                 │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Core Components

#### 1. Master Agent
- Coordinates work of specialized agents
- Routes requests to appropriate specialized agents
- Integrates multiple analysis results
- Manages agent lifecycle

#### 2. Specialized Agents
- **Forecasting Agent**: Uses Prophet model for time series forecasting
- **Anomaly Detection Agent**: Uses multiple methods to detect anomalies
- **Classification Agent**: Identifies time series patterns and behaviors

#### 3. RAG System
- **Vector Database**: ChromaDB for storing prompts
- **Embedding Model**: Sentence Transformers
- **Retrieval System**: Semantic search for relevant prompts

#### 4. Model Pool
- Support for various HuggingFace models
- Dynamic model loading
- Model performance comparison

## ⚡ Quick Start

### Basic Usage Example

```python
import asyncio
from src.agents.master_agent import MasterAgent

async def basic_analysis():
    # Initialize master agent
    master_agent = MasterAgent(agent_id="demo_001")
    await master_agent.initialize()
    
    # Prepare data
    request = {
        "task_type": "comprehensive_analysis",
        "data": {
            "ds": ["2022-01-01", "2022-01-02", "2022-01-03"],
            "y": [100, 105, 98]
        },
        "forecast_horizon": 30,
        "threshold": 2.0
    }
    
    # Execute analysis
    response = await master_agent.process_request(request)
    
    if response.success:
        print(f"Analysis completed, confidence: {response.confidence}")
        results = response.data["results"]
        
        # Extract results
        forecast = results["forecasting"]["results"]
        anomalies = results["anomaly_detection"]["results"]
        classification = results["classification"]["results"]
        
        print(f"Forecast results: {len(forecast['forecast'])} forecast points")
        print(f"Anomaly detection: {anomalies['total_anomalies']} anomalies")
        print(f"Pattern classification: {classification['classification']['pattern']['pattern_type']}")
    
    await master_agent.cleanup()

# Execute analysis
asyncio.run(basic_analysis())
```

## 📦 Installation Guide

### System Requirements

- Python 3.8+
- 8GB+ RAM (16GB recommended)
- 10GB+ disk space

### 1. Clone Repository

```bash
git clone https://github.com/your-username/timeseries_rag_framework.git
cd timeseries_rag_framework
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Environment Configuration

```bash
# Copy configuration example
cp src/config/config.example.py src/config/config.py

# Edit configuration file
nano src/config/config.py
```

### 5. Initialize Database

```bash
python scripts/init_database.py
```

## 📖 Usage Examples

### 1. Financial Analysis

```python
# Stock price analysis
financial_request = {
    "task_type": "comprehensive_analysis",
    "data": stock_data,
    "description": "Analyze stock price trends and abnormal fluctuations",
    "method": "financial_analysis",
    "forecast_horizon": 30,
    "threshold": 2.5,
    "analysis_params": {
        "volatility_analysis": True,
        "risk_assessment": True,
        "trend_breakout_detection": True
    }
}
```

### 2. Industrial Monitoring

```python
# Equipment failure prediction
industrial_request = {
    "task_type": "comprehensive_analysis",
    "data": sensor_data,
    "description": "Predict equipment failures and maintenance needs",
    "method": "predictive_maintenance",
    "forecast_horizon": 7,
    "threshold": 1.8,
    "analysis_params": {
        "vibration_analysis": True,
        "temperature_monitoring": True,
        "performance_degradation": True
    }
}
```

### 3. E-commerce Analysis

```python
# Sales forecasting
ecommerce_request = {
    "task_type": "forecast",
    "data": sales_data,
    "description": "Forecast product sales and demand",
    "method": "sales_forecasting",
    "forecast_horizon": 90,
    "analysis_params": {
        "seasonal_decomposition": True,
        "promotion_impact": True,
        "inventory_optimization": True
    }
}
```

## 🔌 API Documentation

### Input Format

#### Basic Request Structure
```python
request = {
    "task_type": "comprehensive_analysis",  # Task type
    "data": {
        "ds": ["2022-01-01", "2022-01-02", ...],  # Date column
        "y": [10.5, 11.2, 9.8, ...]               # Value column
    },
    "description": "Analyze sales data trends",  # Optional description
    "method": "combined",  # Analysis method
    "forecast_horizon": 30,  # Forecast period
    "threshold": 2.0,  # Anomaly detection threshold
    "window_size": 10  # Rolling window size
}
```

#### Supported Task Types
- `"comprehensive_analysis"`: Comprehensive analysis (default)
- `"forecast"` / `"predict"`: Forecasting analysis
- `"anomaly"` / `"detect_anomaly"`: Anomaly detection
- `"classify"`: Pattern classification
- `"trend"`: Trend analysis
- `"seasonality"`: Seasonality analysis

### Output Format

#### Comprehensive Analysis Output
```python
response = {
    "success": True,
    "message": "Request processed successfully",
    "data": {
        "comprehensive_analysis": True,
        "timestamp": "2024-01-15T10:30:00",
        "results": {
            "forecasting": {
                "results": {
                    "forecast": [...],
                    "forecast_horizon": 30,
                    "confidence": 0.85
                },
                "confidence": 0.85,
                "execution_time": 2.34
            },
            "anomaly_detection": {
                "results": {
                    "anomalies": [...],
                    "total_anomalies": 2,
                    "anomaly_ratio": 0.02
                },
                "confidence": 0.78,
                "execution_time": 1.56
            },
            "classification": {
                "results": {
                    "classification": {
                        "pattern_type": "strong_upward_trend",
                        "trend_type": "increasing",
                        "seasonality_type": "weak_seasonal"
                    }
                },
                "confidence": 0.82,
                "execution_time": 1.23
            }
        },
        "total_execution_time": 5.13
    },
    "confidence": 0.82,
    "execution_time": 5.13
}
```

## ⚙️ Configuration

### Basic Configuration

```python
# src/config/config.py
from dataclasses import dataclass

@dataclass
class ModelConfig:
    base_lm_model: str = "gpt2"
    master_agent_model: str = "gpt2-medium"
    forecasting_agent_model: str = "gpt2"
    anomaly_agent_model: str = "gpt2"
    classification_agent_model: str = "gpt2"

@dataclass
class TimeSeriesConfig:
    forecast_horizon: int = 30
    anomaly_threshold: float = 2.0
    window_size: int = 10

@dataclass
class RAGConfig:
    top_k: int = 5
    similarity_threshold: float = 0.7
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

@dataclass
class Config:
    model: ModelConfig = ModelConfig()
    timeseries: TimeSeriesConfig = TimeSeriesConfig()
    rag: RAGConfig = RAGConfig()
```

### Environment Variables

```bash
# .env
MODEL_BASE_LM=gpt2
MODEL_MASTER_AGENT=gpt2-medium
FORECAST_HORIZON=30
ANOMALY_THRESHOLD=2.0
RAG_TOP_K=5
```

## 🔬 Model Comparison

### Supported Models

| Model | Size | Performance | Use Case |
|-------|------|-------------|----------|
| GPT-2 | 124M | Fast | Basic analysis |
| GPT-2 Medium | 355M | Balanced | General purpose |
| GPT-2 Large | 774M | High accuracy | Complex analysis |
| DialoGPT Medium | 345M | Dialogue optimized | Interactive analysis |
| DialoGPT Large | 762M | High accuracy dialogue | Complex interaction |

### Model Comparison Tools

```bash
# Run model comparison
python available_model_comparison.py

# Run simple comparison
python simple_model_comparison.py

# Run comprehensive comparison
python model_comparison_test.py
```

### Comparison Results

Comparison results generate the following files:
- `available_model_comparison_results.png`: Visualization comparison chart
- `available_model_comparison_results.json`: Detailed data
- `simple_model_comparison_results.png`: Simple comparison chart
- `model_comparison_results.json`: Comprehensive comparison data

## 🚀 Deployment Guide

### Docker Deployment

```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["python", "src/api/main.py"]
```

```bash
# Build image
docker build -t timeseries-rag-framework .

# Run container
docker run -p 8000:8000 timeseries-rag-framework
```

### Kubernetes Deployment

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: timeseries-rag-framework
spec:
  replicas: 3
  selector:
    matchLabels:
      app: timeseries-rag-framework
  template:
    metadata:
      labels:
        app: timeseries-rag-framework
    spec:
      containers:
      - name: app
        image: timeseries-rag-framework:latest
        ports:
        - containerPort: 8000
        env:
        - name: MODEL_BASE_LM
          value: "gpt2-medium"
```

### Production Environment Configuration

```python
# production_config.py
@dataclass
class ProductionConfig:
    # High availability configuration
    load_balancer: bool = True
    auto_scaling: bool = True
    health_check: bool = True
    
    # Monitoring configuration
    logging_level: str = "INFO"
    metrics_collection: bool = True
    alerting: bool = True
    
    # Security configuration
    authentication: bool = True
    rate_limiting: bool = True
    ssl_enabled: bool = True
```

## 🤝 Contributing

### Development Environment Setup

```bash
# Clone repository
git clone https://github.com/your-username/timeseries_rag_framework.git
cd timeseries_rag_framework

# Create development branch
git checkout -b feature/your-feature-name

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Run linting
flake8 src/
black src/
```

### Contribution Process

1. Fork the project
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Code Standards

- Follow PEP 8 code style
- Add appropriate comments and documentation
- Write unit tests
- Ensure all tests pass

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact Information

- **Project Maintainer**: [Your Name](mailto:your.email@example.com)
- **Project Website**: [https://github.com/your-username/timeseries_rag_framework](https://github.com/your-username/timeseries_rag_framework)
- **Issue Reporting**: [Issues](https://github.com/your-username/timeseries_rag_framework/issues)

## 🙏 Acknowledgments

Thanks to all developers and researchers who contributed to this project.

---

⭐ If this project helps you, please give us a star! 