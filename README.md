# 時間序列 RAG 框架 (Time Series RAG Framework)

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Framework](https://img.shields.io/badge/Framework-Next.js%20%7C%20FastAPI%20%7C%20PostgreSQL-blue.svg)]()

一個專為時間序列分析設計的智能代理框架，結合了 RAG (Retrieval-Augmented Generation) 技術和多個專門的 AI 代理，提供全面的時間序列分析解決方案。

## 📋 目錄

- [專案概述](#專案概述)
- [核心功能](#核心功能)
- [架構設計](#架構設計)
- [快速開始](#快速開始)
- [安裝指南](#安裝指南)
- [使用範例](#使用範例)
- [API 文檔](#api-文檔)
- [配置說明](#配置說明)
- [模型比較](#模型比較)
- [部署指南](#部署指南)
- [貢獻指南](#貢獻指南)
- [授權條款](#授權條款)

## 🎯 專案概述

這個框架專為時間序列分析而設計，整合了多種 AI 技術：

- **智能代理系統**: 多個專門的 AI 代理協同工作
- **RAG 技術**: 提升分析準確性和可解釋性
- **多模型支援**: 支援多種 HuggingFace 模型
- **模組化設計**: 易於擴展和維護
- **自動化測試**: 完整的模型比較和評估

### 適用場景

- **金融分析**: 股票價格預測、風險評估
- **工業監控**: 設備異常檢測、預測性維護
- **電商分析**: 銷售預測、需求規劃
- **醫療監測**: 病患數據分析
- **氣象預測**: 天氣模式分析
- **能源管理**: 電力負載預測

## 🚀 核心功能

### 1. 時間序列預測
- 使用 Prophet 模型進行預測
- 支援多種預測時間範圍
- 提供預測置信度評估
- 自動季節性分解

### 2. 異常檢測
- **Z-score 方法**: 基於統計的異常檢測
- **IQR 方法**: 四分位距異常檢測
- **Isolation Forest**: 機器學習異常檢測
- **滾動統計**: 動態異常檢測
- **組合方法**: 多方法融合提高準確性

### 3. 模式分類
- **趨勢分類**: 上升/下降/穩定趨勢識別
- **季節性分類**: 季節性模式檢測
- **行為分類**: 穩定/波動行為分析
- **綜合模式**: 多維度模式識別

### 4. 模型比較
- 支援多種 AI 模型比較
- 自動化性能評估
- 生成詳細的比較報告
- 視覺化結果展示

## 🏗️ 架構設計

### 系統架構
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Master Agent  │    │  Forecasting    │    │  Anomaly       │
│   (協調器)      │◄──►│  Agent          │    │  Detection      │
│                 │    │                 │    │  Agent          │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Classification  │    │  RAG System     │    │  Model Pool     │
│ Agent           │    │  (向量資料庫)   │    │  (多模型支援)   │
│                 │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### 核心組件

#### 1. 主代理 (Master Agent)
- 協調各個專門代理的工作
- 路由請求到適當的專門代理
- 整合多個分析結果
- 管理代理生命週期

#### 2. 專門代理
- **預測代理**: 使用 Prophet 模型進行時間序列預測
- **異常檢測代理**: 使用多種方法檢測異常值
- **分類代理**: 識別時間序列的模式和行為

#### 3. RAG 系統
- **向量資料庫**: ChromaDB 儲存提示詞
- **嵌入模型**: Sentence Transformers
- **檢索系統**: 語義搜尋相關提示詞

#### 4. 模型池
- 支援多種 HuggingFace 模型
- 動態模型載入
- 模型性能比較

## ⚡ 快速開始

### 基本使用範例

```python
import asyncio
from src.agents.master_agent import MasterAgent

async def basic_analysis():
    # 初始化主代理
    master_agent = MasterAgent(agent_id="demo_001")
    await master_agent.initialize()
    
    # 準備數據
    request = {
        "task_type": "comprehensive_analysis",
        "data": {
            "ds": ["2022-01-01", "2022-01-02", "2022-01-03"],
            "y": [100, 105, 98]
        },
        "forecast_horizon": 30,
        "threshold": 2.0
    }
    
    # 執行分析
    response = await master_agent.process_request(request)
    
    if response.success:
        print(f"分析完成，置信度: {response.confidence}")
        results = response.data["results"]
        
        # 提取結果
        forecast = results["forecasting"]["results"]
        anomalies = results["anomaly_detection"]["results"]
        classification = results["classification"]["results"]
        
        print(f"預測結果: {len(forecast['forecast'])} 個預測點")
        print(f"異常檢測: {anomalies['total_anomalies']} 個異常點")
        print(f"模式分類: {classification['classification']['pattern']['pattern_type']}")
    
    await master_agent.cleanup()

# 執行分析
asyncio.run(basic_analysis())
```

## 📦 安裝指南

### 系統需求

- Python 3.8+
- 8GB+ RAM (推薦 16GB)
- 10GB+ 磁碟空間

### 1. 克隆專案

```bash
git clone https://github.com/your-username/timeseries_rag_framework.git
cd timeseries_rag_framework
```

### 2. 創建虛擬環境

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows
```

### 3. 安裝依賴

```bash
pip install -r requirements.txt
```

### 4. 環境配置

```bash
# 複製配置範例
cp src/config/config.example.py src/config/config.py

# 編輯配置檔案
nano src/config/config.py
```

### 5. 初始化資料庫

```bash
python scripts/init_database.py
```

## 📖 使用範例

### 1. 金融分析

```python
# 股票價格分析
financial_request = {
    "task_type": "comprehensive_analysis",
    "data": stock_data,
    "description": "分析股票價格趨勢和異常波動",
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

### 2. 工業監控

```python
# 設備故障預測
industrial_request = {
    "task_type": "comprehensive_analysis",
    "data": sensor_data,
    "description": "預測設備故障和維護需求",
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

### 3. 電商分析

```python
# 銷售預測
ecommerce_request = {
    "task_type": "forecast",
    "data": sales_data,
    "description": "預測產品銷售量和需求",
    "method": "sales_forecasting",
    "forecast_horizon": 90,
    "analysis_params": {
        "seasonal_decomposition": True,
        "promotion_impact": True,
        "inventory_optimization": True
    }
}
```

## 🔌 API 文檔

### 輸入格式

#### 基本請求結構
```python
request = {
    "task_type": "comprehensive_analysis",  # 任務類型
    "data": {
        "ds": ["2022-01-01", "2022-01-02", ...],  # 日期列
        "y": [10.5, 11.2, 9.8, ...]               # 數值列
    },
    "description": "分析銷售數據趨勢",  # 可選描述
    "method": "combined",  # 分析方法
    "forecast_horizon": 30,  # 預測期間
    "threshold": 2.0,  # 異常檢測閾值
    "window_size": 10  # 滾動窗口大小
}
```

#### 支援的任務類型
- `"comprehensive_analysis"`: 綜合分析 (預設)
- `"forecast"` / `"predict"`: 預測分析
- `"anomaly"` / `"detect_anomaly"`: 異常檢測
- `"classify"`: 模式分類
- `"trend"`: 趨勢分析
- `"seasonality"`: 季節性分析

### 輸出格式

#### 綜合分析輸出
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

## ⚙️ 配置說明

### 基本配置

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

### 環境變數

```bash
# .env
MODEL_BASE_LM=gpt2
MODEL_MASTER_AGENT=gpt2-medium
FORECAST_HORIZON=30
ANOMALY_THRESHOLD=2.0
RAG_TOP_K=5
```

## 🔬 模型比較

### 支援的模型

| 模型 | 大小 | 性能 | 適用場景 |
|------|------|------|----------|
| GPT-2 | 124M | 快速 | 基本分析 |
| GPT-2 Medium | 355M | 平衡 | 一般用途 |
| GPT-2 Large | 774M | 高精度 | 複雜分析 |
| DialoGPT Medium | 345M | 對話優化 | 互動分析 |
| DialoGPT Large | 762M | 高精度對話 | 複雜互動 |

### 模型比較工具

```bash
# 執行模型比較
python available_model_comparison.py

# 執行簡單比較
python simple_model_comparison.py

# 執行綜合比較
python model_comparison_test.py
```

### 比較結果

比較結果會生成以下檔案：
- `available_model_comparison_results.png`: 視覺化比較圖
- `available_model_comparison_results.json`: 詳細數據
- `simple_model_comparison_results.png`: 簡單比較圖
- `model_comparison_results.json`: 綜合比較數據

## 🚀 部署指南

### Docker 部署

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
# 建構映像
docker build -t timeseries-rag-framework .

# 執行容器
docker run -p 8000:8000 timeseries-rag-framework
```

### Kubernetes 部署

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

### 生產環境配置

```python
# production_config.py
@dataclass
class ProductionConfig:
    # 高可用性配置
    load_balancer: bool = True
    auto_scaling: bool = True
    health_check: bool = True
    
    # 監控配置
    logging_level: str = "INFO"
    metrics_collection: bool = True
    alerting: bool = True
    
    # 安全配置
    authentication: bool = True
    rate_limiting: bool = True
    ssl_enabled: bool = True
```

## 🤝 貢獻指南

### 開發環境設置

```bash
# 克隆專案
git clone https://github.com/your-username/timeseries_rag_framework.git
cd timeseries_rag_framework

# 創建開發分支
git checkout -b feature/your-feature-name

# 安裝開發依賴
pip install -r requirements-dev.txt

# 運行測試
pytest tests/

# 運行 linting
flake8 src/
black src/
```

### 貢獻流程

1. Fork 專案
2. 創建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交變更 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 開啟 Pull Request

### 代碼規範

- 遵循 PEP 8 代碼風格
- 添加適當的註釋和文檔
- 編寫單元測試
- 確保所有測試通過

## 📄 授權條款

本專案採用 MIT 授權條款 - 詳見 [LICENSE](LICENSE) 檔案

## 📞 聯絡資訊

- **專案維護者**: [Your Name](mailto:your.email@example.com)
- **專案網站**: [https://github.com/your-username/timeseries_rag_framework](https://github.com/your-username/timeseries_rag_framework)
- **問題回報**: [Issues](https://github.com/your-username/timeseries_rag_framework/issues)

## 🙏 致謝

感謝所有為這個專案做出貢獻的開發者和研究人員。

---

⭐ 如果這個專案對您有幫助，請給我們一個星標！ 