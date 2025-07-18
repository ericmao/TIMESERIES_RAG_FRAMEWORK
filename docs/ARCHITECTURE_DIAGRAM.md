# 時間序列 RAG 框架 - 系統架構圖

## 🏗️ 整體架構概覽

```mermaid
graph TB
    subgraph "🌐 用戶層 (User Layer)"
        UI[Web UI / API Client]
        CLI[Command Line Interface]
    end

    subgraph "🚀 API 層 (API Layer)"
        FastAPI[FastAPI Server]
        REST[REST API Endpoints]
        WebSocket[WebSocket Connections]
    end

    subgraph "🧠 智能代理層 (Agent Layer)"
        MA[Master Agent<br/>協調器]
        
        subgraph "專門代理 (Specialized Agents)"
            FA[Forecasting Agent<br/>預測代理]
            AA[Anomaly Detection Agent<br/>異常檢測代理]
            CA[Classification Agent<br/>分類代理]
            TA[Trend Analysis Agent<br/>趨勢分析代理]
            SA[Seasonality Analysis Agent<br/>季節性分析代理]
        end
    end

    subgraph "🔍 RAG 引擎 (RAG Engine)"
        PM[Prompt Manager<br/>提示詞管理器]
        VE[Vector Engine<br/>向量引擎]
        RE[Retrieval Engine<br/>檢索引擎]
        GE[Generation Engine<br/>生成引擎]
    end

    subgraph "🤖 AI 模型層 (AI Model Layer)"
        subgraph "語言模型"
            GPT2[GPT-2]
            DialoGPT[DialoGPT]
            Breeze[Breeze]
        end
        
        subgraph "時間序列模型"
            Prophet[Prophet]
            NeuralProphet[Neural Prophet]
            Darts[Darts]
        end
        
        subgraph "機器學習模型"
            IsolationForest[Isolation Forest]
            ZScore[Z-Score]
            IQR[IQR]
        end
    end

    subgraph "💾 數據層 (Data Layer)"
        subgraph "向量資料庫"
            ChromaDB[ChromaDB]
            FAISS[FAISS]
        end
        
        subgraph "關係資料庫"
            PostgreSQL[PostgreSQL]
        end
        
        subgraph "文件存儲"
            FileSystem[File System]
            Checkpoints[Model Checkpoints]
        end
    end

    subgraph "🛠️ 工具層 (Utility Layer)"
        DP[Data Processor<br/>數據處理器]
        Logger[Logger<br/>日誌系統]
        Config[Config Manager<br/>配置管理器]
        Cache[Cache Manager<br/>緩存管理器]
    end

    %% 連接關係
    UI --> FastAPI
    CLI --> FastAPI
    FastAPI --> MA
    MA --> FA
    MA --> AA
    MA --> CA
    MA --> TA
    MA --> SA
    
    FA --> PM
    AA --> PM
    CA --> PM
    TA --> PM
    SA --> PM
    
    PM --> VE
    VE --> ChromaDB
    VE --> FAISS
    
    RE --> VE
    GE --> GPT2
    GE --> DialoGPT
    GE --> Breeze
    
    FA --> Prophet
    FA --> NeuralProphet
    FA --> Darts
    
    AA --> IsolationForest
    AA --> ZScore
    AA --> IQR
    
    MA --> PostgreSQL
    MA --> FileSystem
    MA --> Checkpoints
    
    MA --> DP
    MA --> Logger
    MA --> Config
    MA --> Cache
```

## 🔄 數據流程圖

```mermaid
sequenceDiagram
    participant U as 用戶
    participant API as FastAPI
    participant MA as Master Agent
    participant FA as Forecasting Agent
    participant AA as Anomaly Agent
    participant CA as Classification Agent
    participant RAG as RAG Engine
    participant DB as Database

    U->>API: 發送分析請求
    API->>MA: 路由到主代理
    MA->>RAG: 檢索相關提示詞
    RAG->>DB: 查詢向量資料庫
    DB-->>RAG: 返回相關提示詞
    RAG-->>MA: 提供上下文

    par 並行處理
        MA->>FA: 預測分析
        FA->>RAG: 獲取預測提示詞
        RAG-->>FA: 返回提示詞
        FA->>FA: 執行 Prophet 預測
        FA-->>MA: 返回預測結果
    and
        MA->>AA: 異常檢測
        AA->>RAG: 獲取異常檢測提示詞
        RAG-->>AA: 返回提示詞
        AA->>AA: 執行多方法異常檢測
        AA-->>MA: 返回異常結果
    and
        MA->>CA: 模式分類
        CA->>RAG: 獲取分類提示詞
        RAG-->>CA: 返回提示詞
        CA->>CA: 執行模式分類
        CA-->>MA: 返回分類結果
    end

    MA->>MA: 整合所有結果
    MA->>DB: 保存分析結果
    MA-->>API: 返回綜合分析
    API-->>U: 返回最終結果
```

## 🏛️ 組件詳細架構

### 1. 主代理 (Master Agent)
```mermaid
graph LR
    subgraph "Master Agent"
        Router[請求路由器]
        Coordinator[協調器]
        Aggregator[結果聚合器]
        LoadBalancer[負載均衡器]
    end
    
    Router --> Coordinator
    Coordinator --> Aggregator
    Aggregator --> LoadBalancer
```

### 2. 預測代理 (Forecasting Agent)
```mermaid
graph LR
    subgraph "Forecasting Agent"
        ProphetModel[Prophet 模型]
        NeuralModel[Neural Prophet]
        DartsModel[Darts 模型]
        Ensemble[集成預測]
    end
    
    ProphetModel --> Ensemble
    NeuralModel --> Ensemble
    DartsModel --> Ensemble
```

### 3. 異常檢測代理 (Anomaly Detection Agent)
```mermaid
graph LR
    subgraph "Anomaly Detection Agent"
        ZScore[Z-Score 檢測]
        IQR[IQR 檢測]
        IsolationForest[Isolation Forest]
        RollingStats[滾動統計]
        Ensemble[多方法融合]
    end
    
    ZScore --> Ensemble
    IQR --> Ensemble
    IsolationForest --> Ensemble
    RollingStats --> Ensemble
```

### 4. RAG 引擎架構
```mermaid
graph TB
    subgraph "RAG Engine"
        subgraph "檢索層"
            QueryProcessor[查詢處理器]
            VectorSearch[向量搜尋]
            RelevanceFilter[相關性過濾]
        end
        
        subgraph "生成層"
            ContextBuilder[上下文構建器]
            PromptRenderer[提示詞渲染器]
            ResponseGenerator[回應生成器]
        end
        
        subgraph "知識庫"
            PromptStore[提示詞存儲]
            VectorDB[向量資料庫]
            MetadataStore[元數據存儲]
        end
    end
    
    QueryProcessor --> VectorSearch
    VectorSearch --> RelevanceFilter
    RelevanceFilter --> ContextBuilder
    ContextBuilder --> PromptRenderer
    PromptRenderer --> ResponseGenerator
    
    VectorSearch --> VectorDB
    PromptRenderer --> PromptStore
    ResponseGenerator --> MetadataStore
```

## 🎯 技術棧架構

```mermaid
graph TB
    subgraph "前端技術"
        NextJS[Next.js]
        React[React]
        TailwindCSS[Tailwind CSS]
        TypeScript[TypeScript]
    end
    
    subgraph "後端技術"
        FastAPI[FastAPI]
        Python[Python 3.12]
        AsyncIO[AsyncIO]
        Pydantic[Pydantic]
    end
    
    subgraph "AI/ML 技術"
        Transformers[Transformers]
        Torch[PyTorch]
        Prophet[Prophet]
        ScikitLearn[Scikit-learn]
    end
    
    subgraph "資料庫技術"
        PostgreSQL[PostgreSQL]
        ChromaDB[ChromaDB]
        FAISS[FAISS]
    end
    
    subgraph "部署技術"
        Docker[Docker]
        Kubernetes[Kubernetes]
        GitHubActions[GitHub Actions]
    end
    
    NextJS --> FastAPI
    React --> FastAPI
    FastAPI --> Transformers
    FastAPI --> PostgreSQL
    Transformers --> ChromaDB
    FastAPI --> Docker
```

## 📊 性能指標架構

```mermaid
graph LR
    subgraph "性能監控"
        subgraph "響應時間"
            RT[響應時間監控]
            QPS[QPS 監控]
            Latency[延遲監控]
        end
        
        subgraph "準確性"
            Accuracy[準確率監控]
            Precision[精確率監控]
            Recall[召回率監控]
        end
        
        subgraph "資源使用"
            CPU[CPU 使用率]
            Memory[記憶體使用率]
            GPU[GPU 使用率]
        end
    end
    
    RT --> CPU
    Accuracy --> Memory
    QPS --> GPU
```

## 🔧 配置管理架構

```mermaid
graph TB
    subgraph "配置層"
        subgraph "環境配置"
            Dev[開發環境]
            Test[測試環境]
            Prod[生產環境]
        end
        
        subgraph "模型配置"
            ModelConfig[模型配置]
            AgentConfig[代理配置]
            RAGConfig[RAG 配置]
        end
        
        subgraph "系統配置"
            DatabaseConfig[資料庫配置]
            CacheConfig[緩存配置]
            LogConfig[日誌配置]
        end
    end
    
    Dev --> ModelConfig
    Test --> AgentConfig
    Prod --> RAGConfig
    
    ModelConfig --> DatabaseConfig
    AgentConfig --> CacheConfig
    RAGConfig --> LogConfig
```

## 🚀 部署架構

```mermaid
graph TB
    subgraph "CI/CD Pipeline"
        GitHub[GitHub Repository]
        Actions[GitHub Actions]
        DockerHub[Docker Hub]
        Registry[Container Registry]
    end
    
    subgraph "部署環境"
        subgraph "開發環境"
            DevK8s[Kubernetes Dev]
            DevDB[PostgreSQL Dev]
        end
        
        subgraph "測試環境"
            TestK8s[Kubernetes Test]
            TestDB[PostgreSQL Test]
        end
        
        subgraph "生產環境"
            ProdK8s[Kubernetes Prod]
            ProdDB[PostgreSQL Prod]
        end
    end
    
    GitHub --> Actions
    Actions --> DockerHub
    DockerHub --> Registry
    Registry --> DevK8s
    Registry --> TestK8s
    Registry --> ProdK8s
    
    DevK8s --> DevDB
    TestK8s --> TestDB
    ProdK8s --> ProdDB
```

## 📈 擴展性架構

```mermaid
graph LR
    subgraph "水平擴展"
        LoadBalancer[負載均衡器]
        Agent1[Agent Instance 1]
        Agent2[Agent Instance 2]
        Agent3[Agent Instance N]
    end
    
    subgraph "垂直擴展"
        CPU[CPU 擴展]
        Memory[記憶體擴展]
        GPU[GPU 擴展]
    end
    
    subgraph "功能擴展"
        NewAgent[新代理類型]
        NewModel[新模型]
        NewFeature[新功能]
    end
    
    LoadBalancer --> Agent1
    LoadBalancer --> Agent2
    LoadBalancer --> Agent3
    
    Agent1 --> CPU
    Agent2 --> Memory
    Agent3 --> GPU
    
    NewAgent --> NewModel
    NewModel --> NewFeature
```

---

## 📋 架構特點總結

### 🎯 **核心優勢**
1. **模組化設計**: 各組件獨立，易於維護和擴展
2. **智能代理系統**: 專門化代理協同工作
3. **RAG 技術整合**: 提升分析準確性和可解釋性
4. **多模型支援**: 支援多種 AI 模型和時間序列模型
5. **高可擴展性**: 支援水平擴展和垂直擴展

### 🔧 **技術特色**
1. **異步處理**: 使用 AsyncIO 提升性能
2. **向量檢索**: 高效的相似性搜尋
3. **模型集成**: 多模型融合提高準確性
4. **自動化部署**: CI/CD 流水線
5. **監控體系**: 完整的性能監控

### 🚀 **部署靈活性**
1. **容器化部署**: Docker 容器化
2. **雲原生**: Kubernetes 編排
3. **多環境支援**: 開發、測試、生產環境
4. **自動化測試**: 完整的測試覆蓋
5. **版本控制**: Git 版本管理

這個架構設計確保了系統的高性能、高可用性和高可擴展性，能夠滿足各種時間序列分析的需求。 