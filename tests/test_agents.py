"""
Tests for Time Series RAG Framework Agents
"""
import pytest
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any

from src.agents.master_agent import MasterAgent
from src.agents.forecasting_agent import ForecastingAgent
from src.agents.anomaly_detection_agent import AnomalyDetectionAgent
from src.agents.classification_agent import ClassificationAgent
from src.config.config import get_config

class TestForecastingAgent:
    """Test cases for ForecastingAgent"""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample time series data"""
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        values = np.sin(np.arange(len(dates)) * 2 * np.pi / 365) + np.random.normal(0, 0.1, len(dates))
        
        return {
            "ds": dates.tolist(),
            "y": values.tolist()
        }
    
    @pytest.fixture
    async def forecasting_agent(self):
        """Create a forecasting agent instance"""
        agent = ForecastingAgent(agent_id="test_forecast")
        await agent.initialize()
        yield agent
        await agent.cleanup()
    
    @pytest.mark.asyncio
    async def test_forecasting_agent_initialization(self, forecasting_agent):
        """Test agent initialization"""
        assert forecasting_agent.agent_id == "test_forecast"
        assert forecasting_agent.agent_type == "forecasting"
        assert forecasting_agent.is_initialized == True
    
    @pytest.mark.asyncio
    async def test_forecasting_basic(self, forecasting_agent, sample_data):
        """Test basic forecasting functionality"""
        request = {
            "data": sample_data,
            "forecast_horizon": 7
        }
        
        response = await forecasting_agent.process_request(request)
        
        assert response.success == True
        assert "forecast" in response.data
        assert len(response.data["forecast"]) == 7
        assert "yhat" in response.data["forecast"][0]
    
    @pytest.mark.asyncio
    async def test_forecasting_invalid_data(self, forecasting_agent):
        """Test forecasting with invalid data"""
        request = {
            "data": {"invalid": "data"},
            "forecast_horizon": 7
        }
        
        response = await forecasting_agent.process_request(request)
        
        assert response.success == False
        assert "error" in response.data
    
    @pytest.mark.asyncio
    async def test_forecasting_missing_columns(self, forecasting_agent):
        """Test forecasting with missing required columns"""
        request = {
            "data": {"ds": ["2023-01-01"], "x": [1.0]},  # Missing 'y' column
            "forecast_horizon": 7
        }
        
        response = await forecasting_agent.process_request(request)
        
        assert response.success == False
        assert "error" in response.data

class TestAnomalyDetectionAgent:
    """Test cases for AnomalyDetectionAgent"""
    
    @pytest.fixture
    def sample_data_with_anomalies(self):
        """Generate sample time series data with anomalies"""
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        values = np.sin(np.arange(len(dates)) * 2 * np.pi / 365) + np.random.normal(0, 0.1, len(dates))
        
        # Add some anomalies
        values[50] = 5.0  # Spike anomaly
        values[150] = -3.0  # Drop anomaly
        values[250:255] = [2.0] * 5  # Sustained anomaly
        
        return {
            "ds": dates.tolist(),
            "y": values.tolist()
        }
    
    @pytest.fixture
    async def anomaly_agent(self):
        """Create an anomaly detection agent instance"""
        agent = AnomalyDetectionAgent(agent_id="test_anomaly")
        await agent.initialize()
        yield agent
        await agent.cleanup()
    
    @pytest.mark.asyncio
    async def test_anomaly_agent_initialization(self, anomaly_agent):
        """Test agent initialization"""
        assert anomaly_agent.agent_id == "test_anomaly"
        assert anomaly_agent.agent_type == "anomaly_detection"
        assert anomaly_agent.is_initialized == True
    
    @pytest.mark.asyncio
    async def test_anomaly_detection_combined(self, anomaly_agent, sample_data_with_anomalies):
        """Test combined anomaly detection"""
        request = {
            "data": sample_data_with_anomalies,
            "method": "combined",
            "threshold": 0.95,
            "window_size": 10
        }
        
        response = await anomaly_agent.process_request(request)
        
        assert response.success == True
        assert "anomalies" in response.data
        assert "confidence" in response.data
        assert len(response.data["anomalies"]) > 0
    
    @pytest.mark.asyncio
    async def test_anomaly_detection_zscore(self, anomaly_agent, sample_data_with_anomalies):
        """Test Z-score anomaly detection"""
        request = {
            "data": sample_data_with_anomalies,
            "method": "zscore",
            "threshold": 2.0,
            "window_size": 10
        }
        
        response = await anomaly_agent.process_request(request)
        
        assert response.success == True
        assert "anomalies" in response.data
        assert response.data["method"] == "zscore"
    
    @pytest.mark.asyncio
    async def test_anomaly_detection_iqr(self, anomaly_agent, sample_data_with_anomalies):
        """Test IQR anomaly detection"""
        request = {
            "data": sample_data_with_anomalies,
            "method": "iqr",
            "threshold": 1.5,
            "window_size": 10
        }
        
        response = await anomaly_agent.process_request(request)
        
        assert response.success == True
        assert "anomalies" in response.data
        assert response.data["method"] == "iqr"
    
    @pytest.mark.asyncio
    async def test_anomaly_detection_isolation_forest(self, anomaly_agent, sample_data_with_anomalies):
        """Test Isolation Forest anomaly detection"""
        request = {
            "data": sample_data_with_anomalies,
            "method": "isolation_forest",
            "threshold": 0.1,
            "window_size": 10
        }
        
        response = await anomaly_agent.process_request(request)
        
        assert response.success == True
        assert "anomalies" in response.data
        assert response.data["method"] == "isolation_forest"
    
    @pytest.mark.asyncio
    async def test_anomaly_detection_invalid_method(self, anomaly_agent, sample_data_with_anomalies):
        """Test anomaly detection with invalid method"""
        request = {
            "data": sample_data_with_anomalies,
            "method": "invalid_method",
            "threshold": 0.95,
            "window_size": 10
        }
        
        response = await anomaly_agent.process_request(request)
        
        assert response.success == False
        assert "error" in response.data

class TestClassificationAgent:
    """Test cases for ClassificationAgent"""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample time series data"""
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        values = np.sin(np.arange(len(dates)) * 2 * np.pi / 365) + np.random.normal(0, 0.1, len(dates))
        
        return {
            "ds": dates.tolist(),
            "y": values.tolist()
        }
    
    @pytest.fixture
    async def classification_agent(self):
        """Create a classification agent instance"""
        agent = ClassificationAgent(agent_id="test_classification")
        await agent.initialize()
        yield agent
        await agent.cleanup()
    
    @pytest.mark.asyncio
    async def test_classification_agent_initialization(self, classification_agent):
        """Test agent initialization"""
        assert classification_agent.agent_id == "test_classification"
        assert classification_agent.agent_type == "classification"
        assert classification_agent.is_initialized == True
    
    @pytest.mark.asyncio
    async def test_classification_comprehensive(self, classification_agent, sample_data):
        """Test comprehensive classification"""
        request = {
            "data": sample_data,
            "method": "comprehensive",
            "classification_type": "pattern"
        }
        
        response = await classification_agent.process_request(request)
        
        assert response.success == True
        assert "classification" in response.data
        assert "confidence" in response.data
    
    @pytest.mark.asyncio
    async def test_classification_pattern(self, classification_agent, sample_data):
        """Test pattern classification"""
        request = {
            "data": sample_data,
            "method": "pattern",
            "classification_type": "pattern"
        }
        
        response = await classification_agent.process_request(request)
        
        assert response.success == True
        assert "classification" in response.data
        assert "pattern_type" in response.data["classification"]
    
    @pytest.mark.asyncio
    async def test_classification_trend(self, classification_agent, sample_data):
        """Test trend classification"""
        request = {
            "data": sample_data,
            "method": "trend",
            "classification_type": "trend"
        }
        
        response = await classification_agent.process_request(request)
        
        assert response.success == True
        assert "classification" in response.data
        assert "trend_type" in response.data["classification"]
    
    @pytest.mark.asyncio
    async def test_classification_seasonality(self, classification_agent, sample_data):
        """Test seasonality classification"""
        request = {
            "data": sample_data,
            "method": "seasonality",
            "classification_type": "seasonality"
        }
        
        response = await classification_agent.process_request(request)
        
        assert response.success == True
        assert "classification" in response.data
        assert "seasonality_type" in response.data["classification"]
    
    @pytest.mark.asyncio
    async def test_classification_behavior(self, classification_agent, sample_data):
        """Test behavior classification"""
        request = {
            "data": sample_data,
            "method": "behavior",
            "classification_type": "behavior"
        }
        
        response = await classification_agent.process_request(request)
        
        assert response.success == True
        assert "classification" in response.data
        assert "behavior_type" in response.data["classification"]
    
    @pytest.mark.asyncio
    async def test_classification_invalid_method(self, classification_agent, sample_data):
        """Test classification with invalid method"""
        request = {
            "data": sample_data,
            "method": "invalid_method",
            "classification_type": "pattern"
        }
        
        response = await classification_agent.process_request(request)
        
        assert response.success == False
        assert "error" in response.data

class TestMasterAgent:
    """Test cases for MasterAgent"""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample time series data"""
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        values = np.sin(np.arange(len(dates)) * 2 * np.pi / 365) + np.random.normal(0, 0.1, len(dates))
        
        return {
            "ds": dates.tolist(),
            "y": values.tolist()
        }
    
    @pytest.fixture
    async def master_agent(self):
        """Create a master agent instance"""
        agent = MasterAgent(agent_id="test_master")
        await agent.initialize()
        yield agent
        await agent.cleanup()
    
    @pytest.mark.asyncio
    async def test_master_agent_initialization(self, master_agent):
        """Test agent initialization"""
        assert master_agent.agent_id == "test_master"
        assert master_agent.agent_type == "master"
        assert master_agent.is_initialized == True
        assert len(master_agent.specialized_agents) > 0
    
    @pytest.mark.asyncio
    async def test_master_agent_comprehensive_analysis(self, master_agent, sample_data):
        """Test comprehensive analysis"""
        request = {
            "data": sample_data,
            "task_type": "comprehensive_analysis"
        }
        
        response = await master_agent.process_request(request)
        
        assert response.success == True
        assert "comprehensive_analysis" in response.data
        assert "results" in response.data
        assert "forecasting" in response.data["results"]
    
    @pytest.mark.asyncio
    async def test_master_agent_forecasting(self, master_agent, sample_data):
        """Test forecasting through master agent"""
        request = {
            "data": sample_data,
            "task_type": "forecasting",
            "forecast_horizon": 7
        }
        
        response = await master_agent.process_request(request)
        
        assert response.success == True
        assert "task_type" in response.data
        assert response.data["task_type"] == "forecasting"
    
    @pytest.mark.asyncio
    async def test_master_agent_anomaly_detection(self, master_agent, sample_data):
        """Test anomaly detection through master agent"""
        request = {
            "data": sample_data,
            "task_type": "anomaly_detection",
            "method": "combined"
        }
        
        response = await master_agent.process_request(request)
        
        assert response.success == True
        assert "task_type" in response.data
        assert response.data["task_type"] == "anomaly_detection"
    
    @pytest.mark.asyncio
    async def test_master_agent_classification(self, master_agent, sample_data):
        """Test classification through master agent"""
        request = {
            "data": sample_data,
            "task_type": "classification",
            "method": "comprehensive"
        }
        
        response = await master_agent.process_request(request)
        
        assert response.success == True
        assert "task_type" in response.data
        assert response.data["task_type"] == "classification"
    
    @pytest.mark.asyncio
    async def test_master_agent_task_routing(self, master_agent):
        """Test automatic task routing"""
        # Test forecasting keyword
        request = {"data": {"ds": ["2023-01-01"], "y": [1.0]}, "forecast": True}
        response = await master_agent.process_request(request)
        assert response.success == True
        
        # Test anomaly keyword
        request = {"data": {"ds": ["2023-01-01"], "y": [1.0]}, "anomaly": True}
        response = await master_agent.process_request(request)
        assert response.success == True
    
    @pytest.mark.asyncio
    async def test_master_agent_status(self, master_agent):
        """Test agent status retrieval"""
        status = await master_agent.get_agent_status()
        
        assert "master_agent" in status
        assert "specialized_agents" in status
        assert len(status["specialized_agents"]) > 0
    
    @pytest.mark.asyncio
    async def test_master_agent_invalid_task(self, master_agent, sample_data):
        """Test handling of invalid task type"""
        request = {
            "data": sample_data,
            "task_type": "invalid_task"
        }
        
        response = await master_agent.process_request(request)
        
        assert response.success == False
        assert "error" in response.data

class TestAgentIntegration:
    """Integration tests for agent interactions"""
    
    @pytest.fixture
    def complex_data(self):
        """Generate complex time series data with multiple patterns"""
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        
        # Create data with trend, seasonality, and anomalies
        trend = np.linspace(0, 2, len(dates))
        seasonality = np.sin(np.arange(len(dates)) * 2 * np.pi / 365) * 0.5
        noise = np.random.normal(0, 0.1, len(dates))
        
        values = trend + seasonality + noise
        
        # Add anomalies
        values[50] = 5.0
        values[150] = -3.0
        values[250:255] = [2.0] * 5
        
        return {
            "ds": dates.tolist(),
            "y": values.tolist()
        }
    
    @pytest.mark.asyncio
    async def test_full_analysis_pipeline(self, complex_data):
        """Test complete analysis pipeline"""
        # Initialize master agent
        master_agent = MasterAgent(agent_id="integration_test")
        await master_agent.initialize()
        
        try:
            # Perform comprehensive analysis
            request = {
                "data": complex_data,
                "task_type": "comprehensive_analysis"
            }
            
            response = await master_agent.process_request(request)
            
            assert response.success == True
            assert "comprehensive_analysis" in response.data
            assert "results" in response.data
            
            results = response.data["results"]
            
            # Check that all agent types are present
            assert "forecasting" in results
            assert "anomaly_detection" in results
            assert "classification" in results
            
            # Check execution times
            assert "execution_times" in response.data
            assert "total_execution_time" in response.data
            
        finally:
            await master_agent.cleanup()
    
    @pytest.mark.asyncio
    async def test_agent_performance_tracking(self, complex_data):
        """Test agent performance tracking"""
        # Initialize agents
        forecasting_agent = ForecastingAgent(agent_id="perf_test_forecast")
        anomaly_agent = AnomalyDetectionAgent(agent_id="perf_test_anomaly")
        classification_agent = ClassificationAgent(agent_id="perf_test_class")
        
        await forecasting_agent.initialize()
        await anomaly_agent.initialize()
        await classification_agent.initialize()
        
        try:
            # Test multiple requests
            for i in range(3):
                # Forecasting
                forecast_request = {"data": complex_data, "forecast_horizon": 7}
                forecast_response = await forecasting_agent.process_request(forecast_request)
                assert forecast_response.success == True
                
                # Anomaly detection
                anomaly_request = {"data": complex_data, "method": "combined"}
                anomaly_response = await anomaly_agent.process_request(anomaly_request)
                assert anomaly_response.success == True
                
                # Classification
                classification_request = {"data": complex_data, "method": "comprehensive"}
                classification_response = await classification_agent.process_request(classification_request)
                assert classification_response.success == True
            
            # Check performance statistics
            forecast_status = forecasting_agent.get_status()
            anomaly_status = anomaly_agent.get_status()
            classification_status = classification_agent.get_status()
            
            assert forecast_status["total_requests"] == 3
            assert anomaly_status["total_requests"] == 3
            assert classification_status["total_requests"] == 3
            
        finally:
            await forecasting_agent.cleanup()
            await anomaly_agent.cleanup()
            await classification_agent.cleanup()

# Utility functions for testing
def create_test_data(length: int = 100, include_anomalies: bool = False) -> Dict[str, list]:
    """Create test time series data"""
    dates = pd.date_range(start='2023-01-01', periods=length, freq='D')
    values = np.sin(np.arange(length) * 2 * np.pi / 365) + np.random.normal(0, 0.1, length)
    
    if include_anomalies:
        # Add some anomalies
        anomaly_indices = [length // 4, length // 2, 3 * length // 4]
        for idx in anomaly_indices:
            values[idx] = values[idx] + np.random.choice([-3, 3])
    
    return {
        "ds": dates.tolist(),
        "y": values.tolist()
    }

def validate_forecast_result(result: Dict[str, Any]) -> bool:
    """Validate forecasting result structure"""
    required_keys = ["forecast", "confidence"]
    return all(key in result for key in required_keys)

def validate_anomaly_result(result: Dict[str, Any]) -> bool:
    """Validate anomaly detection result structure"""
    required_keys = ["anomalies", "method", "confidence"]
    return all(key in result for key in required_keys)

def validate_classification_result(result: Dict[str, Any]) -> bool:
    """Validate classification result structure"""
    required_keys = ["classification", "method", "confidence"]
    return all(key in result for key in required_keys) 