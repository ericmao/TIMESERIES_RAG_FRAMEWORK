#!/usr/bin/env python3
"""
Example usage of the Time Series RAG Framework
"""
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

# Import our framework components
from src.agents.master_agent import MasterAgent
from src.utils.data_processor import TimeSeriesDataProcessor
from src.utils.prompt_manager import PromptManager

async def basic_usage_example():
    """Basic usage example with the master agent"""
    print("🚀 Starting Basic Usage Example...")
    
    # Initialize master agent
    agent = MasterAgent(agent_id="example_master")
    await agent.initialize()
    
    # Create sample time series data
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    values = np.sin(np.arange(len(dates)) * 2 * np.pi / 365) + np.random.normal(0, 0.1, len(dates))
    
    # Add some anomalies for testing
    values[50] = 2.0  # Spike anomaly
    values[150] = -1.5  # Drop anomaly
    
    data = {
        "ds": dates.tolist(),
        "y": values.tolist()
    }
    
    print(f"📊 Sample data created: {len(data['ds'])} data points")
    
    # Perform comprehensive analysis
    print("\n🔍 Performing comprehensive analysis...")
    request = {
        "data": data,
        "task_type": "comprehensive_analysis"
    }
    
    response = await agent.process_request(request)
    
    if response.success:
        print("✅ Analysis completed successfully!")
        # Convert timestamps to strings for JSON serialization
        def convert_timestamps(obj):
            if isinstance(obj, dict):
                return {k: convert_timestamps(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_timestamps(item) for item in obj]
            elif hasattr(obj, 'isoformat'):  # Handle datetime/timestamp objects
                return obj.isoformat()
            else:
                return obj
        
        serializable_data = convert_timestamps(response.data)
        print(f"📈 Results: {json.dumps(serializable_data, indent=2)}")
    else:
        print(f"❌ Analysis failed: {response.message}")
    
    # Cleanup
    await agent.cleanup()
    print("\n🧹 Cleanup completed")

async def forecasting_example():
    """Example of forecasting functionality"""
    print("\n🔮 Starting Forecasting Example...")
    
    # Initialize forecasting agent directly
    from src.agents.forecasting_agent import ForecastingAgent
    
    agent = ForecastingAgent(agent_id="example_forecast")
    await agent.initialize()
    
    # Create sample data
    dates = pd.date_range(start='2023-01-01', end='2023-06-30', freq='D')
    values = np.sin(np.arange(len(dates)) * 2 * np.pi / 365) + np.random.normal(0, 0.1, len(dates))
    
    data = {
        "ds": dates.tolist(),
        "y": values.tolist()
    }
    
    # Perform forecasting
    request = {
        "task_type": "forecast",
        "data": data,
        "forecast_horizon": 30  # 30 days ahead
    }
    
    response = await agent.process_request(request)
    
    if response.success:
        print("✅ Forecasting completed!")
        forecast_data = response.data
        print(f"📊 Forecast horizon: {len(forecast_data['forecast'])} periods")
        print(f"🎯 Confidence: {forecast_data.get('confidence', 'N/A')}")
        
        # Show first few forecast points
        print("📈 First 5 forecast points:")
        for i, point in enumerate(forecast_data['forecast'][:5]):
            print(f"  Day {i+1}: {point['yhat']:.3f} (Range: {point['yhat_lower']:.3f} - {point['yhat_upper']:.3f})")
    else:
        print(f"❌ Forecasting failed: {response.message}")
    
    await agent.cleanup()

async def anomaly_detection_example():
    """Example of anomaly detection"""
    print("\n🚨 Starting Anomaly Detection Example...")
    
    from src.agents.anomaly_detection_agent import AnomalyDetectionAgent
    
    agent = AnomalyDetectionAgent(agent_id="example_anomaly")
    await agent.initialize()
    
    # Create data with known anomalies
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    values = np.sin(np.arange(len(dates)) * 2 * np.pi / 365) + np.random.normal(0, 0.1, len(dates))
    
    # Add obvious anomalies
    values[50] = 3.0   # Large spike
    values[150] = -2.0  # Large drop
    values[250:255] = [1.5] * 5  # Sustained anomaly
    
    data = {
        "ds": dates.tolist(),
        "y": values.tolist()
    }
    
    # Test different anomaly detection methods
    methods = ["combined", "zscore", "iqr", "isolation_forest"]
    
    for method in methods:
        print(f"\n🔍 Testing {method} method...")
        
        request = {
            "task_type": "anomaly_detection",
            "data": data,
            "method": method,
            "threshold": 0.95,
            "window_size": 10
        }
        
        response = await agent.process_request(request)
        
        if response.success:
            result = response.data
            print(f"✅ Found {len(result['anomalies'])} anomalies")
            print(f"🎯 Confidence: {result['confidence']:.3f}")
            
            # Show first few anomalies
            if result['anomalies']:
                print("🚨 First 3 anomalies:")
                for i, anomaly in enumerate(result['anomalies'][:3]):
                    print(f"  Anomaly {i+1}: Index {anomaly['index']}, Value {anomaly['value']:.3f}")
        else:
            print(f"❌ {method} method failed: {response.message}")
    
    await agent.cleanup()

async def classification_example():
    """Example of time series classification"""
    print("\n🏷️ Starting Classification Example...")
    
    from src.agents.classification_agent import ClassificationAgent
    
    agent = ClassificationAgent(agent_id="example_classification")
    await agent.initialize()
    
    # Create different types of time series data
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    
    # Trend data
    trend_values = np.linspace(0, 2, len(dates)) + np.random.normal(0, 0.1, len(dates))
    
    # Seasonal data
    seasonal_values = np.sin(np.arange(len(dates)) * 2 * np.pi / 365) + np.random.normal(0, 0.1, len(dates))
    
    # Volatile data
    volatile_values = np.random.normal(0, 1, len(dates))
    
    datasets = [
        ("trend", trend_values),
        ("seasonal", seasonal_values),
        ("volatile", volatile_values)
    ]
    
    for name, values in datasets:
        print(f"\n📊 Classifying {name} data...")
        
        data = {
            "ds": dates.tolist(),
            "y": values.tolist()
        }
        
        request = {
            "task_type": "classification",
            "data": data,
            "method": "comprehensive",
            "classification_type": "pattern"
        }
        
        response = await agent.process_request(request)
        
        if response.success:
            result = response.data
            print(f"✅ Classification completed!")
            print(f"🏷️ Pattern type: {result['classification']['pattern']['pattern_type']}")
            print(f"🎯 Confidence: {result['confidence']:.3f}")
        else:
            print(f"❌ Classification failed: {response.message}")
    
    await agent.cleanup()

async def data_processing_example():
    """Example of data processing utilities"""
    print("\n🔧 Starting Data Processing Example...")
    
    processor = TimeSeriesDataProcessor()
    
    # Create sample data with issues
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    values = np.sin(np.arange(len(dates)) * 2 * np.pi / 365) + np.random.normal(0, 0.1, len(dates))
    
    # Add some issues
    values[50] = np.nan  # Missing value
    values[100] = 1000.0  # Outlier
    values[150] = -1000.0  # Outlier
    
    data = pd.DataFrame({
        "ds": dates,
        "y": values
    })
    
    print(f"📊 Original data shape: {data.shape}")
    print(f"❌ Missing values: {data.isnull().sum().sum()}")
    
    # Validate data
    print("\n🔍 Validating data...")
    validation_result = processor.validate_data(data)
    print(f"✅ Valid: {validation_result['is_valid']}")
    if validation_result['warnings']:
        print(f"⚠️ Warnings: {validation_result['warnings']}")
    
    # Clean data
    print("\n🧹 Cleaning data...")
    cleaned_data = processor.clean_data(data)
    print(f"✅ Cleaned data shape: {cleaned_data.shape}")
    print(f"✅ Missing values after cleaning: {cleaned_data.isnull().sum().sum()}")
    
    # Engineer features
    print("\n⚙️ Engineering features...")
    engineered_data = processor.engineer_features(cleaned_data)
    print(f"✅ Engineered features: {list(engineered_data.columns)}")
    
    # Transform data
    print("\n🔄 Transforming data...")
    transformed_data = processor.transform_data(engineered_data, "standardize")
    print(f"✅ Data transformed and standardized")
    print(f"📊 Transformed data stats - Mean: {transformed_data['y'].mean():.3f}, Std: {transformed_data['y'].std():.3f}")

async def prompt_management_example():
    """Example of prompt management"""
    print("\n📝 Starting Prompt Management Example...")
    
    pm = PromptManager()
    
    # Create a prompt template
    print("📝 Creating prompt template...")
    prompt_id = pm.create_prompt(
        title="Time Series Analysis Template",
        description="Template for analyzing time series data",
        template="Analyze the following time series data: {data}. Focus on {focus_area}.",
        agent_type="forecasting",
        task_type="analysis",
        parameters={"focus_area": "trends and patterns"},
        tags=["analysis", "template", "example"]
    )
    
    print(f"✅ Created prompt with ID: {prompt_id}")
    
    # List prompts
    print("\n📋 Listing all prompts...")
    prompts = pm.get_prompts_by_agent("forecasting")
    print(f"✅ Found {len(prompts)} prompts for forecasting agent")
    
    # Render a prompt
    print("\n🎭 Rendering prompt...")
    rendered = pm.render_prompt(prompt_id, {
        "data": "sample time series data",
        "focus_area": "anomaly detection"
    })
    
    if rendered:
        print(f"✅ Rendered prompt: {rendered}")
    
    # Get statistics
    print("\n📊 Prompt manager statistics...")
    stats = pm.get_statistics()
    print(f"📈 Total prompts: {stats['total_prompts']}")
    print(f"🏷️ Agent types: {stats['agent_types']}")

async def api_example():
    """Example of API usage (simulated)"""
    print("\n🌐 Starting API Example...")
    print("📡 This would typically involve starting the FastAPI server")
    print("🔗 API would be available at http://localhost:8000")
    print("📚 Documentation at http://localhost:8000/docs")
    
    # Simulate API request
    sample_request = {
        "data": {
            "ds": ["2023-01-01", "2023-01-02", "2023-01-03"],
            "y": [1.0, 2.0, 3.0]
        },
        "task_type": "comprehensive_analysis"
    }
    
    print(f"📤 Sample API request: {json.dumps(sample_request, indent=2)}")
    print("✅ API endpoints would process this request and return results")

async def main():
    """Main function to run all examples"""
    print("🎯 Time Series RAG Framework - Example Usage")
    print("=" * 50)
    
    try:
        # Run all examples
        await basic_usage_example()
        await forecasting_example()
        await anomaly_detection_example()
        await classification_example()
        await data_processing_example()
        await prompt_management_example()
        await api_example()
        
        print("\n🎉 All examples completed successfully!")
        print("🚀 Your Time Series RAG Framework is working perfectly!")
        
    except Exception as e:
        print(f"❌ Error running examples: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main()) 