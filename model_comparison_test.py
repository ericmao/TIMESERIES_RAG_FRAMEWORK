#!/usr/bin/env python3
"""
Model Comparison Test: Breeze-2 8B vs 3B
=========================================

This script compares the performance of Breeze-2 8B vs 3B models
across different time series analysis tasks.
"""

import asyncio
import pandas as pd
import numpy as np
import json
import time
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any

# Import framework components
from src.agents.master_agent import MasterAgent
from src.agents.forecasting_agent import ForecastingAgent
from src.agents.anomaly_detection_agent import AnomalyDetectionAgent
from src.agents.classification_agent import ClassificationAgent
from src.utils.data_processor import TimeSeriesDataProcessor

# Import transformers for model loading
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class ModelComparisonTest:
    """Comprehensive model comparison test"""
    
    def __init__(self):
        self.results = {
            "breeze_3b": {},
            "breeze_8b": {}
        }
        self.test_data = self._generate_test_data()
        
    def _generate_test_data(self) -> Dict[str, Any]:
        """Generate comprehensive test data"""
        print("📊 Generating test data...")
        
        # Create time series with different patterns
        dates = pd.date_range(start='2022-01-01', end='2023-12-31', freq='D')
        
        # 1. Trend data
        trend_data = {
            "ds": dates.strftime('%Y-%m-%d').tolist(),
            "y": np.linspace(0, 10, len(dates)) + np.random.normal(0, 0.5, len(dates))
        }
        
        # 2. Seasonal data
        seasonal_data = {
            "ds": dates.strftime('%Y-%m-%d').tolist(),
            "y": 5 * np.sin(2 * np.pi * np.arange(len(dates)) / 365) + np.random.normal(0, 0.3, len(dates))
        }
        
        # 3. Volatile data with anomalies
        volatile_data = {
            "ds": dates.strftime('%Y-%m-%d').tolist(),
            "y": np.random.normal(0, 1, len(dates))
        }
        # Add anomalies
        volatile_data["y"][50] = 5.0  # Spike
        volatile_data["y"][150] = -4.0  # Drop
        volatile_data["y"][250] = 3.5  # Spike
        
        # 4. Complex pattern data
        complex_data = {
            "ds": dates.strftime('%Y-%m-%d').tolist(),
            "y": (np.linspace(0, 5, len(dates)) + 
                  3 * np.sin(2 * np.pi * np.arange(len(dates)) / 365) +
                  2 * np.sin(2 * np.pi * np.arange(len(dates)) / 30) +
                  np.random.normal(0, 0.2, len(dates)))
        }
        
        return {
            "trend": trend_data,
            "seasonal": seasonal_data,
            "volatile": volatile_data,
            "complex": complex_data
        }
    
    async def test_model(self, model_name: str, model_config: str, use_token: bool = False) -> Dict[str, Any]:
        """Test a specific model configuration"""
        print(f"\n🧪 Testing {model_name}...")
        
        # Update configuration
        from src.config.config import get_config
        config = get_config()
        config.model.base_lm_model = model_config
        config.model.master_agent_model = model_config
        
        # Store token setting for model loading
        self.use_token = use_token
        
        results = {}
        
        # Test each data type
        for data_type, data in self.test_data.items():
            print(f"  📈 Testing {data_type} data...")
            
            try:
                # Initialize agents with token support
                master_agent = MasterAgent(agent_id=f"test_{model_name}_{data_type}")
                
                # Patch the model loading to support tokens
                original_load_models = master_agent._load_models
                async def patched_load_models():
                    try:
                        # Load tokenizer
                        master_agent.tokenizer = AutoTokenizer.from_pretrained(
                            master_agent.model_name,
                            use_auth_token=self.use_token
                        )
                        if master_agent.tokenizer.pad_token is None:
                            master_agent.tokenizer.pad_token = master_agent.tokenizer.eos_token
                        
                        # Load language model
                        master_agent.model = AutoModelForCausalLM.from_pretrained(
                            master_agent.model_name,
                            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                            device_map="auto" if torch.cuda.is_available() else None,
                            use_auth_token=self.use_token
                        )
                        
                        # Load embedding model
                        master_agent.embedding_model = SentenceTransformer(master_agent.config.model.embedding_model)
                        
                        master_agent.logger.info(f"Successfully loaded models for agent {master_agent.agent_id}")
                        
                    except Exception as e:
                        master_agent.logger.error(f"Failed to load models: {str(e)}")
                        raise
                
                master_agent._load_models = patched_load_models
                await master_agent.initialize()
                
                # Test comprehensive analysis
                start_time = time.time()
                request = {
                    "data": data,
                    "task_type": "comprehensive_analysis"
                }
                
                response = await master_agent.process_request(request)
                execution_time = time.time() - start_time
                
                # Extract metrics
                if response.success:
                    forecasting_result = response.data.get("forecasting", {})
                    anomaly_result = response.data.get("anomaly_detection", {})
                    classification_result = response.data.get("classification", {})
                    
                    results[data_type] = {
                        "success": True,
                        "execution_time": execution_time,
                        "forecasting": {
                            "confidence": forecasting_result.get("confidence", 0.0),
                            "forecast_horizon": forecasting_result.get("results", {}).get("forecast_horizon", 0),
                            "forecast_points": len(forecasting_result.get("results", {}).get("forecast", []))
                        },
                        "anomaly_detection": {
                            "confidence": anomaly_result.get("confidence", 0.0),
                            "total_anomalies": anomaly_result.get("results", {}).get("total_anomalies", 0),
                            "anomaly_ratio": anomaly_result.get("results", {}).get("anomaly_ratio", 0.0)
                        },
                        "classification": {
                            "confidence": classification_result.get("confidence", 0.0),
                            "pattern_type": classification_result.get("results", {}).get("classification", {}).get("pattern", {}).get("pattern_type", "unknown")
                        }
                    }
                else:
                    results[data_type] = {
                        "success": False,
                        "error": response.message,
                        "execution_time": execution_time
                    }
                
                await master_agent.cleanup()
                
            except Exception as e:
                print(f"    ❌ Error testing {data_type}: {str(e)}")
                results[data_type] = {
                    "success": False,
                    "error": str(e),
                    "execution_time": 0
                }
        
        return results
    
    async def run_comparison(self):
        """Run the complete model comparison"""
        print("🚀 Starting Model Comparison Test")
        print("=" * 50)
        
        # Test Breeze-2 3B
        self.results["breeze_3b"] = await self.test_model(
            "Breeze-2 3B", 
            "MediaTek-Research/Breeze-2-3B",
            use_token=True  # Enable token authentication
        )
        
        # Test Breeze-2 8B
        self.results["breeze_8b"] = await self.test_model(
            "Breeze-2 8B", 
            "MediaTek-Research/Breeze-2-8B",
            use_token=True  # Enable token authentication
        )
        
        # Generate comparison report
        self._generate_report()
        self._create_visualizations()
    
    def _generate_report(self):
        """Generate detailed comparison report"""
        print("\n📊 Model Comparison Report")
        print("=" * 50)
        
        # Calculate overall metrics
        metrics = {}
        for model_name, results in self.results.items():
            total_time = 0
            success_count = 0
            avg_forecasting_confidence = 0
            avg_anomaly_confidence = 0
            avg_classification_confidence = 0
            
            for data_type, result in results.items():
                if result["success"]:
                    success_count += 1
                    total_time += result["execution_time"]
                    avg_forecasting_confidence += result["forecasting"]["confidence"]
                    avg_anomaly_confidence += result["anomaly_detection"]["confidence"]
                    avg_classification_confidence += result["classification"]["confidence"]
            
            if success_count > 0:
                metrics[model_name] = {
                    "total_time": total_time,
                    "avg_time": total_time / success_count,
                    "success_rate": success_count / len(results),
                    "avg_forecasting_confidence": avg_forecasting_confidence / success_count,
                    "avg_anomaly_confidence": avg_anomaly_confidence / success_count,
                    "avg_classification_confidence": avg_classification_confidence / success_count
                }
        
        # Print comparison
        print(f"\n⏱️  Performance Comparison:")
        print(f"{'Model':<15} {'Total Time':<12} {'Avg Time':<10} {'Success Rate':<12}")
        print("-" * 55)
        for model_name, metric in metrics.items():
            print(f"{model_name:<15} {metric['total_time']:<12.2f} {metric['avg_time']:<10.2f} {metric['success_rate']:<12.2%}")
        
        print(f"\n🎯 Confidence Comparison:")
        print(f"{'Model':<15} {'Forecasting':<12} {'Anomaly':<10} {'Classification':<12}")
        print("-" * 55)
        for model_name, metric in metrics.items():
            print(f"{model_name:<15} {metric['avg_forecasting_confidence']:<12.3f} {metric['avg_anomaly_confidence']:<10.3f} {metric['avg_classification_confidence']:<12.3f}")
        
        # Calculate improvements
        if "breeze_3b" in metrics and "breeze_8b" in metrics:
            print(f"\n📈 Improvements (8B vs 3B):")
            
            # Safe division to avoid ZeroDivisionError
            def safe_improvement(new_val, old_val):
                if old_val == 0:
                    return float('inf') if new_val > 0 else 0.0
                return (new_val - old_val) / old_val * 100
            
            time_improvement = safe_improvement(
                metrics["breeze_8b"]["total_time"], 
                metrics["breeze_3b"]["total_time"]
            )
            forecasting_improvement = safe_improvement(
                metrics["breeze_8b"]["avg_forecasting_confidence"], 
                metrics["breeze_3b"]["avg_forecasting_confidence"]
            )
            anomaly_improvement = safe_improvement(
                metrics["breeze_8b"]["avg_anomaly_confidence"], 
                metrics["breeze_3b"]["avg_anomaly_confidence"]
            )
            classification_improvement = safe_improvement(
                metrics["breeze_8b"]["avg_classification_confidence"], 
                metrics["breeze_3b"]["avg_classification_confidence"]
            )
            
            print(f"  ⏱️  Execution Time: {time_improvement:+.1f}%")
            print(f"  📊 Forecasting Confidence: {forecasting_improvement:+.1f}%")
            print(f"  🚨 Anomaly Detection Confidence: {anomaly_improvement:+.1f}%")
            print(f"  🏷️  Classification Confidence: {classification_improvement:+.1f}%")
    
    def _create_visualizations(self):
        """Create comparison visualizations"""
        print("\n📊 Generating visualizations...")
        
        # Prepare data for plotting
        models = list(self.results.keys())
        data_types = list(self.results[models[0]].keys())
        
        # Execution time comparison
        execution_times = []
        for model in models:
            for data_type in data_types:
                if self.results[model][data_type]["success"]:
                    execution_times.append({
                        "model": model,
                        "data_type": data_type,
                        "time": self.results[model][data_type]["execution_time"]
                    })
        
        df_times = pd.DataFrame(execution_times)
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Breeze-2 8B vs 3B Model Comparison', fontsize=16)
        
        # 1. Execution time comparison
        if not df_times.empty:
            sns.boxplot(data=df_times, x="model", y="time", ax=axes[0,0])
            axes[0,0].set_title('Execution Time Comparison')
            axes[0,0].set_ylabel('Time (seconds)')
        
        # 2. Confidence comparison by task
        confidence_data = []
        for model in models:
            for data_type in data_types:
                if self.results[model][data_type]["success"]:
                    result = self.results[model][data_type]
                    confidence_data.extend([
                        {"model": model, "task": "Forecasting", "confidence": result["forecasting"]["confidence"]},
                        {"model": model, "task": "Anomaly Detection", "confidence": result["anomaly_detection"]["confidence"]},
                        {"model": model, "task": "Classification", "confidence": result["classification"]["confidence"]}
                    ])
        
        df_confidence = pd.DataFrame(confidence_data)
        if not df_confidence.empty:
            sns.boxplot(data=df_confidence, x="task", y="confidence", hue="model", ax=axes[0,1])
            axes[0,1].set_title('Confidence Comparison by Task')
            axes[0,1].set_ylabel('Confidence Score')
        
        # 3. Success rate by data type
        success_rates = []
        for model in models:
            for data_type in data_types:
                success_rates.append({
                    "model": model,
                    "data_type": data_type,
                    "success": self.results[model][data_type]["success"]
                })
        
        df_success = pd.DataFrame(success_rates)
        if not df_success.empty:
            success_pivot = df_success.groupby(['model', 'data_type'])['success'].mean().reset_index()
            sns.barplot(data=success_pivot, x="data_type", y="success", hue="model", ax=axes[1,0])
            axes[1,0].set_title('Success Rate by Data Type')
            axes[1,0].set_ylabel('Success Rate')
        
        # 4. Anomaly detection comparison
        anomaly_data = []
        for model in models:
            for data_type in data_types:
                if self.results[model][data_type]["success"]:
                    anomaly_data.append({
                        "model": model,
                        "data_type": data_type,
                        "anomaly_ratio": self.results[model][data_type]["anomaly_detection"]["anomaly_ratio"]
                    })
        
        df_anomaly = pd.DataFrame(anomaly_data)
        if not df_anomaly.empty:
            sns.barplot(data=df_anomaly, x="data_type", y="anomaly_ratio", hue="model", ax=axes[1,1])
            axes[1,1].set_title('Anomaly Detection Ratio')
            axes[1,1].set_ylabel('Anomaly Ratio')
        
        plt.tight_layout()
        plt.savefig('model_comparison_results.png', dpi=300, bbox_inches='tight')
        print("  ✅ Visualization saved as 'model_comparison_results.png'")
        
        # Save detailed results
        with open('model_comparison_results.json', 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print("  ✅ Detailed results saved as 'model_comparison_results.json'")

async def main():
    """Main function"""
    print("🎯 Breeze-2 8B vs 3B Model Comparison Test")
    print("=" * 60)
    
    # Create test instance
    test = ModelComparisonTest()
    
    # Run comparison
    await test.run_comparison()
    
    print("\n🎉 Model comparison completed!")
    print("📊 Check the generated files for detailed results:")
    print("  - model_comparison_results.png (visualizations)")
    print("  - model_comparison_results.json (detailed data)")

if __name__ == "__main__":
    asyncio.run(main()) 