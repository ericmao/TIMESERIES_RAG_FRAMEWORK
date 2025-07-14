#!/usr/bin/env python3
"""
Available Model Comparison Test
==============================

This script compares different available models on HuggingFace.
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

class AvailableModelComparisonTest:
    """Available model comparison test using real models"""
    
    def __init__(self):
        self.results = {
            "gpt2": {},
            "gpt2_medium": {},
            "gpt2_large": {},
            "dialo_gpt_medium": {},
            "dialo_gpt_large": {}
        }
        self.test_data = self._generate_test_data()
        
    def _generate_test_data(self) -> Dict[str, Any]:
        """Generate test data"""
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
        
        # 3. Volatile data
        volatile_data = {
            "ds": dates.strftime('%Y-%m-%d').tolist(),
            "y": np.random.normal(5, 2, len(dates)) + 0.1 * np.arange(len(dates))
        }
        
        return {
            "trend": trend_data,
            "seasonal": seasonal_data,
            "volatile": volatile_data
        }
    
    async def test_model(self, model_name: str, model_config: str) -> Dict[str, Any]:
        """Test a specific model configuration"""
        print(f"\n🧪 Testing {model_name}...")
        
        # Update configuration
        from src.config.config import get_config
        config = get_config()
        config.model.base_lm_model = model_config
        config.model.master_agent_model = model_config
        
        results = {}
        
        # Test each data type
        for data_type, data in self.test_data.items():
            print(f"  📈 Testing {data_type} data...")
            
            try:
                # Initialize agents
                master_agent = MasterAgent(agent_id=f"test_{model_name}_{data_type}")
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
        print("🚀 Starting Available Model Comparison Test")
        print("=" * 60)
        
        # Test GPT-2 models
        self.results["gpt2"] = await self.test_model(
            "GPT-2", 
            "gpt2"
        )
        
        self.results["gpt2_medium"] = await self.test_model(
            "GPT-2 Medium", 
            "gpt2-medium"
        )
        
        self.results["gpt2_large"] = await self.test_model(
            "GPT-2 Large", 
            "gpt2-large"
        )
        
        # Test DialoGPT models
        self.results["dialo_gpt_medium"] = await self.test_model(
            "DialoGPT Medium", 
            "microsoft/DialoGPT-medium"
        )
        
        self.results["dialo_gpt_large"] = await self.test_model(
            "DialoGPT Large", 
            "microsoft/DialoGPT-large"
        )
        
        # Generate comparison report
        self._generate_report()
        self._create_visualizations()
    
    def _generate_report(self):
        """Generate detailed comparison report"""
        print("\n📊 Model Comparison Report")
        print("=" * 60)
        
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
        print(f"{'Model':<20} {'Total Time':<12} {'Avg Time':<10} {'Success Rate':<12}")
        print("-" * 70)
        for model_name, metric in metrics.items():
            print(f"{model_name:<20} {metric['total_time']:<12.2f} {metric['avg_time']:<10.2f} {metric['success_rate']:<12.2%}")
        
        print(f"\n🎯 Confidence Comparison:")
        print(f"{'Model':<20} {'Forecasting':<12} {'Anomaly':<10} {'Classification':<12}")
        print("-" * 70)
        for model_name, metric in metrics.items():
            print(f"{model_name:<20} {metric['avg_forecasting_confidence']:<12.3f} {metric['avg_anomaly_confidence']:<10.3f} {metric['avg_classification_confidence']:<12.3f}")
        
        # Calculate improvements
        if len(metrics) >= 2:
            print(f"\n📈 Model Performance Rankings:")
            
            # Sort by total time (lower is better)
            time_ranking = sorted(metrics.items(), key=lambda x: x[1]['total_time'])
            print(f"  ⏱️  Fastest to Slowest:")
            for i, (model, metric) in enumerate(time_ranking, 1):
                print(f"    {i}. {model}: {metric['total_time']:.2f}s")
            
            # Sort by success rate (higher is better)
            success_ranking = sorted(metrics.items(), key=lambda x: x[1]['success_rate'], reverse=True)
            print(f"  ✅ Success Rate Ranking:")
            for i, (model, metric) in enumerate(success_ranking, 1):
                print(f"    {i}. {model}: {metric['success_rate']:.1%}")
    
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
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Available Models Comparison', fontsize=16)
        
        # 1. Execution time comparison
        if not df_times.empty:
            sns.boxplot(data=df_times, x="model", y="time", ax=axes[0,0])
            axes[0,0].set_title('Execution Time Comparison')
            axes[0,0].set_ylabel('Time (seconds)')
            axes[0,0].tick_params(axis='x', rotation=45)
        
        # 2. Success rate by model
        success_rates = []
        for model in models:
            success_count = sum(1 for data_type in data_types if self.results[model][data_type]["success"])
            success_rates.append({
                "model": model,
                "success_rate": success_count / len(data_types)
            })
        
        df_success = pd.DataFrame(success_rates)
        if not df_success.empty:
            sns.barplot(data=df_success, x="model", y="success_rate", ax=axes[0,1])
            axes[0,1].set_title('Success Rate by Model')
            axes[0,1].set_ylabel('Success Rate')
            axes[0,1].tick_params(axis='x', rotation=45)
        
        # 3. Model size comparison (approximate)
        model_sizes = {
            "gpt2": "124M",
            "gpt2_medium": "355M", 
            "gpt2_large": "774M",
            "dialo_gpt_medium": "345M",
            "dialo_gpt_large": "774M"
        }
        
        size_data = [{"model": model, "size": size} for model, size in model_sizes.items()]
        df_size = pd.DataFrame(size_data)
        
        # Convert size strings to numbers for plotting
        size_mapping = {"124M": 124, "355M": 355, "345M": 345, "774M": 774}
        df_size["size_num"] = df_size["size"].map(size_mapping)
        
        sns.barplot(data=df_size, x="model", y="size_num", ax=axes[1,0])
        axes[1,0].set_title('Model Size Comparison')
        axes[1,0].set_ylabel('Parameters (M)')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # 4. Performance vs Size scatter plot
        if not df_times.empty and not df_size.empty:
            # Calculate average time per model
            avg_times = df_times.groupby("model")["time"].mean().reset_index()
            performance_data = avg_times.merge(df_size, on="model")
            
            sns.scatterplot(data=performance_data, x="size_num", y="time", ax=axes[1,1])
            axes[1,1].set_title('Performance vs Model Size')
            axes[1,1].set_xlabel('Parameters (M)')
            axes[1,1].set_ylabel('Average Time (seconds)')
            
            # Add model labels
            for _, row in performance_data.iterrows():
                axes[1,1].annotate(row["model"], (row["size_num"], row["time"]), 
                                  xytext=(5, 5), textcoords='offset points')
        
        plt.tight_layout()
        plt.savefig('available_model_comparison_results.png', dpi=300, bbox_inches='tight')
        print("  ✅ Visualization saved as 'available_model_comparison_results.png'")
        
        # Save detailed results
        with open('available_model_comparison_results.json', 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print("  ✅ Detailed results saved as 'available_model_comparison_results.json'")

async def main():
    """Main function"""
    print("🎯 Available Models Comparison Test")
    print("=" * 60)
    print("Testing: GPT-2, GPT-2 Medium, GPT-2 Large, DialoGPT Medium, DialoGPT Large")
    
    # Create test instance
    test = AvailableModelComparisonTest()
    
    # Run comparison
    await test.run_comparison()
    
    print("\n🎉 Model comparison completed!")
    print("📊 Check the generated files for detailed results:")
    print("  - available_model_comparison_results.png (visualizations)")
    print("  - available_model_comparison_results.json (detailed data)")

if __name__ == "__main__":
    asyncio.run(main()) 