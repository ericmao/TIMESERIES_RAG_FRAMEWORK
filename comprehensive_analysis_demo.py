#!/usr/bin/env python3
"""
Comprehensive Time Series Analysis Demo
======================================

This script demonstrates the full capabilities of the Time Series RAG Framework
for real-world time series analysis scenarios.
"""

import asyncio
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

# Import framework components
from src.agents.master_agent import MasterAgent
from src.agents.forecasting_agent import ForecastingAgent
from src.agents.anomaly_detection_agent import AnomalyDetectionAgent
from src.agents.classification_agent import ClassificationAgent
from src.utils.data_processor import TimeSeriesDataProcessor
from src.utils.prompt_manager import PromptManager

class ComprehensiveAnalysisDemo:
    """Demonstrates comprehensive time series analysis capabilities"""
    
    def __init__(self):
        self.master_agent = None
        self.data_processor = TimeSeriesDataProcessor()
        self.prompt_manager = PromptManager()
        
    async def initialize(self):
        """Initialize the framework components"""
        print("🚀 Initializing Time Series RAG Framework...")
        
        # Initialize master agent
        self.master_agent = MasterAgent(agent_id="comprehensive_demo")
        await self.master_agent.initialize()
        
        print("✅ Framework initialized successfully!")
        
    async def create_realistic_dataset(self):
        """Create a realistic time series dataset with known patterns"""
        print("\n📊 Creating realistic time series dataset...")
        
        # Generate 2 years of daily data
        dates = pd.date_range(start='2022-01-01', end='2023-12-31', freq='D')
        
        # Base trend (increasing over time)
        trend = np.linspace(0, 2, len(dates))
        
        # Seasonal patterns (yearly and weekly)
        yearly_seasonal = 0.5 * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25)
        weekly_seasonal = 0.2 * np.sin(2 * np.pi * np.arange(len(dates)) / 7)
        
        # Random noise
        noise = np.random.normal(0, 0.1, len(dates))
        
        # Combine components
        values = trend + yearly_seasonal + weekly_seasonal + noise
        
        # Add known anomalies
        # Large spike in March 2022
        values[60] = values[60] + 3.0
        # Sustained anomaly in July 2022
        values[180:185] = values[180:185] + 2.0
        # Drop in November 2022
        values[300] = values[300] - 2.5
        # Volatile period in March 2023
        values[420:430] = values[420:430] + np.random.normal(1.5, 0.5, 10)
        
        # Create DataFrame
        data = pd.DataFrame({
            'ds': dates,
            'y': values
        })
        
        print(f"✅ Created dataset with {len(data)} data points")
        print(f"📈 Date range: {data['ds'].min()} to {data['ds'].max()}")
        print(f"📊 Value range: {data['y'].min():.3f} to {data['y'].max():.3f}")
        
        return data
    
    async def perform_comprehensive_analysis(self, data):
        """Perform comprehensive analysis using the master agent"""
        print("\n🔍 Performing comprehensive analysis...")
        
        # Prepare data for the framework
        data_dict = {
            "ds": data['ds'].dt.strftime('%Y-%m-%d').tolist(),
            "y": data['y'].tolist()
        }
        
        # Create comprehensive analysis request
        request = {
            "task_type": "comprehensive_analysis",
            "data": data_dict,
            "description": "Comprehensive analysis including forecasting, anomaly detection, pattern classification, and trend analysis"
        }
        
        # Process request
        response = await self.master_agent.process_request(request)
        
        if response.success:
            print("✅ Comprehensive analysis completed successfully!")
            return response.data
        else:
            print(f"❌ Analysis failed: {response.message}")
            return None
    
    async def perform_specialized_analysis(self, data):
        """Perform specialized analysis using individual agents"""
        print("\n🎯 Performing specialized analysis...")
        
        data_dict = {
            "ds": data['ds'].dt.strftime('%Y-%m-%d').tolist(),
            "y": data['y'].tolist()
        }
        
        results = {}
        
        # 1. Forecasting Analysis
        print("\n📈 Forecasting Analysis...")
        forecast_agent = ForecastingAgent(agent_id="demo_forecast")
        await forecast_agent.initialize()
        
        forecast_request = {
            "task_type": "forecast",
            "data": data_dict,
            "forecast_horizon": 90,  # 3 months ahead
            "description": "Generate 3-month forecast with confidence intervals"
        }
        
        forecast_response = await forecast_agent.process_request(forecast_request)
        if forecast_response.success:
            results['forecasting'] = forecast_response.data
            print(f"✅ Forecast generated: {len(forecast_response.data.get('forecast', []))} periods")
        else:
            print(f"❌ Forecasting failed: {forecast_response.message}")
        
        await forecast_agent.cleanup()
        
        # 2. Anomaly Detection Analysis
        print("\n🚨 Anomaly Detection Analysis...")
        anomaly_agent = AnomalyDetectionAgent(agent_id="demo_anomaly")
        await anomaly_agent.initialize()
        
        anomaly_request = {
            "task_type": "anomaly_detection",
            "data": data_dict,
            "method": "combined",
            "threshold": 0.95,
            "window_size": 14,
            "description": "Detect anomalies using combined methods"
        }
        
        anomaly_response = await anomaly_agent.process_request(anomaly_request)
        if anomaly_response.success:
            results['anomaly_detection'] = anomaly_response.data
            print(f"✅ Anomalies detected: {len(anomaly_response.data.get('anomalies', []))} points")
        else:
            print(f"❌ Anomaly detection failed: {anomaly_response.message}")
        
        await anomaly_agent.cleanup()
        
        # 3. Classification Analysis
        print("\n🏷️ Pattern Classification Analysis...")
        classification_agent = ClassificationAgent(agent_id="demo_classification")
        await classification_agent.initialize()
        
        classification_request = {
            "task_type": "classification",
            "data": data_dict,
            "method": "comprehensive",
            "classification_type": "pattern",
            "description": "Classify time series patterns and behavior"
        }
        
        classification_response = await classification_agent.process_request(classification_request)
        if classification_response.success:
            results['classification'] = classification_response.data
            print(f"✅ Classification completed with confidence: {classification_response.data.get('confidence', 0):.3f}")
        else:
            print(f"❌ Classification failed: {classification_response.message}")
        
        await classification_agent.cleanup()
        
        return results
    
    async def analyze_data_quality(self, data):
        """Analyze data quality and preprocessing"""
        print("\n🔧 Data Quality Analysis...")
        
        # Validate data
        validation_result = self.data_processor.validate_data(data)
        print(f"✅ Data validation: {'PASS' if validation_result['is_valid'] else 'FAIL'}")
        
        if validation_result['warnings']:
            print(f"⚠️ Warnings: {validation_result['warnings']}")
        
        # Clean data
        cleaned_data = self.data_processor.clean_data(data)
        print(f"✅ Data cleaning: {data.shape[0] - cleaned_data.shape[0]} rows removed")
        
        # Engineer features
        engineered_data = self.data_processor.engineer_features(cleaned_data)
        print(f"✅ Feature engineering: {len(engineered_data.columns)} features created")
        
        # Show feature summary
        numeric_features = engineered_data.select_dtypes(include=[np.number]).columns
        print(f"📊 Numeric features: {list(numeric_features)}")
        
        return {
            'original': data,
            'cleaned': cleaned_data,
            'engineered': engineered_data,
            'validation': validation_result
        }
    
    def generate_insights_report(self, comprehensive_results, specialized_results, data_quality):
        """Generate a comprehensive insights report"""
        print("\n📋 Generating Comprehensive Insights Report...")
        
        report = {
            "analysis_timestamp": datetime.now().isoformat(),
            "dataset_info": {
                "total_points": len(data_quality['original']),
                "date_range": f"{data_quality['original']['ds'].min()} to {data_quality['original']['ds'].max()}",
                "value_range": f"{data_quality['original']['y'].min():.3f} to {data_quality['original']['y'].max():.3f}",
                "data_quality": data_quality['validation']['is_valid']
            },
            "key_findings": {},
            "recommendations": []
        }
        
        # Extract key findings from comprehensive analysis
        if comprehensive_results:
            if 'forecasting' in comprehensive_results:
                forecast_data = comprehensive_results['forecasting']
                report["key_findings"]["forecasting"] = {
                    "forecast_horizon": len(forecast_data.get('forecast', [])),
                    "confidence": forecast_data.get('confidence', 0),
                    "trend_direction": "increasing" if forecast_data.get('trend_slope', 0) > 0 else "decreasing"
                }
            
            if 'anomaly_detection' in comprehensive_results:
                anomaly_data = comprehensive_results['anomaly_detection']
                report["key_findings"]["anomalies"] = {
                    "total_anomalies": len(anomaly_data.get('anomalies', [])),
                    "anomaly_ratio": anomaly_data.get('anomaly_ratio', 0),
                    "detection_confidence": anomaly_data.get('confidence', 0)
                }
            
            if 'classification' in comprehensive_results:
                classification_data = comprehensive_results['classification']
                report["key_findings"]["patterns"] = {
                    "pattern_type": classification_data.get('classification', {}).get('pattern', {}).get('pattern_type', 'unknown'),
                    "trend_strength": classification_data.get('classification', {}).get('trend', {}).get('trend_strength', 0),
                    "seasonality": classification_data.get('classification', {}).get('seasonality', {}).get('seasonality_type', 'unknown')
                }
        
        # Generate recommendations
        if comprehensive_results:
            if 'forecasting' in comprehensive_results:
                forecast_confidence = comprehensive_results['forecasting'].get('confidence', 0)
                if forecast_confidence > 0.8:
                    report["recommendations"].append("High forecast confidence - predictions are reliable")
                else:
                    report["recommendations"].append("Moderate forecast confidence - consider additional data or features")
            
            if 'anomaly_detection' in comprehensive_results:
                anomaly_ratio = comprehensive_results['anomaly_detection'].get('anomaly_ratio', 0)
                if anomaly_ratio > 0.1:
                    report["recommendations"].append("High anomaly rate detected - investigate data quality and external factors")
                else:
                    report["recommendations"].append("Normal anomaly rate - data appears stable")
            
            if 'classification' in comprehensive_results:
                pattern_type = comprehensive_results['classification'].get('classification', {}).get('pattern', {}).get('pattern_type', '')
                if 'trend' in pattern_type.lower():
                    report["recommendations"].append("Strong trend detected - consider trend-based forecasting models")
                if 'seasonal' in pattern_type.lower():
                    report["recommendations"].append("Seasonal patterns detected - use seasonal decomposition methods")
        
        # Print report summary
        print("\n" + "="*60)
        print("📊 COMPREHENSIVE ANALYSIS REPORT")
        print("="*60)
        
        print(f"\n📅 Analysis Date: {report['analysis_timestamp']}")
        print(f"📈 Dataset: {report['dataset_info']['total_points']} points")
        print(f"📊 Date Range: {report['dataset_info']['date_range']}")
        print(f"✅ Data Quality: {'PASS' if report['dataset_info']['data_quality'] else 'FAIL'}")
        
        if 'key_findings' in report:
            print("\n🔍 KEY FINDINGS:")
            for category, findings in report['key_findings'].items():
                print(f"  {category.upper()}:")
                for key, value in findings.items():
                    print(f"    {key}: {value}")
        
        if report['recommendations']:
            print("\n💡 RECOMMENDATIONS:")
            for i, rec in enumerate(report['recommendations'], 1):
                print(f"  {i}. {rec}")
        
        print("\n" + "="*60)
        
        return report
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.master_agent:
            await self.master_agent.cleanup()
        print("🧹 Cleanup completed")

async def main():
    """Main demonstration function"""
    print("🎯 Time Series RAG Framework - Comprehensive Analysis Demo")
    print("=" * 70)
    
    demo = ComprehensiveAnalysisDemo()
    
    try:
        # Initialize framework
        await demo.initialize()
        
        # Create realistic dataset
        data = await demo.create_realistic_dataset()
        
        # Perform comprehensive analysis
        comprehensive_results = await demo.perform_comprehensive_analysis(data)
        
        # Perform specialized analysis
        specialized_results = await demo.perform_specialized_analysis(data)
        
        # Analyze data quality
        data_quality = await demo.analyze_data_quality(data)
        
        # Generate insights report
        report = demo.generate_insights_report(comprehensive_results, specialized_results, data_quality)
        
        # Save results
        with open('comprehensive_analysis_results.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\n💾 Results saved to 'comprehensive_analysis_results.json'")
        
        print("\n🎉 Comprehensive analysis completed successfully!")
        print("🚀 The Time Series RAG Framework is ready for production use!")
        
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        await demo.cleanup()

if __name__ == "__main__":
    asyncio.run(main()) 