#!/usr/bin/env python3
"""
Real-World Time Series Analysis Examples
=======================================

This script demonstrates practical applications of the Time Series RAG Framework
for common real-world scenarios.
"""

import asyncio
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Import framework components
from src.agents.master_agent import MasterAgent
from src.agents.forecasting_agent import ForecastingAgent
from src.agents.anomaly_detection_agent import AnomalyDetectionAgent
from src.agents.classification_agent import ClassificationAgent
from src.utils.data_processor import TimeSeriesDataProcessor

class RealWorldAnalysisExamples:
    """Demonstrates real-world time series analysis scenarios"""
    
    def __init__(self):
        self.master_agent = None
        self.data_processor = TimeSeriesDataProcessor()
        
    async def initialize(self):
        """Initialize the framework"""
        print("🚀 Initializing Time Series RAG Framework for Real-World Analysis...")
        self.master_agent = MasterAgent(agent_id="real_world_demo")
        await self.master_agent.initialize()
        print("✅ Framework initialized!")
        
    def create_sales_data(self):
        """Create realistic sales data with seasonal patterns and trends"""
        print("\n📈 Creating Sales Data Example...")
        
        # Generate 2 years of daily sales data
        dates = pd.date_range(start='2022-01-01', end='2023-12-31', freq='D')
        
        # Base trend (increasing sales)
        trend = np.linspace(1000, 1500, len(dates))
        
        # Seasonal patterns
        # Weekly pattern (higher sales on weekends)
        weekly = 200 * np.sin(2 * np.pi * np.arange(len(dates)) / 7)
        
        # Monthly pattern (higher sales at month end)
        monthly = 100 * np.sin(2 * np.pi * dates.day / 30)
        
        # Yearly pattern (holiday season boost)
        yearly = 300 * np.sin(2 * np.pi * dates.dayofyear / 365.25)
        
        # Random noise
        noise = np.random.normal(0, 50, len(dates))
        
        # Combine all components
        sales = trend + weekly + monthly + yearly + noise
        
        # Add promotional spikes
        sales = sales.values  # Convert to numpy array for indexing
        sales[60:65] += 500  # March promotion
        sales[180:185] += 400  # July promotion
        sales[300:305] += 600  # November Black Friday
        
        # Add some anomalies
        sales[100] = sales[100] * 0.3  # System outage
        sales[250] = sales[250] * 2.5   # Viral marketing spike
        
        data = pd.DataFrame({
            'ds': dates,
            'y': sales
        })
        
        print(f"✅ Created sales data: {len(data)} days")
        print(f"📊 Sales range: ${data['y'].min():.0f} - ${data['y'].max():.0f}")
        print(f"💰 Average daily sales: ${data['y'].mean():.0f}")
        
        return data
    
    def create_iot_sensor_data(self):
        """Create IoT sensor data with anomalies and patterns"""
        print("\n🌡️ Creating IoT Sensor Data Example...")
        
        # Generate 6 months of hourly temperature data
        dates = pd.date_range(start='2023-01-01', end='2023-06-30', freq='H')
        
        # Base temperature with daily cycle
        base_temp = 20 + 5 * np.sin(2 * np.pi * np.arange(len(dates)) / 24)
        
        # Weekly pattern (weekend vs weekday)
        weekly = 2 * np.sin(2 * np.pi * np.arange(len(dates)) / (24 * 7))
        
        # Seasonal trend (warming over time)
        seasonal = np.linspace(0, 10, len(dates))
        
        # Random noise
        noise = np.random.normal(0, 1, len(dates))
        
        # Combine components
        temperature = base_temp + weekly + seasonal + noise
        
        # Add equipment failures (anomalies)
        temperature = temperature.values  # Convert to numpy array for indexing
        # Sensor malfunction
        temperature[1000:1010] = temperature[1000:1010] + 15
        # HVAC system failure
        temperature[2000:2020] = temperature[2000:2020] - 10
        # Power outage
        temperature[3000:3010] = temperature[3000:3010] + 20
        
        # Add gradual drift (sensor aging)
        temperature[4000:] += np.linspace(0, 5, len(temperature[4000:]))
        
        data = pd.DataFrame({
            'ds': dates,
            'y': temperature
        })
        
        print(f"✅ Created IoT data: {len(data)} hours")
        print(f"🌡️ Temperature range: {data['y'].min():.1f}°C - {data['y'].max():.1f}°C")
        print(f"📊 Average temperature: {data['y'].mean():.1f}°C")
        
        return data
    
    def create_website_traffic_data(self):
        """Create website traffic data with patterns and anomalies"""
        print("\n🌐 Creating Website Traffic Data Example...")
        
        # Generate 1 year of hourly traffic data
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='H')
        
        # Base traffic
        base_traffic = 1000 + 500 * np.sin(2 * np.pi * np.arange(len(dates)) / 24)
        
        # Weekly pattern (lower traffic on weekends)
        weekly = -200 * np.sin(2 * np.pi * np.arange(len(dates)) / (24 * 7))
        
        # Monthly pattern (higher traffic mid-month)
        monthly = 100 * np.sin(2 * np.pi * dates.day / 30)
        
        # Growth trend
        trend = np.linspace(0, 500, len(dates))
        
        # Random noise
        noise = np.random.normal(0, 100, len(dates))
        
        # Combine components
        traffic = base_traffic + weekly + monthly + trend + noise
        
        # Add viral content spikes
        traffic = traffic.values  # Convert to numpy array for indexing
        traffic[1000:1010] *= 3  # Viral post
        traffic[3000:3010] *= 2.5  # Product launch
        traffic[5000:5010] *= 4  # Major announcement
        
        # Add technical issues
        traffic[2000:2005] *= 0.1  # Server outage
        traffic[4000:4010] *= 0.3  # CDN issues
        
        # Add seasonal events
        traffic[6000:6020] *= 1.5  # Holiday season
        traffic[7000:7020] *= 0.7  # Summer slowdown
        
        data = pd.DataFrame({
            'ds': dates,
            'y': traffic
        })
        
        print(f"✅ Created traffic data: {len(data)} hours")
        print(f"👥 Traffic range: {data['y'].min():.0f} - {data['y'].max():.0f} visitors")
        print(f"📊 Average hourly traffic: {data['y'].mean():.0f} visitors")
        
        return data
    
    async def analyze_sales_forecasting(self, data):
        """Analyze sales data for forecasting"""
        print("\n💰 Sales Forecasting Analysis...")
        
        data_dict = {
            "ds": data['ds'].dt.strftime('%Y-%m-%d').tolist(),
            "y": data['y'].tolist()
        }
        
        # Create forecasting request
        request = {
            "task_type": "forecast",
            "data": data_dict,
            "forecast_horizon": 30,  # 30 days ahead
            "description": "Forecast sales for the next 30 days with seasonal patterns"
        }
        
        response = await self.master_agent.process_request(request)
        
        if response.success:
            print("✅ Sales forecasting completed!")
            forecast_data = response.data
            
            # Extract key insights
            forecast_points = forecast_data.get('forecast', [])
            if forecast_points:
                avg_forecast = np.mean([point.get('yhat', 0) for point in forecast_points])
                print(f"📈 Average forecasted sales: ${avg_forecast:.0f}")
                print(f"🎯 Forecast confidence: {forecast_data.get('confidence', 0):.3f}")
                
                # Show trend
                first_forecast = forecast_points[0].get('yhat', 0)
                last_forecast = forecast_points[-1].get('yhat', 0)
                trend = "increasing" if last_forecast > first_forecast else "decreasing"
                print(f"📊 Forecast trend: {trend}")
            
            return response.data
        else:
            print(f"❌ Sales forecasting failed: {response.message}")
            return None
    
    async def analyze_iot_anomalies(self, data):
        """Analyze IoT data for anomalies"""
        print("\n🚨 IoT Anomaly Detection Analysis...")
        
        data_dict = {
            "ds": data['ds'].dt.strftime('%Y-%m-%d %H:%M:%S').tolist(),
            "y": data['y'].tolist()
        }
        
        # Create anomaly detection request
        request = {
            "task_type": "anomaly_detection",
            "data": data_dict,
            "method": "combined",
            "threshold": 0.95,
            "window_size": 24,  # 24-hour window
            "description": "Detect equipment failures and sensor anomalies in IoT data"
        }
        
        response = await self.master_agent.process_request(request)
        
        if response.success:
            print("✅ IoT anomaly detection completed!")
            anomaly_data = response.data
            
            anomalies = anomaly_data.get('anomalies', [])
            print(f"🚨 Detected {len(anomalies)} anomalies")
            print(f"🎯 Detection confidence: {anomaly_data.get('confidence', 0):.3f}")
            
            # Analyze anomaly types
            if anomalies:
                anomaly_values = [anomaly.get('value', 0) for anomaly in anomalies]
                avg_anomaly = np.mean(anomaly_values)
                print(f"📊 Average anomaly value: {avg_anomaly:.1f}°C")
                
                # Categorize anomalies
                high_anomalies = [a for a in anomalies if a.get('value', 0) > avg_anomaly]
                low_anomalies = [a for a in anomalies if a.get('value', 0) <= avg_anomaly]
                
                print(f"🔥 High temperature anomalies: {len(high_anomalies)}")
                print(f"❄️ Low temperature anomalies: {len(low_anomalies)}")
            
            return response.data
        else:
            print(f"❌ IoT anomaly detection failed: {response.message}")
            return None
    
    async def analyze_traffic_patterns(self, data):
        """Analyze website traffic patterns"""
        print("\n🌐 Website Traffic Pattern Analysis...")
        
        data_dict = {
            "ds": data['ds'].dt.strftime('%Y-%m-%d %H:%M:%S').tolist(),
            "y": data['y'].tolist()
        }
        
        # Create classification request
        request = {
            "task_type": "classification",
            "data": data_dict,
            "method": "comprehensive",
            "classification_type": "pattern",
            "description": "Classify website traffic patterns and behavior"
        }
        
        response = await self.master_agent.process_request(request)
        
        if response.success:
            print("✅ Traffic pattern classification completed!")
            classification_data = response.data
            
            classification = classification_data.get('classification', {})
            pattern_info = classification.get('pattern', {})
            
            print(f"🏷️ Pattern type: {pattern_info.get('pattern_type', 'unknown')}")
            print(f"🎯 Classification confidence: {classification_data.get('confidence', 0):.3f}")
            
            # Extract trend and seasonality info
            trend_info = classification.get('trend', {})
            seasonality_info = classification.get('seasonality', {})
            
            print(f"📈 Trend: {trend_info.get('trend_type', 'unknown')}")
            print(f"📅 Seasonality: {seasonality_info.get('seasonality_type', 'unknown')}")
            
            return response.data
        else:
            print(f"❌ Traffic pattern classification failed: {response.message}")
            return None
    
    async def generate_business_insights(self, sales_results, iot_results, traffic_results):
        """Generate business insights from analysis results"""
        print("\n💡 Generating Business Insights...")
        
        insights = {
            "timestamp": datetime.now().isoformat(),
            "sales_insights": {},
            "iot_insights": {},
            "traffic_insights": {},
            "recommendations": []
        }
        
        # Sales insights
        if sales_results:
            forecast_data = sales_results.get('forecasting', {})
            if forecast_data:
                insights["sales_insights"] = {
                    "forecast_confidence": forecast_data.get('confidence', 0),
                    "forecast_horizon": len(forecast_data.get('forecast', [])),
                    "trend_direction": "positive" if forecast_data.get('trend_slope', 0) > 0 else "negative"
                }
                
                if forecast_data.get('confidence', 0) > 0.8:
                    insights["recommendations"].append("High sales forecast confidence - plan inventory accordingly")
                else:
                    insights["recommendations"].append("Moderate sales forecast confidence - monitor closely")
        
        # IoT insights
        if iot_results:
            anomaly_data = iot_results.get('anomaly_detection', {})
            if anomaly_data:
                insights["iot_insights"] = {
                    "total_anomalies": len(anomaly_data.get('anomalies', [])),
                    "anomaly_ratio": anomaly_data.get('anomaly_ratio', 0),
                    "detection_confidence": anomaly_data.get('confidence', 0)
                }
                
                anomaly_ratio = anomaly_data.get('anomaly_ratio', 0)
                if anomaly_ratio > 0.05:
                    insights["recommendations"].append("High anomaly rate in IoT data - investigate equipment health")
                else:
                    insights["recommendations"].append("Normal IoT anomaly rate - system appears healthy")
        
        # Traffic insights
        if traffic_results:
            classification_data = traffic_results.get('classification', {})
            if classification_data:
                pattern_type = classification_data.get('classification', {}).get('pattern', {}).get('pattern_type', '')
                insights["traffic_insights"] = {
                    "pattern_type": pattern_type,
                    "classification_confidence": classification_data.get('confidence', 0)
                }
                
                if 'seasonal' in pattern_type.lower():
                    insights["recommendations"].append("Seasonal traffic patterns detected - optimize for peak periods")
                if 'trend' in pattern_type.lower():
                    insights["recommendations"].append("Growing traffic trend - scale infrastructure accordingly")
        
        # Print insights summary
        print("\n" + "="*60)
        print("💼 BUSINESS INSIGHTS SUMMARY")
        print("="*60)
        
        print(f"\n📅 Analysis Date: {insights['timestamp']}")
        
        if insights["sales_insights"]:
            print(f"\n💰 SALES INSIGHTS:")
            for key, value in insights["sales_insights"].items():
                print(f"  {key}: {value}")
        
        if insights["iot_insights"]:
            print(f"\n🌡️ IoT INSIGHTS:")
            for key, value in insights["iot_insights"].items():
                print(f"  {key}: {value}")
        
        if insights["traffic_insights"]:
            print(f"\n🌐 TRAFFIC INSIGHTS:")
            for key, value in insights["traffic_insights"].items():
                print(f"  {key}: {value}")
        
        if insights["recommendations"]:
            print(f"\n💡 RECOMMENDATIONS:")
            for i, rec in enumerate(insights["recommendations"], 1):
                print(f"  {i}. {rec}")
        
        print("\n" + "="*60)
        
        return insights
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.master_agent:
            await self.master_agent.cleanup()
        print("🧹 Cleanup completed")

async def main():
    """Main function demonstrating real-world analysis"""
    print("🎯 Time Series RAG Framework - Real-World Analysis Examples")
    print("=" * 70)
    
    examples = RealWorldAnalysisExamples()
    
    try:
        # Initialize framework
        await examples.initialize()
        
        # Create different types of real-world data
        sales_data = examples.create_sales_data()
        iot_data = examples.create_iot_sensor_data()
        traffic_data = examples.create_website_traffic_data()
        
        # Perform analyses
        print("\n" + "="*60)
        print("🔍 PERFORMING REAL-WORLD ANALYSES")
        print("="*60)
        
        sales_results = await examples.analyze_sales_forecasting(sales_data)
        iot_results = await examples.analyze_iot_anomalies(iot_data)
        traffic_results = await examples.analyze_traffic_patterns(traffic_data)
        
        # Generate business insights
        insights = await examples.generate_business_insights(sales_results, iot_results, traffic_results)
        
        # Save results
        with open('real_world_analysis_results.json', 'w') as f:
            json.dump(insights, f, indent=2, default=str)
        print(f"\n💾 Results saved to 'real_world_analysis_results.json'")
        
        print("\n🎉 Real-world analysis completed successfully!")
        print("🚀 Framework is ready for production deployment!")
        
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        await examples.cleanup()

if __name__ == "__main__":
    asyncio.run(main()) 