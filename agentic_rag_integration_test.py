#!/usr/bin/env python3
"""
AgenticRAG Integration Test with Markov Chain Anomaly Detection

This script demonstrates proper integration of AgenticRAG with the Markov chain
anomaly detection agent for the CERT insider threat dataset.
"""

import asyncio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import json
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Import the actual Markov agent
from src.agents.markov_anomaly_detection_agent import MarkovAnomalyDetectionAgent

class AgenticRAGIntegrationTest:
    """
    Integration test framework for AgenticRAG with Markov chain anomaly detection
    """
    
    def __init__(self, dataset_path: str = "data/cert_insider_threat"):
        self.dataset_path = dataset_path
        self.results = {}
        self.agent = None
        
    def create_enhanced_cert_dataset(self):
        """Create an enhanced synthetic CERT-like dataset for AgenticRAG testing"""
        print("📊 Creating enhanced CERT dataset for AgenticRAG...")
        
        # Generate enhanced synthetic data
        n_records = 2000
        n_users = 50
        
        data = []
        malicious_users = [12, 25, 38, 42]  # Malicious users
        
        # Create realistic activity patterns
        for i in range(n_records):
            user_id = i % n_users
            timestamp = datetime(2024, 1, 1) + timedelta(hours=i)
            
            # Normal behavior with realistic patterns
            if user_id not in malicious_users:
                # Work hours pattern (9 AM - 5 PM)
                hour = timestamp.hour
                work_hours = 9 <= hour <= 17
                
                if work_hours:
                    activity_level = np.random.normal(0.7, 0.15)
                    file_accesses = np.random.poisson(5)
                    network_connections = np.random.poisson(3)
                    data_transfer = np.random.exponential(8)
                    login_events = np.random.poisson(2)
                    privilege_escalation = 0
                else:
                    activity_level = np.random.normal(0.2, 0.1)
                    file_accesses = np.random.poisson(1)
                    network_connections = np.random.poisson(0.5)
                    data_transfer = np.random.exponential(2)
                    login_events = np.random.poisson(0.2)
                    privilege_escalation = 0
                
                is_malicious = False
            else:
                # Malicious behavior with sophisticated patterns
                if i > 480:  # After 20 days (480 hours)
                    # Different types of malicious activities
                    activity_type = np.random.choice(['data_exfiltration', 'privilege_escalation', 'lateral_movement'])
                    
                    if activity_type == 'data_exfiltration':
                        activity_level = np.random.normal(0.9, 0.05)
                        file_accesses = np.random.poisson(20)
                        network_connections = np.random.poisson(15)
                        data_transfer = np.random.exponential(100)
                        login_events = np.random.poisson(3)
                        privilege_escalation = 0
                    elif activity_type == 'privilege_escalation':
                        activity_level = np.random.normal(0.8, 0.1)
                        file_accesses = np.random.poisson(10)
                        network_connections = np.random.poisson(8)
                        data_transfer = np.random.exponential(20)
                        login_events = np.random.poisson(8)
                        privilege_escalation = np.random.poisson(2)
                    else:  # lateral_movement
                        activity_level = np.random.normal(0.6, 0.15)
                        file_accesses = np.random.poisson(15)
                        network_connections = np.random.poisson(25)
                        data_transfer = np.random.exponential(30)
                        login_events = np.random.poisson(5)
                        privilege_escalation = np.random.poisson(1)
                    
                    is_malicious = True
                else:
                    # Normal behavior before attack
                    hour = timestamp.hour
                    work_hours = 9 <= hour <= 17
                    
                    if work_hours:
                        activity_level = np.random.normal(0.7, 0.15)
                        file_accesses = np.random.poisson(5)
                        network_connections = np.random.poisson(3)
                        data_transfer = np.random.exponential(8)
                        login_events = np.random.poisson(2)
                        privilege_escalation = 0
                    else:
                        activity_level = np.random.normal(0.2, 0.1)
                        file_accesses = np.random.poisson(1)
                        network_connections = np.random.poisson(0.5)
                        data_transfer = np.random.exponential(2)
                        login_events = np.random.poisson(0.2)
                        privilege_escalation = 0
                    
                    is_malicious = False
            
            record = {
                'timestamp': timestamp,
                'user_id': user_id,
                'activity_level': max(0, min(1, activity_level)),
                'file_accesses': max(0, file_accesses),
                'network_connections': max(0, network_connections),
                'data_transfer_mb': max(0, data_transfer),
                'login_events': max(0, login_events),
                'privilege_escalation_attempts': max(0, privilege_escalation),
                'is_malicious': is_malicious
            }
            data.append(record)
        
        df = pd.DataFrame(data)
        
        # Add derived features
        df = self._add_derived_features(df)
        
        # Save dataset
        os.makedirs(self.dataset_path, exist_ok=True)
        df.to_csv(f"{self.dataset_path}/agentic_cert_dataset.csv", index=False)
        
        print(f"   ✅ Created enhanced CERT dataset with {len(df)} records")
        print(f"   ✅ {len(malicious_users)} malicious users identified")
        print(f"   ✅ {df['is_malicious'].sum()} malicious activity records")
        
        return df
    
    def _add_derived_features(self, df: pd.DataFrame):
        """Add sophisticated derived features"""
        # Time-based features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_work_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).astype(int)
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Activity patterns
        df['total_activity'] = (df['file_accesses'] + df['network_connections'] + 
                               df['login_events'] + df['privilege_escalation_attempts'])
        
        # Rate-based features
        df['file_access_rate'] = df['file_accesses'] / (df['total_activity'] + 1)
        df['network_rate'] = df['network_connections'] / (df['total_activity'] + 1)
        df['data_transfer_rate'] = df['data_transfer_mb'] / (df['total_activity'] + 1)
        
        # Anomaly indicators
        df['high_activity'] = (df['activity_level'] > 0.8).astype(int)
        df['high_data_transfer'] = (df['data_transfer_mb'] > 50).astype(int)
        df['high_privilege_attempts'] = (df['privilege_escalation_attempts'] > 0).astype(int)
        
        # User behavior patterns
        user_stats = df.groupby('user_id').agg({
            'activity_level': ['mean', 'std'],
            'file_accesses': ['mean', 'std'],
            'network_connections': ['mean', 'std'],
            'data_transfer_mb': ['mean', 'std']
        }).reset_index()
        
        user_stats.columns = ['user_id', 'avg_activity', 'std_activity', 
                            'avg_files', 'std_files', 'avg_network', 'std_network',
                            'avg_transfer', 'std_transfer']
        
        df = df.merge(user_stats, on='user_id', how='left')
        
        # Deviation from user baseline
        df['activity_deviation'] = (df['activity_level'] - df['avg_activity']) / (df['std_activity'] + 1e-6)
        df['file_deviation'] = (df['file_accesses'] - df['avg_files']) / (df['std_files'] + 1e-6)
        df['network_deviation'] = (df['network_connections'] - df['avg_network']) / (df['std_network'] + 1e-6)
        df['transfer_deviation'] = (df['data_transfer_mb'] - df['avg_transfer']) / (df['std_transfer'] + 1e-6)
        
        return df
    
    def prepare_data_for_agentic_rag(self, df: pd.DataFrame, time_interval: int = 60):
        """Prepare data in the format expected by AgenticRAG"""
        print(f"🔧 Preparing data for AgenticRAG ({time_interval}min intervals)...")
        
        # Resample to specified interval
        df_resampled = df.set_index('timestamp').resample(f'{time_interval}T').agg({
            'activity_level': ['mean', 'std', 'max'],
            'file_accesses': ['sum', 'mean', 'max'],
            'network_connections': ['sum', 'mean', 'max'],
            'data_transfer_mb': ['sum', 'mean', 'max'],
            'login_events': ['sum', 'mean'],
            'privilege_escalation_attempts': ['sum', 'mean'],
            'is_malicious': 'max',
            'is_work_hours': 'mean',
            'is_weekend': 'mean',
            'total_activity': ['sum', 'mean'],
            'high_activity': 'sum',
            'high_data_transfer': 'sum',
            'high_privilege_attempts': 'sum',
            'activity_deviation': ['mean', 'std'],
            'file_deviation': ['mean', 'std'],
            'network_deviation': ['mean', 'std'],
            'transfer_deviation': ['mean', 'std']
        }).fillna(0)
        
        # Flatten column names
        df_resampled.columns = ['_'.join(col).strip() for col in df_resampled.columns.values]
        
        # Create activity score for AgenticRAG
        df_resampled['activity_score'] = (
            df_resampled['activity_level_mean'] * 0.3 +
            df_resampled['file_accesses_sum'] * 0.2 +
            df_resampled['network_connections_sum'] * 0.2 +
            df_resampled['data_transfer_mb_sum'] * 0.15 +
            df_resampled['login_events_sum'] * 0.1 +
            df_resampled['privilege_escalation_attempts_sum'] * 0.05
        )
        
        # Format for AgenticRAG (ds and y columns)
        agentic_data = []
        for idx, row in df_resampled.iterrows():
            agentic_data.append({
                'ds': idx.strftime('%Y-%m-%d %H:%M:%S'),
                'y': row['activity_score'],
                'is_malicious': row['is_malicious_max'],
                'user_id': 'aggregated',
                'activity_level': row['activity_level_mean'],
                'file_accesses': row['file_accesses_sum'],
                'network_connections': row['network_connections_sum'],
                'data_transfer_mb': row['data_transfer_mb_sum'],
                'login_events': row['login_events_sum'],
                'privilege_escalation': row['privilege_escalation_attempts_sum']
            })
        
        print(f"   ✅ Prepared {len(agentic_data)} data points for AgenticRAG")
        return agentic_data
    
    async def initialize_agentic_rag_agent(self):
        """Initialize the AgenticRAG agent with Markov chain anomaly detection"""
        print("🤖 Initializing AgenticRAG with Markov Chain Anomaly Detection...")
        
        try:
            # Initialize the Markov anomaly detection agent
            self.agent = MarkovAnomalyDetectionAgent(
                agent_id="agentic_rag_cert",
                model_name="microsoft/DialoGPT-medium"  # Lightweight model for testing
            )
            
            # Initialize the agent
            success = await self.agent.initialize()
            
            if success:
                print("   ✅ AgenticRAG agent initialized successfully")
                return True
            else:
                print("   ❌ Failed to initialize AgenticRAG agent")
                return False
                
        except Exception as e:
            print(f"   ❌ Error initializing AgenticRAG agent: {str(e)}")
            return False
    
    async def run_agentic_rag_analysis(self, data: list, time_intervals: list = [60, 120, 720]):
        """Run AgenticRAG analysis with Markov chain anomaly detection"""
        print("🔬 Running AgenticRAG Analysis...")
        
        results = {}
        
        for interval in time_intervals:
            print(f"   📊 Processing {interval}min intervals...")
            
            # Prepare data for this interval
            interval_data = self.prepare_data_for_agentic_rag(pd.DataFrame(data), interval)
            
            # Create request for AgenticRAG
            request = {
                'data': interval_data,
                'time_intervals': [interval],
                'similarity_method': 'kl_divergence',
                'reduction_method': 'umap',
                'anomaly_threshold': 0.95,
                'analysis_type': 'insider_threat_detection'
            }
            
            try:
                # Process request through AgenticRAG
                response = await self.agent.process_request(request)
                
                if response.success:
                    print(f"      ✅ AgenticRAG analysis completed for {interval}min")
                    
                    # Extract results
                    analysis_results = response.data
                    
                    # Evaluate against ground truth
                    metrics = self._evaluate_agentic_results(analysis_results, interval_data, f"agentic_rag_{interval}")
                    results[f"agentic_rag_{interval}"] = metrics
                    
                    # Save detailed results
                    self._save_agentic_results(analysis_results, interval)
                    
                else:
                    print(f"      ❌ AgenticRAG analysis failed for {interval}min: {response.message}")
                    results[f"agentic_rag_{interval}"] = {'error': response.message}
                    
            except Exception as e:
                print(f"      ❌ Error in AgenticRAG analysis: {str(e)}")
                results[f"agentic_rag_{interval}"] = {'error': str(e)}
        
        return results
    
    def _evaluate_agentic_results(self, analysis_results: dict, data: list, model_name: str):
        """Evaluate AgenticRAG results against ground truth"""
        if 'anomalies' not in analysis_results:
            return {
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'auc': 0.0,
                'total_anomalies': 0,
                'true_positives': 0,
                'false_positives': 0,
                'false_negatives': 0
            }
        
        # Extract anomalies from AgenticRAG results
        anomalies = analysis_results['anomalies']
        
        # Create prediction array
        predictions = np.zeros(len(data))
        
        for anomaly in anomalies:
            if 'start_time' in anomaly and 'end_time' in anomaly:
                try:
                    start_time = pd.to_datetime(anomaly['start_time'])
                    end_time = pd.to_datetime(anomaly['end_time'])
                    
                    # Find matching time periods
                    for i, record in enumerate(data):
                        record_time = pd.to_datetime(record['ds'])
                        if start_time <= record_time <= end_time:
                            predictions[i] = 1
                except:
                    pass
        
        # Create ground truth array
        true_labels = np.array([record['is_malicious'] for record in data])
        
        # Ensure both arrays are binary
        true_labels = (true_labels > 0).astype(int)
        predictions = (predictions > 0).astype(int)
        
        # Calculate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='binary', zero_division=0)
        
        try:
            auc = roc_auc_score(true_labels, predictions)
        except:
            auc = 0.0
        
        # Calculate confusion matrix components
        tn, fp, fn, tp = confusion_matrix(true_labels, predictions).ravel()
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc,
            'total_anomalies': len(anomalies),
            'true_positives': int(tp),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_negatives': int(tn)
        }
    
    def _save_agentic_results(self, results: dict, interval: int):
        """Save detailed AgenticRAG results"""
        filename = f"agentic_rag_integration_results_{interval}min.json"
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"      💾 Detailed results saved to {filename}")
    
    def generate_agentic_report(self, results: dict, time_intervals: list):
        """Generate comprehensive AgenticRAG report"""
        print("\n📊 Generating AgenticRAG Integration Report...")
        
        # Create summary DataFrame
        summary_data = []
        
        for interval in time_intervals:
            key = f"agentic_rag_{interval}"
            if key in results and 'error' not in results[key]:
                result = results[key]
                summary_data.append({
                    'Model': 'AgenticRAG',
                    'Time_Interval': f"{interval}min",
                    'Precision': result['precision'],
                    'Recall': result['recall'],
                    'F1_Score': result['f1_score'],
                    'AUC': result['auc'],
                    'Total_Anomalies': result['total_anomalies'],
                    'True_Positives': result['true_positives'],
                    'False_Positives': result['false_positives'],
                    'False_Negatives': result['false_negatives']
                })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Check if we have any results
        if summary_df.empty:
            print("⚠️  No valid AgenticRAG results found")
            return
        
        # Save results
        with open('agentic_rag_integration_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        summary_df.to_csv('agentic_rag_integration_summary.csv', index=False)
        
        # Print summary
        print("\n📈 AgenticRAG Integration Analysis Summary:")
        print("=" * 60)
        
        for interval in time_intervals:
            print(f"\n⏰ Time Interval: {interval} minutes")
            print("-" * 40)
            
            interval_data = summary_df[summary_df['Time_Interval'] == f"{interval}min"]
            
            for _, row in interval_data.iterrows():
                print(f"{row['Model']:12} | Precision: {row['Precision']:.3f} | "
                      f"Recall: {row['Recall']:.3f} | F1: {row['F1_Score']:.3f} | "
                      f"AUC: {row['AUC']:.3f}")
        
        print(f"\n✅ AgenticRAG integration results saved to:")
        print(f"   - agentic_rag_integration_results.json")
        print(f"   - agentic_rag_integration_summary.csv")
        
        # Create visualization
        self._create_agentic_visualization(summary_df)
    
    def _create_agentic_visualization(self, summary_df: pd.DataFrame):
        """Create AgenticRAG performance visualization"""
        plt.figure(figsize=(15, 10))
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. F1 Score comparison
        pivot_f1 = summary_df.pivot(index='Time_Interval', columns='Model', values='F1_Score')
        pivot_f1.plot(kind='bar', ax=ax1, title='AgenticRAG F1 Score by Time Interval')
        ax1.set_ylabel('F1 Score')
        ax1.legend(title='Model')
        
        # 2. Precision comparison
        pivot_precision = summary_df.pivot(index='Time_Interval', columns='Model', values='Precision')
        pivot_precision.plot(kind='bar', ax=ax2, title='AgenticRAG Precision by Time Interval')
        ax2.set_ylabel('Precision')
        ax2.legend(title='Model')
        
        # 3. Recall comparison
        pivot_recall = summary_df.pivot(index='Time_Interval', columns='Model', values='Recall')
        pivot_recall.plot(kind='bar', ax=ax3, title='AgenticRAG Recall by Time Interval')
        ax3.set_ylabel('Recall')
        ax3.legend(title='Model')
        
        # 4. AUC comparison
        pivot_auc = summary_df.pivot(index='Time_Interval', columns='Model', values='AUC')
        pivot_auc.plot(kind='bar', ax=ax4, title='AgenticRAG AUC by Time Interval')
        ax4.set_ylabel('AUC')
        ax4.legend(title='Model')
        
        plt.tight_layout()
        plt.savefig('agentic_rag_integration_performance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("   📊 AgenticRAG visualization saved to: agentic_rag_integration_performance.png")
    
    async def run_comprehensive_test(self, time_intervals: list = [60, 120, 720]):
        """Run comprehensive AgenticRAG test on CERT dataset"""
        print("🚀 Starting AgenticRAG CERT Insider Threat Analysis")
        print("=" * 60)
        
        # Create enhanced dataset
        df = self.create_enhanced_cert_dataset()
        
        # Initialize AgenticRAG agent
        agent_initialized = await self.initialize_agentic_rag_agent()
        
        if not agent_initialized:
            print("❌ Failed to initialize AgenticRAG agent. Exiting.")
            return None
        
        # Run AgenticRAG analysis
        results = await self.run_agentic_rag_analysis(df.to_dict('records'), time_intervals)
        
        # Generate comprehensive report
        self.generate_agentic_report(results, time_intervals)
        
        return results

async def main():
    """Main execution function"""
    print("🤖 AgenticRAG Integration with Markov Chain Anomaly Detection")
    print("📊 CERT Insider Threat Dataset Analysis")
    print("=" * 60)
    
    # Initialize test framework
    test_framework = AgenticRAGIntegrationTest()
    
    # Run comprehensive test
    results = await test_framework.run_comprehensive_test(time_intervals=[60, 120, 720])
    
    if results:
        print("\n🎉 AgenticRAG CERT analysis completed successfully!")
        print("📁 Check the generated files for detailed results and visualizations.")
    else:
        print("\n❌ AgenticRAG CERT analysis failed.")

if __name__ == "__main__":
    asyncio.run(main()) 