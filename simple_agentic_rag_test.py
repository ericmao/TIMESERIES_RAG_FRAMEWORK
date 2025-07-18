#!/usr/bin/env python3
"""
Simple AgenticRAG Test with Markov Chain Anomaly Detection for CERT Dataset

This script demonstrates a simplified integration of AgenticRAG with Markov chain
anomaly detection for analyzing the CERT insider threat dataset.
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

class SimpleAgenticRAGTest:
    """
    Simplified test framework for AgenticRAG with Markov chain anomaly detection
    """
    
    def __init__(self, dataset_path: str = "data/cert_insider_threat"):
        self.dataset_path = dataset_path
        self.results = {}
        
    def create_simple_cert_dataset(self):
        """Create a simple synthetic CERT-like dataset for AgenticRAG testing"""
        print("📊 Creating simple CERT dataset for AgenticRAG...")
        
        # Generate simple synthetic data
        n_records = 1000
        n_users = 20
        
        data = []
        malicious_users = [5, 12, 18]  # Specific malicious users
        
        for i in range(n_records):
            user_id = i % n_users
            timestamp = datetime(2024, 1, 1) + timedelta(hours=i)
            
            # Normal behavior
            if user_id not in malicious_users:
                activity_level = np.random.normal(0.6, 0.2)
                file_accesses = np.random.poisson(3)
                network_connections = np.random.poisson(2)
                data_transfer = np.random.exponential(5)
                login_events = np.random.poisson(1)
                privilege_escalation = 0
                is_malicious = False
            else:
                # Malicious behavior (after day 10)
                if i > 240:  # After 10 days (240 hours)
                    if np.random.random() < 0.3:  # 30% chance of malicious activity
                        activity_level = np.random.normal(0.9, 0.1)
                        file_accesses = np.random.poisson(15)
                        network_connections = np.random.poisson(10)
                        data_transfer = np.random.exponential(50)
                        login_events = np.random.poisson(5)
                        privilege_escalation = np.random.poisson(0.5)
                        is_malicious = True
                    else:
                        activity_level = np.random.normal(0.1, 0.1)
                        file_accesses = np.random.poisson(1)
                        network_connections = np.random.poisson(0.5)
                        data_transfer = np.random.exponential(2)
                        login_events = np.random.poisson(0.2)
                        privilege_escalation = 0
                        is_malicious = False
                else:
                    activity_level = np.random.normal(0.6, 0.2)
                    file_accesses = np.random.poisson(3)
                    network_connections = np.random.poisson(2)
                    data_transfer = np.random.exponential(5)
                    login_events = np.random.poisson(1)
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
        
        # Save dataset
        os.makedirs(self.dataset_path, exist_ok=True)
        df.to_csv(f"{self.dataset_path}/simple_agentic_cert_dataset.csv", index=False)
        
        print(f"   ✅ Created simple CERT dataset with {len(df)} records")
        print(f"   ✅ {len(malicious_users)} malicious users identified")
        print(f"   ✅ {df['is_malicious'].sum()} malicious activity records")
        
        return df
    
    def prepare_data_for_markov(self, df: pd.DataFrame, time_interval: int = 60):
        """Prepare data for Markov chain analysis in AgenticRAG format"""
        print(f"🔧 Preparing data for AgenticRAG Markov analysis ({time_interval}min intervals)...")
        
        # Create activity score
        df['activity_score'] = (
            df['activity_level'] * 0.3 +
            df['file_accesses'] * 0.2 +
            df['network_connections'] * 0.2 +
            df['data_transfer_mb'] * 0.15 +
            df['login_events'] * 0.1 +
            df['privilege_escalation_attempts'] * 0.05
        )
        
        # Resample to specified interval
        df_resampled = df.set_index('timestamp').resample(f'{time_interval}T').mean().fillna(0)
        
        # Create sequences by discretizing activity scores
        sequences = []
        activity_scores = df_resampled['activity_score'].values
        
        # Create one sequence from the entire dataset
        if len(activity_scores) > 10:
            states = pd.cut(activity_scores, bins=10, labels=False, duplicates='drop')
            states = states[~np.isnan(states)]
            if len(states) > 5:
                sequences.append(states.tolist())
        
        # Create sequences per user
        for user_id in df['user_id'].unique():
            user_data = df[df['user_id'] == user_id]
            if len(user_data) > 10:
                user_resampled = user_data.set_index('timestamp').resample(f'{time_interval}T').mean().fillna(0)
                if len(user_resampled) > 5:
                    user_scores = user_resampled['activity_score'].values
                    user_states = pd.cut(user_scores, bins=10, labels=False, duplicates='drop')
                    user_states = user_states[~np.isnan(user_states)]
                    if len(user_states) > 3:
                        sequences.append(user_states.tolist())
        
        # Format for AgenticRAG (ds and y columns)
        agentic_data = []
        for idx, row in df_resampled.iterrows():
            agentic_data.append({
                'ds': idx.strftime('%Y-%m-%d %H:%M:%S'),
                'y': row['activity_score'],
                'is_malicious': row['is_malicious'],
                'user_id': 'aggregated',
                'activity_level': row['activity_level'],
                'file_accesses': row['file_accesses'],
                'network_connections': row['network_connections'],
                'data_transfer_mb': row['data_transfer_mb'],
                'login_events': row['login_events'],
                'privilege_escalation': row['privilege_escalation_attempts']
            })
        
        print(f"   ✅ Prepared {len(agentic_data)} data points for AgenticRAG")
        print(f"   ✅ Created {len(sequences)} Markov sequences")
        
        return agentic_data, sequences, df_resampled
    
    def simulate_agentic_rag_markov_analysis(self, sequences: list, data: list, time_interval: int):
        """Simulate AgenticRAG Markov chain analysis"""
        print(f"🤖 Simulating AgenticRAG Markov Chain Analysis ({time_interval}min)...")
        
        if len(sequences) < 2:
            print(f"   ⚠️  Insufficient sequences for Markov analysis")
            return {
                'anomalies': [],
                'sequences_analyzed': len(sequences),
                'method': 'agentic_rag_markov',
                'time_interval': time_interval
            }
        
        # Simple Markov chain analysis
        # Calculate transition probabilities
        all_transitions = []
        for seq in sequences:
            for i in range(len(seq) - 1):
                all_transitions.append((seq[i], seq[i + 1]))
        
        # Count transitions
        transition_counts = {}
        for transition in all_transitions:
            if transition not in transition_counts:
                transition_counts[transition] = 0
            transition_counts[transition] += 1
        
        # Calculate anomaly score based on rare transitions
        total_transitions = len(all_transitions)
        anomaly_scores = []
        
        for seq in sequences:
            seq_anomaly_score = 0
            for i in range(len(seq) - 1):
                transition = (seq[i], seq[i + 1])
                if transition in transition_counts:
                    # Lower probability transitions are more anomalous
                    prob = transition_counts[transition] / total_transitions
                    seq_anomaly_score += -np.log(prob + 1e-10)
            anomaly_scores.append(seq_anomaly_score / max(len(seq) - 1, 1))
        
        # Detect anomalies (top 10% as anomalous)
        threshold = np.percentile(anomaly_scores, 90)
        anomalies = [i for i, score in enumerate(anomaly_scores) if score > threshold]
        
        # Create time-based anomalies
        time_anomalies = []
        for anomaly_idx in anomalies:
            if anomaly_idx < len(data):
                time_anomalies.append({
                    'start_time': data[anomaly_idx]['ds'],
                    'end_time': data[anomaly_idx]['ds'],
                    'confidence': anomaly_scores[anomaly_idx],
                    'method': 'agentic_rag_markov'
                })
        
        return {
            'anomalies': time_anomalies,
            'sequences_analyzed': len(sequences),
            'method': 'agentic_rag_markov',
            'time_interval': time_interval,
            'anomaly_scores': anomaly_scores,
            'threshold': threshold
        }
    
    def evaluate_agentic_results(self, analysis_results: dict, data: list, model_name: str):
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
    
    def run_agentic_rag_test(self, time_intervals: list = [60, 120, 720]):
        """Run AgenticRAG test with Markov chain anomaly detection"""
        print("🚀 Starting AgenticRAG CERT Insider Threat Analysis")
        print("=" * 60)
        
        # Create dataset
        df = self.create_simple_cert_dataset()
        
        results = {}
        
        for interval in time_intervals:
            print(f"\n⏰ Processing {interval}min intervals...")
            
            # Prepare data for AgenticRAG
            agentic_data, sequences, resampled = self.prepare_data_for_markov(df, interval)
            
            # Simulate AgenticRAG Markov analysis
            analysis_results = self.simulate_agentic_rag_markov_analysis(sequences, agentic_data, interval)
            
            # Evaluate results
            metrics = self.evaluate_agentic_results(analysis_results, agentic_data, f"agentic_rag_{interval}")
            results[f"agentic_rag_{interval}"] = metrics
            
            # Save detailed results
            self._save_agentic_results(analysis_results, interval)
        
        # Generate report
        self.generate_agentic_report(results, time_intervals)
        
        return results
    
    def _save_agentic_results(self, results: dict, interval: int):
        """Save detailed AgenticRAG results"""
        filename = f"agentic_rag_results_{interval}min.json"
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"   💾 Detailed results saved to {filename}")
    
    def generate_agentic_report(self, results: dict, time_intervals: list):
        """Generate AgenticRAG report"""
        print("\n📊 Generating AgenticRAG Report...")
        
        # Create summary DataFrame
        summary_data = []
        
        for interval in time_intervals:
            key = f"agentic_rag_{interval}"
            if key in results:
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
        
        # Save results
        with open('agentic_rag_cert_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        summary_df.to_csv('agentic_rag_cert_summary.csv', index=False)
        
        # Print summary
        print("\n📈 AgenticRAG CERT Analysis Summary:")
        print("=" * 50)
        
        for interval in time_intervals:
            print(f"\n⏰ Time Interval: {interval} minutes")
            print("-" * 30)
            
            interval_data = summary_df[summary_df['Time_Interval'] == f"{interval}min"]
            
            for _, row in interval_data.iterrows():
                print(f"{row['Model']:12} | Precision: {row['Precision']:.3f} | "
                      f"Recall: {row['Recall']:.3f} | F1: {row['F1_Score']:.3f} | "
                      f"AUC: {row['AUC']:.3f}")
        
        print(f"\n✅ AgenticRAG results saved to:")
        print(f"   - agentic_rag_cert_results.json")
        print(f"   - agentic_rag_cert_summary.csv")
        
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
        plt.savefig('agentic_rag_cert_performance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("   📊 AgenticRAG visualization saved to: agentic_rag_cert_performance.png")

def main():
    """Main execution function"""
    print("🤖 Simple AgenticRAG with Markov Chain Anomaly Detection")
    print("📊 CERT Insider Threat Dataset Analysis")
    print("=" * 60)
    
    # Initialize test framework
    test_framework = SimpleAgenticRAGTest()
    
    # Run test
    results = test_framework.run_agentic_rag_test(time_intervals=[60, 120, 720])
    
    print("\n🎉 AgenticRAG CERT analysis completed successfully!")
    print("📁 Check the generated files for detailed results and visualizations.")

if __name__ == "__main__":
    main() 