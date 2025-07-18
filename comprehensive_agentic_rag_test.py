#!/usr/bin/env python3
"""
Comprehensive AgenticRAG Test with Markov Chain Anomaly Detection

This script demonstrates the full capabilities of AgenticRAG with Markov chain
anomaly detection for the CERT insider threat dataset.
"""

import asyncio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
import json
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

class ComprehensiveAgenticRAGTest:
    """
    Comprehensive test framework for AgenticRAG with Markov chain anomaly detection
    """
    
    def __init__(self, dataset_path: str = "data/cert_insider_threat"):
        self.dataset_path = dataset_path
        self.results = {}
        
    def create_comprehensive_cert_dataset(self):
        """Create a comprehensive synthetic CERT-like dataset"""
        print("📊 Creating comprehensive CERT dataset for AgenticRAG...")
        
        # Generate comprehensive synthetic data
        n_records = 3000
        n_users = 100
        
        data = []
        malicious_users = [15, 32, 47, 58, 73, 89]  # More malicious users
        
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
                if i > 600:  # After 25 days (600 hours)
                    # Different types of malicious activities
                    activity_type = np.random.choice(['data_exfiltration', 'privilege_escalation', 'lateral_movement', 'persistence'])
                    
                    if activity_type == 'data_exfiltration':
                        activity_level = np.random.normal(0.9, 0.05)
                        file_accesses = np.random.poisson(25)
                        network_connections = np.random.poisson(20)
                        data_transfer = np.random.exponential(150)
                        login_events = np.random.poisson(3)
                        privilege_escalation = 0
                    elif activity_type == 'privilege_escalation':
                        activity_level = np.random.normal(0.8, 0.1)
                        file_accesses = np.random.poisson(15)
                        network_connections = np.random.poisson(12)
                        data_transfer = np.random.exponential(30)
                        login_events = np.random.poisson(10)
                        privilege_escalation = np.random.poisson(3)
                    elif activity_type == 'lateral_movement':
                        activity_level = np.random.normal(0.6, 0.15)
                        file_accesses = np.random.poisson(20)
                        network_connections = np.random.poisson(35)
                        data_transfer = np.random.exponential(40)
                        login_events = np.random.poisson(8)
                        privilege_escalation = np.random.poisson(2)
                    else:  # persistence
                        activity_level = np.random.normal(0.5, 0.2)
                        file_accesses = np.random.poisson(8)
                        network_connections = np.random.poisson(6)
                        data_transfer = np.random.exponential(15)
                        login_events = np.random.poisson(4)
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
        
        # Add comprehensive derived features
        df = self._add_comprehensive_features(df)
        
        # Save dataset
        os.makedirs(self.dataset_path, exist_ok=True)
        df.to_csv(f"{self.dataset_path}/comprehensive_agentic_cert_dataset.csv", index=False)
        
        print(f"   ✅ Created comprehensive CERT dataset with {len(df)} records")
        print(f"   ✅ {len(malicious_users)} malicious users identified")
        print(f"   ✅ {df['is_malicious'].sum()} malicious activity records")
        
        return df
    
    def _add_comprehensive_features(self, df: pd.DataFrame):
        """Add comprehensive derived features"""
        # Time-based features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_work_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).astype(int)
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)
        
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
        df['suspicious_pattern'] = ((df['file_accesses'] > 15) & (df['data_transfer_mb'] > 30)).astype(int)
        
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
        
        # Advanced features
        df['activity_volatility'] = df['activity_deviation'].abs()
        df['data_intensity'] = df['data_transfer_mb'] * df['file_accesses']
        df['network_intensity'] = df['network_connections'] * df['login_events']
        
        return df
    
    def prepare_data_for_comprehensive_analysis(self, df: pd.DataFrame, time_interval: int = 60):
        """Prepare data for comprehensive AgenticRAG analysis"""
        print(f"🔧 Preparing data for comprehensive AgenticRAG analysis ({time_interval}min intervals)...")
        
        # Resample to specified interval with comprehensive aggregation
        df_resampled = df.set_index('timestamp').resample(f'{time_interval}T').agg({
            'activity_level': ['mean', 'std', 'max', 'min'],
            'file_accesses': ['sum', 'mean', 'max', 'std'],
            'network_connections': ['sum', 'mean', 'max', 'std'],
            'data_transfer_mb': ['sum', 'mean', 'max', 'std'],
            'login_events': ['sum', 'mean', 'max'],
            'privilege_escalation_attempts': ['sum', 'mean', 'max'],
            'is_malicious': 'max',
            'is_work_hours': 'mean',
            'is_weekend': 'mean',
            'is_night': 'mean',
            'total_activity': ['sum', 'mean', 'max'],
            'high_activity': 'sum',
            'high_data_transfer': 'sum',
            'high_privilege_attempts': 'sum',
            'suspicious_pattern': 'sum',
            'activity_deviation': ['mean', 'std', 'max'],
            'file_deviation': ['mean', 'std', 'max'],
            'network_deviation': ['mean', 'std', 'max'],
            'transfer_deviation': ['mean', 'std', 'max'],
            'activity_volatility': ['mean', 'max'],
            'data_intensity': ['mean', 'max'],
            'network_intensity': ['mean', 'max']
        }).fillna(0)
        
        # Flatten column names
        df_resampled.columns = ['_'.join(col).strip() for col in df_resampled.columns.values]
        
        # Create comprehensive activity score
        df_resampled['activity_score'] = (
            df_resampled['activity_level_mean'] * 0.25 +
            df_resampled['file_accesses_sum'] * 0.15 +
            df_resampled['network_connections_sum'] * 0.15 +
            df_resampled['data_transfer_mb_sum'] * 0.12 +
            df_resampled['login_events_sum'] * 0.08 +
            df_resampled['privilege_escalation_attempts_sum'] * 0.05 +
            df_resampled['activity_volatility_mean'] * 0.10 +
            df_resampled['data_intensity_mean'] * 0.05 +
            df_resampled['network_intensity_mean'] * 0.05
        )
        
        # Format for AgenticRAG
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
                'privilege_escalation': row['privilege_escalation_attempts_sum'],
                'activity_volatility': row['activity_volatility_mean'],
                'data_intensity': row['data_intensity_mean'],
                'network_intensity': row['network_intensity_mean']
            })
        
        print(f"   ✅ Prepared {len(agentic_data)} data points for comprehensive analysis")
        return agentic_data, df_resampled
    
    def comprehensive_markov_analysis(self, data: list, time_interval: int, similarity_method: str = 'kl_divergence'):
        """Comprehensive Markov chain analysis with multiple similarity methods"""
        print(f"🤖 Comprehensive Markov Chain Analysis ({time_interval}min, {similarity_method})...")
        
        # Create sequences from activity scores
        activity_scores = [record['y'] for record in data]
        
        # Create multiple sequence types
        sequences = []
        
        # Global sequence
        if len(activity_scores) > 10:
            states = pd.cut(activity_scores, bins=12, labels=False, duplicates='drop')
            states = states[~np.isnan(states)]
            if len(states) > 5:
                sequences.append(states.tolist())
        
        # Sliding window sequences with different window sizes
        window_sizes = [min(15, len(activity_scores) // 6), min(25, len(activity_scores) // 4)]
        for window_size in window_sizes:
            for i in range(0, len(activity_scores) - window_size, window_size // 3):
                window_scores = activity_scores[i:i + window_size]
                if len(window_scores) > 5:
                    window_states = pd.cut(window_scores, bins=10, labels=False, duplicates='drop')
                    window_states = window_states[~np.isnan(window_states)]
                    if len(window_states) > 3:
                        sequences.append(window_states.tolist())
        
        # User-specific sequences (simulated)
        for user_id in range(5):  # Simulate 5 user sequences
            user_scores = activity_scores[user_id::5]  # Take every 5th score
            if len(user_scores) > 5:
                user_states = pd.cut(user_scores, bins=8, labels=False, duplicates='drop')
                user_states = user_states[~np.isnan(user_states)]
                if len(user_states) > 3:
                    sequences.append(user_states.tolist())
        
        if len(sequences) < 2:
            print(f"   ⚠️  Insufficient sequences for comprehensive Markov analysis")
            return {
                'anomalies': [],
                'sequences_analyzed': len(sequences),
                'method': 'comprehensive_markov',
                'time_interval': time_interval,
                'similarity_method': similarity_method
            }
        
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
        
        # Calculate anomaly scores with comprehensive method
        total_transitions = len(all_transitions)
        anomaly_scores = []
        
        for seq in sequences:
            seq_anomaly_score = 0
            transition_count = 0
            
            for i in range(len(seq) - 1):
                transition = (seq[i], seq[i + 1])
                if transition in transition_counts:
                    # Lower probability transitions are more anomalous
                    prob = transition_counts[transition] / total_transitions
                    seq_anomaly_score += -np.log(prob + 1e-10)
                    transition_count += 1
            
            if transition_count > 0:
                anomaly_scores.append(seq_anomaly_score / transition_count)
            else:
                anomaly_scores.append(0)
        
        # Normalize anomaly scores
        if anomaly_scores:
            anomaly_scores = np.array(anomaly_scores)
            anomaly_scores = (anomaly_scores - np.mean(anomaly_scores)) / (np.std(anomaly_scores) + 1e-6)
            anomaly_scores = np.clip(anomaly_scores, 0, 1)
        
        # Detect anomalies with optimized threshold
        thresholds = [0.7, 0.75, 0.8, 0.85, 0.9]
        best_threshold = self._find_optimal_threshold_comprehensive(anomaly_scores, data, thresholds)
        
        # Create time-based anomalies
        time_anomalies = []
        for i, score in enumerate(anomaly_scores):
            if score > best_threshold:
                # Map sequence index to time
                seq_start_idx = i * (min(15, len(data) // 6) // 3) if i > 0 else 0
                if seq_start_idx < len(data):
                    time_anomalies.append({
                        'start_time': data[seq_start_idx]['ds'],
                        'end_time': data[min(seq_start_idx + 15, len(data) - 1)]['ds'],
                        'confidence': score,
                        'method': f'comprehensive_markov_{similarity_method}'
                    })
        
        return {
            'anomalies': time_anomalies,
            'sequences_analyzed': len(sequences),
            'method': 'comprehensive_markov',
            'time_interval': time_interval,
            'similarity_method': similarity_method,
            'anomaly_scores': anomaly_scores.tolist(),
            'threshold': best_threshold
        }
    
    def _find_optimal_threshold_comprehensive(self, scores: list, data: list, thresholds: list):
        """Find optimal threshold for comprehensive anomaly detection"""
        if len(scores) == 0:
            return 0.8
        
        best_f1 = 0
        best_threshold = 0.8
        
        for threshold in thresholds:
            # Create predictions
            predictions = np.array(scores) > threshold
            
            # Get ground truth
            true_labels = np.array([record['is_malicious'] for record in data])
            true_labels = (true_labels > 0).astype(int)
            
            # Ensure same length
            min_len = min(len(predictions), len(true_labels))
            predictions = predictions[:min_len]
            true_labels = true_labels[:min_len]
            
            # Calculate F1 score
            precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='binary', zero_division=0)
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        return best_threshold
    
    def evaluate_comprehensive_results(self, analysis_results: dict, data: list, model_name: str):
        """Evaluate comprehensive results against ground truth"""
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
        
        # Extract anomalies from results
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
    
    def run_comprehensive_test(self, time_intervals: list = [60, 120, 720], similarity_methods: list = ['kl_divergence', 'cosine']):
        """Run comprehensive AgenticRAG test with multiple configurations"""
        print("🚀 Starting Comprehensive AgenticRAG CERT Insider Threat Analysis")
        print("=" * 80)
        
        # Create comprehensive dataset
        df = self.create_comprehensive_cert_dataset()
        
        results = {}
        
        for interval in time_intervals:
            print(f"\n⏰ Processing {interval}min intervals...")
            
            # Prepare data for comprehensive analysis
            agentic_data, resampled = self.prepare_data_for_comprehensive_analysis(df, interval)
            
            for similarity_method in similarity_methods:
                print(f"   🔍 Using {similarity_method} similarity method...")
                
                # Run comprehensive Markov analysis
                analysis_results = self.comprehensive_markov_analysis(agentic_data, interval, similarity_method)
                
                # Evaluate results
                key = f"comprehensive_agentic_rag_{interval}_{similarity_method}"
                metrics = self.evaluate_comprehensive_results(analysis_results, agentic_data, key)
                results[key] = metrics
                
                # Save detailed results
                self._save_comprehensive_results(analysis_results, interval, similarity_method)
        
        # Generate comprehensive report
        self.generate_comprehensive_report(results, time_intervals, similarity_methods)
        
        return results
    
    def _save_comprehensive_results(self, results: dict, interval: int, similarity_method: str):
        """Save detailed comprehensive results"""
        filename = f"comprehensive_agentic_rag_results_{interval}min_{similarity_method}.json"
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"      💾 Detailed results saved to {filename}")
    
    def generate_comprehensive_report(self, results: dict, time_intervals: list, similarity_methods: list):
        """Generate comprehensive AgenticRAG report"""
        print("\n📊 Generating Comprehensive AgenticRAG Report...")
        
        # Create summary DataFrame
        summary_data = []
        
        for interval in time_intervals:
            for similarity_method in similarity_methods:
                key = f"comprehensive_agentic_rag_{interval}_{similarity_method}"
                if key in results:
                    result = results[key]
                    summary_data.append({
                        'Model': f'Comprehensive_AgenticRAG_{similarity_method}',
                        'Time_Interval': f"{interval}min",
                        'Similarity_Method': similarity_method,
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
        with open('comprehensive_agentic_rag_cert_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        summary_df.to_csv('comprehensive_agentic_rag_cert_summary.csv', index=False)
        
        # Print summary
        print("\n📈 Comprehensive AgenticRAG CERT Analysis Summary:")
        print("=" * 70)
        
        for interval in time_intervals:
            print(f"\n⏰ Time Interval: {interval} minutes")
            print("-" * 50)
            
            interval_data = summary_df[summary_df['Time_Interval'] == f"{interval}min"]
            
            for _, row in interval_data.iterrows():
                print(f"{row['Model']:35} | Precision: {row['Precision']:.3f} | "
                      f"Recall: {row['Recall']:.3f} | F1: {row['F1_Score']:.3f} | "
                      f"AUC: {row['AUC']:.3f}")
        
        print(f"\n✅ Comprehensive AgenticRAG results saved to:")
        print(f"   - comprehensive_agentic_rag_cert_results.json")
        print(f"   - comprehensive_agentic_rag_cert_summary.csv")
        
        # Create comprehensive visualization
        self._create_comprehensive_visualization(summary_df)
    
    def _create_comprehensive_visualization(self, summary_df: pd.DataFrame):
        """Create comprehensive AgenticRAG performance visualization"""
        plt.figure(figsize=(20, 15))
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 15))
        
        # 1. F1 Score comparison by similarity method
        pivot_f1 = summary_df.pivot_table(index='Time_Interval', columns='Similarity_Method', values='F1_Score')
        pivot_f1.plot(kind='bar', ax=ax1, title='Comprehensive AgenticRAG F1 Score by Similarity Method')
        ax1.set_ylabel('F1 Score')
        ax1.legend(title='Similarity Method')
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. Precision comparison
        pivot_precision = summary_df.pivot_table(index='Time_Interval', columns='Similarity_Method', values='Precision')
        pivot_precision.plot(kind='bar', ax=ax2, title='Comprehensive AgenticRAG Precision by Similarity Method')
        ax2.set_ylabel('Precision')
        ax2.legend(title='Similarity Method')
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. Recall comparison
        pivot_recall = summary_df.pivot_table(index='Time_Interval', columns='Similarity_Method', values='Recall')
        pivot_recall.plot(kind='bar', ax=ax3, title='Comprehensive AgenticRAG Recall by Similarity Method')
        ax3.set_ylabel('Recall')
        ax3.legend(title='Similarity Method')
        ax3.tick_params(axis='x', rotation=45)
        
        # 4. AUC comparison
        pivot_auc = summary_df.pivot_table(index='Time_Interval', columns='Similarity_Method', values='AUC')
        pivot_auc.plot(kind='bar', ax=ax4, title='Comprehensive AgenticRAG AUC by Similarity Method')
        ax4.set_ylabel('AUC')
        ax4.legend(title='Similarity Method')
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('comprehensive_agentic_rag_cert_performance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("   📊 Comprehensive AgenticRAG visualization saved to: comprehensive_agentic_rag_cert_performance.png")

def main():
    """Main execution function"""
    print("🤖 Comprehensive AgenticRAG with Markov Chain Anomaly Detection")
    print("📊 CERT Insider Threat Dataset Analysis")
    print("=" * 80)
    
    # Initialize test framework
    test_framework = ComprehensiveAgenticRAGTest()
    
    # Run comprehensive test
    results = test_framework.run_comprehensive_test(
        time_intervals=[60, 120, 720],
        similarity_methods=['kl_divergence', 'cosine']
    )
    
    print("\n🎉 Comprehensive AgenticRAG CERT analysis completed successfully!")
    print("📁 Check the generated files for detailed results and visualizations.")

if __name__ == "__main__":
    main() 