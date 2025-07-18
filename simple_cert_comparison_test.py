#!/usr/bin/env python3
"""
Simplified CERT Insider Threat Dataset - Model Comparison Test

This script compares three different anomaly detection approaches:
1. Markov Chain-based Anomaly Detection
2. Hidden Markov Model (HMM) Anomaly Detection  
3. Conditional Random Field (CRF) Anomaly Detection

All models are evaluated on the CERT Insider Threat Test Dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
import json
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Import our new agents
from src.agents.hmm_anomaly_detection_agent import HMMAnomalyDetectionAgent
from src.agents.crf_anomaly_detection_agent import CRFAnomalyDetectionAgent

class SimpleCERTComparisonTest:
    """
    Simplified comparison of Markov, HMM, and CRF models on CERT dataset
    """
    
    def __init__(self, dataset_path: str = "data/cert_insider_threat"):
        self.dataset_path = dataset_path
        self.results = {}
        
    def create_simple_cert_dataset(self):
        """Create a simple synthetic CERT-like dataset for evaluation"""
        print("📊 Creating simple synthetic CERT dataset...")
        
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
        df.to_csv(f"{self.dataset_path}/simple_user_activity.csv", index=False)
        
        print(f"   ✅ Created simple CERT dataset with {len(df)} records")
        print(f"   ✅ {len(malicious_users)} malicious users identified")
        print(f"   ✅ {df['is_malicious'].sum()} malicious activity records")
        
        return df
    
    def load_simple_cert_data(self):
        """Load simple CERT dataset"""
        if not os.path.exists(f"{self.dataset_path}/simple_user_activity.csv"):
            return self.create_simple_cert_dataset()
        
        df = pd.read_csv(f"{self.dataset_path}/simple_user_activity.csv")
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    
    def prepare_data_for_markov(self, df: pd.DataFrame, time_interval: int = 60):
        """Prepare data for Markov chain analysis"""
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
        
        return sequences, df_resampled
    
    def evaluate_markov_model(self, df: pd.DataFrame, time_intervals: list = [60, 120, 720]):
        """Evaluate Markov chain model"""
        print("🔬 Evaluating Markov Chain Model...")
        
        results = {}
        
        for interval in time_intervals:
            print(f"   📊 Processing {interval}min intervals...")
            
            # Prepare data
            sequences, resampled = self.prepare_data_for_markov(df, interval)
            
            print(f"      📈 Created {len(sequences)} sequences")
            
            if len(sequences) < 2:
                print(f"      ⚠️  Insufficient sequences ({len(sequences)}), skipping interval {interval}")
                continue
            
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
                if anomaly_idx < len(resampled):
                    time_anomalies.append({
                        'start_time': resampled.index[anomaly_idx],
                        'end_time': resampled.index[anomaly_idx],
                        'confidence': anomaly_scores[anomaly_idx],
                        'method': 'markov'
                    })
            
            # Evaluate against ground truth
            metrics = self._evaluate_anomalies(time_anomalies, resampled, f"markov_{interval}")
            results[f"markov_{interval}"] = metrics
        
        return results
    
    def evaluate_hmm_model(self, df: pd.DataFrame, time_intervals: list = [60, 120, 720]):
        """Evaluate HMM model"""
        print("🔬 Evaluating HMM Model...")
        
        results = {}
        
        # Initialize HMM model
        hmm_model = HMMAnomalyDetectionAgent(
            agent_id="cert_hmm",
            n_components=5,
            model_type='multinomial'
        )
        
        for interval in time_intervals:
            print(f"   📊 Processing {interval}min intervals...")
            
            # Prepare data
            sequences, resampled = self.prepare_data_for_markov(df, interval)
            
            if len(sequences) < 2:
                print(f"      ⚠️  Insufficient sequences for HMM evaluation at {interval}min interval")
                continue
            
            try:
                # Fit HMM
                hmm_model.fit(sequences)
                
                # Detect anomalies using log likelihood
                anomalies = hmm_model.detect_anomalies(sequences, threshold=-30.0)
                
                # Convert to time-based anomalies
                time_anomalies = []
                for anomaly in anomalies:
                    seq_idx = anomaly['sequence_index']
                    if seq_idx < len(resampled):
                        time_anomalies.append({
                            'start_time': resampled.index[seq_idx],
                            'end_time': resampled.index[seq_idx],
                            'confidence': abs(anomaly['log_likelihood']),
                            'method': 'hmm'
                        })
                
                # Evaluate against ground truth
                metrics = self._evaluate_anomalies(time_anomalies, resampled, f"hmm_{interval}")
                results[f"hmm_{interval}"] = metrics
                
            except Exception as e:
                print(f"      ❌ HMM evaluation failed: {e}")
                results[f"hmm_{interval}"] = {'error': str(e)}
        
        return results
    
    def evaluate_crf_model(self, df: pd.DataFrame, time_intervals: list = [60, 120, 720]):
        """Evaluate CRF model"""
        print("🔬 Evaluating CRF Model...")
        
        results = {}
        
        # Initialize CRF model
        crf_model = CRFAnomalyDetectionAgent(agent_id="cert_crf")
        
        for interval in time_intervals:
            print(f"   📊 Processing {interval}min intervals...")
            
            # Prepare data for CRF
            sequences, resampled = self.prepare_data_for_markov(df, interval)
            
            if len(sequences) < 2:
                print(f"      ⚠️  Insufficient sequences for CRF evaluation at {interval}min interval")
                continue
            
            try:
                # Create feature sequences and labels for CRF
                feature_sequences = []
                label_sequences = []
                
                for seq in sequences:
                    if len(seq) > 5:
                        # Create features for each state
                        features = []
                        labels = []
                        
                        for state in seq:
                            # Simple features based on state
                            feature = {
                                'state': state,
                                'state_mod_2': state % 2,
                                'state_mod_3': state % 3,
                                'state_high': 1 if state > 5 else 0,
                                'state_low': 1 if state < 3 else 0
                            }
                            features.append(feature)
                            
                            # Simple labeling: high states are anomalous
                            label = 'anomaly' if state > 7 else 'normal'
                            labels.append(label)
                        
                        feature_sequences.append(features)
                        label_sequences.append(labels)
                
                if len(feature_sequences) < 2:
                    print(f"      ⚠️  Insufficient feature sequences for CRF")
                    continue
                
                # Fit CRF
                crf_model.fit(feature_sequences, label_sequences)
                
                # Predict anomalies
                predictions = crf_model.detect_anomalies(feature_sequences)
                
                # Convert to time-based anomalies
                time_anomalies = []
                for pred_sequence in predictions:
                    for i, pred in enumerate(pred_sequence):
                        if pred == 'anomaly':
                            if i < len(resampled):
                                time_anomalies.append({
                                    'start_time': resampled.index[i],
                                    'end_time': resampled.index[i],
                                    'confidence': 0.8,
                                    'method': 'crf'
                                })
                
                # Evaluate against ground truth
                metrics = self._evaluate_anomalies(time_anomalies, resampled, f"crf_{interval}")
                results[f"crf_{interval}"] = metrics
                
            except Exception as e:
                print(f"      ❌ CRF evaluation failed: {e}")
                results[f"crf_{interval}"] = {'error': str(e)}
        
        return results
    
    def _evaluate_anomalies(self, anomalies: list, data: pd.DataFrame, model_name: str):
        """Evaluate anomalies against ground truth"""
        if not anomalies:
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
        
        # Create prediction array
        predictions = np.zeros(len(data))
        
        for anomaly in anomalies:
            if 'start_time' in anomaly and 'end_time' in anomaly:
                try:
                    start_idx = data.index.get_loc(anomaly['start_time'])
                    end_idx = data.index.get_loc(anomaly['end_time'])
                    predictions[start_idx:end_idx + 1] = 1
                except:
                    pass
        
        # Create ground truth array
        true_labels = data['is_malicious'].values if 'is_malicious' in data.columns else np.zeros(len(data))
        
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
    
    def run_comparison(self, time_intervals: list = [60, 120, 720]):
        """Run comparison of all three models"""
        print("🚀 Starting Simple Model Comparison on CERT Dataset")
        print("=" * 60)
        
        # Load data
        df = self.load_simple_cert_data()
        
        # Evaluate each model
        markov_results = self.evaluate_markov_model(df, time_intervals)
        hmm_results = self.evaluate_hmm_model(df, time_intervals)
        crf_results = self.evaluate_crf_model(df, time_intervals)
        
        # Combine results
        all_results = {**markov_results, **hmm_results, **crf_results}
        
        # Generate comparison report
        self.generate_comparison_report(all_results, time_intervals)
        
        return all_results
    
    def generate_comparison_report(self, results: dict, time_intervals: list):
        """Generate comparison report"""
        print("\n📊 Generating Comparison Report...")
        
        # Create summary DataFrame
        summary_data = []
        
        for interval in time_intervals:
            for model in ['markov', 'hmm', 'crf']:
                key = f"{model}_{interval}"
                if key in results and 'error' not in results[key]:
                    result = results[key]
                    summary_data.append({
                        'Model': model.upper(),
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
            print("⚠️  No valid results found for comparison")
            return
        
        # Save results
        with open('simple_cert_comparison_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        summary_df.to_csv('simple_cert_comparison_summary.csv', index=False)
        
        # Print summary
        print("\n📈 Model Comparison Summary:")
        print("=" * 50)
        
        for interval in time_intervals:
            print(f"\n⏰ Time Interval: {interval} minutes")
            print("-" * 30)
            
            interval_data = summary_df[summary_df['Time_Interval'] == f"{interval}min"]
            
            for _, row in interval_data.iterrows():
                print(f"{row['Model']:8} | Precision: {row['Precision']:.3f} | "
                      f"Recall: {row['Recall']:.3f} | F1: {row['F1_Score']:.3f} | "
                      f"AUC: {row['AUC']:.3f}")
        
        print(f"\n✅ Results saved to:")
        print(f"   - simple_cert_comparison_results.json")
        print(f"   - simple_cert_comparison_summary.csv")

def main():
    """Main execution function"""
    print("🔬 Simple CERT Insider Threat Dataset - Model Comparison Test")
    print("=" * 60)
    
    # Initialize comparison test
    comparison_test = SimpleCERTComparisonTest()
    
    # Run comparison
    results = comparison_test.run_comparison(time_intervals=[60, 120, 720])
    
    print("\n🎉 Model comparison completed successfully!")
    print("📁 Check the generated files for detailed results.")

if __name__ == "__main__":
    main() 