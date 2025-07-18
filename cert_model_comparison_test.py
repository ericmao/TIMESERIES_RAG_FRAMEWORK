#!/usr/bin/env python3
"""
CERT Insider Threat Dataset - Model Comparison Test

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
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Import our existing Markov chain system
from improved_markov_demo import (
    create_event_sequences_improved,
    build_markov_chain,
    compute_similarity_matrix,
    reduce_dimensions,
    detect_anomalies_from_embeddings,
    kl_divergence_similarity,
    cosine_similarity,
    euclidean_similarity
)

# Import our new agents
from src.agents.hmm_anomaly_detection_agent import HMMAnomalyDetectionAgent
from src.agents.crf_anomaly_detection_agent import CRFAnomalyDetectionAgent

class CERTModelComparisonTest:
    """
    Comprehensive comparison of Markov, HMM, and CRF models on CERT dataset
    """
    
    def __init__(self, dataset_path: str = "data/cert_insider_threat"):
        self.dataset_path = dataset_path
        self.results = {}
        self.ground_truth = {}
        self.models = {}
        
        # Initialize models
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize all three model types"""
        # Markov model (uses existing implementation)
        self.models['markov'] = {
            'name': 'Markov Chain',
            'description': 'Markov chain transition matrix with similarity-based anomaly detection'
        }
        
        # HMM model
        self.models['hmm'] = HMMAnomalyDetectionAgent(
            agent_id="cert_hmm",
            n_components=5,
            model_type='multinomial'
        )
        
        # CRF model
        self.models['crf'] = CRFAnomalyDetectionAgent(
            agent_id="cert_crf"
        )
        
    def create_cert_dataset(self):
        """Create synthetic CERT-like dataset for evaluation"""
        print("📊 Creating synthetic CERT Insider Threat dataset...")
        
        # Generate realistic insider threat scenarios
        n_users = 50  # Reduced for faster testing
        n_days = 60
        n_records = n_users * n_days * 24  # Hourly records
        
        # Create synthetic user activity data
        data = []
        malicious_users = np.random.choice(n_users, size=3, replace=False)
        
        for day in range(n_days):
            date = datetime(2024, 1, 1) + timedelta(days=day)
            
            for hour in range(24):
                timestamp = date + timedelta(hours=hour)
                
                for user_id in range(n_users):
                    # Normal user behavior
                    if user_id not in malicious_users:
                        # Regular work hours activity
                        if 8 <= hour <= 18:
                            activity_level = np.random.normal(0.7, 0.2)
                            file_accesses = np.random.poisson(5)
                            network_connections = np.random.poisson(3)
                            data_transfer = np.random.exponential(10)
                            login_events = np.random.poisson(2)
                            privilege_escalation = 0
                        else:
                            activity_level = np.random.normal(0.1, 0.1)
                            file_accesses = np.random.poisson(1)
                            network_connections = np.random.poisson(0.5)
                            data_transfer = np.random.exponential(2)
                            login_events = np.random.poisson(0.2)
                            privilege_escalation = 0
                    
                    # Malicious user behavior
                    else:
                        # Anomalous patterns
                        if day > 20:  # Start malicious activity after 20 days
                            if np.random.random() < 0.4:  # 40% chance of malicious activity
                                activity_level = np.random.normal(0.9, 0.1)
                                file_accesses = np.random.poisson(25)  # High file access
                                network_connections = np.random.poisson(20)  # High network activity
                                data_transfer = np.random.exponential(100)  # High data transfer
                                login_events = np.random.poisson(8)  # Multiple logins
                                privilege_escalation = np.random.poisson(0.3)  # Privilege escalation attempts
                            else:
                                activity_level = np.random.normal(0.1, 0.1)  # Stealth mode
                                file_accesses = np.random.poisson(1)
                                network_connections = np.random.poisson(0.5)
                                data_transfer = np.random.exponential(2)
                                login_events = np.random.poisson(0.2)
                                privilege_escalation = 0
                        else:
                            # Normal behavior initially
                            if 8 <= hour <= 18:
                                activity_level = np.random.normal(0.7, 0.2)
                                file_accesses = np.random.poisson(5)
                                network_connections = np.random.poisson(3)
                                data_transfer = np.random.exponential(10)
                                login_events = np.random.poisson(2)
                                privilege_escalation = 0
                            else:
                                activity_level = np.random.normal(0.1, 0.1)
                                file_accesses = np.random.poisson(1)
                                network_connections = np.random.poisson(0.5)
                                data_transfer = np.random.exponential(2)
                                login_events = np.random.poisson(0.2)
                                privilege_escalation = 0
                    
                    record = {
                        'timestamp': timestamp,
                        'user_id': user_id,
                        'activity_level': max(0, min(1, activity_level)),
                        'file_accesses': max(0, file_accesses),
                        'network_connections': max(0, network_connections),
                        'data_transfer_mb': max(0, data_transfer),
                        'login_events': max(0, login_events),
                        'privilege_escalation_attempts': max(0, privilege_escalation),
                        'is_malicious': user_id in malicious_users and day > 20,
                        'malicious_activity': user_id in malicious_users and day > 20 and np.random.random() < 0.4
                    }
                    data.append(record)
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Save dataset
        os.makedirs(self.dataset_path, exist_ok=True)
        df.to_csv(f"{self.dataset_path}/user_activity.csv", index=False)
        
        # Create ground truth
        ground_truth = {
            'malicious_users': malicious_users.tolist(),
            'malicious_periods': [(21, 60)],  # Days 21-60
            'total_records': len(df),
            'malicious_records': df['malicious_activity'].sum(),
            'normal_records': (~df['malicious_activity']).sum()
        }
        
        with open(f"{self.dataset_path}/ground_truth.json", 'w') as f:
            json.dump(ground_truth, f, indent=2)
        
        print(f"   ✅ Created CERT dataset with {len(df)} records")
        print(f"   ✅ {len(malicious_users)} malicious users identified")
        print(f"   ✅ {ground_truth['malicious_records']} malicious activity records")
        print(f"   ✅ {ground_truth['normal_records']} normal activity records")
        
        return df, ground_truth
    
    def load_cert_data(self):
        """Load CERT dataset"""
        if not os.path.exists(f"{self.dataset_path}/user_activity.csv"):
            return self.create_cert_dataset()
        
        df = pd.read_csv(f"{self.dataset_path}/user_activity.csv")
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        with open(f"{self.dataset_path}/ground_truth.json", 'r') as f:
            ground_truth = json.load(f)
        
        return df, ground_truth
    
    def prepare_data_for_markov(self, df: pd.DataFrame, time_interval: int = 60):
        """Prepare data for Markov chain analysis"""
        # Debug: Check available columns
        print(f"      📊 Available columns: {list(df.columns)}")
        
        # Create time series features
        df = df.sort_values('timestamp')
        
        # Create composite activity score
        df['activity_score'] = (
            df['activity_level'] * 0.3 +
            df['file_accesses'] * 0.2 +
            df['network_connections'] * 0.2 +
            df['data_transfer_mb'] * 0.15 +
            df['login_events'] * 0.1 +
            df['privilege_escalation_attempts'] * 0.05
        )
        
        # Resample to specified interval
        df = df.set_index('timestamp')
        numeric_columns = ['activity_level', 'file_accesses', 'network_connections', 
                          'data_transfer_mb', 'login_events', 'privilege_escalation_attempts',
                          'activity_score', 'is_malicious', 'malicious_activity']
        
        resampled = df[numeric_columns].resample(f'{time_interval}T').mean().fillna(0)
        
        # Create sequences manually for Markov analysis
        # Use the entire dataset as one sequence for simplicity
        sequences = []
        
        # Create activity score for the entire dataset
        df['activity_score'] = (
            df['activity_level'] * 0.3 +
            df['file_accesses'] * 0.2 +
            df['network_connections'] * 0.2 +
            df['data_transfer_mb'] * 0.15 +
            df['login_events'] * 0.1 +
            df['privilege_escalation_attempts'] * 0.05
        )
        
        # Resample the entire dataset
        # Check if timestamp column exists, if not use the first datetime column
        timestamp_col = 'timestamp'
        if timestamp_col not in df.columns:
            # Look for datetime columns
            datetime_cols = df.select_dtypes(include=['datetime64']).columns
            if len(datetime_cols) > 0:
                timestamp_col = datetime_cols[0]
            else:
                print(f"      ⚠️  No timestamp column found, skipping interval {time_interval}")
                return [], resampled
        
        df_resampled = df.set_index(timestamp_col).resample(f'{time_interval}T').mean().fillna(0)
        
        if len(df_resampled) > 10:  # Minimum sequence length
            # Discretize activity scores into states
            activity_scores = df_resampled['activity_score'].values
            states = pd.cut(activity_scores, bins=10, labels=False, duplicates='drop')
            states = states[~np.isnan(states)]  # Remove NaN values
            
            if len(states) > 5:  # Need enough states for transitions
                sequences.append(states.tolist())
        
        # Also create sequences per user if possible
        for user_id in df['user_id'].unique():
            user_data = df[df['user_id'] == user_id].copy()
            if len(user_data) > 10:
                user_resampled = user_data.set_index(timestamp_col).resample(f'{time_interval}T').mean().fillna(0)
                
                if len(user_resampled) > 5:
                    activity_scores = user_resampled['activity_score'].values
                    states = pd.cut(activity_scores, bins=10, labels=False, duplicates='drop')
                    states = states[~np.isnan(states)]
                    
                    if len(states) > 3:  # Shorter minimum for user sequences
                        sequences.append(states.tolist())
        
        return sequences, resampled
    
    def prepare_data_for_hmm(self, df: pd.DataFrame, time_interval: int = 60):
        """Prepare data for HMM analysis"""
        # Use the same sequence creation as Markov but with different state encoding
        sequences, resampled = self.prepare_data_for_markov(df, time_interval)
        
        # For HMM, we'll use the same sequences but ensure they're properly formatted
        hmm_sequences = []
        for seq in sequences:
            if len(seq) > 5:  # Minimum length for HMM
                # Convert to integers and ensure they're within valid range
                hmm_seq = [int(s) for s in seq if not np.isnan(s)]
                if len(hmm_seq) > 5:
                    hmm_sequences.append(hmm_seq)
        
        return hmm_sequences, resampled
    
    def prepare_data_for_crf(self, df: pd.DataFrame, time_interval: int = 60):
        """Prepare data for CRF analysis"""
        # Prepare features for CRF
        resampled = self.prepare_data_for_markov(df, time_interval)
        
        # Create feature sequences
        feature_sequences = []
        label_sequences = []
        
        # Create sequences per user
        for user_id in df['user_id'].unique():
            user_data = df[df['user_id'] == user_id]
            user_resampled = self.prepare_data_for_markov(user_data, time_interval)
            
            if len(user_resampled) > 10:  # Minimum sequence length
                # Create feature dicts for each time step
                user_features = []
                user_labels = []
                
                for idx, row in user_resampled.iterrows():
                    features = {
                        'activity_level': row['activity_level'],
                        'file_accesses': row['file_accesses'],
                        'network_connections': row['network_connections'],
                        'data_transfer': row['data_transfer_mb'],
                        'login_events': row['login_events'],
                        'privilege_escalation': row['privilege_escalation_attempts'],
                        'activity_score': row['activity_score'],
                        'hour': idx.hour,
                        'day_of_week': idx.dayofweek,
                        'is_work_hours': 1 if 8 <= idx.hour <= 18 else 0
                    }
                    user_features.append(features)
                    
                    # Create labels
                    label = 'anomaly' if row['malicious_activity'] else 'normal'
                    user_labels.append(label)
                
                feature_sequences.append(user_features)
                label_sequences.append(user_labels)
        
        return feature_sequences, label_sequences, resampled
    
    def evaluate_markov_model(self, df: pd.DataFrame, ground_truth: dict, time_intervals: list = [60, 120, 720]):
        """Evaluate Markov chain model"""
        print("🔬 Evaluating Markov Chain Model...")
        
        results = {}
        
        for interval in time_intervals:
            print(f"   📊 Processing {interval}min intervals...")
            
            # Prepare data
            sequences, resampled = self.prepare_data_for_markov(df, interval)
            
            # Debug: Check data
            print(f"      📈 Resampled data shape: {resampled.shape}")
            print(f"      📈 Created {len(sequences)} sequences")
            if sequences:
                print(f"      📈 Sequence lengths: {[len(seq) for seq in sequences[:5]]}")
            else:
                print(f"      ⚠️  No sequences created, skipping interval {interval}")
                continue
            
            # Build Markov chains
            markov_chains = []
            for seq in sequences:
                chain = build_markov_chain(seq)
                markov_chains.append(chain)
            
            print(f"      📈 Built {len(markov_chains)} Markov chains")
            
            # Compute similarity matrix
            if len(markov_chains) < 2:
                print(f"      ⚠️  Insufficient Markov chains ({len(markov_chains)}), skipping interval {interval}")
                continue
                
            similarity_matrix = compute_similarity_matrix(markov_chains, kl_divergence_similarity)
            print(f"      📈 Similarity matrix shape: {similarity_matrix.shape}")
            
            # Reduce dimensions
            embeddings = reduce_dimensions(similarity_matrix, method='umap')
            
            # Detect anomalies
            anomalies = detect_anomalies_from_embeddings(embeddings, threshold=0.95)
            
            # Evaluate against ground truth
            metrics = self._evaluate_anomalies(anomalies, resampled, ground_truth, f"markov_{interval}")
            results[f"markov_{interval}"] = metrics
        
        return results
    
    def evaluate_hmm_model(self, df: pd.DataFrame, ground_truth: dict, time_intervals: list = [60, 120, 720]):
        """Evaluate HMM model"""
        print("🔬 Evaluating HMM Model...")
        
        results = {}
        
        for interval in time_intervals:
            print(f"   📊 Processing {interval}min intervals...")
            
            # Prepare data
            sequences, resampled = self.prepare_data_for_hmm(df, interval)
            
            if len(sequences) < 2:
                print(f"      ⚠️  Insufficient sequences for HMM evaluation at {interval}min interval")
                continue
            
            # Split sequences for training/testing
            train_sequences = sequences[:int(len(sequences) * 0.7)]
            test_sequences = sequences[int(len(sequences) * 0.7):]
            
            # Fit HMM
            try:
                self.models['hmm'].fit(train_sequences)
                
                # Detect anomalies
                anomalies = self.models['hmm'].detect_anomalies(test_sequences, threshold=-30.0)
                
                # Convert to time-based anomalies
                time_anomalies = self._convert_sequence_anomalies_to_time(anomalies, resampled, test_sequences)
                
                # Evaluate against ground truth
                metrics = self._evaluate_anomalies(time_anomalies, resampled, ground_truth, f"hmm_{interval}")
                results[f"hmm_{interval}"] = metrics
                
            except Exception as e:
                print(f"      ❌ HMM evaluation failed: {e}")
                results[f"hmm_{interval}"] = {'error': str(e)}
        
        return results
    
    def evaluate_crf_model(self, df: pd.DataFrame, ground_truth: dict, time_intervals: list = [60, 120, 720]):
        """Evaluate CRF model"""
        print("🔬 Evaluating CRF Model...")
        
        results = {}
        
        for interval in time_intervals:
            print(f"   📊 Processing {interval}min intervals...")
            
            # Prepare data
            feature_sequences, label_sequences, resampled = self.prepare_data_for_crf(df, interval)
            
            if len(feature_sequences) < 2:
                print(f"      ⚠️  Insufficient sequences for CRF evaluation at {interval}min interval")
                continue
            
            # Split for training/testing
            split_idx = int(len(feature_sequences) * 0.7)
            train_features = feature_sequences[:split_idx]
            train_labels = label_sequences[:split_idx]
            test_features = feature_sequences[split_idx:]
            test_labels = label_sequences[split_idx:]
            
            # Fit CRF
            try:
                self.models['crf'].fit(train_features, train_labels)
                
                # Predict anomalies
                predictions = self.models['crf'].detect_anomalies(test_features)
                
                # Convert to time-based anomalies
                time_anomalies = self._convert_crf_predictions_to_time(predictions, resampled, test_features)
                
                # Evaluate against ground truth
                metrics = self._evaluate_anomalies(time_anomalies, resampled, ground_truth, f"crf_{interval}")
                results[f"crf_{interval}"] = metrics
                
            except Exception as e:
                print(f"      ❌ CRF evaluation failed: {e}")
                results[f"crf_{interval}"] = {'error': str(e)}
        
        return results
    
    def _convert_sequence_anomalies_to_time(self, anomalies: list, resampled: pd.DataFrame, sequences: list):
        """Convert sequence-based anomalies to time-based anomalies"""
        time_anomalies = []
        
        # Map sequence indices to time indices
        sequence_lengths = [len(seq) for seq in sequences]
        cumulative_length = 0
        
        for anomaly in anomalies:
            seq_idx = anomaly['sequence_index']
            if seq_idx < len(sequences):
                # Map to approximate time indices
                start_idx = cumulative_length
                end_idx = cumulative_length + sequence_lengths[seq_idx]
                
                # Add time-based anomaly
                time_anomalies.append({
                    'start_time': resampled.index[start_idx] if start_idx < len(resampled) else resampled.index[0],
                    'end_time': resampled.index[min(end_idx, len(resampled) - 1)],
                    'confidence': abs(anomaly['log_likelihood']),
                    'method': 'hmm'
                })
            
            cumulative_length += sequence_lengths[seq_idx]
        
        return time_anomalies
    
    def _convert_crf_predictions_to_time(self, predictions: list, resampled: pd.DataFrame, feature_sequences: list):
        """Convert CRF predictions to time-based anomalies"""
        time_anomalies = []
        
        # Map predictions to time indices
        cumulative_length = 0
        
        for pred_sequence in predictions:
            for i, pred in enumerate(pred_sequence):
                if pred == 'anomaly':
                    time_idx = cumulative_length + i
                    if time_idx < len(resampled):
                        time_anomalies.append({
                            'start_time': resampled.index[time_idx],
                            'end_time': resampled.index[time_idx],
                            'confidence': 0.8,  # Default confidence for CRF
                            'method': 'crf'
                        })
            
            cumulative_length += len(pred_sequence)
        
        return time_anomalies
    
    def _evaluate_anomalies(self, anomalies: list, data: pd.DataFrame, ground_truth: dict, model_name: str):
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
                start_idx = data.index.get_loc(anomaly['start_time']) if anomaly['start_time'] in data.index else 0
                end_idx = data.index.get_loc(anomaly['end_time']) if anomaly['end_time'] in data.index else len(data) - 1
                predictions[start_idx:end_idx + 1] = 1
        
        # Create ground truth array
        true_labels = data['malicious_activity'].values
        
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
    
    def run_comprehensive_comparison(self, time_intervals: list = [60, 120, 720]):
        """Run comprehensive comparison of all three models"""
        print("🚀 Starting Comprehensive Model Comparison on CERT Dataset")
        print("=" * 80)
        
        # Load data
        df, ground_truth = self.load_cert_data()
        
        # Evaluate each model
        markov_results = self.evaluate_markov_model(df, ground_truth, time_intervals)
        hmm_results = self.evaluate_hmm_model(df, ground_truth, time_intervals)
        crf_results = self.evaluate_crf_model(df, ground_truth, time_intervals)
        
        # Combine results
        all_results = {**markov_results, **hmm_results, **crf_results}
        
        # Generate comparison report
        self.generate_comparison_report(all_results, time_intervals)
        
        return all_results
    
    def generate_comparison_report(self, results: dict, time_intervals: list):
        """Generate comprehensive comparison report"""
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
        with open('cert_model_comparison_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        summary_df.to_csv('cert_model_comparison_summary.csv', index=False)
        
        # Create visualizations
        self._create_comparison_plots(summary_df, results)
        
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
        print(f"   - cert_model_comparison_results.json")
        print(f"   - cert_model_comparison_summary.csv")
        print(f"   - cert_model_comparison_plots.png")
    
    def _create_comparison_plots(self, summary_df: pd.DataFrame, results: dict):
        """Create comparison plots"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('CERT Dataset: Model Comparison Results', fontsize=16, fontweight='bold')
        
        # Metrics comparison
        metrics = ['Precision', 'Recall', 'F1_Score', 'AUC']
        
        for i, metric in enumerate(metrics):
            ax = axes[0, i]
            
            # Create pivot table for heatmap
            pivot_data = summary_df.pivot(index='Model', columns='Time_Interval', values=metric)
            
            sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='RdYlBu_r', ax=ax)
            ax.set_title(f'{metric} Comparison')
            ax.set_xlabel('Time Interval')
            ax.set_ylabel('Model')
        
        # Anomaly detection counts
        ax1 = axes[1, 0]
        summary_df.groupby('Model')['Total_Anomalies'].mean().plot(kind='bar', ax=ax1)
        ax1.set_title('Average Anomalies Detected')
        ax1.set_ylabel('Count')
        ax1.tick_params(axis='x', rotation=45)
        
        # Confusion matrix components
        ax2 = axes[1, 1]
        confusion_data = summary_df.groupby('Model')[['True_Positives', 'False_Positives', 'False_Negatives']].mean()
        confusion_data.plot(kind='bar', ax=ax2)
        ax2.set_title('Average Confusion Matrix Components')
        ax2.set_ylabel('Count')
        ax2.tick_params(axis='x', rotation=45)
        ax2.legend()
        
        # Model performance radar chart (simplified as bar chart)
        ax3 = axes[1, 2]
        avg_performance = summary_df.groupby('Model')[['Precision', 'Recall', 'F1_Score', 'AUC']].mean()
        avg_performance.plot(kind='bar', ax=ax3)
        ax3.set_title('Average Performance Metrics')
        ax3.set_ylabel('Score')
        ax3.tick_params(axis='x', rotation=45)
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig('cert_model_comparison_plots.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("📊 Comparison plots created and saved!")

def main():
    """Main execution function"""
    print("🔬 CERT Insider Threat Dataset - Model Comparison Test")
    print("=" * 60)
    
    # Initialize comparison test
    comparison_test = CERTModelComparisonTest()
    
    # Run comprehensive comparison
    results = comparison_test.run_comprehensive_comparison(time_intervals=[60, 120, 720])
    
    print("\n🎉 Model comparison completed successfully!")
    print("📁 Check the generated files for detailed results.")

if __name__ == "__main__":
    main() 