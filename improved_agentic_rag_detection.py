#!/usr/bin/env python3
"""
Improved AgenticRAG Detection System

This script implements the immediate fixes identified in the wrong detection analysis:
1. Dynamic thresholding based on user behavior
2. Enhanced feature engineering with sequence-based features
3. Multi-modal detection combining multiple approaches
4. User-specific baselines
5. Better similarity measures
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
import json
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve
from sklearn.preprocessing import StandardScaler
from scipy.stats import entropy
from scipy.spatial.distance import cosine, euclidean
import warnings
warnings.filterwarnings('ignore')

class ImprovedAgenticRAGDetection:
    """
    Improved AgenticRAG detection system with dynamic thresholding and enhanced features
    """
    
    def __init__(self, dataset_path: str = "data/cert_insider_threat"):
        self.dataset_path = dataset_path
        self.user_baselines = {}
        self.dynamic_thresholds = {}
        self.feature_scaler = StandardScaler()
        
    def load_dataset(self):
        """Load the comprehensive CERT dataset"""
        print("📊 Loading comprehensive CERT dataset...")
        
        df = pd.read_csv(f"{self.dataset_path}/comprehensive_agentic_cert_dataset.csv")
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        print(f"   ✅ Loaded dataset with {len(df)} records")
        print(f"   ✅ {df['is_malicious'].sum()} malicious activities")
        print(f"   ✅ {len(df['user_id'].unique())} unique users")
        
        return df
    
    def create_user_baselines(self, df: pd.DataFrame):
        """Create user-specific behavior baselines"""
        print("📊 Creating user-specific behavior baselines...")
        
        # Group by user and calculate baseline statistics
        user_stats = df.groupby('user_id').agg({
            'activity_level': ['mean', 'std', 'min', 'max'],
            'file_accesses': ['mean', 'std', 'max'],
            'network_connections': ['mean', 'std', 'max'],
            'data_transfer_mb': ['mean', 'std', 'max'],
            'login_events': ['mean', 'std'],
            'privilege_escalation_attempts': ['mean', 'std'],
            'total_activity': ['mean', 'std'],
            'activity_deviation': ['mean', 'std'],
            'activity_volatility': ['mean', 'std'],
            'data_intensity': ['mean', 'std'],
            'network_intensity': ['mean', 'std']
        }).fillna(0)
        
        # Flatten column names
        user_stats.columns = ['_'.join(col).strip() for col in user_stats.columns.values]
        
        # Store baselines for each user
        for user_id in df['user_id'].unique():
            if user_id in user_stats.index:
                self.user_baselines[user_id] = user_stats.loc[user_id].to_dict()
        
        print(f"   ✅ Created baselines for {len(self.user_baselines)} users")
        
        return user_stats
    
    def extract_enhanced_features(self, df: pd.DataFrame, interval: int = 720):
        """Extract enhanced features with sequence-based and contextual information"""
        print(f"🔧 Extracting enhanced features for {interval}min intervals...")
        
        # Resample data to the specified interval
        df_resampled = df.set_index('timestamp').resample(f'{interval}T').agg({
            'activity_level': ['mean', 'std', 'max'],
            'file_accesses': ['sum', 'mean', 'max'],
            'network_connections': ['sum', 'mean', 'max'],
            'data_transfer_mb': ['sum', 'mean', 'max'],
            'login_events': ['sum', 'mean'],
            'privilege_escalation_attempts': ['sum', 'mean'],
            'is_malicious': 'max',
            'user_id': 'first',
            'is_work_hours': 'mean',
            'is_weekend': 'mean',
            'total_activity': ['sum', 'mean'],
            'high_activity': 'sum',
            'high_data_transfer': 'sum',
            'high_privilege_attempts': 'sum',
            'activity_deviation': ['mean', 'std'],
            'file_deviation': ['mean', 'std'],
            'network_deviation': ['mean', 'std'],
            'transfer_deviation': ['mean', 'std'],
            'activity_volatility': ['mean', 'max'],
            'data_intensity': ['mean', 'max'],
            'network_intensity': ['mean', 'max']
        }).fillna(0)
        
        # Flatten column names
        df_resampled.columns = ['_'.join(col).strip() for col in df_resampled.columns.values]
        
        # Add enhanced features
        df_resampled = self._add_sequence_features(df_resampled)
        df_resampled = self._add_contextual_features(df_resampled)
        df_resampled = self._add_user_specific_features(df_resampled)
        
        return df_resampled
    
    def _add_sequence_features(self, df: pd.DataFrame):
        """Add sequence-based features"""
        # Activity trend (slope of activity over time)
        df['activity_trend'] = df['activity_level_mean'].rolling(window=3, min_periods=1).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0
        )
        
        # Volatility measures - handle missing columns gracefully
        if 'activity_level_std' in df.columns and 'activity_level_mean' in df.columns:
            df['activity_volatility_ratio'] = df['activity_level_std'] / (df['activity_level_mean'] + 1e-6)
        else:
            df['activity_volatility_ratio'] = 0.0
            
        if 'file_accesses_std' in df.columns and 'file_accesses_mean' in df.columns:
            df['file_volatility_ratio'] = df['file_accesses_std'] / (df['file_accesses_mean'] + 1e-6)
        else:
            df['file_volatility_ratio'] = 0.0
            
        if 'network_connections_std' in df.columns and 'network_connections_mean' in df.columns:
            df['network_volatility_ratio'] = df['network_connections_std'] / (df['network_connections_mean'] + 1e-6)
        else:
            df['network_volatility_ratio'] = 0.0
        
        # Pattern consistency - handle missing columns gracefully
        if 'activity_level_std' in df.columns:
            df['activity_consistency'] = 1 / (1 + df['activity_level_std'])
        else:
            df['activity_consistency'] = 1.0
            
        if 'file_accesses_std' in df.columns:
            df['file_consistency'] = 1 / (1 + df['file_accesses_std'])
        else:
            df['file_consistency'] = 1.0
            
        if 'network_connections_std' in df.columns:
            df['network_consistency'] = 1 / (1 + df['network_connections_std'])
        else:
            df['network_consistency'] = 1.0
        
        # Sudden changes
        df['activity_change'] = df['activity_level_mean'].diff().abs()
        df['file_change'] = df['file_accesses_sum'].diff().abs()
        df['network_change'] = df['network_connections_sum'].diff().abs()
        
        return df
    
    def _add_contextual_features(self, df: pd.DataFrame):
        """Add contextual features"""
        # Time-based features
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17) & (df['day_of_week'] < 5)).astype(int)
        
        # Work pattern features
        df['work_hours_activity'] = df['activity_level_mean'] * df['is_work_hours_mean']
        df['weekend_activity'] = df['activity_level_mean'] * df['is_weekend_mean']
        
        # Intensity ratios
        df['data_intensity_ratio'] = df['data_transfer_mb_sum'] / (df['activity_level_mean'] + 1e-6)
        df['network_intensity_ratio'] = df['network_connections_sum'] / (df['activity_level_mean'] + 1e-6)
        df['file_intensity_ratio'] = df['file_accesses_sum'] / (df['activity_level_mean'] + 1e-6)
        
        return df
    
    def _add_user_specific_features(self, df: pd.DataFrame):
        """Add user-specific features based on individual baselines"""
        # Initialize user-specific features
        df['user_deviation'] = 0.0
        df['user_anomaly_score'] = 0.0
        
        for user_id in df['user_id_first'].unique():
            if user_id in self.user_baselines:
                baseline = self.user_baselines[user_id]
                user_mask = df['user_id_first'] == user_id
                
                # Calculate deviation from user baseline
                for feature in ['activity_level_mean', 'file_accesses_sum', 'network_connections_sum']:
                    if f'{feature}_mean' in baseline:
                        baseline_mean = baseline[f'{feature}_mean']
                        baseline_std = baseline.get(f'{feature}_std', 1.0)
                        
                        if baseline_std > 0:
                            deviation = (df.loc[user_mask, feature] - baseline_mean) / baseline_std
                            df.loc[user_mask, f'{feature}_deviation'] = deviation
                
                # Calculate overall user anomaly score
                deviation_features = [col for col in df.columns if col.endswith('_deviation')]
                if deviation_features:
                    df.loc[user_mask, 'user_anomaly_score'] = df.loc[user_mask, deviation_features].abs().mean(axis=1)
        
        return df
    
    def calculate_dynamic_threshold(self, user_id: str, current_features: dict, window_size: int = 10):
        """Calculate dynamic threshold based on user behavior and recent activity"""
        if user_id not in self.user_baselines:
            return 0.85  # Default threshold
        
        baseline = self.user_baselines[user_id]
        
        # Calculate baseline threshold - make much less conservative
        baseline_threshold = 0.3
        
        # Adjust based on user's historical volatility
        activity_volatility = baseline.get('activity_level_std', 1.0)
        volatility_factor = min(2.0, max(0.5, activity_volatility / 0.5))
        
        # Adjust based on current activity level vs baseline
        current_activity = current_features.get('activity_level_mean', 0)
        baseline_activity = baseline.get('activity_level_mean', 0)
        
        if baseline_activity > 0:
            activity_factor = current_activity / baseline_activity
            activity_factor = min(2.0, max(0.5, activity_factor))
        else:
            activity_factor = 1.0
        
        # Calculate dynamic threshold
        dynamic_threshold = baseline_threshold * volatility_factor * activity_factor
        
        # Ensure threshold is within reasonable bounds - make much less conservative
        dynamic_threshold = max(0.2, min(0.7, dynamic_threshold))
        
        return dynamic_threshold
    
    def multi_modal_detection(self, features: dict, user_id: str):
        """Multi-modal detection combining multiple approaches"""
        # 1. Activity-based detection
        activity_score = self._calculate_activity_score(features)
        
        # 2. Pattern-based detection
        pattern_score = self._calculate_pattern_score(features)
        
        # 3. User-specific detection
        user_score = self._calculate_user_specific_score(features, user_id)
        
        # 4. Contextual detection
        context_score = self._calculate_contextual_score(features)
        
        # Combine scores with weights
        weights = [0.3, 0.25, 0.25, 0.2]  # Adjustable weights
        combined_score = (
            weights[0] * activity_score +
            weights[1] * pattern_score +
            weights[2] * user_score +
            weights[3] * context_score
        )
        
        return combined_score, {
            'activity_score': activity_score,
            'pattern_score': pattern_score,
            'user_score': user_score,
            'context_score': context_score
        }
    
    def _calculate_activity_score(self, features: dict):
        """Calculate activity-based anomaly score"""
        # High activity indicators
        activity_indicators = [
            features.get('activity_level_mean', 0),
            features.get('file_accesses_sum', 0) / 100,  # Normalize
            features.get('network_connections_sum', 0) / 50,  # Normalize
            features.get('data_transfer_mb_sum', 0) / 1000,  # Normalize
            features.get('privilege_escalation_attempts_sum', 0)
        ]
        
        # Calculate weighted activity score
        weights = [0.3, 0.2, 0.2, 0.2, 0.1]
        activity_score = sum(w * indicator for w, indicator in zip(weights, activity_indicators))
        
        # Normalize to [0, 1]
        return min(1.0, activity_score / 10)
    
    def _calculate_pattern_score(self, features: dict):
        """Calculate pattern-based anomaly score"""
        # Pattern consistency and volatility
        volatility_indicators = [
            features.get('activity_volatility_ratio', 0),
            features.get('file_volatility_ratio', 0),
            features.get('network_volatility_ratio', 0)
        ]
        
        # Sudden changes
        change_indicators = [
            features.get('activity_change', 0),
            features.get('file_change', 0),
            features.get('network_change', 0)
        ]
        
        # Calculate pattern score
        volatility_score = np.mean(volatility_indicators)
        change_score = np.mean(change_indicators) / 100  # Normalize
        
        pattern_score = (volatility_score + change_score) / 2
        return min(1.0, pattern_score)
    
    def _calculate_user_specific_score(self, features: dict, user_id: str):
        """Calculate user-specific anomaly score"""
        if user_id not in self.user_baselines:
            return 0.5  # Neutral score
        
        # Use user deviation features
        deviation_score = features.get('user_anomaly_score', 0)
        
        # Normalize to [0, 1]
        return min(1.0, deviation_score / 2)
    
    def _calculate_contextual_score(self, features: dict):
        """Calculate contextual anomaly score"""
        # Time-based anomalies
        is_business_hours = features.get('is_business_hours', 0)
        is_weekend = features.get('is_weekend_mean', 0)
        
        # Activity during unusual times
        weekend_activity = features.get('weekend_activity', 0)
        non_business_activity = features.get('activity_level_mean', 0) * (1 - is_business_hours)
        
        # Intensity anomalies
        data_intensity = features.get('data_intensity_ratio', 0)
        network_intensity = features.get('network_intensity_ratio', 0)
        
        # Calculate contextual score
        time_anomaly = (weekend_activity + non_business_activity) / 2
        intensity_anomaly = (data_intensity + network_intensity) / 2
        
        contextual_score = (time_anomaly + intensity_anomaly) / 2
        return min(1.0, contextual_score)
    
    def detect_anomalies_improved(self, df: pd.DataFrame, interval: int = 720):
        """Improved anomaly detection with dynamic thresholding and multi-modal approach"""
        print(f"🔍 Running improved anomaly detection for {interval}min intervals...")
        
        # Extract enhanced features
        df_features = self.extract_enhanced_features(df, interval)
        
        # Initialize results
        anomalies = []
        anomaly_scores = []
        detection_details = []
        
        # Process each time window
        for idx, row in df_features.iterrows():
            user_id = row.get('user_id_first', 'unknown')
            features = row.to_dict()
            
            # Calculate multi-modal detection score
            combined_score, component_scores = self.multi_modal_detection(features, user_id)
            
            # Calculate dynamic threshold
            dynamic_threshold = self.calculate_dynamic_threshold(user_id, features)
            
            # Determine if anomaly
            is_anomaly = combined_score > dynamic_threshold
            
            # Debug: Print some scores for analysis
            if len(anomaly_scores) < 5:  # Print first 5 for debugging
                print(f"   Debug - Time: {idx}, Score: {combined_score:.3f}, Threshold: {dynamic_threshold:.3f}, Anomaly: {is_anomaly}")
            
            # Store results
            anomaly_scores.append(combined_score)
            detection_details.append({
                'timestamp': idx,
                'user_id': user_id,
                'combined_score': combined_score,
                'threshold': dynamic_threshold,
                'is_anomaly': is_anomaly,
                'component_scores': component_scores
            })
            
            # Add to anomalies list if detected
            if is_anomaly:
                anomalies.append({
                    'start_time': idx.isoformat(),
                    'end_time': (idx + timedelta(minutes=interval)).isoformat(),
                    'confidence': combined_score,
                    'method': f'improved_multi_modal_{interval}min',
                    'user_id': user_id,
                    'component_scores': component_scores
                })
        
        # Calculate performance metrics
        # Anomaly (malicious) = 1 (positive), Normal = 0 (negative)
        true_labels = df_features['is_malicious_max'].values
        predictions = [detail['is_anomaly'] for detail in detection_details]
        
        # Calculate metrics
        tn, fp, fn, tp = confusion_matrix(true_labels, predictions).ravel()
        
        metrics = {
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn,
            'true_negatives': tn,
            'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'f1_score': 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0,
            'accuracy': (tp + tn) / (tp + tn + fp + fn),
            'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
            'false_negative_rate': fn / (fn + tp) if (fn + tp) > 0 else 0
        }
        
        results = {
            'anomalies': anomalies,
            'anomaly_scores': anomaly_scores,
            'detection_details': detection_details,
            'metrics': metrics,
            'method': 'improved_multi_modal',
            'time_interval': interval,
            'sequences_analyzed': len(df_features)
        }
        
        print(f"   ✅ Detected {len(anomalies)} anomalies")
        print(f"   📊 Precision: {metrics['precision']:.3f}")
        print(f"   📊 Recall: {metrics['recall']:.3f}")
        print(f"   📊 F1-Score: {metrics['f1_score']:.3f}")
        
        return results
    
    def save_results(self, results: dict, interval: int):
        """Save improved detection results"""
        filename = f"improved_agentic_rag_results_{interval}min.json"
        
        # Convert numpy types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(v) for v in obj]
            elif isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif hasattr(obj, 'to_pydatetime'):
                return obj.to_pydatetime().isoformat()
            elif hasattr(obj, 'isoformat'):
                return obj.isoformat()
            else:
                return obj
        
        # Convert results
        json_results = convert_numpy_types(results)
        
        with open(filename, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"   💾 Results saved to: {filename}")
    
    def create_comparison_visualization(self, original_results: dict, improved_results: dict, interval: int):
        """Create comparison visualization between original and improved detection"""
        print("📊 Creating comparison visualization...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 15))
        
        # 1. Performance metrics comparison
        metrics = ['precision', 'recall', 'f1_score', 'accuracy']
        original_metrics = [original_results['metrics'][m] for m in metrics]
        improved_metrics = [improved_results['metrics'][m] for m in metrics]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        ax1.bar(x - width/2, original_metrics, width, label='Original', color='red', alpha=0.7)
        ax1.bar(x + width/2, improved_metrics, width, label='Improved', color='green', alpha=0.7)
        
        ax1.set_xlabel('Metrics')
        ax1.set_ylabel('Score')
        ax1.set_title(f'Performance Comparison ({interval}min intervals)')
        ax1.set_xticks(x)
        ax1.set_xticklabels([m.replace('_', ' ').title() for m in metrics])
        ax1.legend()
        ax1.set_ylim(0, 1)
        
        # 2. Error rates comparison
        error_metrics = ['false_positive_rate', 'false_negative_rate']
        original_errors = [original_results['metrics'][m] for m in error_metrics]
        improved_errors = [improved_results['metrics'][m] for m in error_metrics]
        
        ax2.bar(x - width/2, original_errors, width, label='Original', color='red', alpha=0.7)
        ax2.bar(x + width/2, improved_errors, width, label='Improved', color='green', alpha=0.7)
        
        ax2.set_xlabel('Error Types')
        ax2.set_ylabel('Rate')
        ax2.set_title('Error Rate Comparison')
        ax2.set_xticks(x)
        ax2.set_xticklabels(['False Positive Rate', 'False Negative Rate'])
        ax2.legend()
        ax2.set_ylim(0, 1)
        
        # 3. Anomaly score distribution
        original_scores = original_results.get('anomaly_scores', [])
        improved_scores = improved_results.get('anomaly_scores', [])
        
        ax3.hist(original_scores, bins=30, alpha=0.7, label='Original', color='red')
        ax3.hist(improved_scores, bins=30, alpha=0.7, label='Improved', color='green')
        ax3.set_xlabel('Anomaly Score')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Anomaly Score Distribution')
        ax3.legend()
        
        # 4. Detection count comparison
        detection_counts = [
            len(original_results['anomalies']),
            len(improved_results['anomalies'])
        ]
        
        ax4.bar(['Original', 'Improved'], detection_counts, color=['red', 'green'], alpha=0.7)
        ax4.set_ylabel('Number of Detections')
        ax4.set_title('Detection Count Comparison')
        
        plt.tight_layout()
        plt.savefig(f'improved_detection_comparison_{interval}min.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"   📊 Comparison visualization saved to: improved_detection_comparison_{interval}min.png")
    
    def run_improved_detection(self, intervals: list = [60, 120, 720]):
        """Run improved detection on multiple intervals"""
        print("🚀 Running Improved AgenticRAG Detection")
        print("=" * 60)
        
        # Load dataset
        df = self.load_dataset()
        
        # Create user baselines
        self.create_user_baselines(df)
        
        all_results = {}
        
        for interval in intervals:
            print(f"\n⏰ Running improved detection for {interval}min intervals...")
            
            # Run improved detection
            improved_results = self.detect_anomalies_improved(df, interval)
            
            # Save results
            self.save_results(improved_results, interval)
            
            # Store results
            all_results[interval] = improved_results
        
        # Generate summary
        self.generate_improvement_summary(all_results)
        
        return all_results
    
    def generate_improvement_summary(self, all_results: dict):
        """Generate summary of improvements"""
        print("\n📊 Generating Improvement Summary...")
        
        summary_data = []
        
        for interval, results in all_results.items():
            summary_data.append({
                'Interval': f'{interval}min',
                'Precision': results['metrics']['precision'],
                'Recall': results['metrics']['recall'],
                'F1_Score': results['metrics']['f1_score'],
                'Accuracy': results['metrics']['accuracy'],
                'False_Positives': results['metrics']['false_positives'],
                'False_Negatives': results['metrics']['false_negatives'],
                'True_Positives': results['metrics']['true_positives'],
                'True_Negatives': results['metrics']['true_negatives'],
                'Detections': len(results['anomalies'])
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv('improved_detection_summary.csv', index=False)
        
        print("\n📈 Improved Detection Summary:")
        print("=" * 50)
        
        for _, row in summary_df.iterrows():
            print(f"\n⏰ {row['Interval']}")
            print(f"   Precision: {row['Precision']:.3f} | Recall: {row['Recall']:.3f} | F1: {row['F1_Score']:.3f}")
            print(f"   False Positives: {row['False_Positives']} | False Negatives: {row['False_Negatives']}")
            print(f"   Detections: {row['Detections']}")
        
        print(f"\n✅ Improvement summary saved to: improved_detection_summary.csv")

def main():
    """Main execution function"""
    print("🔧 Improved AgenticRAG Detection System")
    print("📊 Enhanced with Dynamic Thresholding and Multi-Modal Detection")
    print("=" * 70)
    
    # Initialize improved detection system
    improved_detection = ImprovedAgenticRAGDetection()
    
    # Run improved detection
    results = improved_detection.run_improved_detection(
        intervals=[60, 120, 720]
    )
    
    print("\n🎉 Improved detection completed successfully!")
    print("📁 Check the generated files for detailed results and comparisons.")

if __name__ == "__main__":
    main() 