"""
Anomaly Detection Agent for the Time Series RAG Framework
"""
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import warnings

from .base_agent import BaseAgent, AgentResponse
from ..config.config import get_config
from ..utils.logger import get_logger

class AnomalyDetectionAgent(BaseAgent):
    """
    Specialized agent for detecting anomalies in time series data.
    
    Supports multiple anomaly detection methods:
    - Statistical methods (Z-score, IQR)
    - Machine learning (Isolation Forest)
    - Rolling statistics
    - Change point detection
    """
    
    def __init__(self, agent_id: str, model_name: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        super().__init__(
            agent_id=agent_id,
            agent_type="anomaly_detection",
            model_name=model_name or get_config().model.anomaly_agent_model,
            config=config
        )
        self.logger = get_logger(f"anomaly_detection_agent_{agent_id}")
        
        # Anomaly detection methods
        self.detection_methods = {
            "zscore": self._detect_anomalies_zscore,
            "iqr": self._detect_anomalies_iqr,
            "isolation_forest": self._detect_anomalies_isolation_forest,
            "rolling_stats": self._detect_anomalies_rolling_stats,
            "combined": self._detect_anomalies_combined
        }
    
    async def _process_request_internal(
        self,
        request: Dict[str, Any],
        context: Optional[Dict[str, Any]],
        relevant_prompts: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Process an anomaly detection request.
        
        Args:
            request: Dict with 'data' (time series) and detection parameters
            context: Optional context
            relevant_prompts: List of retrieved prompts
            
        Returns:
            Dict with anomaly detection results
        """
        # Extract data
        data = request.get('data')
        if data is None:
            return {"error": "No data provided"}
        
        # Convert to DataFrame
        try:
            df = pd.DataFrame(data)
            if 'ds' not in df.columns or 'y' not in df.columns:
                return {"error": "Data must contain 'ds' (date) and 'y' (value) columns"}
        except Exception as e:
            return {"error": f"Invalid data format: {str(e)}"}
        
        # Get detection parameters
        method = request.get('method', 'combined')
        threshold = request.get('threshold', get_config().timeseries.anomaly_threshold)
        window_size = request.get('window_size', get_config().timeseries.window_size)
        
        try:
            # Perform anomaly detection
            if method in self.detection_methods:
                anomalies = await self.detection_methods[method](df, threshold, window_size)
            else:
                return {"error": f"Unsupported detection method: {method}"}
            
            # Calculate confidence based on method agreement
            confidence = self._calculate_confidence(anomalies, method)
            
            return {
                "anomalies": anomalies,
                "method": method,
                "threshold": threshold,
                "confidence": confidence,
                "total_anomalies": len(anomalies),
                "anomaly_ratio": len(anomalies) / len(df),
                "used_prompts": relevant_prompts[:3]
            }
            
        except Exception as e:
            return {"error": f"Anomaly detection failed: {str(e)}"}
    
    async def _detect_anomalies_zscore(self, df: pd.DataFrame, threshold: float, window_size: int) -> List[Dict[str, Any]]:
        """Detect anomalies using Z-score method"""
        try:
            # Calculate rolling mean and std
            rolling_mean = df['y'].rolling(window=window_size, center=True).mean()
            rolling_std = df['y'].rolling(window=window_size, center=True).std()
            
            # Calculate Z-scores
            z_scores = np.abs((df['y'] - rolling_mean) / rolling_std)
            
            # Find anomalies
            anomaly_indices = np.where(z_scores > threshold)[0]
            
            anomalies = []
            for idx in anomaly_indices:
                if not pd.isna(z_scores[idx]):
                    anomalies.append({
                        "index": int(idx),
                        "timestamp": df.iloc[idx]['ds'],
                        "value": float(df.iloc[idx]['y']),
                        "z_score": float(z_scores[idx]),
                        "method": "zscore"
                    })
            
            return anomalies
            
        except Exception as e:
            self.logger.error(f"Z-score anomaly detection failed: {str(e)}")
            return []
    
    async def _detect_anomalies_iqr(self, df: pd.DataFrame, threshold: float, window_size: int) -> List[Dict[str, Any]]:
        """Detect anomalies using IQR method"""
        try:
            # Calculate rolling Q1, Q3, and IQR
            rolling_q1 = df['y'].rolling(window=window_size, center=True).quantile(0.25)
            rolling_q3 = df['y'].rolling(window=window_size, center=True).quantile(0.75)
            rolling_iqr = rolling_q3 - rolling_q1
            
            # Define bounds
            lower_bound = rolling_q1 - threshold * rolling_iqr
            upper_bound = rolling_q3 + threshold * rolling_iqr
            
            # Find anomalies
            anomalies = []
            for idx in range(len(df)):
                if not pd.isna(lower_bound.iloc[idx]) and not pd.isna(upper_bound.iloc[idx]):
                    value = df.iloc[idx]['y']
                    if value < lower_bound.iloc[idx] or value > upper_bound.iloc[idx]:
                        anomalies.append({
                            "index": int(idx),
                            "timestamp": df.iloc[idx]['ds'],
                            "value": float(value),
                            "lower_bound": float(lower_bound.iloc[idx]),
                            "upper_bound": float(upper_bound.iloc[idx]),
                            "method": "iqr"
                        })
            
            return anomalies
            
        except Exception as e:
            self.logger.error(f"IQR anomaly detection failed: {str(e)}")
            return []
    
    async def _detect_anomalies_isolation_forest(self, df: pd.DataFrame, threshold: float, window_size: int) -> List[Dict[str, Any]]:
        """Detect anomalies using Isolation Forest"""
        try:
            # Prepare features
            features = df['y'].values.reshape(-1, 1)
            
            # Handle NaN values
            features = np.nan_to_num(features, nan=np.nanmean(features))
            
            # Fit Isolation Forest
            # Convert threshold to appropriate contamination value (0.01 to 0.5)
            contamination = min(max(threshold * 0.1, 0.01), 0.5)
            iso_forest = IsolationForest(
                contamination=contamination,
                random_state=42,
                n_estimators=100
            )
            
            # Predict anomalies (-1 for anomalies, 1 for normal)
            predictions = iso_forest.fit_predict(features)
            
            # Find anomalies
            anomaly_indices = np.where(predictions == -1)[0]
            
            anomalies = []
            for idx in anomaly_indices:
                anomalies.append({
                    "index": int(idx),
                    "timestamp": df.iloc[idx]['ds'],
                    "value": float(df.iloc[idx]['y']),
                    "method": "isolation_forest"
                })
            
            return anomalies
            
        except Exception as e:
            self.logger.error(f"Isolation Forest anomaly detection failed: {str(e)}")
            return []
    
    async def _detect_anomalies_rolling_stats(self, df: pd.DataFrame, threshold: float, window_size: int) -> List[Dict[str, Any]]:
        """Detect anomalies using rolling statistics"""
        try:
            # Calculate rolling statistics
            rolling_mean = df['y'].rolling(window=window_size, center=True).mean()
            rolling_std = df['y'].rolling(window=window_size, center=True).std()
            
            # Calculate deviation from rolling mean
            deviation = np.abs(df['y'] - rolling_mean)
            
            # Find points that deviate significantly
            significant_deviation = deviation > (threshold * rolling_std)
            
            anomalies = []
            for idx in range(len(df)):
                if significant_deviation.iloc[idx] and not pd.isna(rolling_std.iloc[idx]):
                    anomalies.append({
                        "index": int(idx),
                        "timestamp": df.iloc[idx]['ds'],
                        "value": float(df.iloc[idx]['y']),
                        "rolling_mean": float(rolling_mean.iloc[idx]),
                        "deviation": float(deviation.iloc[idx]),
                        "method": "rolling_stats"
                    })
            
            return anomalies
            
        except Exception as e:
            self.logger.error(f"Rolling stats anomaly detection failed: {str(e)}")
            return []
    
    async def _detect_anomalies_combined(self, df: pd.DataFrame, threshold: float, window_size: int) -> List[Dict[str, Any]]:
        """Combine multiple anomaly detection methods"""
        try:
            # Run all methods
            zscore_anomalies = await self._detect_anomalies_zscore(df, threshold, window_size)
            iqr_anomalies = await self._detect_anomalies_iqr(df, threshold, window_size)
            iso_forest_anomalies = await self._detect_anomalies_isolation_forest(df, threshold, window_size)
            rolling_anomalies = await self._detect_anomalies_rolling_stats(df, threshold, window_size)
            
            # Combine results
            all_anomalies = {}
            
            # Count detections for each point
            for anomaly in zscore_anomalies + iqr_anomalies + iso_forest_anomalies + rolling_anomalies:
                idx = anomaly["index"]
                if idx not in all_anomalies:
                    all_anomalies[idx] = {
                        "index": idx,
                        "timestamp": anomaly["timestamp"],
                        "value": anomaly["value"],
                        "detection_methods": [],
                        "confidence": 0.0
                    }
                
                all_anomalies[idx]["detection_methods"].append(anomaly["method"])
            
            # Calculate confidence based on method agreement
            for idx, anomaly in all_anomalies.items():
                method_count = len(anomaly["detection_methods"])
                anomaly["confidence"] = method_count / 4.0  # 4 methods total
            
            # Filter by minimum confidence
            min_confidence = 0.25  # At least one method must detect
            combined_anomalies = [
                anomaly for anomaly in all_anomalies.values()
                if anomaly["confidence"] >= min_confidence
            ]
            
            return combined_anomalies
            
        except Exception as e:
            self.logger.error(f"Combined anomaly detection failed: {str(e)}")
            return []
    
    def _calculate_confidence(self, anomalies: List[Dict[str, Any]], method: str) -> float:
        """Calculate confidence score for anomaly detection results"""
        if not anomalies:
            return 0.0
        
        if method == "combined":
            # Average confidence from combined method
            confidences = [anomaly.get("confidence", 0.0) for anomaly in anomalies]
            return np.mean(confidences) if confidences else 0.0
        else:
            # Base confidence on number of anomalies and method reliability
            base_confidence = 0.8
            anomaly_ratio = len(anomalies) / 100  # Normalize by expected max anomalies
            
            # Adjust confidence based on anomaly ratio (too many or too few reduces confidence)
            if 0.01 <= anomaly_ratio <= 0.1:  # 1-10% anomalies is reasonable
                confidence = base_confidence
            else:
                confidence = base_confidence * 0.5  # Reduce confidence for extreme ratios
            
            return min(confidence, 1.0) 