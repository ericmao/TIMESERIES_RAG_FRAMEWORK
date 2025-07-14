"""
Classification Agent for the Time Series RAG Framework
"""
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import warnings

from .base_agent import BaseAgent, AgentResponse
from ..config.config import get_config
from ..utils.logger import get_logger

class ClassificationAgent(BaseAgent):
    """
    Specialized agent for classifying time series patterns and behaviors.
    
    Supports multiple classification approaches:
    - Pattern-based classification
    - Statistical feature classification
    - Trend classification
    - Seasonality classification
    """
    
    def __init__(self, agent_id: str, model_name: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        super().__init__(
            agent_id=agent_id,
            agent_type="classification",
            model_name=model_name or get_config().model.classification_agent_model,
            config=config
        )
        self.logger = get_logger(f"classification_agent_{agent_id}")
        
        # Classification methods
        self.classification_methods = {
            "pattern": self._classify_patterns,
            "trend": self._classify_trends,
            "seasonality": self._classify_seasonality,
            "behavior": self._classify_behavior,
            "comprehensive": self._classify_comprehensive
        }
        
        # Pre-trained models (if available)
        self.models = {}
    
    async def _process_request_internal(
        self,
        request: Dict[str, Any],
        context: Optional[Dict[str, Any]],
        relevant_prompts: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Process a classification request.
        
        Args:
            request: Dict with 'data' (time series) and classification parameters
            context: Optional context
            relevant_prompts: List of retrieved prompts
            
        Returns:
            Dict with classification results
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
        
        # Get classification parameters
        method = request.get('method', 'comprehensive')
        classification_type = request.get('classification_type', 'pattern')
        
        try:
            # Perform classification
            if method in self.classification_methods:
                classification_result = await self.classification_methods[method](
                    df, classification_type, context
                )
            else:
                return {"error": f"Unsupported classification method: {method}"}
            
            # Calculate confidence
            confidence = self._calculate_confidence(classification_result, method)
            
            return {
                "classification": classification_result,
                "method": method,
                "classification_type": classification_type,
                "confidence": confidence,
                "used_prompts": relevant_prompts[:3]
            }
            
        except Exception as e:
            return {"error": f"Classification failed: {str(e)}"}
    
    async def _classify_patterns(self, df: pd.DataFrame, classification_type: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Classify time series patterns"""
        try:
            # Extract features for pattern classification
            features = self._extract_pattern_features(df)
            
            # Define pattern classes based on features
            pattern_class = self._determine_pattern_class(features)
            
            return {
                "pattern_type": pattern_class,
                "features": features,
                "confidence_score": self._calculate_pattern_confidence(features)
            }
            
        except Exception as e:
            self.logger.error(f"Pattern classification failed: {str(e)}")
            return {"error": f"Pattern classification failed: {str(e)}"}
    
    async def _classify_trends(self, df: pd.DataFrame, classification_type: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Classify time series trends"""
        try:
            # Calculate trend features
            trend_features = self._extract_trend_features(df)
            
            # Determine trend class
            trend_class = self._determine_trend_class(trend_features)
            
            return {
                "trend_type": trend_class,
                "trend_strength": trend_features.get("trend_strength", 0.0),
                "trend_direction": trend_features.get("trend_direction", "unknown"),
                "features": trend_features
            }
            
        except Exception as e:
            self.logger.error(f"Trend classification failed: {str(e)}")
            return {"error": f"Trend classification failed: {str(e)}"}
    
    async def _classify_seasonality(self, df: pd.DataFrame, classification_type: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Classify time series seasonality"""
        try:
            # Extract seasonality features
            seasonality_features = self._extract_seasonality_features(df)
            
            # Determine seasonality class
            seasonality_class = self._determine_seasonality_class(seasonality_features)
            
            return {
                "seasonality_type": seasonality_class,
                "seasonal_period": seasonality_features.get("seasonal_period", None),
                "seasonal_strength": seasonality_features.get("seasonal_strength", 0.0),
                "features": seasonality_features
            }
            
        except Exception as e:
            self.logger.error(f"Seasonality classification failed: {str(e)}")
            return {"error": f"Seasonality classification failed: {str(e)}"}
    
    async def _classify_behavior(self, df: pd.DataFrame, classification_type: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Classify time series behavior"""
        try:
            # Extract behavior features
            behavior_features = self._extract_behavior_features(df)
            
            # Determine behavior class
            behavior_class = self._determine_behavior_class(behavior_features)
            
            return {
                "behavior_type": behavior_class,
                "volatility": behavior_features.get("volatility", 0.0),
                "stability": behavior_features.get("stability", 0.0),
                "features": behavior_features
            }
            
        except Exception as e:
            self.logger.error(f"Behavior classification failed: {str(e)}")
            return {"error": f"Behavior classification failed: {str(e)}"}
    
    async def _classify_comprehensive(self, df: pd.DataFrame, classification_type: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform comprehensive classification"""
        try:
            # Run all classification methods
            pattern_result = await self._classify_patterns(df, classification_type, context)
            trend_result = await self._classify_trends(df, classification_type, context)
            seasonality_result = await self._classify_seasonality(df, classification_type, context)
            behavior_result = await self._classify_behavior(df, classification_type, context)
            
            # Combine results
            comprehensive_result = {
                "pattern": pattern_result,
                "trend": trend_result,
                "seasonality": seasonality_result,
                "behavior": behavior_result,
                "overall_classification": self._combine_classifications(
                    pattern_result, trend_result, seasonality_result, behavior_result
                )
            }
            
            return comprehensive_result
            
        except Exception as e:
            self.logger.error(f"Comprehensive classification failed: {str(e)}")
            return {"error": f"Comprehensive classification failed: {str(e)}"}
    
    def _extract_pattern_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Extract features for pattern classification"""
        try:
            values = df['y'].values
            
            # Basic statistical features
            features = {
                "mean": np.mean(values),
                "std": np.std(values),
                "skewness": self._calculate_skewness(values),
                "kurtosis": self._calculate_kurtosis(values),
                "range": np.max(values) - np.min(values),
                "iqr": np.percentile(values, 75) - np.percentile(values, 25)
            }
            
            # Trend features
            features.update(self._extract_trend_features(df))
            
            # Seasonality features
            features.update(self._extract_seasonality_features(df))
            
            return features
            
        except Exception as e:
            self.logger.error(f"Feature extraction failed: {str(e)}")
            return {}
    
    def _extract_trend_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Extract trend-related features"""
        try:
            values = df['y'].values
            x = np.arange(len(values))
            
            # Linear trend
            slope, intercept = np.polyfit(x, values, 1)
            trend_strength = np.corrcoef(x, values)[0, 1]
            
            # Moving average trend
            window = min(10, len(values) // 4)
            ma = pd.Series(values).rolling(window=window).mean()
            ma_trend = np.polyfit(range(len(ma.dropna())), ma.dropna(), 1)[0]
            
            return {
                "trend_slope": slope,
                "trend_strength": abs(trend_strength),
                "trend_direction": "increasing" if slope > 0 else "decreasing",
                "ma_trend": ma_trend
            }
            
        except Exception as e:
            self.logger.error(f"Trend feature extraction failed: {str(e)}")
            return {}
    
    def _extract_seasonality_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Extract seasonality-related features"""
        try:
            values = df['y'].values
            
            # Autocorrelation for seasonality detection
            autocorr = np.correlate(values, values, mode='full')
            autocorr = autocorr[len(values)-1:]
            
            # Find peaks in autocorrelation (potential seasonal periods)
            peaks = self._find_peaks(autocorr)
            seasonal_period = peaks[0] if len(peaks) > 0 else None
            
            # Seasonal strength (variance of seasonal component)
            if seasonal_period and seasonal_period > 1:
                seasonal_values = values[::seasonal_period]
                seasonal_strength = np.var(seasonal_values) / np.var(values)
            else:
                seasonal_strength = 0.0
            
            return {
                "seasonal_period": seasonal_period,
                "seasonal_strength": seasonal_strength,
                "autocorr_max": np.max(autocorr[1:]) if len(autocorr) > 1 else 0.0
            }
            
        except Exception as e:
            self.logger.error(f"Seasonality feature extraction failed: {str(e)}")
            return {}
    
    def _extract_behavior_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Extract behavior-related features"""
        try:
            values = df['y'].values
            
            # Volatility (rolling standard deviation)
            window = min(10, len(values) // 4)
            rolling_std = pd.Series(values).rolling(window=window).std()
            volatility = np.mean(rolling_std.dropna())
            
            # Stability (inverse of volatility)
            stability = 1.0 / (1.0 + volatility)
            
            # Change points (sudden changes)
            diff = np.diff(values)
            change_points = np.sum(np.abs(diff) > np.std(diff))
            change_rate = change_points / len(diff)
            
            return {
                "volatility": volatility,
                "stability": stability,
                "change_rate": change_rate,
                "smoothness": 1.0 - change_rate
            }
            
        except Exception as e:
            self.logger.error(f"Behavior feature extraction failed: {str(e)}")
            return {}
    
    def _determine_pattern_class(self, features: Dict[str, Any]) -> str:
        """Determine pattern class based on features"""
        try:
            # Simple rule-based classification
            if features.get("trend_strength", 0) > 0.7:
                if features.get("trend_direction") == "increasing":
                    return "strong_upward_trend"
                else:
                    return "strong_downward_trend"
            elif features.get("seasonal_strength", 0) > 0.5:
                return "seasonal_pattern"
            elif features.get("volatility", 0) > np.mean(list(features.values())):
                return "volatile_pattern"
            else:
                return "stable_pattern"
                
        except Exception as e:
            self.logger.error(f"Pattern classification failed: {str(e)}")
            return "unknown_pattern"
    
    def _determine_trend_class(self, features: Dict[str, Any]) -> str:
        """Determine trend class"""
        try:
            strength = features.get("trend_strength", 0)
            direction = features.get("trend_direction", "unknown")
            
            if strength > 0.8:
                return f"very_strong_{direction}_trend"
            elif strength > 0.6:
                return f"strong_{direction}_trend"
            elif strength > 0.4:
                return f"moderate_{direction}_trend"
            elif strength > 0.2:
                return f"weak_{direction}_trend"
            else:
                return "no_trend"
                
        except Exception as e:
            self.logger.error(f"Trend classification failed: {str(e)}")
            return "unknown_trend"
    
    def _determine_seasonality_class(self, features: Dict[str, Any]) -> str:
        """Determine seasonality class"""
        try:
            strength = features.get("seasonal_strength", 0)
            period = features.get("seasonal_period")
            
            if strength > 0.7:
                return "strong_seasonal"
            elif strength > 0.4:
                return "moderate_seasonal"
            elif strength > 0.2:
                return "weak_seasonal"
            else:
                return "no_seasonality"
                
        except Exception as e:
            self.logger.error(f"Seasonality classification failed: {str(e)}")
            return "unknown_seasonality"
    
    def _determine_behavior_class(self, features: Dict[str, Any]) -> str:
        """Determine behavior class"""
        try:
            volatility = features.get("volatility", 0)
            stability = features.get("stability", 0)
            change_rate = features.get("change_rate", 0)
            
            if volatility > 0.5:
                return "highly_volatile"
            elif change_rate > 0.3:
                return "frequently_changing"
            elif stability > 0.8:
                return "very_stable"
            elif stability > 0.6:
                return "stable"
            else:
                return "moderate_volatility"
                
        except Exception as e:
            self.logger.error(f"Behavior classification failed: {str(e)}")
            return "unknown_behavior"
    
    def _combine_classifications(self, pattern_result: Dict, trend_result: Dict, 
                               seasonality_result: Dict, behavior_result: Dict) -> str:
        """Combine multiple classifications into overall classification"""
        try:
            # Simple combination logic
            classifications = []
            
            if "pattern_type" in pattern_result:
                classifications.append(pattern_result["pattern_type"])
            if "trend_type" in trend_result:
                classifications.append(trend_result["trend_type"])
            if "seasonality_type" in seasonality_result:
                classifications.append(seasonality_result["seasonality_type"])
            if "behavior_type" in behavior_result:
                classifications.append(behavior_result["behavior_type"])
            
            # Return the most specific classification
            if "strong_upward_trend" in classifications or "strong_downward_trend" in classifications:
                return "trend_dominant"
            elif "strong_seasonal" in classifications:
                return "seasonal_dominant"
            elif "highly_volatile" in classifications:
                return "volatile_dominant"
            elif "very_stable" in classifications:
                return "stable_dominant"
            else:
                return "mixed_pattern"
                
        except Exception as e:
            self.logger.error(f"Classification combination failed: {str(e)}")
            return "unknown"
    
    def _calculate_skewness(self, values: np.ndarray) -> float:
        """Calculate skewness of the data"""
        try:
            mean = np.mean(values)
            std = np.std(values)
            if std == 0:
                return 0.0
            return np.mean(((values - mean) / std) ** 3)
        except:
            return 0.0
    
    def _calculate_kurtosis(self, values: np.ndarray) -> float:
        """Calculate kurtosis of the data"""
        try:
            mean = np.mean(values)
            std = np.std(values)
            if std == 0:
                return 0.0
            return np.mean(((values - mean) / std) ** 4) - 3
        except:
            return 0.0
    
    def _find_peaks(self, values: np.ndarray) -> List[int]:
        """Find peaks in the array"""
        try:
            peaks = []
            for i in range(1, len(values) - 1):
                if values[i] > values[i-1] and values[i] > values[i+1]:
                    peaks.append(i)
            return peaks
        except:
            return []
    
    def _calculate_pattern_confidence(self, features: Dict[str, Any]) -> float:
        """Calculate confidence for pattern classification"""
        try:
            # Base confidence on feature quality
            confidence = 0.5
            
            # Adjust based on feature values
            if features.get("trend_strength", 0) > 0.5:
                confidence += 0.2
            if features.get("seasonal_strength", 0) > 0.3:
                confidence += 0.2
            if features.get("volatility", 0) < 0.5:
                confidence += 0.1
            
            return min(confidence, 1.0)
        except:
            return 0.5
    
    def _calculate_confidence(self, classification_result: Dict[str, Any], method: str) -> float:
        """Calculate confidence score for classification results"""
        if "error" in classification_result:
            return 0.0
        
        if method == "comprehensive":
            # Average confidence from all methods
            confidences = []
            for key in ["pattern", "trend", "seasonality", "behavior"]:
                if key in classification_result and "confidence_score" in classification_result[key]:
                    confidences.append(classification_result[key]["confidence_score"])
            
            return np.mean(confidences) if confidences else 0.5
        else:
            # Base confidence on method reliability
            return classification_result.get("confidence_score", 0.7) 