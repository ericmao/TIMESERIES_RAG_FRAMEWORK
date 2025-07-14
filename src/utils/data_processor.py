"""
Data Processing Utilities for the Time Series RAG Framework
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
import warnings
from pathlib import Path
import json

from ..config.config import get_config
from .logger import get_logger

class TimeSeriesDataProcessor:
    """
    Comprehensive data processor for time series data.
    
    Features:
    - Data validation and cleaning
    - Feature engineering
    - Missing value handling
    - Outlier detection and treatment
    - Data transformation
    - Format conversion
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or get_config()
        self.logger = get_logger("data_processor")
        
        # Supported data formats
        self.supported_formats = ['.csv', '.json', '.parquet', '.xlsx', '.xls']
        
        # Processing statistics
        self.processing_stats = {}
    
    def validate_data(self, data: Union[pd.DataFrame, Dict[str, List], str]) -> Dict[str, Any]:
        """
        Validate time series data
        
        Args:
            data: Input data (DataFrame, dict, or file path)
            
        Returns:
            Validation results
        """
        try:
            # Convert to DataFrame if needed
            df = self._ensure_dataframe(data)
            
            validation_results = {
                "is_valid": True,
                "errors": [],
                "warnings": [],
                "statistics": {}
            }
            
            # Check required columns
            required_columns = ['ds', 'y']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                validation_results["is_valid"] = False
                validation_results["errors"].append(f"Missing required columns: {missing_columns}")
            
            # Check data types
            if 'ds' in df.columns:
                if not pd.api.types.is_datetime64_any_dtype(df['ds']):
                    try:
                        df['ds'] = pd.to_datetime(df['ds'])
                        validation_results["warnings"].append("Date column converted to datetime")
                    except:
                        validation_results["errors"].append("Cannot convert 'ds' column to datetime")
            
            if 'y' in df.columns:
                if not pd.api.types.is_numeric_dtype(df['y']):
                    try:
                        df['y'] = pd.to_numeric(df['y'], errors='coerce')
                        validation_results["warnings"].append("Value column converted to numeric")
                    except:
                        validation_results["errors"].append("Cannot convert 'y' column to numeric")
            
            # Check for missing values
            missing_ratio = df.isnull().sum().sum() / (len(df) * len(df.columns))
            if missing_ratio > self.config.data.max_missing_ratio:
                validation_results["warnings"].append(f"High missing value ratio: {missing_ratio:.2%}")
            
            # Check data length
            if len(df) < self.config.timeseries.min_sequence_length:
                validation_results["errors"].append(f"Data too short: {len(df)} < {self.config.timeseries.min_sequence_length}")
            
            if len(df) > self.config.timeseries.max_sequence_length:
                validation_results["warnings"].append(f"Data very long: {len(df)} > {self.config.timeseries.max_sequence_length}")
            
            # Calculate statistics
            if validation_results["is_valid"]:
                validation_results["statistics"] = {
                    "length": len(df),
                    "date_range": {
                        "start": df['ds'].min().isoformat() if 'ds' in df.columns else None,
                        "end": df['ds'].max().isoformat() if 'ds' in df.columns else None
                    },
                    "value_stats": {
                        "mean": df['y'].mean() if 'y' in df.columns else None,
                        "std": df['y'].std() if 'y' in df.columns else None,
                        "min": df['y'].min() if 'y' in df.columns else None,
                        "max": df['y'].max() if 'y' in df.columns else None
                    },
                    "missing_values": df.isnull().sum().to_dict()
                }
            
            return validation_results
            
        except Exception as e:
            self.logger.error(f"Data validation failed: {str(e)}")
            return {
                "is_valid": False,
                "errors": [f"Validation error: {str(e)}"],
                "warnings": [],
                "statistics": {}
            }
    
    def clean_data(self, data: Union[pd.DataFrame, Dict[str, List], str]) -> pd.DataFrame:
        """
        Clean time series data
        
        Args:
            data: Input data
            
        Returns:
            Cleaned DataFrame
        """
        try:
            df = self._ensure_dataframe(data)
            
            # Sort by date
            if 'ds' in df.columns:
                df = df.sort_values('ds').reset_index(drop=True)
            
            # Handle missing values
            df = self._handle_missing_values(df)
            
            # Remove duplicates
            df = df.drop_duplicates(subset=['ds'] if 'ds' in df.columns else None)
            
            # Handle outliers
            df = self._handle_outliers(df)
            
            # Ensure proper data types
            df = self._ensure_data_types(df)
            
            self.logger.info(f"Data cleaned successfully. Shape: {df.shape}")
            return df
            
        except Exception as e:
            self.logger.error(f"Data cleaning failed: {str(e)}")
            raise
    
    def engineer_features(self, data: Union[pd.DataFrame, Dict[str, List], str]) -> pd.DataFrame:
        """
        Engineer features for time series data
        
        Args:
            data: Input data
            
        Returns:
            DataFrame with engineered features
        """
        try:
            df = self._ensure_dataframe(data)
            
            # Ensure data is sorted by date
            if 'ds' in df.columns:
                df = df.sort_values('ds').reset_index(drop=True)
            
            # Time-based features
            if 'ds' in df.columns:
                df = self._add_time_features(df)
            
            # Lag features
            if self.config.timeseries.use_lag_features:
                df = self._add_lag_features(df)
            
            # Rolling statistics
            df = self._add_rolling_features(df)
            
            # Trend features
            if self.config.timeseries.use_trend_features:
                df = self._add_trend_features(df)
            
            # Seasonal features
            if self.config.timeseries.use_seasonal_features:
                df = self._add_seasonal_features(df)
            
            self.logger.info(f"Feature engineering completed. Features: {list(df.columns)}")
            return df
            
        except Exception as e:
            self.logger.error(f"Feature engineering failed: {str(e)}")
            raise
    
    def transform_data(self, data: Union[pd.DataFrame, Dict[str, List], str], 
                      transformation: str = "standardize") -> pd.DataFrame:
        """
        Transform time series data
        
        Args:
            data: Input data
            transformation: Type of transformation ('standardize', 'normalize', 'log', 'diff')
            
        Returns:
            Transformed DataFrame
        """
        try:
            df = self._ensure_dataframe(data)
            
            if transformation == "standardize":
                df = self._standardize_data(df)
            elif transformation == "normalize":
                df = self._normalize_data(df)
            elif transformation == "log":
                df = self._log_transform(df)
            elif transformation == "diff":
                df = self._difference_transform(df)
            else:
                raise ValueError(f"Unsupported transformation: {transformation}")
            
            self.logger.info(f"Data transformed using {transformation}")
            return df
            
        except Exception as e:
            self.logger.error(f"Data transformation failed: {str(e)}")
            raise
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load data from file
        
        Args:
            file_path: Path to data file
            
        Returns:
            Loaded DataFrame
        """
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            file_extension = file_path.suffix.lower()
            
            if file_extension == '.csv':
                df = pd.read_csv(file_path)
            elif file_extension == '.json':
                df = pd.read_json(file_path)
            elif file_extension == '.parquet':
                df = pd.read_parquet(file_path)
            elif file_extension in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
            
            self.logger.info(f"Data loaded from {file_path}. Shape: {df.shape}")
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to load data from {file_path}: {str(e)}")
            raise
    
    def save_data(self, data: pd.DataFrame, file_path: str, format: str = "csv") -> None:
        """
        Save data to file
        
        Args:
            data: DataFrame to save
            file_path: Output file path
            format: Output format ('csv', 'json', 'parquet', 'excel')
        """
        try:
            file_path = Path(file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            if format == "csv":
                data.to_csv(file_path, index=False)
            elif format == "json":
                data.to_json(file_path, orient='records', indent=2)
            elif format == "parquet":
                data.to_parquet(file_path, index=False)
            elif format == "excel":
                data.to_excel(file_path, index=False)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            self.logger.info(f"Data saved to {file_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save data to {file_path}: {str(e)}")
            raise
    
    def _ensure_dataframe(self, data: Union[pd.DataFrame, Dict[str, List], str]) -> pd.DataFrame:
        """Ensure data is in DataFrame format"""
        if isinstance(data, pd.DataFrame):
            return data.copy()
        elif isinstance(data, dict):
            return pd.DataFrame(data)
        elif isinstance(data, str):
            return self.load_data(data)
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the data"""
        # Forward fill for time series
        if 'ds' in df.columns:
            df = df.sort_values('ds')
            df = df.ffill()
        
        # Backward fill for remaining NaNs
        df = df.bfill()
        
        # Drop rows with still missing values
        df = df.dropna()
        
        return df
    
    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle outliers using IQR method"""
        if 'y' not in df.columns:
            return df
        
        Q1 = df['y'].quantile(0.25)
        Q3 = df['y'].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Cap outliers instead of removing them
        df['y'] = df['y'].clip(lower=lower_bound, upper=upper_bound)
        
        return df
    
    def _ensure_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure proper data types"""
        if 'ds' in df.columns:
            df['ds'] = pd.to_datetime(df['ds'])
        
        if 'y' in df.columns:
            df['y'] = pd.to_numeric(df['y'], errors='coerce')
        
        return df
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features"""
        df = df.copy()
        
        # Extract time components
        df['year'] = df['ds'].dt.year
        df['month'] = df['ds'].dt.month
        df['day'] = df['ds'].dt.day
        df['day_of_week'] = df['ds'].dt.dayofweek
        df['day_of_year'] = df['ds'].dt.dayofyear
        df['quarter'] = df['ds'].dt.quarter
        df['is_weekend'] = df['ds'].dt.dayofweek.isin([5, 6]).astype(int)
        
        return df
    
    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add lag features"""
        df = df.copy()
        
        if 'y' in df.columns:
            max_lag = min(self.config.timeseries.max_lag, len(df) // 4)
            
            for lag in range(1, max_lag + 1):
                df[f'y_lag_{lag}'] = df['y'].shift(lag)
        
        return df
    
    def _add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling statistics"""
        df = df.copy()
        
        if 'y' in df.columns:
            window_sizes = [3, 7, 14, 30]
            
            for window in window_sizes:
                if window < len(df):
                    df[f'y_rolling_mean_{window}'] = df['y'].rolling(window=window).mean()
                    df[f'y_rolling_std_{window}'] = df['y'].rolling(window=window).std()
                    df[f'y_rolling_min_{window}'] = df['y'].rolling(window=window).min()
                    df[f'y_rolling_max_{window}'] = df['y'].rolling(window=window).max()
        
        return df
    
    def _add_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add trend features"""
        df = df.copy()
        
        if 'y' in df.columns:
            # Linear trend
            x = np.arange(len(df))
            slope, intercept = np.polyfit(x, df['y'].ffill(), 1)
            df['trend'] = slope * x + intercept
            
            # Trend strength
            df['trend_strength'] = abs(slope)
        
        return df
    
    def _add_seasonal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add seasonal features"""
        df = df.copy()
        
        if 'y' in df.columns and 'ds' in df.columns:
            # Seasonal decomposition (simplified)
            df['seasonal_cycle'] = np.sin(2 * np.pi * df['ds'].dt.dayofyear / 365.25)
            df['seasonal_cycle_cos'] = np.cos(2 * np.pi * df['ds'].dt.dayofyear / 365.25)
        
        return df
    
    def _standardize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize numerical columns"""
        df = df.copy()
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if col != 'ds':  # Don't standardize date column
                mean_val = df[col].mean()
                std_val = df[col].std()
                if std_val > 0:
                    df[col] = (df[col] - mean_val) / std_val
        
        return df
    
    def _normalize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize numerical columns to [0, 1] range"""
        df = df.copy()
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if col != 'ds':  # Don't normalize date column
                min_val = df[col].min()
                max_val = df[col].max()
                if max_val > min_val:
                    df[col] = (df[col] - min_val) / (max_val - min_val)
        
        return df
    
    def _log_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply log transformation to positive values"""
        df = df.copy()
        
        if 'y' in df.columns:
            # Add small constant to avoid log(0)
            df['y'] = np.log(df['y'] + 1e-8)
        
        return df
    
    def _difference_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply differencing transformation"""
        df = df.copy()
        
        if 'y' in df.columns:
            df['y'] = df['y'].diff()
            df = df.dropna()  # Remove NaN from first difference
        
        return df
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return self.processing_stats.copy() 