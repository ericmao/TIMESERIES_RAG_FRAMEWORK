"""
Tests for Data Processing Utilities
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any
import tempfile
import os

from src.utils.data_processor import TimeSeriesDataProcessor
from src.config.config import get_config

class TestTimeSeriesDataProcessor:
    """Test cases for TimeSeriesDataProcessor"""
    
    @pytest.fixture
    def data_processor(self):
        """Create a data processor instance"""
        return TimeSeriesDataProcessor()
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample time series data"""
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        values = np.sin(np.arange(len(dates)) * 2 * np.pi / 365) + np.random.normal(0, 0.1, len(dates))
        
        return pd.DataFrame({
            "ds": dates,
            "y": values
        })
    
    @pytest.fixture
    def sample_data_with_issues(self):
        """Generate sample data with common issues"""
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        values = np.sin(np.arange(len(dates)) * 2 * np.pi / 365) + np.random.normal(0, 0.1, len(dates))
        
        # Add some issues
        values[50] = np.nan  # Missing value
        values[100] = 1000.0  # Outlier
        values[150] = -1000.0  # Outlier
        
        return pd.DataFrame({
            "ds": dates,
            "y": values
        })
    
    def test_data_processor_initialization(self, data_processor):
        """Test data processor initialization"""
        assert data_processor is not None
        assert hasattr(data_processor, 'supported_formats')
        assert hasattr(data_processor, 'processing_stats')
    
    def test_validate_data_valid(self, data_processor, sample_data):
        """Test data validation with valid data"""
        result = data_processor.validate_data(sample_data)
        
        assert result["is_valid"] == True
        assert len(result["errors"]) == 0
        assert "statistics" in result
        assert result["statistics"]["length"] == len(sample_data)
    
    def test_validate_data_missing_columns(self, data_processor):
        """Test data validation with missing columns"""
        invalid_data = pd.DataFrame({
            "date": pd.date_range(start='2023-01-01', periods=10),
            "value": np.random.randn(10)
        })
        
        result = data_processor.validate_data(invalid_data)
        
        assert result["is_valid"] == False
        assert len(result["errors"]) > 0
        assert "Missing required columns" in result["errors"][0]
    
    def test_validate_data_wrong_types(self, data_processor):
        """Test data validation with wrong data types"""
        invalid_data = pd.DataFrame({
            "ds": ["not", "a", "date"],
            "y": ["not", "numeric", "data"]
        })
        
        result = data_processor.validate_data(invalid_data)
        
        assert result["is_valid"] == False
        assert len(result["errors"]) > 0
    
    def test_validate_data_too_short(self, data_processor):
        """Test data validation with too short data"""
        short_data = pd.DataFrame({
            "ds": pd.date_range(start='2023-01-01', periods=10),
            "y": np.random.randn(10)
        })
        
        result = data_processor.validate_data(short_data)
        
        assert result["is_valid"] == False
        assert len(result["errors"]) > 0
        assert "Data too short" in result["errors"][0]
    
    def test_clean_data(self, data_processor, sample_data_with_issues):
        """Test data cleaning"""
        cleaned_data = data_processor.clean_data(sample_data_with_issues)
        
        assert len(cleaned_data) > 0
        assert cleaned_data.isnull().sum().sum() == 0
        assert "ds" in cleaned_data.columns
        assert "y" in cleaned_data.columns
    
    def test_clean_data_duplicates(self, data_processor):
        """Test data cleaning with duplicates"""
        # Create data with duplicates
        dates = pd.date_range(start='2023-01-01', periods=10)
        values = np.random.randn(10)
        
        duplicate_data = pd.DataFrame({
            "ds": list(dates) + [dates[0]],  # Add duplicate
            "y": list(values) + [values[0]]
        })
        
        cleaned_data = data_processor.clean_data(duplicate_data)
        
        assert len(cleaned_data) == len(dates)  # Duplicate should be removed
        assert cleaned_data["ds"].is_unique
    
    def test_engineer_features(self, data_processor, sample_data):
        """Test feature engineering"""
        engineered_data = data_processor.engineer_features(sample_data)
        
        # Check that new features were added
        expected_features = ['year', 'month', 'day', 'day_of_week', 'day_of_year', 'quarter', 'is_weekend']
        for feature in expected_features:
            assert feature in engineered_data.columns
    
    def test_engineer_features_lag(self, data_processor, sample_data):
        """Test lag feature engineering"""
        engineered_data = data_processor.engineer_features(sample_data)
        
        # Check for lag features
        lag_features = [col for col in engineered_data.columns if 'lag' in col]
        assert len(lag_features) > 0
    
    def test_engineer_features_rolling(self, data_processor, sample_data):
        """Test rolling feature engineering"""
        engineered_data = data_processor.engineer_features(sample_data)
        
        # Check for rolling features
        rolling_features = [col for col in engineered_data.columns if 'rolling' in col]
        assert len(rolling_features) > 0
    
    def test_transform_data_standardize(self, data_processor, sample_data):
        """Test data standardization"""
        transformed_data = data_processor.transform_data(sample_data, "standardize")
        
        # Check that y column is standardized
        assert abs(transformed_data['y'].mean()) < 1e-10
        assert abs(transformed_data['y'].std() - 1.0) < 1e-10
    
    def test_transform_data_normalize(self, data_processor, sample_data):
        """Test data normalization"""
        transformed_data = data_processor.transform_data(sample_data, "normalize")
        
        # Check that y column is normalized to [0, 1]
        assert transformed_data['y'].min() >= 0
        assert transformed_data['y'].max() <= 1
    
    def test_transform_data_log(self, data_processor, sample_data):
        """Test log transformation"""
        transformed_data = data_processor.transform_data(sample_data, "log")
        
        # Check that log transformation was applied
        assert "y" in transformed_data.columns
        assert not transformed_data['y'].isnull().any()
    
    def test_transform_data_diff(self, data_processor, sample_data):
        """Test differencing transformation"""
        transformed_data = data_processor.transform_data(sample_data, "diff")
        
        # Check that differencing was applied
        assert "y" in transformed_data.columns
        assert len(transformed_data) < len(sample_data)  # First difference is NaN
    
    def test_transform_data_invalid(self, data_processor, sample_data):
        """Test invalid transformation"""
        with pytest.raises(ValueError):
            data_processor.transform_data(sample_data, "invalid_transformation")
    
    def test_load_data_csv(self, data_processor):
        """Test loading CSV data"""
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("ds,y\n2023-01-01,1.0\n2023-01-02,2.0\n")
            temp_file = f.name
        
        try:
            data = data_processor.load_data(temp_file)
            assert isinstance(data, pd.DataFrame)
            assert len(data) == 2
            assert "ds" in data.columns
            assert "y" in data.columns
        finally:
            os.unlink(temp_file)
    
    def test_load_data_json(self, data_processor):
        """Test loading JSON data"""
        # Create temporary JSON file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write('{"ds": ["2023-01-01", "2023-01-02"], "y": [1.0, 2.0]}')
            temp_file = f.name
        
        try:
            data = data_processor.load_data(temp_file)
            assert isinstance(data, pd.DataFrame)
            assert len(data) == 2
        finally:
            os.unlink(temp_file)
    
    def test_load_data_invalid_format(self, data_processor):
        """Test loading data with invalid format"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("invalid data")
            temp_file = f.name
        
        try:
            with pytest.raises(ValueError):
                data_processor.load_data(temp_file)
        finally:
            os.unlink(temp_file)
    
    def test_load_data_file_not_found(self, data_processor):
        """Test loading non-existent file"""
        with pytest.raises(FileNotFoundError):
            data_processor.load_data("nonexistent_file.csv")
    
    def test_save_data_csv(self, data_processor, sample_data):
        """Test saving data as CSV"""
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            temp_file = f.name
        
        try:
            data_processor.save_data(sample_data, temp_file, "csv")
            assert os.path.exists(temp_file)
            
            # Verify data was saved correctly
            loaded_data = data_processor.load_data(temp_file)
            assert len(loaded_data) == len(sample_data)
        finally:
            os.unlink(temp_file)
    
    def test_save_data_json(self, data_processor, sample_data):
        """Test saving data as JSON"""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            temp_file = f.name
        
        try:
            data_processor.save_data(sample_data, temp_file, "json")
            assert os.path.exists(temp_file)
        finally:
            os.unlink(temp_file)
    
    def test_save_data_invalid_format(self, data_processor, sample_data):
        """Test saving data with invalid format"""
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
            temp_file = f.name
        
        try:
            with pytest.raises(ValueError):
                data_processor.save_data(sample_data, temp_file, "invalid_format")
        finally:
            os.unlink(temp_file)
    
    def test_ensure_dataframe_dataframe(self, data_processor, sample_data):
        """Test _ensure_dataframe with DataFrame input"""
        result = data_processor._ensure_dataframe(sample_data)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_data)
    
    def test_ensure_dataframe_dict(self, data_processor):
        """Test _ensure_dataframe with dict input"""
        dict_data = {
            "ds": ["2023-01-01", "2023-01-02"],
            "y": [1.0, 2.0]
        }
        
        result = data_processor._ensure_dataframe(dict_data)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
    
    def test_ensure_dataframe_invalid(self, data_processor):
        """Test _ensure_dataframe with invalid input"""
        with pytest.raises(ValueError):
            data_processor._ensure_dataframe("invalid_data")
    
    def test_handle_missing_values(self, data_processor):
        """Test missing value handling"""
        # Create data with missing values
        data = pd.DataFrame({
            "ds": pd.date_range(start='2023-01-01', periods=5),
            "y": [1.0, np.nan, 3.0, np.nan, 5.0]
        })
        
        result = data_processor._handle_missing_values(data)
        
        assert result.isnull().sum().sum() == 0
        assert len(result) == 5
    
    def test_handle_outliers(self, data_processor):
        """Test outlier handling"""
        # Create data with outliers
        data = pd.DataFrame({
            "ds": pd.date_range(start='2023-01-01', periods=10),
            "y": [1, 2, 3, 4, 5, 1000, 7, 8, 9, 10]  # 1000 is outlier
        })
        
        result = data_processor._handle_outliers(data)
        
        # Check that outlier was capped
        assert result['y'].max() < 1000
    
    def test_ensure_data_types(self, data_processor):
        """Test data type conversion"""
        data = pd.DataFrame({
            "ds": ["2023-01-01", "2023-01-02"],
            "y": ["1.0", "2.0"]
        })
        
        result = data_processor._ensure_data_types(data)
        
        assert pd.api.types.is_datetime64_any_dtype(result['ds'])
        assert pd.api.types.is_numeric_dtype(result['y'])
    
    def test_add_time_features(self, data_processor, sample_data):
        """Test time feature addition"""
        result = data_processor._add_time_features(sample_data)
        
        expected_features = ['year', 'month', 'day', 'day_of_week', 'day_of_year', 'quarter', 'is_weekend']
        for feature in expected_features:
            assert feature in result.columns
    
    def test_add_lag_features(self, data_processor, sample_data):
        """Test lag feature addition"""
        result = data_processor._add_lag_features(sample_data)
        
        # Check for lag features
        lag_features = [col for col in result.columns if 'lag' in col]
        assert len(lag_features) > 0
    
    def test_add_rolling_features(self, data_processor, sample_data):
        """Test rolling feature addition"""
        result = data_processor._add_rolling_features(sample_data)
        
        # Check for rolling features
        rolling_features = [col for col in result.columns if 'rolling' in col]
        assert len(rolling_features) > 0
    
    def test_add_trend_features(self, data_processor, sample_data):
        """Test trend feature addition"""
        result = data_processor._add_trend_features(sample_data)
        
        assert 'trend' in result.columns
        assert 'trend_strength' in result.columns
    
    def test_add_seasonal_features(self, data_processor, sample_data):
        """Test seasonal feature addition"""
        result = data_processor._add_seasonal_features(sample_data)
        
        assert 'seasonal_cycle' in result.columns
        assert 'seasonal_cycle_cos' in result.columns
    
    def test_get_processing_stats(self, data_processor):
        """Test getting processing statistics"""
        stats = data_processor.get_processing_stats()
        assert isinstance(stats, dict)

# Integration tests
class TestDataProcessorIntegration:
    """Integration tests for data processing pipeline"""
    
    @pytest.fixture
    def data_processor(self):
        return TimeSeriesDataProcessor()
    
    def test_full_processing_pipeline(self, data_processor):
        """Test complete data processing pipeline"""
        # Create sample data with issues
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        values = np.sin(np.arange(len(dates)) * 2 * np.pi / 365) + np.random.normal(0, 0.1, len(dates))
        
        # Add issues
        values[50] = np.nan
        values[100] = 1000.0
        values[150] = -1000.0
        
        data = pd.DataFrame({
            "ds": dates,
            "y": values
        })
        
        # Validate data
        validation_result = data_processor.validate_data(data)
        assert validation_result["is_valid"] == True
        
        # Clean data
        cleaned_data = data_processor.clean_data(data)
        assert cleaned_data.isnull().sum().sum() == 0
        
        # Engineer features
        engineered_data = data_processor.engineer_features(cleaned_data)
        assert len(engineered_data.columns) > len(cleaned_data.columns)
        
        # Transform data
        transformed_data = data_processor.transform_data(engineered_data, "standardize")
        assert abs(transformed_data['y'].mean()) < 1e-10
    
    def test_data_persistence(self, data_processor):
        """Test data loading and saving"""
        # Create sample data
        data = pd.DataFrame({
            "ds": pd.date_range(start='2023-01-01', periods=10),
            "y": np.random.randn(10)
        })
        
        # Save data
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            temp_file = f.name
        
        try:
            data_processor.save_data(data, temp_file, "csv")
            
            # Load data
            loaded_data = data_processor.load_data(temp_file)
            
            # Compare data
            assert len(loaded_data) == len(data)
            assert list(loaded_data.columns) == list(data.columns)
            
        finally:
            os.unlink(temp_file) 