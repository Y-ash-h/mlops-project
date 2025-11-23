"""Unit tests for preprocessing pipeline logic."""
import pytest
import pandas as pd
import numpy as np
import sys
import os

# Allow importing modules from 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../mlops-pipeline')))

# Import real preprocessing functions
from src.data_preprocessing.preprocess_router import clean_logic, feature_engineering_logic

# --- TESTS ---

def test_cleaning_removes_nulls():
    """Test that rows with NaN targets are removed."""
    raw_data = pd.DataFrame({
        'feature1': [1, 2, 3],
        'target': [1, np.nan, 0]  # One bad row
    })
    
    cleaned = clean_logic(raw_data)
    
    assert len(cleaned) == 2, "Should have dropped 1 row"
    assert cleaned['target'].isna().sum() == 0, "No NaNs allowed in target"

def test_feature_engineering_creates_column():
    """Test that the interaction column is created correctly."""
    data = pd.DataFrame({
        'feature1': [2, 3],
        'feature2': [4, 5]
    })
    
    processed = feature_engineering_logic(data)
    
    assert 'feature1_x_feature2' in processed.columns, "New feature missing"
    # 2*4 = 8
    assert processed.iloc[0]['feature1_x_feature2'] == 8

def test_feature_engineering_handles_missing_columns():
    """Test that feature engineering doesn't break if columns are missing."""
    data = pd.DataFrame({
        'feature1': [2, 3]
    })
    
    processed = feature_engineering_logic(data)
    
    # Should not create the interaction column
    assert 'feature1_x_feature2' not in processed.columns

def test_cleaning_preserves_good_data():
    """Test that cleaning doesn't drop rows with valid targets."""
    raw_data = pd.DataFrame({
        'feature1': [1, 2, 3],
        'target': [1, 0, 1]  # All valid
    })
    
    cleaned = clean_logic(raw_data)
    
    assert len(cleaned) == 3, "Should not drop any rows"
    assert cleaned['target'].isna().sum() == 0, "No NaNs should exist"
