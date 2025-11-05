import pytest
import polars as pl
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from signals.idio_vol import compute_idio_vol

def test_compute_idio_vol_basic():
    """Test basic idiosyncratic volatility computation"""
    data = pl.DataFrame({
        'date': ['2023-01-01', '2023-01-02', '2023-01-03'] * 3,
        'permno': [1, 1, 1, 2, 2, 2, 3, 3, 3],
        'returns': [0.01, -0.02, 0.015, -0.01, 0.02, -0.015, 0.005, -0.01, 0.01]
    })
    
    config = {
        'lookback': 2,
        'min_periods': 1
    }
    
    result = compute_idio_vol(data, 'test_idio_vol', config)
    
    assert 'test_idio_vol' in result.columns
    assert len(result) == len(data)
    # Add more specific assertions based on your idio_vol logic

def test_compute_idio_vol_missing_columns():
    """Test that appropriate errors are raised for missing data"""
    data = pl.DataFrame({
        'date': ['2023-01-01'],
        'permno': [1]
        # Missing 'returns' column
    })
    
    config = {'lookback': 20}
    
    with pytest.raises(Exception):  # Or more specific exception
        compute_idio_vol(data, 'test_signal', config)