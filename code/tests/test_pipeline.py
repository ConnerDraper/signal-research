import pytest
import polars as pl
from unittest.mock import patch, MagicMock
import sys
import os

# Add the parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from pipeline import compute_alphas

def test_compute_alphas_structure(sample_data, config_file):
    """Test that compute_alphas returns the expected structure"""
    # Mock the actual signal computation to just add a dummy signal column
    with patch('pipeline.compute_idio_vol') as mock_compute:
        mock_compute.return_value = sample_data.with_columns(
            pl.lit([1.0, 2.0, 1.5, 2.5, 1.8] * 2).alias('test_signal')
        )
        
        result = compute_alphas(sample_data, str(config_file))
        
        # Check that expected columns are added
        assert 'test_signal' in result.columns
        assert 'test_signal_score' in result.columns
        assert 'test_signal_alpha' in result.columns
        
        # Check that original columns are preserved
        assert 'specific_risk' in result.columns
        assert 'date' in result.columns

def test_compute_alphas_unknown_signal_type(sample_data, tmp_path):
    """Test that unknown signal types raise appropriate errors"""
    config = {
        'signal': {
            'name': 'test_signal',
            'type': 'unknown_signal'
        }
    }
    config_path = tmp_path / "bad_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    with pytest.raises(ValueError, match="unknown signal type"):
        compute_alphas(sample_data, str(config_path))

def test_compute_alphas_alpha_calculation(sample_data, config_file):
    """Test the alpha calculation logic"""
    # Create data where we can predict the z-score calculation
    test_data = pl.DataFrame({
        'date': ['2023-01-01'] * 3,
        'specific_risk': [1.0, 1.0, 1.0],
        'dummy_signal': [1.0, 2.0, 3.0]
    })
    
    with patch('pipeline.compute_idio_vol') as mock_compute:
        mock_compute.return_value = test_data
        
        result = compute_alphas(test_data, str(config_file))
        
        # The z-scores should be [-1, 0, 1] for values [1, 2, 3]
        scores = result['test_signal_score'].to_list()
        alphas = result['test_signal_alpha'].to_list()
        
        # Since specific_risk is 1.0, alpha should equal score
        assert scores == alphas