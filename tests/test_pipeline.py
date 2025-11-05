import pytest
import polars as pl
from unittest.mock import patch, MagicMock
import yaml

from signal_loader import compute_alphas

def test_compute_alphas_structure(pipeline_input_data):
    """Test that compute_alphas returns the expected structure"""
    # Mock the actual signal computation to just add a dummy signal column
    with patch('signal_loader.signals') as mock_signals:
        signal_config = {'name': 'test_signal', 'type': 'idio_vol'}
        mock_compute_func = MagicMock()
        mock_compute_func.return_value = pipeline_input_data.with_columns(
            pl.lit([1.0, 2.0, 1.5, 2.5, 1.8] * 2).alias(signal_config["name"])
        )
        mock_signals.compute_idio_vol = mock_compute_func
        result = compute_alphas(pipeline_input_data, signal_config)
        
        # Check that expected columns are added
        assert signal_config["name"] in result.columns
        assert f'{signal_config["name"]}_score' in result.columns
        assert f'{signal_config["name"]}_alpha' in result.columns
        
        # Check that original columns are preserved
        assert 'specific_risk' in result.columns
        assert 'date' in result.columns

def test_compute_alphas_unknown_signal_type(pipeline_input_data, tmp_path):
    """Test that unknown signal types raise appropriate errors"""
    signal_config = {
        'name': 'test_signal',
        'type': 'unknown_signal'
    }
    
    with pytest.raises(ValueError, match="unknown signal type"):
        compute_alphas(pipeline_input_data, signal_config)

def test_compute_alphas_alpha_calculation():
    """Test the alpha calculation logic"""
    # Create data where we can predict the z-score calculation
    test_data = pl.DataFrame({
        'date': ['2023-01-01'] * 3,
        'specific_risk': [1.0, 1.0, 1.0],
        'permno': [1, 2, 3] # Add permno for grouping
    })
    
    with patch('signal_loader.signals') as mock_signals:
        signal_config = {'name': 'test_signal', 'type': 'idio_vol'}
        # Mock the dynamically accessed compute function
        mock_compute_func = MagicMock(return_value=test_data.with_columns(
            pl.Series(signal_config["name"], [1.0, 2.0, 3.0])
        ))
        mock_signals.compute_idio_vol = mock_compute_func
        result = compute_alphas(test_data, signal_config)
        # The z-scores should be [-1, 0, 1] for values [1, 2, 3]
        scores = result['test_signal_score'].to_list()
        alphas = result['test_signal_alpha'].to_list()

        # Use pytest.approx for robust floating point comparisons
        expected_scores = [-1.0, 0.0, 1.0] # Standardized values (ddof=1)
        assert scores == pytest.approx(expected_scores)
        # Since specific_risk is 1.0, alpha should equal score
        assert alphas == pytest.approx(expected_scores)