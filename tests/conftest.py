import pytest
import polars as pl
import yaml
import os

@pytest.fixture
def barra_data() -> pl.DataFrame:
    """Pytest fixture to create a sample DataFrame mimicking raw Barra data."""
    return pl.DataFrame(
        {
            "date": ["2023-01-01", "2023-01-01", "2023-01-02"],
            "barrid": ["US123", "CA456", "US789"],
            "rootid": ["US123", "CA456", "US789"],
            "iso_country_code": ["USA", "CAN", "USA"],
            "return": [1.0, -0.5, 2.0],
            "specific_return": [1.1, -0.4, 2.2],
            "specific_risk": [5.0, 10.0, 4.0],
            "daily_volume": [1000, 0, 5000],
            "price": [10.0, 5.0, 20.0],
            "market_cap": [1e9, 5e8, 2e9],
        }
    )


@pytest.fixture
def pipeline_input_data() -> pl.DataFrame:
    """Create sample data representing the input for the main pipeline compute step."""
    return pl.DataFrame(
        {
            "date": ["2023-01-01"] * 5 + ["2023-01-02"] * 5,
            "permno": list(range(5)) * 2,
            "specific_risk": [1.0, 1.5, 2.0, 1.2, 0.8] * 2,
            "returns": [0.01, -0.02, 0.015, -0.01, 0.02] * 2,
            "volume": [1000000, 2000000, 1500000, 1800000, 1200000] * 2,
            "price": [100.0, 50.0, 75.0, 120.0, 80.0] * 2,
        }
    )

@pytest.fixture
def test_config():
    """Test configuration"""
    return {
        'signal': {
            'name': 'test_signal',
            'type': 'idio_vol',
            'lookback': 20,
            'min_periods': 10
        }
    }

@pytest.fixture
def config_file(tmp_path):
    """Create a temporary config file for testing"""
    config = {
        'signal': {
            'name': 'test_signal',
            'type': 'idio_vol',
            'lookback': 20,
            'min_periods': 10
        }
    }
    config_path = tmp_path / "test_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    return config_path