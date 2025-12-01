import pytest
import polars as pl
from polars.testing import assert_frame_equal

from src.signals.idio_vol import compute_idio_vol


def test_compute_idio_vol_basic():
    """Test basic idiosyncratic volatility computation."""
    data = pl.DataFrame(
        {
            "date": ["2023-01-01", "2023-01-02", "2023-01-03"] * 2,
            "barrid": ["A", "A", "A", "B", "B", "B"],
            "return": [0.01, -0.02, 0.015, -0.01, 0.02, -0.015],
        }
    )

    config = {"window_size": 3, "min_periods": 2, "direction": 1, "shift": 0}

    result = compute_idio_vol(data, "test_idio_vol", config)

    assert "test_idio_vol" in result.columns
    assert len(result) == len(data)

    # Check that the calculation is correct for one group
    # sample std of [0.01, -0.02, 0.015] is approx 0.018929
    expected_vol = 0.018929694486000914
    actual_vol = result.filter(pl.col("barrid") == "A")["test_idio_vol"].item(2)
    assert actual_vol == pytest.approx(expected_vol)


def test_compute_idio_vol_missing_columns():
    """Test that appropriate errors are raised for missing data."""
    data = pl.DataFrame({"date": ["2023-01-01"], "barrid": ["A"]})  # Missing 'return'

    config = {"window_size": 20, "direction": 1, "shift": 0}

    with pytest.raises(pl.exceptions.ColumnNotFoundError):
        compute_idio_vol(data, "test_signal", config)