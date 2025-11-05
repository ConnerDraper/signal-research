import pytest
import polars as pl
import yaml
from unittest.mock import patch
from pathlib import Path

from code.data_loader import load_barra_data
from code.signal_loader import compute_alphas
from code.pipeline import main
from code.visualization import create_core_visualizations


@pytest.fixture(scope="module")
def integration_config(tmp_path_factory):
    """
    Creates a temporary config file and output directory for integration tests.
    This config uses the 'idio_vol' signal.
    """
    tmp_dir = tmp_path_factory.mktemp("integration_data")
    config = {
        "data_loading": {
            "start_date": "2023-01-01",
            "end_date": "2023-01-05",
            "russell_filter": True,
            "columns": [
                "date", "barrid", "rootid", "iso_country_code", "return",
                "specific_return", "specific_risk", "daily_volume", "price",
                "market_cap", "predicted_beta"
            ],
        },
        "data_cleaning": {
            "convert_returns_to_decimal": True,
            "replace_zero_volume": True,
            "filters": [{"usa_only": True, "min_price": 5.0, "require_returns": True}],
        },
        "signal": {
            "name": "test_idio_vol",
            "type": "idio_vol",
            "window_size": 3,
            "min_periods": 2,
            "direction": -1,
            "shift": 0,
        },
        "backtest": {
            "constraints": [
                "FullInvestment",
                "LongOnly",
                "NoBuyingOnMargin"
            ],
            "gamma": 400
        },
        "output": {"results_path": str(tmp_dir / "results")},
    }
    config_path = tmp_dir / "integration_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    return config_path


@pytest.fixture(scope="module")
def mock_integration_data():
    """
    A small, realistic DataFrame that will be "loaded" by the mock.
    Includes data that should be filtered out.
    """
    return pl.DataFrame({
        "date": ["2023-01-02", "2023-01-03", "2023-01-04"] * 2 + ["2023-01-02"],
        "barrid": ["US1", "US1", "US1", "US2", "US2", "US2", "CA1"],
        "rootid": ["US1", "US1", "US1", "US2", "US2", "US2", "CA1"],
        "iso_country_code": ["USA", "USA", "USA", "USA", "USA", "USA", "CAN"],
        "return": [1.0, -0.5, 2.0, 0.5, 0.2, -0.1, 3.0],
        "specific_return": [1.1, -0.4, 2.2, 0.6, 0.3, -0.0, 3.1],
        "specific_risk": [5.0, 5.1, 4.9, 10.0, 10.1, 9.9, 8.0],
        "daily_volume": [1000, 2000, 1500, 0, 5000, 4000, 10000],
        "price": [10.0, 10.1, 9.9, 4.0, 20.0, 20.1, 50.0], # US2 has a low price day
        "market_cap": [1e9, 1.1e9, 1.2e9, 5e8, 2e9, 2.1e9, 3e9],
        "predicted_beta": [1.0, 1.0, 1.0, 1.2, 1.2, 1.2, 0.9],
    })


@patch("code.data_loader.sfd.load_assets")
def test_load_and_clean_data(mock_load_assets, mock_integration_data, integration_config):
    """Test the data loading and cleaning step of the pipeline."""
    mock_load_assets.return_value = mock_integration_data

    data = load_barra_data(integration_config)

    # Check that non-USA and low-price rows were filtered
    assert "CA1" not in data["barrid"].to_list()
    assert data["price"].min() >= 5.0
    # Check that returns were converted to decimal
    assert data["return"].max() < 1.0


@patch("code.data_loader.sfd.load_assets")
def test_compute_alphas_integration(mock_load_assets, mock_integration_data, integration_config):
    """Test that alphas are computed correctly on loaded data."""
    mock_load_assets.return_value = mock_integration_data
    data = load_barra_data(integration_config)

    with open(integration_config) as f:
        config = yaml.safe_load(f)
    alpha_data = compute_alphas(data, config["signal"])

    # Check for signal, score, and alpha columns
    assert "test_idio_vol" in alpha_data.columns
    assert "test_idio_vol_score" in alpha_data.columns
    assert "test_idio_vol_alpha" in alpha_data.columns
    assert alpha_data["test_idio_vol_alpha"].is_not_null().any()


@patch("code.pipeline.sfp.generate_returns_from_weights")
@patch("code.backtester.sfb.backtest_parallel")
@patch("code.data_loader.sfd.load_assets")
def test_full_pipeline_run(
    mock_load_assets,
    mock_backtest_parallel,
    mock_gen_returns,
    mock_integration_data,
    integration_config,
):
    """Test that the main pipeline runs end-to-end and creates output files."""
    mock_load_assets.return_value = mock_integration_data

    # Configure mocks to return plausible data structures
    mock_backtest_parallel.return_value = pl.DataFrame(
        {"date": ["2023-01-04"], "barrid": ["US1"], "weight": [1.0]}
    )
    mock_gen_returns.return_value = pl.DataFrame(
        # The sf_quant library expects a "portfolio" column for labeling
        {
            "date": ["2023-01-04"],
            "portfolio_return": [0.01],
            "portfolio": "test_portfolio",
            "return": [0.005], # Add benchmark return column
        }
    )

    main(integration_config)

    # Check that output files were created
    config = yaml.safe_load(Path(integration_config).read_text())
    output_dir = Path(config["output"]["results_path"])
    for file in ["weights.parquet", "returns.parquet", "alphas.parquet"]:
        output_file = output_dir / file
        assert output_file.exists()
        assert pl.read_parquet(output_file).height > 0


@patch("code.visualization.sfp.generate_returns_from_weights")
@patch("matplotlib.pyplot.savefig")
def test_visualization_script_runs(mock_savefig, mock_gen_returns, integration_config):
    """
    Test that the visualization script runs without errors.
    We mock savefig to avoid creating actual image files.
    """
    config = yaml.safe_load(Path(integration_config).read_text())
    output_dir = Path(config["output"]["results_path"])
    output_dir.mkdir(parents=True, exist_ok=True)
    signal_name = config["signal"]["name"]

    # Create dummy data files that the script expects
    weights_data = pl.DataFrame({"date": ["2023-01-04"], "barrid": ["US1"], "weight": [1.0]})
    alpha_data = pl.DataFrame({
        "date": ["2023-01-04"], "barrid": ["US1"], f"{signal_name}_alpha": [0.5],
        "specific_risk": [0.1], "return": [0.01]
    })
    # Mock the return generation to isolate the visualization logic
    mock_gen_returns.return_value = pl.DataFrame({
        "date": ["2023-01-04"],
        "portfolio_return": [0.01],
        "portfolio": "test_portfolio",
        "return": [0.005], # Add benchmark return column
    })

    # Run the visualization function
    create_core_visualizations(weights_data, alpha_data, signal_name, str(output_dir))

    # Assert that savefig was called for the custom plots in visualization.py
    assert mock_savefig.call_count >= 4