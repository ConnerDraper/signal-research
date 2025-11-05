#!/usr/bin/env python3
"""
Complete research pipeline: data → signals → backtest → results
Runs on supercomputer, saves minimal outputs for visualization.
"""

import polars as pl
import yaml

from data_loader import load_barra_data
from signal_loader import compute_alphas
from backtester import run_mvo_backtest
import sf_quant.performance as sfp

def main(config_path: str = "config_files/research_config.yaml"):
    # load config
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # 1. load data
    print("loading data...")
    data = load_barra_data(config_path)
    
    # 2. compute signals
    print("computing signals...")
    alpha_data = compute_alphas(data, config_path)
    
    # 3. run backtests
    print("running backtests...")
    weights = run_mvo_backtest(
        alpha_data=alpha_data,
        constraints=config["backtest"]["constraints"],
        gamma=config["backtest"]["gamma"]
    )
    
    # 4. generate returns for visualization
    print("generating returns...")
    returns = sfp.generate_returns_from_weights(weights=weights)
    
    # 5. save minimal results for visualization
    print("saving results...")
    output_dir = config["output"]["results_path"]
    
    weights.write_parquet(f"{output_dir}/weights.parquet")
    returns.write_parquet(f"{output_dir}/returns.parquet")
    
    # save only essential alpha columns for visualization
    signal_name = config["signal"]["name"]
    alpha_data.select([
        "date", "barrid", f"{signal_name}_alpha", "specific_risk"
    ]).write_parquet(f"{output_dir}/alphas.parquet")
    
    print("pipeline complete!")

if __name__ == "__main__":
    main()