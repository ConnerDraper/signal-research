#!/usr/bin/env python3
"""
Complete research pipeline: data → signals → backtest → results
"""

import polars as pl
import yaml
import argparse
from pathlib import Path
import sys

# Absolute imports
from src.data_loader import load_barra_data
from src.signal_loader import compute_alphas
from src.backtester import run_mvo_backtest
import sf_quant.performance as sfp

def main(config_path: str):
    config_file = Path(config_path)
    
    # 1. Derive Run Name from Filename
    # e.g., "str_22_low_quality.yaml" -> "str_22_low_quality"
    run_name = config_file.stem 
    
    print(f"======================================================")
    print(f"Starting Pipeline for: {run_name}")
    print(f"Config File: {config_file}")
    print(f"======================================================")

    # 2. Load Config
    with open(config_file) as f:
        config = yaml.safe_load(f)
    
    # 3. Extract Signal Config
    if "signal" not in config:
        print(f"Error: Key 'signal' not found in {config_file}")
        sys.exit(1)

    signal_config = config["signal"]
    signal_name = signal_config["name"]

    # 4. Prepare Output Directory
    # e.g., data/results/str_22_low_quality/
    output_dir = Path(config["output"]["results_path"]) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # 5. Load Data
    # Note: passing the path string as your loader likely expects a string
    print("Loading data...")
    data = load_barra_data(str(config_file))
    
    # 6. Compute Signals
    print(f"Computing signals (Type: {signal_config.get('type')})...")
    alpha_data = compute_alphas(data, signal_config)
    
    # 7. Run Backtest
    print("Running backtests...")
    weights = run_mvo_backtest(
        alpha_data=alpha_data,
        signal_name=signal_name, 
        constraints=config["backtest"]["constraints"],
        gamma=config["backtest"]["gamma"],
        n_cpus=config["backtest"]["n_cpus"]
    )
    returns = sfp.generate_returns_from_weights(weights=weights)
    
    # 8. Save Core Artifacts
    print(f"Saving results to: {output_dir}")
    
    weights.write_parquet(output_dir / f"{run_name}_weights.parquet")
    #alpha_data.write_parquet(output_dir / f"{run_name}_alphas.parquet")    # takes up too much space & is unnecessary
    returns.write_parquet(output_dir / f"{run_name}_returns.parquet")
    
    # # 10. Generate Visualizations
    # print("Generating visualizations...")
    # create_core_visualizations(
    #     weights=weights,
    #     alpha_data=alpha_data,
    #     signal_name=run_name,
    #     output_path=str(output_dir)
    # )
    
    print(f"Pipeline complete for: {run_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the research pipeline for a specific signal.")
    
    parser.add_argument(
        "--config", 
        type=str, 
        required=True,
        dest="config_path", # Maps the input to 'args.config_path'
        help="Path to the configuration file."
    )
    
    args = parser.parse_args()
    # passes the string (e.g., "config_files/str_22_low_quality.yaml") to main()
    main(config_path=args.config_path)