#!/usr/bin/env python3
"""orchestrate signal computation"""

import polars as pl
import yaml
from signals import *

def compute_alphas(data: pl.DataFrame, config_path: str = "config_files/research_config.yaml") -> pl.DataFrame:
    """compute the configured signal"""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    signal_config = config["signal"]
    signal_name = signal_config["name"]
    signal_type = signal_config["type"]
    
    # compute signal using getattr
    compute_function_name = f"compute_{signal_type}"
    try:
        compute_function = getattr(signals, compute_function_name)
    except AttributeError:
        # Fallback to global namespace if not found in signals module
        compute_function = globals().get(compute_function_name)
        if compute_function is None:
            raise ValueError(f"unknown signal type: {signal_type}")
    
    data = compute_function(data, signal_name, signal_config)
    
    # convert to alpha
    data = data.with_columns(
        ((pl.col(signal_name) - pl.col(signal_name).mean().over("date")) 
         / pl.col(signal_name).std().over("date")).alias(f"{signal_name}_score")
    ).with_columns(
        (pl.col(f"{signal_name}_score") * pl.col("specific_risk")).alias(f"{signal_name}_alpha")
    )
    
    return data