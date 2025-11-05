#!/usr/bin/env python3
"""core scientific visualizations using silverfund"""

import polars as pl
import sf_quant.performance as sfp
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

def create_core_visualizations(weights: pl.DataFrame, alpha_data: pl.DataFrame, output_path: str):
    """create all core visualizations for signal research"""
    
    Path(output_path).mkdir(parents=True, exist_ok=True)
    
    # get signal name from alpha data
    signal_name = _get_signal_name(alpha_data)
    
    # 1. generate returns from weights
    returns = sfp.generate_returns_from_weights(weights=weights)
    
    # 2. cumulative returns chart
    sfp.generate_returns_chart(
        returns=returns,
        title=f"{signal_name} - Cumulative Returns",
        log_scale=True,
        file_name=Path(output_path) / "cumulative_returns.png"
    )
    
    # 3. drawdown chart
    sfp.generate_drawdown_chart(
        returns=returns,
        title=f"{signal_name} - Drawdowns", 
        file_name=Path(output_path) / "drawdowns.png"
    )
    
    # 4. performance summary table
    summary = sfp.generate_summary_table(returns=returns)
    summary.write_parquet(Path(output_path) / "performance_summary.parquet")
    
    # 5. information coefficient over time
    _plot_information_coefficient(alpha_data, signal_name, output_path)
    
    # 6. quantile portfolio returns
    _plot_quantile_returns(alpha_data, signal_name, output_path)
    
    # 7. z-score distribution
    _plot_zscore_distribution(alpha_data, signal_name, output_path)
    
    # 8. weight distribution
    _plot_weight_distribution(weights, output_path)
    
    # 9. turnover analysis
    turnover = sfp.calculate_turnover(weights=weights)
    turnover.write_parquet(Path(output_path) / "turnover.parquet")

def _get_signal_name(alpha_data: pl.DataFrame) -> str:
    """extract signal name from alpha columns"""
    alpha_cols = [col for col in alpha_data.columns if col.endswith('_alpha')]
    if not alpha_cols:
        raise ValueError("no alpha columns found")
    return alpha_cols[0].replace('_alpha', '')

def _plot_information_coefficient(alpha_data: pl.DataFrame, signal_name: str, output_path: str):
    """plot information coefficient over time"""
    try:
        # calculate IC (correlation between signal and forward returns)
        # this is simplified - you'd need to align dates properly
        ic_data = alpha_data.with_columns(
            pl.col(f"{signal_name}_alpha").shift(-1).alias('forward_alpha')
        ).group_by("date").agg(
            pl.corr(f"{signal_name}_alpha", "forward_alpha").alias('IC')
        ).filter(pl.col("IC").is_not_null())
        
        plt.figure(figsize=(10, 6))
        plt.plot(ic_data["date"], ic_data["IC"])
        plt.title(f"{signal_name} - Information Coefficient")
        plt.xlabel("Date")
        plt.ylabel("IC")
        plt.grid(True)
        plt.savefig(Path(output_path) / "information_coefficient.png")
        plt.close()
        
    except Exception as e:
        print(f"error creating IC plot: {e}")

def _plot_quantile_returns(alpha_data: pl.DataFrame, signal_name: str, output_path: str):
    """plot quantile portfolio returns"""
    try:
        # create quintile portfolios based on signal
        quantile_data = alpha_data.with_columns(
            pl.col(f"{signal_name}_alpha").qcut(5).over("date").alias('quantile')
        ).group_by(["date", "quantile"]).agg(
            pl.col("return").mean().alias('quantile_return')
        )
        
        # pivot to get top and bottom quintile returns
        pivot_data = quantile_data.pivot(
            index="date", 
            columns="quantile", 
            values="quantile_return"
        )
        
        plt.figure(figsize=(10, 6))
        plt.plot(pivot_data["date"], pivot_data["4"] - pivot_data["0"], label="Top-Bottom Spread")
        plt.title(f"{signal_name} - Quantile Spread Returns")
        plt.xlabel("Date")
        plt.ylabel("Return Spread")
        plt.legend()
        plt.grid(True)
        plt.savefig(Path(output_path) / "quantile_returns.png")
        plt.close()
        
    except Exception as e:
        print(f"error creating quantile plot: {e}")

def _plot_zscore_distribution(alpha_data: pl.DataFrame, signal_name: str, output_path: str):
    """plot z-score distribution histogram"""
    try:
        zscores = alpha_data.select(
            (pl.col(f"{signal_name}_alpha") / pl.col("specific_risk")).alias('zscore')
        ).filter(pl.col('zscore').is_not_null())
        
        plt.figure(figsize=(10, 6))
        plt.hist(zscores["zscore"], bins=50, alpha=0.7)
        plt.title(f"{signal_name} - Z-score Distribution")
        plt.xlabel("Z-score")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.savefig(Path(output_path) / "zscore_distribution.png")
        plt.close()
        
    except Exception as e:
        print(f"error creating zscore plot: {e}")

def _plot_weight_distribution(weights: pl.DataFrame, output_path: str):
    """plot weight distribution histogram"""
    try:
        plt.figure(figsize=(10, 6))
        plt.hist(weights["weight"], bins=50, alpha=0.7)
        plt.title("Portfolio Weight Distribution")
        plt.xlabel("Weight")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.savefig(Path(output_path) / "weight_distribution.png")
        plt.close()
        
    except Exception as e:
        print(f"error creating weight distribution plot: {e}")
