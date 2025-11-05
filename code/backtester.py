#!/usr/bin/env python3
"""MVO backtesting engine"""

import polars as pl
import sf_quant.backtester as sfb
import sf_quant.optimizer as sfo

def run_mvo_backtest(alpha_data: pl.DataFrame, constraints: list, gamma: int) -> pl.DataFrame:
    """run MVO backtest on alpha data"""
    
    # convert constraint names to objects using getattr
    constraint_objects = []
    for constraint_name in constraints:
        try:
            constraint_class = getattr(sfo.constraints, constraint_name)
            constraint_objects.append(constraint_class())
        except AttributeError:
            raise ValueError(f"unknown constraint: {constraint_name}")
    
    # get the signal name (assume there's only one alpha column)
    alpha_columns = [col for col in alpha_data.columns if col.endswith('_alpha')]
    if not alpha_columns:
        raise ValueError("no alpha columns found in data")
    
    signal_name = alpha_columns[0].replace('_alpha', '')
    
    # prepare data for backtesting
    backtest_data = alpha_data.filter(
        pl.col(f"{signal_name}_alpha").is_not_null()
    ).select([
        'date', 'barrid', f'{signal_name}_alpha', 'predicted_beta'
    ]).rename({f'{signal_name}_alpha': 'alpha'})
    
    if backtest_data.is_empty():
        print("warning: no data after filtering for backtest")
        return pl.DataFrame()
    
    # run backtest
    weights = sfb.backtest_parallel(
        data=backtest_data,
        constraints=constraint_objects,
        gamma=gamma
    )
    
    return weights
