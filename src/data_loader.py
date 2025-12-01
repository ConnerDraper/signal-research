#!/usr/bin/env python3
"""data loader with cleaning and filtering"""

import polars as pl
import sf_quant.data as sfd
import datetime as dt
import yaml

def load_barra_data(config_path: str = "config_files/research_config.yaml") -> pl.DataFrame:
    """
    Load Barra data using silverfund library
    Then clean data
    Then apply filters according to config
    Then validate data
    """

    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    data_config = config["data_loading"]
    cleaning_config = config.get("data_cleaning", {})
    
    start = dt.datetime.strptime(data_config["start_date"], "%Y-%m-%d").date()
    end = dt.datetime.strptime(data_config["end_date"], "%Y-%m-%d").date()
    
    print(f"loading data: {start} to {end}")
    
    data = sfd.load_assets(
        start=start,
        end=end,
        in_universe=data_config["russell_filter"],
        columns=data_config["columns"]
    )
    
    print(f"loaded {len(data)} rows")
    
    # apply all data preparation steps
    data = prepare_data(data, cleaning_config)
    
    if "filters" in cleaning_config:
        data = apply_filters(data, cleaning_config["filters"])
        print(f"after filtering: {len(data)} rows")
    
    if not validate_data(data):
        raise ValueError("data validation failed")
    
    return data

def prepare_data(data: pl.DataFrame, cleaning_config: dict) -> pl.DataFrame:
    """
    Load Bara data
    Then clean and prepare data for our calcualtions
    """
    transforms = []
    
    if cleaning_config.get("convert_returns_to_decimal", True):
        transforms.extend([
            pl.col('return').truediv(100),
            pl.col('specific_return').truediv(100),
            pl.col('specific_risk').truediv(100)
        ])
    
    if cleaning_config.get("replace_zero_volume", True):
        transforms.append(pl.col("daily_volume").replace(0, None))
    
    if transforms:
        data = data.with_columns(transforms)
    
    return data

def apply_filters(data: pl.DataFrame, filters: list) -> pl.DataFrame:
    """
    Apply data filters found in config
    """
    filter_conditions = []
    
    for filter_config in filters:
        if filter_config.get("usa_only", True):
            filter_conditions.extend([
                pl.col('iso_country_code').eq("USA"),
                pl.col('rootid').eq(pl.col('barrid')),
                pl.col('barrid').str.starts_with('US')
            ])
        
        if filter_config.get("min_price"):
            filter_conditions.append(pl.col("price") >= filter_config["min_price"])
        
        if filter_config.get("min_market_cap"):
            filter_conditions.append(pl.col("market_cap") >= filter_config["min_market_cap"])
        
        if filter_config.get("require_returns", True):
            filter_conditions.append(pl.col("return").is_not_null())
        
        if filter_config.get("require_specific_risk", True):
            filter_conditions.append(pl.col("specific_risk").is_not_null())
    
    if filter_conditions:
        data = data.filter(*filter_conditions)
    
    return data

def validate_data(data: pl.DataFrame) -> bool:
    """basic data validation"""
    if data.is_empty():
        print("warning: data is empty")
        return False
    
    required_cols = ['date', 'barrid', 'return', 'specific_risk']
    missing_cols = [col for col in required_cols if col not in data.columns]
    
    if missing_cols:
        print(f"error: missing columns: {missing_cols}")
        return False
    
    return True