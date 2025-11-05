"""idio volatility signal"""
import polars as pl

def compute_idio_vol(data: pl.DataFrame, name: str, config: dict) -> pl.DataFrame:
    return data.with_columns(
        pl.col("specific_risk")
        .rolling_mean(window_size=config["window_size"])
        .mul(config["direction"])
        .shift(config["shift"])
        .over('barrid')
        .alias(name)
    )
