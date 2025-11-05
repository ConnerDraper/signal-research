"""idio volatility signal"""
import polars as pl

def compute_idio_vol(data: pl.DataFrame, name: str, config: dict) -> pl.DataFrame:
    return data.with_columns(
        pl.col("return")
        .rolling_std(window_size=config["window_size"], min_samples=config.get("min_periods"))
        .mul(config["direction"])
        .shift(config["shift"])
        .over('barrid')
        .alias(name)
    )
