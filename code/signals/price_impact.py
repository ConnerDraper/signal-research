"""price impact signal"""
import polars as pl

def compute_price_impact(data: pl.DataFrame, name: str, config: dict) -> pl.DataFrame:
    return data.with_columns(
        (pl.col("return").abs() / (pl.col('daily_volume') * pl.col('price')))
        .rolling_mean(window_size=config["window_size"])
        .mul(config["direction"])
        .shift(config["shift"])
        .over('barrid')
        .alias(name)
    )
