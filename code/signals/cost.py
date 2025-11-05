"""cost signal"""
import polars as pl

def compute_cost(data: pl.DataFrame, name: str, config: dict) -> pl.DataFrame:
    return data.with_columns(
        (pl.col("bid_ask_spread") / pl.col('price'))
        .rolling_mean(window_size=config["window_size"])
        .truediv(pl.col("return").rolling_std(window_size=44))
        .shift(config["shift"])
        .over('barrid')
        .alias(name)
    )
