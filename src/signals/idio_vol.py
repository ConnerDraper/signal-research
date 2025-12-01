"""idio volatility signal"""
import polars as pl

def compute_idio_vol(data: pl.DataFrame, name: str, config: dict) -> pl.DataFrame:
    # long low idio vol stocks, short high idio vol stocks
    window_size = config.get("window_size", 252)
    min_periods = config.get("min_periods", 252)
    idio_vol = (
        data.with_columns(
            (pl.col("specific_risk").rolling_mean(window_size=window_size, min_samples=min_periods))
            .mul(-1)
            .shift(2)
            .over('barrid')
            .alias('idio_vol')
        )
        .sort(['barrid', 'date'])
    )
    return idio_vol
