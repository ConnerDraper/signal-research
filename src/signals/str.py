"""short-term reversal signal"""
import polars as pl

def compute_str(data: pl.DataFrame, name: str, config: dict) -> pl.DataFrame:
    # get log return
    data = (
        data.with_columns(
            pl.col('return')
            .log1p()
            .alias('log_return')
        )
    )

    signal_config = config.get("signal", {})
    period = signal_config.get("period", 22)

    # get scores
    data = (
        data.with_columns(
            pl.col('log_return')
            .rolling_sum(window_size=period)
            .over('barrid')
            .mul(-1)
            .alias('str')
        )
    ).sort(['barrid', 'date'])

    return data