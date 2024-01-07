import polars as pl
import numpy as np 
from finml.labeling.get_label import create_label 



def cusum_filter(bars, threshold):
    '''
    return "count_index" select by time series 
    '''
    events_indexs = []
    s_pos = 0
    s_neg = 0
    # log returns
    returns = bars.select([pl.col("count_index"),pl.col("close").log().diff().alias("log_return")])
    # drop first nan 
    returns = returns[1:,:]
    # perform cusum 
    for index,log_return in returns.iter_rows():
        pos = float(s_pos + log_return)
        neg = float(s_neg + log_return)
        s_pos = max(0.0, pos)
        s_neg = min(0.0, neg)

        if s_neg < -threshold:
            s_neg = 0
            events_indexs.append(index)

        elif s_pos > threshold:
            s_pos = 0
            events_indexs.append(index)
    events_indexs = [item - bars[0,"count_index"] for item in events_indexs]
    return events_indexs


import polars as pl

def zscore_filter(
    bars,
    period,
    threshold,
    lambda_value,
    price_col_name="price",
    quantity_col_name="quantity",
    datetime_col_name="timestamp"
):
    """
    Filters financial bars based on z-score conditions and aggregates information for significant events.

    Parameters:
    - bars (polars.DataFrame): Input DataFrame containing financial bars.
    - period (int): The period for calculating z-score.
    - threshold (float): Threshold for z-score to trigger filtering.
    - lambda_value (float): Threshold for dollar value to identify significant events.
    - price_col_name (str): Column name for price data in the input DataFrame.
    - quantity_col_name (str): Column name for quantity data in the input DataFrame.
    - datetime_col_name (str): Column name for timestamp data in the input DataFrame.

    Returns:
    - polars.DataFrame: Resulting DataFrame with aggregated information for significant events.
    """

    # Calculate tick direction and fill nulls
    bars = bars.with_columns(
        [
            pl.arange(0, pl.count()).alias("count_index"),
            (pl.col(price_col_name).diff().sign().fill_null(1).alias("tick_direction"))
        ]
    )
    
    # Clean up tick direction nulls
    bars = bars.with_columns(
        pl.when(pl.col("tick_direction") == 0)
        .then(pl.lit(None))
        .otherwise(pl.col("tick_direction"))
        .fill_null(strategy="forward").alias("tick_direction")
    )
    
    # Calculate imbalance and dollar value
    bars = bars.with_columns([
        (pl.col(price_col_name) * pl.col(quantity_col_name) * pl.col("tick_direction")).alias("imbalance"),
        (pl.col(price_col_name) * pl.col(quantity_col_name)).alias("dollor_value"),
    ])
    
    # Calculate z-score of imbalance
    imbalance = bars.select(
        [
            pl.col("count_index"),
            ((pl.col("imbalance") - pl.col("imbalance").rolling_mean(period)) / pl.col("imbalance").rolling_std(period)).alias("zscore_imbalance")
        ]
    )
    
    # Shift the imbalance to align with the original data
    imbalance = imbalance[period - 1:]
    
    # Filter bars based on z-score thresholds
    imbalance = imbalance.filter(
        ((pl.col("zscore_imbalance") < threshold) & (pl.col("zscore_imbalance").shift() > threshold)) |
        ((pl.col("zscore_imbalance") > -threshold) & (pl.col("zscore_imbalance").shift() < -threshold))
    )
    
    # Add group_id to the original bars
    imbalance = imbalance.with_columns(
        pl.col("count_index").alias("group_id"),
    )
    bars = bars.join(imbalance, on="count_index", how="left")
    bars = bars.with_columns(
        [
            pl.col("group_id").fill_null(strategy="backward").alias("group_id")
        ]
    )
    
    # Drop null values
    bars = bars.filter(~pl.col("group_id").is_null())
    
    
    # Aggregate bars based on group_id
    newbars = bars.group_by("group_id").agg(
        [
            pl.col(datetime_col_name).first().alias("ts_init"),
            pl.col(datetime_col_name).last().alias("ts_event"),
            pl.col(price_col_name).first().alias("open"),
            pl.col(price_col_name).max().alias("high"),
            pl.col(price_col_name).min().alias("low"),
            pl.col(price_col_name).last().alias("close"),
            pl.col(quantity_col_name).sum().alias("volume"),
            pl.count().alias("group_length"),
            pl.col("dollor_value").filter((pl.col("tick_direction") > 0) & (pl.col("dollor_value") > lambda_value)).sum().alias("big_buy_dollor_sum"),
            pl.when((pl.col("tick_direction") > 0) & (pl.col("dollor_value") > lambda_value))
            .then(pl.col("dollor_value"))
            .otherwise(0.0)
            .std().alias("big_buy_dollor_std"),
            (pl.col("dollor_value")
             .filter((pl.col("tick_direction") > 0) & (pl.col("dollor_value") > lambda_value))
             .sum() - pl.col("dollor_value")
             .filter((pl.col("tick_direction") < 0) & (pl.col("dollor_value") > lambda_value))
             .sum()).alias("big_dif_sum"),
            (pl.when((pl.col("tick_direction") > 0) & (pl.col("dollor_value") > lambda_value))
            .then(pl.col("dollor_value"))
            .otherwise(0.0) - pl.when((pl.col("tick_direction") < 0) & (pl.col("dollor_value") > lambda_value))
            .then(pl.col("dollor_value"))
            .otherwise(0.0)).std().alias("big_dif_std")
        ]
    )
    
    # Sort bars based on event timestamp
    newbars = newbars.sort("ts_event")
    
    # Drop the first bar
    newbars = newbars[1:]
    
    # Select columns for the final result
    newbars = newbars.select(
        [
            pl.col("ts_init"),
            pl.col("ts_event"),
            pl.col("open"),
            pl.col("high"),
            pl.col("low"),
            pl.col("close"),
            pl.col("volume"),
            (pl.col("big_buy_dollor_sum") / pl.col("volume")).alias("big_buy_ratio"),
            (pl.col("big_dif_sum") / pl.col("volume")).alias("big_net_buy_ratio"),
            (pl.col("big_buy_dollor_sum") / pl.col("group_length") / (pl.col("big_buy_dollor_std")+1e-9)).alias("big_buy_power"),
            (pl.col("big_dif_sum") / pl.col("group_length") / (pl.col("big_dif_std")+1e-9)).alias("big_net_buy_power")
        ]
    )
    
    return newbars


