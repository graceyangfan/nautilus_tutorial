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


def zscore_filter(bars,period,threshold,lambda_value):
    bars = bars.with_columns(
        [
            pl.arange(0, pl.count()).alias("count_index"),
            (pl.col("price").diff().sign().fill_null(1).alias("tick_direction"))
        ]
    )
    bars =  bars.with_columns(
        pl.when(pl.col("tick_direction")==0)
        .then(pl.lit(None))
        .otherwise(pl.col("tick_direction"))
        .fill_null(strategy="forward").alias("tick_direction")
    )
    bars = bars.with_columns(
        (pl.col("price").pct_change()*pl.col("quantity")*pl.col("tick_direction")).alias("imbalance")
    )
    imbalance = bars.select(
        [
            pl.col("count_index"),
            ((pl.col("imbalance") - pl.col("imbalance").rolling_mean(period))/pl.col("imbalance").rolling_std(period)).alias("zscore_imbalance")
        ]
    )
    imbalance = imbalance[period-1:]
     
    imbalance = imbalance.filter(pl.col("zscore_imbalance").abs() > threshold)
    imbalance = imbalance.with_columns(
        pl.col("count_index").alias("group_id"),
    )
    bars = bars.join(imbalance, on = "count_index",  how = "left")
    bars = bars.with_columns(
        [
            pl.col("group_id").fill_null(strategy = "backward").alias("group_id")
        ]
    )
    # drop null 
    bars = bars.filter(~pl.col("group_id").is_null())
    print(bars.select(
    [
        pl.col("price"),
        pl.col("quantity"),
        pl.col("buyer_maker")
    ]))
    # aggregate bars 
    newbars = bars.groupby("group_id").agg(
        [
            pl.col("timestamp").first().alias("ts_init"), 
            pl.col("timestamp").last().alias("ts_event"),
            pl.col("price").first().alias("open"),
            pl.col("price").max().alias("high"),
            pl.col("price").min().alias("low"),
            pl.col("price").last().alias("close"),
            pl.col("quantity").sum().alias("volume"),
            (
                (pl.col("price")*pl.col("quantity")).filter(
                        (pl.col("buyer_maker")==False)&(pl.col("price")*pl.col("quantity")<lambda_value)
                )
            ).sum().alias("small_buy_value"),
            (
                (pl.col("price")*pl.col("quantity")).filter(
                        (pl.col("buyer_maker")==False)&(pl.col("price")*pl.col("quantity")>lambda_value)
                )
            ).sum().alias("big_buy_value"),
            (
                (pl.col("price")*pl.col("quantity")).filter(
                        (pl.col("buyer_maker")==True)&(pl.col("price")*pl.col("quantity")<lambda_value)
                )
            ).sum().alias("small_sell_value"),
            (
                (pl.col("price")*pl.col("quantity")).filter(
                        (pl.col("buyer_maker")==True)&(pl.col("price")*pl.col("quantity")>lambda_value)
                )
            ).sum().alias("big_sell_value"),
        ]
    )
    #fill null for volume aggregate 
    newbars = newbars.with_columns(
        [
            pl.col("small_buy_value").fill_null(0.0).alias("small_buy_value"),
            pl.col("big_buy_value").fill_null(0.0).alias("big_buy_value"),
            pl.col("small_sell_value").fill_null(0.0).alias("small_sell_value"),
            pl.col("big_sell_value").fill_null(0.0).alias("big_sell_value"),
        ]
    )
    newbars = newbars.sort("ts_event")
    # drop the first bar 
    newbars = newbars[1:]
    # select columns
    newbars = newbars.select(
        [
            pl.col("open"),
            pl.col("high"),
            pl.col("low"),
            pl.col("close"),
            pl.col("volume"),
            pl.col("small_buy_value"),
            pl.col("big_buy_value"),
            pl.col("small_sell_value"),
            pl.col("big_sell_value"),
            pl.col("ts_init"),
            pl.col("ts_event"),
            (pl.col("big_buy_value") - pl.col("big_sell_value")).alias("buyer_maker_imbalance"),
        ]
    )
    return newbars

