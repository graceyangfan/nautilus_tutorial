import polars as pl 

def agg_timebar(bars,period):
    
    start_group = bars[0,"timestamp"]//(10**3*period)
    bars = bars.with_columns(
        [
            (pl.col("timestamp")//(10**3*period) - start_group).alias("group_id"),
        ]
    )
    bars = bars.with_columns(
        [
            (pl.col("group_id").shift(1).fill_null(strategy='backward')).alias("group_id")
        ]
    )
    print(bars["timestamp"])
    print(bars["group_id"])
    # aggregate bars 
    newbars = bars.groupby("group_id").agg(
        [
            (pl.col("timestamp").last()-pl.col("timestamp").last()%(10**3*period)).alias("ts_event"),
            pl.col("price").first().alias("open"),
            pl.col("price").max().alias("high"),
            pl.col("price").min().alias("low"),
            pl.col("price").last().alias("close"),
            pl.col("quantity").sum().alias("volume"),
        ]
    )
    newbars = newbars.sort("ts_event")
    # select columns
    newbars = newbars.select(
        [
            pl.col("open"),
            pl.col("high"),
            pl.col("low"),
            pl.col("close"),
            pl.col("volume"),
            pl.col("ts_event").alias("ts_init"),
            pl.col("ts_event"),
        ]
    )
    return newbars