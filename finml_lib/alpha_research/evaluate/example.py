from factor_preprocessing import preprocessing_extreme, preprocessing_scale
from single_factor_analysis import multi_period_ic_analysis
import polars as pl 
import numpy as np 

if __name__ == "__main__":
    filename = "../example_data/data.parquet"
    df = pl.read_parquet(filename)
    df = df.with_columns(
        [
            pl.col("date").alias("datetime"),
            pl.col("asset").alias("symbol")
        ]
    )
    df = df.sort(["symbol", "datetime"])
    for i in np.arange(3,20):
        df = df.with_columns(
            (pl.col("close").shift(-i)/pl.col("close")-1.0).over("symbol").alias(f"label_{i}")
        )
    df = df.drop_nulls()
    df = df.sort(["symbol", "datetime"])
    df = df.with_columns(
        pl.int_range(pl.len()).alias("index")
    )
    df = df.filter((pl.col("symbol")!=pl.lit("s_0091")) & (pl.col("symbol")!=pl.lit("s_0465")))
    #print(df.columns)
    factor_name = "close"
    df = df.group_by("symbol").map_groups(
        lambda x: preprocessing_extreme(
            x,
            method = "three_sigma",
            process_columns = [factor_name],
            n = 1
        )
    )
    #print(df.select(pl.all().is_null().sum()))
    #w = df.group_by("symbol").agg([pl.col("close_low").min(),pl.col("close_high").max()])
    #print(w.filter((pl.col("close_low")-pl.col("close_high")).abs()<1e-6))
    df = df.group_by("symbol").map_groups(
        lambda x: preprocessing_scale(
            x,
            method = "zscore",
            process_columns = [factor_name+"_clip"]
        )
    )
    #print(df.select(pl.all().is_null().sum()))
    ic_df = multi_period_ic_analysis(
        df = df,
        analysis_column = factor_name+"_clip_zscore",
        label_column_prefix = "label", 
        method = "IC",
        abs_threshold = 0.02,
        p_value_threshold = 0.05,
        start_period = 3,
        decay_period = 20 
    )
   
    print(ic_df)