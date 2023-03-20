import numpy as np
import os 
import glob 
import tqdm 
import pandas as pd 
import polars as pl 
from finml.labeling.get_label import create_label 

from sklearn.linear_model import LinearRegression 
import pyarrow as pa 

bar_schema   = pa.schema(
        {
            "bar_type": pa.dictionary(pa.int8(), pa.string()),
            "instrument_id": pa.dictionary(pa.int64(), pa.string()),
            "open": pa.float64(),
            "high": pa.float64(),
            "low": pa.float64(),
            "close": pa.float64(),
            "volume": pa.float64(),
            "ts_event": pa.int64(),
            "ts_init": pa.int64(),
        }
)


if __name__ == "__main__":
    filename1 = "catalog/data/bar.parquet/instrument_id=BTCUSDT-PERP.BINANCE/1654041610000000000-1678406390000000000-BINANCE-10-SECOND-LAST-EXTERNAL-0.parquet"
    filename2 = "catalog/data/bar.parquet/instrument_id=OPUSDT-PERP.BINANCE/1654092080000000000-1678406390000000000-BINANCE-10-SECOND-LAST-EXTERNAL-0.parquet"
    df1 = pl.read_parquet(filename1,use_pyarrow = True,pyarrow_options = {"schema":bar_schema})
    df2 = pl.read_parquet(filename2,use_pyarrow = True,pyarrow_options = {"schema":bar_schema})
    df2 = create_label(
        df2.with_columns([pl.col("ts_event").alias("datetime")]),
        0.02,
        0.001,
        False 
    )
    df =  df2.join(
        df1.select([pl.col("ts_event"),pl.col("close").alias("base_close")]),
        on = "ts_event", how = "left")

    results = pd.DataFrame()
    #for period in np.arange(6,300,30):
    period = 130 
    df = df.with_columns(
        [
            pl.col("close").pct_change(period).alias("pct_change_{}".format(period)),
            pl.col("base_close").pct_change(period).alias("base_pct_change_{}".format(period)),
        ]
    )
    df = df.filter(~pl.col("pct_change_{}".format(period)).is_null())
    x = df[:,"base_pct_change_{}".format(period)].to_numpy().reshape(-1,1)
    y = df[:,"pct_change_{}".format(period)].to_numpy().reshape(-1,1) 
    model = LinearRegression().fit(x,y) 
    residual = y - model.predict(x) 
    # zscore residual 
    residual = abs((residual - residual.mean())/residual.std())

    residual = pl.DataFrame({"residual":residual.reshape(-1)})
    df = pl.concat([df,residual],how='horizontal')
    for value in np.arange(0.5,10,5):
        sub_df = df.filter(pl.col("residual")>value)
        corr = df.select(pl.corr(f"base_pct_change_{period}","label",method = "spearman"))[0,f"base_pct_change_{period}"]
        print(f"the correction {value} of residuals and label is {corr}")
    # select the best residal for select open point! 



