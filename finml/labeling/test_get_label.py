import polars as pl
import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
from get_label import * 
import pyarrow as pa

ExtendedBar_SCHEMA  = pa.schema(
        {
            "bar_type": pa.dictionary(pa.int8(), pa.string()),
            #"instrument_id": pa.dictionary(pa.int64(), pa.string()),
            "open": pa.float64(),
            "high": pa.float64(),
            "low": pa.float64(),
            "close": pa.float64(),
            "volume": pa.float64(),
            "bids_value_level_0": pa.float64(),
            "bids_value_level_1": pa.float64(),
            "bids_value_level_2": pa.float64(),
            "bids_value_level_3": pa.float64(),
            "bids_value_level_4": pa.float64(),
            "asks_value_level_0": pa.float64(),
            "asks_value_level_1": pa.float64(),
            "asks_value_level_2": pa.float64(),
            "asks_value_level_3": pa.float64(),
            "asks_value_level_4": pa.float64(),
            "ts_event": pa.int64(),
            "ts_init": pa.int64(),
        },
    metadata={"type": "ExtendedBar"},
)

if __name__ == "__main__":
    filenames = glob.glob("../../catalog/data/genericdata_extended_bar.parquet/*.parquet")
    df = pl.read_parquet(
        filenames[0],
        use_pyarrow=True,
        pyarrow_options={"schema": ExtendedBar_SCHEMA}
    )
    df = df.select(
    [pl.all(),pl.col("ts_event").alias("datetime")]
    )
    labeled_df = create_label(
        df,
        threshold = 0.01,
        stop_loss = 0.005,
    )
