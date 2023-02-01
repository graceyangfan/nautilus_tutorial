#!/usr/bin/env python
# Usage:
#  $ POLARS_MAX_THREADS=6 python test_sample.py

import polars as pl
import pandas as pd
import numpy as np
import glob
import os
#import matplotlib.pyplot as plt
import pyarrow as pa
from finml.labeling.get_label import create_label
from finml.sampling.get_weights import * 

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
    import os
    bar_type = "ETHBUSD-PERP.BINANCE-850000-VALUE-LAST-EXTERNAL"
    df = pl.read_parquet(os.path.join("train",bar_type+".parquet"))
    if '__index_level_0__' in df.columns:
        df = df.drop(['__index_level_0__'])
    labeled_df = create_label(
            df,
            threshold = 0.01,
            stop_loss = 0.005,
        )
    del df 
    total_indexs = [] 
    labeled_df_size = labeled_df.shape[0]
    for i in range(10):
        sub_df = labeled_df[int(labeled_df_size/10*i):int(labeled_df_size/10*(i+1)),:]
        np.random.seed(42)
        events = sub_df.select([pl.col("event_starts"),pl.col("event_ends")])
        bars = sub_df.select([pl.col("datetime"),pl.col("close")])
        sample_size = sub_df.shape[0]/10
        del sub_df 
        events_indicators = get_event_indicators(bars,events,njobs =1)
        sample_indexs = sample_sequential_bootstrap(events_indicators, size= sample_size)
        del events_indicators
        real_indexs = [item + int(labeled_df_size/10*i)  for item in sample_indexs]
        total_indexs.extend(real_indexs)

    
    import pickle
    with open('sample_indexs.pkl', 'wb') as f:
        pickle.dump(total_indexs, f)


