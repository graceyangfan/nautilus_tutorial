import polars as pl
import pandas as pd 
from sklearn.model_selection import train_test_split
from tuneta.tune_ta import TuneTA
import os 
import glob 
import pyarrow as pa
from finml.labeling.get_label import create_label

def tuneta_select(
    df,
    indictaors,
    range_start = 30,
    range_end = 500,
    trials = 30 ,
    early_stop = 10,
    prune_threshold = 0.7,
    num_workers = 200,
):
    if isinstance(df,pl.DataFrame):
        df = df.to_pandas() 
    df["Date"] = df["datetime"].apply(lambda x: pd.Timestamp(x, unit="ns"))
    df = df.set_index("Date",drop=True)
    df = df.loc[df.index.unique()]
    print(df.columns)
    X = df[["open","high","low","close","volume"]]
    y = df["label"]
    num_workers = int(os.cpu_count ()/2.0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, shuffle=False)
    # Initialize with x cores and show trial results
    tt = TuneTA(n_jobs=num_workers, verbose=True)
    tt.fit(X_train, y_train,
        indicators=indictaors,
        ranges = [(range_start,range_end)],
        trials = trials,
        early_stop = early_stop,
    )

    # Show time duration in seconds per indicator
    tt.fit_times()

    # Show correlation of indicators to target
    tt.report(target_corr=True, features_corr=True)

    # Select features with at most x correlation between each other
    tt.prune(max_inter_correlation=prune_threshold)

    # Show correlation of indicators to target and among themselves
    tt.report(target_corr=True, features_corr=True)

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
    filenames = glob.glob("catalog/data/extended_bar.parquet/*.parquet")
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
    print(labeled_df[:,"label"].max(),labeled_df[:,"label"].min())
    labeled_df = labeled_df.to_pandas()
    df = df.to_pandas()
    min_datetime = max(labeled_df.datetime.iloc[0],df.datetime.iloc[0])
    max_datetime = min(labeled_df.datetime.iloc[-1],df.datetime.iloc[-1])
    df = df[(df.datetime >= min_datetime)&(df.datetime <= max_datetime)]
    df = df.merge(labeled_df.drop(columns=["close"]),left_on="datetime",right_on="datetime",how="right")
    tuneta_select(
        df,
        #indictaors = ['pta'],
        indictaors = ['pta.zscore',
                      'pta.cfo',
                      'pta.willr',
                      'pta.bias',
                      'pta.dpo',
                      'pta.qstick',
                      'pta.slope',
                      'pta.mom',
                      'pta.fisher',
                      'pta.rsi',
                      'pta.cmo',
                      'pta.uo',
                      'pta.cci',
                      'pta.cti',
                      'pta.efi',
                      'pta.pgo',
                      'pta.eri',
                      'pta.rvi',
                      'pta.rsx',
                      'pta.cg',
                      'pta.brar',
                      'pta.kvo',
                      'pta.vortex',
                      'pta.ttm',
                      'pta.mfi',
                      'pta.bop',
                      'pta.pvol',
                      'pta.stoch',
                      'pta.roc',
                      'pta.entropy',
                      'pta.cmf',
                      'pta.decreasing'
                     ],
        range_start = 30,
        range_end = 500,
        trials = 30 ,
        early_stop = 10,
        prune_threshold = 0.7,
    )
    