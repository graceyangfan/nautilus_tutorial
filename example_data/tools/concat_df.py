import polars as pl 
import pandas as pd 
import glob 
from datetime import datetime 
from tqdm import tqdm

metrics_columns = ['create_time', 'symbol', 'sum_open_interest', 'sum_open_interest_value',
       'count_toptrader_long_short_ratio', 'sum_toptrader_long_short_ratio',
       'count_long_short_ratio', 'sum_taker_long_short_vol_ratio']


kline_columns = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time',
    'quote_volume', 'count', 'taker_buy_volume', 'taker_buy_quote_volume',
    'ignore']

def concat_file(filename_str,columns):
    df = pl.DataFrame({})
    filenames = sorted(glob.glob(filename_str))
    for file in tqdm(filenames):
        df_in = pd.read_csv(file,compression="zip")
        df_in = pl.from_pandas(df_in)
        df_in.columns = columns
        df = pl.concat([df,df_in], how="vertical")
    #convert into timestamp(nano second)
    if "create_time" in df.columns:
        df = df.with_columns([pl.col("create_time").str.strptime(pl.Datetime, None, strict=False).dt.timestamp("ns").alias("create_time")])
        df = df.with_columns([pl.col("create_time"),pl.all().exclude(["create_time","symbol"]).shift(1)])
        df = df.filter(pl.col("create_time")>1e9*datetime(2022,6,1).timestamp())
        #df = df.drop_nulls()
    
    if "open_time" in df.columns:
        df = df.select([pl.all().exclude(["close_time","ignore"])])
        df = df.with_columns([(pl.col("open_time")*1e6).cast(pl.Int64)])
        df = df.filter(pl.col("open_time")>1e9*datetime(2022,6,1).timestamp())
    return df 

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', default='BTCUSDT')
    parser.add_argument("--venue",default='BINANCE') 
    parser.add_argument('--startDate', default='2022-01-01')
    parser.add_argument('--endDate', default='2023-09-01')
    parser.add_argument('--store_dir', default='../example_data/')
    parser.add_argument('--interval', default='5m')
    args = parser.parse_args()
    metrics_str = f"{args.store_dir}/data/futures/um/daily/metrics/{args.symbol}/{args.startDate}_{args.endDate}/*zip"
    klines_str = f"{args.store_dir}/data/futures/um/monthly/klines/{args.symbol}/{args.interval}/{args.startDate}_{args.endDate}/*zip"

    save_path = f"{args.symbol}-PERP.{args.venue}-{args.interval}-MINUTE-LAST-EXTERNAL.parquet"
    df_metrics = concat_file(metrics_str,metrics_columns)
    df_kline = concat_file(klines_str,kline_columns)
    df = df_kline.join(df_metrics,left_on="open_time",right_on="create_time",how="left")
    df = df.with_columns(pl.col(['sum_open_interest', 'sum_open_interest_value',
       'count_toptrader_long_short_ratio', 'sum_toptrader_long_short_ratio',
       'count_long_short_ratio', 'sum_taker_long_short_vol_ratio']).interpolate())
    df = df.with_columns(pl.col(['open', 'high', 'low', 'close', 'volume', 
    'quote_volume', 'taker_buy_volume', 'taker_buy_quote_volume', 
    'sum_open_interest', 'sum_open_interest_value', 'count_toptrader_long_short_ratio', 
    'sum_toptrader_long_short_ratio', 'count_long_short_ratio', 'sum_taker_long_short_vol_ratio']).cast(pl.Float64))
    df = df.with_columns([
        pl.col('open_time').cast(pl.Int64),
        pl.col('count').cast(pl.Int64),
        pl.col("symbol").cast(pl.Utf8),
    ])
    if len(df) >=113855:
        df.write_parquet(save_path)


