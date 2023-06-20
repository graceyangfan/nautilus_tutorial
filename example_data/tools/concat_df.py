import polars as pl 
import pandas as pd 
import glob 
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
    if "time" in df.columns[0]:
        df = df.with_columns([
            pl.col(df.columns[0]).str.strptime(pl.Datetime, fmt=None, strict=False),
        ])
    return df 

if __name__ == "__main__":
    # filename_str = "../example_data/data/futures/um/daily/metrics/1000PEPEUSDT/2023-05-01_2023-06-17/*"
    # save_path = "../PEPEUSDT_metrics.parquet"

    # df = concat_file(filename_str,metrics_columns)
    # df.write_parquet(save_path)

    filename_str = "../example_data/data/futures/um/daily/klines/1000PEPEUSDT/5m/2023-05-01_2023-06-17/*"
    save_path = "../PEPEUSDT_kline.parquet"

    df = concat_file(filename_str,kline_columns)
    df.write_parquet(save_path)