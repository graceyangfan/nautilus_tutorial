from nautilus_trader.indicators.zigzag import Zigzag 
import polars as pl 
def batch_zizag(
    _df,
    threshold = 0.02,
    only_close = False,
):
    zigzag = Zigzag(threshold,only_close)
    datetimes = [] 
    zigzag_values = [] 
    for j in range(_df.shape[0]):
        if only_close:
            zigzag.update_raw(_df[j,"open"],_df[j,"close"],_df[j,"close"],_df[j,"close"],_df[j,"volume"],_df[j,"ts_event"])
        else:
            zigzag.update_raw(_df[j,"open"],_df[j,"high"],_df[j,"low"],_df[j,"close"],_df[j,"volume"],_df[j,"ts_event"])

        if zigzag.initialized:
            datetime = zigzag.zigzags_datetime[-2] 
            value = zigzag.zigzags_values[-2]
            if len(datetimes) < 1 or datetime!=datetimes[-1]:
                datetimes.append(datetime)
                zigzag_values.append(value)
    zigzag_points = pl.DataFrame({
        "ts_event":datetimes,
        "zigzag_points":zigzag_values,
    })

    return zigzag_points 