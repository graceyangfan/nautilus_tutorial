import pyarrow as pa 
import polars as pl 

allow_compute_dtypes  = [
        pl.Int8,
        pl.Int16,
        pl.Int32,
        pl.Int64,
        pl.UInt8,
        pl.UInt16,
        pl.UInt32,
        pl.UInt64,
        pl.Float32,
        pl.Float64,
    ]

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

schema_dict = {
    "Bar":bar_schema,
}