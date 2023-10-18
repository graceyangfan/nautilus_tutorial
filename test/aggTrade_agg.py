from decimal import Decimal
import argparse
from datetime import datetime
import polars as pl 
import numpy as np
import pandas as pd 
import os 
import glob 
from tqdm import tqdm
from joblib import Parallel,delayed 
from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog
from nautilus_trader.adapters.binance.common.enums import BinanceAccountType
from nautilus_trader.adapters.binance.factories import get_cached_binance_http_client
from nautilus_trader.adapters.binance.futures.providers import BinanceFuturesInstrumentProvider
from nautilus_trader.adapters.binance.http.client import BinanceHttpClient
from nautilus_trader.common.clock import LiveClock, TestClock
from nautilus_trader.common.logging import Logger 
from nautilus_trader.model.data.bar import BarSpecification, BarType
from nautilus_trader.model.enums import AggressorSide, BarAggregation, PriceType
from nautilus_trader.persistence.external.core import write_objects
from nautilus_trader.model.instruments.currency_pair import CurrencyPair
from nautilus_trader.model.identifiers import Symbol
from nautilus_trader.model.identifiers import Venue
from nautilus_trader.model.currencies import BUSD,USDT
from nautilus_trader.model.currencies import ETH
from nautilus_trader.model.objects import Money
from nautilus_trader.model.objects import Price, Quantity
from nautilus_trader.model.identifiers import InstrumentId

#from nautilus_trader.model.data.aggregate_extended_bar import ExtendedBarBuilder
#from nautilus_trader.model.data.extended_bar import ExtendedBar
#import pyarrow as pa
#from nautilus_trader.serialization.base import register_serializable_object
#from nautilus_trader.serialization.arrow.serializer import register_parquet
from finml.labeling.get_label import create_label
from finml.sampling.filter import zscore_filter



def ts_parser(time_in_secs: str) -> datetime:
    """Parses timestamp string into a datetime object."""
    return datetime.utcfromtimestamp(int(time_in_secs) / 1_000.0)


def get_ETH_BUSD_instrument(exchange):
    return CurrencyPair(
        instrument_id=InstrumentId(
            symbol=Symbol("ETHBUSD-PERP"),
            venue=Venue(exchange),
        ),
        native_symbol=Symbol("ETHBUSD-PERP"),
        base_currency=ETH,
        quote_currency=BUSD,
        price_precision=2,
        size_precision=3,
        price_increment=Price(1e-02, precision=2),
        size_increment=Quantity(1e-3, precision=3),
        lot_size=None,
        max_quantity=Quantity(9000, precision=3),
        min_quantity=Quantity(1e-03, precision=3),
        max_notional=None,
        min_notional=Money(1.00, BUSD),
        max_price=Price(1000000, precision=2),
        min_price=Price(0.01, precision=2),
        margin_init=Decimal("0"),
        margin_maint=Decimal("0"),
        maker_fee=Decimal("0.0004"),
        taker_fee=Decimal("0.0004"),
        ts_event=0,
        ts_init=0,
        )

def test_imbalance_corr(df):
    df = df.select(
        [pl.col("buyer_maker_imbalance"),pl.col("label")]
    )
    df_train = df[:int(0.5*df.shape[0])]
    df_test = df[int(0.5*df.shape[0]):]
    IS_Corr = df_train.select(pl.corr("buyer_maker_imbalance","label",method = 'spearman'))[0,0]
    OS_Corr = df_test.select(pl.corr("buyer_maker_imbalance","label",method = 'spearman'))[0,0]
    return (IS_Corr,OS_Corr)

def select_params(df,njobs=1):
    def compute_value(params):
        newbars = zscore_filter(params["df"],params["period"].item(),params["threshold"],params["value"])
        if len(newbars) < 1000:
            return (1,-1)
        labeled_df =create_label(
            newbars.select([
            pl.col("ts_event").alias("datetime"),
            pl.col("close"),
            pl.col("high"),
            pl.col("low"),
            pl.col("buyer_maker_imbalance")]),
            0.02,
            False
        )

        labeled_df = labeled_df.with_columns([
            (pl.when(pl.col("buyer_maker_imbalance")>pl.col("buyer_maker_imbalance").mean() +5.0*pl.col("buyer_maker_imbalance").std())
             .then(pl.col("buyer_maker_imbalance").mean() +5.0*pl.col("buyer_maker_imbalance").std())
             .otherwise(pl.col("buyer_maker_imbalance"))).alias("buyer_maker_imbalance"),
        ])
        labeled_df = labeled_df.with_columns([
            (pl.when(pl.col("buyer_maker_imbalance")< pl.col("buyer_maker_imbalance").mean() - 5.0*pl.col("buyer_maker_imbalance").std())
             .then(pl.col("buyer_maker_imbalance").mean() - 5.0*pl.col("buyer_maker_imbalance").std())
             .otherwise(pl.col("buyer_maker_imbalance"))).alias("buyer_maker_imbalance"),
        ])
        
        #normal 
        labeled_df = labeled_df.with_columns([
            ((pl.col("buyer_maker_imbalance")-pl.col("buyer_maker_imbalance").mean())/pl.col("buyer_maker_imbalance").std()).alias("buyer_maker_imbalance"),
            ((pl.col("label")-pl.col("label").mean())/pl.col("label").std()).alias("label")
        ])
    
        
        in_sample_corr,out_sample_corr = test_imbalance_corr(labeled_df)  
        if np.isnan(in_sample_corr) or np.isnan(out_sample_corr):
            return (1,-1)
        if np.sign(in_sample_corr)!=np.sign(out_sample_corr):
            return (1,-1)
        timeduring = newbars.select(pl.col("ts_event")-pl.col("ts_init")).median().row(0)[0] 
        return in_sample_corr,out_sample_corr,timeduring,{"period":params["period"].item(),"threshold":params["threshold"],"value":params["value"]}
    if isinstance(df,pd.DataFrame):
        df = pl.from_pandas(df)
    values = df.select(pl.col("price")*pl.col("quantity"))
    params=[]
    for period in np.arange(50,150,30):
        for threshold in np.arange(7,15,1):
            for value in np.arange(10000,11000,10000):
                params.append({
                    "df":df,
                    "period":period,
                    "threshold":threshold,
                    "value":value
                })
    results = Parallel(n_jobs=njobs)(delayed(compute_value)(param) for param in tqdm(params))
    params = [] 
    in_sample_corr_list = [] 
    out_sample_corr_list = [] 
    timedurings = [] 
    for idx,item in enumerate(results):
        if np.sign(item[0]) == np.sign(item[1]):
            params.append(item[-1])
            in_sample_corr_list.append(item[0])
            out_sample_corr_list.append(item[1])
            timedurings.append(item[2])
    results = pd.DataFrame({
        "params":params,
        "in_sample_corr":in_sample_corr_list,
        "out_sample_corr":out_sample_corr_list,
        "timedurings":timedurings,
    })
    results = results.sort_values("out_sample_corr",ascending=False)
    return results



if __name__ == "__main__":
    ## register extendBar 
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', default="*.zip")
    parser.add_argument('--symbol', default='MATICUSDT')
    parser.add_argument("--venue",default='BINANCE') 
    parser.add_argument('--threshold', type=int, default=10)
    parser.add_argument('--api_key_file', default = "config.json")
    parser.add_argument('--account_type', default=BinanceAccountType.FUTURES_USDT)
    args = parser.parse_args()

    catalog = ParquetDataCatalog("/notebooks/nautilus_trader-develop/examples/catalog")

    instrument_id = f"{args.symbol}-PERP.{args.venue}"
    # Get the instrument for the given symbol
    #instrument = asyncio.run(get_instrument(instrument_id , api_key, api_secret, args.account_type))
    if args.symbol == 'ETHBUSD':
        instrument = get_ETH_BUSD_instrument(args.venue)
    else:
        instrument = catalog.instruments(instrument_ids=[instrument_id],as_nautilus=True)[0]


    ## read data 
    if os.path.exists(os.path.join("/tmp",instrument_id+".parquet")):
        df = pl.read_parquet(os.path.join("/tmp",instrument_id+".parquet"))
        if "buyer_maker" not in df.columns:
            df= df.with_columns(pl.col("is_buyer_maker").alias("buyer_maker"))
        if "timestamp" not in df.columns:
            df= df.with_columns(pl.col("transact_time").alias("timestamp"))
    else:
        df = pd.DataFrame({})
        filenames = sorted(glob.glob(args.filename))
        for file in tqdm(filenames):
            df_in = pd.read_csv(file,compression="zip")
            #df_in.colmuns = ['agg_trade_id', 'price', 'quantity', 'first_trade_id', 'last_trade_id',
       #'transact_time', 'is_buyer_maker']
            df = pd.concat([df,df_in])

        df.to_parquet(os.path.join("/tmp",instrument_id+".parquet"))
    results = select_params(df,njobs=1)
    results.to_csv("select.csv")