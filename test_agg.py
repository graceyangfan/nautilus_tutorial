
from nautilus_trader.model.data.aggregate_extended_bar import ExtendedBarBuilder
from nautilus_trader.model.data.extended_bar import ExtendedBar
import argparse
from datetime import datetime
import pandas as pd
import os 
import glob 
import tqdm 
from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog
from nautilus_trader.adapters.binance.common.enums import BinanceAccountType
from nautilus_trader.adapters.binance.factories import get_cached_binance_http_client
from nautilus_trader.adapters.binance.futures.providers import BinanceFuturesInstrumentProvider
from nautilus_trader.adapters.binance.http.client import BinanceHttpClient
from nautilus_trader.backtest.data.wranglers import TradeTickDataWrangler
from nautilus_trader.common.clock import LiveClock, TestClock
from nautilus_trader.common.logging import Logger 
from nautilus_trader.data.aggregation import ValueBarAggregator
from nautilus_trader.model.data.bar import BarSpecification, BarType
from nautilus_trader.model.enums import AggressorSide, BarAggregation, PriceType
from nautilus_trader.persistence.external.core import write_objects
from nautilus_trader.model.instruments.currency_pair import CurrencyPair
from nautilus_trader.model.identifiers import Symbol
from nautilus_trader.model.identifiers import Venue
from nautilus_trader.model.currencies import BUSD
from nautilus_trader.model.currencies import ETH
from nautilus_trader.model.objects import Money
from tqdm import tqdm 

def ts_parser(time_in_secs: str) -> datetime:
    """Parses timestamp string into a datetime object."""
    return datetime.utcfromtimestamp(int(time_in_secs) / 1_000.0)


catalog = ParquetDataCatalog("catalog/.")

instrument_id = "BTCBUSD-PERP.BINANCE"
instrument = catalog.instruments(instrument_ids=[instrument_id],as_nautilus=True)[0]
filenames = glob.glob("example_data//*.csv")

## read data 
wrangler = TradeTickDataWrangler(instrument=instrument)
df = pd.DataFrame([])
filenames = glob.glob("example_data//*.csv")
for file in tqdm(filenames):
        df_in = pd.read_csv(
            file,
            index_col="timestamp",
            date_parser=ts_parser,
            parse_dates=True,
            skiprows=1,
            names = ["trade_id","price","quantity","quoteQty","timestamp","buyer_maker"],
            #compression='zip'
        )
        df = pd.concat([df,df_in],axis=0)
df["price"] = df["price"].astype(float)
df["quantity"] = df["quantity"].astype(float)
values = df["price"]*df["quantity"]

threshold = int(values.to_numpy().sum()/518400*6)
print(f"the threshold value  is {threshold}")

bar_spec = BarSpecification(threshold, BarAggregation.VALUE, PriceType.LAST)
bar_type = BarType(instrument.id, bar_spec)

handlers = []   
def on_bar(bar):                                            
    handlers.append(bar) 

bar_builder = ExtendedBarBuilder(
    instrument,
    bar_type,
    on_bar,
)
#df = df.to_parquet("example_data/ETHBUSD_trade.parquet")
ticks = wrangler.process(df)
for tick in tqdm(ticks):
    bar_builder.handle_trade_tick(tick)
write_objects(catalog, chunk = handlers)