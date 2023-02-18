from decimal import Decimal
import argparse
from datetime import datetime
import pandas as pd
import polars as pl 
import os 
import glob 
from tqdm import tqdm
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
from nautilus_trader.model.objects import Price, Quantity
from nautilus_trader.model.identifiers import InstrumentId

from nautilus_trader.model.data.aggregate_imbalance_bar import ImbalanceBarBuilder
from nautilus_trader.model.data.imbalance_bar import ImbalanceBar
import pyarrow as pa
from nautilus_trader.serialization.base import register_serializable_object
from nautilus_trader.serialization.arrow.serializer import register_parquet

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

ImbalanceBar_SCHEMA  = pa.schema(
        {
            "bar_type": pa.dictionary(pa.int8(), pa.string()),
            #"instrument_id": pa.dictionary(pa.int64(), pa.string()),
            "open": pa.string(),
            "high": pa.string(),
            "low": pa.string(),
            "close": pa.string(),
            "volume": pa.string(),
            "small_buy_value": pa.float64(),
            "big_buy_value": pa.float64(),
            "small_sell_value": pa.float64(),
            "big_sell_value": pa.float64(),
            "ts_event": pa.uint64(),
            "ts_init": pa.uint64(),
        },
    metadata={"type": "ImbalanceBar"},
)


if __name__ == "__main__":
    ## register extendBar 
    #register_serializable_object(ImbalanceBar, ImbalanceBar.to_dict, ImbalanceBar.from_dict)
    register_parquet(
        cls=ImbalanceBar, 
        serializer=ImbalanceBar.to_dict,
        deserializer= ImbalanceBar.from_dict, 
        schema=ImbalanceBar_SCHEMA
    )
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', default="/tmp/ETHBUSD*.csv")
    parser.add_argument('--symbol', default='ETHBUSD')
    parser.add_argument("--venue",default='BINANCE') 
    parser.add_argument('--period', type=int, default=170)
    parser.add_argument('--threshold', type=float, default=11.5)
    parser.add_argument('--imbalance_value', type=float, default=250)
    parser.add_argument('--account_type', default=BinanceAccountType.FUTURES_USDT)
    args = parser.parse_args()

    catalog = ParquetDataCatalog("catalog/.")

    instrument_id = f"{args.symbol}-PERP.{args.venue}"
    # Get the instrument for the given symbol
    #instrument = asyncio.run(get_instrument(instrument_id , api_key, api_secret, args.account_type))
    if args.symbol == 'ETHBUSD':
        instrument = get_ETH_BUSD_instrument(args.venue)
    else:
        instrument = catalog.instruments(instrument_ids=[instrument_id],as_nautilus=True)[0]


    ## read data 
    wrangler = TradeTickDataWrangler(instrument=instrument)
    if os.path.exists(os.path.join("example_data",instrument_id+".parquet")):
        df = pl.read_parquet(os.path.join("example_data",instrument_id+".parquet"))
    else:
        df = pl.DataFrame({})
        filenames = sorted(glob.glob(args.filename))
        for file in tqdm(filenames):
            df_in = pl.read_csv(file)
            df_in.columns = ["trade_id","price","quantity","quoteQty","timestamp","buyer_maker"]

            df = pl.concat([df,df_in], how="vertical")

    # aggregate bars 
    newbars = zscore_filter(df,args.period,args.threshold,args.imbalance_value)
    handlers = [] 
    for i in range(newbars.sahpe[0]):
        bar = ImbalanceBar(

        )
        handlers.append(bar) 
    write_objects(catalog, chunk = handlers)
    write_objects(catalog, chunk = [instrument])