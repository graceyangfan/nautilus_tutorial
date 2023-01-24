
from finml.bar_aggregation.aggregate_extended_bar import ExtendedBarBuilder
from finml.bar_aggregation.extended_bar import *
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


def get_ETH_BUSD_instrument(exchange):
    return CurrencyPair(
        instrument_id=InstrumentId(
            symbol=Symbol("ETHBUSD"),
            venue=Venue(exchange),
        ),
        native_symbol=Symbol("ETHBUSD"),
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

if __name__ == "__main__":
    ## register extendBar 

    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', default="example_data/ETHBUSD*.zip")
    parser.add_argument('--symbol', default='ETHBUSD')
    parser.add_argument("--venue",default='BINANCE') 
    parser.add_argument('--threshold', type=int, default=10)
    parser.add_argument('--api_key_file', default = "config.json")
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
    df = pd.DataFrame([])
    filenames = sorted(glob.glob(args.filename))
    for file in tqdm(filenames):
        df_in = pd.read_csv(
            file,
            index_col="timestamp",
            date_parser=ts_parser,
            parse_dates=True,
            skiprows=1,
            names = ["trade_id","price","quantity","quoteQty","timestamp","buyer_maker"],
            compression='zip'
        )
        df = pd.concat([df,df_in],axis=0)
    df["price"] = df["price"].astype(float)
    df["quantity"] = df["quantity"].astype(float)
    values = df["price"]*df["quantity"]
    threshold = int(values.to_numpy().sum()/518400)
    print(f"the threshold value of {args.symbol} is {threshold}")


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
        bar_builder.apply_update(tick)

    write_objects(catalog, chunk = handlers)