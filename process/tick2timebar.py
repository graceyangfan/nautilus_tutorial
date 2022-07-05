## timeBar Aggregator
from datetime import datetime
from decimal import Decimal
from gettext import Catalog
import pandas as pd
import asyncio 
from nautilus_trader.backtest.data.providers import TestDataProvider
from nautilus_trader.backtest.data.loaders import CSVTickDataLoader
from nautilus_trader.backtest.data.wranglers import TradeTickDataWrangler
from nautilus_trader.common.clock import TestClock
from nautilus_trader.common.logging import Logger
from nautilus_trader.data.aggregation import TimeBarAggregator
from nautilus_trader.model.data.bar import Bar
from nautilus_trader.model.data.bar import BarSpecification
from nautilus_trader.model.data.bar import BarType
from nautilus_trader.model.data.tick import TradeTick
from nautilus_trader.model.enums import AggressorSide
from nautilus_trader.model.enums import BarAggregation
from nautilus_trader.model.enums import PriceType
from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog
from nautilus_trader.persistence.external.core import write_objects
from tests.test_kit.mocks.object_storer import ObjectStorer

from nautilus_trader.adapters.binance.common.enums import BinanceAccountType
from nautilus_trader.adapters.binance.factories import get_cached_binance_http_client
from nautilus_trader.adapters.binance.futures.providers import BinanceFuturesInstrumentProvider
from nautilus_trader.common.clock import LiveClock
from nautilus_trader.common.logging import Logger 
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.identifiers import Symbol
from nautilus_trader.model.identifiers import Venue

def _ts_parser(time_in_secs: str) -> datetime:
    return datetime.utcfromtimestamp(int(time_in_secs) / 1_000_000.0)


def tick2timebar(
        instrument,
        filename,
        bar_spec = BarSpecification(10, BarAggregation.SECOND, PriceType.LAST),
    ):

    clock = TestClock()
    bar_store = ObjectStorer()
    handler = bar_store.store
    bar_type = BarType(instrument.id, bar_spec)

    aggregator = TimeBarAggregator(
            instrument,
            bar_type,
            handler,
            clock,
            Logger(clock),
        )

    wrangler = TradeTickDataWrangler(instrument=instrument)
    ticks = wrangler.process(pd.read_csv(
            filename,
            index_col="timestamp",
            date_parser=_ts_parser,
            parse_dates=True,
        ))

    for tick in ticks:
        aggregator.handle_trade_tick(tick)
    events = clock.advance_time(ticks[-1].ts_event)
    for event in events:
        event.handle()
    bars = bar_store.get_store()   
    return bars 

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename',default="processedGALBUSD-trades-2022-06.csv")
    parser.add_argument('--symbol',default='APEBUSD')
    parser.add_argument("--venue",default='BINANCE') 
    args = parser.parse_args()
    catalog = ParquetDataCatalog("catalog")
    from get_all_instruments import write_all_instruments 
    import json 
    global data 
    with open('config.json') as f:
        data = json.load(f)
    instrument_id = InstrumentId(Symbol(f"{args.symbol}-PERP"),Venue("BINANCE"))
    instrument = asyncio.run(write_all_instruments(data["api_key"],data["secret_key"],instrument_id))  
    #
    bars = tick2timebar(instrument,args.filename)
    write_objects(catalog, chunk = bars)