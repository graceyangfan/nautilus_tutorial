import asyncio
import argparse
from datetime import datetime
import pandas as pd
import os 

from nautilus_trader.adapters.binance.common.enums import BinanceAccountType
from nautilus_trader.adapters.binance.factories import get_cached_binance_http_client
from nautilus_trader.adapters.binance.futures.providers import BinanceFuturesInstrumentProvider
from nautilus_trader.adapters.binance.http.client import BinanceHttpClient
from nautilus_trader.backtest.data.wranglers import TradeTickDataWrangler
from nautilus_trader.common.clock import LiveClock, TestClock
from nautilus_trader.common.logging import Logger 
from nautilus_trader.data.aggregation import VolumeBarAggregator
from nautilus_trader.model.data.bar import BarSpecification, BarType
from nautilus_trader.model.enums import AggressorSide, BarAggregation, PriceType
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog
from nautilus_trader.persistence.external.core import write_objects


async def get_instrument(
    sym: str, api_key: str, api_secret: str, account_type: BinanceAccountType
) -> InstrumentId:
    """Asynchronously connects to the Binance API and retrieves an instrument object for a given symbol."""
    clock = LiveClock()
    client = get_cached_binance_http_client(
        loop=asyncio.get_event_loop(),
        clock=clock,
        logger=Logger(clock=clock),
        account_type=account_type,
        key=api_key,
        secret=api_secret,
        is_testnet=True,
    )

    symbol = InstrumentId.from_str(sym)

    try:
        await client.connect()
    except Exception as e:
        print(f"Failed to connect to Binance API: {e}")
        return None

    try:
        provider = BinanceFuturesInstrumentProvider(
            client=client,
            logger=Logger(clock=clock),
            account_type=account_type,
        )

        await provider.load_all_async()

        instrument = provider.find(instrument_id=symbol)
    except Exception as e:
        print(f"Failed to retrieve instrument: {e}")
        instrument = None
    finally:
        await client.disconnect()

    return instrument


def ts_parser(time_in_secs: str) -> datetime:
    """Parses timestamp string into a datetime object."""
    return datetime.utcfromtimestamp(int(time_in_secs) / 1_000.0)


def aggregate_volumebar(instrument, filename, threshold):
    """Aggregates trade tick data into volume bars of a specified threshold."""
    if not instrument:
        return []

    # Read in trade tick data from CSV file and process with TradeTickDataWrangler
    try:
        wrangler = TradeTickDataWrangler(instrument=instrument)
        df = pd.read_csv(
            filename,
            index_col="timestamp",
            date_parser=ts_parser,
            parse_dates=True,
            skiprows=1,
            names=["trade_id","price","quantity","quoteQty","timestamp","buyer_maker"]
        )
        ticks = wrangler.process(df)
    except Exception as e:
        print(f"Failed to process trade tick data: {e}")
        return []

    # Set up clock, logger, and handler for volume bar aggregation
    clock = TestClock()
    handler = [] 
    bar_spec = BarSpecification(threshold, BarAggregation.VOLUME, PriceType.LAST)
    bar_type = BarType(instrument.id, bar_spec)
    aggregator = VolumeBarAggregator(instrument, bar_type, handler.append, Logger(clock))

    # Iterate over trade ticks and process through VolumeBarAggregator
    for tick in ticks:
        aggregator.handle_trade_tick(tick)

    return handler


import json

def main():
    # Set up command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', default="../../example_data/BTCBUSD-trades-2022-12.csv")
    parser.add_argument('--symbol', default='BTCBUSD')
    parser.add_argument("--venue",default='BINANCE') 
    parser.add_argument('--threshold', type=int, default=10)
    parser.add_argument('--api_key_file', default = "config.json")
    parser.add_argument('--account_type', default=BinanceAccountType.FUTURES_USDT)
    args = parser.parse_args()

    catalog = ParquetDataCatalog("../../catalog/.")
    # Read API key and secret from JSON file
    with open(args.api_key_file, 'r') as f:
        api_key_data = json.load(f)
    api_key = api_key_data['api_key']
    api_secret = api_key_data['api_secret']
    instrument_id = f"{args.symbol}-PERP.{args.venue}"
    # Get the instrument for the given symbol
    instrument = asyncio.run(get_instrument(instrument_id , api_key, api_secret, args.account_type))
    #instrument = catalog.instruments(instrument_ids=[instrument_id],as_nautilus=True)[0]
    # Aggregate trade tick data into volume bars
    volume_bars = aggregate_volumebar(instrument, args.filename, args.threshold)

    # Write volume bars to catalog
    if len(volume_bars) > 0:
        write_objects(catalog, chunk = volume_bars)
        #write_objects(catalog, chunk = [instrument])

if __name__ == "__main__":
    main()

