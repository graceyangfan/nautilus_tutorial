import asyncio
from gettext import Catalog 
import os 
import json
import secrets 
from nautilus_trader.adapters.binance.common.enums import BinanceAccountType
from nautilus_trader.adapters.binance.factories import get_cached_binance_http_client
from nautilus_trader.adapters.binance.futures.providers import BinanceFuturesInstrumentProvider
from nautilus_trader.common.clock import LiveClock
from nautilus_trader.common.logging import Logger
from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog
from nautilus_trader.persistence.external.core import write_objects 

async def write_all_instruments(key,secret,instrument_id = "BTCUSDT.BINANCE"):
    clock = LiveClock()
    account_type = BinanceAccountType.FUTURES_USDT

    client = get_cached_binance_http_client(
        loop=asyncio.get_event_loop(),
        clock=clock,
        logger=Logger(clock=clock),
        account_type=account_type,
        key=key,
        secret=secret,
        is_testnet=True,
    )
    #client.connect()

    provider = BinanceFuturesInstrumentProvider(
        client=client,
        logger=Logger(clock=clock),
        account_type=BinanceAccountType.FUTURES_USDT,
    )
    await client.connect()

    await provider.load_all_async()

    instruments = provider.list_all()
    catalog = ParquetDataCatalog("../catalog/.")
    #write_objects(catalog,instruments)
    await client.disconnect()
    return provider.find(instrument_id)

if __name__ == "__main__":
    global data 
    with open("config.json", 'r') as f:
        api_key_data = json.load(f)
    api_key = api_key_data['api_key']
    api_secret = api_key_data['api_secret']
    asyncio.run(write_all_instruments(api_key,api_secret))