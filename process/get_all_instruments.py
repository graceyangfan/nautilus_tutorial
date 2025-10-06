import asyncio
from gettext import Catalog 
import os 
import json
import secrets 
from nautilus_trader.adapters.binance.common.enums import BinanceAccountType
from nautilus_trader.adapters.binance.factories import get_cached_binance_http_client
from nautilus_trader.adapters.binance.futures.providers import BinanceFuturesInstrumentProvider
from nautilus_trader.common.component import LiveClock
from nautilus_trader.common.component import Logger
from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog
from nautilus_trader.config import InstrumentProviderConfig

async def write_all_instruments(key,secret,instrument_id = "BTCUSDT.BINANCE"):
    clock = LiveClock()


    client = get_cached_binance_http_client(
        clock=clock,
        account_type=BinanceAccountType.USDT_FUTURE,
        is_testnet=False,
        api_key=key,
        api_secret=secret,
    )

    binance_provider = BinanceFuturesInstrumentProvider(
        client=client,
        clock=clock,
        config=InstrumentProviderConfig(load_all=True, log_warnings=False),
    )

    await binance_provider.load_all_async()

    instruments = binance_provider.list_all()
    catalog = ParquetDataCatalog("catalog/.")
    catalog.write_data(instruments)
    await client.disconnect()
    return binance_provider.find(instrument_id)

if __name__ == "__main__":
    global data 
    with open("config.json", 'r') as f:
        api_key_data = json.load(f)
    api_key = api_key_data['api_key']
    api_secret = api_key_data['api_secret']
    asyncio.run(write_all_instruments(api_key,api_secret))