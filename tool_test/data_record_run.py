#!/usr/bin/env python3
# -------------------------------------------------------------------------------------------------
#  Copyright (C) 2015-2024 Nautech Systems Pty Ltd. All rights reserved.
#  https://nautechsystems.io
#
#  Licensed under the GNU Lesser General Public License Version 3.0 (the "License");
#  You may not use this file except in compliance with the License.
#  You may obtain a copy of the License at https://www.gnu.org/licenses/lgpl-3.0.en.html
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# -------------------------------------------------------------------------------------------------

import os 
import json 
from decimal import Decimal

from nautilus_trader.adapters.binance.common.enums import BinanceAccountType
from nautilus_trader.adapters.binance.config import BinanceDataClientConfig
from nautilus_trader.adapters.binance.config import BinanceExecClientConfig
from nautilus_trader.adapters.binance.factories import BinanceLiveDataClientFactory
from nautilus_trader.adapters.binance.factories import BinanceLiveExecClientFactory
from nautilus_trader.config import CacheConfig
from nautilus_trader.config import InstrumentProviderConfig
from nautilus_trader.config import LiveExecEngineConfig
from nautilus_trader.config import LoggingConfig
from nautilus_trader.config import TradingNodeConfig
from nautilus_trader.live.config import LiveDataEngineConfig
from nautilus_trader.live.node import TradingNode
from nautilus_trader.model.data import BarType
from nautilus_trader.model.identifiers import ClientId
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.identifiers import TraderId
from  nautilus_trader.examples.strategies.data_record import (
    DataRecordConfig,
    DataRecord
)
from nautilus_trader.persistence.config import StreamingConfig
from nautilus_trader.model.data import OrderBookDeltas

with open("config.json", 'r') as f:
    api_key_data = json.load(f)
api_key = api_key_data['api_key']
api_secret = api_key_data['api_secret']

# Configure the trading node
config_node = TradingNodeConfig(
    trader_id=TraderId("TESTER-001"),
    logging=LoggingConfig(
        log_level="INFO",
        # log_level_file="DEBUG",
        # log_file_format="json",
        log_colors=True,
        use_pyo3=True,
    ),
    exec_engine=LiveExecEngineConfig(
        reconciliation=True,
    ),
    cache=CacheConfig(
        # database=DatabaseConfig(timeout=2),
        timestamps_as_iso8601=True,
        flush_on_start=False,
    ),
    streaming=StreamingConfig(
        catalog_path="catalog",
        include_types = [OrderBookDeltas]
    ),
    data_clients={
        "BINANCE": BinanceDataClientConfig(
            api_key=api_key,  # 'BINANCE_API_KEY' env var
            api_secret=api_secret,  # 'BINANCE_API_SECRET' env var
            account_type=BinanceAccountType.USDT_FUTURE,
            base_url_http=None,  # Override with custom endpoint
            base_url_ws=None,  # Override with custom endpoint
            us=False,  # If client is for Binance US
            testnet=False,  # If client uses the testnet
            instrument_provider=InstrumentProviderConfig(load_all=True),
        ),
    },
    exec_clients={
        "BINANCE": BinanceExecClientConfig(
            api_key=api_key,  # 'BINANCE_API_KEY' env var
            api_secret=api_secret,  # 'BINANCE_API_SECRET' env var
            account_type=BinanceAccountType.USDT_FUTURE,
            base_url_http=None,  # Override with custom endpoint
            base_url_ws=None,  # Override with custom endpoint
            us=False,  # If client is for Binance US
            testnet=False,  # If client uses the testnet
            instrument_provider=InstrumentProviderConfig(load_all=True),
        ),
    },
    timeout_connection=30.0,
    timeout_reconciliation=10.0,
    timeout_portfolio=10.0,
    timeout_disconnection=10.0,
    timeout_post_stop=5.0,
)

# Instantiate the node with a configuration
node = TradingNode(config=config_node)

configs = [] 
strategys = [] 
symbols = [
    "FTMUSDT-PERP.BINANCE",
]
for index,symbol in enumerate(symbols):
    configs.append(
        DataRecordConfig(
            instrument_id = symbol,
            book_depth = 50,
            order_id_tag= str(index),
            oms_type="NETTING",
        )
    )
    strategys.append(DataRecord(config=configs[-1]))
# Add your strategies and modules
node.trader.add_strategies(strategys)

# Register your client factories with the node (can take user-defined factories)
node.add_data_client_factory("BINANCE", BinanceLiveDataClientFactory)
node.add_exec_client_factory("BINANCE", BinanceLiveExecClientFactory)
node.build()


# Stop and dispose of the node with SIGINT/CTRL+C
if __name__ == "__main__":
    try:
        node.run()
    finally:
        node.dispose()
