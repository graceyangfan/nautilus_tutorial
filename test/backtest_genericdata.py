
# -------------------------------------------------------------------------------------------------
#  Copyright (C) 2015-2023 Nautech Systems Pty Ltd. All rights reserved.
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

import gc 
import os 
import numpy as np 
import argparse
from decimal import Decimal
from tqdm import tqdm 
from joblib import Parallel,delayed 

from nautilus_trader.config import (
    BacktestEngineConfig,
    BacktestDataConfig,
    BacktestRunConfig,
    BacktestVenueConfig,
    ImportableStrategyConfig,
    RiskEngineConfig,
)
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.model.identifiers import Venue
from nautilus_trader.backtest.node import BacktestNode
from nautilus_trader.model.data.extended_bar import ExtendedBar
from nautilus_trader.config import CacheConfig
from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog

from nautilus_trader.model.instruments.currency_pair import CurrencyPair
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.identifiers import Symbol
from nautilus_trader.model.identifiers import Venue
from nautilus_trader.model.currencies import BUSD
from nautilus_trader.model.currencies import ETH
from nautilus_trader.model.enums import AccountType
from nautilus_trader.model.enums import OmsType
from nautilus_trader.model.objects import Money
from nautilus_trader.model.objects import Price, Quantity
from nautilus_trader.model.data.bar import BarSpecification, BarType
from nautilus_trader.model.enums import AggressorSide, BarAggregation, PriceType

from  nautilus_trader.examples.strategies.genericdata_strategy import GenericData,GenericDataConfig

import pyarrow as pa
from nautilus_trader.serialization.base import register_serializable_object
from nautilus_trader.serialization.arrow.serializer import register_parquet
ExtendedBar_SCHEMA  = pa.schema(
        {
            "bar_type": pa.dictionary(pa.int8(), pa.string()),
            #"instrument_id": pa.dictionary(pa.int64(), pa.string()),
            "open": pa.string(),
            "high": pa.string(),
            "low": pa.string(),
            "close": pa.string(),
            "volume": pa.string(),
            "bids_value_level_0": pa.float64(),
            "bids_value_level_1": pa.float64(),
            "bids_value_level_2": pa.float64(),
            "bids_value_level_3": pa.float64(),
            "bids_value_level_4": pa.float64(),
            "asks_value_level_0": pa.float64(),
            "asks_value_level_1": pa.float64(),
            "asks_value_level_2": pa.float64(),
            "asks_value_level_3": pa.float64(),
            "asks_value_level_4": pa.float64(),
            "ts_event": pa.uint64(),
            "ts_init": pa.uint64(),
        },
    metadata={"type": "ExtendedBar"},
)


def  backtest_instance(params):
    config = BacktestEngineConfig(
        trader_id="BACKTESTER-001",
    )
    # Build the backtest engine
    engine = BacktestEngine(config=config)

    # Setup trading instruments
    SIM = Venue("BINANCE")
    # Add starting balances for single-currency or multi-currency accounts
    engine.add_venue(
        venue=SIM,
        oms_type=OmsType.NETTING,
        account_type=AccountType.MARGIN,
        base_currency=BUSD,  # Standard single-currency account
        starting_balances=[Money(10_000_000, BUSD)],
    )
    engine.add_instrument(params["instrument"])
    
    data_config = BacktestDataConfig(
            data_cls=ExtendedBar,
            client_id="BINANCE",
            catalog_path=str(params["catalog"].path),
            catalog_fs_protocol=params["catalog"].fs_protocol,
            catalog_fs_storage_options=params["catalog"].fs_storage_options,
            #instrument_id=params["instrument_id"],
            start_time=params["start_time"],
            end_time=params["end_time"],
            metadata = params["bar_type"]
            )
    engine.add_data(data_config.load()["data"])
    
    # Configure your strategy
    config = GenericDataConfig(
        instrument_id= params["instrument_id"],
        bar_type=params["bar_type"],
    )
    # Instantiate and add your strategy
    strategy = GenericData(config=config)
    engine.add_strategy(strategy=strategy)
    # Run the engine (from start to end of data)
    engine.run()
    results = engine.get_result()
    return  {"total_orders":results.total_orders,
            "stats_pnls":results.stats_pnls["BUSD"]['PnL%'],
             "win rate":results.stats_pnls["BUSD"]['Win Rate'],
             'avg_win':results.stats_pnls["BUSD"]['Avg Winner'],
             "maxloss":results.stats_pnls["BUSD"]['Max Loser'],
            "sharpratio":results.stats_returns['Sharpe Ratio (252 days)'],
             "sortin":results.stats_returns['Sortino Ratio (252 days)'],
            }

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
    register_serializable_object(ExtendedBar, ExtendedBar.to_dict, ExtendedBar.from_dict)
    register_parquet(
        cls=ExtendedBar, 
        serializer=ExtendedBar.to_dict,
        deserializer= ExtendedBar.from_dict, 
        schema=ExtendedBar_SCHEMA
    )
    num_workers = int(os.cpu_count ()/2.0)
    catalog = ParquetDataCatalog(path="catalog")
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', default='ETHBUSD')
    parser.add_argument("--venue",default='BINANCE') 
    parser.add_argument("--threshold",default=833719) 
    args = parser.parse_args()
    instrument_id = f"{args.symbol}-PERP.{args.venue}"
    # Get the instrument for the given symbol
    #instrument = asyncio.run(get_instrument(instrument_id , api_key, api_secret, args.account_type))
    if args.symbol == 'ETHBUSD':
        instrument = get_ETH_BUSD_instrument(args.venue)
    else:
        instrument = catalog.instruments(instrument_ids=[instrument_id],as_nautilus=True)[0]
    

    bar_type = instrument_id + '-'+str(args.threshold) +"-VALUE-LAST-EXTERNAL"
    params = {
        "catalog":catalog,
        "instrument_id":instrument_id,
        "instrument":instrument,
        "bar_type":bar_type,
        "log_level":"WARNING",
        "bypass_logging":False,
        "persistence":False,
        "start_time":"2022-06-30",
        "end_time":None
    }

    backtest_instance(params)