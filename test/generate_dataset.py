
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
from nautilus_trader.model.data.imbalance_bar import ImbalanceBar
from nautilus_trader.model.data.bar import Bar
from nautilus_trader.config import CacheConfig
from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog

from nautilus_trader.model.instruments.currency_pair import CurrencyPair
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.identifiers import Symbol
from nautilus_trader.model.identifiers import Venue
from nautilus_trader.model.currencies import BUSD,USDT 
from nautilus_trader.model.currencies import ETH
from nautilus_trader.model.enums import AccountType
from nautilus_trader.model.enums import OmsType
from nautilus_trader.model.objects import Money
from nautilus_trader.model.objects import Price, Quantity
from nautilus_trader.model.data.bar import BarSpecification, BarType
from nautilus_trader.model.enums import AggressorSide, BarAggregation, PriceType

from  nautilus_trader.examples.strategies.imbalance_bar_strategy import GenericData,GenericDataConfig

import pyarrow as pa
ImbalanceBar_SCHEMA  = pa.schema(
        {
            "bar_type": pa.dictionary(pa.int8(), pa.string()),
            #"instrument_id": pa.dictionary(pa.int64(), pa.string()),
            "open": pa.float64(),
            "high": pa.float64(),
            "low": pa.float64(),
            "close": pa.float64(),
            "volume": pa.float64(),
            "small_buy_value": pa.float64(),
            "big_buy_value": pa.float64(),
            "small_sell_value": pa.float64(),
            "big_sell_value": pa.float64(),
            "ts_event": pa.int64(),
            "ts_init": pa.int64(),
        },
    metadata={"type": "ImbalanceBar"},
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
        base_currency=USDT,  # Standard single-currency account
        starting_balances=[Money(10_000_000, USDT)],
    )
    engine.add_instrument(params["instrument"])
    engine.add_instrument(params["base_instrument"])
    
    print(params["bar_type"])
    data_config1 = BacktestDataConfig(
            data_cls=ImbalanceBar,
            client_id="BINANCE",
            catalog_path=str(params["catalog"].path),
            catalog_fs_protocol=params["catalog"].fs_protocol,
            catalog_fs_storage_options=params["catalog"].fs_storage_options,
            #instrument_id=params["instrument_id"],
            start_time=params["start_time"],
            end_time=params["end_time"],
            metadata = params["bar_type"]
            )
    engine.add_data(data_config1.load()["data"])

    data_config2 = BacktestDataConfig(
            data_cls=Bar,
            client_id="BINANCE",
            catalog_path=str(params["catalog"].path),
            catalog_fs_protocol=params["catalog"].fs_protocol,
            catalog_fs_storage_options=params["catalog"].fs_storage_options,
            instrument_id=params["base_instrument_id"],
            start_time=params["start_time"],
            end_time=params["end_time"],
            )
    engine.add_data(data_config2.load()["data"])
    
    
    # Configure your strategy
    config = GenericDataConfig(
        instrument_id= params["instrument_id"],
        bar_type=params["bar_type"],
        base_instrument_id = params["base_instrument_id"],
        base_bar_type = params["base_bar_type"],
    )
    # Instantiate and add your strategy
    strategy = GenericData(config=config)
    engine.add_strategy(strategy=strategy)
    # Run the engine (from start to end of data)
    engine.run()
    results = engine.get_result()
    return  {"total_orders":results.total_orders,
            "stats_pnls":results.stats_pnls["USDT"]['PnL%'],
             "win rate":results.stats_pnls["USDT"]['Win Rate'],
             'avg_win':results.stats_pnls["USDT"]['Avg Winner'],
             "maxloss":results.stats_pnls["USDT"]['Max Loser'],
            "sharpratio":results.stats_returns['Sharpe Ratio (252 days)'],
             "sortin":results.stats_returns['Sortino Ratio (252 days)'],
            }

def get_ETH_USDT_instrument(exchange):
    return CurrencyPair(
        instrument_id=InstrumentId(
            symbol=Symbol("ETHUSDT-PERP"),
            venue=Venue(exchange),
        ),
        native_symbol=Symbol("ETHUSDT-PERP"),
        base_currency=ETH,
        quote_currency=USDT,
        price_precision=2,
        size_precision=3,
        price_increment=Price(1e-02, precision=2),
        size_increment=Quantity(1e-3, precision=3),
        lot_size=None,
        max_quantity=Quantity(9000, precision=3),
        min_quantity=Quantity(1e-03, precision=3),
        max_notional=None,
        min_notional=Money(1.00, USDT),
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
    num_workers = int(os.cpu_count ()/2.0)
    catalog = ParquetDataCatalog(path="catalog")
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', default='OPUSDT')
    parser.add_argument('--base_symbol', default='BTCUSDT')
    parser.add_argument("--venue",default='BINANCE') 
    parser.add_argument("--threshold",default=5) 
    args = parser.parse_args()
    instrument_id = f"{args.symbol}-PERP.{args.venue}"
    base_instrument_id = f"{args.base_symbol}-PERP.{args.venue}"
    base_instrument = catalog.instruments(instrument_ids=[base_instrument_id],as_nautilus=True)[0]
    # Get the instrument for the given symbol
    #instrument = asyncio.run(get_instrument(instrument_id , api_key, api_secret, args.account_type))
    if args.symbol == 'ETHUSDT':
        instrument = get_ETH_USDT_instrument(args.venue)
    else:
        instrument = catalog.instruments(instrument_ids=[instrument_id],as_nautilus=True)[0]
    

    bar_type = instrument_id + '-'+str(args.threshold) +"-VALUE_IMBALANCE-LAST-EXTERNAL"
    base_bar_type = base_instrument_id + '-1-MINUTE-LAST-EXTERNAL' 
    print(base_bar_type)
    params = {
        "catalog":catalog,
        "instrument_id":instrument_id,
        "instrument":instrument,
        "base_instrument_id":base_instrument_id,
        "base_instrument":base_instrument,
        "base_bar_type":base_bar_type,
        "bar_type":bar_type,
        "log_level":"WARNING",
        "bypass_logging":False,
        "persistence":False,
        "start_time":"2022-06-01",
        "end_time":None
    }

    backtest_instance(params)