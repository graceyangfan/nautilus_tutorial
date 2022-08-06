#!/usr/bin/env python3
# -------------------------------------------------------------------------------------------------
#  Copyright (C) 2015-2022 Nautech Systems Pty Ltd. All rights reserved.
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

from datetime import datetime
from decimal import Decimal
import gc 
import pandas as pd
from nautilus_trader.backtest.data.providers import TestDataProvider
from nautilus_trader.backtest.data.providers import TestInstrumentProvider
from nautilus_trader.backtest.data.wranglers import QuoteTickDataWrangler
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.backtest.engine import BacktestEngineConfig
from nautilus_trader.backtest.models import FillModel
from nautilus_trader.backtest.modules import FXRolloverInterestModule
import os
from nautilus_trader.examples.strategies.double_ma_strategyV2 import DoubleMa,DoubleMaConfig 
from nautilus_trader.config.backtest import BacktestRunConfig, BacktestVenueConfig, BacktestDataConfig
from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog
from pathlib import Path
from nautilus_trader.model.currencies import USDT
from nautilus_trader.model.enums import AccountType
from nautilus_trader.model.enums import OMSType
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.identifiers import Symbol
from nautilus_trader.model.identifiers import Venue
from nautilus_trader.model.objects import Money
from nautilus_trader.model.instruments.currency_pair import CurrencyPair
from nautilus_trader.model.objects import Price, Quantity
from nautilus_trader.persistence.external.core import process_files, write_objects
from nautilus_trader.persistence.external.readers import ParquetReader
from nautilus_trader.model.data.bar import Bar, BarType, BarSpecification
from functools import partial
from nautilus_trader.model.enums import AggregationSource, PriceType
from nautilus_trader.model.c_enums.bar_aggregation import BarAggregation
from nautilus_trader.core.datetime import dt_to_unix_nanos
from nautilus_trader.backtest.data.wranglers import BarDataWrangler
from nautilus_trader.backtest.data.providers import TestDataProvider
import gc 
from joblib import Parallel,delayed 
from tqdm import tqdm 

from nautilus_trader.model.c_enums.currency_type import CurrencyType
from nautilus_trader.model.currency import Currency
ETH = Currency("ETH", precision=8, iso4217=0, name="ETH", currency_type=CurrencyType.CRYPTO)


def get_ETH_USDT_instrument(exchange):
    return CurrencyPair(
        instrument_id=InstrumentId(
            symbol=Symbol("ETHUSDT"),
            venue=Venue(exchange),
        ),
        native_symbol=Symbol("ETHUSDT"),
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
        maker_fee=Decimal("0.0002"),
        taker_fee=Decimal("0.0004"),
        ts_event=0,
        ts_init=0,
        )


def backtest_one(params):
    # Configure backtest engine
    CATALOG_PATH = os.getcwd() + "/catalog"
    ETH_USDT = get_ETH_USDT_instrument("BINANCE")
    #import pickle 
    #pickle.dump(catalog.bars(as_nautilus=True),"bnb_USDT.pkl")
    config = BacktestEngineConfig(
        trader_id="BACKTESTER-001",
    )
    # Build the backtest engine
    engine = BacktestEngine(config=config)

    # Setup trading instruments
    SIM = Venue("BINANCE")
   
    engine.add_instrument(ETH_USDT)
    
    data_config = BacktestDataConfig(
            catalog_path=str(CATALOG_PATH),
            data_cls="nautilus_trader.model.data.bar:Bar",
            instrument_id="ETHUSDT.BINANCE",
            start_time = "2022-6-1",
            end_time = "2022-7-1"
        )
    engine.add_data(data_config.load()["data"])
    
    # Create a fill model (optional)
    fill_model = FillModel(
        prob_fill_on_limit=1,
        prob_fill_on_stop=1,
        prob_slippage=0,
        random_seed=42,
    )
    # Add a trading venue (multiple venues possible)
    # Add starting balances for single-currency or multi-currency accounts
    engine.add_venue(
        venue=SIM,
        oms_type=OMSType.NETTING,
        account_type=AccountType.MARGIN,
        base_currency=USDT,  # Standard single-currency account
        starting_balances=[Money(1_000_000, USDT)],
        bar_execution=True,  # Recommended for running on bar data
        fill_model =fill_model,
    )

    # Configure your strategy
    config = DoubleMaConfig(
        instrument_id=str(ETH_USDT.id),
        bar_type="ETHUSDT.BINANCE-1-MINUTE-LAST-EXTERNAL",
        ml_models_dir = "models",
        fast_ema_period = params["fast_period"],
        slow_ema_period = params["slow_period"],
        trade_size=Decimal(0.1),
        order_id_tag="001",
    )
    # Instantiate and add your strategy
    strategy = DoubleMa(config=config)
    engine.add_strategy(strategy=strategy)


    # Run the engine (from start to end of data)
    engine.run()

    # Optionally view reports
    with pd.option_context(
        "display.max_rows",
        100,
        "display.max_columns",
        None,
        "display.width",
        300,
    ):
        print(engine.trader.generate_account_report(SIM))
        print(engine.trader.generate_order_fills_report())
        print(engine.trader.generate_positions_report())
    engine.trader.generate_order_fills_report().to_csv("orders.csv")
    engine.trader.generate_account_report(SIM).to_csv("account.csv")
    engine.trader.generate_positions_report().to_csv("positions.csv")
    results = engine.get_result()
    #engine.cache.cache_positions()
    print(engine.cache.positions())
    # For repeated backtest runs make sure to reset the engine
    engine.reset()

    # Good practice to dispose of the object when done
    engine.dispose()
    gc.collect()
    return  {"total_orders":results.total_orders,
            "stats_pnls":results.stats_pnls["USDT"]['PnL%'],
             "win rate":results.stats_pnls["USDT"]['Win Rate'],
             'avg_win':results.stats_pnls["USDT"]['Avg Winner'],
             "maxloss":results.stats_pnls["USDT"]['Max Loser'],
            "sharpratio":results.stats_returns['Sharpe Ratio (252 days)'],
             "sortin":results.stats_returns['Sortino Ratio (252 days)'],
            }





if __name__ == "__main__":
    num_workers = int(os.cpu_count ()/2.0)
    PARAM_SET = [
    ]
    import numpy as np 

    #for fp in np.arange(10,30,10):
        #for sp in np.arange(fp+10,100,10):
            #for sig in np.arange(10,30,5):
    #for ml_period in np.arange(40,100,10):
    for i in [1]:
        #for RS in np.arange(30,100,10):
    #for mp in np.arange(1,4.5,0.5):
           # for mlp in np.arange(0,1,0.1):
                        params={
                            "fast_period":10,
                            "slow_period":60,
                            "trade_size":Decimal(0.1)}
                        PARAM_SET.append(params)     
    backtest_one(PARAM_SET[0])
    total_results  = Parallel(n_jobs = num_workers)(
        delayed(backtest_one)(params) for params in tqdm(PARAM_SET )
        )
    total_results = pd.DataFrame(total_results)
    total_results["params"] = PARAM_SET
    total_results = total_results.sort_values(by="stats_pnls",ascending=False)
    total_results.to_csv("backtets_results.csv")
