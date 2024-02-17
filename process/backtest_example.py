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
from nautilus_trader.examples.strategies.as_market_maker import ASMarketMaker,ASMarketMakerConfig 
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
from nautilus_trader.model.enums import BookType
from nautilus_trader.model.orderbook.book import L2OrderBook
from nautilus_trader.model.orderbook.data import OrderBookData
from nautilus_trader.model.orderbook.data import OrderBookDeltas
from nautilus_trader.model.orderbook.data import OrderBookSnapshot
from functools import partial
from nautilus_trader.model.enums import AggregationSource, PriceType
from nautilus_trader.core.datetime import dt_to_unix_nanos
from nautilus_trader.backtest.data.providers import TestDataProvider
import gc 
from joblib import Parallel,delayed 
from tqdm import tqdm 


from nautilus_trader.config import StreamingConfig
def streaming_config(
        catalog: ParquetDataCatalog,
        kind: str = "backtest",
    ) -> StreamingConfig:
        return StreamingConfig(
            catalog_path=str(catalog.path),
            fs_protocol=catalog.fs_protocol,
            kind=kind,
        )


def backtest_one(params):
    # Configure backtest engine
    CATALOG_PATH = "catalog"
    #import pickle 
    
    streaming = streaming_config(ParquetDataCatalog(CATALOG_PATH))

    config = BacktestEngineConfig(
        trader_id="BACKTESTER-001",
        streaming=streaming,
    )
    # Build the backtest engine
    engine = BacktestEngine(config=config)

    # Setup trading instruments
    BINANCE = Venue("BINANCE")
   
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
        venue=BINANCE,
        oms_type=OMSType.NETTING,
        account_type=AccountType.MARGIN,
        base_currency=USDT,  # Standard single-currency account
        starting_balances=[Money(1_000_000, USDT)],
        book_type=BookType.L2_MBP,
    )

    engine.add_instrument(params["instrument"])
    data_config = BacktestDataConfig(
            catalog_path=str(CATALOG_PATH),
            data_cls=OrderBookData.fully_qualified_name(),
            instrument_id=params["instrument_id"],
            start_time = params["start_time"],
            end_time = params["end_time"],
        )
    engine.add_data(data_config.load()["data"])
    
    # Configure your strategy
    config = ASMarketMakerConfig(
        instrument_id=params["instrument_id"],
        order_qty= params["order_qty"],
        n_spreads= params["n_spreads"],
        estimate_window = params["estimate_window"],
        period = params["period"],
        sigma_tick_period = params["sigma_tick_period"],
        sigma_multiplier = params["sigma_multiplier"],
        gamma = params["gamma"],
        ema_tick_period = params["ema_tick_period"],
        stop_loss= params["stop_loss"],
        stoploss_sleep = params["stoploss_sleep"],
        stopprofit = params["stopprofit"],
        trailling_stop = params["trailling_stop"],
        order_id_tag="001",
    )
    # Instantiate and add your strategy
    strategy = ASMarketMaker(config=config)
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
        print(engine.trader.generate_account_report(BINANCE))
        print(engine.trader.generate_order_fills_report())
        print(engine.trader.generate_positions_report())
    engine.trader.generate_order_fills_report().to_csv("orders.csv")
    engine.trader.generate_account_report(BINANCE).to_csv("account.csv")
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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol',default='MATICUSDT')
    parser.add_argument("--venue",default='BINANCE') 
    parser.add_argument("--fileloction",default="data/compressed") 
    args = parser.parse_args()
    CATALOG_PATH = "catalog"
    catalog = ParquetDataCatalog(CATALOG_PATH)
    instrument_id = f"{args.symbol}-PERP.{args.venue}"
    instrument = catalog.instruments(instrument_ids=[instrument_id],as_nautilus=True)[0]
   
    num_workers = int(os.cpu_count ()/2.0)
    PARAM_SET = [
    ]
    import numpy as np 

    for i in [1]:
                        params=dict(
                            instrument = instrument,
                            instrument_id = instrument_id,
                            order_qty = 100,
                            n_spreads = 10,
                            estimate_window = 600000,
                            period = 2000,
                            sigma_tick_period = 500,
                            sigma_multiplier = 1.0,
                            gamma = 0.2,
                            ema_tick_period = 200,
                            stop_loss = 0.0618,
                            stoploss_sleep = 30000,
                            stopprofit = 0.00618,
                            trailling_stop = 0.00382,
                            start_time = '2022-11-13',
                            end_time =  '2022-11-14',
                        )
                        PARAM_SET.append(params)     
    total_results  = Parallel(n_jobs = num_workers)(
        delayed(backtest_one)(params) for params in tqdm(PARAM_SET )
        )
    total_results = pd.DataFrame(total_results)
    total_results["params"] = PARAM_SET
    total_results = total_results.sort_values(by="stats_pnls",ascending=False)
    total_results.to_csv("backtets_results.csv")
    
  
