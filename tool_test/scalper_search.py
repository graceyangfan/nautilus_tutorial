
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
import pandas as pd 
import argparse
from decimal import Decimal
from tqdm import tqdm 
from joblib import Parallel,delayed 
from nautilus_trader.backtest.node import BacktestNode
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.backtest.engine import BacktestEngineConfig
from nautilus_trader.config import BacktestDataConfig
from nautilus_trader.model.identifiers import Venue
from nautilus_trader.model.enums import AccountType
from nautilus_trader.model.enums import OmsType
from nautilus_trader.model.objects import Money
from nautilus_trader.model.currencies import USDT
from nautilus_trader.config import LoggingConfig
from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog
from hyperopt import RayBacktestNode 

def node_setup(params):
    data_configs = [
        BacktestDataConfig(
            catalog_path="../catalog",
            data_cls="nautilus_trader.model.data:Bar",
            instrument_id = params["instrument_id"],
            start_time = params["start_time"],
            end_time = params["end_time"],
        )
    ]
    venues_configs = [
        BacktestVenueConfig(
            name="BINANCE",
            oms_type="NETTING",
            account_type="MARGIN",
            base_currency="USDT",
            starting_balances=["100000 USDT"],
        )
    ]

    strategies = [
        ImportableStrategyConfig(
            strategy_path = "nautilus_trader.examples.strategies.reverse_scalper:Scapler",
            config_path = "nautilus_trader.examples.strategies.reverse_scalper:ScaplerConfig",
            config=dict(
                instrument_id=params["instrument_id"],
                bar_type= params["bar_type"],
                high_bar_type = params["high_bar_type"],
                hour_bar_type = params["hour_bar_type"],     
                trade_usd = 100.0,
                strength = 1.0,
                buffer_pct = 0.001,
                auto_reduce_start_pct = 0.06,
                min_notional = 6,
                check_timestamp_interval = pd.Timedelta(minutes=1),
                supertrend_period = 20.0,
                supertrend_mul = 4.,
                hour_supertrend_period = 30.,
                hour_supertrend_mul = 3.,
                atr_period = 8,
                mfi_period = 8,
                rsi_period = 8,
                lookback = 8,
                zigzag_pct = 0.02,
                take_profit_multiple = 4.5,
                reissue_threshold = 0.01,
            ),
        ),
    ]
    base_config = BacktestRunConfig(
        engine=BacktestEngineConfig(strategies=strategies),
        venues=venues_configs,
        data=data_configs,
    )
    node = RayBacktestNode(base_config)
    node.set_strategy_config(
        strategy_path = "nautilus_trader.examples.strategies.reverse_scalper:Scapler",
        config_path = "nautilus_trader.examples.strategies.reverse_scalper:ScaplerConfig",
    )

    return node 


if __name__ == "__mian__":

    num_workers = int(os.cpu_count ()//2)
    catalog = ParquetDataCatalog(path="../catalog")
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', default='WLDUSDT')
    parser.add_argument("--venue",default='BINANCE') 
    parser.add_argument("--minute1",type = int,default=  1)
    parser.add_argument("--minute2",type = int,default=  5)
    args = parser.parse_args()
    instrument_id = f"{args.symbol}-PERP.{args.venue}"
    instrument = catalog.instruments(instrument_ids=[instrument_id],as_nautilus=True)[0]
    bar_type =instrument_id + '-'+str(args.minute1) +"-MINUTE-LAST-EXTERNAL"
    high_bar_type = instrument_id + '-'+str(args.minute2) +"-MINUTE-LAST-EXTERNAL"
    hour_bar_type = instrument_id + "-1-HOUR-LAST-EXTERNAL"
   
    params = {
        "instrument_id":instrument_id,
        "bar_type":bar_type,
        "high_bar_type":high_bar_type,
        "hour_bar_type":hour_bar_type,
        "start_time": "2024-1-1",
        "end_time": "2024-3-1",
        "trade_usd": 100,
        "strength": 1.0,
        "buffer_pct":0.001,
        "auto_reduce_start_pct": 0.06,
        "min_notional": 6,
        "check_timestamp_interval":pd.Timedelta(minutes=1),
        "supertrend_period":tune.randint(10,100),
        "supertrend_mul":tune.uniform(1.0,9.),
        "hour_supertrend_period":tune.randint(10,100),
        "hour_supertrend_mul":tune.uniform(1.0,9.),
        "atr_period":8,
        "mfi_period":8,
        "rsi_period":8,
        "lookback":tune.randint(1,12),
        "zigzag_pct":0.02,
        "take_profit_multiple":tune.uniform(1.0,9.),
        "reissue_threshold":0.01,
    }
    
    best_params = node.ray_search(
        params=params,
        minimum_positions=30,
        max_evals=100
    )

    print("Best parameters found:", best_params)
