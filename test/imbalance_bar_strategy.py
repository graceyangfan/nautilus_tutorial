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

import numpy as np 
import pandas as pd 
from typing import Optional


from nautilus_trader.config import StrategyConfig
from nautilus_trader.trading.strategy import Strategy
from nautilus_trader.model.data.bar import BarType
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.instruments.base import Instrument
from nautilus_trader.model.data.imbalance_bar import ImbalanceBar
from nautilus_trader.model.data.bar import Bar
from nautilus_trader.model.data.tick import TradeTick
from nautilus_trader.model.data.base import DataType
from nautilus_trader.model.identifiers import ClientId
from nautilus_trader.core.data import Data

from nautilus_trader.indicators.zscore import Zscore
from nautilus_trader.indicators.zigzag import Zigzag
#from nautilus_trader.indicators.linear_regression import LinearRegression #cfo
#from nautilus_trader.indicators.bias import Bias 
from nautilus_trader.indicators.rvi import RelativeVolatilityIndex
#from nautilus_trader.indicators.kvo import KlingerVolumeOscillator
#from nautilus_trader.indicators.mfi import MoneyFlowIndex
#from nautilus_trader.indicators.stochastics import Stochastics
#from nautilus_trader.indicators.rsi import RelativeStrengthIndex
#from nautilus_trader.indicators.brar import BRAR 
#from nautilus_trader.indicators.bop import BalanceOfPower
from features.features import Entropy,MicroStructucture,RollStats
#from nautilus_trader.indicators.average.sma import SimpleMovingAverage
#from nautilus_trader.indicators.uo import UltimateOscillator
#from nautilus_trader.indicators.slope import Slope
#from nautilus_trader.indicators.ad import AccumulationDistribution #cmf 



class GenericDataConfig(StrategyConfig, kw_only=True):
    instrument_id: str
    bar_type: str



class GenericData(Strategy):

    def __init__(self, config: GenericDataConfig):
        super().__init__(config)

        # Configuration
        self.instrument_id = InstrumentId.from_str(config.instrument_id)
        self.bar_type = BarType.from_str(config.bar_type)
        self.instrument: Optional[Instrument] = None  # Initialized in on_start
        self.zscore = Zscore(40)
        self.rvi = RelativeVolatilityIndex(70)
        self.zigzag = Zigzag(0.022,True)

        self.ms = MicroStructucture(250,1)
        self.entropy = Entropy(30,10,-0.2,0.2)
        self.rollstats_level_0 = RollStats(100)
        self.rollstats_level_1 = RollStats(100)

       

        self.columns = [
            'zscore',
            'rvi',
            "zizag_strength",
            "zigzag_value",
            'vpin',
            'roll_effective_spread_1',
            'bar_based_kyle_lambda',
            'shannon_entropy',
            'level_0_diff_kurt',
            'level_1_diff_kurt',
            "datetime",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
          
        
        self.series_list = [] 
        self.df = pd.DataFrame([],columns=self.columns)


    def on_start(self):
        """Actions to be performed on strategy start."""
        self.instrument = self.cache.instrument(self.instrument_id)
        if self.instrument is None:
            self.log.error(f"Could not find instrument for {self.instrument_id}")
            self.stop()
            return

        # Get historical data
        #self.request_trade_ticks(self.instrument_id)

        # Subscribe to live data
        #self.subscribe_trade_ticks(self.instrument_id)
        self.data_type = DataType(ImbalanceBar, metadata={"bar_type":self.bar_type})
        #self.subscribe_data(
        #    data_type=self.data_type,
        #    client_id=ClientId("BINANCE")
        #)
        self.subscribe_bars(self.bar_type)
        
    def on_trade_tick(self, tick: TradeTick):
        """
        Actions to be performed when the strategy is running and receives a trade tick.

        Parameters
        ----------
        tick : TradeTick
            The tick received.

        """
        pass


    def on_bar(self, data: ImbalanceBar):
        self.bar_close = data.close.as_double()
        self.zscore.update_raw(
            data.close.as_double(),
        )
        self.rvi.update_raw(
            data.close.as_double(),
        )
        self.zigzag.update_raw(
            data.open.as_double(),
            data.high.as_double(),
            data.low.as_double(),
            data.close.as_double(),
            pd.Timestamp(data.ts_event, tz="UTC")
        )
      
        self.ms.update_raw(
            data.high.as_double(),
            data.low.as_double(),
            data.close.as_double(),
            data.volume.as_double(), 
        )

        self.entropy.update_raw(
            data.close.as_double(),
        )

        self.rollstats_level_0.update_raw(
            data.small_buy_value - data.small_sell_value
        )

        self.rollstats_level_1.update_raw(
            data.big_buy_value - data.big_sell_value
        )

        # check if indicators has initialized 
        if not self.zscore.initialized:
                return 
        if not self.rvi.initialized:
            return  
        if not self.zigzag.initialized:
            return 

        if not self.ms.initialized():
            return 

        if not self.entropy.initialized():
            return 

        if not self.rollstats_level_0.initialized():
            return 
    

        indicator_info = np.zeros((1,16)) 
 
        indicator_info[0,0] = self.zscore.value 
        indicator_info[0,1] = self.rvi.value 
        if self.zigzag.zigzag_direction == 1:
            indicator_info[0,2] = self.zigzag.zigzag_direction*self.zigzag.length/self.zigzag.low_price
            indicator_info[0,3] = (self.zigzag.high_price-self.bar_close)/self.zigzag.length
        else:
            indicator_info[0,2] = self.zigzag.zigzag_direction*self.zigzag.length/self.zigzag.high_price
            indicator_info[0,3] = (-self.zigzag.low_price+self.bar_close)/self.zigzag.length

        indicator_info[0,4] = self.ms.vpin()
        indicator_info[0,5] = self.ms.roll_effective_spread()[1]
        indicator_info[0,6] = self.ms.bar_based_kyle_lambda()
        indicator_info[0,7] = self.entropy.shannon_entropy()
       
        indicator_info[0,8] = self.rollstats_level_0.kurt()
        indicator_info[0,9] = self.rollstats_level_1.kurt()

        indicator_info[0,10] = data.ts_event 
        indicator_info[0,11] = data.open.as_double()
        indicator_info[0,12] = data.high.as_double() 
        indicator_info[0,13] = data.low.as_double() 
        indicator_info[0,14] = data.close.as_double() 
        indicator_info[0,15] = data.volume.as_double() 
        # update df 

        x= pd.DataFrame(indicator_info,columns=self.columns)
        self.series_list.append(x)

    def on_data(self, data: Data):
        """
        Actions to be performed when the strategy is running and receives generic data.

        Parameters
        ----------
        data : Data
            The data received.

        """
        if isinstance(data, ImbalanceBar):
            # update indicators 
           pass 

    
    def on_stop(self):
        """
        Actions to be performed when the strategy is stopped.
        """
        self.df = pd.concat(self.series_list)
        self.df.to_parquet(self.bar_type.__str__()+".parquet")
        self.cancel_all_orders(self.instrument_id)
        self.close_all_positions(self.instrument_id)

        # Unsubscribe from data
        #self.unsubscribe_trade_ticks(self.instrument_id)
        #self.unsubscribe_data(self.data_type)
        self.unsubscribe_bars(self.bar_type)

    def on_reset(self):
        """
        Actions to be performed when the strategy is reset.
        """
        # Reset indicators here
        self.zscore.reset()
        self.rvi.reset()
        self.zigzag.reset()

        self.ms.reset()
        self.entropy.reset()
        self.rollstats_level_0.reset()
        self.rollstats_level_1.reset()

