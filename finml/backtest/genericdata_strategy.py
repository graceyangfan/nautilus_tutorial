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
from nautilus_trader.model.data.extended_bar import ExtendedBar
from nautilus_trader.model.data.bar import Bar
from nautilus_trader.model.data.tick import TradeTick
from nautilus_trader.model.data.base import DataType
from nautilus_trader.model.identifiers import ClientId
from nautilus_trader.core.data import Data
from nautilus_trader.indicators.zscore import Zscore
from nautilus_trader.indicators.linear_regression import LinearRegression
from nautilus_trader.indicators.bias import Bias 
from nautilus_trader.indicators.rvi import RelativeVolatilityIndex
from nautilus_trader.indicators.kvo import KlingerVolumeOscillator
from nautilus_trader.indicators.mfi import MoneyFlowIndex



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
        self.zscore = Zscore(157)
        self.linear_regression = LinearRegression(120)
        self.bias = Bias(196)
        self.rvi = RelativeVolatilityIndex(369)
        self.kvo = KlingerVolumeOscillator(185,272,242)
        self.mfi = MoneyFlowIndex(193)

        self.columns = [
            #"close",
            #"vlome",
            "zscore",
            "cfo",
            "bias",
            "rvi",
            "kvo",
            "mfi"
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
        self.data_type = DataType(ExtendedBar, metadata={"bar_type":self.bar_type})
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


    def on_bar(self, data: ExtendedBar):
        self.zscore.update_raw(data.close.as_double())
        self.linear_regression.update_raw(data.close.as_double())
        self.bias.update_raw(data.close.as_double())
        self.rvi.update_raw(data.close.as_double())
        self.kvo.update_raw(
            data.high.as_double(),
            data.low.as_double(),
            data.close.as_double(),
            data.volume.as_double(),
        )
        self.mfi.update_raw(
            data.high.as_double(),
            data.low.as_double(),
            data.close.as_double(),
            data.volume.as_double(),
        )
        self.log.info(
            f"current bar is {data}",
        )
        self.log.info(
            f"current zscore value  is {self.zscore.value}",
        )
        # check if indicators has initialized 
        if not self.zscore.initialized:
                return 
        if not self.linear_regression.initialized:
            return 
        if not self.bias.initialized:
            return 
        if not self.rvi.initialized:
            return  
        if not self.kvo.initialized:
            return 
        if not self.mfi.initialized:
            return 
        
        # update df 
        indicator_info = np.zeros((1,6)) 
        indicator_info[0,0] = self.zscore.value 
        indicator_info[0,1] = self.linear_regression.cfo 
        indicator_info[0,2] = self.bias.value  
        indicator_info[0,3] = self.rvi.value  
        indicator_info[0,4] = self.kvo.value
        indicator_info[0,5] = self.mfi.value 

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
        if isinstance(data, ExtendedBar):
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
        self.linear_regression.reset()
        self.bias.reset()
        self.rvi.reset()
        self.kvo.reset()
        self.mfi.reset()

