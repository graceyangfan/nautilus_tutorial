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

#from nautilus_trader.indicators.zscore import Zscore
from nautilus_trader.indicators.linear_regression import LinearRegression #cfo
#from nautilus_trader.indicators.bias import Bias 
from nautilus_trader.indicators.rvi import RelativeVolatilityIndex
from nautilus_trader.indicators.kvo import KlingerVolumeOscillator
from nautilus_trader.indicators.mfi import MoneyFlowIndex
from nautilus_trader.indicators.stochastics import Stochastics
from nautilus_trader.indicators.rsi import RelativeStrengthIndex
from nautilus_trader.indicators.brar import BRAR 
from nautilus_trader.indicators.bop import BalanceOfPower
from features.features import FracDiff,Entropy,MicroStructucture
from nautilus_trader.indicators.average.sma import SimpleMovingAverage
from nautilus_trader.indicators.uo import UltimateOscillator
from nautilus_trader.indicators.slope import Slope
from nautilus_trader.indicators.ad import AccumulationDistribution #cmf 



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
        #self.zscore = Zscore(157)
        #self.linear_regression = LinearRegression(120)
        #self.bias = Bias(196)
        #
        self.rsi = RelativeStrengthIndex(69)
        self.uo = UltimateOscillator(107,145,193)
        self.slope = Slope(63)
        self.ad = AccumulationDistribution() 
        self.mfi = MoneyFlowIndex(72)
        self.brar = BRAR(83)
        self.rvi = RelativeVolatilityIndex(170)
        self.linear_regression = LinearRegression(334)
        self.kvo = KlingerVolumeOscillator(90,158,290)

        #self.stoch = Stochastics(222,346)
        #self.brar = BRAR(70)
        #self.kvo = KlingerVolumeOscillator(56,209,216)
        #self.bop = BalanceOfPower()
        self.ms = MicroStructucture(50,1)
        #self.close_frac_diff = FracDiff(0.8,200)
        #self.volume_frac_diff = FracDiff(0.8,200)
        self.bids_value_level_0_ma = SimpleMovingAverage(70)
        self.bids_value_level_1_ma = SimpleMovingAverage(70)
        self.bids_value_level_2_ma = SimpleMovingAverage(70)
        self.bids_value_level_3_ma = SimpleMovingAverage(70)
        self.bids_value_level_4_ma = SimpleMovingAverage(70)
        self.asks_value_level_0_ma = SimpleMovingAverage(70)
        self.asks_value_level_1_ma = SimpleMovingAverage(70)
        self.asks_value_level_2_ma = SimpleMovingAverage(70)
        self.asks_value_level_3_ma = SimpleMovingAverage(70)
        self.asks_value_level_4_ma = SimpleMovingAverage(70)
        self.value_level_0_diff_ma = SimpleMovingAverage(70)
        self.value_level_1_diff_ma = SimpleMovingAverage(70)
        self.value_level_2_diff_ma = SimpleMovingAverage(70)
        self.value_level_3_diff_ma = SimpleMovingAverage(70)
        self.value_level_4_diff_ma = SimpleMovingAverage(70)

        self.columns = [
            'bids_value_level_0',
            'bids_value_level_1',
            'bids_value_level_2',
            'bids_value_level_3',
            'bids_value_level_4',
            'asks_value_level_0',
            'asks_value_level_1',
            'asks_value_level_2',
            'asks_value_level_3',
            'asks_value_level_4',
            'value_level_0_diff',
            'value_level_1_diff',
            'value_level_2_diff',
            'value_level_3_diff',
            'value_level_4_diff',
            "rsi",
            "uo",
            "slope",
            "cmf",
            "ad",
            "mr",
            "mfi",
            #'stoch_k',
            #'stoch_d',
            'brar_ar',
            'brar_br',
            "rvi",
            "degree",
            "cfo",
            "R2",
            'kvo', 
            #'bop',
            "bar_based_amihud_lambda",
            "high_low_volatility", 
            "bids_value_level_0_ma",
            "bids_value_level_1_ma",
            "bids_value_level_2_ma",
            "bids_value_level_3_ma",
            "bids_value_level_4_ma",
            "asks_value_level_0_ma",
            "asks_value_level_1_ma",
            "asks_value_level_2_ma",
            "asks_value_level_3_ma",
            "asks_value_level_4_ma",
            "value_level_0_diff_ma",
            "value_level_1_diff_ma",
            "value_level_2_diff_ma",
            "value_level_3_diff_ma",
            "value_level_4_diff_ma",
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

        self.rsi.update_raw(
            data.close.as_double()
        )
        self.uo.update_raw(
            data.high.as_double(),
            data.low.as_double(),
            data.close.as_double(),        
        )

        #self.stoch.update_raw(
        #    data.high.as_double(),
        #    data.low.as_double(),
        #    data.close.as_double(),        
        #)
        self.slope.update_raw(
            data.close.as_double(),
        )
        self.ad.update_raw(
            data.open.as_double(),
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
        self.brar.update_raw(
            data.open.as_double(),
            data.high.as_double(),
            data.low.as_double(),
            data.close.as_double(),
        )
        self.rvi.update_raw(
            data.close.as_double(),
        )
        self.linear_regression.update_raw(
            data.close.as_double(),
        )

        self.kvo.update_raw(
            data.high.as_double(),
            data.low.as_double(),
            data.close.as_double(),
            data.volume.as_double(),
        )
        #self.bop.update_raw(
        #    data.open.as_double(),
        #    data.high.as_double(),
        #    data.low.as_double(),
        #    data.close.as_double(),
        #)
        self.ms.update_raw(
            data.high.as_double(),
            data.low.as_double(),
            data.close.as_double(),
            data.volume.as_double(), 
        )

        # frac diff 
        #self.close_frac_diff.update_raw(
            #data.close.as_double()
        #)
        #self.volume_frac_diff.update_raw(
            #data.volume.as_double()
        #)

        # aggreate value 
        self.bids_value_level_0_ma.update_raw(
            data.bids_value_level_0
        )
        self.bids_value_level_1_ma.update_raw(
            data.bids_value_level_1
        )
        self.bids_value_level_2_ma.update_raw(
            data.bids_value_level_2
        )
        self.bids_value_level_3_ma.update_raw(
            data.bids_value_level_3
        )
        self.bids_value_level_4_ma.update_raw(
            data.bids_value_level_4
        )
        self.asks_value_level_0_ma.update_raw(
            data.asks_value_level_0
        )
        self.asks_value_level_1_ma.update_raw(
            data.asks_value_level_1
        )
        self.asks_value_level_2_ma.update_raw(
            data.asks_value_level_2
        )
        self.asks_value_level_3_ma.update_raw(
            data.asks_value_level_3
        )
        self.asks_value_level_4_ma.update_raw(
            data.asks_value_level_4
        )
        self.value_level_0_diff_ma.update_raw(
            data.bids_value_level_0 - data.asks_value_level_0
        )
        self.value_level_1_diff_ma.update_raw(
            data.bids_value_level_1 - data.asks_value_level_1
        )
        self.value_level_2_diff_ma.update_raw(
            data.bids_value_level_2 - data.asks_value_level_2
        )
        self.value_level_3_diff_ma.update_raw(
            data.bids_value_level_3 - data.asks_value_level_3
        )
        self.value_level_4_diff_ma.update_raw(
            data.bids_value_level_4 - data.asks_value_level_4
        )

        # check if indicators has initialized 
        #if not self.zscore.initialized:
                #return 
        #if not self.linear_regression.initialized:
            #return 
        #if not self.bias.initialized:
            #return 

        if not self.rsi.initialized:
            return 
        #if not self.stoch.initialized:
            #return 
        if not self.uo.initialized:
            return 
        if not self.slope.initialized:
            return 
        if not self.ad.initialized:
            return 
        if not self.mfi.initialized:
            return 
        if not self.brar.initialized:
            return 
        if not self.rvi.initialized:
            return  
        if not self.linear_regression.initialized:
            return 
        if not self.kvo.initialized:
            return 
        #if not self.bop.initialized:
            #return 

        
        if not self.ms.initialized():
            return 
        #if not self.close_frac_diff.initialized():
            #return 
        #if not self.volume_frac_diff.initialized():
            #return 
        
        if not self.value_level_0_diff_ma.initialized:
            return 
        
        indicator_info = np.zeros((1,52)) 
        indicator_info[0,0] = data.bids_value_level_0
        indicator_info[0,1] = data.bids_value_level_1
        indicator_info[0,2] = data.bids_value_level_2
        indicator_info[0,3] = data.bids_value_level_3
        indicator_info[0,4] = data.bids_value_level_4
        indicator_info[0,5] = data.asks_value_level_0
        indicator_info[0,6] = data.asks_value_level_1
        indicator_info[0,7] = data.asks_value_level_2
        indicator_info[0,8] = data.asks_value_level_3
        indicator_info[0,9] = data.asks_value_level_4
        indicator_info[0,10] = data.bids_value_level_0 - data.asks_value_level_0
        indicator_info[0,11] = data.bids_value_level_1 - data.asks_value_level_1
        indicator_info[0,12] = data.bids_value_level_2 - data.asks_value_level_2
        indicator_info[0,13] = data.bids_value_level_3 - data.asks_value_level_3
        indicator_info[0,14] = data.bids_value_level_4 - data.asks_value_level_4
        indicator_info[0,15] = self.rsi.value 
        indicator_info[0,16] = self.uo.value 
        indicator_info[0,17] = self.slope.value 
        indicator_info[0,18] = self.ad.cmf 
        indicator_info[0,19] = self.ad.value 
        indicator_info[0,20] = self.mfi.mr 
        indicator_info[0,21] = self.mfi.value 
        #indicator_info[0,16] = self.stoch.value_k
        #indicator_info[0,17] = self.stoch.value_d
        indicator_info[0,22] = self.brar.ar 
        indicator_info[0,23] = self.brar.br 
        indicator_info[0,24] = self.rvi.value 
        indicator_info[0,25] = self.linear_regression.degree
        indicator_info[0,26] = self.linear_regression.cfo
        indicator_info[0,27] = self.linear_regression.R2 
        indicator_info[0,28] = self.kvo.value
        #indicator_info[0,21] = self.bop.value 
       
        indicator_info[0,29] = self.ms.bar_based_amihud_lambda()
        indicator_info[0,30] = self.ms.high_low_volatility()
        indicator_info[0,31] = self.bids_value_level_0_ma.value
        indicator_info[0,32] = self.bids_value_level_1_ma.value
        indicator_info[0,33] = self.bids_value_level_2_ma.value
        indicator_info[0,34] = self.bids_value_level_3_ma.value
        indicator_info[0,35] = self.bids_value_level_4_ma.value
        indicator_info[0,36] = self.asks_value_level_0_ma.value
        indicator_info[0,37] = self.asks_value_level_1_ma.value
        indicator_info[0,38] = self.asks_value_level_2_ma.value
        indicator_info[0,39] = self.asks_value_level_3_ma.value
        indicator_info[0,40] = self.asks_value_level_4_ma.value
        indicator_info[0,41] = self.value_level_0_diff_ma.value
        indicator_info[0,42] = self.value_level_1_diff_ma.value
        indicator_info[0,43] = self.value_level_2_diff_ma.value
        indicator_info[0,44] = self.value_level_3_diff_ma.value
        indicator_info[0,45] = self.value_level_4_diff_ma.value

        indicator_info[0,46] = data.ts_event 
        indicator_info[0,47] = data.open.as_double()
        indicator_info[0,48] = data.high.as_double() 
        indicator_info[0,49] = data.low.as_double() 
        indicator_info[0,50] = data.close.as_double() 
        indicator_info[0,51] = data.volume.as_double() 
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
        #self.zscore.reset()
        #self.linear_regression.reset()
        #self.bias.reset()
        #self.rvi.reset()
        self.rsi.reset()
        self.uo.reset()
        self.slope.reset()
        self.ad.reset()
        self.mfi.reset()
        self.brar.reset()
        self.rvi.reset()
        self.linear_regression.reset()
        self.kvo.reset()
        #self.close_frac_diff.reset()
        #self.volume_frac_diff.reset()
        self.bids_value_level_0_ma.reset()
        self.bids_value_level_1_ma.reset()
        self.bids_value_level_2_ma.reset()
        self.bids_value_level_3_ma.reset()
        self.bids_value_level_4_ma.reset()
        self.asks_value_level_0_ma.reset()
        self.asks_value_level_1_ma.reset()
        self.asks_value_level_2_ma.reset()
        self.asks_value_level_3_ma.reset()
        self.asks_value_level_4_ma.reset()
        self.value_level_0_diff_ma.reset()
        self.value_level_1_diff_ma.reset()
        self.value_level_2_diff_ma.reset()
        self.value_level_3_diff_ma.reset()
        self.value_level_4_diff_ma.reset()

