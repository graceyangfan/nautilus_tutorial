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

from collections import deque 
from nautilus_trader.core.correctness cimport Condition

from nautilus_trader.indicators.average.ma_factory import MovingAverageType
from nautilus_trader.indicators.atr cimport AverageTrueRange
from nautilus_trader.indicators.average.ema cimport ExponentialMovingAverage
from nautilus_trader.indicators.base.indicator cimport Indicator
from nautilus_trader.model.data cimport Bar


cdef class SuperTrend(Indicator):
    """
    SuperTrend Indicator 
    Parameters
    ----------
    period : int
        The rolling window period for the indicator (> 0).
    k_multiplier : double
        The multiplier for the ATR (> 0).
    ma_type : MovingAverageType
        The moving average type for the middle band (cannot be None).
    ma_type_atr : MovingAverageType
        The moving average type for the internal ATR (cannot be None).
    use_previous : bool
        The boolean flag indicating whether previous price values should be used.
    atr_floor : double
        The ATR floor (minimum) output value for the indicator (>= 0).
    """
    def __init__(
        self,
        int period,
        double k_multiplier,
        bint use_close,
        ma_type_atr not None: MovingAverageType=MovingAverageType.EXPONENTIAL,
        bint use_previous=True,
        double atr_floor=0,
    ):
        Condition.positive_int(period, "period")
        Condition.positive(k_multiplier, "k_multiplier")
        Condition.not_negative(atr_floor, "atr_floor")

        params = [
            period,
            k_multiplier,
            use_close,
            ma_type_atr.name,
            use_previous,
            atr_floor,
        ]
        super().__init__(params=params)

        self.period = period
        self.k_multiplier = k_multiplier
        self.use_close= use_close
        self._atr = AverageTrueRange(period, ma_type_atr, use_previous, atr_floor)
        self._previous_upper_band = 0
        self._previous_lower_band = 0
        self._current_upper_band = 0
        self._current_lower_band = 0
        self.direction = 0
        self.previous_direction = 0 
        self.change_direction = False

    cpdef void handle_bar(self, Bar bar) except *:
        """
        Update the indicator with the given bar.

        Parameters
        ----------
        bar : Bar
            The update bar.

        """
        Condition.not_none(bar, "bar")
        self.update_raw(
            bar.high.as_double(),
            bar.low.as_double(),
            bar.close.as_double()
        )
    cpdef void update_raw(
        self,
        double high,
        double low,
        double close,
    ) except *:
        """
        Update the indicator with the given raw values.

        Parameters
        ----------
        high : double
            The high price.
        low : double
            The low price.
        close : double
            The close price.

        """
        #update current_close
        self._atr.update_raw(high, low, close)

        cdef double typical_price = (high + low) / 2.0
        if self.use_close:
            typical_price = close 
        
        self._current_upper_band = typical_price  + self._atr.value * self.k_multiplier
        self._current_lower_band = typical_price  - self._atr.value * self.k_multiplier

        if close > self._previous_upper_band:
            self.direction = 1 
        elif close < self._previous_lower_band:
            self.direction = -1 
        else:
            self.direction = self.previous_direction 
            if self.direction > 0 and self._current_lower_band < self._previous_lower_band:
                self._current_lower_band = self._previous_lower_band 
            if self.direction < 0 and self._current_upper_band > self._previous_upper_band:
                self._current_upper_band = self._previous_upper_band 

        if self.previous_direction != self.direction:
            self.change_direction = True 
        else:
            self.change_direction = False 

        self._previous_lower_band = self._current_lower_band
        self._previous_upper_band = self._current_upper_band
        self.previous_direction = self.direction 
   
        # Initialization logic
        if not self.initialized:
            self._set_has_inputs(True)
            if self._atr.initialized:
                self._set_initialized(True)

    cpdef void _reset(self) except *:
        """
        Reset the indicator.

        All stateful fields are reset to their initial value.
        """
        self._atr.reset()
        self._previous_upper_band = 0
        self._previous_lower_band = 0
        self._current_upper_band = 0
        self._current_lower_band = 0
        self.direction = 0
        self.previous_direction = 0
        self.change_direction = False
