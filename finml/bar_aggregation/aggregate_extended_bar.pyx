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

from decimal import Decimal
from typing import Callable
import numpy as np

cimport numpy as np

from nautilus_trader.core.correctness cimport Condition
from nautilus_trader.model.data.bar cimport BarType
from nautilus_trader.model.instruments.base cimport Instrument
from nautilus_trader.model.enums_c cimport AggressorSide
from nautilus_trader.model.data.tick cimport TradeTick
from nautilus_trader.model.data.extended_bar cimport ExtendedBar
from nautilus_trader.model.objects cimport Price
from nautilus_trader.model.objects cimport Quantity


cdef class ExtendedBarBuilder:
    """
    Provides a generic extendedbar builder for aggregation.

    Parameters
    ----------
    instrument : Instrument
        The instrument for the builder.
    bar_type : BarType
        The bar type for the builder.

    Raises
    ------
    ValueError
        If `instrument.id` != `bar_type.instrument_id`.
    """
    def __init__(
        self,
        Instrument instrument not None,
        BarType bar_type not None,
        on_bar not None: Callable,
    ):
        Condition.equal(instrument.id, bar_type.instrument_id, "instrument.id", "bar_type.instrument_id")

        self._bar_type = bar_type

        self.price_precision = instrument.price_precision
        self.size_precision = instrument.size_precision
        self.level_array = np.asarray([1,10,100,1000,10000], dtype=np.float64)
        self.level_array = self.level_array * self._bar_type.spec.step /sum(self.level_array)

        self.on_bar = on_bar 
        self.initialized = False
        self.ts_last = 0
        self.count = 0

        self._partial_set = False
        self._last_close = None
        self._open = None
        self._high = None
        self._low = None
        self._close = None
        self.volume = Quantity.zero_c(precision=self.size_precision)

        self.bids_value_level_0 = 0.0
        self.bids_value_level_1 = 0.0
        self.bids_value_level_2 = 0.0
        self.bids_value_level_3 = 0.0
        self.bids_value_level_4 = 0.0
        self.asks_value_level_0 = 0.0
        self.asks_value_level_1 = 0.0
        self.asks_value_level_2 = 0.0
        self.asks_value_level_3 = 0.0
        self.asks_value_level_4 = 0.0

        self._cum_value = Decimal(0.0)  # Cumulative value
    
    cpdef void set_partial(self, ExtendedBar partial_bar) except *:
        """
        Set the initial values for a partially completed bar.

        This method can only be called once per instance.

        Parameters
        ----------
        partial_bar : ExtendedBar
            The partial bar with values to set.

        """
        if self._partial_set:
            return  # Already updated

        self._open = partial_bar.open

        if self._high is None or partial_bar.high > self._high:
            self._high = partial_bar.high

        if self._low is None or partial_bar.low < self._low:
            self._low = partial_bar.low

        if self._close is None:
            self._close = partial_bar.close

        self.volume = partial_bar.volume
        self.bids_value_level_0 = partial_bar.bids_value_level_0
        self.bids_value_level_1 = partial_bar.bids_value_level_1
        self.bids_value_level_2 = partial_bar.bids_value_level_2
        self.bids_value_level_3 = partial_bar.bids_value_level_3
        self.bids_value_level_4 = partial_bar.bids_value_level_4
        self.asks_value_level_0 = partial_bar.asks_value_level_0
        self.asks_value_level_1 = partial_bar.asks_value_level_1
        self.asks_value_level_2 = partial_bar.asks_value_level_2
        self.asks_value_level_3 = partial_bar.asks_value_level_3
        self.asks_value_level_4 = partial_bar.asks_value_level_4

        if self.ts_last == 0:
            self.ts_last = partial_bar.ts_init

        self._partial_set = True
        self.initialized = True

    cpdef void update(self, Price price, Quantity size, uint64_t ts_event) except *:
        """
        Update the bar builder.

        Parameters
        ----------
        price : Price
            The update price.
        size : Decimal
            The update size.
        ts_event : uint64_t
            The UNIX timestamp (nanoseconds) of the update.

        """
        Condition.not_none(price, "price")
        Condition.not_none(size, "size")

        # TODO: What happens if the first tick updates before a partial bar is applied?
        if ts_event < self.ts_last:
            return  # Not applicable

        if self._open is None:
            # Initialize builder
            self._open = price
            self._high = price
            self._low = price
            self.initialized = True
        elif price._mem.raw > self._high._mem.raw:
            self._high = price
        elif price._mem.raw < self._low._mem.raw:
            self._low = price

        self._close = price
        self.volume._mem.raw += size._mem.raw
        self.count += 1
        self.ts_last = ts_event

    cpdef void handle_trade_tick(self, TradeTick tick) except *:
        """
        Update the aggregator with the given tick.

        Parameters
        ----------
        tick : TradeTick
            The tick for the update.

        """
        Condition.not_none(tick, "tick")

        self._apply_update(
            price=tick.price,
            size=tick.size,
            aggressor_side=tick.aggressor_side,
            ts_event=tick.ts_event,
        )
    
    cdef void _apply_update(self, Price price, Quantity size, AggressorSide aggressor_side, uint64_t ts_event) except *:

        cdef double dollar_value
        dollar_value = price.as_double() * size.as_double() 
        if aggressor_side ==  AggressorSide.BUYER:
            if dollar_value < self.level_array[0]:
                self.bids_value_level_0 += dollar_value
            elif (dollar_value >= self.level_array[0] and dollar_value < self.level_array[1]):
                self.bids_value_level_1 += dollar_value
            elif (dollar_value >= self.level_array[1] and dollar_value < self.level_array[2]):
                self.bids_value_level_2 += dollar_value
            elif (dollar_value >= self.level_array[2] and dollar_value < self.level_array[3]):
                self.bids_value_level_3 += dollar_value
            elif dollar_value >= self.level_array[3]:
                self.bids_value_level_4 += dollar_value
        else:
            if dollar_value < self.level_array[0]:
                self.asks_value_level_0 += dollar_value
            elif (dollar_value >= self.level_array[0] and dollar_value < self.level_array[1]):
                self.asks_value_level_1 += dollar_value
            elif (dollar_value >= self.level_array[1] and dollar_value < self.level_array[2]):
                self.asks_value_level_2 += dollar_value
            elif (dollar_value >= self.level_array[2] and dollar_value < self.level_array[3]):
                self.asks_value_level_3 += dollar_value
            elif dollar_value >= self.level_array[3]:
                self.asks_value_level_4 += dollar_value


        size_update = size

        while size_update > 0:
            value_update = price * size_update 
            if self._cum_value + value_update < self._bar_type.spec.step:
                 self._cum_value = self._cum_value + value_update
                 self.update(
                    price=price,
                    size=Quantity(size_update, precision=size._mem.precision),
                    ts_event=ts_event,
                 )
                 break 
            value_diff: Decimal = self._bar_type.spec.step - self._cum_value
            size_diff: Decimal = size_update * (value_diff / value_update)
            # Update builder to the step threshold
            self.update(
                price=price,
                size=Quantity(size_diff, precision=size._mem.precision),
                ts_event=ts_event,
            )
            new_bar = self.build(self.ts_last)
            # generate New bar and call strategy on_bar to handle this bar 
            self.on_bar(new_bar) 
            self._cum_value = Decimal(0.0)

            # Decrement the update size
            size_update -= size_diff
            assert size_update >= 0

    cpdef void reset(self) except *:
        """
        Reset the bar builder.

        All stateful fields are reset to their initial value.
        """
        self._open = None
        self._high = None
        self._low = None

        self.volume = Quantity.zero_c(precision=self.size_precision)
        self.bids_value_level_0 = 0.0
        self.bids_value_level_1 = 0.0
        self.bids_value_level_2 = 0.0
        self.bids_value_level_3 = 0.0
        self.bids_value_level_4 = 0.0
        self.asks_value_level_0 = 0.0
        self.asks_value_level_1 = 0.0
        self.asks_value_level_2 = 0.0
        self.asks_value_level_3 = 0.0
        self.asks_value_level_4 = 0.0
        self.count = 0
    
    cpdef ExtendedBar build(self, uint64_t ts_event):
        """
        Return the aggregated bar with the given closing timestamp, and reset.

        Parameters
        ----------
        ts_event : uint64_t
            The UNIX timestamp (nanoseconds) of the bar close.

        Returns
        -------
        Bar

        """
        if self._open is None:  # No tick was received
            self._open = self._last_close
            self._high = self._last_close
            self._low = self._last_close
            self._close = self._last_close

        bar =ExtendedBar(
            bar_type=self._bar_type,
            open=self._open,
            high=self._high,
            low=self._low,
            close=self._close,
            volume=Quantity(self.volume, self.size_precision),
            bids_value_level_0 = self.bids_value_level_0, 
            bids_value_level_1 = self.bids_value_level_1, 
            bids_value_level_2 = self.bids_value_level_2, 
            bids_value_level_3 = self.bids_value_level_3, 
            bids_value_level_4 = self.bids_value_level_4, 
            asks_value_level_0 = self.asks_value_level_0, 
            asks_value_level_1 = self.asks_value_level_1, 
            asks_value_level_2 = self.asks_value_level_2, 
            asks_value_level_3 = self.asks_value_level_3, 
            asks_value_level_4 = self.asks_value_level_4, 
            ts_event=ts_event,
            ts_init=ts_event,
        )

        self._last_close = self._close
        self.reset()
        return bar