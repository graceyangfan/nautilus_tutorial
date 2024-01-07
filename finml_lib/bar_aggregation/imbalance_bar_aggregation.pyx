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
cimport numpy as np
from collections import deque
from decimal import Decimal
from typing import Callable
from cpython.datetime cimport datetime
from cpython.datetime cimport timedelta
from libc.math cimport isnan
from libc.stdint cimport uint64_t
from nautilus_trader.common.clock cimport Clock
from nautilus_trader.common.clock cimport TimeEvent
from nautilus_trader.common.logging cimport Logger
from nautilus_trader.common.logging cimport LoggerAdapter
from nautilus_trader.core.correctness cimport Condition
from nautilus_trader.core.datetime cimport dt_to_unix_nanos
from nautilus_trader.core.rust.core cimport millis_to_nanos
from nautilus_trader.core.rust.core cimport secs_to_nanos
from nautilus_trader.model.data cimport BarAggregation
from nautilus_trader.model.data cimport BarType
#from nautilus_trader.model.data cimport QuoteTick
from nautilus_trader.model.data cimport TradeTick
from nautilus_trader.model.functions cimport bar_aggregation_to_str
from nautilus_trader.model.instruments.base cimport Instrument
from nautilus_trader.model.objects cimport Price
from nautilus_trader.model.objects cimport Quantity
from nautilus_trader.model.imbalance_bar cimport ImbalanceBar
from nautilus_trader.indicators.zscore cimport Zscore
from nautilus_trader.core.stats cimport fast_mean
from nautilus_trader.core.stats cimport fast_std_with_mean

cdef class ImbalanceBarBuilder:
    """
    Provides a generic ImbalanceBar builder for aggregation.

    Parameters
    ----------
    instrument : Instrument
        The instrument for the builder.
    bar_type : BarType
        The bar type for the builder.
    on_bar : Callable
        The callback function for when a bar is completed.

    Raises
    ------
    ValueError
        If `instrument.id` != `bar_type.instrument_id`.
    """
    # Constructor
    def __init__(
        self,
        Instrument instrument not None,
        BarType bar_type not None,
        on_bar not None: Callable
    ):
        # Checking if instrument ID matches bar type's instrument ID
        Condition.equal(instrument.id, bar_type.instrument_id, "instrument.id", "bar_type.instrument_id")

        # Initializing variables
        self._bar_type = bar_type
        self.on_bar = on_bar 
        self.value_level = self._bar_type.spec.step 

        self.price_precision = instrument.price_precision
        self.size_precision = instrument.size_precision
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

        self.buy_dollor_value = deque()
        self.sell_dollor_value = deque()

        # Extra variables 
        self.big_buy_ratio = 0.0
        self.big_net_buy_ratio = 0.0
        self.big_buy_power = 0.0
        self.big_net_buy_power = 0.0
        self.value_delta = 0.0
        self.tag = 0 


    # Method to set partial values
    cpdef void set_partial(self, ImbalanceBar partial_bar):
        """
        Set the initial values for a partially completed bar.

        This method can only be called once per instance.

        Parameters
        ----------
        partial_bar : Bar
            The partial bar with values to set.
        """
        if self._partial_set:
            return  # Already updated

        # Setting values from the partial bar
        self._open = partial_bar.open

        if self._high is None or partial_bar.high > self._high:
            self._high = partial_bar.high

        if self._low is None or partial_bar.low < self._low:
            self._low = partial_bar.low

        if self._close is None:
            self._close = partial_bar.close

        self.volume = partial_bar.volume

        if self.ts_last == 0:
            self.ts_last = partial_bar.ts_init

        self._partial_set = True
        self.initialized = True

    # Method to update the builder
    cpdef void update(self, Price price, Quantity size, uint64_t ts_event):
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


    # Method to reset the builder
    cpdef void reset(self) except *:
        """
        Reset the bar builder.
        All stateful fields are reset to their initial value.
        """
        self._open = None
        self._high = None
        self._low = None

        self.volume = Quantity.zero_c(precision=self.size_precision)
        self.count = 0

        self.buy_dollor_value.clear()
        self.sell_dollor_value.clear()
        # Extra variables 
        self.big_buy_ratio = 0.0
        self.big_net_buy_ratio = 0.0
        self.big_buy_power = 0.0
        self.big_net_buy_power = 0.0
        self.value_delta = 0.0
        self.tag = 0 


    # Method to build the bar immediately and reset
    cpdef ImbalanceBar build_now(self):
        """
        Return the aggregated bar and reset.

        Returns
        -------
        Bar
        """
        return self.build(self.ts_last, self.ts_last)
        

    # Internal method to build now and send the bar
    cdef void _build_now_and_send(self):
        cdef ImbalanceBar bar = self.build_now()
        self.on_bar(bar)

    # Internal method to build and send the bar with specific timestamps
    cdef void _build_and_send(self, uint64_t ts_event, uint64_t ts_init):
        cdef ImbalanceBar bar = self.build(ts_event=ts_event, ts_init=ts_init)
        self.on_bar(bar)


    # Method to build the bar with given timestamps
    cpdef ImbalanceBar build(self, uint64_t ts_event, uint64_t ts_init):
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

        # Calculating some statistics
        cdef np.ndarray buy_dollor_value_array = np.asarray(self.buy_dollor_value, dtype=np.float64)
        cdef np.ndarray big_dif_value_array  = buy_dollor_value_array - np.asarray(self.sell_dollor_value, dtype=np.float64)
        cdef double buy_dollor_value_mean = fast_mean(buy_dollor_value_array)
        cdef double big_dif_value_mean = fast_mean(big_dif_value_array)
        cdef double buy_dollor_value_std = fast_std_with_mean(buy_dollor_value_array,buy_dollor_value_mean) * np.sqrt(self.count/(self.count-1.0))
        cdef double big_dif_value_std = fast_std_with_mean(big_dif_value_array,big_dif_value_mean) * np.sqrt(self.count/(self.count-1.0))
        
        self.big_buy_ratio = self.count * buy_dollor_value_mean / self.volume.as_double()
        self.big_net_buy_ratio = self.count * big_dif_value_mean / self.volume.as_double()
        self.big_buy_power = buy_dollor_value_mean / (buy_dollor_value_std + 1e-9)
        self.big_net_buy_power = big_dif_value_mean / (big_dif_value_std + 1e-9)
        
        # Creating an ImbalanceBar instance
        cdef ImbalanceBar bar = ImbalanceBar(
            bar_type=self._bar_type,
            open=self._open,
            high=self._high,
            low=self._low,
            close=self._close,
            volume=Quantity(self.volume, self.size_precision),
            big_buy_ratio=self.big_buy_ratio,
            big_net_buy_ratio=self.big_net_buy_ratio,
            big_buy_power=self.big_buy_power,
            big_net_buy_power=self.big_net_buy_power,
            value_delta=self.value_delta,
            tag=self.tag,
            ts_event=ts_event,
            ts_init=ts_init,
        )
        
        # Updating last close and resetting the builder
        self._last_close = self._close
        self.reset()
        return bar


    # Method to handle trade ticks and update the aggregator
    cpdef void handle_trade_tick(self, TradeTick tick):
        """
        Update the aggregator with the given tick.

        Parameters
        ----------
        tick : TradeTick
            The tick for the update.
        """
        Condition.not_none(tick, "tick")
        
        # Determining tick direction
        if not self.last_price:
            self.tick_direction = 1 
        else:
            if abs(tick.price.as_double() - self.last_price.as_double()) < 1e-20:
                self.tick_direction = self.tick_direction 
            elif tick.price > self.last_price:
                self.tick_direction = 1 
            elif tick.price < self.last_price:
                self.tick_direction = -1 

        cdef double dollar_value
        dollar_value = tick.price.as_double() * tick.size.as_double() 
        self.value_delta +=  self.tick_direction * dollar_value
        
        # Saving info based on tick direction and value level
        if self.tick_direction > 0 and dollar_value > self.value_level:
            self.buy_dollor_value.append(dollar_value)
        else:
            self.buy_dollor_value.append(0.0)
        
        if self.tick_direction < 0 and dollar_value > self.value_level:
            self.sell_dollor_value.append(dollar_value)
        else:
            self.sell_dollor_value.append(0.0)
            
        # Applying the update
        self._apply_update(
            price=tick.price,
            size=tick.size,
            ts_event=tick.ts_event,
        )
        self.last_price = tick.price 


    # Internal method to apply an update
    cdef void _apply_update(self, Price price, Quantity size, uint64_t ts_event):
        raise NotImplementedError("method must be implemented in the subclass")



cdef class ZscoreImbalanceBarAggregator(ImbalanceBarBuilder):
    """
    ZscoreImbalanceBarAggregator extends ImbalanceBarBuilder and adds Z-score based aggregation.

    Parameters
    ----------
    instrument : Instrument
        The instrument for the aggregator.
    bar_type : BarType
        The bar type for the aggregator.
    on_bar : Callable
        The callback function for when a bar is completed.
    zs_threshold : double, optional
        The Z-score threshold for triggering aggregation, defaults to 4.0.
    """
    # Constructor
    def __init__(
        self,
        Instrument instrument not None,
        BarType bar_type not None,
        on_bar not None: Callable,
        int zs_period,
        double zs_threshold = 4.0,
    ):
        # Call the constructor of the base class (ImbalanceBarBuilder)
        super().__init__(
            instrument=instrument,
            bar_type=bar_type,
            on_bar=on_bar
        )
        # Initialize Z-score related attributes
        self.zs_threshold = zs_threshold 
        self.zscore = Zscore(zs_period)
        self.last_zscore_value = float("nan")

    # Internal method to apply an update with Z-score calculation
    cdef void _apply_update(self, Price price, Quantity size, uint64_t ts_event):
        # Call the update method from the base class
        self.update(price, size, ts_event)

        # Update the Z-score with the product of tick direction, price, and size
        self.zscore.update_raw(
            self.tick_direction * price.as_double() * size.as_double()
        )

        # Check if Z-score is initialized
        if not self.zscore.initialized:
            return 
        # Check if last Z-score value is NaN (not a number)
        if isnan(self.last_zscore_value):
            self.last_zscore_value = self.zscore.value 
            return 

        # Determine Z-score-based events and trigger aggregation
        if self.last_zscore_value > self.zs_threshold and self.zscore.value < self.zs_threshold:
            self.tag = 2 
            self._build_now_and_send()
        elif self.last_zscore_value < -self.zs_threshold and self.zscore.value > -self.zs_threshold:
            self.tag = 4 
            self._build_now_and_send()

        # Update last Z-score value
        self.last_zscore_value = self.zscore.value 



cdef class CumsumImbalanceBarAggregator(ImbalanceBarBuilder):
    """
    CumsumImbalanceBarAggregator extends ImbalanceBarBuilder and adds aggregation based on cumulative returns.

    Parameters
    ----------
    instrument : Instrument
        The instrument for the aggregator.
    bar_type : BarType
        The bar type for the aggregator.
    on_bar : Callable
        The callback function for when a bar is completed.
    """
    # Constructor
    def __init__(
        self,
        Instrument instrument not None,
        BarType bar_type not None,
        on_bar not None: Callable,
    ):
        # Call the constructor of the base class (ImbalanceBarBuilder)
        super().__init__(
            instrument=instrument,
            bar_type=bar_type,
            on_bar=on_bar,
        )
        # Initialize return_of_bar attribute
        self.return_of_bar = 0.0

    # Internal method to apply an update with cumulative return calculation
    cdef void _apply_update(self, Price price, Quantity size, uint64_t ts_event):
        # Call the update method from the base class
        self.update(price, size, ts_event)

        # Check if last close price is available
        if not self._last_close:
            self._last_close = price 
            return 

        # Calculate the cumulative return of the bar
        self.return_of_bar = (price.as_double() - self._last_close.as_double()) / self._last_close.as_double()

        # Evaluate cumulative return-based conditions and trigger aggregation
        if self.return_of_bar < -self._bar_type.spec.step / 10000.0:
            self.tag = 0 
            self._build_now_and_send()
        elif self.return_of_bar > self._bar_type.spec.step / 10000.0:
            self.tag = 1 
            self._build_now_and_send()

        

cdef class ValueImbalanceBarAggregator(ImbalanceBarBuilder):
    """
    ValueImbalanceBarAggregator extends ImbalanceBarBuilder and aggregates bars based on cumulative value.

    Parameters
    ----------
    instrument : Instrument
        The instrument for the aggregator.
    bar_type : BarType
        The bar type for the aggregator.
    on_bar : Callable
        The callback function for when a bar is completed.
    """
    # Constructor
    def __init__(
        self,
        Instrument instrument not None,
        BarType bar_type not None,
        on_bar not None: Callable,
    ):
        # Call the constructor of the base class (ImbalanceBarBuilder)
        super().__init__(
            instrument=instrument,
            bar_type=bar_type,
            on_bar=on_bar,
        )

        # Initialize cumulative value attribute
        self._cum_value = Decimal(0)

    # Method to get the current cumulative value of the aggregator
    cpdef object get_cumulative_value(self):
        """
        Return the current cumulative value of the aggregator.

        Returns
        -------
        Decimal
        """
        return self._cum_value

    # Internal method to apply an update with cumulative value calculation
    cdef void _apply_update(self, Price price, Quantity size, uint64_t ts_event):
        # Copy the size for incremental updates
        size_update = size

        # While there is value to apply
        while size_update > 0:
            # Calculated value in quote currency
            value_update = price * size_update  

            if self._cum_value + value_update < self._bar_type.spec.step:
                # Update cumulative value and break
                self._cum_value = self._cum_value + value_update
                self.update(
                    price=price,
                    size=Quantity(size_update, precision=size._mem.precision),
                    ts_event=ts_event,
                )
                break

            # Calculate the remaining value and size to reach the step threshold
            value_diff: Decimal = self._bar_type.spec.step - self._cum_value
            size_diff: Decimal = size_update * (value_diff / value_update)

            # Update builder to the step threshold
            self.update(
                price=price,
                size=Quantity(size_diff, precision=size._mem.precision),
                ts_event=ts_event,
            )

            # Build a bar and reset builder and cumulative value
            self._build_now_and_send()
            self._cum_value = Decimal(0)

            # Decrement the update size
            size_update -= size_diff
            assert size_update >= 0



cdef class TimeImbalanceBarAggregator(ImbalanceBarBuilder):
    def __init__(
        self,
        Instrument instrument not None,
        BarType bar_type not None,
        on_bar not None: Callable,
        Clock clock not None,
        bint build_with_no_updates = True,
        bint timestamp_on_close = True,
    ):
        super().__init__(
            instrument=instrument,
            bar_type=bar_type,
            on_bar=on_bar,
        )
        # for time bar aggregate 
        self._clock = clock
        self._timer_name = None 
        self.interval = self._get_interval()
        self._set_build_timer()
        self.next_close_ns = self._clock.next_time_ns(self._timer_name)
        self._build_on_next_tick = False
        self._stored_open_ns = dt_to_unix_nanos(self.get_start_time())
        self._stored_close_ns = 0
        self._build_with_no_updates = build_with_no_updates
        self._timestamp_on_close = timestamp_on_close


    cpdef datetime get_start_time(self):
        """
        Return the start time for the aggregators next bar.

        Returns
        -------
        datetime
            The timestamp (UTC).

        """
        cdef datetime now = self._clock.utc_now()
        cdef int step = self._bar_type.spec.step

        cdef datetime start_time
        if self._bar_type.spec.aggregation == BarAggregation.MILLISECOND:
            diff_microseconds = now.microsecond % step // 1000
            diff_seconds = 0 if diff_microseconds == 0 else max(0, (step // 1000) - 1)
            diff = timedelta(
                seconds=diff_seconds,
                microseconds=now.microsecond,
            )
            start_time = now - diff
        elif self._bar_type.spec.aggregation == BarAggregation.SECOND:
            diff_seconds = now.second % step
            diff_minutes = 0 if diff_seconds == 0 else max(0, (step // 60) - 1)
            start_time = now - timedelta(
                minutes=diff_minutes,
                seconds=diff_seconds,
                microseconds=now.microsecond,
            )
        elif self._bar_type.spec.aggregation == BarAggregation.MINUTE:
            diff_minutes = now.minute % step
            diff_hours = 0 if diff_minutes == 0 else max(0, (step // 60) - 1)
            start_time = now - timedelta(
                hours=diff_hours,
                minutes=diff_minutes,
                seconds=now.second,
                microseconds=now.microsecond,
            )
        elif self._bar_type.spec.aggregation == BarAggregation.HOUR:
            diff_hours = now.hour % step
            diff_days = 0 if diff_hours == 0 else max(0, (step // 24) - 1)
            start_time = now - timedelta(
                days=diff_days,
                hours=diff_hours,
                minutes=now.minute,
                seconds=now.second,
                microseconds=now.microsecond,
            )
        elif self._bar_type.spec.aggregation == BarAggregation.DAY:
            start_time = now - timedelta(
                days=now.day % step,
                hours=now.hour,
                minutes=now.minute,
                seconds=now.second,
                microseconds=now.microsecond,
            )
        else:  # pragma: no cover (design-time error)
            raise ValueError(
                f"Aggregation type not supported for time bars, "
                f"was {bar_aggregation_to_str(self._bar_type.spec.aggregation)}",
            )

        return start_time


    cpdef void stop(self):
        """
        Stop the bar aggregator.
        """
        self._clock.cancel_timer(str(self._bar_type))
        self._timer_name = None

    cdef timedelta _get_interval(self):
        cdef BarAggregation aggregation = self._bar_type.spec.aggregation
        cdef int step = self._bar_type.spec.step

        if aggregation == BarAggregation.MILLISECOND:
            return timedelta(milliseconds=(1 * step))
        elif aggregation == BarAggregation.SECOND:
            return timedelta(seconds=(1 * step))
        elif aggregation == BarAggregation.MINUTE:
            return timedelta(minutes=(1 * step))
        elif aggregation == BarAggregation.HOUR:
            return timedelta(hours=(1 * step))
        elif aggregation == BarAggregation.DAY:
            return timedelta(days=(1 * step))
        else:
            # Design time error
            raise ValueError(
                f"Aggregation not time based, was {bar_aggregation_to_str(aggregation)}",
            )

    cpdef void _set_build_timer(self):
        self._timer_name = str(self._bar_type)
        self._clock.set_timer(
            name=self._timer_name,
            interval=self.interval,
            start_time=self.get_start_time(),
            stop_time=None,
            callback=self._build_bar,
        )


    cpdef void _build_bar(self, TimeEvent event):
        if not self.initialized:
            # Set flag to build on next close with the stored close time
            self._build_on_next_tick = True
            self._stored_close_ns = self.next_close_ns
            return

        if not self._build_with_no_updates and self.count == 0:
            return  # Do not build and emit bar

        cdef uint64_t ts_init = event.ts_event
        cdef uint64_t ts_event = event.ts_event
        if not self._timestamp_on_close:
            # Timestamp on open
            ts_event = self._stored_open_ns
        #build and send
        self._build_and_send(ts_event=ts_event, ts_init=ts_init)
        # Close time becomes the next open time
        self._stored_open_ns = event.ts_event

        # On receiving this event, timer should now have a new `next_time_ns`
        self.next_close_ns = self._clock.next_time_ns(self._timer_name)

    cdef void _apply_update(self, Price price, Quantity size, uint64_t ts_event):
        self.update(price, size, ts_event)
         #only call once for first bar 
        if self._build_on_next_tick:  # (fast C-level check)
            ts_init = ts_event
            ts_event = self._stored_close_ns
            if not self._timestamp_on_close:
                # Timestamp on open
                ts_event = self._stored_open_ns
            self._build_and_send(ts_event=ts_event, ts_init=ts_init)
            # Reset flag and clear stored close
            self._build_on_next_tick = False
            self._stored_close_ns = 0

