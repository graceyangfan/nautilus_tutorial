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
from libc.stdint cimport uint64_t


from nautilus_trader.common.clock cimport Clock
from nautilus_trader.common.timer cimport TimeEvent
from nautilus_trader.core.correctness cimport Condition
from nautilus_trader.model.data.orderflowbar cimport  OrderFlowBar
from nautilus_trader.model.data.bar cimport BarType
from nautilus_trader.model.data.tick cimport TradeTick
from nautilus_trader.model.enums_c cimport AggressorSide
from nautilus_trader.model.enums_c cimport BarAggregation
from nautilus_trader.model.enums_c cimport bar_aggregation_to_str
from nautilus_trader.model.instruments.base cimport Instrument
from nautilus_trader.model.objects cimport Price
from nautilus_trader.model.objects cimport Quantity
from nautilus_trader.core.datetime cimport dt_to_unix_nanos


cdef class OrderFlowBarBuilder:
    """
    Provides a generic OrderFlowBar builder for aggregation.
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
        int imbalance_ratio,
        Clock clock not None,
        double scaler = 10.0,
        bint build_with_no_updates = True,
        bint timestamp_on_close = True,
    ):
        Condition.equal(instrument.id, bar_type.instrument_id, "instrument.id", "bar_type.instrument_id")

        self._bar_type = bar_type

        #self.price_precision = instrument.price_precision
        self.size_precision = instrument.size_precision
        self.price_increment = instrument.price_increment.as_double()*scaler
        self.size_increment = instrument.size_increment.as_double()
        self.initialized = False
        self.ts_last = 0
        self.count = 0

        self._partial_set = False
        self.last_price = None 
        self.tick_direction = 1 
        self._last_close = None
        self._open = None
        self._high = None
        self._low = None
        self._close = None
        self.volume = Quantity.zero_c(precision=self.size_precision)

        self.on_bar = on_bar 
        self.imbalance_ratio = imbalance_ratio 
        self.ask_price = deque()
        self.bid_price = deque()
        self.ask_volume = deque()
        self.bid_volume = deque()

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
        #extra variables 
        self.pressure_levels = 0 
        self.imbalance_pressure_price = 0.0 
        self.support_levels = 0 
        self.imbalance_support_price = 0.0 
        self.top_imbalance_level1 = 0.0 
        self.top_imbalance_level2 = 0.0 
        self.bottom_imbalance_level1 = 0.0   
        self.bottom_imbalance_level2 = 0.0 
        self.top_index1 = 0 
        self.top_index2 = 0 
        self.bottom_index1 = 0 
        self.bottom_index2 = 0 
        self.max_volume = 0.0 
        self.total_volume = 0.0 
        self.point_of_control = 0.0 
        self.delta = 0.0 
    
    cpdef void set_partial(self, OrderFlowBar partial_bar):
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

    cpdef void update(
        self, 
        Price price, 
        Quantity size, 
        AggressorSide aggressor_side,
        uint64_t ts_event
    ) except *:
        """
        Update the bar builder.
        Parameters
        ----------
        price : Price
            The update price.
        size : Decimal
            The update size.
        aggressor_side: AggressorSide
            The aggressorside of the size.
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

        #save info 
        if self.tick_direction > 0:
            self.bid_price.append(price.as_double())
            self.bid_volume.append(size.as_double())
        else:
            self.ask_price.append(price.as_double())
            self.ask_volume.append(size.as_double())

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
    
    cdef void _apply_update(
        self, 
        Price price, 
        Quantity size, 
        AggressorSide aggressor_side, 
        uint64_t ts_event
    ) except *:

        if not self.last_price:
            self.tick_direction = 1 
        else:
            if abs(price.as_double() - self.last_price.as_double()) < 1e-20:
                self.tick_direction = self.tick_direction 
            elif price > self.last_price:
                self.tick_direction = 1 
            elif price < self.last_price:
                self.tick_direction = -1 
        self.delta += self.tick_direction * size.as_double() 

        self.update(price, size, aggressor_side, ts_event)
        self.last_price = price 

        #only call once for first bar 
        if self._build_on_next_tick:  # (fast C-level check)
            ts_init = ts_event
            ts_event = self._stored_close_ns
            if not self._timestamp_on_close:
                # Timestamp on open
                ts_event = self._stored_open_ns
            #build and send
            self._build_and_send(ts_event=ts_event, ts_init=ts_init)
            # Reset flag and clear stored close
            self._build_on_next_tick = False
            self._stored_close_ns = 0
    
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
    
        self.ask_price.clear()
        self.bid_price.clear()
        self.ask_volume.clear()
        self.bid_volume.clear()
        self.pressure_levels = 0 
        self.imbalance_pressure_price = 0.0 
        self.support_levels = 0 
        self.imbalance_support_price = 0.0 
        self.top_imbalance_level1 = 0.0 
        self.top_imbalance_level2 = 0.0 
        self.bottom_imbalance_level1 = 0.0   
        self.bottom_imbalance_level2 = 0.0 
        self.top_index1 = 0 
        self.top_index2 = 0 
        self.bottom_index1 = 0 
        self.bottom_index2 = 0 
        self.max_volume = 0.0 
        self.total_volume = 0.0 
        self.point_of_control = 0.0 
        self.delta = 0.0 

    cdef void _build_and_send(self, uint64_t ts_event, uint64_t ts_init):
        cdef OrderFlowBar bar = self.build(ts_event=ts_event, ts_init=ts_init)
        self.on_bar(bar)

    cpdef OrderFlowBar build(self, uint64_t ts_event, uint64_t ts_init):
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

        cdef int aggregate_price_list_length = int((self._high.as_double() - self._low.as_double()) // self.price_increment + 1)
        cdef np.ndarray[np.float64_t, ndim=1] aggregate_bid_volume, aggregate_ask_volume
        # Create arrays to aggregate bid and ask volumes
        aggregate_bid_volume = np.zeros(aggregate_price_list_length, dtype=np.float64)
        aggregate_ask_volume = np.zeros(aggregate_price_list_length, dtype=np.float64)

        for i in range(len(self.bid_price)):
            key = int((self.bid_price[i] - self._low.as_double()) // self.price_increment)
            aggregate_bid_volume[key] += self.bid_volume[i]

        for i in range(len(self.ask_price)):
            key = int((self.ask_price[i] - self._low.as_double()) // self.price_increment)
            aggregate_ask_volume[key] += self.ask_volume[i]
        

        #stack compute 
        for i in range(aggregate_price_list_length - 1):
            if aggregate_ask_volume[i+1] > self.imbalance_ratio * aggregate_bid_volume[i]:
                self.pressure_levels +=1 
                self.imbalance_pressure_price = self._low.as_double() + i * self.price_increment
        for i in range(aggregate_price_list_length - 2, -1, -1):
            if aggregate_bid_volume[i] > self.imbalance_ratio * aggregate_ask_volume[i + 1]:
                self.support_levels += 1 
                self.imbalance_support_price  = self._low.as_double() + i * self.price_increment

        self.top_index1 = int(aggregate_price_list_length/8.0*6.0)
        self.top_index2 = int(aggregate_price_list_length/8.0*7.0)
        self.bottom_index1 = int(aggregate_price_list_length/8.0)
        self.bottom_index2 = int(aggregate_price_list_length/8.0*2.0)

        if aggregate_price_list_length < 2:
            self.top_imbalance_level1 = 0.0 
            self.top_imbalance_level2 = 0.0 
            self.bottom_imbalance_level1 = 0.0   
            self.bottom_imbalance_level2 = 0.0 
        elif aggregate_price_list_length < 8:
            self.top_imbalance_level1 = (aggregate_bid_volume[-1] - aggregate_ask_volume[-1]) 
            self.top_imbalance_level2 = (aggregate_bid_volume[-2] - aggregate_ask_volume[-2]) 
            self.bottom_imbalance_level1 = (aggregate_bid_volume[0] - aggregate_ask_volume[0]) 
            self.bottom_imbalance_level2 = (aggregate_bid_volume[1] - aggregate_ask_volume[1]) 
        else:
            self.top_imbalance_level1 = (aggregate_bid_volume[self.top_index2:].sum() - aggregate_ask_volume[self.top_index2:].sum())
            self.top_imbalance_level2 = (aggregate_bid_volume[self.top_index1:self.top_index2].sum() - aggregate_ask_volume[self.top_index1:self.top_index2].sum())
            
            self.bottom_imbalance_level1 =(aggregate_bid_volume[:self.bottom_index1].sum() - aggregate_ask_volume[:self.bottom_index1].sum())
            self.bottom_imbalance_level2 =(aggregate_bid_volume[self.bottom_index1:self.bottom_index2].sum() - aggregate_ask_volume[self.bottom_index1:self.bottom_index2].sum())

        #compute poc 
        for i in range(aggregate_price_list_length):
            self.total_volume  = aggregate_ask_volume[i] + aggregate_bid_volume[i]
            if self.total_volume > self.max_volume:
                self.max_volume = self.total_volume 
                self.point_of_control = self._low.as_double() + i * self.price_increment 

        cdef OrderFlowBar bar = OrderFlowBar(
                bar_type=self._bar_type,
                open=self._open,
                high=self._high,
                low=self._low,
                close=self._close,
                volume=Quantity(self.volume, self.size_precision),
                imbalance_pressure_price=self.imbalance_pressure_price,
                imbalance_support_price=self.imbalance_support_price,
                pressure_levels=self.pressure_levels,
                support_levels=self.support_levels,
                point_of_control=self.point_of_control,
                top_imbalance_level1=self.top_imbalance_level1,
                top_imbalance_level2=self.top_imbalance_level2, 
                bottom_imbalance_level1=self.bottom_imbalance_level1,
                bottom_imbalance_level2=self.bottom_imbalance_level2,
                delta=self.delta,
                ts_event=ts_event,
                ts_init=ts_init,
            )
        self._last_close = self._close
        self.reset()
        return bar


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

