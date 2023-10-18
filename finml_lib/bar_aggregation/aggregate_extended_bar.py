
from extended_bar import * 
import numpy as np 
from nautilus_trader.core.correctness import PyCondition
from nautilus_trader.model.data.bar import BarType
from nautilus_trader.model.instruments.base import Instrument
from nautilus_trader.model.enums import AggressorSide
from extended_bar import ExtendedBar 

class ExtendedBarBuilder:
    def __init__(
        self,
        instrument: Instrument,
        bar_type:BarType,
        on_bar,
    ):
        PyCondition.equal(instrument.id, bar_type.instrument_id, "instrument.id", "bar_type.instrument_id")

        self._bar_type = bar_type
        self.price_precision = instrument.price_precision
        self.size_precision = instrument.size_precision
        self.level_array = np.array([1,10,100,1000,10000])
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
        self.volume =  Quantity.zero(precision=self.size_precision)

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


        self._cum_value = Decimal(0)  # Cumulative value

    def apply_update(
        self,
        ticker : Ticker,
    ):
        price = ticker.price
        size = ticker.size
        aggressor_side= ticker.aggressor_side
        ts_event = ticker.ts_event 
        size_update = size 
        dollar_value = price.as_double() * size.as_double()
        # recieve new ticker and update aggregation value 
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

        while size_update > 0:
            value_update = price * size_update 
            if self._cum_value + value_update < self._bar_type.spec.step:
                 self._cum_value = self._cum_value + value_update
                 self.update(
                    price=price,
                    size=Quantity(size_update, precision=self.size_precision),
                    aggressor_side = aggressor_side,
                    ts_event=ts_event,
                 )
                 break 
            value_diff: Decimal = self._bar_type.spec.step - self._cum_value
            size_diff: Decimal = size_update * (value_diff / value_update)
            # Update builder to the step threshold
            self.update(
                price=price,
                size=Quantity(size_diff, precision=self.size_precision),
                aggressor_side = aggressor_side,
                ts_event=ts_event,
            )
            new_bar = self.build()
            # generate New bar and call strategy on_bar to handle this bar 
            self.on_bar(new_bar) 
            self._cum_value = Decimal(0)

            # Decrement the update size
            size_update -= size_diff
            assert size_update >= 0

    def update(
        self,
        price: Price, 
        size: Quantity,
        aggressor_side: AggressorSide,
        ts_event: int,
    ):
        # TODO: What happens if the first tick updates before a partial bar is applied?
        if ts_event < self.ts_last:
            return  # Not applicable

        if self._open is None:
            # Initialize builder
            self._open = price
            self._high = price
            self._low = price
            self.initialized = True
        elif price > self._high:
            self._high = price
        elif price < self._low:
            self._low = price

        self._close = price
        self.volume += size

        self.count += 1
        self.ts_last = ts_event

    def set_partial(self,partial_bar: ExtendedBar):

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

    def reset(self):
        self._open = None
        self._high = None
        self._low = None

        self.volume = Quantity.zero(precision=self.size_precision)
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

    def build(self):
        ts_event = self.ts_last 
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
            volume= Quantity(self.volume, self.size_precision),
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