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

import numpy as np 
from sys.float_info import epsilon
from decimal import Decimal
from typing import Optional
from collections import deque 
from nautilus_trader.config import StrategyConfig
from nautilus_trader.model.enums import BookType
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.enums import TimeInForce
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.instruments.base import Instrument
from nautilus_trader.model.orderbook.book import OrderBook
from nautilus_trader.model.orderbook.data import OrderBookData
from nautilus_trader.trading.strategy import Strategy
from nautilus_trader.model.events.position import PositionChanged
from nautilus_trader.model.events.position import PositionClosed
from nautilus_trader.model.events.position import PositionOpened
from nautilus_trader.core.message import Event
from nautilus_trader.model.orders.limit import LimitOrder
from nautilus_trader.indicators.average.ema import ExponentialMovingAverage 
from avellaneda_stoikov.avellaneda_stoikov import AvellanedaStoikov 


# *** THIS IS A TEST STRATEGY WITH NO ALPHA ADVANTAGE WHATSOEVER. ***
# *** IT IS NOT INTENDED TO BE USED TO TRADE LIVE WITH REAL MONEY. ***


class ASMarketMakerConfig(StrategyConfig):
    """
    Configuration for ``ASMarketMaker`` instances.

    Parameters
    ----------
    instrument_id : InstrumentId
        The instrument ID for the strategy.
    order_qty : float 
        how many base_asset each time to trade.
    n_spread : int 
        Number of spread increments  
    estimate_window : int 
        estimate window for ak solver (milliseconds).
    period : int 
        bid and cancel order period (milliseconds)
    sigma_tick_period: int 
        how many tick u want to use for calc the volatility parameter
    sigma_multiplier: float 
        when the sigma esitmate did't doing will, u use this hack the paramter new_sigma = sigma * sigma_multiplier
    gamma: float 
        gamma form the paper
    ema_tick_period: int 
        EMA indictaor params 
    stop_loss: float 
        max loss percent of  the position 
    stoploss_sleep: int 
        after stop loss, how many milliseconds u want to stop trading
    stopprofit: float 
        max profit percent of a position 
    trailling_stop: float 
        trailing_stop percent to save profit.
    """

    instrument_id: str
    
    order_qty: float = 1 
    n_spreads: int = 10 
    estimate_window: int = 6000000
    period: int = 2000 
    sigma_tick_period: int = 500 
    sigma_multiplier: float = 1.0 
    gamma: float = 0.2 
    ema_tick_period: int = 200 
    stoploss: float = 0.00618  
    stoploss_sleep: int = 300000
    stopprofit: float = 0.00618 
    trailling_stop: float = 0.00382 


class ASMarketMaker(Strategy):
    """
    A simple strategy that use the AvellanedaStoikov model to do market maker.

    Parameters
    ----------
    config : ASMarketMakerConfig
        The configuration for the instance.
    """

    def __init__(self, config: ASMarketMakerConfig):
        super().__init__(config)

        # Configuration
        self.instrument_id = InstrumentId.from_str(config.instrument_id)
        self.sigma_multiplier = config.sigma_multiplier 
        self.order_qty = config.order_qty 
        self.trailing_stop = config.trailing_stop
        self.stop_loss = config.stop_loss 
        self.stopprofit = config.stopprofit 
        self.stoploss_sleep = config.stoploss_sleep
        self.period = config.period 
        self.sigma_tick_period = config.sigma_tick_period
        self.ema_tick_period = config.ema_tick_period 
        self.n_spreads = config.n_spreads 
        self.estimate_window = config.estimate_window 
        self.gamma = config.gamma 

        self.ema_array = deque(maxlen=3)
        self.wap = deque(maxlen = self.sigma_tick_period) 
        self.imb = deque(maxlen = self.sigma_tick_period) 
        self.spread = deque(maxlen = self.sigma_tick_period) 
        self.tv = deque(maxlen = self.sigma_tick_period) 
        self.ema = ExponentialMovingAverage(config.ema_tick_period)
        self.as_model: Optional[AvellanedaStoikov] = None  
        self.instrument: Optional[Instrument] = None
        self.timer = 0 
        self.position_amount = 0 
        self.entry_price = 0 
        self.unrealized_pnl  = 0 
        self.in_stoploss = False 
        self.active_trailing_stop = False 
        self._book = None  # type: Optional[OrderBook]

    def on_start(self):
        """Actions to be performed on strategy start."""
        self.instrument = self.cache.instrument(self.instrument_id)
        self.as_model = AvellanedaStoikov(
            self.instrument.min_quantity,
            self.n_spreads,
            self.estimate_window,
            self.period,
            self.clock.timestamp_ms()
        )

        if self.instrument is None:
            self.log.error(f"Could not find instrument for {self.instrument_id}")
            self.stop()
            return

        self.subscribe_order_book_deltas(
            instrument_id=self.instrument.id,
            book_type=BookType.L2_MBP,
        )
        self._book = OrderBook.create(
            instrument=self.instrument,
            book_type=BookType.L2_MBP,
        )

    def on_order_book_delta(self, data: OrderBookData):
        """Actions to be performed when a delta is received."""
        self._book.apply(data)
        if self._book.spread():
            self.on_tick() 

    def on_order_book(self, order_book: OrderBook):
        """Actions to be performed when an order book update is received."""
        self._book = order_book
        if self._book.spread():
            self.on_tick() 

    def on_event(self, event: Event):
        if isinstance(event, (PositionOpened, PositionChanged)):
            self.position_amount = event.quantity
            self.entry_price = event.avg_px_open


    def on_tick(self):
        ##update data 
        self.wap.append( 
            (self._book.best_bid_price * self._book.best_ask_qty \
            + self._book.best_ask_price * self._book.best_bid_qty) /(self._book.best_ask_qty + self._book.best_bid_qty)
        )
        self.imb.append(self._book.best_bid_qty / (self._book.best_ask_qty + self._book.best_bid_qty))
        self.spread.append((self._book.best_ask_price - self._book.best_bid_price) / self.wap[-1])
        self.tv.append(abs(self.wap[-1] / self.wap[0] - 1.0) + self.spread[-1] / self.wap[-1])
        self.ema.update_raw(self.wap[-1]) 
        self.ema_array.append(self.ema.value)


        buy_a, buy_k, sell_a, sell_k = self.as_model.calculate_intensity_info(
            self._book.best_ask_price,
            self._book.best_bid_price,
            self._book.ts_last / 10**6
            )
        
        ## make sure the model hase been initlized 
        if len(self.wap) < self.sigma_tick_period  or (not self.ema.initlized) or (not self.as_model.initlized):
            return 

        self.buy_a = buy_a + epsilon 
        self.buy_k = buy_k + epsilon 
        self.sell_a = sell_a + epsilon 
        self.sell_k = sell_k + epsilon  

        spread_ask, spread_bid = self.calculate_spread() 

        if not self.in_stoploss:
            if self.position_amount > 0 :
                self.unrealized_pnl = self._book.best_bid_price / self.entry_price  - 1.0
            elif self.position_amount < 0:
                self.unrealized_pnl = 1.0 - self._book.best_ask_price / self.entry_price 
            else:
                self.unrealized_pnl = 0.0 

            ## try to stop profit 
            if (self.unrealized_pnl > self.trailing_stop) and (self.timer <  self._book.ts_last / 10**9 -10):
                self.active_trailing_stop = True 
            if self.active_trailing_stop and (self.unrealized_pnl < self.trailing_stop):
                self.close_all_positions(self.instrument.id)
                self.unrealized_pnl = 0.0  
                self.active_trailing_stop = False 
                self.timer =  self._book.ts_last / 10**9
            
            # try to stop loss 
            if self.unrealized_pnl < -self.stoploss:
                self.cancel_all_orders(self.instrument.id)
                self.close_all_positions(self.instrument.id)
                self.in_stoploss = True 
                self.unrealized_pnl = 0.0  
                self.active_trailing_stop = False 
                self.timer =  self._book.ts_last / 10**9
            #try to stop profit 
            elif self.unrealized_pnl > self.stopprofit and  (self.timer <= self._book.ts_last / 10**9  - self.period / 10**3):
                self.cancel_all_orders(self.instrument.id)
                self.close_all_positions(self.instrument.id)
                self.unrealized_pnl = 0.0  
                self.timer =  self._book.ts_last / 10**9

            elif  (self.timer <= self._book.ts_last / 10**9  - self.period / 10**3):
                sell_price = self.wap[-1] + spread_ask 
                buy_price = self.wap[-1] - spread_bid 

                self.buy(buy_price, self.order_qty)
                self.sell(sell_price, self.order_qty) 
                self.timer =  self._book.ts_last / 10**9

        elif self.timer <= self._book.ts_last / 10**9 - self.stoploss_sleep:
            self.in_stoploss = False 
    

    def calculate_spread(self):
        
        self.sigma = self.calculate_gk_volatility()
        sigma_fix = self.sigma * self.sigma_multiplier
        q_fix = self.position_amount / self.order_qty

        bid =  np.log(1. + self.gamma / self.sell_k) / self.gamma  + (q_fix + 0.5)* \
            np.sqrt(
                (sigma_fix * sigma_fix * self.gamma) / (2. * self.sell_k * self.sell_a)* \
                   np.power(1. + self.gamma / self.sell_k, 1. + self.sell_k / self.gamma)
            )

        ask = np.log(1. + self.gamma / self.buy_k) / self.gamma +  (q_fix - 0.5)* \
            np.sqrt(
                (sigma_fix * sigma_fix * self.gamma) / (2. * self.buy_k * self.buy_a)* \
                    np.power(1. + self.gamma / self.buy_k, 1. + self.buy_k / self.gamma)
            )

        # direction_long 
        if self.ema_array[-1] > self.ema_array[-2] and self.ema_array[-2] > self.ema_array[-3]:
            ask = ask * 1.618
            bid = bid * 0.618
        # direction short 
        elif self.ema_array[-1] < self.ema_array[-2] and self.ema_array[-2] < self.ema_array[-3]:
            ask = ask * 0.618
            bid = bid * 1.618
        
        return ask,bid 


    def calculate_gk_volatility(self):

        wap_vec = np.array(self.wap) 
        t = 10.0 
        garman_klass_hv = 0. 
        step = 30 
        for i in range(0,len(wap_vec) - step,step):
            co = np.log((wap_vec[i+step] / wap_vec[i]))
            hl = np.log(max(wap_vec[i:i+step]) / min (wap_vec[i:i+step]))
            res = 0.5 * np.power(hl, 2) - (2.0* np.log(2) -1.0) * np.power(co, 2)
            garman_klass_hv += res
        
        return np.sqrt(garman_klass_hv / t)


    def buy(self, price, quantity):
        """
        Users simple buy method (example).
        """
        order: LimitOrder = self.order_factory.limit(
            instrument_id=self.instrument_id,
            order_side=OrderSide.BUY,
            quantity=self.instrument.make_qty(quantity),
            price=self.instrument.make_price(price),
            time_in_force=TimeInForce.GTX,
            post_only=True,  # default value is True
            # display_qty=self.instrument.make_qty(self.trade_size / 2),  # iceberg
        )

        self.submit_order(order)

    def sell(self, price, quantity):
        """
        Users simple sell method (example).
        """
        order: LimitOrder = self.order_factory.limit(
            instrument_id=self.instrument_id,
            order_side=OrderSide.SELL,
            quantity=self.instrument.make_qty(quantity),
            price=self.instrument.make_price(price),
            time_in_force=TimeInForce.GTX,
            post_only=True,  # default value is True
            # display_qty=self.instrument.make_qty(self.trade_size / 2),  # iceberg
        )

        self.submit_order(order)


    def on_reset(self):
        """
        Actions to be performed when the strategy is reset.
        """
        # Reset indicators here
        self.ema_array.clear()
        self.wap.clear()
        self.imb.clear()
        self.spread.clear()
        self.tv.clear()
        self.ema.reset() 

    def on_stop(self):
        """Actions to be performed when the strategy is stopped."""
        if self.instrument is None:
            return
        self.cancel_all_orders(self.instrument.id)
        self.close_all_positions(self.instrument.id)
