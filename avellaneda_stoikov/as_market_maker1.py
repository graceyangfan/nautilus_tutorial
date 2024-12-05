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


import datetime
import numpy as np 
import sys 
from decimal import Decimal
from typing import Optional
from collections import deque 
from nautilus_trader.config import StrategyConfig
from nautilus_trader.core.data import Data
from nautilus_trader.model.data.bar import Bar
from nautilus_trader.model.data.bar import BarType
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


from nautilus_trader.czsc.enums import (
    DivergenceType,
    Mark,
    Direction,
    BIType,
    LineType,
    TradePointType,
    ZSProcessType,
    SupportType
)
from nautilus_trader.czsc.twist import TwistConfig,Twist
from nautilus_trader.czsc.base_object import BI, LINE 
from nautilus_trader.czsc.utils import last_confirm_line 

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
        trailling_stop percent to save profit.
    """

    instrument_id: str
    bar_type: str
    order_qty: float = 1 
    n_spreads: int = 10 
    estimate_window: int = 600000
    period: int = 2000 
    sigma_tick_period: int = 500 
    sigma_multiplier: float = 1.0 
    gamma: float = 0.2 
    #ema_tick_period: int = 200 
    stop_loss: float = 0.00618  
    stoploss_sleep: int = 300000
    stopprofit: float = 0.00618 
    trailling_stop: float = 0.00382 
    request_bar_days: int = 10 


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
        self.bar_type = BarType.from_str(config.bar_type)
        self.request_bar_days = config.request_bar_days 
        self.order_qty = config.order_qty 
        self.n_spreads = config.n_spreads 
        self.estimate_window = config.estimate_window 
        self.period = config.period 
        self.sigma_tick_period = config.sigma_tick_period
        self.sigma_multiplier = config.sigma_multiplier 
        self.gamma = config.gamma 
        #self.ema_tick_period = config.ema_tick_period 
        self.stop_loss = config.stop_loss 
        self.stoploss_sleep = config.stoploss_sleep
        self.stopprofit = config.stopprofit 
        self.trailling_stop = config.trailling_stop
     

        #self.ema_array = deque(maxlen=3)
        self.wap = deque(maxlen = self.sigma_tick_period) 
        self.imb = deque(maxlen = self.sigma_tick_period) 
        self.spread = deque(maxlen = self.sigma_tick_period) 
        self.tv = deque(maxlen = self.sigma_tick_period) 
        #self.ema = ExponentialMovingAverage(config.ema_tick_period)
        self.as_model: Optional[AvellanedaStoikov] = None  
        self.instrument: Optional[Instrument] = None
        self.timer = 0 
        self.position_amount = 0 
        self.entry_price = 0 
        self.unrealized_pnl  = 0 
        self.in_stoploss = False 
        self.active_trailling_stop = False 
        self._book = None  # type: Optional[OrderBook]


        self.twist_config = TwistConfig(
        instrument_id=config.instrument_id,
        bar_type=config.bar_type,
        )
        self.twist = Twist(self.twist_config)
        self.direction = 0 
    def on_start(self):
        """Actions to be performed on strategy start."""
        self.instrument = self.cache.instrument(self.instrument_id)

        if self.instrument is None:
            self.log.error(f"Could not find instrument for {self.instrument_id}")
            self.stop()
            return

        self.as_model = AvellanedaStoikov(
            self.instrument.price_increment,
            self.n_spreads,
            self.estimate_window,
            self.period,
            self.clock.timestamp_ms()
        )

        self.subscribe_order_book_snapshots(
            instrument_id=self.instrument.id,
            book_type=BookType.L1_TBBO,
            #depth=5, 
            interval_ms=1000
        )
        self._book = OrderBook.create(
            instrument=self.instrument,
            book_type=BookType.L1_TBBO,
        )

        ## twist 
        self.twist.on_start(self)
        # Get historical data
        from_datetime = datetime.datetime.utcnow() -datetime.timedelta(days=self.request_bar_days)
        self.request_bars(
            bar_type=self.bar_type,
            from_datetime = from_datetime,
            )
        #subscribe real data 
        self.subscribe_bars(self.bar_type)

    def on_historical_data(self, data: Data):
        """
        preprocessing bars.
        """
        if isinstance(data, Bar):
            self.twist.on_bar(data) 

    def on_bar(self, bar: Bar):
        self.on_historical_data(bar)
        ## predict directions with twist bars 
        if len(self.twist.xds) < 1:
            return 
        last_xd = self.twist.xds[-1] 
        self.last_done_bi = last_confirm_line(self.twist.bis)
        if last_xd.divergence_type_exists([DivergenceType.OSCILLATION,DivergenceType.TREND]) and last_xd.direction_type == Direction.DOWN and self.last_done_bi.direction_type == Direction.UP:
            self.direction = 1 
        elif last_xd.direction_type == Direction.UP and not last_xd.divergence_type_exists([DivergenceType.OSCILLATION,DivergenceType.TREND]) and self.last_done_bi.direction_type == Direction.UP:
            self.direction = 1 
        elif last_xd.divergence_type_exists([DivergenceType.OSCILLATION,DivergenceType.TREND]) and last_xd.direction_type == Direction.UP and self.last_done_bi.direction_type == Direction.DOWN:
            self.direction = -1 
        elif last_xd.direction_type == Direction.DOWN and not last_xd.divergence_type_exists([DivergenceType.OSCILLATION,DivergenceType.TREND]) and self.last_done_bi.direction_type == Direction.DOWN:
            self.direction = -1 
        else:
            self.direction = 0 

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
            (self._book.best_bid_price() * self._book.best_ask_qty() \
            + self._book.best_ask_price() * self._book.best_bid_qty()) /(self._book.best_ask_qty() + self._book.best_bid_qty())
        )
        self.imb.append(self._book.best_bid_qty() / (self._book.best_ask_qty() + self._book.best_bid_qty()))
        self.spread.append(self._book.spread()/ self.wap[-1])
        self.tv.append(abs(self.wap[-1] / self.wap[0] - 1.0) + self.spread[-1] / self.wap[-1])
        #self.ema.update_raw(self.wap[-1]) 
        #self.ema_array.append(self.ema.value)
        
        buy_a, buy_k, sell_a, sell_k = self.as_model.calculate_intensity_info(
            self._book.best_ask_price(),
            self._book.best_bid_price(),
            int(self._book.ts_last / 10**6)
        )

        ## make sure the model hase been initialized
        if len(self.wap) < self.sigma_tick_period or (not self.as_model.initialized())  or len(self.twist.xds) < 1:
            return         

        self.buy_a = buy_a + sys.float_info.epsilon
        self.buy_k = buy_k + sys.float_info.epsilon
        self.sell_a = sell_a + sys.float_info.epsilon
        self.sell_k = sell_k + sys.float_info.epsilon 

        spread_value = self.calculate_spread() 
        if spread_value:
             spread_ask, spread_bid  = spread_value
        else:
            return 
        self.log.info(f"current as estimate are {buy_a},{buy_k},{sell_a},{sell_k}")
        self.log.info(f"current spread are {spread_ask},{spread_bid}")
        self.log.info(f"current postions are ampount:{self.position_amount},price:{self.entry_price}")


        if not self.in_stoploss:
            if self.position_amount > 0 :
                self.unrealized_pnl = self._book.best_bid_price() / self.entry_price  - 1.0
            elif self.position_amount < 0:
                self.unrealized_pnl = 1.0 - self._book.best_ask_price() / self.entry_price 
            else:
                self.unrealized_pnl = 0.0 

            ## try to stop profit 
            if (self.unrealized_pnl > self.trailling_stop) and (self.timer <  self._book.ts_last / 10**9 -10):
                self.active_trailling_stop = True 
            if self.active_trailling_stop and (self.unrealized_pnl < self.trailling_stop):
                self.cancel_all_orders(self.instrument.id)
                self.close_all_positions(self.instrument.id)
                self.unrealized_pnl = 0.0  
                self.active_trailling_stop = False 
                self.timer =  self._book.ts_last / 10**9
            
            # try to stop loss 
            if self.unrealized_pnl < -self.stop_loss:
                self.cancel_all_orders(self.instrument.id)
                self.close_all_positions(self.instrument.id)
                self.in_stoploss = True 
                self.unrealized_pnl = 0.0  
                self.active_trailling_stop = False 
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
                self.log.info(f'current wap is {self.wap[-1]}')
                self.log.info(f'current spread ask is {spread_ask}')
                self.log.info(f'current spread bid is {spread_bid}')
    
                if  self.portfolio.is_flat(self.instrument_id) and self.direction > 0:
                    self.buy(buy_price, self.order_qty)
                    self.timer =  self._book.ts_last / 10**9
                if  self.portfolio.is_flat(self.instrument_id) and self.direction < 0:
                    self.sell(sell_price, self.order_qty) 
                    self.timer =  self._book.ts_last / 10**9
                else:
                    self.buy(buy_price, self.order_qty)
                    self.sell(sell_price, self.order_qty) 
                    self.timer =  self._book.ts_last / 10**9

        elif self.timer <= self._book.ts_last / 10**9 - self.stoploss_sleep:
            self.in_stoploss = False 
    

    def calculate_spread(self):
        
        self.sigma = self.calculate_gk_volatility()
        self.log.info(f"current sigma is {self.sigma}")
        if  np.isnan(self.sigma):
            return  False 
        sigma_fix = self.sigma * self.sigma_multiplier
        q_fix = self.position_amount / self.order_qty

        bid =  np.log(1. + self.gamma / self.sell_k) / self.gamma  + (q_fix + 0.5)* \
            np.sqrt(
                (sigma_fix * sigma_fix * self.gamma) / (2. * self.sell_k * self.sell_a)* \
                   np.power(1. + self.gamma / self.sell_k, 1. + self.sell_k / self.gamma)
            )

        ask = np.log(1. + self.gamma / self.buy_k) / self.gamma - (q_fix - 0.5)* \
            np.sqrt(
                (sigma_fix * sigma_fix * self.gamma) / (2. * self.buy_k * self.buy_a)* \
                    np.power(1. + self.gamma / self.buy_k, 1. + self.buy_k / self.gamma)
            )

        # direction_long 
        if self.direction > 0 :
            ask = ask * 1.618
            bid = bid * 0.618
        # direction short 
        elif self.direction < 0 :
            ask = ask * 0.618
            bid = bid * 1.618
        
        return ask,bid 


    def calculate_gk_volatility(self):

        wap_vec = np.array(self.wap) 
        t = 0 
        garman_klass_hv = 0. 
        step = 30 
        for i in range(0,len(wap_vec) - step,step):
            co = np.log((wap_vec[i+step] / wap_vec[i]))
            hl = np.log(max(wap_vec[i:i+step]) / min (wap_vec[i:i+step]))
            res = 0.5 * np.power(hl, 2) - (2.0* np.log(2) -1.0) * np.power(co, 2)
            garman_klass_hv += res
            t = t + 1
            
        return np.sqrt(garman_klass_hv / t)

    def calculate_p_volatility(self):
        wap_vec = np.array(self.wap) 
        t = 10. 
        parkinson_hv = 0. 
        step = 30 
        for i in range(0,len(wap_vec) - step,step):
            hl = np.log(max(wap_vec[i:i+step]) / min (wap_vec[i:i+step]))
            res = hl.powi(2) 
            parkinson_hv += res 
        
        res = np.sqrt(parkinson_hv / (4. * t * (2. * np.log(2)))) 

        return res 


    def buy(self, price, quantity):
        """
        Users simple buy method (example).
        """
        order: LimitOrder = self.order_factory.limit(
            instrument_id=self.instrument_id,
            order_side=OrderSide.BUY,
            quantity=self.instrument.make_qty(quantity),
            price=self.instrument.make_price(price),
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
            post_only=True,  # default value is True
            # display_qty=self.instrument.make_qty(self.trade_size / 2),  # iceberg
        )

        self.submit_order(order)


    def on_reset(self):
        """
        Actions to be performed when the strategy is reset.
        """
        # Reset indicators here
        #self.ema_array.clear()
        self.wap.clear()
        self.imb.clear()
        self.spread.clear()
        self.tv.clear()
        self.twist.on_reset()
        #self.ema.reset() 

    def on_stop(self):
        """Actions to be performed when the strategy is stopped."""
        self.unsubscribe_order_book_snapshots(self.instrument.id)
        self.unsubscribe_bars(self.bar_type)
        self.cancel_all_orders(self.instrument.id)
        self.close_all_positions(self.instrument.id)
