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


from cProfile import run
from sklearn.linear_model import LinearRegression
from  nautilus_trader.czsc.base_object import (
    BI,
    NewBar,
    TwistBar,
    FX,
    LINE,
    ZS,
    TradePoint,
    BC,
    TZXL,
    XLFX,
    XD
)
import numpy as np
from  nautilus_trader.czsc.enums import (
    DivergenceType,
    Mark,
    Direction,
    BIType,
    LineType,
    TradePointType,
    ZSProcessType,
    SupportType
)
from collections import deque 
from typing import List, Dict, Union
from nautilus_trader.trading.strategy import Strategy
from nautilus_trader.config import StrategyConfig
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.instruments.base import Instrument
from nautilus_trader.model.data.bar import Bar
from nautilus_trader.model.data.bar import BarType
from nautilus_trader.indicators.average.ema import ExponentialMovingAverage
from nautilus_trader.indicators.macd import MovingAverageConvergenceDivergence
from nautilus_trader.indicators.bollinger_bands import BollingerBands
from nautilus_trader.indicators.rsi import RelativeStrengthIndex


class TwistConfig(StrategyConfig):
    bar_type: str
    instrument_id: str
    macd_fast: int 
    macd_slow: int 
    macd_signal = 9 
    boll_period = 20 
    boll_k = 2 
    ma_period = 5 
    rsi_period = 14 
    fake_bi = False  # if recognize un_confirmed bi 
    bi_type = BIType.OLD 
    fx_included = False 
    zs_process_type = ZSProcessType.INTERATE
    zs_support_type = SupportType.HL 
    bar_capacity: int = 2000
    twist_bar_capacity: int = 2000
    fx_capacity: int = 1000
    bi_capacity: int = 1000
    xd_capacity: int = 1000
    trend_capacity:int = 1000
    bi_zs_capacity: int = 500  
    xd_zs_capacity: int = 500 
    use_rsi: bool = True
    use_boll:bool = False 
    use_ma:bool = False 


    ##不继承Strategy 
class Twist:
    """
    行情数据缠论分析
    """

    def __init__(self, config: TwistConfig):
        """
        """
        # Configuration
        self.instrument_id = InstrumentId.from_str(config.instrument_id)
        self.bar_type = BarType.from_str(config.bar_type)
        self.fake_bi = config.fake_bi 
        self.bi_type =config.bi_type 
        self.fx_included = config.fx_included 
        self.zs_process_type = config.zs_process_type 
        self.zs_support_type = config.zs_support_type 

        self.macd_fast = config.macd_fast 
        self.macd_slow = config.macd_slow 
        self.macd_signal = config.macd_signal 
        self.rsi_period = config.rsi_period 
        self.boll_period = config.boll_period 
        self.boll_k = config.boll_k
        self.ma_period = config.ma_period 

        self.bar_capacity = config.bar_capacity
        self.twist_bar_capacity = config.twist_bar_capacity
        self.fx_capacity = config.fx_capacity
        self.bi_capacity = config.bi_capacity
        self.xd_capacity = config.xd_capacity
        self.trend_capacity = config.trend_capacity
        self.bi_zs_capacity = config.bi_zs_capacity
        self.xd_zs_capacity = config.xd_zs_capacity

        self.use_boll = config.use_boll
        self.use_ma = config.use_ma 
        self.use_rsi = config.use_rsi 

        self.macd = MovingAverageConvergenceDivergence(
            self.macd_fast,
            self.macd_slow,
            self.macd_signal
        )
        if self.use_boll:
            self.boll = BollingerBands(self.boll_period,self.boll_k)
        if self.use_ma:
            self.ma = ExponentialMovingAverage(self.ma_period) 
        if self.use_rsi:
            self.rsi = RelativeStrengthIndex(self.rsi_period)
        # 计算后保存的值
        self.newbars = []       # 整理后的原始K线
        self.twist_bars = []  # 缠论K线
        self.fxs = []             # 分型列表 
        self.real_fxs = []   
        self.bis = [] 
        self.xds = []
        self.big_trends = []
        self.bi_zss = []
        self.xd_zss = []
        self.bar_count= 0 

        self.macd_dif = [] 
        self.macd_dem = [] 
        self.macd_value = [] 
        self.boll_up = [] 
        self.boll_middle = [] 
        self.boll_down = [] 
        self.ma_value = [] 
        self.rsi_value = [] 

    def on_start(self,strategy: Strategy):
        """
        call from outside stratgies 
        """
        self.instrument = strategy.cache.instrument(self.instrument_id)
        if self.instrument is None:
            self.log.error(f"Could not find instrument for {self.instrument_id}")
            self.stop()
            return

        # Register the indicators for updating
        strategy.register_indicator_for_bars(self.bar_type, self.macd)
        if self.use_boll:
            strategy.register_indicator_for_bars(self.bar_type, self.boll) 
        if self.use_ma:
            strategy.register_indicator_for_bars(self.bar_type, self.ma)

        if self.use_rsi:
            strategy.register_indicator_for_bars(self.bar_type, self.rsi)

    def update_raw(
        self,
        ts_event,
        open,
        high,
        low,
        close,
        volume
    ):
        """
        Update the raw data
        """
        info = None 
        new_bar = NewBar(
            bar_type = self.bar_type,
            index = self.bar_count,
            ts_opened = ts_event,
            ts_closed =  ts_event,
            open = open,
            high = high,
            low = low,
            close = close,
            volume = volume,
            info = info,
        )
        self.newbars.append(new_bar) 
        ##update twist bars 
        self.process_twist_bar() 
        ##update fx 
        self.process_fx() 
        ##update bi 
        bi_update = self.process_bi()

        ##update xd from bi 
        xd_update = self.process_up_line(LineType.BI) if bi_update else False
        ##update big_trend 
        self.process_up_line(LineType.XD) if xd_update else False
        ## process bi and xd zs 
        self.process_zs(LineType.BI) if bi_update else None
        self.process_zs(LineType.XD) if xd_update else None 

        self.process_trade_point(LineType.BI) if bi_update else None
        self.process_trade_point(LineType.XD) if xd_update else None
        ##update bar_count 
        self.bar_count += 1
        

    def on_bar(self,bar: Bar):
        ## update newbar 
        ## update indciator 
        #self.memory_reduce()
        self.update_info(bar)
        self.update_raw(
            ts_event = bar.ts_event,
            open = bar.open.as_double(),
            high = bar.high.as_double(),
            low = bar.low.as_double(),
            close = bar.close.as_double(),
            volume = bar.volume.as_double(),
        )

    def update_info(self,bar: Bar):
        #self.macd.update_raw(bar.close.as_double()) 
        #self.boll.update_raw(bar.high.as_double(),bar.low.as_double(),bar.close.as_double())
        #self.ma.update_raw(bar.close.as_double()) 
        self.macd_dif.append(self.macd.dif)
        self.macd_dem.append(self.macd.dem)
        self.macd_value.append(self.macd.value)
        if self.use_boll:
            self.boll_up.append(self.boll.upper)
            self.boll_middle.append(self.boll.middle)
            self.boll_down.append(self.boll.lower)
        if self.use_ma:
            self.ma_value.append(self.ma.value)
        if self.use_rsi:
            self.rsi_value.append(self.rsi.value)

        return True 

    def process_twist_bar(self):
        """
        Aggregate ordinary bars in twist bars.
        """
        new_bar = self.newbars[-1] 
        if len(self.twist_bars) < 1:
            twist_bar = TwistBar(
                index = 0,
                b_index = new_bar.index,
                ts_opened = new_bar.ts_opened,
                ts_closed = new_bar.ts_closed,
                elements= [new_bar],
                open = new_bar.open,
                high = new_bar.high,
                low = new_bar.low,
                close = new_bar.close,
                volume = new_bar.volume,
                jump = False,
            )
            self.twist_bars.append(twist_bar)
            return True 
        if len(self.twist_bars) >=4:
            up_twist_bars = [self.twist_bars[-4],self.twist_bars[-3]]
        elif len(self.twist_bars) <=2:
            up_twist_bars = [] 
        else:
            up_twist_bars =[self.twist_bars[-3]] 
        twisit_bar_1 = self.twist_bars[-1]
        twisit_bar_2 = self.twist_bars[-2] if len(self.twist_bars) >=2 else None 

        raw_bars = twisit_bar_2.elements if twisit_bar_2  else [] 
        raw_bars.extend(twisit_bar_1.elements) 
        if new_bar.ts_opened != raw_bars[-1].ts_opened:
            raw_bars.append(new_bar) 

        post_twist_bars = self.bars_inlcuded(raw_bars,up_twist_bars) 
        if twisit_bar_2:
            self.twist_bars.pop() 
            self.twist_bars.pop() 
        else:
            self.twist_bars.pop() 
        ##add processed twist_bars 
        for item in post_twist_bars:
            if len(self.twist_bars) < 1:
                item.index = 0 
            else:
                item.index = self.twist_bars[-1].index + 1 
            self.twist_bars.append(item) 
        return True 

    def process_fx(self):
        """
        Aggregate twist bars into FX.
        """
        if len(self.twist_bars) < 3:
            return False 
        
        b1, b2, b3 = self.twist_bars[-3:]
        fx = None 
        if (b1.high < b2.high and b2.high > b3.high) and (b1.low < b2.low and b2.low > b3.low):
            jump = True if (b1.high < b2.low or b2.low > b3.high) else False 
            fx = FX(
                mark_type = Mark.DING,
                middle_twist_bar = b2,
                twist_bars = [b1,b2,b3],
                value = b2.high,
                jump = jump,
                real = True,
                is_confirm= True, 
            )
        if (b1.high > b2.high and b2.high < b3.high) and (b1.low > b2.low and b2.low < b3.low):
            jump = True  if (b1.low > b2.high or b2.high < b3.low) else False
            fx = FX(
                mark_type = Mark.DI,
                middle_twist_bar = b2,
                twist_bars = [b1,b2,b3],
                value = b2.low,
                jump = jump,
                real = True,
                is_confirm= True, 
            )
        
        if fx is None:
            ##check un_confirmed FX 
            if self.fake_bi:
                b1,b2 = self.twist_bars[-2:]
                b3 = None 
                if b2.high > b1.high:
                    fx = FX(
                        mark_type = Mark.DING,
                        middle_twist_bar = b2,
                        twist_bars = [b1,b2],
                        value = b2.high,
                        jump = False,
                        real = True,
                        is_confirm= False,  
                    )
                elif b2.low < b1.low:
                    fx = FX(
                        mark_type = Mark.DI,
                        middle_twist_bar = b2,
                        twist_bars = [b1,b2],
                        value = b2.low,
                        jump = False,
                        real = True,
                        is_confirm= False,  
                    )
                else:
                    return False 
            else:
                return False 

        if len(self.fxs) == 0 and fx.is_confirm is False:
            return False
        elif len(self.fxs) == 0 and fx.is_confirm is True:
            fx.index = 0 
            self.fxs.append(fx)
            return True    

        ##check if fx should be updated 
        is_update = False  
        end_fx = self.fxs[-1]
        if fx.ts_opened == end_fx.ts_opened:
            end_fx_index = end_fx.index
            self.fxs[-1] = fx 
            self.fxs[-1].index = end_fx_index
            is_update = True 
        
        
        up_fx = None
        # record max and min value in un_real fxs 
        fx_interval_high = None
        fx_interval_low = None
        for _fx in self.fxs[::-1]:
            if is_update and _fx.ts_opened == fx.ts_opened:
                continue 
            fx_interval_high = _fx.value if fx_interval_high is None else max(fx_interval_high, _fx.value) 
            fx_rang_low = _fx.value if fx_interval_low is None else min(fx_interval_low,_fx.value) 
            if _fx.real:
                up_fx = _fx 
                break 
        
        if up_fx is None:
            return False

        if self.bi_type == BIType.TB:
            if not is_update:
                fx.index = self.fxs[-1].index + 1 
                self.fxs.append(fx) 
            return True 

        
        #新增分型及其上一个真实分型 
        if fx.mark_type == Mark.DING and up_fx.mark_type == Mark.DING and up_fx.middle_twist_bar.high <= fx.middle_twist_bar.high:
            up_fx.real = False 
        elif fx.mark_type == Mark.DI and up_fx.mark_type == Mark.DI and up_fx.middle_twist_bar.low >= fx.middle_twist_bar.low:
            up_fx.real = False
        elif fx.mark_type == up_fx.mark_type:
            fx.real = False  ## continue fx, DING - prev_high > last_high ,drop last 
        elif fx.mark_type == Mark.DING and up_fx.mark_type == Mark.DI and \
            (
                fx.middle_twist_bar.high <= up_fx.middle_twist_bar.low 
                or fx.middle_twist_bar.low <= up_fx.middle_twist_bar.high 
                or (not self.fx_included  and fx.high() < up_fx.high())
        ):
            fx.real = False 
        elif fx.mark_type == Mark.DI and up_fx.mark_type == Mark.DING and \
            (
                fx.middle_twist_bar.low >= up_fx.middle_twist_bar.high 
                or fx.middle_twist_bar.high >= up_fx.middle_twist_bar.low 
                or (not self.fx_included and fx.low() > up_fx.low())
        ):
            fx.real = False 
        else:
            if self.bi_type == BIType.OLD and fx.middle_twist_bar.index - up_fx.middle_twist_bar.index < 4:
                fx.real = False 
            if self.bi_type == BIType.NEW and (fx.middle_twist_bar.index - up_fx.middle_twist_bar.index < 3 \
                or fx.middle_twist_bar.elements[-1].index - up_fx.middle_twist_bar.elements[-1].index < 4):
                fx.real = False 
        if not is_update:
            fx.index = self.fxs[-1].index + 1 
            self.fxs.append(fx)
        return True
        
    def process_bi(self):
        """
        Aggregate FXs into bis.
        """
        if len(self.fxs) == 0:
            return False


        if len(self.bis) > 0 and  not self.bis[-1].start.real:
            self.bis.pop()
        
        bi = self.bis[-1] if len(self.bis) > 0 else None 
        ##check bi  pause 
        if bi:
            close = self.newbars[-1].close 
            if bi.is_confirm and bi.direction_type == Direction.UP and close < bi.end.twist_bars[-1].low:
                bi.pause = True 
            elif bi.is_confirm and bi.direction_type == Direction.DOWN and close > bi.end.twist_bars[-1].high:
                bi.pause = True 
            else:
                bi.pause = False 

        if bi is None: ## the first time to generate bi 
            real_fx = [_fx for _fx in self.fxs if _fx.real]
            if len(real_fx) < 2:
                return False 
            for fx in real_fx:
                if bi is None:
                    bi = BI(start = fx, index = 0) 
                    continue 
                if bi.start.mark_type == fx.mark_type:
                    continue 
                bi.end = fx 
                bi.direction_type = Direction.UP if bi.start.mark_type == Mark.DI else Direction.DOWN 
                bi.is_confirm = fx.is_confirm  
                bi.pause = False 
                self.process_line_power(bi)
                self.process_line_hl(bi)
                self.bis.append(bi)
                return True 

        # 确定最后一个有效分型
        end_real_fx = None
        for _fx in self.fxs[::-1]:
            if _fx.real:
                end_real_fx = _fx
                break
        if (bi.end.real is False and bi.end.mark_type == end_real_fx.mark_type):
            #or  (bi.end.index == end_real_fx.index and bi.is_confirm != end_real_fx.is_confirm):
            bi.end = end_real_fx 
            bi.is_confirm = end_real_fx.is_confirm 
            self.process_line_power(bi)
            self.process_line_hl(bi)
            return True 

        if bi.end.index < end_real_fx.index and bi.end.mark_type != end_real_fx.mark_type:
            # new bi generate 
            new_bi = BI(start=bi.end, end=end_real_fx)
            new_bi.index = self.bis[-1].index + 1
            new_bi.direction_type = Direction.UP if new_bi.start.mark_type == Mark.DI else Direction.DOWN 
            new_bi.is_confirm = end_real_fx.is_confirm 
            new_bi.pause  = False 
            self.process_line_power(new_bi)
            self.process_line_hl(new_bi)
            self.bis.append(new_bi)
            return True 

        return False

    def process_up_line(
        self,
        base_line_type = LineType.BI,
    ):
        """
        Aggregate bis into XLFX and XD.
        """
        is_update = False
        if base_line_type == LineType.BI:
            up_lines = self.xds
            base_lines = self.bis
        elif base_line_type == LineType.XD:
            up_lines = self.big_trends
            base_lines = self.xds
        else:
            raise ('high level xd name is wrong：%s' % base_line_type)

        if len(base_lines) == 0:
            return False
        ##first time update XD 
        if len(up_lines) == 0:
            bi_0 = base_lines[0] 
            start_fx = XLFX(
                mark_type= Mark.DI if bi_0.direction_type == Direction.UP else Mark.DING,
                high = bi_0.high,
                low = bi_0.low,
                line = bi_0, 
            )
            end_fx = None 
            if start_fx.mark_type == Mark.DI:
                dis = self.cal_line_xlfx(base_lines, Mark.DI)
                for di in dis:
                    if di.line.index > start_fx.line.index: 
                        start_fx = di 
                dings = self.cal_line_xlfx(base_lines[start_fx.line.index:], Mark.DING) 
                for ding in dings:
                    if ding.line.index - start_fx.line.index >= 2:
                        ## general new XD 
                        end_fx = ding
                        break
            elif start_fx.mark_type == Mark.DING:
                dings = self.cal_line_xlfx(base_lines, Mark.DING)
                for ding in dings:
                    if ding.line.index > start_fx.line.index:
                        start_fx = ding 
                dis  = self.cal_line_xlfx(base_lines[start_fx.line.index:], Mark.DI)
                for di in dis:
                    if di.line.index - start_fx.line.index >=2:
                        end_fx = di 

            if start_fx and end_fx:
                start_line = start_fx.line 
                end_line = base_lines[end_fx.line.index-1]
                new_up_line = XD(
                    start = start_line.start,
                    end = end_line.end,
                    start_line= start_line,
                    end_line = end_line,
                    direction_type = Direction.UP if end_fx.mark_type == Mark.DING else Direction.DOWN,
                    ding_fx = start_fx if start_fx.mark_type == Mark.DING else end_fx,
                    di_fx = start_fx if start_fx.mark_type == Mark.DI else end_fx,
                    is_confirm= end_fx.is_confirm,
                )
                self.process_line_power(new_up_line)
                self.process_line_hl(new_up_line)
                up_lines.append(new_up_line)
                return True
            else:
                return False


        ## generally update XD 
        up_line = up_lines[-1]
        ## if extened 
        if up_line.direction_type == Direction.UP:
            dings = self.cal_line_xlfx(base_lines[up_line.start_line.index:],Mark.DING) 
            for ding in dings:
                if ding.line.index >=up_line.end_line.index:
                    end_line = base_lines[ding.line.index - 1]
                    up_line.end = end_line.end 
                    up_line.end_line = end_line 
                    up_line.ding_fx = ding 
                    up_line.is_confirm = ding.is_confirm 
                    self.process_line_power(up_line)
                    self.process_line_hl(up_line)
                    is_update = True 
        elif up_line.direction_type == Direction.DOWN:
            dis = self.cal_line_xlfx(base_lines[up_line.start_line.index:], Mark.DI)
            for di in dis:
                if di.line.index >= up_line.end_line.index:
                    end_line = base_lines[di.line.index - 1]
                    up_line.end = end_line.end
                    up_line.end_line = end_line 
                    up_line.di_fx = di 
                    up_line.is_confirm = di.is_confirm 
                    self.process_line_power(up_line)
                    self.process_line_hl(up_line)
                    is_update = True

        ##check if has inverse-direction XLFX to generate new XD 
        if up_line.direction_type == Direction.UP:
            dis = self.cal_line_xlfx(base_lines[up_line.end_line.index+1:],Mark.DI)
            for di in dis:
                if di.line.index - up_line.end_line.index >=2 :
                    start_line = base_lines[up_line.end_line.index+1]
                    end_line = base_lines[di.line.index-1]
                    new_up_line = XD(
                        start= start_line.start,
                        end = end_line.end,
                        start_line= start_line,
                        end_line = end_line,
                        direction_type= Direction.DOWN,
                        ding_fx= up_line.ding_fx,
                        di_fx = di,
                        index = up_line.index + 1,
                        is_confirm= di.is_confirm,
                    )
                    self.process_line_power(new_up_line)
                    self.process_line_hl(new_up_line)
                    # two DD uncomplete 
                    up_line.is_confirm = True 
                    up_lines.append(new_up_line)
                    is_update = True
                    break
        elif up_line.direction_type == Direction.DOWN:
            dings = self.cal_line_xlfx(base_lines[up_line.end_line.index + 1:], Mark.DING)
            for ding in dings:
                if ding.line.index - up_line.end_line.index >= 2: 
                    start_line = base_lines[up_line.end_line.index + 1]
                    end_line = base_lines[ding.line.index - 1]
                    new_up_line = XD(
                        start=start_line.start,
                        end=end_line.end,
                        start_line=start_line,
                        end_line=end_line,
                        direction_type= Direction.UP,
                        ding_fx=ding,
                        di_fx=up_line.di_fx,
                        index = up_line.index + 1,
                        is_confirm= ding.is_confirm,
                    )
                    self.process_line_power(new_up_line)
                    self.process_line_hl(new_up_line)
                    ## two DD uncomplete 
                    up_line.is_confirm = True
                    up_lines.append(new_up_line)
                    is_update = True
                    break

        return is_update



    def process_zs(
        self,
        run_type: LineType,
    ):
        """
        generate zss.
        """
        if run_type is None:
            return False
        if run_type == LineType.BI:
            lines = self.bis
            up_lines = self.xds
            zss = self.bi_zss
        elif run_type == LineType.XD:
            lines = self.xds
            up_lines = self.big_trends
            zss = self.xd_zss
        else:
            raise Exception('error zs run_type as %s' % run_type)

        if len(lines) < 4:
            return False
        if self.zs_process_type == ZSProcessType.INTERATE:
            self.process_interate_zs(lines, zss, run_type)
        elif self.zs_process_type == ZSProcessType.INSIDE:
            self.process_inside_zs(lines, up_lines, zss, run_type)
        else:
            raise Exception('error zs run_type as %s' % run_type)

        return True


    def process_trade_point(
        self, 
        run_type=None
        ):
        """
        compute Divergence and TradePoints
        """
        if run_type is None:
            return False

        if run_type == LineType.BI:
            lines: List[BI] = self.bis
            zss: List[ZS] = self.bi_zss
        elif run_type == LineType.XD:
            lines: List[XD] = self.xds
            zss: List[ZS] = self.xd_zss
        else:
            raise Exception('trade point based line_type error ：%s' % run_type)

        if len(zss) == 0:
            return True

        line = lines[-1]
        # clear trade points and recompute 
        line.divergences = {}
        line.trade_points = {} 

        # add bi divergence 
        if run_type == LineType.BI:
            line.add_divergence(DivergenceType.BI, None, lines[-3], self.divergence_line(lines[-3], line))
        elif run_type == LineType.XD:
            line.add_divergence(DivergenceType.XD, None, lines[-3], self.divergence_line(lines[-3], line))
        
        # find all zss end with current line 
        line_zss: List[ZS] = [
            _zs for _zs in zss
            if (_zs.lines[-1].index == line.index and _zs.real and _zs.level == 0)
        ]
        for _zs in line_zss:
            line.add_divergence(DivergenceType.OSCILLATION, _zs, _zs.lines[0], self.divergence_oscillation(_zs, line))
            line.add_divergence(DivergenceType.TREND, _zs, _zs.lines[0], self.divergence_trend(zss, _zs, line))

        #  trend divergence (1buy,1sell)
        for divergence_type,bcs in line.divergences.items():
            for bc in bcs:
                if bc.divergence_type == DivergenceType.TREND and bc.is_divergence:
                    if line.direction_type == Direction.UP:
                        line.add_trade_point(TradePointType.OneSell, bc.zs)
                    if line.direction_type == Direction.DOWN:
                        line.add_trade_point(TradePointType.OneBuy, bc.zs)

        # 2buy,2sell, two bi same direction,lst_bi break,new bi get back or divergence 
        for _zs in line_zss:
            if len(_zs.lines) < 7:
                continue
            tx_line: Union[BI, XD] = _zs.lines[-3]
            if _zs.lines[0].direction_type == Direction.UP and line.direction_type == Direction.UP:
                if tx_line.high == _zs.gg and (tx_line.high > line.high or line.divergence_type_exists([DivergenceType.OSCILLATION,DivergenceType.TREND])):
                    line.add_trade_point(TradePointType.TwoSell, _zs)
            if _zs.lines[0].direction_type == Direction.DOWN and line.direction_type == Direction.DOWN:
                if tx_line.low == _zs.dd and (tx_line.low < line.low or line.divergence_type_exists([DivergenceType.OSCILLATION,DivergenceType.TREND])):
                    line.add_trade_point(TradePointType.TwoBuy, _zs)

        # l2buy,l2sell, When first bir is (2buy,2sell) and leave power is weaker than back power
        for _zs in line_zss:
            # if ZS has invere trade_points or divergence trade_point,then no l2 trade_points 
            have_buy = False
            have_sell = False
            have_bc = False
            for _line in _zs.lines[:-1]:
                if _line.trade_type_exists([
                    TradePointType.OneBuy,
                    TradePointType.TwoBuy,
                    TradePointType.ThreeBuy,
                    TradePointType.L2Buy,
                    TradePointType.L3Buy,
                ]):
                    have_buy = True
                if _line.trade_type_exists([
                    TradePointType.OneSell,
                    TradePointType.TwoSell,
                    TradePointType.ThreeSell,
                    TradePointType.L2Sell,
                    TradePointType.L3Sell,
                    ]):
                    have_sell = True
                if _line.divergence_type_exists([DivergenceType.OSCILLATION, DivergenceType.TREND]):
                    have_bc = True
            if TradePointType.TwoBuy in _zs.lines[1].trade_points.keys() and line.direction_type == Direction.DOWN:
                if have_sell is False and have_bc is False and self.compare_power_divergence(_zs.lines[1].power, line.power):
                    line.add_trade_point(TradePointType.L2Buy, _zs)
            if TradePointType.TwoSell in _zs.lines[1].trade_points.keys() and line.direction_type == Direction.UP:
                if have_buy is False and have_bc is False and self.compare_power_divergence(_zs.lines[1].power, line.power):
                    line.add_trade_point(TradePointType.L2Sell, _zs)

        # 3buy,3sell, ZS's end line is just the previous of the last bi.
        line_3trade_point_zss: List[ZS] = [
            _zs for _zs in zss
            if (_zs.lines[-1].index == line.index - 1 and _zs.real and _zs.level == 0)
        ]
        for _zs in line_3trade_point_zss:
            if len(_zs.lines) < 5:
                continue
            if line.direction_type == Direction.UP and line.high < _zs.zd:
                line.add_trade_point(TradePointType.ThreeSell, _zs)
            if line.direction_type == Direction.DOWN and line.low > _zs.zg:
                line.add_trade_point(TradePointType.ThreeBuy, _zs)

        # l3buy,l3sell trade point 
        for _zs in line_zss:
            # if ZS has invere trade_points or divergence trade_point,then no l3 trade_points 
            have_buy = False
            have_sell = False
            have_bc = False
            for _line in _zs.lines[:-1]:
                # 不包括当前笔
                if _line.trade_type_exists([
                    TradePointType.OneBuy, 
                    TradePointType.TwoBuy, 
                    TradePointType.L2Buy, 
                    TradePointType.ThreeBuy, 
                    TradePointType.L3Buy
                ]):
                    have_buy = True
                if _line.trade_type_exists([
                    TradePointType.OneSell,
                     TradePointType.TwoSell, 
                     TradePointType.L2Sell, 
                     TradePointType.ThreeSell, 
                     TradePointType.L3Sell
                ]):
                    have_sell = True
                if _line.divergence_type_exists([DivergenceType.OSCILLATION, DivergenceType.TREND]):
                    have_bc = True
            for side_type,trade_points in _zs.lines[1].trade_points.items():
                for trade_point in trade_points:
                    if trade_point.side_type == TradePointType.ThreeBuy:
                        if have_sell is False and have_bc is False and line.direction_type == Direction.DOWN \
                                and line.low > trade_point.zs.zg \
                                and self.compare_power_divergence(_zs.lines[0].power, line.power):
                            line.add_trade_point(TradePointType.L3Buy, trade_point.zs)
                    if trade_point.side_type == TradePointType.ThreeSell:
                        if have_buy is False and have_bc is False and line.direction_type == Direction.UP \
                                and line.high < trade_point.zs.zd \
                                and self.compare_power_divergence(_zs.lines[0].power, line.power):
                            line.add_trade_point(TradePointType.L3Sell, trade_point.zs)

        return True



    def process_inside_zs(
        self, 
        lines: List[LINE], 
        up_lines: List[XD], 
        zss: List[ZS], 
        run_type: LineType,
    ):
        """
        compute dn zs 
        """
        un_real_zs_index =[] 
        if len(up_lines) >= 2:
            _up_l = up_lines[-2]
            _run_lines = lines[_up_l.start_line.index:_up_l.end_line.index + 1]
            _up_zss = [_zs for _zs in zss if _up_l.start.index <= _zs.start.index < _up_l.end.index]
            _new_zss = self.create_inside_zs(run_type, _run_lines)
            for _u_zs in _up_zss:
                if _u_zs.start.index not in [_z.start.index for _z in _new_zss]:
                    _u_zs.real = False
                    un_real_zs_index.append(_u_zs.index)
                    continue
                for _n_zs in _new_zss:
                    if _u_zs.start.index == _n_zs.start.index:
                        self.__copy_zs(_n_zs, _u_zs)
                        _u_zs.is_confirm = True
                if not _u_zs.real:
                    un_real_zs_index.append(_u_zs.index)

        # compute dn zs 
        run_lines: List[LINE]
        if len(up_lines) == 0:
            run_lines = lines
        else:
            run_lines = lines[up_lines[-1].start_line.index:]

        exists_zs = [_zs for _zs in zss if _zs.start.index >= run_lines[0].start.index]
        new_zs = self.create_inside_zs(run_type, run_lines)
        #update or remove zs 
        for _ex_zs in exists_zs:
            if _ex_zs.start.index not in [_z.start.index for _z in new_zs]:
                _ex_zs.real = False
                un_real_zs_index.append(_ex_zs.index)
                continue
            for _n_zs in new_zs:
                if _n_zs.start.index == _ex_zs.start.index:
                    self.__copy_zs(_n_zs, _ex_zs)
            if not _ex_zs.real:
                un_real_zs_index.append(_ex_zs.index)
        # get real zs 
        if len(un_real_zs_index) > 0:
            start_index = min(un_real_zs_index)
            back_real_zs = [_zs for _zs in zss[start_index:] if _zs.real] 
            if run_type == LineType.BI:
                self.bi_zss =  self.bi_zss[:start_index]
                zss = self.bi_zss
            elif run_type == LineType.XD:
                self.xd_zss = self.xd_zss[:start_index]
                zss = self.xd_zss 
            for zs in back_real_zs:
                zs.index = zss[-1].index + 1
                zss.append(zs)
        ## add new zs 
        for _n_zs in new_zs:
            if _n_zs.start.index not in [_z.start.index for _z in exists_zs]:
                _n_zs.index = zss[-1].index + 1 if len(zss) > 0 else 0
                zss.append(_n_zs)
        return

    def process_interate_zs(
        self,
        lines: List[LINE], 
        zss: List[ZS], 
        run_type: LineType,
    ):
        """
        compute BL ZS.
        """
        un_real_zs_index =[] 
        if len(zss) == 0:
            _ls = lines[-4:]
            _zs = self.create_zs(run_type, None, _ls)
            if _zs:
                zss.append(_zs)
            return True

        line = lines[-1]
        start_zs_index = len(zss) - 1
        start_line_index = zss[-1].lines[0].index 
        for idx,_zs in enumerate(zss):
            if not _zs.is_confirm:
                start_zs_index = idx 
                start_line_index = _zs.lines[0].index 
                break 
        run_lines = lines[start_line_index:]
        new_zs = self.create_inside_zs(run_type, run_lines)

        if run_type == LineType.BI:
            self.bi_zss =  self.bi_zss[:start_zs_index]
            zss = self.bi_zss
        elif run_type == LineType.XD:
            self.xd_zss = self.xd_zss[:start_zs_index]
            zss = self.xd_zss 
        for  _n_zs in new_zs:
            _n_zs.index = zss[-1].index + 1 if len(zss) > 0 else 0
            zss.append(_n_zs)

        return True


    def create_zs(
        self, 
        run_type: LineType, 
        zs: Union[ZS, None], 
        lines: List[LINE],
    ) -> Union[ZS, None]:
        """
        if pass zs:update its property 
        if not pass zs:create a new zs 
        """
        if len(lines) <= 3:
            return None
    

        run_lines = []
        zs_confirm  = False

        ## record overlap of zs 
        _high,_low = self.line_hl(lines[0]) 

        cross = self.cross_interval(
            self.line_hl(lines[1]),
            self.line_hl(lines[3])
        )

        if not cross:
            return None 
        else:
            interval_high,interval_low = cross
   
        for _l in lines:
            #get all lines overlap to zs,once not overlap the zs has finished.
            _l_hl = self.line_hl(_l)
            if self.cross_interval(
                [interval_high,interval_low],
                _l_hl,
            ):
                _high = max(_high, _l_hl[0]) 
                _low = min(_low, _l_hl[1])
                run_lines.append(_l)
            else:
                zs_confirm = True 
                break 
        if len(run_lines) < 4:
            return None 

        _last_line = run_lines[-1] 
        _last_hl = self.line_hl(_last_line) 
        last_line_in_zs = True 
        if (_last_line.direction_type == Direction.UP and _last_hl[0] == _high) \
            or (_last_line.direction_type == Direction.DOWN and _last_hl[1] == _low):
            #if the last line is the highest or lowest,it do not belong to zs 
            last_line_in_zs = False 

        if zs is None:
            zs = ZS(
                zs_type = run_type,
                start = run_lines[1].start,
                direction_type = Direction.OSCILLATION,
            )
        zs.is_confirm = zs_confirm 
   
        zs.lines = []
        zs.add_line(run_lines[0])
        zs_range = [interval_high, interval_low ]
        zs_gg = run_lines[1].high 
        zs_dd = run_lines[1].low 

        for i in range(1,len(run_lines)):
            _l = run_lines[i] 
            _l_hl = self.line_hl(_l) 
            cross_range = self.cross_interval(zs_range,_l_hl)
            if not cross_range:
                raise Exception("A ZS must have overlap interval")

            if i == len(run_lines) - 1 and last_line_in_zs is False:
                # the last line and it is not incuded in zs 
                pass
            else:
                zs_gg = max(zs_gg, _l_hl[0])
                zs_dd = min(zs_dd, _l_hl[1])
                # compute level based on line_num 
                zs.line_num = len(zs.lines) - 1
                zs.level = int(zs.line_num / 9)
                zs.end = _l.end
                # record max power 
                if zs.max_power is None:
                    zs.max_power = _l.power 
                elif _l.power:
                    zs.max_power = zs.max_power if self.compare_power_divergence(zs.max_power, _l.power) else _l.power
            zs.add_line(_l)

        zs.zg = zs_range[0]
        zs.zd = zs_range[1]
        zs.gg = zs_gg
        zs.dd = zs_dd
        zs.high_supported = zs.zg
        zs.low_supported = zs.zd
        # compute zs direction 

        if zs.lines[0].direction_type == zs.lines[-1].direction_type:
            _l_start_hl = self.line_hl(zs.lines[0])
            _l_end_hl = self.line_hl(zs.lines[-1])
            if zs.lines[0].direction_type == Direction.UP and _l_start_hl[1] <= zs.dd and _l_end_hl[0] >= zs.gg:
                zs.direction_type= zs.lines[0].direction_type
            elif zs.lines[0].direction_type == Direction.DOWN and _l_start_hl[0] >= zs.gg and _l_end_hl[1] <= zs.dd:
                zs.direction_type = zs.lines[0].direction_type
            else:
                zs.direction_type = Direction.OSCILLATION
        else:
            zs.direction_type = Direction.OSCILLATION

        return zs

    def create_inside_zs(
        self, 
        run_type: LineType,
        lines: List[LINE]
    ) -> List[ZS]:
        """
        compute dn zs.
        """
        zss: List[ZS] = []
        if len(lines) <= 4:
            return zss

        start = 0
        while True:
            run_lines = lines[start:]
            if len(run_lines) == 0:
                break
            zs = self.create_zs(run_type, None, run_lines)
            if zs is None:
                start += 1
            else:
                zss.append(zs)
                start += len(zs.lines) - 1

        return zss



    def process_line_power(
        self, 
        line: LINE,
        key: str = 'macd',
        ):
        """
        process Line power 
        """
        line.power = {
            key: self.query_macd_power(line.start, line.end)
        }
        return True 

    def line_hl(
        self,
        line: LINE
    ):
        if self.zs_support_type == SupportType.HL:
            return [line.high,line.low]
        else:
            return [line.top_high(),line.bottom_low()]

    def process_line_hl(self, line: LINE):
        """
        process line real high low point 
        """
        start = line.start.middle_twist_bar.b_index
        end = line.end.middle_twist_bar.b_index+1
        assert start < end 
        fx_bars = self.newbars[start:end]
        b_h = [_b.high for _b in fx_bars]
        b_l = [_b.low for _b in fx_bars]
        line.high = np.array(b_h).max()
        line.low = np.array(b_l).min()
        line.vwap = np.sum([_b.close * _b.volume for _b in fx_bars]) / np.sum([_b.volume for _b in fx_bars])
        # process high_rsi and low_rsi 
        if self.use_rsi:
            line._high_rsi = np.max(self.rsi_value[start:end])
            line._low_rsi = np.min(self.rsi_value[start:end])
        return True 

    def query_macd_power(self, start_fx: FX, end_fx: FX):
        if start_fx.ts_opened > end_fx.ts_opened:
            raise Exception("start_fx start time should small than end_fx's start time ")

        start_index = start_fx.middle_twist_bar.b_index
        end_index = end_fx.middle_twist_bar.b_index + 1
        assert start_index < end_index 
        dif = np.array(self.macd_dif[start_index:end_index])
        dem = np.array(self.macd_dem[start_index:end_index])
        macd_value = np.array(self.macd_value[start_index:end_index])
        if len(macd_value) == 0:
            macd_value = np.array([0])
        if len(dem) == 0:
            dem = np.array([0])
        if len(dif) == 0:
            dif = np.array([0])

        macd_value_abs = abs(macd_value)
        macd_value_up = macd_value.clip(0,None)
        macd_value_down = macd_value.clip(None,0)
        macd_value_sum = macd_value_abs.sum()
        macd_value_up_sum = macd_value_up.sum()
        macd_value_down_sum = macd_value_down.sum()
        end_dem = dem[-1]
        end_dif = dif[-1]
        end_macd_value = macd_value[-1]
        return dict(
            dem = dict(end = end_dem, max = np.max(dem), min = np.min(dem)),
            dif = dict(end = end_dif, max = np.max(dif), min = np.min(dif)),
            macd_value = dict(
                sum =macd_value_sum, 
                up_sum = macd_value_up_sum, 
                down_sum = macd_value_down_sum,
                end =  end_macd_value),
            macd_dif = dict(
                sum = abs(dif).sum(),
                up_sum = dif.clip(0,None).sum(),
                down_sum = dif.clip(None,0).sum(),
            )
        )


    def divergence_line(
        self, 
        pre_line: LINE, 
        now_line: LINE
    ):
        """
        compare if there has a divergence between two lines 
        """
        if pre_line.direction_type != now_line.direction_type:
            return False 
        if pre_line.direction_type == Direction.UP and now_line.high < pre_line.high:
            return False 
        if pre_line.direction_type == Direction.DOWN and now_line.low > pre_line.low:
            return False 

        return self.compare_power_divergence(pre_line.power, now_line.power)

    def divergence_oscillation(
        self, zs: ZS, 
        now_line: LINE
    ):
        """
        decied if the ZS is in pz divergence 
        """
        if zs.lines[-1].index != now_line.index:
            return False
        if zs.direction_type not in [Direction.UP,Direction.DOWN]:
            return False 
 
        return self.compare_power_divergence(zs.lines[0].power, now_line.power)

    def divergence_trend(
        self, 
        zss: List[ZS], 
        zs: ZS, 
        now_line: LINE
    ):
        """
        decide  if the ZS us in trend divergence 
        """
        if zs.direction_type not in [Direction.UP,Direction.DOWN]:
            return False 

        # check if has same diretion ZS 
        pre_zs = [
            _zs for _zs in zss
            if (_zs.lines[-1].index == zs.lines[0].index and _zs.direction_type == zs.direction_type and _zs.level == zs.level)
        ]
        if len(pre_zs) == 0:
            return False
        # if high and low overlaped 
        pre_overlap_zs = []
        for _zs in pre_zs:
            if (_zs.direction_type == Direction.UP and _zs.gg < zs.dd) or (_zs.direction_type == Direction.DOWN and _zs.dd > zs.gg):
                pre_overlap_zs.append(_zs)

        if len(pre_overlap_zs) == 0:
            return False

        return self.compare_power_divergence(zs.lines[0].power, now_line.power)

    def macd_divergence(
        self, 
        pre_line: LINE, 
        now_line: LINE
    ):
        """
        compare if there has a divergence between two lines 
        """
        pre_line_dif_change = self.macd_dif[pre_line.end.middle_twist_bar.b_index] - self.macd_dif[pre_line.start.middle_twist_bar.b_index]
        now_line_dif_change = self.macd_dif[now_line.end.middle_twist_bar.b_index] - self.macd_dif[now_line.start.middle_twist_bar.b_index]
        if pre_line.direction_type != now_line.direction_type:
            return False 
        if pre_line.direction_type == Direction.UP:
            if now_line.high > pre_line.high:
                return False 
            if pre_line_dif_change > 0 or now_line_dif_change > 0:
                return False 
            if abs(pre_line_dif_change) < abs(now_line_dif_change):
                return False 
        if pre_line.direction_type == Direction.DOWN:
            if  now_line.low < pre_line.low:
                return False 
            if pre_line_dif_change < 0 or now_line_dif_change < 0:
                return False 
            if abs(pre_line_dif_change) < abs(now_line_dif_change):
                return False 
        return True 

    def trend_channel_model(
        self,
        base_line_type = LineType.XD,
        line_num: int = 5,
    ):
        if base_line_type == LineType.BI:
            base_lines = self.bis
        elif base_line_type == LineType.XD:
            base_lines = self.xds
        else:
            raise ('high level xd name is wrong：%s' % base_line_type)


        trend_lines = []
        for i in range(1,len(base_lines)):
            bi = base_lines[-i]
            if bi.is_confirm:
                trend_lines.append(bi)
            if len(trend_lines) == line_num:
                break
        if len(trend_lines) < line_num:
            raise ('trend line init has not complete!')

        line_highs = [{"val": line.high ,"index":line.end.middle_twist_bar.b_index} for line in trend_lines if line.direction_type == Direction.UP]
        line_lows = [{"val": line.low ,"index":line.end.middle_twist_bar.b_index} for line in trend_lines if line.direction_type == Direction.DOWN]

        line_highs = sorted(line_highs, key=lambda v: v['val'], reverse=True)
        line_lows = sorted(line_lows, key=lambda v: v['val'], reverse=False)

        X_up = np.array([item["index"] for item in line_highs]).reshape(-1,1)
        Y_up = np.array([item["val"] for item in line_highs])
        up_model = LinearRegression().fit(X_up,Y_up)

        X_down = np.array([item["index"] for item in line_lows]).reshape(-1,1)
        Y_down = np.array([item["val"] for item in line_lows])
        down_model = LinearRegression().fit(X_down,Y_down)

        #predict_up = up_model.predict(np.array([self.newbars[-1].index]).reshape(-1,1))[0] 
        #predict_down = down_model.predict(np.array([self.newbars[-1].index]).reshape(-1,1))[0]     

        return up_model,down_model   

   
    @staticmethod
    def bars_inlcuded(
        newbars: List[NewBar], 
        up_twist_bars: List[TwistBar]
    ) -> List[TwistBar]:
        """
        Aggregate ordinary bars in twist bars.

        Parameters
        ----------
        newbars : List[NewBar]
            The original bars.
        up_twist_bars : List[TwistBar]
            The Third and fourth twist bars away from current time.
        """
        twist_bars = [] 
        twist_bar = TwistBar(
                index = 0, ##will be replaced,no worry 
                b_index = newbars[0].index,
                ts_opened = newbars[0].ts_opened,
                ts_closed = newbars[0].ts_closed,
                elements = [newbars[0]],
                open = newbars[0].open,
                high = newbars[0].high,
                low = newbars[0].low,
                close = newbars[0].close,
                volume = newbars[0].volume,
                jump = False,
            )
        twist_bars.append(twist_bar)
        up_twist_bars.append(twist_bar)

        for i in range(1,len(newbars)):
            twist_b = twist_bars[-1]
            newbar = newbars[i] 
            if (twist_b.high >= newbar.high and twist_b.low <= newbar.low) or (twist_b.high <= newbar.high and twist_b.low >= newbar.low):
                ## direct aggregate 
                if len(up_twist_bars) < 2:
                    #twist_b.index = twist_b.index 
                    twist_b.high = max(twist_b.high, newbar.high) 
                    twist_b.low = min(twist_b.low, newbar.low)
                else: 
                    #up direction 
                    if up_twist_bars[-2].high < twist_b.high:
                        twist_b.b_index = twist_b.b_index  if twist_b.high > newbar.high else newbar.index  
                        twist_b.high = max(twist_b.high, newbar.high) 
                        twist_b.low = max(twist_b.low, newbar.low)
                        twist_b.previous_trend = Direction.UP 
                    else:
                        twist_b.b_index = twist_b.b_index  if twist_b.low < newbar.low else newbar.index 
                        twist_b.high = min(twist_b.high, newbar.high) 
                        twist_b.low = min(twist_b.low, newbar.low)
                        twist_b.previous_trend = Direction.DOWN 

                twist_b.ts_opened = twist_b.ts_opened
                twist_b.ts_closed = newbar.ts_closed 
                twist_b.open = twist_b.open 
                twist_b.close = newbar.close 
                twist_b.volume = twist_b.volume + newbar.volume 
                twist_b.elements.append(newbar) 
            else:
                twist_bar = TwistBar(
                    index= 0,
                    b_index = newbar.index,
                    ts_opened = newbar.ts_opened,
                    ts_closed = newbar.ts_closed,
                    elements = [newbar],
                    open = newbar.open,
                    high = newbar.high,
                    low = newbar.low,
                    close = newbar.close,
                    volume = newbar.volume,
                    jump = False,
                )
                twist_bars.append(twist_bar)
                up_twist_bars.append(twist_bar) 
                
        return  twist_bars 

    @staticmethod
    def cal_line_xlfx(
        lines: List[LINE],
        fx_type= Mark.DING,
    ) -> List[XLFX]:
        """
        use line high low point two compute XLFXS
        """
        sequence= []
        for line in lines:
            if (fx_type == Mark.DING and line.direction_type == Direction.DOWN) or (fx_type == Mark.DI and line.direction_type == Direction.UP):
                now_xl = TZXL(
                        high = line.top_high(),
                        low = line.bottom_low(),
                        line = line,
                        line_broken= False,
                )
                if len(sequence) == 0:
                    sequence.append(now_xl)
                    continue 

                trend = Direction.UP if fx_type == Mark.DING else Direction.DOWN 
                up_xl = sequence[-1] 

                if up_xl.high >= now_xl.high and up_xl.low <=now_xl.low:
                    if trend == Direction.UP:
                        now_xl.line = now_xl.line if now_xl.high >= up_xl.high else up_xl.line 
                        now_xl.high = max(up_xl.high,now_xl.high) 
                        now_xl.low = max(up_xl.low,now_xl.low)
                    else:
                        now_xl.line = now_xl.line if now_xl.low <= up_xl.low else up_xl.line 
                        now_xl.high = min(up_xl.high,now_xl.high) 
                        now_xl.low = min(up_xl.low,now_xl.low) 
                    sequence.pop() 
                    sequence.append(now_xl) 
                elif up_xl.high < now_xl.high and up_xl.low > now_xl.low:
                    #strong included ,current xl include front xl
                    now_xl.line_broken = True 
                    sequence.append(now_xl)
                else:
                    sequence.append(now_xl)
        xlfxs: List[XLFX] = [] 
        for i in range(1,len(sequence)):
            up_xl = sequence[i-1]
            now_xl = sequence[i] 
            if len(sequence) > (i+1):
                next_xl = sequence[i+1]
            else:
                next_xl = None 

            jump = True if up_xl.high < now_xl.low or up_xl.low > now_xl.high else False 

            if next_xl:
                fx_high = max(up_xl.high, now_xl.high, next_xl.high)
                fx_low = min(up_xl.low, now_xl.low, next_xl.low)

                if fx_type == Mark.DING and up_xl.high < now_xl.high and now_xl.high > next_xl.high:
                    now_xl.mark_type = Mark.DING 
                    xlfxs.append(
                        XLFX(
                            mark_type =Mark.DING,
                            high = now_xl.high,
                            low = now_xl.low,
                            line = now_xl.line,
                            jump = jump,
                            line_broken= now_xl.line_broken,
                            fx_high = fx_high,
                            fx_low = fx_low,
                            is_confirm = True,
                        )
                    )
                if fx_type == Mark.DI and up_xl.low > now_xl.low and now_xl.low < next_xl.low:
                    now_xl.mark_type = Mark.DI
                    xlfxs.append(
                        XLFX(
                            mark_type = Mark.DI,
                            high = now_xl.high,
                            low = now_xl.low,
                            line = now_xl.line,
                            jump =jump,
                            line_broken= now_xl.line_broken,
                            fx_high = fx_high,
                            fx_low = fx_low,
                            is_confirm = True,
                        )
                    )
            else:
                ##uncomplete FX 
                fx_high = max(up_xl.high,now_xl.high)
                fx_low = min(up_xl.low,now_xl.low)

                if fx_type == Mark.DING and up_xl.high < now_xl.high:
                    now_xl.mark_type = Mark.DING 
                    xlfxs.append(
                        XLFX(
                            mark_type =Mark.DING,
                            high = now_xl.high,
                            low = now_xl.low,
                            line = now_xl.line,
                            jump = jump,
                            line_broken= now_xl.line_broken,
                            fx_high = fx_high,
                            fx_low = fx_low,
                            is_confirm = False,
                        )
                    )
                if fx_type == Mark.DI and up_xl.low > now_xl.low:
                    now_xl.mark_type = Mark.DI
                    xlfxs.append(
                        XLFX(
                            mark_type = Mark.DI,
                            high = now_xl.high,
                            low = now_xl.low,
                            line = now_xl.line,
                            jump =jump,
                            line_broken= now_xl.line_broken,
                            fx_high = fx_high,
                            fx_low = fx_low,
                            is_confirm = False,
                        )
                    )

        return xlfxs

    @staticmethod
    def __copy_zs(copy_zs: ZS, to_zs: ZS):
        """
        copy from  zs to zs 
        """
        to_zs.zs_type = copy_zs.zs_type
        to_zs.direction_type = copy_zs.direction_type 
        to_zs.start = copy_zs.start
        to_zs.lines = copy_zs.lines
        to_zs.end = copy_zs.end
        to_zs.zg = copy_zs.zg
        to_zs.zd = copy_zs.zd
        to_zs.gg = copy_zs.gg
        to_zs.dd = copy_zs.dd
        to_zs.high_supported = copy_zs.high_supported
        to_zs.low_supported = copy_zs.low_supported 
        to_zs.line_num = copy_zs.line_num
        to_zs.level = copy_zs.level
        to_zs.max_power = copy_zs.max_power
        to_zs.ts_opened = copy_zs.ts_opened 
        to_zs.ts_closed = copy_zs.ts_closed 
        to_zs.is_confirm = copy_zs.is_confirm
        to_zs.real = copy_zs.real
        return

    @staticmethod
    def cross_interval(interval_one, interval_two):
        """
        compute the cross of two interval 
        :param interval_one:
        :param interval_two:
        :return:
        """

        max_one = max(interval_one[0], interval_one[1])
        min_one = min(interval_one[0], interval_one[1])
        max_two = max(interval_two[0], interval_two[1])
        min_two = min(interval_two[0], interval_two[1])

        cross_max_val = min(max_two, max_one)
        cross_min_val = max(min_two, min_one)

        if cross_max_val >= cross_min_val:
            return  cross_max_val,cross_min_val
        else:
            return None 

    @staticmethod
    def compare_power_divergence(
        one_power: dict, 
        two_power: dict,
        indicator_key: str = 'macd',
        indicator_element:str = 'macd_value',
        post_key:str = 'sum',
        ):
        """
        compute  if there has a divergence with macd value sum.
        """
        if two_power[indicator_key][indicator_element][post_key] < one_power[indicator_key][indicator_element][post_key]:
            return True
        else:
            return False

    def get_anchored_vwap(
        self,
        run_type: LineType = LineType.BI,
    ):
        if run_type == LineType.BI:
            line = self.bis[-1]
        else:
            line = self.xds[-1]
        start = line.end.middle_twist_bar.b_index+1
        fx_bars = self.newbars[start:]
        return np.sum([bar.close*bar.volume for bar in fx_bars])/np.sum([bar.volume for bar in fx_bars])

    def memory_reduce(self):
        if len(self.newbars) > self.bar_capacity:
            middle_xd = self.xds[len(self.xds)//2]
            start_index = middle_xd.start.twist_bars[0].elements[0].index
            newbars = self.newbars[max(0,start_index-1):]

            # reset info 
            self.newbars = []       
            self.twist_bars = [] 
            self.fxs = []          
            self.real_fxs = []   
            self.bis = [] 
            self.xds = []
            self.big_trends = []
            self.bi_zss = []
            self.xd_zss = []

            self.bar_count= 0 
            
            self.macd_dif = [] 
            self.macd_dem = [] 
            self.macd_value = [] 
            self.boll_up = [] 
            self.boll_middle = [] 
            self.boll_down = [] 
            self.ma_value = [] 

            for bar in newbars:
                bar.index = self.bar_count 
                
                self.newbars.append(bar) 
                ##update twist bars 
                self.process_twist_bar() 
                ##update fx 
                self.process_fx() 
                ##update bi 
                bi_update = self.process_bi()

                ##update xd from bi 
                xd_update = self.process_up_line(LineType.BI) if bi_update else False
                ##update big_trend 
                self.process_up_line(LineType.XD) if xd_update else False
                ## process bi and xd zs 
                self.process_zs(LineType.BI) if bi_update else None
                self.process_zs(LineType.XD) if xd_update else None 

                self.process_trade_point(LineType.BI) if bi_update else None
                self.process_trade_point(LineType.XD) if xd_update else None
                ##update bar_count 
                self.bar_count += 1

    def on_reset(self):
        """
        Actions to be performed when the strategy is reset.
        """
        # Reset indicators here
        self.macd.reset()
        self.boll.reset() 
        self.ma.reset() 

        self.newbars = []       
        self.twist_bars = [] 
        self.fxs = []          
        self.real_fxs = []   
        self.bis = [] 
        self.xds = []
        self.big_trends = []
        self.bi_zss = []
        self.xd_zss = []

        self.bar_count= 0 
        
        self.macd_dif = [] 
        self.macd_dem = [] 
        self.macd_value = [] 
        self.boll_up = [] 
        self.boll_middle = [] 
        self.boll_down = [] 
        self.ma_value = [] 