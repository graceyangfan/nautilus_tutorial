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

from dataclasses import dataclass
from os import set_inheritable
from typing import Dict, List, Union
from nautilus_trader.czsc.enums import Direction, DivergenceType, LineType, Mark, TradePointType
from nautilus_trader.core.data import Data
from nautilus_trader.model.objects import Price
from nautilus_trader.model.objects import Quantity
from nautilus_trader.model.data.bar import Bar,BarType 
import math 

'''
Basic  objects for twist compute 
'''

class NewBar:
    def __init__(
        self,
        bar_type,
        index,
        ts_opened,
        ts_closed,
        open,
        high,
        low,
        close,
        volume,
        info,
    ):
        self.bar_type = bar_type 
        self.ts_opened = ts_opened 
        self.ts_closed = ts_closed 
        self.index = index 
        self.open = open 
        self.high = high 
        self.low = low 
        self.close = close 
        self.volume = volume 
        self.info = info 

class TwistBar:
    def __init__(
        self,
        index,
        b_index,
        ts_opened,
        ts_closed,
        elements,
        open,
        high,
        low,
        close,
        volume,
        jump = False, #gap  #是否有缺口
    ):
        self.index = index 
        self.b_index = b_index
        self.ts_opened = ts_opened 
        self.ts_closed = ts_closed 
        self.elements = elements 
        self.open = open 
        self.high = high 
        self.low = low 
        self.close = close 
        self.volume = volume 
        self.jump = jump 
        self.previous_trend = None  

    
    def raw_bars(self):
        return self.elements 

    def bar_type(self):
        return self.elements[0].bar_type 


class FX:
    def __init__(
        self,
        mark_type, # Mark.DING  
        middle_twist_bar,
        twist_bars,
        value,
        jump,#跳空 
        real,
        is_confirm,
    ):
        self.mark_type = mark_type 
        self.middle_twist_bar = middle_twist_bar
        self.twist_bars = twist_bars 
        self.value = value 
        self.jump = jump  
        self.real = real 
        self.is_confirm = is_confirm 
        self.ts_opened = self.twist_bars[0].ts_opened 
        self.ts_closed = self.twist_bars[-1].ts_closed 
        self.index = 0 

    def power(self):
        """
        the power of FX 
        """
        power = 0
        first_twistbar = self.twist_bars[0]
        second_twistbar = self.twist_bars[1]
        third_twistbar = self.twist_bars[2]
        if third_twistbar is None:
            return power
        if self.mark_type == Mark.DING:
            # 第三个缠论K线要一根单阴线
            if len(third_twistbar.elements) > 1:
                return power
            if third_twistbar.elements[0].close > third_twistbar.elements[0].open:
                return power
            # 第三个K线的高点，低于第二根的 50% 以下
            if third_twistbar.high < (second_twistbar.high - ((second_twistbar.high - second_twistbar.low) * 0.5)):
                power += 1
            # 第三个最低点是三根中最低的
            if third_twistbar.low < first_twistbar.low and third_twistbar.low < second_twistbar.low:
                power += 1
            # 第三根的K线的收盘价要低于前两个K线
            if third_twistbar.elements[0].close < first_twistbar.low and third_twistbar.elements[0].close < second_twistbar.low:
                power += 1
            # 第三个缠论K线的实体，要大于第二根缠论K线
            if (third_twistbar.high - third_twistbar.low) > (second_twistbar.high - second_twistbar.low):
                power += 1
            # 第三个K线不能有太大的下影线
            if (third_twistbar.elements[0].high - third_twistbar.elements[0].low) != 0 and \
                    (third_twistbar.elements[0].close - third_twistbar.elements[0].low) / (third_twistbar.elements[0].high - third_twistbar.elements[0].low) < 0.3:
                power += 1
        elif self.mark_type == Mark.DI:
            # 第三个缠论K线要一根单阳线
            if len(third_twistbar.elements) > 1:
                return power
            if third_twistbar.elements[0].close < third_twistbar.elements[0].open:
                return power
            # 第三个K线的低点，高于第二根的 50% 之上
            if third_twistbar.low > (second_twistbar.low + ((second_twistbar.high - second_twistbar.low) * 0.5)):
                power += 1
            # 第三个最高点是三根中最高的
            if third_twistbar.high > first_twistbar.high and third_twistbar.high > second_twistbar.high:
                power += 1
            # 第三根的K线的收盘价要高于前两个K线
            if third_twistbar.elements[0].close > first_twistbar.high and third_twistbar.elements[0].close > second_twistbar.high:
                power += 1
            # 第三个缠论K线的实体，要大于第二根缠论K线
            if (third_twistbar.high - third_twistbar.low) > (second_twistbar.high - second_twistbar.low):
                power += 1
            # 第三个K线不能有太大的上影线
            if (third_twistbar.elements[0].high - third_twistbar.elements[0].low) != 0 and \
                    (third_twistbar.elements[0].high - third_twistbar.elements[0].close) / (third_twistbar.elements[0].high - third_twistbar.elements[0].low) < 0.3:
                power += 1
        return power

    def high(self):
        return max([item.high for item in self.twist_bars])
    
    def low(self):
        return min([item.low for item in self.twist_bars])


class LINE:
    def __init__(
        self,
        start,
        end,
        index,
        direction_type, ##Direction 
        power,
        is_confirm,
    ):
        self.start = start 
        self.end = end 
        self.index = index 
        self.direction_type = direction_type 
        self.power = power 
        self.is_confirm = is_confirm 

        self.ts_opened = self.start.ts_opened 
        self.ts_closed = self.end.ts_closed if self.end else self.ts_opened
        self.high = 0 
        self.low = 0 
        self.vwap = 0 
        self._high_rsi = 0 
        self._low_rsi =  1 

    @property
    def high_rsi(self):
        return self._high_rsi
    
    @property
    def low_rsi(self):
        return self._low_rsi
    
    def top_high(self):
        return self.end.value if self.direction_type == Direction.UP else self.start.value

    def bottom_low(self):
        return self.end.value if self.direction_type == Direction.DOWN else self.start.value

    def dd_high_low(self):
        """
        返回线 顶底端点 的高低点
        """
        if self.direction_type == Direction.UP:
            return {
                'high': self.end.value, 
                'low': self.start.value
                }
        else:
            return {
                'high': self.start.value, 
                'low': self.end.value
                }

    def real_high_low(self):
        """
        返回线 两端 实际的 高低点
        """
        return {
            'high': self.high,
             'low': self.low
             }
    def angle(self) -> float:
        """
        计算线段与坐标轴呈现的角度（正为上，负为下）
        """
        # 计算斜率
        # convert to minute 
        duration = self.end.index - self.start.index 
        k = (self.end.value - self.start.value)/(self.start.value) / duration 
        # 斜率转弧度
        k = math.atan(k)
        # 弧度转角度
        j = math.degrees(k)
        return j


class ZS:
    def __init__(
        self,
        zs_type: LineType,
        start: FX, 
        end: FX = None, 
        zg: float = None, 
        zd: float = None,
        gg: float = None, 
        dd: float = None,
        direction_type = None,
        index: int = 0,
        level: int = 0, 
        max_power: dict = None,
    ):  
        self.zs_type = zs_type 
        self.direction_type = direction_type 
        self.start = start 
        self.end = end 
        self.zg = zg 
        self.zd = zd 
        self.gg = gg 
        self.dd = dd 
        self.index = index 
        self.level = level 
        self.max_power = max_power 
        self.lines =[] 
        self.line_num = len(self.lines)
        self.ts_opened = self.start.ts_opened 
        self.ts_closed = self.end.ts_closed if self.end else self.start.ts_opened
        self.is_confirm = False 
        self.real = True 
        self.high_supported = self.zg
        self.low_supported = self.zd 

    def add_line(
        self, 
        line: LINE
        ) -> bool:
        """
        add BI or XD 
        """
        self.lines.append(line)
        return True

    def zf(self) -> float:
        """
        overlap zone / total zone 
        """
        zgzd = self.zg - self.zd
        if zgzd == 0:
            zgzd = 1
        return (zgzd / (self.gg - self.dd)) * 100
    
    def update_zs_supported(
            self, 
            threshold,
        ):
        cur_line = self.lines[-1]
        if self.high_supported is None or self.low_supported is None:
            return 
        for line in self.lines[-2::-1]:
            if abs(line.high - cur_line.high) < threshold and line.high > self.high_supported:
                self.high_supported = line.high 
        cur_line = self.lines[-1]
        for line in self.lines[-2::-1]:
            if abs(line.low - cur_line.low) < threshold and line.low < self.low_supported:
                self.low_supported = line.low

class TradePoint:
    """
    Trade point on ZS 
    """

    def __init__(
        self, 
        side_type: TradePointType,
        zs: ZS
    ):
        self.side_type = side_type 
        self.zs: ZS = zs  # zs object 

        self.ts_opened = self.zs.ts_opened 
        self.ts_closed = self.zs.ts_closed 

    def __str__(self):
        return 'TradePoint: %s ZS: %s' % (self.side_type, self.zs)


class BC:
    """
    Divergence object 
    """

    def __init__(
        self,
        divergence_type: DivergenceType, 
        zs: ZS, 
        compare_line: LINE, 
        is_divergence: bool
    ):
        self.divergence_type: DivergenceType = divergence_type  ## divergence type 
        self.zs: ZS = zs  # 
        self.compare_line: LINE = compare_line  ##  compared xd or bi 
        self.is_divergence = is_divergence  #  if is divergence 

    def __str__(self):
        return 'DivergenceType: %s is_divergence: %s zs: %s' % (self.divergence_type, self.is_divergence, self.zs)

class BI(LINE):
    """
    笔对象
    """

    def __init__(
        self, 
        start: FX, 
        end: FX = None, 
        index: int = 0,
        direction_type: str = None,
        power: dict = None, 
        is_confirm: bool = None, 
        pause: bool = False, 
    ):
        super().__init__(start, end, index, direction_type, power, is_confirm)
        self.pause: bool = pause  # paused 
        self.trade_points: Dict[TradePointType, List[TradePoint]] = {}
        self.divergences: Dict[DivergenceType, List[BC]] = {} # Divergence info 

    def get_trade_points(
        self, 
        side_type: TradePointType,
    ) -> List[TradePoint]:

        if not side_type:
            return self.trade_points 
        if  side_type not in self.trade_points.keys():
            return []
        return self.trade_points[side_type] 

    def get_divergences(
        self, 
        divergence_type: DivergenceType, 
    ) -> List[BC]:
        if not divergence_type:
            return self.divergences 
        elif divergence_type not in self.divergences.keys():
            return [] 
    
        return self.divergences[divergence_type] 


    def add_trade_point(
        self, 
        side_type:TradePointType,
        zs:ZS) -> bool:
        """
        add trade point 
        """
        trade_point_obj = TradePoint(side_type, zs)
        if side_type not in self.trade_points.keys():
            self.trade_points[side_type] = [] 
        self.trade_points[side_type].append(trade_point_obj) 
        return True


    def add_divergence(
            self,
            divergence_type: DivergenceType,
            zs: Union[ZS, None],
            compare_line: Union[LINE, None],
       is_divergence: bool
    ) -> bool:
        """
        add divergences 
        """
        divergence_obj = BC(divergence_type, zs, compare_line, is_divergence)
        if divergence_type  not in self.divergences.keys():
            self.divergences[divergence_type] = [] 
        self.divergences[divergence_type].append(divergence_obj) 

        return True

    def trade_type_exists(
        self, 
        side_type_list: list
    ) -> bool:
        """
        check if has special trade_type 
        """
        all_types = self.trade_points.keys() 
        return len(set(side_type_list) & set(all_types)) > 0

    def divergence_type_exists(
        self,
        divergence_type_list: list
    ) -> bool:
        """
        check if has special divergence type 
        """
        all_types = self.divergences.keys() 
        return len(set(divergence_type_list) & set(all_types)) > 0


class TZXL:
    """
    feature sequence 
    """

    def __init__(
        self, 
        high: float, 
        low: float, 
        line: Union[LINE, None], 
        line_broken: bool = False,
    ):
        self.high: float = high
        self.low: float = low 
        self.line: Union[LINE, None] = line
        self.line_broken: bool = line_broken

class XLFX:
    """
    Three bi => one xlfx 
    """

    def __init__(
        self,
        mark_type: str, 
        high: float, 
        low: float, 
        line: LINE,
        jump: bool = False, 
        line_broken: bool = False,
        fx_high: float = None, 
        fx_low: float = None, 
        is_confirm: bool = True
    ):
        self.mark_type = mark_type
        self.high = high
        self.low = low
        self.line = line

        self.jump = jump  # 
        self.line_broken = line_broken  # 
        self.fx_high = fx_high  # 
        self.fx_low = fx_low  #
        self.is_confirm = is_confirm  # 



class XD(LINE):
    """
    线段对象
    """

    def __init__(
        self, 
        start: FX, 
        end: FX, 
        start_line: LINE, 
        end_line: LINE = None, 
        direction_type: str = None,
        high: float = None,
        low: float = None,
        ding_fx: XLFX = None, 
        di_fx: XLFX = None, 
        power: dict = None,
        index: int = 0,
        is_confirm: bool = True,
    ):
        super().__init__(start, end, index, direction_type, power, is_confirm)
        self.high = high 
        self.low = low 

        self.start_line: LINE = start_line  # 线段起始笔
        self.end_line: LINE = end_line  # 线段结束笔
        self.ding_fx: XLFX = ding_fx
        self.di_fx: XLFX = di_fx
        self.index = index 

        self.trade_points: Dict[TradePointType, List[TradePoint]] = {}
        self.divergences: Dict[DivergenceType, List[BC]] = {} # Divergence info 

    def is_jump(self):
        """
        成线段的分型是否有缺口
        """
        if self.direction_type == Direction.UP:
            return self.ding_fx.jump
        else:
            return self.di_fx.jump

    def is_line_broken(self):
        """
        成线段的分数，是否背笔破坏（被笔破坏不等于线段结束，但是有大概率是结束了）
        """
        if self.direction_type == Direction.UP:
            return self.ding_fx.line_broken
        else:
            return self.di_fx.line_broken

    def get_trade_points(
        self, 
        side_type: TradePointType,
    ) -> List[TradePoint]:

        if not side_type:
            return self.trade_points 
        if  side_type not in self.trade_points.keys():
            return []
        return self.trade_points[side_type] 

    def get_divergences(
        self, 
        divergence_type: DivergenceType, 
    ) -> List[BC]:
        if not divergence_type:
            return self.divergences 
        elif divergence_type not in self.divergences.keys():
            return [] 
    
        return self.divergences[divergence_type] 


    def add_trade_point(
        self, 
        side_type:TradePointType,
        zs:ZS) -> bool:
        """
        add trade point 
        """
        trade_point_obj = TradePoint(side_type, zs)
        if side_type not in self.trade_points.keys():
            self.trade_points[side_type] = [] 
        self.trade_points[side_type].append(trade_point_obj) 
        return True


    def add_divergence(
            self,
            divergence_type: DivergenceType,
            zs: Union[ZS, None],
            compare_line: Union[LINE, None],
       is_divergence: bool
    ) -> bool:
        """
        add divergences 
        """
        divergence_obj = BC(divergence_type, zs, compare_line, is_divergence)
        if divergence_type  not in self.divergences.keys():
            self.divergences[divergence_type] = [] 
        self.divergences[divergence_type].append(divergence_obj) 

        return True

    def trade_type_exists(
        self, 
        side_type_list: list
    ) -> bool:
        """
        check if has special trade_type 
        """
        all_types = self.trade_points.keys() 
        return len(set(side_type_list) & set(all_types)) > 0

    def divergence_type_exists(
        self,
        divergence_type_list: list
    ) -> bool:
        """
        check if has special divergence type 
        """
        all_types = self.divergences.keys() 
        return len(set(divergence_type_list) & set(all_types)) > 0
   
    def is_confirm(self) -> bool:
        """
        线段是否完成
        """
        return self.ding_fx.is_confirm if self.direction_type == Direction.UP  else self.di_fx.is_confirm



@dataclass
class LOW_LEVEL_TREND:
    zss: List[ZS]  # low level zss 
    lines: List[Union[LINE, BI, XD]]  # low level lines 
    zs_num: int = 0
    line_num: int = 0
    divergence_line: Union[LINE, None] = None  # 
    last_line: Union[LINE, BI, XD, None] = None  # 最后一个线
    trend: bool = False  
    oscillation: bool = False 
    line_divergence: bool = False  # bi/zss 
    trend_divergence: bool = False  # trena 
    oscillation_divergence: bool = False 


@dataclass
class MACD_INFOS:
    dif_up_cross_num = 0  
    dem_up_cross_num = 0  
    dif_down_cross_num = 0  
    dem_down_cross_num = 0  
    gold_cross_num = 0  
    silver_cross_num = 0  
    last_dif = 0
    last_dem = 0