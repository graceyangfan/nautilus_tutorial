from typing import Dict, Optional, List, Any, Callable, Generic, TypeVar, Union, Protocol
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from collections import deque
from enum import Enum
from abc import ABC, abstractmethod
from nautilus_trader.common.component import Clock
from nautilus_trader.model.enums import AggressorSide
from nautilus_trader.model.data import TradeTick
T = TypeVar('T')

NANOS_IN_SECOND = 1_000_000_000  # 每秒的纳秒数

class Aggregator(Protocol):
    """聚合函数协议"""
    name: str
    def __call__(self, data: List[T]) -> Any:
        """执行聚合计算"""
        ...

class WindowType(Enum):
    """时间窗口类型"""
    TUMBLING = "tumbling"  # 固定时间窗口，不重叠
    SLIDING = "sliding"    # 滑动时间窗口，以当前时间为结束

@dataclass
class TimeWindow:
    """时间窗口"""
    start_time: int  # 纳秒时间戳
    end_time: int    # 纳秒时间戳
    window_type: WindowType
    
    def contains(self, timestamp: int, behavior: str = "right_open") -> bool:
        """
        判断时间戳是否在窗口内
        
        Parameters
        ----------
        timestamp : int
            纳秒时间戳
        behavior : str, default "left_open"
            窗口行为:
            - "left_open": 左开右闭 (default)
            - "right_open": 左闭右开
            - "closed": 左闭右闭
            - "open": 左开右开
        """
        if behavior == "right_open":
            return self.start_time <= timestamp < self.end_time
        elif behavior == "left_open":
            return self.start_time < timestamp <= self.end_time
        elif behavior == "closed":
            return self.start_time <= timestamp <= self.end_time
        elif behavior == "open":
            return self.start_time < timestamp < self.end_time
        else:
            raise ValueError(f"Unknown window behavior: {behavior}")

class TimeAggregator(Generic[T]):
    """
    时间聚合器
    支持Tumbling和Sliding两种窗口模式
    """
    def __init__(
        self,
        window_type: WindowType,
        interval: timedelta,
        aggregators: List[Aggregator],
        clock: Clock,
        align_to_interval: bool = True,
        window_behavior: str = "right_open",
        max_history: Optional[timedelta] = None,
        auto_compute: bool = True,  # 是否自动计算
    ):
        self.window_type = window_type
        self.interval = interval
        self.interval_nanos = int(interval.total_seconds() * NANOS_IN_SECOND)
        self.align_to_interval = align_to_interval
        self.window_behavior = window_behavior 
        self.max_history = max_history or interval * 2
        self.max_history_nanos = int(self.max_history.total_seconds() * NANOS_IN_SECOND)
        self.auto_compute = auto_compute
        self.clock = clock
        
        # 聚合器
        self._aggregators = {agg.name: agg for agg in aggregators}
        
        # 数据存储
        self._data = deque()  # [(timestamp_ns, value), ...]
        
        # 缓存
        self._cache = {}
        
        # 窗口状态
        self._current_window: Optional[TimeWindow] = None
        if self.window_type == WindowType.TUMBLING:
            self._init_tumbling_window()
        else:
            self._current_window = TimeWindow(
                start_time=self.clock.timestamp_ns() - self.interval_nanos,
                end_time=self.clock.timestamp_ns(),
                window_type=WindowType.SLIDING,
            )
            
        # 最新计算结果
        self._latest_results = {}

        self.generate_new_data = False # 是否生成新数据
    
    def _init_tumbling_window(self) -> None:
        """初始化Tumbling Window"""
        now_ns = self.clock.timestamp_ns()
        if self.align_to_interval:
            # 对齐到下一个整数时间间隔
            window_start_ns = int(now_ns // self.interval_nanos) * self.interval_nanos
        else:
            window_start_ns = now_ns
        
        self._current_window = TimeWindow(
            start_time=window_start_ns,
            end_time=window_start_ns + self.interval_nanos,
            window_type=WindowType.TUMBLING,
        )
    
    def should_generate_new_data(self, timestamp_ns: int) -> bool:
        if self.window_type == WindowType.TUMBLING:
            if self.window_behavior == "right_open":
                return timestamp_ns >= self._current_window.end_time
            else:
                return timestamp_ns > self._current_window.end_time
        else:
            return True

    def _update_window(self, timestamp_ns: int):
        """更新窗口状态，返回需要计算的窗口"""
        if self.window_type == WindowType.TUMBLING:
            window_start_ns = self._current_window.end_time
            self._current_window = TimeWindow(
                start_time=window_start_ns,
                end_time=window_start_ns + self.interval_nanos,
                window_type=WindowType.TUMBLING,
            )
            return 
        else:
            # Sliding窗口
            self._current_window = TimeWindow(
                start_time=timestamp_ns - self.interval_nanos,
                end_time=timestamp_ns,
                window_type=WindowType.SLIDING,
            )
            return 

    def _get_window(self, timestamp_ns: int) -> TimeWindow:
        """获取当前窗口，不触发计算"""
        if self.window_type == WindowType.TUMBLING:
            if not self._current_window:
                self._init_tumbling_window()
            return self._current_window
        else:
            return TimeWindow(
                start_time=timestamp_ns - self.interval_nanos,
                end_time=timestamp_ns,
                window_type=WindowType.SLIDING,
            )
    
    def _cleanup_expired_data(self, current_time_ns: int) -> None:
        """清理过期数据"""
        expire_time_ns = current_time_ns - self.max_history_nanos
        while self._data and self._data[0][0] < expire_time_ns:
            self._data.popleft()
    
    def add(self, value: T, timestamp_ns: Optional[int] = None) -> None:
        """
        添加数据并更新窗口状态，根据设置决定是否自动计算
        
        Parameters
        ----------
        value : T
            数据值
        timestamp_ns : Optional[int]
            纳秒时间戳，None表示使用当前时间
        """
        ts_ns = timestamp_ns or self.clock.timestamp_ns()
        self._data.append((ts_ns, value))
        
        # 清理过期数据和缓存
        self._cleanup_expired_data(ts_ns)
        self._cache.clear()
        
        # 是否生成新数据
        self.generate_new_data = self.should_generate_new_data(ts_ns)
        if self.window_type == WindowType.SLIDING:
            #slide 先更新窗口后计算
            self._update_window(ts_ns)
            if self.auto_compute:
                self._compute_window(self._current_window)
        elif self.window_type == WindowType.TUMBLING:
            #tumbling 先计算后更新窗口
            if self.generate_new_data:
                if self.auto_compute:
                    self._compute_window(self._current_window)
                # 更新窗口状态
                self._update_window(ts_ns)


    def _compute_window(self, window: TimeWindow) -> Dict[str, Any]:
        """
        计算指定窗口的聚合结果
        """
        # 获取窗口内的数据
        window_data = [
            value for ts_ns, value in self._data
            if window.contains(ts_ns, behavior=self.window_behavior)
        ]
        if not window_data:
            return {}
        
        # 计算所有聚合器的结果
        results = {}
        for name, aggregator in self._aggregators.items():
            result = aggregator(window_data)
            results[name] = result
        
        # 更新最新结果
        self._latest_results = results
        return results
    
    def get_latest(self) -> Dict[str, Any]:
        """获取最新计算结果"""
        return self._latest_results

    def manual_compute(
        self
    ):
        """ 
            手动触发计算,默认self.generate_new_data 为True 
        """
        self.generate_new_data = True
        #计算值 
        self._compute_window(self._current_window)
        # 更新窗口状态
        self._update_window(self.clock.timestamp_ns())

   
# 示例聚合函数
class TradeVWAP:
    """VWAP计算"""
    def __init__(self):
        self.name = "trade_vwap"
    
    def __call__(self, trades: List[T]) -> float:
        total_value = sum(t.price.as_double() * t.size.as_double() for t in trades)
        total_volume = sum(t.size.as_double() for t in trades)
        return total_value / total_volume if total_volume > 0 else 0.0

class TradeImbalance:
    """交易失衡计算"""
    def __init__(self, lookback: Optional[int] = None):
        self.name = "trade_imbalance"
        self.lookback = lookback
    
    def __call__(self, trades: List[T]) -> float:
        if self.lookback:
            trades = trades[-self.lookback:]
        buy_volume = sum(t.size.as_double() for t in trades if t.aggressor_side == AggressorSide.BUYER)
        sell_volume = sum(t.size.as_double() for t in trades if t.aggressor_side == AggressorSide.SELLER)
        total_volume = buy_volume + sell_volume
        return (buy_volume - sell_volume) / total_volume if total_volume > 0 else 0.0

def create_trade_aggregator(
    window_type: WindowType,
    interval: timedelta,
    clock: Clock,
    auto_compute: bool = True,
    align_to_interval: bool = True,
) -> TimeAggregator[TradeTick]:
    """创建交易数据聚合器"""
    
    aggregators = [
        TradeVWAP(),
        TradeImbalance(lookback=100),
    ]
    
    return TimeAggregator[TradeTick](
        window_type=window_type,
        interval=interval,
        aggregators=aggregators,
        clock=clock,
        align_to_interval=align_to_interval,
        auto_compute=auto_compute
    )