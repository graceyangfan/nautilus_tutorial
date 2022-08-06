from zenquant.ctastrategy.backtesting import BacktestingEngine, OptimizationSetting
from zenquant.ctastrategy.strategies.double_ma_strategy import (
    DoubleMaStrategy
)
from datetime import datetime

##注意start和end要在数据库中，
##调用utils中的download_binance 下载数据存储于/home/username/.vntrader
engine = BacktestingEngine()
engine.set_parameters(
    vt_symbol="ETHUSDT.BINANCE",
    interval="1m",
    start=datetime(2022, 6, 1),
    end=datetime(2022, 7, 1),
    rate=0.0004, ## commision_rate 
    slippage=0,
    size=1,
    pricetick=0.01,
    capital=1_000_000,
)
engine.add_strategy(DoubleMaStrategy, {
    "fast_window":10,
    "slow_windw":60,
    "trade_size":0.1,
})

engine.load_data()
engine.run_backtesting()
df = engine.calculate_result()
engine.calculate_statistics()
engine.show_chart()
df = engine.daily_df
import matplotlib.pyplot as plt
plt.plot(df.index,df.balance)
plt.savefig("vnpy_result.jpeg")