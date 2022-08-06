import requests 
from datetime import date,datetime
import time
import pandas as pd

def unix_time_millis(date):
    # epoch = datetime.utcfromtimestamp(0)
    dt = datetime.strptime(date, "%Y-%m-%d")
    # return int((dt - epoch).total_seconds() * 1000)
    return int(dt.timestamp() * 1000)
 
def GetKlines_USDT(symbol='BTC',start='2020-8-10',end='2021-8-10',period='1h'):
    Klines = []
    start_time = unix_time_millis(start)
    end_time = unix_time_millis(end)
    while start_time < end_time:
        res = requests.get('https://fapi.binance.com/fapi/v1/klines?symbol=%sUSDT&interval=%s&startTime=%s&limit=1000'%(symbol,period,start_time))
        res_list = res.json()
        Klines += res_list
        #print(datetime.utcfromtimestamp(start_time/1000).strftime('%Y-%m-%d %H:%M:%S') ,len(res_list))
        start_time = res_list[-1][0]
    return pd.DataFrame(Klines,columns=[    'open_time','open','high','low','close','amount','close_time','volume','count','buy_amount','buy_volume','null']).astype('float')

def GetKlines_BUSD(symbol='BTC',start='2020-8-10',end='2021-8-10',period='1h'):
    Klines = []
    start_time = unix_time_millis(start)
    end_time = unix_time_millis(end)
    while start_time < end_time:
        res = requests.get('https://fapi.binance.com/fapi/v1/klines?symbol=%sUSDT&interval=%s&startTime=%s&limit=1000'%(symbol,period,start_time))
        res_list = res.json()
        Klines += res_list
        #print(datetime.utcfromtimestamp(start_time/1000).strftime('%Y-%m-%d %H:%M:%S') ,len(res_list))
        start_time = res_list[-1][0]
    df = pd.DataFrame(Klines,columns=['open_time','open','high','low','close','amount','close_time','volume','count','buy_amount','buy_volume','null']).astype('float')
    df["open_time"] = df["open_time"]/1000
    return df 

if __name__ == "__main__":
   # all_coins =  ['BTC', 'ETH', 'BNB', 'ADA', 'XRP', 'DOGE', 'SOL', 'FTT', 'AVAX', 'NEAR', 'GMT', 'APE', 'GAL', 'FTM', 'DODO', 'ANC', 
  #  'GALA', 'TRX', '1000LUNC', 'LUNA2', 'DOT', 'TLM', 'ICP', 'WAVES', 'LINK']
    all_coins =  ['ETH']
    for coin in all_coins:
        df = GetKlines_USDT(symbol=coin,start='2022-6-1',end='2022-7-3',period='1m')
        df.to_parquet(f"compressed/{coin}-USDT.parquet")
