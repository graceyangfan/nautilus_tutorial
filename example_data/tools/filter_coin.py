import ccxt
import datetime

# Initialize Binance Futures
exchange = ccxt.binance({
    'enableRateLimit': True,  # this option is mandatory to minimize the risk of being banned by the exchange
    'options': {
        'defaultType': 'future',  # this line enables the futures trading
    },
})

# Get all symbols
markets = exchange.load_markets()
symbols = [symbol for symbol in markets.keys() if symbol.endswith('/USDT')]

# Filter symbols that were up before 2022-1-1
filtered_symbols = []
for symbol in symbols:
    try:
        # Fetch the earliest available OHLCV data
        ohlcv = exchange.fetch_ohlcv(symbol, '1d', since=0, limit=1)
        if ohlcv:
            # The timestamp is the first element of the OHLCV data
            timestamp = ohlcv[0][0]
            up_market_day = datetime.datetime.fromtimestamp(timestamp / 1e3)
            if up_market_day < datetime.datetime(2022, 1, 1):
                filtered_symbols.append(symbol)
    except ccxt.BaseError as e:
        print(f"An error occurred while fetching OHLCV data for {symbol}: {str(e)}")

# Write filtered symbols to a text file
with open('filtered_symbols.txt', 'w') as f:
    for symbol in filtered_symbols:
        f.write(symbol + '\n')

print(f"Filtered symbols have been written to filtered_symbols.txt")