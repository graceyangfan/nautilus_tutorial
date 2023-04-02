import polars as pl

def get_kline_plot_setting(
    df
):
    candlestick_data = df[["time", "open", "high", "low", "close"]].to_dicts()
    volume_data = df.select([pl.col("time"),pl.col("volume").alias("value")]).to_dicts()
    # Define the chart options
    chart_options = {
        "layout": {
            "textColor": 'black',
            "background": {
                "type": 'solid',
                "color": 'white'
            }
        }
    }

    # Define the series options
    series_options = [
        {
            "type": 'Candlestick',
            "data": candlestick_data,
            "options": {
                "upColor": "#26a69a",
                "downColor": "#ef5350",
                "borderDownColor": "#ef5350",
                "borderUpColor": "#26a69a",
                "wickDownColor": "#ef5350",
                "wickUpColor": "#26a69a"
            }
        },
        {
            "type": 'Histogram',
            "data": volume_data,
            "options": {
                "color": '#26a69a',
                "priceFormat": {
                    "type": 'volume',
                },
                "priceScaleId": "" # set as an overlay setting,
            },
            "priceScale": {
                "scaleMargins": {
                    "top": 0.7,
                    "bottom": 0,
                }
            }
        }
    ]

    kline_options = {"chart": chart_options, "series": series_options}
    return kline_options 

def get_trade_lines_setting(
    trades
):
     #买入为绿色，卖空为红色
    series_options = [] 
    for i in range(trades.shape[0]):
        line_data = [{"time":trades[i,"ts_opened"],"value":trades[i,"avg_px_open"]},{"time":trades[i,"ts_closed"],"value":trades[i,"avg_px_close"]}]
        side = trades[i,"entry"] == "BUY"
        series_options.append(
            {
                "type": 'Line',
                "data": line_data,
                "options": {
                    "color": "#00ff00" if side  else "#ff0000",
                    "lineWidth": 2,
                    "lineStyle": 0 # 0-Solid, 1-Dotted, 2-Dashed, 3-LargeDashed
                }
            }
        )

    return series_options 
