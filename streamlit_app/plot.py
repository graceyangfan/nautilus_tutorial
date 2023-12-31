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


def get_trade_lines_setting(trades):
    # Buy is represented by green, short is represented by red
    series_options = []

    for trade in trades.iter_rows(named=True):
        line_data = [
            {"time": trade["open_time"], "value": trade["avg_px_open"]},
            {"time": trade["close_time"], "value": trade["avg_px_close"]}
        ]

        side = trade["entry"] == "BUY"
        color = "#00ff00" if side else "#ff0000"

        series_options.append({
            "type": 'Line',
            "data": line_data,
            "options": {
                "color": color,
                "lineWidth": 2,
                "lineStyle": 0  # 0-Solid, 1-Dotted, 2-Dashed, 3-LargeDashed
            }
        })

    return series_options

