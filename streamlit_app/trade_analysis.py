from datetime import datetime 
import streamlit as st
from streamlit_lightweight_charts import renderLightweightCharts
import polars as pl 
from io_tool import load_parquet,load_csv
from data_scheme import bar_schema,schema_dict
from stream2batch import batch_zizag
from plot import get_kline_plot_setting,get_trade_lines_setting


if __name__ == "__main__":
    # write a sidebar to select factor analysis
    st.set_page_config(layout = "wide")
    st.sidebar.subheader('APP:')
    app_name = sorted(['Trade Analysis'])
    app = st.sidebar.selectbox('', app_name, index = 0)
    if app == 'Trade Analysis':
        data_type = st.sidebar.radio(
            "Select data type",
            ("Bar","Tick")
        )
        if data_type == "Bar":
            #df = load_parquet(schema_dict[data_type],True,"load the bar data")
            df = load_csv(True,"load the bar data")
            trades = load_csv(False,"load the trade data")
            if df is not None and trades is not None:
                #convert ts_event to datetime 
                df = df.with_columns(
                    [
                        (pl.col("close_time")/10**3).alias("time")
                    ]
                )
                trades = trades.with_columns(
                    [
                        (pl.col('ts_last')/10**9).alias("close_time"),
                        ((pl.col('ts_last')-pl.col("duration_ns"))/10**9).alias("open_time")
                        #pl.col('ts_closed').str.strptime(pl.Datetime).dt.timestamp().alias("ts_closed")
                    ]
                )
                kline_options = get_kline_plot_setting(df)
                trade_lines_option = get_trade_lines_setting(trades)
                kline_options["series"].extend(trade_lines_option)
                st.subheader("Candlestick Chart")
                renderLightweightCharts([kline_options], 'candlestick')


                

