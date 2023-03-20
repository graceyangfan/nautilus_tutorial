import polars as pl 
import pandas as pd 
import numpy as np 
import streamlit as st 
from data_scheme import allow_compute_dtypes 
from finml.labeling.get_label import create_label 
from finml.utils.stats import corrections 
from nautilus_trader.indicators.zscore import Zscore
 
def load_parquet(
    data_scheme,
    load_from_sidebar = False,
):
    if load_from_sidebar:
        uploaded_file = st.sidebar.file_uploader("Choose a file")
    else:
        uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        df = pl.read_parquet(
            uploaded_file,
            use_pyarrow = True,
            pyarrow_options = {"schema":data_scheme}
        )
        return df 
    

#@st.cache_data
def get_label(
    _df,
    pct_change = 0.02,
    stoploss = 0.001,
    logreturn = False 
):
    _df = create_label(
        _df.with_columns([pl.col("ts_event").alias("datetime")]),
        pct_change,
        stoploss,
        logreturn 
    )
    return _df


def create_zscore_tab(_df):
    col = st.selectbox(
        "choose_columns",
        [_df.columns[idx] for idx,item in enumerate(_df.dtypes) if item in allow_compute_dtypes]
    )
    start_period = st.slider( 
        "start_period",
        1,
        100,
        5
    )
    end_period = st.slider(
        "end_period",
        start_period,
        600,
        100
    )

    step = st.slider(
        "step",
        1,
        100,
        5
    )
    return (col,start_period,end_period,step)

@st.cache_data
def zscore_analysis(
    _df,
    col,
    start_period = 6,
    end_period = 300,
    step = 30,
):
    results = []
    label = _df.select([pl.col('datetime'),pl.col("label")])
    for period in range(start_period,end_period,step):
        zscore = Zscore(period)
        datetime = [] 
        zscore_list = [] 
    
        for j in range(_df.shape[0]):
            zscore.update_raw(_df[j,col])
            datetime.append(_df[j,"datetime"])
            if not zscore.initialized:
                zscore_list.append(np.nan)
            else:
                zscore_list.append(zscore.value)
        factor = pd.DataFrame({
            "datetime":datetime,
            f"zscore_{period}" : zscore_list,
        })
        label = _df.select([pl.col('datetime'),pl.col("label")])
        corr = corrections(
            factor,
            label,
            corr_type ="spearman",
        )
        results.append(corr) 
    st.dataframe(pd.concat(results,axis=0))



