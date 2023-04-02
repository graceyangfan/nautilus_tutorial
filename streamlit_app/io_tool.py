import polars as pl 
import streamlit as st 
from data_scheme import allow_compute_dtypes 
 
def load_parquet(
    data_scheme,
    load_from_sidebar = False,
    info ="Choose a file",
):
    if load_from_sidebar:
        uploaded_file = st.sidebar.file_uploader(info)
    else:
        uploaded_file = st.file_uploader(info)
    if uploaded_file is not None:
        df = pl.read_parquet(
            uploaded_file,
            use_pyarrow = True,
            pyarrow_options = {"schema":data_scheme}
        )
        return df 
    
def load_csv(
    load_from_sidebar = False,
    info ="Choose a file",
):
    if load_from_sidebar:
        uploaded_file = st.sidebar.file_uploader(info)
    else:
        uploaded_file = st.file_uploader(info)
    if uploaded_file is not None:
        df = pl.read_csv(uploaded_file)
        return df
