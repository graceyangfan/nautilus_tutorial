import streamlit as st 
from data_scheme import bar_schema,schema_dict
from factor_analysis import (
    load_parquet,
    get_label,
    create_zscore_tab,
    zscore_analysis
)



if __name__ == "__main__":
    # write a sidebar to select factor analysis
    st.set_page_config(layout = "wide")
    st.sidebar.subheader('APP:')
    app_name = sorted(['Factor Analysis'])
    app = st.sidebar.selectbox('', app_name, index = 0)
    if app == 'Factor Analysis':
        data_type = st.sidebar.radio(
            "Select data type",
            ("Bar","Tick")
        )
        if data_type == "Bar":
            df = load_parquet(schema_dict[data_type],True)
            if df is not None:
                df = get_label(df)
        st.title(f'Factor Analysis')
        options = st.selectbox(
            '', 
            ["Zscore"],
            index = 0
        )
        if options == "Zscore" and df is not None: 
            zscore_widgets = create_zscore_tab(df)
            zscore_analysis(df,*zscore_widgets)

