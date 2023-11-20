from scipy.stats import rv_continuous
import polars as pl 
import pandas as pd 
import numpy as np 
class KDERv(rv_continuous):

    def __init__(self, kde, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._kde = kde

    def _pdf(self, x, *args):
        return self._kde.pdf(x)

def corrections(
    factor:pd.DataFrame,
    label:pd.DataFrame,
    drop_feature_corr_threshold = 0.7,
    corr_type: str ="spearman",
):
    '''
    corr_type: {‘pearson’, ‘kendall’, ‘spearman’} or callable
    '''
    if isinstance(factor,pl.DataFrame):
        factor = factor.to_pandas()
    if isinstance(label,pl.DataFrame):
        label = label.to_pandas()
    factor = factor.dropna()
    label = label.dropna() 
    correlations = factor.drop(columns=["datetime"]).corr(method=corr_type)
    upper_tri = correlations.where(np.triu(np.ones(correlations.shape),k=1).astype(np.bool_))
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > drop_feature_corr_threshold)]
    factor = factor.drop(columns=to_drop)
    df = factor.merge(label,left_on="datetime",right_on="datetime",how="right")
    df = df.drop(columns=["datetime"])
    # split 
    df_in = df.iloc[:int(0.5*len(df))]
    df_out = df.iloc[int(0.5*len(df)):]
    in_sample_corr = df_in.corr(method=corr_type)["label"]
    out_sample_corr = df_out.corr(method=corr_type)["label"]
    results = pd.concat([in_sample_corr,out_sample_corr],axis=1)
    results.columns=["IS_label_corr","OS_label_corr"]
    results["OS_corr_abs"] = results["OS_label_corr"].abs()
    corr = results[results.index!="label"]
    corr = corr[corr["IS_label_corr"]*corr["OS_label_corr"]>0]
    corr = corr.sort_values(by = "OS_corr_abs",ascending=False)
    return corr.drop(columns=["OS_corr_abs"])
    

def test_imbalance_corr(df):
    df = df.select(
    [
        pl.col("buyer_maker_imbalance"),
        pl.col("label"),
    ])
    df_train = df[:int(0.7*df.shape[0])]
    df_test = df[int(0.7*df.shape[0]):]
    in_sample_corr = df_train.select(pl.corr("buyer_maker_imbalance","label"))[0,0]
    out_sample_corr = df_test.select(pl.corr("buyer_maker_imbalance","label"))[0,0]
    print(f'the in sample corr of buyer_maker_imbalance  is {in_sample_corr}')
    print(f'the out sample corr of buyer_maker_imbalance  is {out_sample_corr}')
    return [in_sample_corr,out_sample_corr]