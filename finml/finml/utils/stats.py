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
    drop_feature_corr_threshold = 0.85,
    corr_type: str ="pearson",
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
    min_datetime = max(factor.datetime.iloc[0],label.datetime.iloc[0])
    max_datetime = min(factor.datetime.iloc[-1],label.datetime.iloc[-1])
    factor = factor[(factor.datetime >= min_datetime)&(factor.datetime <= max_datetime)]
    label = label[(label.datetime >= min_datetime)&(label.datetime <= max_datetime)]
    df = factor.merge(label,left_on="datetime",right_on="datetime",how="right")
    df = df.drop(columns=["datetime"])
    return df.corr(method=corr_type)