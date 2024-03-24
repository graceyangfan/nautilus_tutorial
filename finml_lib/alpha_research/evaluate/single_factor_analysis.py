import numpy as np 
import polars as pl
import statsmodels.api as sm
from scipy import stats
from basic import half_decay_fit 

def ic_analysis(
    df: pl.DataFrame,
    analysis_column: str = "score",
    label_column: str = "label",
    method: str = "IC",
    abs_threshold: float = 0.02,
    p_value_threshold: float = 0.05
) -> dict[str, float] :
    if method == "IC":
        ic_df = df.group_by("datetime").agg(
                pl.map_groups(
                    exprs=[analysis_column, label_column],
                    function = lambda list_of_series:
                    stats.pearsonr(list_of_series[0].to_numpy(),list_of_series[1].to_numpy())
                ).alias(f'{analysis_column}_corr_p_value')
            ).sort("datetime")
        ic_df = ic_df.with_columns(
            [
                pl.col(f'{analysis_column}_corr_p_value').list.get(0).alias(f'{analysis_column}_corr'),
                pl.col(f'{analysis_column}_corr_p_value').list.get(1).alias(f'{analysis_column}_p_value')
            ]
        )
    elif method == "Rank IC":
        ic_df = df.group_by("datetime").agg(
                pl.map_groups(
                    exprs=[analysis_column, label_column],
                    function = lambda list_of_series:
                    stats.spearmanr(list_of_series[0].to_numpy(),list_of_series[1].to_numpy())
                ).alias(f'{analysis_column}_corr_p_value')
            ).sort("datetime")
        ic_df = ic_df.with_columns(
            [
                pl.col(f'{analysis_column}_corr_p_value').list.get(0).alias(f'{analysis_column}_corr'),
                pl.col(f'{analysis_column}_corr_p_value').list.get(1).alias(f'{analysis_column}_p_value')
            ]
        )
    ic_mean = ic_df[f'{analysis_column}_corr'].mean()
    ic_std = ic_df[f'{analysis_column}_corr'].std()
    ic_ir = ic_mean / ic_std
    ic_cout = ic_df[f'{analysis_column}_corr'].count()
    ic_great_than_zero = (ic_df[f'{analysis_column}_corr']>0).count()
    ic_ratio = ic_great_than_zero / ic_cout
    ic_abs_ratio = (ic_df[f'{analysis_column}_corr'].abs() > abs_threshold).count() / ic_cout
    ic_skewness = ic_df[f'{analysis_column}_corr'].skew()
    ic_kurtosis = ic_df[f'{analysis_column}_corr'].kurtosis()

    p_value_significant = (ic_df[f'{analysis_column}_p_value'] < p_value_threshold).count()

    ic_positive_ratio = p_value_significant / ic_cout * 100 
    ic_negative_ratio = (ic_cout - p_value_significant) / ic_cout * 100
    ic_change_num = ic_df[f'{analysis_column}_corr'].sign().diff().abs().sum() 
    ic_change_ratio = ic_change_num / ic_cout * 100
    ic_unchange_ratio = (ic_cout - ic_change_num) / ic_cout * 100 
    return {
            "ic_mean": ic_mean,
            "ic_std": ic_std,
            "ic_ir": ic_ir,
            "ic_ratio": ic_ratio,
            "ic_abs_ratio": ic_abs_ratio,
            "ic_skewness": ic_skewness,
            "ic_kurtosis": ic_kurtosis,
            "ic_positive_ratio": ic_positive_ratio,
            "ic_negative_ratio": ic_negative_ratio,
            "ic_change_ratio": ic_change_ratio,
            "ic_unchange_ratio": ic_unchange_ratio
        }


def multi_period_ic_analysis(
    df: pl.DataFrame,
    analysis_column: str = "score",
    label_column_prefix: str = "label",
    method: str = "IC",
    abs_threshold: float = 0.02,
    p_value_threshold: float = 0.05,
    start_period: int = 3, 
    decay_period: int = 20 
) -> pl.DataFrame:
    for period in np.arange(start_period,decay_period):
        if label_column_prefix+"_"+str(period) not in df.columns:
            raise ValueError(f"{label_column_prefix}_{period} not in df columns")
            return     
    ic_dicts = {}  
    ic_array = []           
    for period in np.arange(start_period,decay_period):
        ic_dict = ic_analysis(
            df = df,
            analysis_column = analysis_column,
            label_column = label_column_prefix+"_"+str(period),
            method = method,
            abs_threshold = abs_threshold,
            p_value_threshold = p_value_threshold
        )
        ic_array.append(ic_dict["ic_mean"]) 
        ic_dict = {k+"_"+str(period):v for k,v in ic_dict.items()}
        ic_dicts.update(ic_dict)

    multi_period_ic_analysis = pl.DataFrame(ic_dicts)
    halflife = half_decay_fit(np.arange(start_period,decay_period),ic_array)
    multi_period_ic_analysis.with_columns(ic_decay_halflife = halflife) 
    multi_period_ic_analysis.with_columns(factor_name=analysis_column)
    return multi_period_ic_analysis 
