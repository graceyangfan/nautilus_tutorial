import  polars as pl 
from basic import (
    three_sigma,
    mad,
    quantile,
    zscore_scale,
    robust_scale,
    minmax_scale
)


def preprocessing_extreme(
    df,
    method = "mad",
    process_columns = ["close"],
    n = 3,
    l = 0.025,
    h = 0.975
) -> pl.DataFrame:
    expressions = [] 
    if method == "three_sigma":
        for col_name in process_columns:
            low,high = three_sigma(pl.col(col_name),n)
            expressions.append(
                low.alias(f"{col_name}_low")
            )
            expressions.append(
                high.alias(f"{col_name}_high")
            )
            expressions.append(
                pl.col(col_name).clip(low,high).alias(f"{col_name}_clip")
            )
    elif method == "mad":
        for col_name in process_columns:
            low,high = mad(pl.col(col_name),n)
            expressions.append(
                low.alias(f"{col_name}_low")
            )
            expressions.append(
                high.alias(f"{col_name}_high")
            )
            expressions.append(
                pl.col(col_name).clip(low,high).alias(f"{col_name}_clip")
            )
    elif method == "quantile":
        for col_name in process_columns:
            low,high = quantile(pl.col(col_name),l,h)
            expressions.append(
                low.alias(f"{col_name}_low")
            )
            expressions.append(
                high.alias(f"{col_name}_high")
            )
            expressions.append(
                pl.col(col_name).clip(low,high).alias(f"{col_name}_clip")
            )
    else:
        raise ValueError("method not supported") 
    return df.with_columns(expressions)


def preprocessing_scale(
    df,
    method = "zscore",
    process_columns = ["close"],
) -> pl.DataFrame:
    expressions = [] 
    if method == "zscore":
        for col_name in process_columns:
            mean, std, zscore = zscore_scale(pl.col(col_name))
            expressions.append(
                mean.alias(f"{col_name}_mean")
            )
            expressions.append(
                std.alias(f"{col_name}_std")
            )
            expressions.append(
                zscore.alias(f"{col_name}_zscore")
            )
    elif method == "robust":
        for col_name in process_columns:
            median, mad_median, robust = robust_scale(pl.col(col_name))
            expressions.append(
                median.alias(f"{col_name}_median")
            )
            expressions.append(
                mad_median.alias(f"{col_name}_mad_median")
            )
            expressions.append(
                robust.alias(f"{col_name}_robust")
            )
    elif method == "minmax":
        for col_name in process_columns:
            min_expr, max_expr, minmax = minmax_scale(pl.col(col_name))
            expressions.append(
                min_expr.alias(f"{col_name}_min")
            )
            expressions.append(
                max_expr.alias(f"{col_name}_max")
            )
            expressions.append(
                minmax.alias(f"{col_name}_minmax")
            )
    else:
        raise ValueError("method not supported") 
    return df.with_columns(expressions)
    

