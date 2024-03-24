import polars as pl 
import numpy as np 
from scipy.optimize import curve_fit

def half_decay_fit(x,y):
    # Exponential decay function
    def exp_decay(x, a, c):
        return c * np.exp(a * x)

    # Fit the curve
    popt, _ = curve_fit(exp_decay, x, y)
    a, c = popt
    halflife = np.log(2) / a
    return halflife

def three_sigma(expr: pl.Expr, n: int = 3) ->  [pl.Expr]:
    mean = expr.mean()
    std = expr.std()
    low = mean - n * std
    high = mean + n * std
    return low,high

def mad(expr: pl.Expr, n: int = 3)  ->  [pl.Expr]:
    median = expr.median() 
    mad_median = (expr - median).abs().median()
    high = median + n * mad_median
    low = median - n * mad_median
    return low,high

def quantile(expr: pl.Expr, l: float = 0.025, h: float = 0.975)  ->  [pl.Expr]:
    low = expr.quantile(l)
    high = expr.quantile(h)
    return low,high

def zscore_scale(expr: pl.Expr) -> [pl.Expr]:
    mean = expr.mean()
    std = expr.std()
    return mean, std, (expr - mean) / std

def robust_scale(expr: pl.Expr) -> [pl.Expr]:
    median = expr.median()
    mad_median = (expr - median).abs().median()
    return median, mad_median, (expr - median) / mad_median

def minmax_scale(expr: pl.Expr) -> [pl.Expr]:
    min_expr = expr.min()
    max_expr = expr.max()
    return min_expr, max_expr, (expr - min_expr) / (max_expr - min_expr)


    



