import unittest
import polars as pl
import pandas as pd
import numpy as np 
from scipy.stats import percentileofscore


from expr_engine import ExprEngine

class TestAlphaFactor(unittest.TestCase):
    def setUp(self):
        Load sample data
        filename = "../example_data/data.parquet"
        self.df = pl.read_parquet(filename)
        self.df = self.df.with_columns(
            [
                pl.col("date").alias("datetime"),
                pl.col("asset").alias("symbol")
            ]
        )
        self.df = self.df.sort(["symbol", "datetime"])
        self.df = self.df.with_columns(
            pl.int_range(pl.len()).alias("index")
        )
        self.pd_df = self.df.to_pandas()
        self.df = self.df.lazy()
        self.pd_df = self.pd_df.sort_values(by=["symbol","datetime"]).reset_index(drop=True)
        self.expr_engine = ExprEngine()
        self.expr_engine.init()

    def test_if_operator(self):
        expression = "If($close/Ref($close,1) < 1, $close,0)"
        expr = self.expr_engine.get_expression(expression)
        se = expr.batch_update(self.df,True)
        result = se.collect()
        print(result)

    def test_Idxmin_operator(self):
        expression = "IdxMin($close,5)" 
        expr = self.expr_engine.get_expression(expression)
        se = expr.batch_update(self.df,True)
        result = se.collect()
        print(result.filter(pl.col("symbol") =="s_0000"))
        pandas_result = self.pd_df[self.pd_df["symbol"]=="s_0000"]["close"].rolling(5, min_periods=1).apply(lambda x: x.argmin()+1, raw=True)
        print(pandas_result)

    def test_Mad_operator(self):
        def mad(x):
            x1 = x[~np.isnan(x)]
            return np.mean(np.abs(x1 - x1.mean()))
        expression = "Mad($close,5)" 
        expr = self.expr_engine.get_expression(expression)
        se = expr.batch_update(self.df,True)
        result = se.collect()
        print(result.filter(pl.col("symbol") =="s_0000"))
        pandas_result = self.pd_df[self.pd_df["symbol"]=="s_0000"]["close"].rolling(5, min_periods=1).apply(lambda x: mad(x), raw=True)
        print(pandas_result) 

    def test_Rank_operator(self):
        def rank(x):
            if np.isnan(x[-1]):
                return np.nan
            x1 = x[~np.isnan(x)]
            if x1.shape[0] == 0:
                return np.nan
            return percentileofscore(x1, x1[-1]) / 100
        expression = "Rank($close,5)" 
        expr = self.expr_engine.get_expression(expression)
        se = expr.batch_update(self.df,True)
        result = se.collect()
        print(result.filter(pl.col("symbol") =="s_0000"))
        pandas_result = self.pd_df[self.pd_df["symbol"]=="s_0000"]["close"].rolling(5, min_periods=1).apply(lambda x: rank(x), raw=True)
        print(pandas_result) 

    def test_count_operator(self):
        expression = "Count($close,5)" 
        expr = self.expr_engine.get_expression(expression)
        se = expr.batch_update(self.df,True)
        result = se.collect()
        print(result.filter(pl.col("symbol") =="s_0000"))
        pandas_result =  getattr(self.pd_df[self.pd_df["symbol"]=="s_0000"]["close"].rolling(5, min_periods=1) , "count")()
        print(pandas_result) 

    def test_Slope_operator(self):
        def slope(y):
            import statsmodels.api as sm
            x = np.arange(len(y))
            X = sm.add_constant(x)
            model  = sm.OLS(y,X).fit() 
            return model.params[1]

        expression = "Slope($close,5)" 
        expr = self.expr_engine.get_expression(expression)
        se = expr.batch_update(self.df,True)
        result = se.collect()
        print(result.filter(pl.col("symbol") =="s_0000"))
        pandas_result = self.pd_df[self.pd_df["symbol"]=="s_0000"]["close"].rolling(5, min_periods=1).apply(lambda x: slope(x), raw=True)
        print(pandas_result)
    
    def test_rsquare_operator(self):
        def rsquare(y):
            import statsmodels.api as sm
            x = np.arange(len(y))
            X = sm.add_constant(x)
            model  = sm.OLS(y,X).fit() 
            return model.rsquared

        expression = "Rsquare($close,5)" 
        expr = self.expr_engine.get_expression(expression)
        se = expr.batch_update(self.df,True)
        result = se.collect()
        print(result.filter(pl.col("symbol") =="s_0000"))
        pandas_result = self.pd_df[self.pd_df["symbol"]=="s_0000"]["close"].rolling(5, min_periods=1).apply(lambda x: rsquare(x), raw=True)
        print(pandas_result)

    def test_residual_operator(self):
        def residual(y):
            import statsmodels.api as sm
            x = np.arange(len(y))
            X = sm.add_constant(x)
            model  = sm.OLS(y,X).fit() 
            return model.resid[-1]

        expression = "Resi($close,5)" 
        expr = self.expr_engine.get_expression(expression)
        se = expr.batch_update(self.df,True)
        result = se.collect()
        print(result.filter(pl.col("symbol") =="s_0000"))
        pandas_result = self.pd_df[self.pd_df["symbol"]=="s_0000"]["close"].rolling(5, min_periods=1).apply(lambda x: residual(x), raw=True)
        print(pandas_result)

    def test_cs_rank_operator(self):
        def csrank(df):
            return df[["close","high"]].rank(pct=True)
        expression = "CSRank($close)"
        expr = self.expr_engine.get_expression(expression)
        se = expr.batch_update(self.df,True)
        result = se.collect()
        time = self.df.collect()[0,"datetime"]
        print(result.filter(pl.col("datetime") ==time))
        pandas_result = self.pd_df.groupby(by=["datetime"], group_keys=True).apply(csrank)
        print(pandas_result)

    def test_cs_scale_operator(self):
        def scale(df, scale=1):
            x = df["close"]
            return x / x.abs().sum() * scale
        expression = "CSScale($close)"
        expr = self.expr_engine.get_expression(expression)
        se = expr.batch_update(self.df,True)
        result = se.collect()
        time = self.df.collect()[0,"datetime"]
        print(result.filter(pl.col("datetime") ==time))
        pandas_result = self.pd_df.groupby(by=["datetime"], group_keys=True).apply(scale)
        print(pandas_result)
   
if __name__ == '__main__':
    unittest.main()