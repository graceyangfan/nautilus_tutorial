import numpy as np
import polars as pl 
from Type import Union
from .base import (
    Expression,
    Feature,
    Constant,
    UnaryOperator,
    BinaryOperator,
    TripleOperator,
    RollingOperator,
    PairRollingOperator
)

class Abs(UnaryOperator):
    @property 
    def expr(self) -> Union[pl.Expr, float, int]:
        if not self._expr_update:
            self._expr = self._hs.expr.abs() 
            self._expr_update = True 
        return self._expr 

class Sign(UnaryOperator):
    @property
    def expr(self) -> Union[pl.Expr, float, int]:
        if not self._expr_update:
            self._expr = self._hs.expr.sign() 
            self._expr_update = True 
        return self._expr

class Log(UnaryOperator):
    @property
    def expr(self) -> Union[pl.Expr, float, int]:
        if not self._expr_update:
            self._expr = self._hs.expr.log() 
            self._expr_update = True 
        return self._expr

class Add(BinaryOperator):
    @property
    def expr(self) -> Union[pl.Expr, float, int]:
        if not self._expr_update:
            self._expr = self._lhs.expr + self._rhs.expr
            self._expr_update = True 
        return self._expr

class Sub(BinaryOperator):
    @property
    def expr(self) -> Union[pl.Expr, float, int]:
        if not self._expr_update:
            self._expr = self._lhs.expr - self._rhs.expr
            self._expr_update = True 
        return self._expr

class Mul(BinaryOperator):
    @property
    def expr(self) -> Union[pl.Expr, float, int]:
        if not self._expr_update:
            self._expr = self._lhs.expr * self._rhs.expr
            self._expr_update = True 
        return self._expr

class Div(BinaryOperator):
    @property
    def expr(self) -> Union[pl.Expr, float, int]:
        if not self._expr_update:
            self._expr = self._lhs.expr / self._rhs.expr
            self._expr_update = True 
        return self._expr

class Greater(BinaryOperator):
    #greater elements taken from the input two features
    @property
    def expr(self) -> Union[pl.Expr, float, int]:
        if not self._expr_update:
            self._expr = pl.max(self._lhs.expr, self._rhs.expr)
            self._expr_update = True 
        return self._expr

class Less(BinaryOperator):
    #less elements taken from the input two features
    @property
    def expr(self) -> Union[pl.Expr, float, int]:
        if not self._expr_update:
            self._expr = pl.min(self._lhs.expr, self._rhs.expr)
            self._expr_update = True 
        return self._expr

class Gt(BinaryOperator):
    # return bool series indicate `left > right`
    @property
    def expr(self) -> Union[pl.Expr, float, int]:
        if not self._expr_update:
            self._expr = self._lhs.expr > self._rhs.expr
            self._expr_update = True 
        return self._expr


class Ge(BinaryOperator):
    # return bool series indicate `left >= right`
    @property
    def expr(self) -> Union[pl.Expr, float, int]:
        if not self._expr_update:
            self._expr = self._lhs.expr >= self._rhs.expr
            self._expr_update = True 
        return self._expr

class Lt(BinaryOperator):
    # return bool series indicate `left < right`
    @property
    def expr(self) -> Union[pl.Expr, float, int]:
        if not self._expr_update:
            self._expr = self._lhs.expr < self._rhs.expr
            self._expr_update = True 
        return self._expr

class Le(BinaryOperator):
    # return bool series indicate `left <= right`
    @property
    def expr(self) -> Union[pl.Expr, float, int]:
        if not self._expr_update:
            self._expr = self._lhs.expr <= self._rhs.expr
            self._expr_update = True 
        return self._expr

class Eq(BinaryOperator):
    # return bool series indicate `left == right`
    @property
    def expr(self) -> Union[pl.Expr, float, int]:
        if not self._expr_update:
            self._expr = self._lhs.expr == self._rhs.expr
            self._expr_update = True 
        return self._expr

class Ne(BinaryOperator):
    # return bool series indicate `left != right`
    @property
    def expr(self) -> Union[pl.Expr, float, int]:
        if not self._expr_update:
            self._expr = self._lhs.expr != self._rhs.expr
            self._expr_update = True 
        return self._expr

class And(BinaryOperator):
    # two features' row by row & output
    @property
    def expr(self) -> Union[pl.Expr, float, int]:
        if not self._expr_update:
            self._expr = self._lhs.expr & self._rhs.expr
            self._expr_update = True 
        return self._expr

class Or(BinaryOperator):
    # two features' row by row | output
    @property
    def expr(self) -> Union[pl.Expr, float, int]:
        if not self._expr_update:
            self._expr = self._lhs.expr | self._rhs.expr
            self._expr_update = True 
        return self._expr

class If(TripleOperator):
    # if condition is true, return true_value, else return false_value
    @property
    def expr(self) -> Union[pl.Expr, float, int]:
        if not self._expr_update:
            self._expr = pl.when(self._lhs.expr).then(self._mhs.expr).otherwise(self._rhs.expr)
            self._expr_update = True 
        return self._expr

class Ref(RollingOperator):
    @property
    def expr(self) -> Union[pl.Expr, float, int]:
        if not self._expr_update:
            self._expr = self._hs.expr.shift(self._window_size)
            self._expr_update = True 
        return self._expr

class SMA(RollingOperator):
    @property
    def expr(self) -> Union[pl.Expr, float, int]:
        if not self._expr_update:
            self._expr = self._hs.expr.rolling_mean(window_size = self._window_size)
            self._expr_update = True 
        return self._expr

class EMA(RollingOperator):
    @property
    def expr(self) -> Union[pl.Expr, float, int]:
        if not self._expr_update:
            self._expr = self._hs.expr.ewm_mean(span = self._window_size)
            self._expr_update = True 
        return self._expr

class Sum(RollingOperator):
    @property
    def expr(self) -> Union[pl.Expr, float, int]:
        if not self._expr_update:
            self._expr = self._hs.expr.rolling_sum(window_size = self._window_size)
            self._expr_update = True 
        return self._expr

class Std(RollingOperator):
    @property
    def expr(self) -> Union[pl.Expr, float, int]:
        if not self._expr_update:
            self._expr = self._hs.expr.rolling_std(window_size = self._window_size)
            self._expr_update = True 
        return self._expr

class Var(RollingOperator):
    @property
    def expr(self) -> Union[pl.Expr, float, int]:
        if not self._expr_update:
            self._expr = self._hs.expr.rolling_var(window_size = self._window_size)
            self._expr_update = True 
        return self._expr

class Skew(RollingOperator):
    @property
    def expr(self) -> Union[pl.Expr, float, int]:
        if not self._expr_update:
            self._expr = self._hs.expr.rolling_skew(window_size = self._window_size)
            self._expr_update = True 
        return self._expr

class Kurt(RollingOperator):
    @property
    def expr(self) -> Union[pl.Expr, float, int]:
        if not self._expr_update:
            self._expr = self._hs.expr.rolling_kurt(window_size = self._window_size)
            self._expr_update = True

class Max(RollingOperator):
    #max elements taken from the input feature
    @property
    def expr(self) -> Union[pl.Expr, float, int]:
        if not self._expr_update:
            self._expr = self._hs.expr.rolling_max(window_size = self._window_size)
            self._expr_update = True 
        return self._expr

class Min(RollingOperator):
    #min elements taken from the input feature
    @property
    def expr(self) -> Union[pl.Expr, float, int]:
        if not self._expr_update:
            self._expr = self._hs.expr.rolling_min(window_size = self._window_size)
            self._expr_update = True 
        return self._expr

class Quantile(RollingOperator):
    #quantile elements taken from the input feature
    def __init__(self, feature: Union[Feature, float, int], window_size: int, quantile: float) -> None:
        super().__init__(feature, window_size)
        self._quantile = quantile
    @property
    def expr(self) -> Union[pl.Expr, float, int]:
        if not self._expr_update:
            self._expr = self._hs.expr.rolling_quantile(quantile = self._quantile, window_size = self._window_size)
            self._expr_update = True 
        return self._expr

    @classmethod
    def n_args(cls) -> int: 
        return 3 

    def __str__(self) -> str:
        return f"{type(self).__name__}({self._hs},{self._window},{self._quantile})"

class Median(RollingOperator):
    #median elements taken from the input feature
    @property
    def expr(self) -> Union[pl.Expr, float, int]:
        if not self._expr_update:
            self._expr = self._hs.expr.rolling_median(window_size = self._window_size)
            self._expr_update = True 
        return self._expr


class Delta(RollingOperator):
    #a feature instance with end minus start in rolling window
    @property
    def expr(self) -> Union[pl.Expr, float, int]:
        if not self._expr_update:
            self._expr = self._hs.expr.diff(n = self._window_size)
            self._expr_update = True 
        return self._expr

#Pair-Wise Rolling 
class Corr(PairRollingOperator):
    #correlation between two features in rolling window
    @property
    def expr(self) -> Union[pl.Expr, float, int]:
        if not self._expr_update:
            self._expr = pl.rolling_corr(self._hs.expr, self._rhs.expr, window_size = self._window_size)
            self._expr_update = True 
        return self._expr

class  Cov(PairRollingOperator):
    #covariance between two features in rolling window
    @property
    def expr(self) -> Union[pl.Expr, float, int]:
        if not self._expr_update:
            self._expr = pl.rolling_cov(self._hs.expr, self._rhs.expr, window_size = self._window_size)
            self._expr_update = True 
        return self._expr
