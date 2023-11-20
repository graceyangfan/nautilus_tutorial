import numpy as np
import polars as pl 
from Type import Union
import warnings
from plexpr_func import slope, rsqure, residual
from base import (
    Expression,
    Feature,
    Constant,
    UnaryOperator,
    BinaryOperator,
    TripleOperator,
    RollingOperator,
    PairRollingOperator,
    CrossSectionalOperator
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

class Power(BinaryOperator):
    # two features' row by left.pow(right)
    @property
    def expr(self) -> Union[pl.Expr, float, int]:
        if not self._expr_update:
            self._expr = self._lhs.expr.pow(self._rhs.expr)
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


# class DiscreteNeutralize():
#     @property 
#     def expr(self) -> Union[pl.Expr, float, int]:
#         if not self._expr_update:
#             self._expr = self._hs.expr - self._hs.expr.mean()
#             self._expr_update = True 
#         return self._expr

class Ref(RollingOperator):
    @property
    def expr(self) -> Union[pl.Expr, float, int]:
        if not self._expr_update:
            self._expr = self._hs.expr.shift(self._window_size)
            self._expr_update = True 
        return self._expr.over("symbol")


class Mean(RollingOperator):
    @property
    def expr(self) -> Union[pl.Expr, float, int]:
        if not self._expr_update:
            self._expr = self._hs.expr.rolling_mean(window_size = self._window_size)
            self._expr_update = True 
        return self._expr.over("symbol")

class EMA(RollingOperator):
    @property
    def expr(self) -> Union[pl.Expr, float, int]:
        if not self._expr_update:
            self._expr = self._hs.expr.ewm_mean(span = self._window_size)
            self._expr_update = True 
        return self._expr.over("symbol")

class  WMA(RollingOperator):
    @property 
    def expr(self) -> Union[pl.Expr, float, int]:
        if not self._expr_update:
            w =  np.arange(self._window_size) + 1
            w = w / w.sum()
            self._expr = self._hs.expr.rolling_mean(window_size = self._window_size,weights=w)
            self._expr_update = True 
        return self._expr.over("symbol")


class Sum(RollingOperator):
    @property
    def expr(self) -> Union[pl.Expr, float, int]:
        if not self._expr_update:
            self._expr = self._hs.expr.rolling_sum(window_size = self._window_size)
            self._expr_update = True 
        return self._expr.over("symbol")

class Prod(RollingOperator):
    @property
    def expr(self) -> Union[pl.Expr, float, int]:
        if not self._expr_update: 
            self._expr = pl.arange(0,pl.count()).alias("index")
                                                .groupby_rolling("index",period=self._window_size)
                                                .agg(self._hs.product())
            self._expr_update = True 
        return self._expr.over("symbol")

class Std(RollingOperator):
    @property
    def expr(self) -> Union[pl.Expr, float, int]:
        if not self._expr_update:
            self._expr = self._hs.expr.rolling_std(window_size = self._window_size)
            self._expr_update = True 
        return self._expr.over("symbol")

class Var(RollingOperator):
    @property
    def expr(self) -> Union[pl.Expr, float, int]:
        if not self._expr_update:
            self._expr = self._hs.expr.rolling_var(window_size = self._window_size)
            self._expr_update = True 
        return self._expr.over("symbol")

class Skew(RollingOperator):
    @property
    def expr(self) -> Union[pl.Expr, float, int]:
        if not self._expr_update:
            self._expr = self._hs.expr.rolling_skew(window_size = self._window_size)
            self._expr_update = True 
        return self._expr.over("symbol")

class Kurt(RollingOperator):
    @property
    def expr(self) -> Union[pl.Expr, float, int]:
        if not self._expr_update:
            self._expr = self._hs.expr.rolling_kurt(window_size = self._window_size)
            self._expr_update = True
        return self._expr.over("symbol")

class Max(RollingOperator):
    #max elements taken from the input feature
    @property
    def expr(self) -> Union[pl.Expr, float, int]:
        if not self._expr_update:
            self._expr = self._hs.expr.rolling_max(window_size = self._window_size)
            self._expr_update = True 
        return self._expr.over("symbol")


class IdxMax(RollingOperator):
    @property
    def expr(self) -> Union[pl.Expr, float, int]:
        if not self._expr_update: 
            self._expr = pl.arange(0,pl.count()).alias("index").groupby_rolling("index",period=self._window_size).agg(self._hs.arg_max())
            self._expr_update = True 
        return self._expr.over("symbol")


class Min(RollingOperator):
    #min elements taken from the input feature
    @property
    def expr(self) -> Union[pl.Expr, float, int]:
        if not self._expr_update:
            self._expr = self._hs.expr.rolling_min(window_size = self._window_size)
            self._expr_update = True 
        return self._expr.over("symbol")

class IdxMin(RollingOperator):
    @property
    def expr(self) -> Union[pl.Expr, float, int]:
        if not self._expr_update: 
            self._expr = pl.arange(0,pl.count()).alias("index").groupby_rolling("index",period=self._window_size).agg(self._hs.arg_min())
            self._expr_update = True 
        return self._expr.over("symbol")

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
        return self._expr.over("symbol")

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
        return self._expr.over("symbol")

class Mad(RollingOperator):
    #mad elements taken from the input feature
    @property
    def expr(self) -> Union[pl.Expr, float, int]:
        if not self._expr_update:
            self._expr = pl.arange(0,pl.count()).alias("index")
            .groupby_rolling("index",period=self._window_size)
            .agg((self._hs - self._hs.median()).abs().median())
            self._expr_update = True 
        return self._expr.over("symbol")

class Rank(RollingOperator):
    """Rolling Rank (Percentile)
    """
    @property
    def expr(self) -> Union[pl.Expr, float, int]:
        if not self._expr_update: 
            self._expr = pl.arange(0,pl.count()).alias("index")
            .groupby_rolling("index",period=self._window_size)
            .agg(pl.col("close").rank().last()/pl.count()/pl.count()*100) 
            self._expr_update = True 
        return self._expr.over("symbol")


class Delta(RollingOperator):
    #a feature instance with end minus start in rolling window
    @property
    def expr(self) -> Union[pl.Expr, float, int]:
        if not self._expr_update:
            self._expr = self._hs.expr.diff(n = self._window_size)
            self._expr_update = True 
        return self._expr.over("symbol")

class Slope(Rolling):
    """Rolling Slope
    """
    @property
    def expr(self) -> Union[pl.Expr, float, int]:
        if not self._expr_update: 
            self._expr = pl.arange(0,pl.count()).alias("index")
            .groupby_rolling("index",period=self._window_size)
            .agg(slope(pl.col("index"),self._hs.expr))
            self._expr_update = True 
        return self._expr.over("symbol")

class Rsquare(Rolling):
    """Rolling Rsquare
    """
    @property
    def expr(self) -> Union[pl.Expr, float, int]:
        if not self._expr_update: 
            self._expr = pl.arange(0,pl.count()).alias("index")
            .groupby_rolling("index",period=self._window_size)
            .agg(rsqure(pl.col("index"),self._hs.expr))
            self._expr_update = True 
        return self._expr.over("symbol")

class Resi(Rolling):
    """Rolling Regression Residuals
    """
    @property
    def expr(self) -> Union[pl.Expr, float, int]:
        if not self._expr_update: 
            self._expr = pl.arange(0,pl.count()).alias("index")
            .groupby_rolling("index",period=self._window_size)
            .agg(residual(pl.col("index"),self._hs.expr))
            self._expr_update = True 
        return self._expr.over("symbol")

#Pair-Wise Rolling 
class Corr(PairRollingOperator):
    #correlation between two features in rolling window
    @property
    def expr(self) -> Union[pl.Expr, float, int]:
        if not self._expr_update:
            self._expr = pl.rolling_corr(self._hs.expr, self._rhs.expr, window_size = self._window_size)
            self._expr_update = True 
        return self._expr.over("symbol")

class  Cov(PairRollingOperator):
    #covariance between two features in rolling window
    @property
    def expr(self) -> Union[pl.Expr, float, int]:
        if not self._expr_update:
            self._expr = pl.rolling_cov(self._hs.expr, self._rhs.expr, window_size = self._window_size)
            self._expr_update = True 
        return self._expr.over("symbol")


#################### cross section operator ####################

class CSRank(CrossSectionalOperator):
    #cross section rank of a feature
    @property
    def expr(self) -> Union[pl.Expr, float, int]:
        if not self._expr_update:
            self._expr = self._hs.rank()/pl.count()
            self._expr_update = True 
        return self._expr.over("datetime")

class CSScale(CrossSectionalOperator):
    #cross section scale of a feature
    @property
    def expr(self) -> Union[pl.Expr, float, int]:
        if not self._expr_update:
            self._expr = self._hs / self._hs.sum()
            self._expr_update = True 
        return self._expr.over("datetime")



#register Operators 

OpsList = [
    Abs,
    Sign,
    Log,
    Add,
    Sub,
    Mul,
    Div,
    Greater,
    Less,
    Gt,
    Ge,
    Lt,
    Le,
    Eq,
    Ne,
    And,
    Or,
    Power,
    If,
    Ref,
    Mean,
    EMA,
    WMA,
    Sum,
    Prod,
    Std,
    Var,
    Skew,
    Kurt,
    Max,
    IdxMax,
    Min,
    IdxMin,
    Quantile,
    Median,
    Mad,
    Rank,
    Delta,
    Slope,
    Rsquare,
    Resi,
    Corr,
    Cov,
    CSRank,
    CSScale
]


class OpsWrapper:
    """Ops Wrapper"""

    def __init__(self):
        self._ops = {}

    def reset(self):
        self._ops = {}

    def register(self, ops_list):
        for operator in ops_list:
            if isinstance(operator, dict):
                ops_class, _ = get_callable_kwargs(operator)
            else:
                ops_class = operator

            if not issubclass(ops_class, Expression):
                raise TypeError("operator must be subclass of ExpressionOps, not {}".format(_ops_class))
            if operator.__name__ in self._ops:
                warnings.warn("The custom operator [{}] will override the finml default definition".format(operator.__name__))
            self._ops[operator.__name__] = operator

    def __getattr__(self, key):
        if key not in self._ops:
            raise AttributeError("The operator [{0}] is not registered".format(key))
        return self._ops[key]


Operators = OpsWrapper()


def register_all_ops(C=None):
    """register all operator"""

    Operators.reset()
    Operators.register(OpsList)

    if getattr(C, "custom_ops", None) is not None:
        Operators.register(C.custom_ops)