import numpy as np
import polars as pl 

from .base import (
    Expression,
    Feature,
    Constant,
    UnaryOperator,
    BinaryOperator
)

class Abs(UnaryOperator):
    @property 
    def expr(self) -> Union[pl.Expr, float, int]:
        if not self._expr_update:
            self._expr = self._feature.expr.abs() 
            self._expr_update = True 
        return self._expr 

class Sign(UnaryOperator):
    @property
    def expr(self) -> Union[pl.Expr, float, int]:
        if not self._expr_update:
            self._expr = self._feature.expr.sign() 
            self._expr_update = True 
        return self._expr

class log(UnaryOperator):
    @property
    def expr(self) -> Union[pl.Expr, float, int]:
        if not self._expr_update:
            self._expr = self._feature.expr.log() 
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

class Ref(BinaryOperator):
    def __init__(self, feature: Union[Feature, float, int], window_size: int) -> None:
        super().__init__(feature, window_size)
        self._window_size = window_size
