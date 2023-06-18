from abc import ABCMeta, abstractmethod
from typing import List, Type, Union
import polars as pl 
import numpy as np 

class Expression(metaclass=ABCMeta):
    """Expression base class"""

    _expr: pl.Expr 
    _expr_update: bool = False 
    @property
    def expr(self) -> Union[pl.Expr, float, int]:
        raise NotImplementedError("This function must be implemented in your newly defined feature")

    def batch_update(self, data: pl.LazyFrame) -> Union[pl.LazyFrame, float, int]:
        return data.select(self.expr)

    def __str__(self):
        return type(self).__name__

    def __add__(self, other: Union["Expression", float]) -> "Add":
        if isinstance(other, Expression):
            return Add(self, other)

    @property
    def is_featured(self): 
        raise NotImplementedError("This function must be implemented in your newly defined feature")
    

class Feature(Expression):
    def __init__(self, feature_name: str) -> None:
        self._feature_name = feature_name

    @property
    def expr(self) -> Union[pl.Expr, float, int]:
        if not self._expr_update:
            self._expr = pl.col(self._feature_name)
            self._expr_update = True 
        return self._expr 

    def __str__(self) -> str: 
        return f'Feature({self._feature_name})'

    @property
    def is_featured(self): 
        return True


class Constant(Expression):
    '''
        The is used for helping building simper expression tree.
    '''
    def __init__(self, value: float) -> None:
        self._value = value
        self._feature_name = f'Constant({str(self._value)})'

    @property
    def expr(self) -> float:
        if not self._expr_update:
            self._expr = self._value
            self._expr_update = True 
        return self._expr 

    def batch_update(self, data: pl.LazyFrame) -> float:
        return  self.expr  

    def __str__(self) -> str: 
        return f'Constant({str(self._value)})'

    @property
    def is_featured(self): 
        return False


class Operator(Expression):
    @classmethod
    @abstractmethod
    def n_args(cls) -> int: ...

    @classmethod
    @abstractmethod
    def category_type(cls) -> Type['Operator']: ...


class UnaryOperator(Operator):
    def __init__(self, feature: Union[Feature, float, int]) -> None:
        self._feature = feature if isinstance(feature, Feature) else Constant(feature)

    @classmethod
    def n_args(cls) -> int: 
        return 1

    @classmethod
    def category_type(cls) -> Type['Operator']: 
        return UnaryOperator

    def __str__(self) -> str:
        return f"{type(self).__name__}({self._feature})"

    @property
    def is_featured(self): 
        return self._feature.is_featured


class BinaryOperator(Operator):
    def __init__(self, lhs: Union[Feature, float, int], rhs: Union[Feature, float, int]) -> None:
        self._lhs = lhs if isinstance(lhs, Feature) else Constant(lhs)
        self._rhs = rhs if isinstance(rhs, Feature) else Constant(rhs)

    @classmethod
    def n_args(cls) -> int: 
        return 2

    @classmethod
    def category_type(cls) -> Type['Operator']:
        return BinaryOperator

    def __str__(self) -> str:
        return f"{type(self).__name__}({self._lhs},{self._rhs})"

    @property
    def is_featured(self): 
        return self._lhs.is_featured or self._rhs.is_featured

# class Add(BinaryOperator):
#     @property
#     def expr(self) ->pl.Expr:
#         if not self._expr_update:
#             self._expr = self._lhs.expr + self._rhs.expr
#             self._expr_update = True 
#         return self._expr 

# class ExpressionBuilder:
#     stack: List[Expression]
#     def __init__(self, stack):
#         self.stack = stack
#     def batch_update(self,data):
#         for item in self.stack:
#             data = data.pipe(item.batch_update)

if __name__ == "__main__":
    filename = "../../../example_data/PEPEUSDT_kline.parquet"
    df = pl.read_parquet(filename)
    df = df.lazy()
    f1 = Feature("open")
    #print(df.select(pl.col("open")))
    f = Add(Feature("close"),Feature("open"))
    print(f.batch_update(df).collect())