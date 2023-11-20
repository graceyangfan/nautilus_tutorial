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
        raise NotImplementedError("This function must be implemented in your newly defined expression")

    def batch_update(self, data: pl.LazyFrame) -> Union[pl.LazyFrame, float, int]:
        """This function performance real computing on data, users must define it by themselves
        """
        return data.select(self.expr)

    def __str__(self):
        return type(self).__name__

    def __neg__(self):
        from ops import Neg  # pylint: disable=C0415

        return Neg(self)

    def __gt__(self, other):
        from ops import Gt  # pylint: disable=C0415

        return Gt(self, other)

    def __ge__(self, other):
        from ops import Ge  # pylint: disable=C0415

        return Ge(self, other)

    def __lt__(self, other):
        from ops import Lt  # pylint: disable=C0415

        return Lt(self, other)

    def __le__(self, other):
        from ops import Le  # pylint: disable=C0415

        return Le(self, other)

    def __eq__(self, other):
        from ops import Eq  # pylint: disable=C0415

        return Eq(self, other)

    def __ne__(self, other):
        from ops import Ne  # pylint: disable=C0415

        return Ne(self, other)

    def __add__(self, other):
        from ops import Add  # pylint: disable=C0415

        return Add(self, other)

    def __radd__(self, other):
        from ops import Add  # pylint: disable=C0415

        return Add(other, self)

    def __sub__(self, other):
        from ops import Sub  # pylint: disable=C0415

        return Sub(self, other)

    def __rsub__(self, other):
        from ops import Sub  # pylint: disable=C0415

        return Sub(other, self)

    def __mul__(self, other):
        from ops import Mul  # pylint: disable=C0415

        return Mul(self, other)

    def __rmul__(self, other):
        from ops import Mul  # pylint: disable=C0415

        return Mul(self, other)

    def __div__(self, other):
        from ops import Div  # pylint: disable=C0415

        return Div(self, other)

    def __rdiv__(self, other):
        from ops import Div  # pylint: disable=C0415

        return Div(other, self)

    def __truediv__(self, other):
        from ops import Div  # pylint: disable=C0415

        return Div(self, other)

    def __rtruediv__(self, other):
        from ops import Div  # pylint: disable=C0415

        return Div(other, self)

    def __pow__(self, other):
        from ops import Power  # pylint: disable=C0415

        return Power(self, other)

    def __rpow__(self, other):
        from ops import Power  # pylint: disable=C0415

        return Power(other, self)

    def __and__(self, other):
        from ops import And  # pylint: disable=C0415

        return And(self, other)

    def __rand__(self, other):
        from ops import And  # pylint: disable=C0415

        return And(other, self)

    def __or__(self, other):
        from ops import Or  # pylint: disable=C0415

        return Or(self, other)

    def __ror__(self, other):
        from ops import Or  # pylint: disable=C0415

        return Or(other, self)

    @property
    def is_featured(self): 
        raise NotImplementedError("This function must be implemented in your newly defined expression")
    

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
    def __init__(self, hs: Union[Expression, float, int]) -> None:
        self._hs = hs if isinstance(hs, Expression) else Constant(hs)

    @classmethod
    def n_args(cls) -> int: 
        return 1

    @classmethod
    def category_type(cls) -> Type['Operator']: 
        return UnaryOperator

    def __str__(self) -> str:
        return f"{type(self).__name__}({self._hs})"

    @property
    def is_featured(self): 
        return self._hs.is_featured


class BinaryOperator(Operator):
    def __init__(
        self, 
        lhs: Union[Expression, float, int], 
        rhs: Union[Expression, float, int]
    ) -> None:
        self._lhs = lhs if isinstance(lhs, Expression) else Constant(lhs)
        self._rhs = rhs if isinstance(rhs, Expression) else Constant(rhs)

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

class TripleOperator(Operator):
    def __init__(
        self, 
        lhs: Union[Expression, float, int], 
        mhs: Union[Expression, float, int], 
        rhs: Union[Expression, float, int]
    ) -> None:
        self._lhs = lhs if isinstance(lhs, Expression) else Constant(lhs)
        self._mhs = mhs if isinstance(mhs, Expression) else Constant(mhs)
        self._rhs = rhs if isinstance(rhs, Expression) else Constant(rhs)

    @classmethod
    def n_args(cls) -> int: 
        return 3

    @classmethod
    def category_type(cls) -> Type['Operator']:
        return TripleOperator

    def __str__(self) -> str:
        return f"{type(self).__name__}({self._lhs},{self._mhs},{self._rhs})"

    @property
    def is_featured(self): 
        return self._lhs.is_featured or self._mhs.is_featured or self._rhs.is_featured

class RollingOperator(Operator):
    def __init__(
        self, 
        hs: Union[Expression, float, int],
        window_size: int
    ) -> None:
        self._hs = hs if isinstance(hs, Expression) else Constant(hs)
        self._window_size = window_size


    @classmethod
    def n_args(cls) -> int: 
        return 2

    @classmethod
    def category_type(cls) -> Type['Operator']:
        return RollingOperator

    def __str__(self) -> str:
        return f"{type(self).__name__}({self._hs},{self._window})"

    @property
    def is_featured(self):
        return self._hs.is_featured

class PairRollingOperator(Operator):
    def __init__(
        self, 
        lhs: Union[Expression, float, int], 
        rhs: Union[Expression, float, int], 
        window_size: int
    ) -> None:
        self._lhs = lhs if isinstance(lhs, Expression) else Constant(lhs)
        self._rhs = rhs if isinstance(rhs, Expression) else Constant(rhs)
        self._window_size = window_size 

    @classmethod
    def n_args(cls) -> int: 
        return 3

    @classmethod
    def category_type(cls) -> Type['Operator']:
        return PairRollingOperator

    def __str__(self) -> str:
        return f"{type(self).__name__}({self._lhs},{self._rhs},{self._window})"

    @property
    def is_featured(self):
        return self._lhs.is_featured or self._rhs.is_featured

class CrossSectionalOperator(Operator):
    def __init__(
        self, 
        hs: Union[Expression, float, int],
    ) -> None:
        self._hs = hs if isinstance(hs, Expression) else Constant(hs)

    @classmethod
    def n_args(cls) -> int: 
        return 1

    @classmethod
    def category_type(cls) -> Type['Operator']:
        return CrossSectionalOperator

    def __str__(self) -> str:
        return f"{type(self).__name__}({self._hs})"

    @property
    def is_featured(self):
        return self._hs.is_featured


# # class Add(BinaryOperator):
# #     @property
# #     def expr(self) ->pl.Expr:
# #         if not self._expr_update:
# #             self._expr = self._lhs.expr + self._rhs.expr
# #             self._expr_update = True 
# #         return self._expr 

# # class ExpressionBuilder:
# #     stack: List[Expression]
# #     def __init__(self, stack):
# #         self.stack = stack
# #     def batch_update(self,data):
# #         for item in self.stack:
# #             data = data.pipe(item.batch_update)

# if __name__ == "__main__":
#     filename = "../../../example_data/PEPEUSDT_kline.parquet"
#     df = pl.read_parquet(filename)
#     df = df.lazy()
#     f1 = Feature("open")
#     #print(df.select(pl.col("open")))
#     f = Add(Feature("close"),Feature("open"))
#     print(f.batch_update(df).collect())