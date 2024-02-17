from __future__ import annotations
import polars as pl
from polars.utils.udfs import _get_shared_lib_location
from typing import Union 

_lib = _get_shared_lib_location(__file__)


@pl.api.register_expr_namespace("hop")
class RollingOps:
    
    def __init__(self, expr: pl.Expr):
        self._expr: pl.Expr = expr
    def rolling_idxmax(
        self,
        window: int,
        by_columns: Union[str, list[str]] = None,
    ):
        if by_columns is None:
            return self._expr.register_plugin(
                lib=_lib,
                symbol="pl_rolling_idxmax",
                kwargs={"window": window, "by_columns": None}, 
                is_elementwise=True,
            )
        else:
            if isinstance(by_columns, str):
                by_columns = [by_columns]
            return self._expr.register_plugin(
                lib=_lib,
                symbol="pl_rolling_idxmax",
                args= [pl.col(name) for name in by_columns],
                kwargs={"window": window, "by_columns": by_columns}, 
                is_elementwise=True,
            )