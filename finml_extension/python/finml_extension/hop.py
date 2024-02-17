from __future__ import annotations
import polars as pl
from polars.utils.udfs import _get_shared_lib_location

_lib = _get_shared_lib_location(__file__)


@pl.api.register_expr_namespace("hop")
class RollingOps:
    
    def __init__(self, expr: pl.Expr):
        self._expr: pl.Expr = expr
    def rolling_idxmax(
        self,
        window: int,
        by_columns: list[str] = [],
    ):
        return self._expr.register_plugin(
            lib=_lib,
            symbol="pl_rolling_idxmax",
            kwargs={"window": window, "by_columns": by_columns}, 
            is_elementwise=True,
        )