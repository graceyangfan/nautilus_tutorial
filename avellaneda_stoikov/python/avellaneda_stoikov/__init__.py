"""Python package wrapper for the pyo3 calibrator.

Exposes AvellanedaStoikov from the compiled extension as a top-level import.
"""

try:
    from .avellaneda_stoikov import AvellanedaStoikov  # type: ignore
except Exception as _e:  # pragma: no cover
    AvellanedaStoikov = None  # type: ignore

