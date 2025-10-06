"""Reusable quote and sigma utilities with optional Numba acceleration.

Exports:
  - select_sigma(wap_deque, tv_deque, mode) -> (sigma, mode_used, details)
  - compute_half_spreads(gamma, sigma, buy_a, buy_k, sell_a, sell_k, q_fix, min_a, min_k)
"""

from __future__ import annotations

import math
from collections import deque
from typing import Dict, Tuple

import numpy as np

try:  # pragma: no cover - optional JIT
    from .numba_utils import (
        gk_sigma as _jit_gk,
        parkinson_sigma as _jit_pk,
        classical_sigma as _jit_cls,
        tv_mean as _jit_tv,
        as_half_spreads as _jit_half,
    )
    _HAS_NUMBA = True
except Exception:  # pragma: no cover
    _HAS_NUMBA = False


def _py_gk(arr: np.ndarray) -> float:
    if arr.size < 32:
        return float("nan")
    step = 30
    t = 0
    gk = 0.0
    for i in range(0, arr.size - step, step):
        first = arr[i]
        last = arr[i + step]
        if first <= 0 or last <= 0:
            continue
        co = math.log(last / first)
        hi = float(np.max(arr[i : i + step]))
        lo = float(np.min(arr[i : i + step]))
        if lo <= 0:
            continue
        hl = math.log(hi / lo)
        res = 0.5 * hl * hl - ((2.0 * math.log(2.0)) - 1.0) * co * co
        gk += res
        t += 1
    return math.sqrt(gk / t) if t > 0 else float("nan")


def _py_pk(arr: np.ndarray) -> float:
    if arr.size < 32:
        return float("nan")
    step = 30
    t = 0
    acc = 0.0
    for i in range(0, arr.size - step, step):
        hi = float(np.max(arr[i : i + step]))
        lo = float(np.min(arr[i : i + step]))
        if lo <= 0:
            continue
        hl = math.log(hi / lo)
        acc += hl * hl
        t += 1
    return math.sqrt(acc / (4.0 * t * math.log(2.0))) if t > 0 else float("nan")


def _py_cls(arr: np.ndarray) -> float:
    if arr.size < 2:
        return float("nan")
    t = 10.0
    d = np.diff(arr)
    hv = float(np.dot(d, d))
    return math.sqrt(hv / t)


def _py_tv(arr: np.ndarray) -> float:
    if arr.size == 0:
        return float("nan")
    return float(arr.mean())


def select_sigma(
    wap: deque[float],
    tv: deque[float],
    mode: str = "auto",
) -> Tuple[float, str, Dict[str, float]]:
    """Return (sigma, mode_used, details) from wap/tv and preferred mode.

    mode in {"gk","parkinson","classical","tv","auto"}
    """
    arr = np.asarray(list(wap), dtype=np.float64)
    tva = np.asarray(list(tv), dtype=np.float64)
    if _HAS_NUMBA:
        s_gk = _jit_gk(arr)
        s_pk = _jit_pk(arr)
        s_cls = _jit_cls(arr)
        s_tv = _jit_tv(tva)
    else:
        s_gk = _py_gk(arr)
        s_pk = _py_pk(arr)
        s_cls = _py_cls(arr)
        s_tv = _py_tv(tva)

    details = {
        "gk": s_gk,
        "parkinson": s_pk,
        "classical": s_cls,
        "tv": s_tv,
    }

    use = (mode or "auto").lower()
    if use == "gk":
        return s_gk, "gk", details
    if use == "parkinson":
        return s_pk, "parkinson", details
    if use == "classical":
        return s_cls, "classical", details
    if use == "tv":
        return s_tv, "tv", details

    # auto fallback order
    for key in ("gk", "parkinson", "classical", "tv"):
        s = details[key]
        if s and math.isfinite(s) and s > 0.0:
            return s, key, details
    return float("nan"), "none", details


def compute_half_spreads(
    gamma: float,
    sigma: float,
    buy_a: float,
    buy_k: float,
    sell_a: float,
    sell_k: float,
    q_fix: float,
    *,
    min_a: float = 1e-6,
    min_k: float = 1e-6,
) -> Tuple[float, float, bool]:
    """Compute (half_bid, half_ask, ok) with safeguards and optional JIT.

    - Applies floors to A/k to avoid domain errors
    - Returns ok=False if any invalid/NaN
    """
    ba = max(buy_a, min_a)
    sa = max(sell_a, min_a)
    bk = max(buy_k, min_k)
    sk = max(sell_k, min_k)

    if _HAS_NUMBA:
        hb, ha = _jit_half(float(gamma), float(sigma), float(ba), float(bk), float(sa), float(sk), float(q_fix))
    else:
        base1 = 1.0 + gamma / sk
        base2 = 1.0 + gamma / bk
        if base1 <= 0.0 or base2 <= 0.0:
            return 0.0, 0.0, False
        term_bid = (sigma * sigma * gamma) / (2.0 * sk * sa)
        term_ask = (sigma * sigma * gamma) / (2.0 * bk * ba)
        if term_bid <= 0.0 or term_ask <= 0.0:
            return 0.0, 0.0, False
        hb = math.log(base1) / gamma + (q_fix + 0.5) * math.sqrt(term_bid * (base1 ** (1.0 + sk / gamma)))
        ha = math.log(base2) / gamma - (q_fix - 0.5) * math.sqrt(term_ask * (base2 ** (1.0 + bk / gamma)))

    ok = (isinstance(hb, float) and isinstance(ha, float) and math.isfinite(hb) and math.isfinite(ha) and hb > 0.0 and ha > 0.0)
    return float(hb), float(ha), ok

