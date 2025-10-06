"""Numba-accelerated helpers for sigma estimation and AS half-spread computation.

These functions are pure numerical routines intended to be reused across strategies.
They gracefully degrade if Numba is unavailable by being imported conditionally
from a wrapper (see quote_utils.py).
"""

from __future__ import annotations

import math
import numpy as np

try:  # pragma: no cover - numba optional at runtime
    from numba import njit
except Exception:  # pragma: no cover
    # Stubs to allow import without numba installed
    def njit(*args, **kwargs):  # type: ignore
        def wrapper(fn):
            return fn

        return wrapper


@njit(cache=True, fastmath=True)
def _safe_log(x: float) -> float:
    if x <= 0.0 or not math.isfinite(x):
        return np.nan
    return math.log(x)


@njit(cache=True, fastmath=True)
def gk_sigma(arr: np.ndarray) -> float:
    n = arr.shape[0]
    if n < 32:
        return np.nan
    step = 30
    t = 0
    gk = 0.0
    for i in range(0, n - step, step):
        first = arr[i]
        last = arr[i + step]
        if first <= 0.0 or last <= 0.0:
            continue
        co = _safe_log(last / first)
        if not math.isfinite(co):
            continue
        # local hi/lo
        hi = -1e308
        lo = 1e308
        for j in range(i, i + step):
            v = arr[j]
            if v > hi:
                hi = v
            if v < lo:
                lo = v
        if lo <= 0.0:
            continue
        hl = _safe_log(hi / lo)
        if not math.isfinite(hl):
            continue
        res = 0.5 * (hl * hl) - ((2.0 * math.log(2.0)) - 1.0) * (co * co)
        gk += res
        t += 1
    if t == 0:
        return np.nan
    return math.sqrt(gk / t)


@njit(cache=True, fastmath=True)
def parkinson_sigma(arr: np.ndarray) -> float:
    n = arr.shape[0]
    if n < 32:
        return np.nan
    step = 30
    t = 0
    acc = 0.0
    for i in range(0, n - step, step):
        hi = -1e308
        lo = 1e308
        for j in range(i, i + step):
            v = arr[j]
            if v > hi:
                hi = v
            if v < lo:
                lo = v
        if lo <= 0.0:
            continue
        hl = _safe_log(hi / lo)
        if not math.isfinite(hl):
            continue
        acc += hl * hl
        t += 1
    if t == 0:
        return np.nan
    return math.sqrt(acc / (4.0 * t * math.log(2.0)))


@njit(cache=True, fastmath=True)
def classical_sigma(arr: np.ndarray) -> float:
    n = arr.shape[0]
    if n < 2:
        return np.nan
    t = 10.0
    hv = 0.0
    for i in range(n - 1):
        d = arr[i + 1] - arr[i]
        hv += d * d
    return math.sqrt(hv / t)


@njit(cache=True, fastmath=True)
def tv_mean(arr: np.ndarray) -> float:
    n = arr.shape[0]
    if n == 0:
        return np.nan
    s = 0.0
    for i in range(n):
        s += arr[i]
    return s / n


@njit(cache=True, fastmath=True)
def as_half_spreads(
    gamma: float,
    sigma: float,
    buy_a: float,
    buy_k: float,
    sell_a: float,
    sell_k: float,
    q_fix: float,
) -> tuple:
    # domain checks
    if not (math.isfinite(gamma) and math.isfinite(sigma)):
        return (np.nan, np.nan)
    if gamma <= 0.0 or sigma <= 0.0:
        return (np.nan, np.nan)
    if buy_a <= 0.0 or sell_a <= 0.0 or buy_k <= 0.0 or sell_k <= 0.0:
        return (np.nan, np.nan)
    base1 = 1.0 + gamma / sell_k
    base2 = 1.0 + gamma / buy_k
    if base1 <= 0.0 or base2 <= 0.0:
        return (np.nan, np.nan)
    term_bid = (sigma * sigma * gamma) / (2.0 * sell_k * sell_a)
    term_ask = (sigma * sigma * gamma) / (2.0 * buy_k * buy_a)
    if term_bid <= 0.0 or term_ask <= 0.0:
        return (np.nan, np.nan)
    bid_half = math.log(base1) / gamma + (q_fix + 0.5) * math.sqrt(term_bid * (base1 ** (1.0 + sell_k / gamma)))
    ask_half = math.log(base2) / gamma - (q_fix - 0.5) * math.sqrt(term_ask * (base2 ** (1.0 + buy_k / gamma)))
    if not (math.isfinite(bid_half) and math.isfinite(ask_half)):
        return (np.nan, np.nan)
    return (bid_half, ask_half)

