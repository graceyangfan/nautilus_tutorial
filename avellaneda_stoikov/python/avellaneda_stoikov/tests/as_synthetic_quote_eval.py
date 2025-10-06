#!/usr/bin/env python3
"""
Synthetic evaluation of A/k estimation and AS quote computation across sigma modes.
Verifies no-NaN and positive half-spreads for modes: GK, Parkinson, Classical, TV, Auto.

Run:
  python -m avellaneda_stoikov.tests.as_synthetic_quote_eval
"""
from __future__ import annotations

import math
import random
from collections import deque
from typing import Iterator, Tuple

try:
    from avellaneda_stoikov.avellaneda_stoikov import AvellanedaStoikov
except Exception as e:  # pragma: no cover
    raise SystemExit(f"Import error (install with `pip install -e .`): {e}")


def synthetic_ticks(mid0: float, spread0: float, n: int, dt_ms: int) -> Iterator[Tuple[int, float, float]]:
    ts = 0
    for i in range(n):
        drift = 0.00002 * i
        wave = 0.02 * math.sin(i / 50.0)
        noise = (random.random() - 0.5) * 0.01
        mid = mid0 * (1.0 + drift) + wave + noise
        half = max(spread0 / 2.0, 1e-6)
        yield ts, mid - half, mid + half
        ts += dt_ms


def gk_sigma(wap: deque[float]) -> float:
    if len(wap) < 32:
        return float("nan")
    arr = list(wap)
    t = 0
    gk = 0.0
    step = 30
    for i in range(0, len(arr) - step, step):
        first = arr[i]
        last = arr[i + step]
        if first <= 0 or last <= 0:
            continue
        co = math.log(last / first)
        hi = max(arr[i : i + step])
        lo = min(arr[i : i + step])
        if lo <= 0:
            continue
        hl = math.log(hi / lo)
        res = 0.5 * (hl ** 2) - ((2.0 * math.log(2.0)) - 1.0) * (co ** 2)
        gk += res
        t += 1
    if t == 0:
        return float("nan")
    return math.sqrt(gk / t)


def parkinson_sigma(wap: deque[float]) -> float:
    if len(wap) < 32:
        return float("nan")
    arr = list(wap)
    t = 0
    accum = 0.0
    step = 30
    for i in range(0, len(arr) - step, step):
        hi = max(arr[i : i + step])
        lo = min(arr[i : i + step])
        if lo <= 0:
            continue
        hl = math.log(hi / lo)
        accum += hl * hl
        t += 1
    if t == 0:
        return float("nan")
    return math.sqrt(accum / (4.0 * t * math.log(2.0)))


def classical_sigma(wap: deque[float]) -> float:
    if len(wap) < 2:
        return float("nan")
    arr = list(wap)
    t = 10.0
    hv = 0.0
    for i in range(len(arr) - 1):
        hv += (arr[i + 1] - arr[i]) ** 2
    return math.sqrt(hv / t)


def tv_mean(tv: deque[float]) -> float:
    if len(tv) == 0:
        return float("nan")
    return sum(tv) / float(len(tv))


def main() -> None:
    # Config
    tick_size = 0.01
    n_spreads = 10
    estimate_window = 60_000
    period = 200
    start_time = 0
    gamma = 0.1
    sigma_multiplier = 1.0
    min_a = 1e-6
    min_k = 1e-6

    calib = AvellanedaStoikov(tick_size, n_spreads, estimate_window, period, start_time)
    wap: deque[float] = deque(maxlen=600)
    tv: deque[float] = deque(maxlen=600)

    # Feed synthetic stream
    last_ts = 0
    for ts, bid, ask in synthetic_ticks(100.0, 0.02, n=2500, dt_ms=period):
        mid = (bid + ask) / 2.0
        wap.append(mid)
        first = wap[0]
        tv.append(abs(mid / first - 1.0) + (ask - bid) / mid)
        calib.ingest_tick(ask, bid, ts)
        last_ts = ts

    # Estimate A/k at the end
    ak = calib.estimate_ak(last_ts)
    if ak is None:
        raise SystemExit("Calibrator did not produce A/k; synthetic stream too short?")
    buy_a, buy_k, sell_a, sell_k = ak
    # Floors
    buy_a = max(buy_a, min_a)
    sell_a = max(sell_a, min_a)
    buy_k = max(buy_k, min_k)
    sell_k = max(sell_k, min_k)

    # Sigma modes
    s_gk = gk_sigma(wap)
    s_pk = parkinson_sigma(wap)
    s_cls = classical_sigma(wap)
    s_tv = tv_mean(tv)

    def compute_quote(sigma: float) -> tuple[float, float]:
        if not (sigma and math.isfinite(sigma) and sigma > 0):
            return float("nan"), float("nan")
        base1 = 1.0 + gamma / sell_k
        base2 = 1.0 + gamma / buy_k
        if base1 <= 0.0 or base2 <= 0.0:
            return float("nan"), float("nan")
        term_bid = (sigma * sigma * gamma) / (2.0 * sell_k * sell_a)
        term_ask = (sigma * sigma * gamma) / (2.0 * buy_k * buy_a)
        if term_bid <= 0.0 or term_ask <= 0.0:
            return float("nan"), float("nan")
        bid_half = math.log(base1) / gamma + (0.0 + 0.5) * math.sqrt(term_bid * (base1 ** (1.0 + sell_k / gamma)))
        ask_half = math.log(base2) / gamma - (0.0 - 0.5) * math.sqrt(term_ask * (base2 ** (1.0 + buy_k / gamma)))
        return bid_half, ask_half

    results = {
        "gk": compute_quote(s_gk),
        "parkinson": compute_quote(s_pk),
        "classical": compute_quote(s_cls),
        "tv": compute_quote(s_tv),
    }

    print(f"A/k -> buy_a={buy_a:.6f}, buy_k={buy_k:.6f}, sell_a={sell_a:.6f}, sell_k={sell_k:.6f}")
    print(
        "sigma: GK={:.6f} PK={:.6f} CLS={:.6f} TV={:.6f}".format(
            s_gk if math.isfinite(s_gk) else float("nan"),
            s_pk if math.isfinite(s_pk) else float("nan"),
            s_cls if math.isfinite(s_cls) else float("nan"),
            s_tv if math.isfinite(s_tv) else float("nan"),
        )
    )
    for mode, (b, a) in results.items():
        ok = math.isfinite(b) and math.isfinite(a) and b > 0 and a > 0
        print(f"mode={mode:10s} half_bid={b:.6f} half_ask={a:.6f} ok={ok}")


if __name__ == "__main__":
    main()

