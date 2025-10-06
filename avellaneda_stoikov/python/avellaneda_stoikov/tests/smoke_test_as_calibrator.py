#!/usr/bin/env python3
"""
Minimal smoke test for the pyo3 AS calibrator with synthetic L1 data.

Usage:
  1) Build/install the extension (in editable mode is fine):
       cd avellaneda_stoikov
       python -m pip install -U pip setuptools wheel setuptools-rust
       python -m pip install -e .
  2) Run this script:
       python -m avellaneda_stoikov.tests.smoke_test_as_calibrator
"""

import math
import random
from typing import Iterator, Tuple

try:
    # Package layout uses nested module name per setup.py
    from avellaneda_stoikov.avellaneda_stoikov import AvellanedaStoikov
except Exception as e:  # pragma: no cover - import error diagnostics
    raise SystemExit(
        "Failed to import avellaneda_stoikov extension.\n"
        "Make sure you've installed the package (pip install -e .) from the avellaneda_stoikov folder.\n"
        f"Import error: {e}"
    )


def synthetic_ticks(
    mid0: float,
    spread0: float,
    n: int,
    dt_ms: int,
) -> Iterator[Tuple[int, float, float]]:
    """Generate (ts_ms, bid, ask) ticks around a drifting mid with mild noise."""
    ts = 0
    mid = mid0
    for i in range(n):
        # Slow drift + small sinusoidal + random micro noise
        drift = 0.00002 * i
        wave = 0.02 * math.sin(i / 50.0)
        noise = (random.random() - 0.5) * 0.01
        mid = mid0 * (1.0 + drift) + wave + noise

        # Keep a small positive spread
        half = max(spread0 / 2.0, 1e-6)
        bid = mid - half
        ask = mid + half

        yield ts, bid, ask
        ts += dt_ms


def main() -> None:
    # Estimator config
    tick_size = 0.01       # used internally as spread step
    n_spreads = 10
    estimate_window = 60_000  # ms
    period = 200               # ms (dt)
    start_time = 0

    calibrator = AvellanedaStoikov(
        tick_size,
        n_spreads,
        estimate_window,
        period,
        start_time,
    )

    print("Created calibrator. initialized?", calibrator.initialized())

    # Feed synthetic ticks
    last_non_zero = None
    last_ts = 0
    for ts, bid, ask in synthetic_ticks(100.0, 0.02, n=2_000, dt_ms=period):
        # new API: separate ingest and estimate
        calibrator.ingest_tick(ask, bid, ts)
        last_ts = ts

        if ts % 5_000 == 0:
            ak = calibrator.estimate_ak(ts)
            if ak is not None:
                last_non_zero = ak
                ba, bk, sa, sk = ak
                print(f"ts={ts} A/k -> buy_a={ba:.6f}, buy_k={bk:.6f}, sell_a={sa:.6f}, sell_k={sk:.6f}")

    # Backward compatible API check (will just call estimate internally)
    ba, bk, sa, sk = calibrator.calculate_intensity_info(ask=101.0, bid=99.0, ts=last_ts + 10_000)
    print("compat calculate_intensity_info:", ba, bk, sa, sk)

    if last_non_zero is None and (ba, bk, sa, sk) == (0.0, 0.0, 0.0, 0.0):
        print("WARNING: did not obtain positive A/k in this run (may happen if synthetic flow too benign).")
    else:
        print("OK: obtained at least one non-zero A/k estimate.")


if __name__ == "__main__":
    main()
