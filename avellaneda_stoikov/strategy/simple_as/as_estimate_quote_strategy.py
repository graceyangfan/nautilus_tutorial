# -------------------------------------------------------------------------------------------------
#  Minimal AS calibrator strategy for nautilus_trader
#  - Subscribes L1 order book deltas for a single instrument
#  - Feeds best bid/ask ticks to the Rust calibrator (A/k estimation)
#  - Computes AS half-spreads and logs quote prices (no order placement)
# -------------------------------------------------------------------------------------------------
from __future__ import annotations

import math
from collections import deque
from typing import Optional

from nautilus_trader.common.component import TimeEvent
from nautilus_trader.config import StrategyConfig
from nautilus_trader.core.data import Data
from nautilus_trader.model.book import OrderBook
from nautilus_trader.model.data import OrderBookDeltas
from nautilus_trader.model.enums import BookType
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.instruments import Instrument
from nautilus_trader.trading.strategy import Strategy


try:
    from avellaneda_stoikov.avellaneda_stoikov import AvellanedaStoikov  # type: ignore
except Exception:  # pragma: no cover
    from avellaneda_stoikov import AvellanedaStoikov  # type: ignore

from .quote_utils import select_sigma, compute_half_spreads


class SimpleASConfig(StrategyConfig, kw_only=True):
    instrument_id: str
    # Calibrator parameters
    n_spreads: int = 10
    estimate_window_ms: int = 60_000
    period_ms: int = 200
    # Quote parameters (purely for half-spread calculation; no orders placed)
    gamma: float = 0.1
    sigma_tick_period: int = 600
    sigma_multiplier: float = 1.0
    order_qty: float = 1.0  # only for inventory normalization; this strategy stays flat
    # Sigma mode: one of {"gk","parkinson","classical","tv","auto"}
    sigma_mode: str = "auto"
    # Minimum safety floors for A/k in quote computation
    min_a: float = 1e-6
    min_k: float = 1e-6
    # Optional safety caps for half-spread (in ticks)
    half_spread_cap_ticks: float | None = None
    min_half_spread_ticks: float | None = None
    # Timer
    timer_interval_ms: int = 200


class SimpleASEstimateQuoteStrategy(Strategy):
    def __init__(self, config: SimpleASConfig) -> None:
        super().__init__(config)
        self.instrument_id = InstrumentId.from_str(config.instrument_id)
        self.interval = config.timer_interval_ms

        self.instrument: Optional[Instrument] = None
        self.book: Optional[OrderBook] = None

        # Windows for sigma estimation (align with avellaneda_stoikov.rs helpers)
        self._wap: deque[float] = deque(maxlen=config.sigma_tick_period)
        self._spread_rel: deque[float] = deque(maxlen=config.sigma_tick_period)
        self._tv: deque[float] = deque(maxlen=config.sigma_tick_period)

        # Rust calibrator instance (pyo3)
        self.calibrator: Optional[AvellanedaStoikov] = None

        # Cached last mid for logging
        self._last_mid: float = math.nan

    # Lifecycle ---------------------------------------------------------------------------------
    def on_start(self) -> None:
        self.instrument = self.cache.instrument(self.instrument_id)
        if self.instrument is None:
            self.log.error(f"Instrument not found: {self.instrument_id}")
            self.stop()
            return

        # Local order book to accumulate deltas
        self.book = OrderBook(instrument_id=self.instrument.id, book_type=BookType.L1_TBBO)
        self.subscribe_order_book_deltas(self.instrument.id)

        # Instantiate calibrator
        tick_size = float(self.instrument.price_increment.as_double())
        self.calibrator = AvellanedaStoikov(
            tick_size,
            int(self.config.n_spreads),
            int(self.config.estimate_window_ms),
            int(self.config.period_ms),
            int(self.clock.timestamp_ms()),
        )
        self.clock.set_timer("simple_as_timer", self.interval, callback=self.on_time_event)
        self.log.info(
            f"Started SimpleASEstimateQuoteStrategy for {self.instrument.id}, tick_size={tick_size}"
        )

    def on_stop(self) -> None:
        if self.instrument:
            self.unsubscribe_order_book_deltas(self.instrument.id)

    # Data --------------------------------------------------------------------------------------
    def on_data(self, data: Data) -> None:
        if isinstance(data, OrderBookDeltas):
            if self.book is None or data.instrument_id != self.instrument_id:
                return
            self.book.apply_deltas(data)
            # Update WAP window when best prices present
            best_bid = self.book.best_bid_price()
            best_ask = self.book.best_ask_price()
            if best_bid is None or best_ask is None:
                return
            bid = float(best_bid.as_double())
            ask = float(best_ask.as_double())
            # Use equal weights as L1 TBBO doesn't carry sizes; equals mid
            wap = (bid + ask) / 2.0
            self._wap.append(wap)
            self._last_mid = wap
            # Maintain relative spread and TV (as in avellaneda_stoikov.rs)
            if wap > 0:
                self._spread_rel.append((ask - bid) / wap)
                first = self._wap[0] if len(self._wap) > 0 else wap
                tv = abs(wap / first - 1.0) + ((ask - bid) / wap)
                self._tv.append(tv)

            # Feed calibrator with the current tick
            if self.calibrator:
                self.calibrator.ingest_tick(ask, bid, int(self.clock.timestamp_ms()))

    def on_time_event(self, event: TimeEvent) -> None:
        # Only compute when we have enough WAP points
        if len(self._wap) < 30 or not math.isfinite(self._last_mid):
            return
        if self.book is None:
            return
        best_bid = self.book.best_bid_price()
        best_ask = self.book.best_ask_price()
        if best_bid is None or best_ask is None:
            return

        ts = int(self.clock.timestamp_ms())
        # Try estimate A/k
        ak = None
        if self.calibrator:
            ak = self.calibrator.estimate_ak(ts)
        if ak is None:
            # Not yet ready
            return
        buy_a, buy_k, sell_a, sell_k = ak

        # Floors to avoid domain errors (align with safety in production)
        min_a = float(self.config.min_a)
        min_k = float(self.config.min_k)
        buy_a = max(buy_a, min_a)
        sell_a = max(sell_a, min_a)
        buy_k = max(buy_k, min_k)
        sell_k = max(sell_k, min_k)

        # Estimate sigma (Garman–Klass)
        sigma_mode_cfg = (self.config.sigma_mode or "auto").lower()
        sigma, sigma_mode_used, sigma_details = select_sigma(self._wap, self._tv, sigma_mode_cfg)
        if not math.isfinite(sigma):
            return
        sigma *= float(self.config.sigma_multiplier)

        # Inventory is zero in this simple strategy
        q_fix = 0.0
        gamma = float(self.config.gamma)
        bid_half, ask_half, ok = compute_half_spreads(
            gamma,
            sigma,
            buy_a,
            buy_k,
            sell_a,
            sell_k,
            q_fix,
            min_a=float(self.config.min_a),
            min_k=float(self.config.min_k),
        )
        if not ok:
            return

        # Quote prices around current mid
        mid = self._last_mid
        # Optional half-spread clamping in ticks
        tsz = float(self.instrument.price_increment.as_double())
        hb_tick = bid_half / tsz
        ha_tick = ask_half / tsz
        min_h = self.config.min_half_spread_ticks
        cap_h = self.config.half_spread_cap_ticks
        if min_h is not None and hb_tick < min_h:
            hb_tick = min_h
        if min_h is not None and ha_tick < min_h:
            ha_tick = min_h
        if cap_h is not None and hb_tick > cap_h:
            hb_tick = cap_h
        if cap_h is not None and ha_tick > cap_h:
            ha_tick = cap_h
        bid_px = self.instrument.make_price(mid - hb_tick * tsz)
        ask_px = self.instrument.make_price(mid + ha_tick * tsz)

        self.log.info(
            f"AS estimate ts={ts} mid={mid:.6f} "
            f"A/k(buy)={buy_a:.6f}/{buy_k:.6f} A/k(sell)={sell_a:.6f}/{sell_k:.6f} "
            f"sigma_mode={sigma_mode_used} sigma={sigma:.6f}"
        )
        self.log.info(
            f"Quotes bid={bid_px.as_double():.6f} ask={ask_px.as_double():.6f}"
        )

    # Helpers -----------------------------------------------------------------------------------
    # All numerical helpers moved into quote_utils (optionally jitted)
