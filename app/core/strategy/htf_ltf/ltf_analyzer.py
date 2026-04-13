from dataclasses import dataclass
from typing import Optional

from app.core.indicators.atr import ATRIndicator
from app.core.indicators.ema import EMAIndicator
from app.core.indicators.rsi import RSIIndicator
from app.core.market.candle import Candle
from app.core.market.market_series import MarketSeries


# ──────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────

@dataclass
class LTFConfig:
    """
    EMA channel
    -----------
    ema_period  : Period for both High EMA (source=high) and Low EMA (source=low).
    ema_offset  : Lookback window used by EMAIndicator for slope helpers.

    RSI
    ---
    rsi_period       : RSI lookback period.
    rsi_slope_offset : Candle window for the internal regression slope.
    rsi_strict_slope : True  → slope > 0.1 AND R² > 0.6 (smooth, filtered).
                       False → any positive/negative slope qualifies.

    ATR
    ---
    atr_period        : ATR lookback period.
    atr_tp_multiplier : take_profit = entry ± (ATR × multiplier).
    """
    # EMA channel
    ema_period:  int = 9
    ema_offset:  int = 3

    # RSI
    rsi_period:       int  = 14
    rsi_slope_offset: int  = 3
    rsi_strict_slope: bool = False

    # ATR
    atr_period:        int   = 14
    atr_tp_multiplier: float = 2.0


# ──────────────────────────────────────────────────────────────────────
# LTF Analyzer
# ──────────────────────────────────────────────────────────────────────

class LTFAnalyzer:
    """
    Evaluates entry and exit conditions on a lower-timeframe (1m) feed.

    Indicators
    ----------
    EMA channel — two separate EMAs:
        High EMA : EMAIndicator(source="high")  →  dynamic resistance
        Low EMA  : EMAIndicator(source="low")   →  dynamic support

    RSI — entry requires BOTH a price-side-of-50 check AND a slope check.
    ATR — sizes the take-profit distance from entry.

    Entry conditions (long)
    -----------------------
    1. pullback — candle.low  ≤ ema_low   (price tagged or swept the low EMA)
    2. reclaim  — candle.close > ema_high  (bullish body close above high EMA)
    3. rsi > 50
    4. RSI slope is positive

    Entry conditions (short)  — exact mirror
    ----------------------------------------
    1. rally     — candle.high ≥ ema_high
    2. rejection — candle.close < ema_low
    3. rsi < 50
    4. RSI slope is negative

    Force-close conditions (long) — any one sufficient
    ---------------------------------------------------
    - candle.close < ema_low  (structure break below support)
    - RSI slope turns negative
    - Passed-in HA character is RANGING  (HTF regime degrades)

    Force-close conditions (short) — exact mirror

    Usage
    -----
        ltf = LTFAnalyzer(LTFConfig(ema_period=9, rsi_period=14))
        ltf.calculate(ltf_series)            # seed from history

        # on each new 1m candle:
        ltf.update(candle)
        if ltf.long_entry_ok(candle):
            sl = ltf.sl_long
            tp = ltf.tp_long(candle.close)
    """

    def __init__(self, config: LTFConfig = None):
        self.config = config or LTFConfig()
        cfg = self.config

        self._ema_high = EMAIndicator(
            period=cfg.ema_period, source="high", offset=cfg.ema_offset
        )
        self._ema_low = EMAIndicator(
            period=cfg.ema_period, source="low", offset=cfg.ema_offset
        )
        self._rsi = RSIIndicator(
            period=cfg.rsi_period, slope_offset=cfg.rsi_slope_offset
        )
        self._atr = ATRIndicator(period=cfg.atr_period)

    # ── Initialization ────────────────────────────────────────────────

    def calculate(self, series: MarketSeries) -> None:
        """Seed all indicators from a historical MarketSeries."""
        self._ema_high.calculate(series)
        self._ema_low.calculate(series)
        self._rsi.calculate(series)
        self._atr.calculate(series)

    # ── Live update ───────────────────────────────────────────────────

    def update(self, candle: Candle) -> None:
        """
        Advance all LTF indicators by one candle.
        Call this before any entry / exit checks on the same candle.
        """
        self._ema_high.update(candle)
        self._ema_low.update(candle)
        self._rsi.update(candle)
        self._atr.update(candle)

    # ── Entry conditions ──────────────────────────────────────────────

    def long_entry_ok(self, candle: Candle) -> bool:
        """
        True when all long conditions are met on this candle:
          pullback  — low touched or swept below Low EMA
          reclaim   — close back above High EMA (body close confirmation)
          rsi_above — RSI > 50
          rsi_slope — RSI regression slope is positive
        """
        if not self._ltf_ready():
            return False

        pullback  = candle.low  <= self._ema_low.current_value
        reclaim   = candle.close > self._ema_high.current_value
        rsi_above = self._rsi.current_value > 50
        rsi_slope = self._rsi.is_rising(strict=self.config.rsi_strict_slope)

        return pullback and reclaim and rsi_above and rsi_slope

    def short_entry_ok(self, candle: Candle) -> bool:
        """
        True when all short conditions are met on this candle:
          rally      — high touched or swept above High EMA
          rejection  — close back below Low EMA
          rsi_below  — RSI < 50
          rsi_slope  — RSI regression slope is negative
        """
        if not self._ltf_ready():
            return False

        rally      = candle.high  >= self._ema_high.current_value
        rejection  = candle.close  < self._ema_low.current_value
        rsi_below  = self._rsi.current_value < 50
        rsi_slope  = self._rsi.is_falling(strict=self.config.rsi_strict_slope)

        return rally and rejection and rsi_below and rsi_slope

    # ── Force-close conditions ────────────────────────────────────────

    def should_force_close_long(self, candle: Candle, htf_ranging: bool = False) -> bool:
        """
        True if any exit condition fires for an open long:
          - Structure break : close < Low EMA
          - RSI degradation : slope turns negative
          - HTF degradation : caller signals HA has become RANGING
        """
        if self._ema_low.current_value is None:
            return False

        structure_break = candle.close < self._ema_low.current_value
        rsi_lost        = self._rsi.is_falling(strict=self.config.rsi_strict_slope)

        return structure_break or rsi_lost or htf_ranging

    def should_force_close_short(self, candle: Candle, htf_ranging: bool = False) -> bool:
        """
        True if any exit condition fires for an open short:
          - Structure break : close > High EMA
          - RSI recovery    : slope turns positive
          - HTF degradation : caller signals HA has become RANGING
        """
        if self._ema_high.current_value is None:
            return False

        structure_break = candle.close  > self._ema_high.current_value
        rsi_recovered   = self._rsi.is_rising(strict=self.config.rsi_strict_slope)

        return structure_break or rsi_recovered or htf_ranging

    # ── Stop-loss and take-profit ──────────────────────────────────────

    @property
    def sl_long(self) -> Optional[float]:
        """Stop-loss for a long position = Low EMA."""
        return self._ema_low.current_value

    @property
    def sl_short(self) -> Optional[float]:
        """Stop-loss for a short position = High EMA."""
        return self._ema_high.current_value

    def tp_long(self, entry_price: float) -> float:
        """Take-profit for a long = entry + (ATR × multiplier)."""
        return entry_price + (self.atr or 0.0) * self.config.atr_tp_multiplier

    def tp_short(self, entry_price: float) -> float:
        """Take-profit for a short = entry - (ATR × multiplier)."""
        return entry_price - (self.atr or 0.0) * self.config.atr_tp_multiplier

    # ── Readable indicator values ─────────────────────────────────────

    @property
    def ema_high(self) -> Optional[float]:
        return self._ema_high.current_value

    @property
    def ema_low(self) -> Optional[float]:
        return self._ema_low.current_value

    @property
    def rsi(self) -> Optional[float]:
        return self._rsi.current_value

    @property
    def atr(self) -> Optional[float]:
        return self._atr.current_value

    @property
    def channel_width(self) -> Optional[float]:
        if self.ema_high is None or self.ema_low is None:
            return None
        return self.ema_high - self.ema_low

    # ── Debug ─────────────────────────────────────────────────────────

    def snapshot(self) -> dict:
        return {
            "ema_high":       _r(self.ema_high),
            "ema_low":        _r(self.ema_low),
            "channel_width":  _r(self.channel_width),
            "rsi":            _r(self.rsi, 2),
            "rsi_rising":     self._rsi.is_rising(strict=self.config.rsi_strict_slope),
            "rsi_falling":    self._rsi.is_falling(strict=self.config.rsi_strict_slope),
            "atr":            _r(self.atr),
        }

    # ── Internal ──────────────────────────────────────────────────────

    def _ltf_ready(self) -> bool:
        """Guard: all three indicator values must exist before any check."""
        return not any(
            v is None
            for v in (self._ema_high.current_value, self._ema_low.current_value, self._rsi.current_value)
        )


def _r(v: Optional[float], d: int = 5) -> Optional[float]:
    return round(v, d) if v is not None else None