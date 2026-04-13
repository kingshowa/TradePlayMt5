from dataclasses import dataclass
from enum import Enum
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
    Optimized 1m execution config for a HTF/LTF trend-following scalper.

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
    rsi_long_min     : Minimum RSI required for long entries.
    rsi_short_max    : Maximum RSI required for short entries.

    ATR / entry quality
    -------------------
    atr_period             : ATR lookback period.
    atr_tp_multiplier      : take_profit = entry ± (ATR × multiplier).
    min_atr                : Minimum tradable ATR. Prevents dead-market entries.
    min_reclaim_body_atr   : Minimum reclaim/rejection candle body as a fraction of ATR.
    min_channel_atr_ratio  : Minimum EMA-channel width relative to ATR.
    max_setup_bars         : Maximum age for a pullback/rally setup.
    max_pullback_atr       : Maximum overshoot depth from the channel before invalidation.
    close_location_min     : Long candles must close in at least this fraction of their range.
    close_location_max     : Short candles must close in at most this fraction of their range.
    require_rsi_reset      : If True, long requires RSI to have dipped to/under 50 during the
                             pullback and short requires RSI to have lifted to/above 50 during
                             the rally.
    force_close_confirm_bars : Require this many consecutive adverse momentum bars before using
                               RSI alone to force-close.
    """

    # EMA channel
    ema_period: int = 13
    ema_offset: int = 3

    # RSI
    rsi_period: int = 14
    rsi_slope_offset: int = 3
    rsi_strict_slope: bool = False
    rsi_long_min: float = 50.0
    rsi_short_max: float = 50.0

    # ATR / trade quality
    atr_period: int = 14
    atr_tp_multiplier: float = 2.0
    min_atr: float = 0.0
    min_reclaim_body_atr: float = 0.15
    min_channel_atr_ratio: float = 0.05
    max_setup_bars: int = 6
    max_pullback_atr: float = 1.50
    close_location_min: float = 0.60
    close_location_max: float = 0.40
    require_rsi_reset: bool = False
    force_close_confirm_bars: int = 2


# ──────────────────────────────────────────────────────────────────────
# Internal state
# ──────────────────────────────────────────────────────────────────────

class LTFEntryState(Enum):
    IDLE = "IDLE"
    PULLBACK_LONG = "PULLBACK_LONG"
    RALLY_SHORT = "RALLY_SHORT"


# ──────────────────────────────────────────────────────────────────────
# LTF Analyzer
# ──────────────────────────────────────────────────────────────────────

class LTFAnalyzer:
    """
    Evaluates entry and exit conditions on a lower-timeframe (1m) feed.

    Design goals
    ------------
    1. Stateful pullback/rally detection.
    2. Reclaim/rejection must happen AFTER setup starts (no same-candle setup+entry).
    3. EMA channel defines structure and invalidation.
    4. RSI confirms momentum quality.
    5. ATR normalizes body quality, channel quality, and TP distance.

    Long setup lifecycle
    --------------------
    IDLE → PULLBACK_LONG
        Triggered when price crosses below the High EMA after previously trading on/above it.

    PULLBACK_LONG → BUY-ready
        A later candle closes back above High EMA and passes quality filters:
          - RSI above threshold
          - RSI slope positive
          - candle body strong enough vs ATR
          - channel width not too compressed vs ATR
          - setup age still fresh
          - pullback depth not too extreme

    Setup invalidation (long)
    -------------------------
    - close < Low EMA
    - setup too old
    - ATR too low
    - overshoot depth too large

    Short is the exact mirror image.
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

        # Stateful setup tracking
        self._state: LTFEntryState = LTFEntryState.IDLE
        self._setup_bars: int = 0
        self._pullback_low: Optional[float] = None
        self._rally_high: Optional[float] = None
        self._min_rsi_in_setup: Optional[float] = None
        self._max_rsi_in_setup: Optional[float] = None

        # Entry-ready flags produced in update() and consumed by *_entry_ok()
        self._long_entry_ready: bool = False
        self._short_entry_ready: bool = False
        self._last_trigger_side: Optional[str] = None

        # Previous values for cross detection
        self._prev_close: Optional[float] = None
        self._prev_ema_high: Optional[float] = None
        self._prev_ema_low: Optional[float] = None

        # Exit-quality tracking
        self._long_adverse_bars: int = 0
        self._short_adverse_bars: int = 0

    # ── Initialization ────────────────────────────────────────────────

    def calculate(self, series: MarketSeries) -> None:
        """Seed all indicators from a historical MarketSeries."""
        self._ema_high.calculate(series)
        self._ema_low.calculate(series)
        self._rsi.calculate(series)
        self._atr.calculate(series)

        last = series.last()
        if last is not None:
            self._prev_close = last.close
            self._prev_ema_high = self._ema_high.current_value
            self._prev_ema_low = self._ema_low.current_value

        self._reset_setup()
        self._long_adverse_bars = 0
        self._short_adverse_bars = 0

    # ── Live update ───────────────────────────────────────────────────

    def update(self, candle: Candle) -> None:
        """
        Advance all LTF indicators by one candle.
        Call this before any entry / exit checks on the same candle.
        """
        self._long_entry_ready = False
        self._short_entry_ready = False
        self._last_trigger_side = None

        self._ema_high.update(candle)
        self._ema_low.update(candle)
        self._rsi.update(candle)
        self._atr.update(candle)

        if not self._ltf_ready():
            return

        self._update_force_close_counters(candle)
        self._advance_state_machine(candle)

        # Preserve state for next cross detection
        self._prev_close = candle.close
        self._prev_ema_high = self._ema_high.current_value
        self._prev_ema_low = self._ema_low.current_value

    # ── Entry conditions ──────────────────────────────────────────────

    def long_entry_ok(self, candle: Candle) -> bool:
        """
        True only once, on the reclaim-confirmation candle produced by update().
        This prevents same-candle pullback+reclaim entries.
        """
        if self._long_entry_ready:
            self._long_entry_ready = False
            self._last_trigger_side = "BUY"
            self._reset_setup()
            return True
        return False

    def short_entry_ok(self, candle: Candle) -> bool:
        """
        True only once, on the rejection-confirmation candle produced by update().
        This prevents same-candle rally+rejection entries.
        """
        if self._short_entry_ready:
            self._short_entry_ready = False
            self._last_trigger_side = "SELL"
            self._reset_setup()
            return True
        return False

    # ── Force-close conditions ────────────────────────────────────────

    def should_force_close_long(self, candle: Candle, htf_ranging: bool = False) -> bool:
        """
        True if any exit condition fires for an open long:
          - Structure break : close < Low EMA
          - Momentum failure: RSI < 50 and adverse momentum persisted for N bars
          - HTF degradation : caller signals HA has become RANGING
        """
        if self._ema_low.current_value is None:
            return False

        structure_break = candle.close < self._ema_low.current_value
        momentum_fail = (
            self._rsi.current_value is not None
            and self._rsi.current_value < 50
            and self._long_adverse_bars >= self.config.force_close_confirm_bars
        )

        return structure_break or momentum_fail or htf_ranging

    def should_force_close_short(self, candle: Candle, htf_ranging: bool = False) -> bool:
        """
        True if any exit condition fires for an open short:
          - Structure break : close > High EMA
          - Momentum recovery: RSI > 50 and recovery persisted for N bars
          - HTF degradation : caller signals HA has become RANGING
        """
        if self._ema_high.current_value is None:
            return False

        structure_break = candle.close > self._ema_high.current_value
        momentum_fail = (
            self._rsi.current_value is not None
            and self._rsi.current_value > 50
            and self._short_adverse_bars >= self.config.force_close_confirm_bars
        )

        return structure_break or momentum_fail or htf_ranging

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

    @property
    def entry_state(self) -> str:
        return self._state.value

    # ── Debug ─────────────────────────────────────────────────────────

    def snapshot(self) -> dict:
        return {
            "state": self._state.value,
            "setup_bars": self._setup_bars,
            "ema_high": _r(self.ema_high),
            "ema_low": _r(self.ema_low),
            "channel_width": _r(self.channel_width),
            "rsi": _r(self.rsi, 2),
            "min_rsi_in_setup": _r(self._min_rsi_in_setup, 2),
            "max_rsi_in_setup": _r(self._max_rsi_in_setup, 2),
            "rsi_rising": self._rsi.is_rising(strict=self.config.rsi_strict_slope),
            "rsi_falling": self._rsi.is_falling(strict=self.config.rsi_strict_slope),
            "atr": _r(self.atr),
            "pullback_low": _r(self._pullback_low),
            "rally_high": _r(self._rally_high),
            "long_entry_ready": self._long_entry_ready,
            "short_entry_ready": self._short_entry_ready,
            "last_trigger_side": self._last_trigger_side,
            "long_adverse_bars": self._long_adverse_bars,
            "short_adverse_bars": self._short_adverse_bars,
        }

    # ── Internal: state machine ───────────────────────────────────────

    def _advance_state_machine(self, candle: Candle) -> None:
        if self._state == LTFEntryState.IDLE:
            self._try_start_long_pullback(candle)
            if self._state == LTFEntryState.IDLE:
                self._try_start_short_rally(candle)
            return

        if self._state == LTFEntryState.PULLBACK_LONG:
            self._update_long_setup(candle)
            return

        if self._state == LTFEntryState.RALLY_SHORT:
            self._update_short_setup(candle)
            return

    def _try_start_long_pullback(self, candle: Candle) -> None:
        if self.ema_high is None or self.ema_low is None:
            return

        crossed_below_high_ema = (
            candle.low < self.ema_high
            and self._prev_close is not None
            and self._prev_ema_high is not None
            and self._prev_close >= self._prev_ema_high
        )

        if crossed_below_high_ema:
            self._state = LTFEntryState.PULLBACK_LONG
            self._setup_bars = 0
            self._pullback_low = candle.low
            self._rally_high = None
            self._min_rsi_in_setup = self.rsi
            self._max_rsi_in_setup = self.rsi

    def _try_start_short_rally(self, candle: Candle) -> None:
        if self.ema_high is None or self.ema_low is None:
            return

        crossed_above_low_ema = (
            candle.high > self.ema_low
            and self._prev_close is not None
            and self._prev_ema_low is not None
            and self._prev_close <= self._prev_ema_low
        )

        if crossed_above_low_ema:
            self._state = LTFEntryState.RALLY_SHORT
            self._setup_bars = 0
            self._rally_high = candle.high
            self._pullback_low = None
            self._min_rsi_in_setup = self.rsi
            self._max_rsi_in_setup = self.rsi

    def _update_long_setup(self, candle: Candle) -> None:
        self._setup_bars += 1
        self._pullback_low = min(self._pullback_low, candle.low) if self._pullback_low is not None else candle.low
        self._min_rsi_in_setup = self._min_value(self._min_rsi_in_setup, self.rsi)
        self._max_rsi_in_setup = self._max_value(self._max_rsi_in_setup, self.rsi)

        if self._invalidate_long_setup(candle):
            self._reset_setup()
            return

        reclaim_confirmed = candle.close > self.ema_high
        if reclaim_confirmed and self._setup_bars >= 1 and self._long_entry_filters_pass(candle):
            self._long_entry_ready = True

    def _update_short_setup(self, candle: Candle) -> None:
        self._setup_bars += 1
        self._rally_high = max(self._rally_high, candle.high) if self._rally_high is not None else candle.high
        self._min_rsi_in_setup = self._min_value(self._min_rsi_in_setup, self.rsi)
        self._max_rsi_in_setup = self._max_value(self._max_rsi_in_setup, self.rsi)

        if self._invalidate_short_setup(candle):
            self._reset_setup()
            return

        rejection_confirmed = candle.close < self.ema_low
        if rejection_confirmed and self._setup_bars >= 1 and self._short_entry_filters_pass(candle):
            self._short_entry_ready = True

    # ── Internal: entry-quality filters ───────────────────────────────

    def _long_entry_filters_pass(self, candle: Candle) -> bool:
        atr = self.atr or 0.0
        if atr < self.config.min_atr:
            return False

        body = abs(candle.close - candle.open)
        if body < atr * self.config.min_reclaim_body_atr:
            return False

        if self.channel_width is None or self.channel_width < atr * self.config.min_channel_atr_ratio:
            return False

        if self.rsi is None or self.rsi <= self.config.rsi_long_min:
            return False

        if not self._rsi.is_rising(strict=self.config.rsi_strict_slope):
            return False

        if self.config.require_rsi_reset and self._min_rsi_in_setup is not None:
            if self._min_rsi_in_setup > self.config.rsi_long_min:
                return False

        if self._close_location_value(candle) < self.config.close_location_min:
            return False

        return True

    def _short_entry_filters_pass(self, candle: Candle) -> bool:
        atr = self.atr or 0.0
        if atr < self.config.min_atr:
            return False

        body = abs(candle.close - candle.open)
        if body < atr * self.config.min_reclaim_body_atr:
            return False

        if self.channel_width is None or self.channel_width < atr * self.config.min_channel_atr_ratio:
            return False

        if self.rsi is None or self.rsi >= self.config.rsi_short_max:
            return False

        if not self._rsi.is_falling(strict=self.config.rsi_strict_slope):
            return False

        if self.config.require_rsi_reset and self._max_rsi_in_setup is not None:
            if self._max_rsi_in_setup < self.config.rsi_short_max:
                return False

        if self._close_location_value(candle) > self.config.close_location_max:
            return False

        return True

    # ── Internal: setup invalidation ──────────────────────────────────

    def _invalidate_long_setup(self, candle: Candle) -> bool:
        if self.ema_low is None or self.ema_high is None:
            return True

        if (self.atr or 0.0) < self.config.min_atr:
            return True

        if candle.close < self.ema_low:
            return True

        if self._setup_bars > self.config.max_setup_bars:
            return True

        if self._pullback_low is not None and self.atr is not None:
            depth = self.ema_high - self._pullback_low
            if depth > self.atr * self.config.max_pullback_atr:
                return True

        return False

    def _invalidate_short_setup(self, candle: Candle) -> bool:
        if self.ema_low is None or self.ema_high is None:
            return True

        if (self.atr or 0.0) < self.config.min_atr:
            return True

        if candle.close > self.ema_high:
            return True

        if self._setup_bars > self.config.max_setup_bars:
            return True

        if self._rally_high is not None and self.atr is not None:
            depth = self._rally_high - self.ema_low
            if depth > self.atr * self.config.max_pullback_atr:
                return True

        return False

    # ── Internal: helpers ─────────────────────────────────────────────

    def _update_force_close_counters(self, candle: Candle) -> None:
        if self._ltf_ready() and self.rsi is not None:
            long_adverse = candle.close < (self.ema_low or candle.close) or (
                self.rsi < 50 and self._rsi.is_falling(strict=self.config.rsi_strict_slope)
            )
            short_adverse = candle.close > (self.ema_high or candle.close) or (
                self.rsi > 50 and self._rsi.is_rising(strict=self.config.rsi_strict_slope)
            )

            self._long_adverse_bars = self._long_adverse_bars + 1 if long_adverse else 0
            self._short_adverse_bars = self._short_adverse_bars + 1 if short_adverse else 0

    def _reset_setup(self) -> None:
        self._state = LTFEntryState.IDLE
        self._setup_bars = 0
        self._pullback_low = None
        self._rally_high = None
        self._min_rsi_in_setup = None
        self._max_rsi_in_setup = None
        self._long_entry_ready = False
        self._short_entry_ready = False

    def _ltf_ready(self) -> bool:
        """Guard: all indicator values must exist before any check."""
        return not any(
            v is None
            for v in (
                self._ema_high.current_value,
                self._ema_low.current_value,
                self._rsi.current_value,
                self._atr.current_value,
            )
        )

    @staticmethod
    def _min_value(current: Optional[float], new: Optional[float]) -> Optional[float]:
        if current is None:
            return new
        if new is None:
            return current
        return min(current, new)

    @staticmethod
    def _max_value(current: Optional[float], new: Optional[float]) -> Optional[float]:
        if current is None:
            return new
        if new is None:
            return current
        return max(current, new)

    @staticmethod
    def _close_location_value(candle: Candle) -> float:
        rng = candle.high - candle.low
        if rng <= 0:
            return 0.5
        return (candle.close - candle.low) / rng


def _r(v: Optional[float], d: int = 5) -> Optional[float]:
    return round(v, d) if v is not None else None
