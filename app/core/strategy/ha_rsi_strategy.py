from typing import Optional

from app.core.indicators.atr import ATRIndicator
from app.core.indicators.ha import HeikinAshiIndicator, TrendState, MomentumState
from app.core.indicators.rsi import RSIIndicator
from app.core.market.candle import Candle
from app.core.market.market_series import MarketSeries
from app.core.strategy.trade_signal import StrategySignal


class HeikinAshiRsiStrategy:
    """
    HA + RSI swing strategy.

    Core philosophy
    ───────────────
    Ride a trend for as long as it is gaining momentum.
    Enter on the first candle where momentum turns GAINING within a trend —
    not necessarily the flip candle — so slow-starting trends are not missed.
    At most one entry is taken per trend run.

    Entry rules  (ALL must be true on the same candle)
    ───────────────────────────────────────────────────
    1. An active HA trend exists (BULLISH or BEARISH).
    2. This trend has not already been traded this run.
    3. Momentum is GAINING  (last_candle_growth > avg_candle_growth).
    4. RSI confirms direction     (long: RSI > 50  |  short: RSI < 50).
    5. RSI is not overextended   (long: RSI < 70  |  short: RSI > 30).

    Exit rules  — first condition that fires wins
    ─────────────────────────────────────────────
    Priority 1 — SL breach:
        Raw candle low < entry SL (long)  |  raw candle high > entry SL (short).
        Trade closes immediately; strategy re-scans for a new entry on the
        same candle.

    Priority 2 — Trend flip:
        HA trend direction reverses while in trade.
        The move is structurally over; close without waiting for RSI.

    Priority 3 — Momentum + RSI confirmed deterioration:
        Momentum = LOSING  AND  RSI weakening with strict regression check
        (slope + R² both cross their thresholds).
        This is the normal swing exit — both indicators must agree.
    """

    def __init__(
        self,
        market_series: MarketSeries,
        ha_max_candles: int = 500,
        ha_max_trends: int = 100,
        rsi_period: int = 14,
        rsi_slope_offset: int = 3,
        atr_period: int = 14,
        rsi_bull_min: float = 35.0,    # RSI must be above this for a long entry 50
        rsi_bear_max: float = 65.0,    # RSI must be below this for a short entry 50
        rsi_overbought: float = 70.0,  # long entries blocked at or above this
        rsi_oversold: float = 30.0,    # short entries blocked at or below this
    ):
        # ── Indicators ────────────────────────────────────────────────────────
        self._ha = HeikinAshiIndicator(
            max_candles=ha_max_candles,
            max_trends=ha_max_trends,
        )
        self._rsi = RSIIndicator(
            period=rsi_period,
            slope_offset=rsi_slope_offset,
        )
        self._atr = ATRIndicator(period=atr_period)

        # ── RSI level thresholds ───────────────────────────────────────────────
        self._rsi_bull_min = rsi_bull_min
        self._rsi_bear_max = rsi_bear_max
        self._rsi_overbought_level = rsi_overbought
        self._rsi_oversold_level = rsi_oversold

        # ── Trade state ───────────────────────────────────────────────────────
        # Holds the full StrategySignal of the open trade, or None when flat.
        # Storing the signal object gives access to the original SL, direction,
        # and all entry metadata without maintaining separate fields.
        self._open_trade: Optional[StrategySignal] = None

        # Tracks which trend run has been entered so the strategy never takes
        # a second entry inside the same trend. Keyed on (state, start_open)
        # which uniquely identifies a trend run because HA open is continuous.
        self._traded_trend_id: Optional[tuple] = None

        # ── Bootstrap indicators from historical data ─────────────────────────
        self._ha.calculate(market_series)
        self._rsi.calculate(market_series)
        self._atr.calculate(market_series)

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    @property
    def in_trade(self) -> bool:
        return self._open_trade is not None

    @property
    def open_trade(self) -> Optional[StrategySignal]:
        """The StrategySignal of the currently open trade, or None if flat."""
        return self._open_trade

    def update(self, candle: Candle) -> Optional[StrategySignal]:
        """
        Feed one new (or live-updated) candle into the strategy.

        Returns
        ───────
        StrategySignal(signal='BUY'|'SELL')  — new trade opened.
        StrategySignal(signal='CLOSE')       — open trade closed.
        None                                 — no actionable event.

        Note: an SL breach closes the trade and then immediately re-scans for
        a fresh entry on the same candle, so a CLOSE and a new BUY/SELL can
        both occur within a single update() call. In that case only the new
        entry signal is returned; the caller's position manager must treat the
        absence of an open trade as an implicit close.
        """
        self._ha.update(candle)
        self._rsi.update(candle)
        self._atr.update(candle)

        # ── Manage open trade ─────────────────────────────────────────────────
        if self._open_trade is not None:
            close_signal = self._check_exit(candle)
            if close_signal is not None:
                # Trade is now flat. Re-scan immediately — a flip exit might
                # coincide with a valid entry in the new direction.
                entry_signal = self._scan_entry(candle)
                return entry_signal if entry_signal is not None else close_signal

            return None

        # ── Flat — scan for entry ─────────────────────────────────────────────
        return self._scan_entry(candle)

    # ──────────────────────────────────────────────────────────────────────────
    # Exit logic
    # ──────────────────────────────────────────────────────────────────────────

    def _check_exit(self, candle: Candle) -> Optional[StrategySignal]:
        """
        Evaluate all exit conditions in priority order.
        Returns a CLOSE signal and resets trade state if any condition fires.
        Returns None if the trade should remain open.
        """
        direction = self._open_trade.signal

        # Priority 1 — SL breach
        if self._sl_breached(candle, direction):
            return self._close_trade(candle, reason="SL breach")

        # Priority 2 — Trend flip
        if self._trend_flipped_against_trade(direction):
            return self._close_trade(candle, reason="Trend flip")

        # Priority 3 — Momentum LOSING confirmed by RSI regression
        if self._momentum_losing() and self._rsi_weakening(direction):
            return self._close_trade(candle, reason="Momentum loss + RSI weakening")

        return None

    def _sl_breached(self, candle: Candle, direction: str) -> bool:
        sl = self._open_trade.sl
        if sl is None:
            return False
        if direction == "BUY":
            return candle.low < sl
        return candle.high > sl

    def _trend_flipped_against_trade(self, direction: str) -> bool:
        trend = self._ha.current_trend
        if trend is None:
            return False
        if direction == "BUY":
            return trend.state == TrendState.BEARISH
        return trend.state == TrendState.BULLISH

    def _momentum_losing(self) -> bool:
        trend = self._ha.current_trend
        return trend is not None and trend.momentum == MomentumState.LOSING

    def _rsi_weakening(self, direction: str) -> bool:
        """
        strict=True: slope threshold + R² > 0.6.
        Prevents a single noisy LOSING candle from closing a healthy trade.
        """
        if direction == "BUY":
            return self._rsi.is_falling(strict=True)
        return self._rsi.is_rising(strict=True)

    def _close_trade(self, candle: Candle, reason: str) -> StrategySignal:
        trend = self._ha.current_trend
        rsi_val = self._rsi.current_value
        original = self._open_trade

        full_reason = (
            f"CLOSE {original.signal} — {reason}. "
            f"HA momentum={trend.momentum.value} "
            f"({trend.last_candle_growth_pct:.3f}% vs avg {trend.avg_candle_growth_pct:.3f}%), "
            f"RSI={rsi_val:.2f}"
        )
        pattern = (
            f"HA_trend={trend.state.value} "
            f"candles={trend.candle_count} "
            f"total_growth={trend.growth_pct:.3f}% "
            f"RSI={rsi_val:.2f}"
        )

        self._open_trade = None
        self._traded_trend_id = None

        return StrategySignal(
            signal="CLOSE",
            strategy_type="TRENDING",
            reason=full_reason,
            pattern_name=pattern,
            atr=self._atr.current_value,
            sl=None,
            tp=None,
            candle=candle,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Entry logic
    # ──────────────────────────────────────────────────────────────────────────

    def _scan_entry(self, candle: Candle) -> Optional[StrategySignal]:
        if self._is_long_entry():
            return self._open_entry("BUY", candle)
        if self._is_short_entry():
            return self._open_entry("SELL", candle)
        return None

    def _is_long_entry(self) -> bool:
        return (
            self._trend_is_bullish()
            # and self._trend_not_yet_traded()
            and self._momentum_gaining()
            and self._rsi_confirms_long()
            and not self._rsi_overbought()
        )

    def _is_short_entry(self) -> bool:
        return (
            self._trend_is_bearish()
            # and self._trend_not_yet_traded()
            and self._momentum_gaining()
            and self._rsi_confirms_short()
            and not self._rsi_oversold()
        )

    def _trend_is_bullish(self) -> bool:
        trend = self._ha.current_trend
        return trend is not None and trend.state == TrendState.BULLISH

    def _trend_is_bearish(self) -> bool:
        trend = self._ha.current_trend
        return trend is not None and trend.state == TrendState.BEARISH

    def _trend_not_yet_traded(self) -> bool:
        """
        True when the current trend run has not already been entered.
        Prevents re-entry on every subsequent GAINING candle within the same
        trend. The ID is reset only when a new trend run begins (different
        state or different start_open), so a stopped-out trade does not
        re-enter the same trend that stopped it.
        """
        return self._current_trend_id() != self._traded_trend_id

    def _current_trend_id(self) -> Optional[tuple]:
        trend = self._ha.current_trend
        if trend is None:
            return None
        return (trend.state, trend._start_open)

    def _momentum_gaining(self) -> bool:
        trend = self._ha.current_trend
        return trend is not None and trend.momentum == MomentumState.GAINING

    def _rsi_confirms_long(self) -> bool:
        return (
            self._rsi.current_value is not None
            and self._rsi.current_value > self._rsi_bull_min
            # and 30.0 > self._rsi.current_value > 70.0
            and self._rsi.is_rising(strict=True)
        )

    def _rsi_confirms_short(self) -> bool:
        return (
            self._rsi.current_value is not None
            and self._rsi.current_value < self._rsi_bear_max
            # and 40.0 > self._rsi.current_value > 70.0
            and self._rsi.is_falling(strict=True)
        )

    def _rsi_overbought(self) -> bool:
        return (
            self._rsi.current_value is not None
            and self._rsi.current_value >= self._rsi_overbought_level
        )

    def _rsi_oversold(self) -> bool:
        return (
            self._rsi.current_value is not None
            and self._rsi.current_value <= self._rsi_oversold_level
        )

    def _open_entry(self, direction: str, candle: Candle) -> StrategySignal:
        trend = self._ha.current_trend
        rsi_val = self._rsi.current_value
        sl = self._compute_sl(direction)

        reason = (
            f"{direction}: HA {trend.state.value} candle {trend.candle_count}, "
            f"momentum=GAINING ({trend.last_candle_growth_pct:.3f}% vs avg {trend.avg_candle_growth_pct:.3f}%), "
            f"RSI={rsi_val:.2f}"
        )
        pattern = (
            f"HA={trend.state.value} "
            f"candle={trend.candle_count} "
            f"growth={trend.growth_pct:.3f}% "
            f"RSI={rsi_val:.2f}"
        )

        signal = StrategySignal(
            signal=direction,
            strategy_type="TRENDING",
            reason=reason,
            pattern_name=pattern,
            atr=self._atr.current_value,
            sl=sl,
            tp=None,
            candle=candle,
        )

        self._open_trade = signal
        self._traded_trend_id = self._current_trend_id()

        return signal

    # ──────────────────────────────────────────────────────────────────────────
    # Stop-loss helper
    # ──────────────────────────────────────────────────────────────────────────

    def _compute_sl(self, direction: str) -> float:
        """
        SL is anchored to the structural extreme of the current HA trend.

        Long:  lowest HA low across all candles since the trend started.
        Short: highest HA high across all candles since the trend started.

        Uses trend.candle_count candles from the HA history, capped at what
        is available, so the stop always reflects the actual trend footprint.
        """
        trend = self._ha.current_trend
        ha_candles = self._ha.values()
        lookback = min(trend.candle_count, len(ha_candles))
        window = ha_candles[-lookback:]

        if direction == "BUY":
            return min(c.low for c in window)
        return max(c.high for c in window)