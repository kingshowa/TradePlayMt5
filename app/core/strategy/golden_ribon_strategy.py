from collections import deque
from typing import Any, Deque, Optional

from app.core.indicators.atr import ATRIndicator
from app.core.indicators.ema import EMAIndicator
from app.core.indicators.rsi import RSIIndicator
from app.core.market.candle import Candle
from app.core.market.market_series import MarketSeries
from app.core.strategy.trade_signal import StrategySignal


class GoldenRibbonStrategy:
    """
    Golden Ribbon — Single Timeframe Trend Scalper.

    Designed for fast bot execution on a single chart (recommended: 5M Gold).
    Replaces the HTF/LTF two-chart approach by embedding a local trend proxy
    (EMA 50) alongside the fast ribbon (EMA 9 / EMA 21) on the same timeframe.

    ── Indicator stack ───────────────────────────────────────────────────────
    EMA 50  (close)   Local trend proxy — price side determines bias.
                      Replaces the HTF 1H 200 EMA. No higher chart needed.
    EMA 9   (close)   Fast ribbon line — primary cross trigger.
    EMA 21  (close)   Slow ribbon line — cross anchor.
    RSI 14            Momentum gate. Thresholds at 52 / 48 (not 50) to create
                      a chop-zone buffer and avoid neutral-momentum entries.
    ATR 14            Volatility normaliser. Signals are skipped when ATR is
                      below 0.8× or above 2.0× its 20-bar rolling average —
                      filtering dead markets and news spikes respectively.

    ── Entry logic ───────────────────────────────────────────────────────────
    BUY  — EMA9 crosses above EMA21  AND  close > EMA50  AND  RSI > 52
           AND  RSI is rising  AND  close > EMA9  AND  ATR in normal range
           AND  cooldown elapsed since last stop-out.

    SELL — EMA9 crosses below EMA21  AND  close < EMA50  AND  RSI < 48
           AND  RSI is falling  AND  close < EMA9  AND  ATR in normal range
           AND  cooldown elapsed since last stop-out.

    ── Exit logic (use_close_signal=True) ────────────────────────────────────
    CLOSE on either condition (whichever fires first):
      1. RSI crosses back through 50 against the open trade direction.
      2. EMA9 / EMA21 cross reverses against the open trade direction.

    ── Risk management ───────────────────────────────────────────────────────
    SL and TP are handled by the external risk engine / executor.
    The contract between this class and the outside world is three calls:

      sync_trade(trade)
          Call once immediately after the risk engine builds or modifies a
          trade. The strategy stores the SL / TP and open direction so that
          close-signal logic and the state() snapshot reflect the live levels.

      on_trade_closed(exit_type, exit_price)
          Call whenever the trade closes for any reason. The strategy clears
          its trade snapshot and, when exit_type is "SL", also resets the
          post-stop cooldown counter so the bot pauses before re-entering.

      state() -> dict
          Returns a snapshot of every indicator value and all trade-state
          fields for logging, dashboards, or unit-test assertions.

    This class generates only StrategySignal objects.
    SL and TP values on those signals are always None — the risk engine is
    responsible for computing and tracking actual levels.
    """

    # ── Class-level constants (override via subclass if needed) ───────────
    _COOLDOWN_BARS: int   = 3    # bars to wait after a stop before re-entry
    _ATR_AVG_PERIOD: int  = 25   # rolling window for ATR average modified from 20
    _ATR_LOW_MULT: float  = 0.8  # skip below this × ATR average (dead market)
    _ATR_HIGH_MULT: float = 1.8  # skip above this × ATR average (news spike) modified from 2.0

    def __init__(
        self,
        market_series: MarketSeries,
        # ── Ribbon ────────────────────────────────────────────────────────
        ema_fast_period: int   = 9,
        ema_slow_period: int   = 21,
        ema_trend_period: int  = 50,
        ema_offset: int        = 1,
        ema_fast_slope_threshold: float = 0.04,
        # ── RSI ───────────────────────────────────────────────────────────
        rsi_period: int        = 14,
        rsi_slope_offset: int  = 3,
        rsi_buy_level: float   = 52.0,
        rsi_sell_level: float  = 48.0,
        # ── ATR ───────────────────────────────────────────────────────────
        atr_period: int        = 14,
        # ── Behaviour toggles ─────────────────────────────────────────────
        use_close_signal: bool = True,
        max_candles: int       = 200,
    ):
        self.rsi_buy_level    = rsi_buy_level
        self.rsi_sell_level   = rsi_sell_level
        self.use_close_signal = use_close_signal

        self.ema_fast_slope_threshold = ema_fast_slope_threshold # New

        # ── Indicators ────────────────────────────────────────────────────
        self.ema_fast = EMAIndicator(
            period=ema_fast_period,
            source="close",
            offset=ema_offset,
        )
        self.ema_slow = EMAIndicator(
            period=ema_slow_period,
            source="close",
            offset=ema_offset,
        )
        self.ema_trend = EMAIndicator(
            period=ema_trend_period,
            source="close",
            offset=ema_offset,
        )
        self.rsi = RSIIndicator(
            period=rsi_period,
            slope_offset=rsi_slope_offset,
        )
        self.atr = ATRIndicator(period=atr_period)

        # ── Seed all indicators from historical series ─────────────────
        self.current_ema_fast  = self.ema_fast.calculate(market_series)
        self.current_ema_slow  = self.ema_slow.calculate(market_series)
        self.current_ema_trend = self.ema_trend.calculate(market_series)
        self.current_rsi       = self.rsi.calculate(market_series)
        self.current_atr       = self.atr.calculate(market_series)

        # ── ATR rolling average for the volatility filter ──────────────
        self._atr_history: Deque[float] = deque(maxlen=self._ATR_AVG_PERIOD)
        if self.current_atr is not None:
            self._atr_history.append(self.current_atr)

        # ── Candle buffer ──────────────────────────────────────────────
        self.candles: Deque[Candle] = deque(maxlen=max_candles)
        for candle in market_series._candles:
            self.candles.append(candle)

        # ── Trade state ────────────────────────────────────────────────
        # _open_direction  — "LONG" | "SHORT" | None.
        #                    Set by _set_direction_from_trade() via sync_trade()
        #                    and cleared by clear_open_trade() on any close.
        # _current_trade   — the full trade object from the risk engine.
        #                    Stored for diagnostics; the strategy never mutates it.
        # _current_trade_sl / _tp — mirrors of the live stop and target levels
        #                    as last reported by sync_trade(). Updated on every
        #                    sync_trade() call so the close-signal logic and
        #                    state() always reflect the risk engine's current view.
        # _last_exit_*     — filled by on_trade_closed() for post-trade logging.
        self._open_direction:    Optional[str]   = None
        self._current_trade:     Optional[Any]   = None
        self._current_trade_sl:  Optional[float] = None
        self._current_trade_tp:  Optional[float] = None
        self._last_exit_type:    Optional[str]   = None
        self._last_exit_price:   Optional[float] = None

        # Start at COOLDOWN_BARS so the bot is ready to trade immediately
        # on the first valid signal after initialisation.
        self._bars_since_stop: int = self._COOLDOWN_BARS

    # ──────────────────────────────────────────────────────────────────────
    # Public API — called by the backtest / live executor
    # ──────────────────────────────────────────────────────────────────────

    def update(self, candle: Candle) -> Optional[StrategySignal]:
        """
        Process one closed candle and return a StrategySignal or None.

        Call order within each bar:
          1. Advance all indicators.
          2. If a position is open and use_close_signal is True, check exits.
             Exit signal is returned immediately; no entry check on the same bar.
          3. If flat, check for a new entry cross.
        """
        self.current_ema_fast  = self.ema_fast.update(candle)
        self.current_ema_slow  = self.ema_slow.update(candle)
        self.current_ema_trend = self.ema_trend.update(candle)
        self.current_rsi       = self.rsi.update(candle)
        self.current_atr       = self.atr.update(candle)

        if self.current_atr is not None:
            self._atr_history.append(self.current_atr)

        self.candles.append(candle)
        self._bars_since_stop += 1

        # Guard: all indicators must be ready and cross detection needs
        # a previous value on both ribbon EMAs.
        if None in (
            self.current_ema_fast,
            self.current_ema_slow,
            self.current_ema_trend,
            self.current_rsi,
            self.current_atr,
        ):
            return None

        if self.ema_fast.previous_value is None or self.ema_slow.previous_value is None:
            return None

        series = MarketSeries(list(self.candles))

        # Step 1: exit check takes priority over new entries
        if self.use_close_signal and self._open_direction is not None:
            close_signal = self._check_close(series.last())
            if close_signal is not None:
                # Do NOT clear the trade here — the backtest calls
                # on_trade_closed() after logging the exit, which performs
                # the authoritative state clear. Clearing here as well would
                # cause a double-clear that resets cooldown incorrectly.
                return close_signal

        # Step 2: new entry — only when flat
        if self._open_direction is None:
            if self._ribbon_crossed_above() and self._is_long_valid(candle):
                self._open_direction = "LONG"
                return self._build_entry_signal("BUY", series.last())

            if self._ribbon_crossed_below() and self._is_short_valid(candle):
                self._open_direction = "SHORT"
                return self._build_entry_signal("SELL", series.last())

        return None

    def sync_trade(self, trade: Any) -> None:
        """
        Synchronise the strategy with the live / backtest trade object.

        Call this once immediately after the risk engine builds a new trade,
        and again any time the risk engine modifies the trade's SL or TP
        (e.g. after a breakeven move or trailing stop update).

        The strategy stores the latest SL / TP so that:
          - state() exposes the current risk levels for logging and dashboards.
          - _check_close() can be extended in future to incorporate SL/TP
            proximity as an additional exit gate.
          - _open_direction is set from trade.direction, keeping both sides
            in sync without requiring the caller to manage direction separately.
        """
        self._current_trade    = trade
        self._current_trade_sl = getattr(trade, "stop_loss",   None)
        self._current_trade_tp = getattr(trade, "take_profit", None)
        self._set_direction_from_trade(trade)

    def on_trade_closed(
        self,
        exit_type: str,
        exit_price: Optional[float] = None,
    ) -> None:
        """
        Notify the strategy that the externally managed trade has closed.

        Must be called by the backtest / executor for every exit, including
        force-closes at end of data. This is the single authoritative place
        where trade state is cleared.

        Parameters
        ----------
        exit_type
            Recognised values: "SL", "TP", "StrategyClose", "ForceClose",
            "Manual". Only "SL" triggers the post-stop cooldown.
        exit_price
            Actual execution price. Stored for diagnostics and state().

        Behaviour
        ---------
        SL  → clear the trade snapshot and reset the cooldown counter.
              The bot will not enter a new trade for _COOLDOWN_BARS bars.
        TP / StrategyClose / ForceClose / Manual
              → clear the trade snapshot; cooldown counter is unchanged.
        """
        self._last_exit_type  = exit_type
        self._last_exit_price = exit_price

        if exit_type.upper() == "SL":
            self._reset_after_stop()
        else:
            self.clear_open_trade()

    def clear_open_trade(self) -> None:
        """
        Clear the trade snapshot and open direction without touching the
        cooldown counter.

        Called internally after TP / strategy / force exits, and can be
        called externally when the executor needs to forcibly flatten the
        strategy state (e.g. end-of-session, manual intervention).
        """
        self._open_direction   = None
        self._current_trade    = None
        self._current_trade_sl = None
        self._current_trade_tp = None

    def state(self) -> dict:
        """
        Snapshot of every indicator value and all trade-state fields.

        Intended for logging, dashboards, and unit-test assertions.
        ribbon_fanning is True when the gap between EMA 9 and EMA 21 is
        widening — a qualitative sign that momentum behind the cross is real.
        """
        atr_avg = self._atr_average()
        return {
            # Indicators
            "ema_fast":          self.current_ema_fast,
            "ema_slow":          self.current_ema_slow,
            "ema_trend":         self.current_ema_trend,
            "rsi":               self.current_rsi,
            "atr":               self.current_atr,
            "atr_avg":           atr_avg,
            "atr_ratio":         (
                round(self.current_atr / atr_avg, 3)
                if atr_avg and atr_avg > 0 else None
            ),
            # Trade state
            "open_direction":    self._open_direction,
            "current_trade_sl":  self._current_trade_sl,
            "current_trade_tp":  self._current_trade_tp,
            # Cooldown
            "bars_since_stop":   self._bars_since_stop,
            "cooldown_active":   self._bars_since_stop < self._COOLDOWN_BARS,
            # Last exit
            "last_exit_type":    self._last_exit_type,
            "last_exit_price":   self._last_exit_price,
            # Ribbon quality
            "ribbon_fanning":    self._ribbon_is_fanning(),
        }

    # ──────────────────────────────────────────────────────────────────────
    # Internal state helpers
    # ──────────────────────────────────────────────────────────────────────

    def _set_direction_from_trade(self, trade: Any) -> None:
        direction = str(getattr(trade, "direction", "")).upper()
        if direction == "BUY":
            self._open_direction = "LONG"
        elif direction == "SELL":
            self._open_direction = "SHORT"

    def _reset_after_stop(self) -> None:
        """Clear the trade snapshot and start the cooldown countdown."""
        self.clear_open_trade()
        self._bars_since_stop = 0

    # ──────────────────────────────────────────────────────────────────────
    # Entry validation
    # ──────────────────────────────────────────────────────────────────────

    def _is_long_valid(self, candle: Candle) -> bool:
        """
        All conditions required after a bullish ribbon cross.

          Cooldown       — enough bars must have elapsed since the last stop.
          ATR filter     — volatility is neither dead nor spiking.
          Trend gate     — close is above EMA 50 (session trend is up).
          RSI gate       — RSI above 52 and rising (momentum is bullish).
          Close confirm  — this candle closed above the 9 EMA, avoiding
                           entries triggered by intrabar wicks.
        """
        return (
            self._bars_since_stop >= self._COOLDOWN_BARS
            and self._atr_in_range()
            and candle.close > self.current_ema_trend
            and self.current_rsi > self.rsi_buy_level
            and self.rsi.is_rising(strict=False)
            and candle.close > self.current_ema_fast
            and self.ema_fast.is_uptrend(candle.close, self.ema_fast_slope_threshold)
        )

    def _is_short_valid(self, candle: Candle) -> bool:
        """
        All conditions required after a bearish ribbon cross.
        Mirror image of _is_long_valid with direction flipped.
        """
        return (
            self._bars_since_stop >= self._COOLDOWN_BARS
            and self._atr_in_range()
            and candle.close < self.current_ema_trend
            and self.current_rsi < self.rsi_sell_level
            and self.rsi.is_falling(strict=False)
            and candle.close < self.current_ema_fast
            and self.ema_fast.is_downtrend(candle.close, self.ema_fast_slope_threshold)
        )

    # ──────────────────────────────────────────────────────────────────────
    # Close signal logic
    # ──────────────────────────────────────────────────────────────────────

    def _check_close(self, candle: Candle) -> Optional[StrategySignal]:
        """
        Evaluate indicator-based exit conditions for the open direction.

        Two triggers (first one hit wins):
          1. RSI crosses back through 50 against the trade — momentum is dead.
          2. Ribbon reverses against the trade — the thesis has expired.

        Returns a CLOSE StrategySignal or None if no exit condition is met.
        The caller (update) returns this signal immediately without checking
        for a new entry on the same bar.
        """
        reason:  Optional[str] = None
        pattern: Optional[str] = None
        prev_rsi = self.rsi.previous_value

        if self._open_direction == "LONG":
            rsi_exit    = (
                prev_rsi is not None
                and prev_rsi >= 50.0
                and self.current_rsi < 50.0
            )
            ribbon_exit = self._ribbon_crossed_below()

            if rsi_exit:
                reason = (
                    f"RSI dropped below 50 — bullish momentum exhausted | "
                    f"PrevRSI: {prev_rsi:.2f}  CurrRSI: {self.current_rsi:.2f}"
                )
                pattern = "RSI_50_CROSS_DOWN"

            elif ribbon_exit:
                reason = (
                    f"Ribbon reversed bearish — EMA9 crossed below EMA21 | "
                    f"EMA9: {self.current_ema_fast:.4f}  "
                    f"EMA21: {self.current_ema_slow:.4f}"
                )
                pattern = "RIBBON_CROSS_BEARISH"

        elif self._open_direction == "SHORT":
            rsi_exit    = (
                prev_rsi is not None
                and prev_rsi <= 50.0
                and self.current_rsi > 50.0
            )
            ribbon_exit = self._ribbon_crossed_above()

            if rsi_exit:
                reason = (
                    f"RSI rose above 50 — bearish momentum exhausted | "
                    f"PrevRSI: {prev_rsi:.2f}  CurrRSI: {self.current_rsi:.2f}"
                )
                pattern = "RSI_50_CROSS_UP"

            elif ribbon_exit:
                reason = (
                    f"Ribbon reversed bullish — EMA9 crossed above EMA21 | "
                    f"EMA9: {self.current_ema_fast:.4f}  "
                    f"EMA21: {self.current_ema_slow:.4f}"
                )
                pattern = "RIBBON_CROSS_BULLISH"

        if reason is None:
            return None

        return StrategySignal(
            signal="CLOSE",
            strategy_type="TRENDING",
            reason=reason,
            pattern_name=pattern,
            atr=self.current_atr,
            sl=None,
            tp=None,
            candle=candle,
        )

    # ──────────────────────────────────────────────────────────────────────
    # Entry signal builder
    # ──────────────────────────────────────────────────────────────────────

    def _build_entry_signal(self, direction: str, candle: Candle) -> StrategySignal:
        """
        Construct the BUY or SELL StrategySignal.

        reason        — EMA state at the moment of the cross.
        pattern_name  — RSI, ATR, and trend context for logging / debugging.
        sl / tp       — always None; the risk engine owns those levels.
        """
        atr_avg    = self._atr_average()
        trend_side = "UP" if candle.close > self.current_ema_trend else "DOWN"

        reason = (
            f"PreEMAfast: {self.ema_fast.previous_value:.4f}  "
            f"CurrEMAfast: {self.ema_fast.current_value:.4f} | "
            f"PreEMAslow: {self.ema_slow.previous_value:.4f}  "
            f"CurrEMAslow: {self.ema_slow.current_value:.4f} | "
            f"EMA50: {self.current_ema_trend:.4f}"
        )
        pattern_name = (
            f"PrevRSI: {self.rsi.previous_value:.2f}  "
            f"CurrRSI: {self.current_rsi:.2f} | "
            f"ATR: {self.current_atr:.4f}  "
            f"ATRavg: {atr_avg:.4f} | "
            f"Trend: {trend_side}"
        )

        return StrategySignal(
            signal=direction,
            strategy_type="TRENDING",
            reason=reason,
            pattern_name=pattern_name,
            atr=self.current_atr,
            sl=None,
            tp=None,
            candle=candle,
        )

    # ──────────────────────────────────────────────────────────────────────
    # Cross detection
    # ──────────────────────────────────────────────────────────────────────

    def _ribbon_crossed_above(self) -> bool:
        """True when EMA9 crossed above EMA21 on this bar (bar N-1 → bar N)."""
        return (
            self.ema_fast.previous_value <= self.ema_slow.previous_value
            and self.ema_fast.current_value > self.ema_slow.current_value
        )

    def _ribbon_crossed_below(self) -> bool:
        """True when EMA9 crossed below EMA21 on this bar (bar N-1 → bar N)."""
        return (
            self.ema_fast.previous_value >= self.ema_slow.previous_value
            and self.ema_fast.current_value < self.ema_slow.current_value
        )

    # ──────────────────────────────────────────────────────────────────────
    # ATR volatility filter
    # ──────────────────────────────────────────────────────────────────────

    def _atr_average(self) -> float:
        """Rolling mean of the last _ATR_AVG_PERIOD ATR readings."""
        if not self._atr_history:
            return self.current_atr or 0.0
        return sum(self._atr_history) / len(self._atr_history)

    def _atr_in_range(self) -> bool:
        """
        True when current ATR is within the accepted volatility band.

        Below _ATR_LOW_MULT × avg  → market is ranging; crosses are noise.
        Above _ATR_HIGH_MULT × avg → news spike; slippage risk is high.
        """
        avg = self._atr_average()
        if avg == 0.0:
            return False
        ratio = self.current_atr / avg
        return self._ATR_LOW_MULT <= ratio <= self._ATR_HIGH_MULT

    # ──────────────────────────────────────────────────────────────────────
    # Ribbon quality
    # ──────────────────────────────────────────────────────────────────────

    def _ribbon_is_fanning(self) -> Optional[bool]:
        """
        True when the gap between EMA 9 and EMA 21 is widening vs last bar.
        A fanning ribbon indicates real momentum behind the cross.
        None when previous values are not yet available.
        """
        if None in (self.ema_fast.previous_value, self.ema_slow.previous_value):
            return None
        current_gap  = abs(self.current_ema_fast - self.current_ema_slow)
        previous_gap = abs(self.ema_fast.previous_value - self.ema_slow.previous_value)
        return current_gap > previous_gap