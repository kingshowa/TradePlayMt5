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
    EMA 50  (close)  Local trend proxy — price side determines bias.
                     Replaces the HTF 1H 200 EMA. No higher chart needed.
    EMA 9   (close)  Fast ribbon line — primary cross trigger.
    EMA 21  (close)  Slow ribbon line — cross anchor.
    RSI 14           Momentum gate. Thresholds at 52 / 48 (not 50) to create
                     a chop-zone buffer and avoid neutral-momentum entries.
    ATR 14           Volatility normaliser. Signals are skipped when ATR is
                     below 0.8× or above 2.0× its 20-bar rolling average —
                     filtering dead markets and news spikes respectively.

    ── Entry logic ───────────────────────────────────────────────────────────
    BUY  — EMA9 crosses above EMA21  AND  close > EMA50  AND  RSI > 52
           AND  RSI is rising  AND  close > EMA9  AND  ATR in normal range
           AND  cooldown elapsed since last stop-out.

    SELL — EMA9 crosses below EMA21  AND  close < EMA50  AND  RSI < 48
           AND  RSI is falling  AND  close < EMA9  AND  ATR in normal range
           AND  cooldown elapsed since last stop-out.

    ── Exit logic (use_close_signal=True) ───────────────────────────────────
    CLOSE on either condition (whichever fires first):
      1. RSI crosses back through 50 against the open trade direction.
      2. EMA9 / EMA21 cross reverses against the open trade direction.

    ── Risk management ───────────────────────────────────────────────────────
    SL and TP are handled by the external risk engine / executor.
    Once a trade is built or broker-modified, call sync_trade(...) so the
    strategy knows the current stop and target. When the trade closes, call
    on_trade_closed(...). Stops automatically reset cooldown; TP / manual /
    strategy exits simply clear the open state.

    This class generates only StrategySignal objects.
    """

    # ── Class-level constants (override via subclass if needed) ───────────
    _COOLDOWN_BARS: int = 3       # bars to wait after a stop before re-entry
    _ATR_AVG_PERIOD: int = 20     # rolling window for ATR average
    _ATR_LOW_MULT: float = 0.8    # skip below this × ATR average (dead market)
    _ATR_HIGH_MULT: float = 2.0   # skip above this × ATR average (news spike)

    def __init__(
        self,
        market_series: MarketSeries,
        # ── Ribbon ────────────────────────────────────────────────────────
        ema_fast_period: int = 9,
        ema_slow_period: int = 21,
        ema_trend_period: int = 50,
        ema_offset: int = 1,
        # ── RSI ───────────────────────────────────────────────────────────
        rsi_period: int = 14,
        rsi_slope_offset: int = 3,
        rsi_buy_level: float = 52.0,
        rsi_sell_level: float = 48.0,
        # ── ATR ───────────────────────────────────────────────────────────
        atr_period: int = 14,
        # ── Behaviour toggles ─────────────────────────────────────────────
        use_close_signal: bool = True,
        max_candles: int = 200,
    ):
        self.rsi_buy_level = rsi_buy_level
        self.rsi_sell_level = rsi_sell_level
        self.use_close_signal = use_close_signal

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
        self.current_ema_fast = self.ema_fast.calculate(market_series)
        self.current_ema_slow = self.ema_slow.calculate(market_series)
        self.current_ema_trend = self.ema_trend.calculate(market_series)
        self.current_rsi = self.rsi.calculate(market_series)
        self.current_atr = self.atr.calculate(market_series)

        # ── ATR rolling average for the volatility filter ──────────────
        self._atr_history: Deque[float] = deque(maxlen=self._ATR_AVG_PERIOD)
        if self.current_atr is not None:
            self._atr_history.append(self.current_atr)

        # ── Candle buffer ──────────────────────────────────────────────
        self.candles: Deque[Candle] = deque(maxlen=max_candles)
        for candle in market_series._candles:
            self.candles.append(candle)

        # ── Internal state ─────────────────────────────────────────────
        self._open_direction: Optional[str] = None  # "LONG" | "SHORT" | None
        self._current_trade: Optional[Any] = None
        self._current_trade_sl: Optional[float] = None
        self._current_trade_tp: Optional[float] = None
        self._last_exit_type: Optional[str] = None
        self._last_exit_price: Optional[float] = None

        # Start at COOLDOWN_BARS so the bot is ready to trade immediately
        # on the first valid signal after initialisation.
        self._bars_since_stop: int = self._COOLDOWN_BARS

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def update(self, candle: Candle) -> Optional[StrategySignal]:
        """
        Process one closed candle. Returns a StrategySignal or None.

        Call order within each bar:
          1. Update all indicators.
          2. If a position is open and use_close_signal is True, check exits.
          3. If flat (or just closed), check for a new entry cross.
        """
        self.current_ema_fast = self.ema_fast.update(candle)
        self.current_ema_slow = self.ema_slow.update(candle)
        self.current_ema_trend = self.ema_trend.update(candle)
        self.current_rsi = self.rsi.update(candle)
        self.current_atr = self.atr.update(candle)

        if self.current_atr is not None:
            self._atr_history.append(self.current_atr)

        self.candles.append(candle)
        self._bars_since_stop += 1

        # Guard: need at least two bars for cross detection (previous + current)
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

        # ── Step 1: exit check takes priority over new entries ─────────
        if self.use_close_signal and self._open_direction is not None:
            close_signal = self._check_close(series.last())
            if close_signal is not None:
                self.clear_open_trade()
                return close_signal

        # ── Step 2: new entry — only when flat ─────────────────────────
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
        Synchronise the strategy with the live/backtest trade object after the
        risk engine builds or modifies it.

        The strategy stores the latest SL/TP so that dashboards, logs, and any
        external trade-state checks can see the same levels as the risk engine.
        """
        self._current_trade = trade
        self._current_trade_sl = getattr(trade, "stop_loss", None)
        self._current_trade_tp = getattr(trade, "take_profit", None)

        direction = str(getattr(trade, "direction", "")).upper()
        if direction == "BUY":
            self._open_direction = "LONG"
        elif direction == "SELL":
            self._open_direction = "SHORT"

    def clear_open_trade(self) -> None:
        """Clear the strategy-side trade snapshot without starting cooldown."""
        self._open_direction = None
        self._current_trade = None
        self._current_trade_sl = None
        self._current_trade_tp = None

    def on_trade_closed(self, exit_type: str, exit_price: Optional[float] = None) -> None:
        """
        Notify the strategy that the externally managed trade has closed.

        Parameters
        ----------
        exit_type
            Examples: "SL", "TP", "StrategyClose", "ForceClose", "Manual".
        exit_price
            Optional execution price for diagnostics.

        Behaviour
        ---------
        - SL   → clear the trade and reset cooldown.
        - TP   → clear the trade and keep cooldown unchanged.
        - else → clear the trade and keep cooldown unchanged.
        """
        self._last_exit_type = exit_type
        self._last_exit_price = exit_price

        if str(exit_type).upper() == "SL":
            self.on_stop_hit()
            return

        self.clear_open_trade()

    def on_stop_hit(self) -> None:
        """
        Notify the strategy that the risk engine triggered a stop-loss.

        Resets the cooldown counter and clears the open direction so the
        bot does not attempt re-entry until COOLDOWN_BARS have elapsed.
        Must be called from the external risk/execution engine.
        """
        self.clear_open_trade()
        self._bars_since_stop = 0

    # ──────────────────────────────────────────────────────────────────────
    # Entry validation
    # ──────────────────────────────────────────────────────────────────────

    def _is_long_valid(self, candle: Candle) -> bool:
        """
        All conditions that must hold after a bullish ribbon cross.

        - Cooldown: enough bars since the last stop-out.
        - ATR filter: volatility is neither dead nor spiking.
        - Trend gate: close is above EMA 50 (local trend is up).
        - RSI gate: RSI is above 52 and rising (momentum is genuinely bullish).
        - Close confirmation: this candle closed above the fast ribbon line,
          avoiding entries triggered by intrabar wicks that cross and retrace.
        """
        return (
            self._bars_since_stop >= self._COOLDOWN_BARS
            and self._atr_in_range()
            and candle.close > self.current_ema_trend
            and self.current_rsi > self.rsi_buy_level
            and self.rsi.is_rising(strict=False)
            and candle.close > self.current_ema_fast
        )

    def _is_short_valid(self, candle: Candle) -> bool:
        """
        All conditions that must hold after a bearish ribbon cross.

        Mirror image of _is_long_valid with direction flipped.
        """
        return (
            self._bars_since_stop >= self._COOLDOWN_BARS
            and self._atr_in_range()
            and candle.close < self.current_ema_trend
            and self.current_rsi < self.rsi_sell_level
            and self.rsi.is_falling(strict=False)
            and candle.close < self.current_ema_fast
        )

    # ──────────────────────────────────────────────────────────────────────
    # Close signal logic
    # ──────────────────────────────────────────────────────────────────────

    def _check_close(self, candle: Candle) -> Optional[StrategySignal]:
        """
        Evaluate exit conditions for the currently open direction.

        Two triggers (first one hit wins):
          1. RSI crosses back through 50 against the trade — momentum is dead.
          2. Ribbon reverses against the trade — the thesis has expired.

        Returns a CLOSE StrategySignal or None.
        """
        reason: Optional[str] = None
        pattern: Optional[str] = None

        prev_rsi = self.rsi.previous_value

        if self._open_direction == "LONG":
            rsi_exit = (
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
                    f"EMA9: {self.current_ema_fast:.4f}  EMA21: {self.current_ema_slow:.4f}"
                )
                pattern = "RIBBON_CROSS_BEARISH"

        elif self._open_direction == "SHORT":
            rsi_exit = (
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
                    f"EMA9: {self.current_ema_fast:.4f}  EMA21: {self.current_ema_slow:.4f}"
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
    # Signal builder
    # ──────────────────────────────────────────────────────────────────────

    def _build_entry_signal(self, direction: str, candle: Candle) -> StrategySignal:
        """
        Construct the BUY or SELL StrategySignal.

        reason      — EMA state at the moment of the cross.
        pattern_name — RSI, ATR context and trend side for logging/debugging.
        """
        atr_avg = self._atr_average()
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
        """
        True when EMA9 crossed above EMA21 on this bar.
        Uses indicator.previous_value (bar N-1) vs current_value (bar N).
        """
        return (
            self.ema_fast.previous_value <= self.ema_slow.previous_value
            and self.ema_fast.current_value > self.ema_slow.current_value
        )

    def _ribbon_crossed_below(self) -> bool:
        """
        True when EMA9 crossed below EMA21 on this bar.
        """
        return (
            self.ema_fast.previous_value >= self.ema_slow.previous_value
            and self.ema_fast.current_value < self.ema_slow.current_value
        )

    # ──────────────────────────────────────────────────────────────────────
    # ATR volatility filter
    # ──────────────────────────────────────────────────────────────────────

    def _atr_average(self) -> float:
        """Rolling mean of the last ATR_AVG_PERIOD ATR readings."""
        if not self._atr_history:
            return self.current_atr
        return sum(self._atr_history) / len(self._atr_history)

    def _atr_in_range(self) -> bool:
        """
        True when the current ATR is within the accepted volatility band.

        Below _ATR_LOW_MULT  → market is too quiet / ranging, crosses are noise.
        Above _ATR_HIGH_MULT → news spike in progress, slippage risk is high.
        """
        avg = self._atr_average()
        if avg == 0.0:
            return False
        ratio = self.current_atr / avg
        return self._ATR_LOW_MULT <= ratio <= self._ATR_HIGH_MULT

    # ──────────────────────────────────────────────────────────────────────
    # Diagnostics
    # ──────────────────────────────────────────────────────────────────────

    def state(self) -> dict:
        """
        Snapshot of current indicator values and strategy state.
        Useful for logging, dashboards, and unit-test assertions.
        """
        return {
            "ema_fast": self.current_ema_fast,
            "ema_slow": self.current_ema_slow,
            "ema_trend": self.current_ema_trend,
            "rsi": self.current_rsi,
            "atr": self.current_atr,
            "atr_avg": self._atr_average(),
            "atr_ratio": (
                round(self.current_atr / self._atr_average(), 3)
                if self._atr_average() else None
            ),
            "open_direction": self._open_direction,
            "bars_since_stop": self._bars_since_stop,
            "cooldown_active": self._bars_since_stop < self._COOLDOWN_BARS,
            "current_trade_sl": self._current_trade_sl,
            "current_trade_tp": self._current_trade_tp,
            "last_exit_type": self._last_exit_type,
            "last_exit_price": self._last_exit_price,
            "ribbon_fanning": (
                abs(self.current_ema_fast - self.current_ema_slow)
                > abs(self.ema_fast.previous_value - self.ema_slow.previous_value)
                if None not in (self.ema_fast.previous_value, self.ema_slow.previous_value)
                else None
            ),
        }
