from collections import deque
from typing import Any, Deque, Optional

from app.core.indicators.atr import ATRIndicator
from app.core.indicators.ema import EMAIndicator
from app.core.indicators.psar import PSARIndicator
from app.core.market.candle import Candle
from app.core.market.market_series import MarketSeries
from app.core.strategy.trade_signal import StrategySignal


class PrecisionPsarStrategy:
    """
    Precision PSAR strategy using the existing indicator components.

    Live idea:
    - update(closed_candle) advances PSAR/EMA/ATR only after a candle closes.
    - on_tick(price, candle) checks whether live price crosses the next clamped
      PSAR trigger level calculated from the last confirmed PSAR state.
    - If a reversal is triggered, the estimated reversal PSAR SL is the prior EP,
      because PSARIndicator._reverse_to_up/_reverse_to_down sets
      current_value = self.ep on reversal.

    Important:
    - For live trading use entry_on_update=False.
    - Do not use strategy CLOSE signals for this PSAR trailing model.
    """

    _COOLDOWN_BARS: int = 0

    def __init__(
        self,
        market_series: MarketSeries,
        psar_step: float = 0.02,
        psar_max_step: float = 0.2,
        use_ema_trend: bool = True,
        ema_trend_period: int = 200,
        ema_offset: int = 3,
        ema_slope_threshold: float = 0.0,
        atr_period: int = 14,
        use_close_signal: bool = False,
        entry_on_update: bool = False,
        require_trigger_ema_side: bool = True,
        use_ema_distance_filter: bool = False,
        max_ema_distance_atr: float = 2.0,
        min_entry_sl_atr: float = 0.05,
        max_entry_sl_atr: Optional[float] = None,
        max_candles: int = 500,
    ):
        self.use_ema_trend = use_ema_trend
        self.use_close_signal = use_close_signal
        self.entry_on_update = entry_on_update
        self.ema_slope_threshold = ema_slope_threshold
        self.require_trigger_ema_side = require_trigger_ema_side
        self.use_ema_distance_filter = use_ema_distance_filter
        self.max_ema_distance_atr = max_ema_distance_atr
        self.min_entry_sl_atr = min_entry_sl_atr
        self.max_entry_sl_atr = max_entry_sl_atr

        self.psar = PSARIndicator(
            step=psar_step,
            max_step=psar_max_step,
            history_size=max_candles,
        )
        self.atr = ATRIndicator(period=atr_period)
        self.ema_trend = (
            EMAIndicator(
                period=ema_trend_period,
                source="close",
                offset=ema_offset,
            )
            if use_ema_trend
            else None
        )

        self.current_psar = self.psar.calculate(market_series)
        self.current_atr = self.atr.calculate(market_series)
        self.current_ema_trend = (
            self.ema_trend.calculate(market_series)
            if self.ema_trend is not None
            else None
        )

        self.candles: Deque[Candle] = deque(maxlen=max_candles)
        for candle in market_series.candles():
            self.candles.append(candle)

        self._open_direction: Optional[str] = None
        self._current_trade: Optional[Any] = None
        self._current_trade_sl: Optional[float] = None
        self._current_trade_tp: Optional[float] = None
        self._last_exit_type: Optional[str] = None
        self._last_exit_price: Optional[float] = None
        self._bars_since_stop: int = self._COOLDOWN_BARS

        # Tick-state guards.
        self._last_seen_price: Optional[float] = self._last_close()
        self._last_triggered_key: Optional[tuple[str, float]] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, candle: Candle) -> Optional[StrategySignal]:
        """
        Process one confirmed/closed candle.

        In live trading this should be called only once per newly closed candle.
        It advances indicators and optionally generates a backtest entry when
        entry_on_update=True.
        """
        self.current_psar = self.psar.update(candle)
        self.current_atr = self.atr.update(candle)

        if self.ema_trend is not None:
            self.current_ema_trend = self.ema_trend.update(candle)

        self.candles.append(candle)
        self._last_seen_price = candle.close
        self._bars_since_stop += 1

        if self.current_psar is None or self.current_atr is None:
            return None

        if self.use_close_signal and self._open_direction is not None:
            close_signal = self._check_close(candle)
            if close_signal is not None:
                return close_signal

        if not self.entry_on_update or self._open_direction is not None:
            return None

        # Backtest-only fallback: emulate an intrabar cross with OHLC after the
        # previous confirmed state. Live trading should use on_tick instead.
        return self.check_candle_psar_cross(candle)

    def on_tick(self, price: float, candle: Optional[Candle] = None) -> Optional[StrategySignal]:
        """
        Live tick-level precision entry.

        Sequence:
        1. Use the last confirmed PSAR state only.
        2. Compute the next clamped SAR trigger for the current forming candle.
        3. Detect crossing of that trigger.
        4. Estimate the reversal PSAR SL using the prior EP.
        5. Build a StrategySignal whose candle.close equals the live entry price.

        This avoids lookahead because it does not call psar.update() on the
        forming candle before checking the entry.
        """
        if price is None or price <= 0:
            return None

        price = float(price)

        if self._open_direction is not None:
            self._last_seen_price = price
            return None

        if self.current_psar is None or self.current_atr is None:
            self._last_seen_price = price
            return None

        trigger = self._next_psar_trigger_level()
        previous_price = self._last_seen_price
        self._last_seen_price = price

        if trigger is None or previous_price is None:
            return None

        direction = self._cross_direction(previous_price, price, trigger)
        if direction is None:
            return None

        if self._last_triggered_key == (direction, round(trigger, 10)):
            return None

        if not self._is_context_valid(direction, price=price, trigger_level=trigger):
            return None

        estimated_sl = self._estimated_reversal_psar_sl(direction=direction, entry_price=price)
        if estimated_sl is None:
            return None

        if not self._entry_sl_is_acceptable(direction, entry_price=price, sl=estimated_sl):
            return None

        entry_candle = self._candle_for_entry(price, candle)
        self._last_triggered_key = (direction, round(trigger, 10))

        return self._build_entry_signal(
            direction=direction,
            candle=entry_candle,
            entry_price=price,
            trigger_level=trigger,
            psar_sl=estimated_sl,
            entry_mode="LIVE_NEXT_PSAR_TRIGGER_CROSS",
        )

    def check_candle_psar_cross(self, candle: Candle) -> Optional[StrategySignal]:
        """
        Backtest OHLC approximation using the same last-confirmed state as on_tick.

        Call this before update(candle) in a live-equivalent backtest, or let
        update(candle) call it when entry_on_update=True.
        """
        if self._open_direction is not None:
            return None

        trigger = self._next_psar_trigger_level()
        if trigger is None or self.current_atr is None:
            return None

        direction = None
        entry_price = None

        if self.psar.is_downtrend() and candle.high > trigger:
            direction = "BUY"
            entry_price = trigger
        elif self.psar.is_uptrend() and candle.low < trigger:
            direction = "SELL"
            entry_price = trigger

        if direction is None or entry_price is None:
            return None

        if self._last_triggered_key == (direction, round(trigger, 10)):
            return None

        if not self._is_context_valid(direction, price=entry_price, trigger_level=trigger):
            return None

        estimated_sl = self._estimated_reversal_psar_sl(direction=direction, entry_price=entry_price)
        if estimated_sl is None:
            return None

        if not self._entry_sl_is_acceptable(direction, entry_price=entry_price, sl=estimated_sl):
            return None

        self._last_triggered_key = (direction, round(trigger, 10))
        return self._build_entry_signal(
            direction=direction,
            candle=candle,
            entry_price=entry_price,
            trigger_level=trigger,
            psar_sl=estimated_sl,
            entry_mode="BACKTEST_NEXT_PSAR_TRIGGER_CROSS",
        )

    def get_psar_sl(self, direction: str) -> Optional[float]:
        """
        Return the best available PSAR trailing stop for an already-open trade.

        If trade direction is ahead of the confirmed PSAR trend, use EP as the
        estimated reversal PSAR until the closed-candle update confirms the flip.
        """
        direction = str(direction).upper()
        psar_trend = self.state().get("psar_trend")

        trade_ahead_of_confirmed_flip = (
                (direction == "BUY" and psar_trend == "DOWN")
                or
                (direction == "SELL" and psar_trend == "UP")
        )

        if trade_ahead_of_confirmed_flip:
            ep = self.psar.ep
            if ep is None:
                return None
            return float(ep)

        if self.current_psar is None:
            return None

        return float(self.current_psar)

    def sync_trade(self, trade: Any) -> None:
        self._current_trade = trade
        self._current_trade_sl = getattr(trade, "stop_loss", None)
        self._current_trade_tp = getattr(trade, "take_profit", None)
        self._set_direction_from_trade(trade)

    def on_trade_closed(self, exit_type: str, exit_price: Optional[float] = None) -> None:
        self._last_exit_type = exit_type
        self._last_exit_price = exit_price

        if exit_type.upper() == "SL":
            self._reset_after_stop()
        else:
            self.clear_open_trade()

    def clear_open_trade(self) -> None:
        self._open_direction = None
        self._current_trade = None
        self._current_trade_sl = None
        self._current_trade_tp = None

    def state(self) -> dict:
        trigger = self._next_psar_trigger_level()
        return {
            "strategy": "PrecisionPsarStrategy",
            "entry_on_update": self.entry_on_update,
            "psar": self.current_psar,
            "psar_next_trigger": trigger,
            "psar_trend": self.psar.trend,
            "psar_previous_value": self.psar.previous_value,
            "psar_ep": self.psar.ep,
            "psar_af": self.psar.af,
            "atr": self.current_atr,
            "ema_trend": self.current_ema_trend,
            "ema_slope": self.ema_trend.slope() if self.ema_trend is not None else None,
            "open_direction": self._open_direction,
            "bars_since_stop": self._bars_since_stop,
            "cooldown_active": self._bars_since_stop < self._COOLDOWN_BARS,
            "last_triggered_key": self._last_triggered_key,
        }

    # ------------------------------------------------------------------
    # PSAR live trigger / estimated SL logic
    # ------------------------------------------------------------------

    def _next_psar_trigger_level(self) -> Optional[float]:
        """
        Compute the SAR level that the current forming candle must cross to flip.

        This mirrors PSARIndicator.update() before the reversal check:
        - candidate = current_sar + AF * (EP - current_sar)
        - UP trend clamps to min(previous two lows)
        - DOWN trend clamps to max(previous two highs)
        """
        if self.current_psar is None or self.psar.ep is None or self.psar.trend is None:
            return None

        if len(self.candles) < 2:
            return None

        prev = self.candles[-1]
        prev_prev = self.candles[-2]

        candidate = float(self.current_psar) + float(self.psar.af) * (
            float(self.psar.ep) - float(self.current_psar)
        )

        if self.psar.is_uptrend():
            candidate = min(candidate, float(prev.low), float(prev_prev.low))
        elif self.psar.is_downtrend():
            candidate = max(candidate, float(prev.high), float(prev_prev.high))
        else:
            return None

        return float(candidate)

    def _cross_direction(
        self,
        previous_price: float,
        current_price: float,
        trigger_level: float,
    ) -> Optional[str]:
        """
        Return BUY/SELL only when price crosses the correct PSAR trigger for
        the current confirmed PSAR trend.
        """
        if self.psar.is_downtrend():
            if previous_price <= trigger_level < current_price:
                return "BUY"
            return None

        if self.psar.is_uptrend():
            if previous_price >= trigger_level > current_price:
                return "SELL"
            return None

        return None

    def _estimated_reversal_psar_sl(
        self,
        direction: str,
        entry_price: float,
    ) -> Optional[float]:
        """
        Estimate the PSAR value after reversal using the current EP.

        This is not a guess from price action; it follows the indicator's own
        reversal rule. In PSARIndicator._reverse_to_up/_reverse_to_down,
        current_value is assigned to self.ep before EP is reset to the reversal
        candle's high/low.
        """
        ep = self.psar.ep
        if ep is None:
            return None

        sl = float(ep)
        direction = direction.upper()

        if direction == "BUY" and sl < entry_price:
            return sl
        if direction == "SELL" and sl > entry_price:
            return sl

        return None

    def _entry_sl_is_acceptable(
        self,
        direction: str,
        entry_price: float,
        sl: float,
    ) -> bool:
        """
        Reject unusable estimated PSAR stops before the risk manager sees them.

        min_entry_sl_atr prevents a stop effectively sitting on the entry.
        max_entry_sl_atr optionally prevents a huge old EP from creating an
        untradably large stop.
        """
        if self.current_atr is None or self.current_atr <= 0:
            return False

        distance = abs(entry_price - sl)

        if direction == "BUY" and sl >= entry_price:
            return False
        if direction == "SELL" and sl <= entry_price:
            return False

        if self.min_entry_sl_atr is not None and self.min_entry_sl_atr > 0:
            if distance < float(self.current_atr) * self.min_entry_sl_atr:
                return False

        if self.max_entry_sl_atr is not None and self.max_entry_sl_atr > 0:
            if distance > float(self.current_atr) * self.max_entry_sl_atr:
                return False

        return True

    # ------------------------------------------------------------------
    # Context filters
    # ------------------------------------------------------------------

    def _is_context_valid(self, direction: str, price: float, trigger_level: float) -> bool:
        if self._bars_since_stop < self._COOLDOWN_BARS:
            return False

        if direction == "BUY":
            return (
                self._ema_allows_buy_price(price)
                and self._trigger_ema_side_allows_buy(trigger_level)
                and self._ema_distance_allows(price)
            )

        if direction == "SELL":
            return (
                self._ema_allows_sell_price(price)
                and self._trigger_ema_side_allows_sell(trigger_level)
                and self._ema_distance_allows(price)
            )

        return False

    def _ema_allows_buy_price(self, price: float) -> bool:
        if not self.use_ema_trend:
            return True
        if self.ema_trend is None or self.current_ema_trend is None:
            return False
        return self.ema_trend.is_uptrend(
            current_price=price,
            slope_threshold=self.ema_slope_threshold,
        )

    def _ema_allows_sell_price(self, price: float) -> bool:
        if not self.use_ema_trend:
            return True
        if self.ema_trend is None or self.current_ema_trend is None:
            return False
        return self.ema_trend.is_downtrend(
            current_price=price,
            slope_threshold=self.ema_slope_threshold,
        )

    def _trigger_ema_side_allows_buy(self, trigger_level: float) -> bool:
        if not self.require_trigger_ema_side or not self.use_ema_trend:
            return True
        if self.current_ema_trend is None:
            return False
        return trigger_level > self.current_ema_trend

    def _trigger_ema_side_allows_sell(self, trigger_level: float) -> bool:
        if not self.require_trigger_ema_side or not self.use_ema_trend:
            return True
        if self.current_ema_trend is None:
            return False
        return trigger_level < self.current_ema_trend

    def _ema_distance_allows(self, price: float) -> bool:
        if not self.use_ema_distance_filter:
            return True
        if self.current_ema_trend is None or self.current_atr is None or self.current_atr <= 0:
            return False
        return abs(price - self.current_ema_trend) <= self.current_atr * self.max_ema_distance_atr

    # ------------------------------------------------------------------
    # Close / signal builders
    # ------------------------------------------------------------------

    def _check_close(self, candle: Candle) -> Optional[StrategySignal]:
        if self._open_direction == "LONG" and self.psar.flipped_sell():
            return StrategySignal(
                signal="CLOSE",
                strategy_type="TRENDING",
                reason="PSAR flipped bearish against LONG",
                pattern_name="PRECISION_PSAR_BEARISH_FLIP_EXIT",
                atr=self.current_atr,
                sl=None,
                tp=None,
                candle=candle,
            )

        if self._open_direction == "SHORT" and self.psar.flipped_buy():
            return StrategySignal(
                signal="CLOSE",
                strategy_type="TRENDING",
                reason="PSAR flipped bullish against SHORT",
                pattern_name="PRECISION_PSAR_BULLISH_FLIP_EXIT",
                atr=self.current_atr,
                sl=None,
                tp=None,
                candle=candle,
            )

        return None

    def _build_entry_signal(
        self,
        direction: str,
        candle: Candle,
        entry_price: float,
        trigger_level: float,
        psar_sl: Optional[float],
        entry_mode: str,
    ) -> StrategySignal:
        signal_candle = self._entry_candle(candle, entry_price)

        ema_text = (
            f"EMA: {self.current_ema_trend:.4f} | EMA slope: {self.ema_trend.slope():.4f}"
            if self.ema_trend is not None and self.current_ema_trend is not None
            else "EMA filter disabled"
        )
        psar_text = f"Estimated reversal PSAR SL: {psar_sl:.4f}" if psar_sl is not None else "Estimated reversal PSAR SL: None"

        reason = (
            f"Precision PSAR {direction} entry | Mode: {entry_mode} | "
            f"Next PSAR trigger: {trigger_level:.4f} | "
            f"Entry: {entry_price:.4f} | {psar_text} | "
            f"Prior trend: {self.psar.trend} | Prior EP: {self.psar.ep:.4f}"
        )

        pattern_name = (
            f"PRECISION_PSAR_{direction}_NEXT_TRIGGER | "
            f"{ema_text} | ATR: {self.current_atr:.4f}"
        )

        return StrategySignal(
            signal=direction,
            strategy_type="TRENDING",
            reason=reason,
            pattern_name=pattern_name,
            atr=self.current_atr,
            sl=psar_sl,
            tp=None,
            candle=signal_candle,
        )

    def _entry_candle(self, base_candle: Candle, entry_price: float) -> Candle:
        return Candle(
            base_candle.time,
            base_candle.open,
            max(base_candle.high, entry_price),
            min(base_candle.low, entry_price),
            entry_price,
            volume=base_candle.volume,
        )

    def _candle_for_entry(self, price: float, candle: Optional[Candle]) -> Candle:
        if candle is not None:
            return self._entry_candle(candle, price)

        last = self.candles[-1] if self.candles else None
        if last is None:
            raise ValueError("Cannot build entry candle without candle history")

        return Candle(
            last.time,
            last.close,
            max(last.close, price),
            min(last.close, price),
            price,
            volume=getattr(last, "volume", 0),
        )

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

    def _last_close(self) -> Optional[float]:
        if not self.candles:
            return None
        return float(self.candles[-1].close)

    def _set_direction_from_trade(self, trade: Any) -> None:
        direction = str(getattr(trade, "direction", "")).upper()
        if direction == "BUY":
            self._open_direction = "LONG"
        elif direction == "SELL":
            self._open_direction = "SHORT"

    def _reset_after_stop(self) -> None:
        self.clear_open_trade()
        self._bars_since_stop = 0
