from collections import deque

from app.core.indicators.base_indicator import BaseIndicator
from app.core.market.market_series import MarketSeries
from app.core.market.candle import Candle


class PSARIndicator(BaseIndicator):
    """
    Parabolic SAR — standard Wilder implementation, MT5-compatible.

    Calculation rules:
      Uptrend:
        candidate SAR = prev_SAR + AF * (EP - prev_SAR)
        clamp candidate to min(prev_two_lows)
        if candle.low < SAR  → reverse to downtrend
        else update EP / AF if new high made

      Downtrend:
        candidate SAR = prev_SAR + AF * (EP - prev_SAR)
        clamp candidate to max(prev_two_highs)
        if candle.high > SAR → reverse to uptrend
        else update EP / AF if new low made

    Features:
    - Batch calculation for backtesting
    - Incremental update for live trading
    - Flip detection with BUY / SELL direction
    - Price-relative helpers (is_below_price, is_above_price)
    """

    TREND_UP = "UP"
    TREND_DOWN = "DOWN"

    def __init__(
        self,
        step: float = 0.02,
        max_step: float = 0.2,
        history_size: int = 100,
    ):
        super().__init__(period=2)

        if step <= 0:
            raise ValueError("step must be greater than 0")

        if max_step <= 0:
            raise ValueError("max_step must be greater than 0")

        if step > max_step:
            raise ValueError("step cannot be greater than max_step")

        if history_size <= 0:
            raise ValueError("history_size must be greater than 0")

        self.step = step
        self.max_step = max_step

        self._history: deque[float] = deque(maxlen=history_size)
        self._candles: deque[Candle] = deque(maxlen=3)

        self.current_value: float | None = None
        self.previous_value: float | None = None

        self.trend: str | None = None
        self.previous_trend: str | None = None

        self.ep: float | None = None
        self.af: float = step

        self.has_flipped: bool = False
        self.flip_direction: str | None = None     # "BUY", "SELL", or None

    # ------------------------------------------------------------------
    # Batch calculation (Backtest)
    # ------------------------------------------------------------------
    def calculate(self, series: MarketSeries):
        if len(series) < 3:
            raise ValueError("PSAR requires at least 3 candles to calculate.")

        self._reset()

        candles = series._candles
        first = candles[0]
        second = candles[1]

        self.trend = self._initial_trend(first, second)

        if self.trend == self.TREND_UP:
            self.current_value = min(first.low, second.low)
            self.ep = max(first.high, second.high)
        else:
            self.current_value = max(first.high, second.high)
            self.ep = min(first.low, second.low)

        self.af = self.step
        self._candles.append(first)
        self._candles.append(second)
        self._history.append(self.current_value)

        for candle in candles[2:]:
            self.update(candle)

        return self.current_value

    # ------------------------------------------------------------------
    # Incremental update (Live)
    # ------------------------------------------------------------------
    def update(self, candle: Candle):
        if self.current_value is None:
            raise ValueError("PSAR must be initialized using calculate() before update().")

        if len(self._candles) < 2:
            raise ValueError("PSAR requires at least 2 previous candles before update().")

        self.previous_value = self.current_value
        self.previous_trend = self.trend
        self.has_flipped = False
        self.flip_direction = None

        prev = self._candles[-1]
        prev_prev = self._candles[-2]

        candidate_sar = self._candidate_sar()

        if self.trend == self.TREND_UP:
            # SAR must stay below the prior two lows
            candidate_sar = min(candidate_sar, prev.low, prev_prev.low)

            if candle.low < candidate_sar:
                self._reverse_to_down(candle)
            else:
                self.current_value = candidate_sar
                self._continue_uptrend(candle)

        else:
            # SAR must stay above the prior two highs
            candidate_sar = max(candidate_sar, prev.high, prev_prev.high)

            if candle.high > candidate_sar:
                self._reverse_to_up(candle)
            else:
                self.current_value = candidate_sar
                self._continue_downtrend(candle)

        self._candles.append(candle)
        self._history.append(self.current_value)

        return self.current_value

    # ------------------------------------------------------------------
    # Trend / flip helpers
    # ------------------------------------------------------------------
    def is_uptrend(self) -> bool:
        return self.trend == self.TREND_UP

    def is_downtrend(self) -> bool:
        return self.trend == self.TREND_DOWN

    def flipped_buy(self) -> bool:
        """True on the bar when SAR crosses below price (new uptrend)."""
        return self.has_flipped and self.flip_direction == "BUY"

    def flipped_sell(self) -> bool:
        """True on the bar when SAR crosses above price (new downtrend)."""
        return self.has_flipped and self.flip_direction == "SELL"

    def is_below_price(self, candle: Candle) -> bool:
        if self.current_value is None:
            return False
        return self.current_value < candle.close

    def is_above_price(self, candle: Candle) -> bool:
        if self.current_value is None:
            return False
        return self.current_value > candle.close

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------
    def value(self) -> float:
        if self.current_value is None:
            raise ValueError("PSAR has not been initialized.")
        return self.current_value

    def history(self) -> list[float]:
        return list(self._history)

    def state(self) -> dict:
        return {
            "current_value": self.current_value,
            "previous_value": self.previous_value,
            "trend": self.trend,
            "previous_trend": self.previous_trend,
            "ep": self.ep,
            "af": self.af,
            "has_flipped": self.has_flipped,
            "flip_direction": self.flip_direction,
        }

    # ------------------------------------------------------------------
    # Internal logic
    # ------------------------------------------------------------------
    def _reset(self):
        self._history.clear()
        self._candles.clear()

        self.current_value = None
        self.previous_value = None

        self.trend = None
        self.previous_trend = None

        self.ep = None
        self.af = self.step

        self.has_flipped = False
        self.flip_direction = None

    def _initial_trend(self, first: Candle, second: Candle) -> str:
        return self.TREND_UP if second.close >= first.close else self.TREND_DOWN

    def _candidate_sar(self) -> float:
        """SAR candidate before direction-specific clamping."""
        return self.current_value + self.af * (self.ep - self.current_value)

    def _continue_uptrend(self, candle: Candle):
        if candle.high > self.ep:
            self.ep = candle.high
            self.af = min(self.af + self.step, self.max_step)

    def _continue_downtrend(self, candle: Candle):
        if candle.low < self.ep:
            self.ep = candle.low
            self.af = min(self.af + self.step, self.max_step)

    def _reverse_to_down(self, candle: Candle):
        self.trend = self.TREND_DOWN
        self.current_value = self.ep          # SAR jumps to the prior EP
        self.ep = candle.low
        self.af = self.step
        self.has_flipped = True
        self.flip_direction = "SELL"

    def _reverse_to_up(self, candle: Candle):
        self.trend = self.TREND_UP
        self.current_value = self.ep          # SAR jumps to the prior EP
        self.ep = candle.high
        self.af = self.step
        self.has_flipped = True
        self.flip_direction = "BUY"