from collections import deque

from app.core.indicators.base_indicator import BaseIndicator
from app.core.market.market_series import MarketSeries
from app.core.market.candle import Candle


class ADXIndicator(BaseIndicator):
    """
    Average Directional Index (ADX) — Wilder's smoothing, MT5-compatible.

    Calculation pipeline:
      1. Build TR, +DM, -DM arrays from raw candles.
      2. Seed smoothed totals = sum of first `period` values (Wilder's initial sum).
      3. For every bar after the seed window:
           a. Apply Wilder smoothing to TR/+DM/-DM.
           b. Compute +DI, -DI, DX from the smoothed totals.
           c. Collect DX values until we have `period` of them.
      4. First ADX  = SMA of those first `period` DX values.
      5. Subsequent = Wilder smoothing on ADX itself.

    Why this matches MT5:
      The previous implementation computed a "seed DX" from the raw initial sum
      (before any smoothing step) and included it in the SMA window.  MT5 only
      uses DX values that have been produced *after* at least one Wilder smoothing
      step, so the seed DX must not participate.  Removing it also shifts the
      minimum-candle requirement from 2*period to 2*period + 1.

    Features:
    - Batch calculation for backtesting
    - Incremental update for live trading
    - Exposes ADX, +DI, -DI, DX
    - Trend strength and directional bias helpers
    """

    BIAS_BUY = "BUY"
    BIAS_SELL = "SELL"
    BIAS_NEUTRAL = "NEUTRAL"

    def __init__(
        self,
        period: int = 14,
        trend_threshold: float = 25.0,
        history_size: int = 100,
    ):
        super().__init__(period)

        if period <= 0:
            raise ValueError("period must be greater than 0")

        if trend_threshold <= 0:
            raise ValueError("trend_threshold must be greater than 0")

        if history_size <= 0:
            raise ValueError("history_size must be greater than 0")

        self.trend_threshold = trend_threshold

        self._history = deque(maxlen=history_size)
        self._dx_history = deque(maxlen=history_size)

        self.current_value = None       # ADX
        self.previous_value = None

        self.plus_di = None
        self.minus_di = None
        self.dx = None

        self.smoothed_tr = None
        self.smoothed_plus_dm = None
        self.smoothed_minus_dm = None

        self.previous_candle = None

    # ------------------------------------------------------------------
    # Batch calculation (Backtest)
    # ------------------------------------------------------------------
    def calculate(self, series: MarketSeries):
        # FIX: was 2*period — off by one because the seed DX was incorrectly
        # included in the SMA window.  Without the seed DX we need one extra
        # bar from the loop to fill the N-item DX window.
        min_candles = self.period * 2 + 1
        if len(series) < min_candles:
            raise ValueError(
                f"ADX({self.period}) requires at least {min_candles} candles."
            )

        self._reset()

        candles = series._candles

        # Step 1 — build raw TR / DM arrays (one entry per candle pair)
        tr_values = []
        plus_dm_values = []
        minus_dm_values = []

        for i in range(1, len(candles)):
            tr, plus_dm, minus_dm = self._calculate_tr_dm(candles[i], candles[i - 1])
            tr_values.append(tr)
            plus_dm_values.append(plus_dm)
            minus_dm_values.append(minus_dm)

        # Step 2 — Wilder seed: raw sum of first `period` values
        self.smoothed_tr = sum(tr_values[:self.period])
        self.smoothed_plus_dm = sum(plus_dm_values[:self.period])
        self.smoothed_minus_dm = sum(minus_dm_values[:self.period])

        # Step 3 — loop over every bar after the seed window
        # FIX: DX collection starts here (no pre-loop seed DX).
        # Each iteration: smooth first, then compute DX from the updated totals.
        dx_values = []

        for i in range(self.period, len(tr_values)):
            self._smooth_values(tr_values[i], plus_dm_values[i], minus_dm_values[i])
            plus_di, minus_di, dx = self._calculate_di_dx()
            dx_values.append(dx)

            if len(dx_values) < self.period:
                # Still filling the SMA window — not enough DX values yet.
                continue

            if len(dx_values) == self.period:
                # Step 4 — first ADX: plain SMA of the N collected DX values.
                self.current_value = sum(dx_values) / self.period
            else:
                # Step 5 — subsequent ADX: Wilder smoothing.
                self.previous_value = self.current_value
                self.current_value = (
                    (self.current_value * (self.period - 1) + dx) / self.period
                )

            # FIX: state assignment is now shared — no duplication between branches.
            self.plus_di = plus_di
            self.minus_di = minus_di
            self.dx = dx
            self._history.append(self.current_value)
            self._dx_history.append(dx)

        if self.current_value is None:
            raise ValueError("ADX could not be calculated from the provided series.")

        self.previous_candle = candles[-1]
        return self.current_value

    # ------------------------------------------------------------------
    # Incremental update (Live)
    # ------------------------------------------------------------------
    def update(self, candle: Candle):
        if self.current_value is None:
            raise ValueError("ADX must be initialized using calculate() before update().")

        if self.previous_candle is None:
            raise ValueError("ADX has no previous candle state.")

        tr, plus_dm, minus_dm = self._calculate_tr_dm(candle, self.previous_candle)

        self._smooth_values(tr, plus_dm, minus_dm)
        plus_di, minus_di, dx = self._calculate_di_dx()

        self.previous_value = self.current_value
        self.current_value = (
            (self.current_value * (self.period - 1) + dx) / self.period
        )

        self.plus_di = plus_di
        self.minus_di = minus_di
        self.dx = dx
        self.previous_candle = candle

        self._history.append(self.current_value)
        self._dx_history.append(dx)

        return self.current_value

    # ------------------------------------------------------------------
    # Trend helpers
    # ------------------------------------------------------------------
    def is_trending(self, threshold: float | None = None) -> bool:
        if self.current_value is None:
            return False
        threshold = threshold if threshold is not None else self.trend_threshold
        return self.current_value >= threshold

    def is_weak_or_ranging(self, threshold: float | None = None) -> bool:
        return not self.is_trending(threshold=threshold)

    def directional_bias(self) -> str:
        if self.plus_di is None or self.minus_di is None:
            return self.BIAS_NEUTRAL
        if self.plus_di > self.minus_di:
            return self.BIAS_BUY
        if self.minus_di > self.plus_di:
            return self.BIAS_SELL
        return self.BIAS_NEUTRAL

    def has_buy_bias(self) -> bool:
        return self.directional_bias() == self.BIAS_BUY

    def has_sell_bias(self) -> bool:
        return self.directional_bias() == self.BIAS_SELL

    def is_strengthening(self) -> bool:
        if self.current_value is None or self.previous_value is None:
            return False
        return self.current_value > self.previous_value

    def is_weakening(self) -> bool:
        if self.current_value is None or self.previous_value is None:
            return False
        return self.current_value < self.previous_value

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------
    def value(self) -> float:
        if self.current_value is None:
            raise ValueError("ADX has not been initialized.")
        return self.current_value

    def history(self) -> list[float]:
        return list(self._history)

    def dx_history(self) -> list[float]:
        return list(self._dx_history)

    def state(self) -> dict:
        return {
            "adx": self.current_value,
            "previous_adx": self.previous_value,
            "plus_di": self.plus_di,
            "minus_di": self.minus_di,
            "dx": self.dx,
            "smoothed_tr": self.smoothed_tr,
            "smoothed_plus_dm": self.smoothed_plus_dm,
            "smoothed_minus_dm": self.smoothed_minus_dm,
            "trend_threshold": self.trend_threshold,
            "is_trending": self.is_trending(),
            "directional_bias": self.directional_bias(),
        }

    # ------------------------------------------------------------------
    # Internal logic
    # ------------------------------------------------------------------
    def _reset(self):
        self._history.clear()
        self._dx_history.clear()

        self.current_value = None
        self.previous_value = None

        self.plus_di = None
        self.minus_di = None
        self.dx = None

        self.smoothed_tr = None
        self.smoothed_plus_dm = None
        self.smoothed_minus_dm = None

        self.previous_candle = None

    def _calculate_tr_dm(self, current: Candle, previous: Candle):
        tr = max(
            current.high - current.low,
            abs(current.high - previous.close),
            abs(current.low - previous.close),
        )

        up_move = current.high - previous.high
        down_move = previous.low - current.low

        plus_dm = up_move if up_move > down_move and up_move > 0 else 0.0
        minus_dm = down_move if down_move > up_move and down_move > 0 else 0.0

        return tr, plus_dm, minus_dm

    def _smooth_values(self, tr: float, plus_dm: float, minus_dm: float):
        """
        Wilder's smoothing for the running totals:
          new = old - old/N + current
            = old * (N-1)/N + current
        Note: operates on sums (not averages), so the +DM/-DM scale matches
        the TR scale when computing +DI/-DI ratios.
        """
        self.smoothed_tr = self.smoothed_tr - self.smoothed_tr / self.period + tr
        self.smoothed_plus_dm = self.smoothed_plus_dm - self.smoothed_plus_dm / self.period + plus_dm
        self.smoothed_minus_dm = self.smoothed_minus_dm - self.smoothed_minus_dm / self.period + minus_dm

    def _calculate_di_dx(self):
        if self.smoothed_tr == 0:
            return 0.0, 0.0, 0.0

        plus_di = 100 * self.smoothed_plus_dm / self.smoothed_tr
        minus_di = 100 * self.smoothed_minus_dm / self.smoothed_tr

        denominator = plus_di + minus_di
        dx = 100 * abs(plus_di - minus_di) / denominator if denominator != 0 else 0.0

        return plus_di, minus_di, dx