from collections import deque

from app.core.indicators.base_indicator import BaseIndicator
from app.core.market.market_series import MarketSeries
from app.core.market.candle import Candle


class RSIIndicator(BaseIndicator):
    """
    Relative Strength Index (RSI) using Wilder's smoothing.

    Features:
    - Batch calculation for backtesting
    - Incremental update for live trading
    - Tracks current and previous RSI values
    - Offset-based rising/falling checks for noisy lower timeframes
    - Uses Wilder smoothing to match common trading platforms
    """

    def __init__(self, period: int, slope_offset: int = 3, history_size: int | None = None):
        super().__init__(period)

        if period <= 0:
            raise ValueError("period must be greater than 0")

        if slope_offset <= 0:
            raise ValueError("slope_offset must be greater than 0")

        self.previous_close = None
        self.avg_gain = None
        self.avg_loss = None

        self.current_value = None
        self.previous_value = None

        self.slope_offset = slope_offset

        # Must always be large enough to compare current RSI with RSI N candles ago
        required_history = slope_offset + 1
        self._history = deque(maxlen=max(history_size or required_history, required_history))

    # ---------------------------------
    # Batch calculation (Backtest)
    # ---------------------------------
    def calculate(self, series: MarketSeries):
        candles = series._candles

        if len(candles) < self.period + 1:
            raise ValueError(
                f"RSI requires at least {self.period + 1} candles."
            )

        closes = [c.close for c in candles]

        gains = []
        losses = []

        # 1. Compute gains and losses from close-to-close changes
        for i in range(1, len(closes)):
            change = closes[i] - closes[i - 1]
            gains.append(max(change, 0.0))
            losses.append(max(-change, 0.0))

        # 2. Initial average gain/loss using SMA of first N values
        avg_gain = sum(gains[:self.period]) / self.period
        avg_loss = sum(losses[:self.period]) / self.period

        # 3. First RSI value
        rsi = self._compute_rsi(avg_gain, avg_loss)
        rsi_values = [rsi]

        # 4. Wilder smoothing for remaining values
        for i in range(self.period, len(gains)):
            avg_gain = ((avg_gain * (self.period - 1)) + gains[i]) / self.period
            avg_loss = ((avg_loss * (self.period - 1)) + losses[i]) / self.period

            rsi = self._compute_rsi(avg_gain, avg_loss)
            rsi_values.append(rsi)

        self.avg_gain = avg_gain
        self.avg_loss = avg_loss
        self.previous_close = closes[-1]

        self.current_value = rsi_values[-1]
        self.previous_value = rsi_values[-2] if len(rsi_values) >= 2 else None

        self._history.clear()
        for value in rsi_values[-self._history.maxlen:]:
            self._history.append(value)

        return self.current_value

    # ---------------------------------
    # Incremental update (Live)
    # ---------------------------------
    def update(self, candle: Candle):
        if (
            self.current_value is None
            or self.previous_close is None
            or self.avg_gain is None
            or self.avg_loss is None
        ):
            raise ValueError(
                "RSI must be initialized with calculate() before update()."
            )

        change = candle.close - self.previous_close
        gain = max(change, 0.0)
        loss = max(-change, 0.0)

        # Wilder smoothing
        self.avg_gain = ((self.avg_gain * (self.period - 1)) + gain) / self.period
        self.avg_loss = ((self.avg_loss * (self.period - 1)) + loss) / self.period

        self.previous_value = self.current_value
        self.current_value = self._compute_rsi(self.avg_gain, self.avg_loss)

        self._history.append(self.current_value)
        self.previous_close = candle.close

        return self.current_value

    # ---------------------------------
    # Momentum / direction checks
    # ---------------------------------
    # def is_rising(self, offset: int | None = None) -> bool:
    #     """
    #     Returns True if current RSI is greater than RSI 'offset' candles ago.
    #     """
    #     offset = offset or self.slope_offset
    #
    #     if offset <= 0:
    #         raise ValueError("offset must be greater than 0")
    #
    #     if len(self._history) < offset + 1:
    #         return False
    #
    #     return self._history[-1] > self._history[-(offset + 1)]
    #
    # def is_falling(self, offset: int | None = None) -> bool:
    #     """
    #     Returns True if current RSI is less than RSI 'offset' candles ago.
    #     """
    #     offset = offset or self.slope_offset
    #
    #     if offset <= 0:
    #         raise ValueError("offset must be greater than 0")
    #
    #     if len(self._history) < offset + 1:
    #         return False
    #
    #     return self._history[-1] < self._history[-(offset + 1)]

    def is_rising(self, offset: int | None = None, strict: bool = False) -> bool:
        """
        Determines if RSI is rising based on a Linear Regression slope
        calculated across the entire window.
        """
        slope, r_squared = self._calculate_regression_stats(offset)

        # A true rising RSI should have a positive slope.
        # If 'strict' is true, we also check if the fit (r_squared) is high,
        # meaning the move is smooth, not erratic.
        if strict:
            return slope > 0.1 and r_squared > 0.6

        return slope > 0

    def is_falling(self, offset: int | None = None, strict: bool = False) -> bool:
        """
        Determines if RSI is falling based on a Linear Regression slope.
        """
        slope, r_squared = self._calculate_regression_stats(offset)

        if strict:
            return slope < -0.1 and r_squared > 0.6

        return slope < 0

    def get_strength(self, offset: int | None = None) -> float:
        """
        Returns a 'Velocity' score.
        Higher = RSI is moving fast and consistently.
        """
        slope, r_squared = self._calculate_regression_stats(offset)
        return slope * r_squared

    # ---------------------------------
    # The Math Engine: Linear Regression
    # ---------------------------------

    def _calculate_regression_stats(self, offset: int | None = None) -> tuple[float, float]:
        """
        Calculates the Slope and R-Squared (consistency) of the RSI window.
        """
        n = offset or self.slope_offset
        if len(self._history) < n + 1:
            return 0.0, 0.0

        # Get the slice of history we care about
        y_values = list(self._history)[-(n + 1):]
        x_values = list(range(len(y_values)))  # 0, 1, 2, 3...

        sum_x = sum(x_values)
        sum_y = sum(y_values)
        sum_xx = sum(x * x for x in x_values)
        sum_xy = sum(x * y for x, y in zip(x_values, y_values))

        count = len(y_values)

        # Calculate Slope (m)
        denominator = (count * sum_xx - sum_x ** 2)
        if denominator == 0:
            return 0.0, 0.0

        slope = (count * sum_xy - sum_x * sum_y) / denominator

        # Calculate R-Squared (Coefficient of Determination)
        # This tells us how 'noisy' the slope is. 1.0 = perfect line.
        y_mean = sum_y / count
        ss_tot = sum((y - y_mean) ** 2 for y in y_values)
        if ss_tot == 0:
            return slope, 1.0

        ss_res = sum((y - (slope * x + (sum_y - slope * sum_x) / count)) ** 2
                     for x, y in zip(x_values, y_values))

        r_squared = 1 - (ss_res / ss_tot)

        return slope, r_squared

    def slope(self, offset: int | None = None) -> float:
        """
        Returns RSI change between current value and RSI 'offset' candles ago.
        Positive => rising
        Negative => falling
        """
        offset = offset or self.slope_offset

        if offset <= 0:
            raise ValueError("offset must be greater than 0")

        if len(self._history) < offset + 1:
            raise ValueError(
                f"Not enough RSI history to compute slope with offset={offset}"
            )

        return self._history[-1] - self._history[-(offset + 1)]

    # ---------------------------------
    # Helpers
    # ---------------------------------
    def _compute_rsi(self, avg_gain: float, avg_loss: float) -> float:
        """
        Compute RSI safely from average gain/loss.
        Edge cases:
        - avg_loss == 0 and avg_gain == 0 -> RSI = 50
        - avg_loss == 0 -> RSI = 100
        - avg_gain == 0 -> RSI = 0
        """
        if avg_loss == 0 and avg_gain == 0:
            return 50.0
        if avg_loss == 0:
            return 100.0
        if avg_gain == 0:
            return 0.0

        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    # ---------------------------------
    # Optional convenience methods
    # ---------------------------------
    def value(self) -> float:
        if self.current_value is None:
            raise ValueError("RSI has not been initialized.")
        return self.current_value

    def history(self) -> list[float]:
        return list(self._history)