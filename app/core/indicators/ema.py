from collections import deque

from app.core.indicators.base_indicator import BaseIndicator
from app.core.market.market_series import MarketSeries
from app.core.market.candle import Candle


class EMAIndicator(BaseIndicator):
    """
    Exponential Moving Average (EMA)

    Features:
    - Supports open, high, low, or close as price source
    - Batch calculation for backtesting
    - Incremental update for live trading
    - Offset-based slope comparison
    - Trend and flatness helpers
    """

    VALID_SOURCES = {"open", "high", "low", "close"}

    def __init__(self, period: int, source: str = "close", offset: int = 3):
        super().__init__(period)

        if period <= 0:
            raise ValueError("period must be greater than 0")

        if offset <= 0:
            raise ValueError("offset must be greater than 0")

        source = source.lower()
        if source not in self.VALID_SOURCES:
            raise ValueError(
                f"Invalid source '{source}'. Must be one of: {sorted(self.VALID_SOURCES)}"
            )

        self.source = source
        self.offset = offset
        self.alpha = 2 / (period + 1)

        # Stores latest EMA values so slope can be measured against N candles ago
        self._history = deque(maxlen=offset + 1)

        self.current_value = None
        self.previous_value = None  # EMA value N candles ago

    # -----------------------------
    # Batch calculation (Backtest)
    # -----------------------------
    def calculate(self, series: MarketSeries):
        prices = self._extract_prices(series)

        if len(prices) < self.period:
            raise ValueError(
                f"EMA requires at least {self.period} values to calculate."
            )

        # Initial seed = SMA of first period values
        ema = sum(prices[:self.period]) / self.period
        ema_values = [ema]

        # Continue EMA from remaining prices
        for price in prices[self.period:]:
            ema = (price * self.alpha) + (ema * (1 - self.alpha))
            ema_values.append(ema)

        self.current_value = ema_values[-1]

        self._history.clear()
        for value in ema_values[-self._history.maxlen:]:
            self._history.append(value)

        self.previous_value = (
            self._history[0] if len(self._history) >= self.offset + 1 else None
        )

        return self.current_value

    # -----------------------------
    # Incremental update (Live)
    # -----------------------------
    def update(self, candle: Candle):
        if self.current_value is None:
            raise ValueError("EMA must be initialized using calculate() before update().")

        price = self._extract_price_from_candle(candle)
        new_ema = (price * self.alpha) + (self.current_value * (1 - self.alpha))

        self._history.append(new_ema)
        self.current_value = new_ema
        self.previous_value = (
            self._history[0] if len(self._history) >= self.offset + 1 else None
        )

        return self.current_value

    # -----------------------------
    # Slope / Gradient helpers
    # -----------------------------
    def slope(self, offset: int | None = None) -> float:
        """
        Raw EMA slope over the configured or provided offset.
        Positive => rising EMA
        Negative => falling EMA
        """
        offset = offset or self.offset

        if offset <= 0:
            raise ValueError("offset must be greater than 0")

        if len(self._history) < offset + 1:
            return 0.0

        return self._history[-1] - self._history[-(offset + 1)]

    def slope_per_candle(self, offset: int | None = None) -> float:
        """
        Average EMA slope per candle over the configured or provided offset.
        """
        offset = offset or self.offset

        if offset <= 0:
            raise ValueError("offset must be greater than 0")

        return self.slope(offset=offset) / offset

    def is_rising(self, offset: int | None = None) -> bool:
        """
        True if current EMA is above EMA 'offset' candles ago.
        """
        return self.slope(offset=offset) > 0.01

    def is_falling(self, offset: int | None = None) -> bool:
        """
        True if current EMA is below EMA 'offset' candles ago.
        """
        return self.slope(offset=offset) < -0.01

    def is_flat(self, slope_threshold: float, offset: int | None = None) -> bool:
        """
        True if EMA slope magnitude is below threshold.
        """
        return abs(self.slope(offset=offset)) < slope_threshold

    def is_uptrend(self, current_price: float, slope_threshold: float = 0.0, offset: int | None = None) -> bool:
        """
        Confirmed uptrend:
        - current price above EMA
        - EMA slope positive enough
        """
        if self.current_value is None:
            return False

        return (
            current_price > self.current_value
            and self.slope(offset=offset) >= slope_threshold
        )

    def is_downtrend(self, current_price: float, slope_threshold: float = 0.0, offset: int | None = None) -> bool:
        """
        Confirmed downtrend:
        - current price below EMA
        - EMA slope negative enough
        """
        if self.current_value is None:
            return False

        return (
            current_price < self.current_value
            and self.slope(offset=offset) <= -slope_threshold
        )

    # -----------------------------
    # Convenience methods
    # -----------------------------
    def value(self) -> float:
        if self.current_value is None:
            raise ValueError("EMA has not been initialized.")
        return self.current_value

    def history(self) -> list[float]:
        return list(self._history)

    # -----------------------------
    # Internal helpers
    # -----------------------------
    def _extract_prices(self, series: MarketSeries) -> list[float]:
        if self.source == "open":
            return series.opens()
        if self.source == "high":
            return series.highs()
        if self.source == "low":
            return series.lows()
        return series.closes()

    def _extract_price_from_candle(self, candle: Candle) -> float:
        return getattr(candle, self.source)