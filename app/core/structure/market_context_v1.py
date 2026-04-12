# app/core/market/market_context.py

from dataclasses import dataclass
from typing import List, Optional, Tuple

from app.core.market.market_series import MarketSeries
from app.core.market.candle import Candle
from app.core.indicators.atr import ATRIndicator


@dataclass
class MarketContextState:
    trend: str  # "UP", "DOWN", "RANGE"
    last_swing_high: Optional[float]
    last_swing_low: Optional[float]
    support_zone: Optional[Tuple[float, float]]
    resistance_zone: Optional[Tuple[float, float]]
    in_support: bool
    in_resistance: bool
    atr: Optional[float]


class MarketContext:
    """
    Institutional-grade market structure engine.
    Handles:
    - Swing detection
    - Trend classification
    - Range detection
    - Structure-based S/R zones
    - ATR volatility integration
    """

    def __init__(self,
                 swing_lookback: int = 3,
                 atr_period: int = 14,
                 zone_atr_multiplier: float = 1.0):

        self.swing_lookback = swing_lookback
        self.atr_indicator = ATRIndicator(atr_period)
        self.zone_atr_multiplier = zone_atr_multiplier

        self.swing_highs: List[float] = []
        self.swing_lows: List[float] = []

        self.state: Optional[MarketContextState] = None

    # ==========================================================
    # PUBLIC API
    # ==========================================================

    def calculate(self, series: MarketSeries) -> MarketContextState:
        """
        Full batch calculation (used in backtesting).
        """

        atr = self.atr_indicator.calculate(series)

        self._detect_swings(series)
        trend = self._determine_trend()

        support_zone, resistance_zone = self._build_zones(atr)

        current_price = series.closes()[-1]

        in_support = self._is_in_zone(current_price, support_zone)
        in_resistance = self._is_in_zone(current_price, resistance_zone)

        self.state = MarketContextState(
            trend=trend,
            last_swing_high=self.swing_highs[-1] if self.swing_highs else None,
            last_swing_low=self.swing_lows[-1] if self.swing_lows else None,
            support_zone=support_zone,
            resistance_zone=resistance_zone,
            in_support=in_support,
            in_resistance=in_resistance,
            atr=atr
        )

        return self.state

    def update(self, series: MarketSeries, candle: Candle) -> MarketContextState:
        """
        Incremental update for live trading.
        Assumes series already includes the new closed candle.
        """

        atr = self.atr_indicator.update(candle)

        self._update_swings(series)
        trend = self._determine_trend()

        support_zone, resistance_zone = self._build_zones(atr)

        current_price = candle.close

        in_support = self._is_in_zone(current_price, support_zone)
        in_resistance = self._is_in_zone(current_price, resistance_zone)

        self.state = MarketContextState(
            trend=trend,
            last_swing_high=self.swing_highs[-1] if self.swing_highs else None,
            last_swing_low=self.swing_lows[-1] if self.swing_lows else None,
            support_zone=support_zone,
            resistance_zone=resistance_zone,
            in_support=in_support,
            in_resistance=in_resistance,
            atr=atr
        )

        return self.state

    # ==========================================================
    # INTERNAL LOGIC
    # ==========================================================

    def _detect_swings(self, series: MarketSeries):
        candles = series._candles

        self.swing_highs.clear()
        self.swing_lows.clear()

        for i in range(self.swing_lookback, len(candles) - self.swing_lookback):
            window = candles[i - self.swing_lookback:i + self.swing_lookback + 1]
            center = candles[i]

            if center.high == max(c.high for c in window):
                self.swing_highs.append(center.high)

            if center.low == min(c.low for c in window):
                self.swing_lows.append(center.low)

    def _update_swings(self, series: MarketSeries):
        """
        Only checks the latest possible swing to avoid full recalculation.
        """
        candles = series._candles
        i = len(candles) - self.swing_lookback - 1

        if i <= self.swing_lookback:
            return

        window = candles[i - self.swing_lookback:i + self.swing_lookback + 1]
        center = candles[i]

        if center.high == max(c.high for c in window):
            self.swing_highs.append(center.high)

        if center.low == min(c.low for c in window):
            self.swing_lows.append(center.low)

    def _determine_trend(self) -> str:
        previous_trend = self.state.trend if self.state else "RANGE"

        if len(self.swing_highs) < 2 or len(self.swing_lows) < 2:
            return "RANGE"

        last_high = self.swing_highs[-1]
        prev_high = self.swing_highs[-2]

        last_low = self.swing_lows[-1]
        prev_low = self.swing_lows[-2]

        # === Structure Conditions ===
        higher_high = last_high > prev_high
        higher_low = last_low > prev_low

        lower_high = last_high < prev_high
        lower_low = last_low < prev_low

        # === Strong Trend Confirmation ===
        if higher_high and higher_low:
            return "UP"

        if lower_high and lower_low:
            return "DOWN"

        # === Structural Break Logic ===
        # Only invalidate trend on decisive violation

        if previous_trend == "UP" and lower_low:
            # We broke previous higher low
            return "RANGE"

        if previous_trend == "DOWN" and higher_high:
            # We broke previous lower high
            return "RANGE"

        # Otherwise remain in prior state
        return previous_trend

    def _build_zones(self, atr: Optional[float]):
        if atr is None:
            return None, None

        buffer = atr * self.zone_atr_multiplier

        support = None
        resistance = None

        if self.swing_lows:
            level = self.swing_lows[-1]
            support = (level - buffer, level + buffer)

        if self.swing_highs:
            level = self.swing_highs[-1]
            resistance = (level - buffer, level + buffer)

        return support, resistance

    def _is_in_zone(self, price: float, zone: Optional[Tuple[float, float]]) -> bool:
        if zone is None:
            return False

        lower, upper = zone
        return lower <= price <= upper