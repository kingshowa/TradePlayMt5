# app/core/market/market_context.py
import statistics
from dataclasses import dataclass
from typing import List, Optional, Tuple
from collections import deque

from app.core.market.market_series import MarketSeries
from app.core.market.candle import Candle


# ==========================================================
# STATE
# ==========================================================

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
    trend_strength: Optional[str]  # "HIGH", "LOW", None


# ==========================================================
# STREAMING MARKET CONTEXT ENGINE
# ==========================================================

class MarketContext:

    def __init__(self,
                 swing_lookback: int = 3,
                 zone_atr_multiplier: float = 1.0,
                 max_candles: int = 500,
                 max_swings: int = 500):

        self.swing_lookback = swing_lookback
        self.zone_atr_multiplier = zone_atr_multiplier

        # Internal rolling buffers (memory safe)
        self.candles: deque = deque(maxlen=max_candles)
        self.swing_highs: deque = deque(maxlen=max_swings)
        self.swing_lows: deque = deque(maxlen=max_swings)

        self.state: Optional[MarketContextState] = None

    # ==========================================================
    # INITIALIZATION (BATCH ONCE)
    # ==========================================================

    def initialize(self, series: MarketSeries, atr: Optional[float]) -> MarketContextState:
        """
        Called once at startup.
        """

        for candle in series._candles:
            self.candles.append(candle)

        self._detect_initial_swings()

        return self._build_state(atr)

    # ==========================================================
    # LIVE UPDATE (STREAMING)
    # ==========================================================

    def update(self,
               new_candle: Candle,
               atr: Optional[float]) -> MarketContextState:
        """
        Called on every new CLOSED candle.
        """

        self.candles.append(new_candle)

        self._update_swings()

        return self._build_state(atr)

    # ==========================================================
    # STATE BUILDER
    # ==========================================================

    def _build_state(self, atr: Optional[float]) -> MarketContextState:

        trend = self._determine_trend()
        trend_strength = self._determine_trend_strength(trend)

        support_zone, resistance_zone = self._build_zones(atr)

        current_price = self.candles[-1].close

        state = MarketContextState(
            trend=trend,
            last_swing_high=self.swing_highs[-1] if self.swing_highs else None,
            last_swing_low=self.swing_lows[-1] if self.swing_lows else None,
            support_zone=support_zone,
            resistance_zone=resistance_zone,
            in_support=self._is_in_zone(current_price, support_zone),
            in_resistance=self._is_in_zone(current_price, resistance_zone),
            atr=atr,
            trend_strength=trend_strength
        )

        self.state = state
        return state

    # ==========================================================
    # SWING DETECTION
    # ==========================================================

    def _detect_initial_swings(self):
        candles = list(self.candles)

        for i in range(self.swing_lookback, len(candles) - self.swing_lookback):
            window = candles[i - self.swing_lookback:i + self.swing_lookback + 1]
            center = candles[i]

            if center.high == max(c.high for c in window):
                self.swing_highs.append(center.high)

            if center.low == min(c.low for c in window):
                self.swing_lows.append(center.low)

    def _update_swings(self):
        """
        Only checks latest potential swing.
        """

        candles = list(self.candles)

        i = len(candles) - self.swing_lookback - 1
        if i <= self.swing_lookback:
            return

        window = candles[i - self.swing_lookback:i + self.swing_lookback + 1]
        center = candles[i]

        if center.high == max(c.high for c in window):
            self.swing_highs.append(center.high)

        if center.low == min(c.low for c in window):
            self.swing_lows.append(center.low)

    # ==========================================================
    # TREND LOGIC
    # ==========================================================

    # def _determine_trend(self) -> str:
    #
    #     previous_trend = self.state.trend if self.state else "RANGE"
    #
    #     if len(self.swing_highs) < 2 or len(self.swing_lows) < 2:
    #         return "RANGE"
    #
    #     last_high = self.swing_highs[-1]
    #     prev_high = self.swing_highs[-2]
    #
    #     last_low = self.swing_lows[-1]
    #     prev_low = self.swing_lows[-2]
    #
    #     higher_high = last_high > prev_high
    #     higher_low = last_low > prev_low
    #
    #     lower_high = last_high < prev_high
    #     lower_low = last_low < prev_low
    #
    #     if higher_high and higher_low:
    #         return "UP"
    #
    #     if lower_high and lower_low:
    #         return "DOWN"
    #
    #     if previous_trend == "UP" and lower_low:
    #         return "RANGE"
    #
    #     if previous_trend == "DOWN" and higher_high:
    #         return "RANGE"
    #
    #     return previous_trend

    def _determine_trend(self) -> str:
        previous_trend = self.state.trend if self.state else "RANGE"

        if len(self.swing_highs) < 2 or len(self.swing_lows) < 2:
            return "RANGE"

        current_close = self.candles[-1].close

        last_high = self.swing_highs[-1]
        prev_high = self.swing_highs[-2]

        last_low = self.swing_lows[-1]
        prev_low = self.swing_lows[-2]

        # Fast live structure break
        if previous_trend == "DOWN" and current_close > last_high:
            return "RANGE F"

        if previous_trend == "UP" and current_close < last_low:
            return "RANGE F"

        # if previous_trend == "RANGE" and current_close < last_low:
        #     return "DOWN"
        #
        # if previous_trend == "RANGE" and current_close > last_high:
        #     return "UP"

        # Confirmed swing structure fallback
        higher_high = last_high > prev_high
        higher_low = last_low > prev_low

        lower_high = last_high < prev_high
        lower_low = last_low < prev_low

        if higher_high and higher_low:
            return "UP"

        if lower_high and lower_low:
            return "DOWN"

        return previous_trend

    # ==========================================================
    # TREND STRENGTH
    # ==========================================================

    def _determine_trend_strength(self, trend: str) -> Optional[str]:

        if trend == "RANGE":
            return None

        if len(self.swing_highs) < 3 or len(self.swing_lows) < 3:
            return "LOW"

        confirmations = 0

        if trend == "UP":
            if self.swing_highs[-1] > self.swing_highs[-2]:
                confirmations += 1
            if self.swing_lows[-1] > self.swing_lows[-2]:
                confirmations += 1

        if trend == "DOWN":
            if self.swing_highs[-1] < self.swing_highs[-2]:
                confirmations += 1
            if self.swing_lows[-1] < self.swing_lows[-2]:
                confirmations += 1

        return "HIGH" if confirmations == 2 else "LOW"

    # ==========================================================
    # ZONES
    # ==========================================================

    # def _build_zones(self, atr: Optional[float], zone_points: int = 3):
    #
    #     if atr is None:
    #         return None, None
    #
    #     buffer = atr * self.zone_atr_multiplier
    #
    #     support = None
    #     resistance = None
    #
    #     if len(self.swing_lows) >= 1:
    #         recent_lows = list(self.swing_lows)[-zone_points:]
    #         support_level = sum(recent_lows) / len(recent_lows)
    #         support = (support_level - buffer, support_level + buffer)
    #
    #     if len(self.swing_highs) >= 1:
    #         recent_highs = list(self.swing_highs)[-zone_points:]
    #         resistance_level = sum(recent_highs) / len(recent_highs)
    #         resistance = (resistance_level - buffer, resistance_level + buffer)
    #
    #     return support, resistance

    def _build_zones(self, atr: Optional[float]):
        if atr is None or not self.swing_lows or not self.swing_highs:
            return None, None

        # 1. Dynamic buffer based on ATR
        buffer = atr * self.zone_atr_multiplier

        # 2. Define how many recent swings to look at (3-5 is usually ideal)
        lookback_n = 3

        # Ensure we have enough data
        num_lows = min(lookback_n, len(self.swing_lows))
        num_highs = min(lookback_n, len(self.swing_highs))

        # 3. Calculate Median Levels
        # This ignores the "freak" outlier wicks and finds the true structural center
        recent_lows = list(self.swing_lows)[-num_lows:]
        recent_highs = list(self.swing_highs)[-num_highs:]

        median_low = statistics.median(recent_lows)
        median_high = statistics.median(recent_highs)

        # 4. Construct the "Median Zones"
        support = (median_low - buffer, median_low + buffer)
        resistance = (median_high - buffer, median_high + buffer)

        return support, resistance

    # ==========================================================
    # ZONE CHECK
    # ==========================================================

    def _is_in_zone(self,
                    price: float,
                    zone: Optional[Tuple[float, float]]) -> bool:

        if zone is None:
            return False

        lower, upper = zone
        return lower <= price <= upper