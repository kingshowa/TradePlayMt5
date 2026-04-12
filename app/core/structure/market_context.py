# app/core/market/market_context.py

import statistics
from dataclasses import dataclass
from typing import Deque, List, Optional, Tuple, Literal
from collections import deque

from app.core.market.market_series import MarketSeries
from app.core.market.candle import Candle


Trend = Literal["UP", "DOWN", "RANGE"]
TrendStrength = Literal["HIGH", "LOW"]
StructureEvent = Literal["BOS_UP", "BOS_DOWN", "CHOCH_UP", "CHOCH_DOWN", "NONE"]
SwingLabel = Literal["HH", "HL", "LH", "LL", "H", "L"]


# ==========================================================
# SWING MODELS
# ==========================================================

@dataclass
class SwingPoint:
    price: float
    index: int
    kind: Literal["HIGH", "LOW"]
    label: Optional[SwingLabel] = None


# ==========================================================
# STATE
# ==========================================================

@dataclass
class MarketContextState:
    trend: Trend
    structure_event: StructureEvent
    last_swing_high: Optional[float]
    last_swing_low: Optional[float]
    protected_high: Optional[float]
    protected_low: Optional[float]
    support_zone: Optional[Tuple[float, float]]
    resistance_zone: Optional[Tuple[float, float]]
    in_support: bool
    in_resistance: bool
    atr: Optional[float]
    trend_strength: Optional[TrendStrength]


# ==========================================================
# STREAMING MARKET CONTEXT ENGINE
# ==========================================================

class MarketContext:
    def __init__(
        self,
        swing_lookback: int = 3,
        zone_atr_multiplier: float = 1.0,
        break_atr_multiplier: float = 0.15,
        max_candles: int = 500,
        max_swings: int = 500,
    ):
        self.swing_lookback = swing_lookback
        self.zone_atr_multiplier = zone_atr_multiplier
        self.break_atr_multiplier = break_atr_multiplier

        self.candles: Deque[Candle] = deque(maxlen=max_candles)
        self.swing_highs: Deque[SwingPoint] = deque(maxlen=max_swings)
        self.swing_lows: Deque[SwingPoint] = deque(maxlen=max_swings)

        self.state: Optional[MarketContextState] = None

    # ==========================================================
    # INITIALIZATION (BATCH ONCE)
    # ==========================================================

    def initialize(self, series: MarketSeries, atr: Optional[float]) -> MarketContextState:
        for candle in series._candles:
            self.candles.append(candle)

        self._detect_initial_swings()
        return self._build_state(atr)

    # ==========================================================
    # LIVE UPDATE (STREAMING)
    # ==========================================================

    def update(self, new_candle: Candle, atr: Optional[float]) -> MarketContextState:
        self.candles.append(new_candle)
        self._update_swings()
        return self._build_state(atr)

    # ==========================================================
    # STATE BUILDER
    # ==========================================================

    def _build_state(self, atr: Optional[float]) -> MarketContextState:
        trend, structure_event, protected_high, protected_low = self._determine_structure(atr)
        trend_strength = self._determine_trend_strength(trend, structure_event)
        support_zone, resistance_zone = self._build_zones(atr)

        current_price = self.candles[-1].close

        state = MarketContextState(
            trend=trend,
            structure_event=structure_event,
            last_swing_high=self.swing_highs[-1].price if self.swing_highs else None,
            last_swing_low=self.swing_lows[-1].price if self.swing_lows else None,
            protected_high=protected_high,
            protected_low=protected_low,
            support_zone=support_zone,
            resistance_zone=resistance_zone,
            in_support=self._is_in_zone(current_price, support_zone),
            in_resistance=self._is_in_zone(current_price, resistance_zone),
            atr=atr,
            trend_strength=trend_strength,
        )

        self.state = state
        return state

    # ==========================================================
    # SWING DETECTION
    # ==========================================================

    def _detect_initial_swings(self) -> None:
        candles = list(self.candles)

        for i in range(self.swing_lookback, len(candles) - self.swing_lookback):
            self._try_register_swing(candles, i)

        self._relabel_swings()

    def _update_swings(self) -> None:
        candles = list(self.candles)

        i = len(candles) - self.swing_lookback - 1
        if i <= self.swing_lookback:
            return

        self._try_register_swing(candles, i)
        self._relabel_swings()

    def _try_register_swing(self, candles: List[Candle], i: int) -> None:
        window = candles[i - self.swing_lookback : i + self.swing_lookback + 1]
        center = candles[i]

        is_swing_high = center.high == max(c.high for c in window)
        is_swing_low = center.low == min(c.low for c in window)

        if is_swing_high:
            self._append_swing_high(SwingPoint(price=center.high, index=i, kind="HIGH"))

        if is_swing_low:
            self._append_swing_low(SwingPoint(price=center.low, index=i, kind="LOW"))

    def _append_swing_high(self, swing: SwingPoint) -> None:
        if self.swing_highs and self.swing_highs[-1].index == swing.index:
            return

        # Replace weaker consecutive high with stronger one if needed
        if self.swing_highs and len(self.swing_lows) == 0:
            if swing.price >= self.swing_highs[-1].price:
                self.swing_highs[-1] = swing
            return

        self.swing_highs.append(swing)

    def _append_swing_low(self, swing: SwingPoint) -> None:
        if self.swing_lows and self.swing_lows[-1].index == swing.index:
            return

        # Replace weaker consecutive low with stronger one if needed
        if self.swing_lows and len(self.swing_highs) == 0:
            if swing.price <= self.swing_lows[-1].price:
                self.swing_lows[-1] = swing
            return

        self.swing_lows.append(swing)

    def _relabel_swings(self) -> None:
        highs = list(self.swing_highs)
        lows = list(self.swing_lows)

        for idx in range(len(highs)):
            if idx == 0:
                highs[idx].label = "H"
            else:
                highs[idx].label = "HH" if highs[idx].price > highs[idx - 1].price else "LH"

        for idx in range(len(lows)):
            if idx == 0:
                lows[idx].label = "L"
            else:
                lows[idx].label = "HL" if lows[idx].price > lows[idx - 1].price else "LL"

        self.swing_highs = deque(highs, maxlen=self.swing_highs.maxlen)
        self.swing_lows = deque(lows, maxlen=self.swing_lows.maxlen)

    # ==========================================================
    # STRUCTURE INTERPRETATION
    # ==========================================================

    def _determine_structure(
        self, atr: Optional[float]
    ) -> Tuple[Trend, StructureEvent, Optional[float], Optional[float]]:
        previous_trend: Trend = self.state.trend if self.state else "RANGE"

        if len(self.swing_highs) < 2 or len(self.swing_lows) < 2:
            return "RANGE", "NONE", self._get_last_high(), self._get_last_low()

        current_close = self.candles[-1].close

        last_high = self.swing_highs[-1].price
        prev_high = self.swing_highs[-2].price

        last_low = self.swing_lows[-1].price
        prev_low = self.swing_lows[-2].price

        bullish_structure = last_high > prev_high and last_low > prev_low
        bearish_structure = last_high < prev_high and last_low < prev_low

        protected_high = self._get_protected_high(previous_trend)
        protected_low = self._get_protected_low(previous_trend)

        # Fallback if protected points are not yet mature
        if protected_high is None:
            protected_high = last_high

        if protected_low is None:
            protected_low = last_low

        # ------------------------------------------------------
        # EVENT DETECTION (live price vs protected structure)
        # ------------------------------------------------------
        if previous_trend == "UP":
            if self._is_valid_break_up(current_close, last_high, atr):
                return "UP", "BOS_UP", protected_high, protected_low

            if self._is_valid_break_down(current_close, protected_low, atr):
                return "RANGE", "CHOCH_DOWN", protected_high, protected_low

            if bullish_structure:
                return "UP", "NONE", protected_high, protected_low

            if bearish_structure:
                return "RANGE", "NONE", protected_high, protected_low

            return "RANGE", "NONE", protected_high, protected_low

        if previous_trend == "DOWN":
            if self._is_valid_break_down(current_close, last_low, atr):
                return "DOWN", "BOS_DOWN", protected_high, protected_low

            if self._is_valid_break_up(current_close, protected_high, atr):
                return "RANGE", "CHOCH_UP", protected_high, protected_low

            if bearish_structure:
                return "DOWN", "NONE", protected_high, protected_low

            if bullish_structure:
                return "RANGE", "NONE", protected_high, protected_low

            return "RANGE", "NONE", protected_high, protected_low

        # ------------------------------------------------------
        # RANGE / NEUTRAL LOGIC
        # ------------------------------------------------------
        if self._is_valid_break_up(current_close, last_high, atr):
            return "UP", "BOS_UP", protected_high, protected_low

        if self._is_valid_break_down(current_close, last_low, atr):
            return "DOWN", "BOS_DOWN", protected_high, protected_low

        if bullish_structure:
            return "UP", "NONE", protected_high, protected_low

        if bearish_structure:
            return "DOWN", "NONE", protected_high, protected_low

        return "RANGE", "NONE", protected_high, protected_low

    # ==========================================================
    # PROTECTED STRUCTURE
    # ==========================================================

    def _get_protected_low(self, trend: Trend) -> Optional[float]:
        if not self.swing_lows:
            return None

        lows = list(self.swing_lows)

        if trend == "UP":
            # Last higher low is the important invalidation point
            for swing in reversed(lows):
                if swing.label == "HL":
                    return swing.price

        return lows[-1].price

    def _get_protected_high(self, trend: Trend) -> Optional[float]:
        if not self.swing_highs:
            return None

        highs = list(self.swing_highs)

        if trend == "DOWN":
            # Last lower high is the important invalidation point
            for swing in reversed(highs):
                if swing.label == "LH":
                    return swing.price

        return highs[-1].price

    def _get_last_high(self) -> Optional[float]:
        return self.swing_highs[-1].price if self.swing_highs else None

    def _get_last_low(self) -> Optional[float]:
        return self.swing_lows[-1].price if self.swing_lows else None

    # ==========================================================
    # BREAK VALIDATION
    # ==========================================================

    def _is_valid_break_up(self, price: float, level: float, atr: Optional[float]) -> bool:
        threshold = self._break_threshold(atr)
        return price > level + threshold

    def _is_valid_break_down(self, price: float, level: float, atr: Optional[float]) -> bool:
        threshold = self._break_threshold(atr)
        return price < level - threshold

    def _break_threshold(self, atr: Optional[float]) -> float:
        if atr is None:
            return 0.0
        return atr * self.break_atr_multiplier

    # ==========================================================
    # TREND STRENGTH
    # ==========================================================

    def _determine_trend_strength(
        self, trend: Trend, structure_event: StructureEvent
    ) -> Optional[TrendStrength]:
        if trend == "RANGE":
            return None

        if len(self.swing_highs) < 2 or len(self.swing_lows) < 2:
            return "LOW"

        last_high = self.swing_highs[-1].price
        prev_high = self.swing_highs[-2].price

        last_low = self.swing_lows[-1].price
        prev_low = self.swing_lows[-2].price

        confirmations = 0

        if trend == "UP":
            if last_high > prev_high:
                confirmations += 1
            if last_low > prev_low:
                confirmations += 1
            if structure_event == "BOS_UP":
                confirmations += 1

        elif trend == "DOWN":
            if last_high < prev_high:
                confirmations += 1
            if last_low < prev_low:
                confirmations += 1
            if structure_event == "BOS_DOWN":
                confirmations += 1

        return "HIGH" if confirmations >= 2 else "LOW"

    # # ==========================================================
    # # ZONES
    # # ==========================================================

    # def _build_zones(self, atr: Optional[float]) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
    #     if atr is None or not self.swing_lows or not self.swing_highs:
    #         return None, None
    #
    #     buffer = atr * self.zone_atr_multiplier
    #     lookback_n = 3
    #
    #     recent_lows = [s.price for s in list(self.swing_lows)[-lookback_n:]]
    #     recent_highs = [s.price for s in list(self.swing_highs)[-lookback_n:]]
    #
    #     # current_price = self.candles[-1].close
    #     # recent_lows.append(current_price)
    #     # recent_highs.append(current_price)
    #
    #     median_low = statistics.median(recent_lows)
    #     median_high = statistics.median(recent_highs)
    #
    #     support = (median_low - buffer, median_low + buffer)
    #     resistance = (median_high - buffer, median_high + buffer)
    #
    #     return support, resistance

    # ==========================================================
    # ZONES
    # ==========================================================

    def _build_zones(
        self, atr: Optional[float]
    ) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
        if atr is None:
            return None, None

        trend = self.state.trend if self.state else "RANGE"

        support_candidates = self._collect_support_candidates(trend)
        resistance_candidates = self._collect_resistance_candidates(trend)

        support = self._build_clustered_zone(
            candidates=support_candidates,
            atr=atr,
            zone_type="support"
        )

        resistance = self._build_clustered_zone(
            candidates=resistance_candidates,
            atr=atr,
            zone_type="resistance"
        )

        return support, resistance

    def _collect_support_candidates(self, trend: Trend) -> List[Tuple[float, float]]:
        """
        Returns support candidates as:
            [(price, weight), ...]
        Higher weight = more important level.
        """
        candidates: List[Tuple[float, float]] = []
        lows = list(self.swing_lows)

        if not lows:
            return candidates

        recent_lows = lows[-8:]  # enough depth without being too stale

        for i, swing in enumerate(recent_lows):
            weight = 1.0

            # Structure-aware weighting
            if swing.label == "HL":
                weight += 2.5
            elif swing.label == "LL":
                weight += 1.0
            else:
                weight += 0.5

            # Recency weighting
            recency_boost = (i + 1) / len(recent_lows)
            weight += recency_boost * 1.5

            candidates.append((swing.price, weight))

        # Protected low is very important in bullish structure
        protected_low = self._get_protected_low(trend)
        if protected_low is not None:
            protected_weight = 4.5 if trend == "UP" else 2.5
            candidates.append((protected_low, protected_weight))

        # Last swing low still matters
        if self.swing_lows:
            candidates.append((self.swing_lows[-1].price, 2.0))

        return candidates

    def _collect_resistance_candidates(self, trend: Trend) -> List[Tuple[float, float]]:
        """
        Returns resistance candidates as:
            [(price, weight), ...]
        Higher weight = more important level.
        """
        candidates: List[Tuple[float, float]] = []
        highs = list(self.swing_highs)

        if not highs:
            return candidates

        recent_highs = highs[-8:]

        for i, swing in enumerate(recent_highs):
            weight = 1.0

            # Structure-aware weighting
            if swing.label == "LH":
                weight += 2.5
            elif swing.label == "HH":
                weight += 1.0
            else:
                weight += 0.5

            # Recency weighting
            recency_boost = (i + 1) / len(recent_highs)
            weight += recency_boost * 1.5

            candidates.append((swing.price, weight))

        # Protected high is very important in bearish structure
        protected_high = self._get_protected_high(trend)
        if protected_high is not None:
            protected_weight = 4.5 if trend == "DOWN" else 2.5
            candidates.append((protected_high, protected_weight))

        # Last swing high still matters
        if self.swing_highs:
            candidates.append((self.swing_highs[-1].price, 2.0))

        return candidates

    def _build_clustered_zone(
        self,
        candidates: List[Tuple[float, float]],
        atr: float,
        zone_type: str
    ) -> Optional[Tuple[float, float]]:
        """
        Build a zone by clustering nearby candidate levels and selecting
        the strongest cluster.

        candidates: [(price, weight), ...]
        """
        if not candidates:
            return None

        # How close levels must be to belong to same zone
        cluster_tolerance = max(atr * 0.35, 1e-9)

        clusters: List[dict] = []

        # Sort by price so nearby levels group naturally
        for price, weight in sorted(candidates, key=lambda x: x[0]):
            placed = False

            for cluster in clusters:
                if abs(price - cluster["center"]) <= cluster_tolerance:
                    cluster["levels"].append((price, weight))
                    cluster["total_weight"] += weight
                    cluster["center"] = (
                        sum(p * w for p, w in cluster["levels"]) /
                        sum(w for _, w in cluster["levels"])
                    )
                    placed = True
                    break

            if not placed:
                clusters.append({
                    "levels": [(price, weight)],
                    "total_weight": weight,
                    "center": price
                })

        if not clusters:
            return None

        # Prefer the strongest cluster, but with slight bias toward relevant side
        current_price = self.candles[-1].close if self.candles else None

        best_cluster = None
        best_score = float("-inf")

        for cluster in clusters:
            score = cluster["total_weight"]

            if current_price is not None:
                distance = abs(current_price - cluster["center"])
                score -= distance / max(atr, 1e-9)

                # Encourage support below/near price and resistance above/near price
                if zone_type == "support" and cluster["center"] <= current_price:
                    score += 0.75
                elif zone_type == "resistance" and cluster["center"] >= current_price:
                    score += 0.75

            if score > best_score:
                best_score = score
                best_cluster = cluster

        if best_cluster is None:
            return None

        level_prices = [p for p, _ in best_cluster["levels"]]
        weighted_center = best_cluster["center"]

        # Zone width based on both ATR and actual spread of cluster
        cluster_spread = max(level_prices) - min(level_prices) if len(level_prices) > 1 else 0.0
        half_width = max(atr * self.zone_atr_multiplier * 0.6, cluster_spread * 0.75)

        lower = weighted_center - half_width
        upper = weighted_center + half_width

        return (lower, upper)

    # ==========================================================
    # ZONE CHECK
    # ==========================================================

    def _is_in_zone(self, price: float, zone: Optional[Tuple[float, float]]) -> bool:
        if zone is None:
            return False

        lower, upper = zone
        return lower <= price <= upper