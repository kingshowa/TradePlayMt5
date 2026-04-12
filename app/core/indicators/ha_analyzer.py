from dataclasses import dataclass
from enum import Enum
from typing import List

from app.core.market.candle import Candle


class HACandleBias(str, Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    INDECISIVE = "INDECISIVE"


class HATrendStrength(str, Enum):
    STRONG = "STRONG"
    WEAK = "WEAK"
    INDECISIVE = "INDECISIVE"


class HATrendState(str, Enum):
    STRONG_BULLISH = "STRONG_BULLISH"
    WEAK_BULLISH = "WEAK_BULLISH"
    STRONG_BEARISH = "STRONG_BEARISH"
    WEAK_BEARISH = "WEAK_BEARISH"
    INDECISIVE = "INDECISIVE"


@dataclass
class HAAnalysisResult:
    state: HATrendState
    latest_bias: HACandleBias
    bullish_count: int
    bearish_count: int
    indecisive_count: int
    bullish_streak: int
    bearish_streak: int
    momentum_slowing: bool
    bullish_score: float
    bearish_score: float


class HeikinAshiAnalyzer:
    """
    Lightweight analyzer for Heikin Ashi candles.

    Responsibilities:
    - Classify candles as bullish / bearish / indecisive
    - Count direction dominance in a recent lookback
    - Measure bullish/bearish streaks
    - Estimate strength of recent trend
    - Detect momentum slowing

    It assumes input candles are already Heikin Ashi candles.
    """

    def __init__(
        self,
        indecision_body_ratio: float = 0.25,
        strong_body_ratio: float = 0.55,
        small_wick_ratio: float = 0.15,
    ):
        self.indecision_body_ratio = indecision_body_ratio
        self.strong_body_ratio = strong_body_ratio
        self.small_wick_ratio = small_wick_ratio

    # --------------------------------------------------
    # Basic candle measurements
    # --------------------------------------------------
    @staticmethod
    def body_size(candle: Candle) -> float:
        return abs(candle.close - candle.open)

    @staticmethod
    def range_size(candle: Candle) -> float:
        return candle.high - candle.low

    @staticmethod
    def upper_wick(candle: Candle) -> float:
        return candle.high - max(candle.open, candle.close)

    @staticmethod
    def lower_wick(candle: Candle) -> float:
        return min(candle.open, candle.close) - candle.low

    def body_ratio(self, candle: Candle) -> float:
        candle_range = self.range_size(candle)
        if candle_range <= 0:
            return 0.0
        return self.body_size(candle) / candle_range

    def upper_wick_ratio(self, candle: Candle) -> float:
        candle_range = self.range_size(candle)
        if candle_range <= 0:
            return 0.0
        return self.upper_wick(candle) / candle_range

    def lower_wick_ratio(self, candle: Candle) -> float:
        candle_range = self.range_size(candle)
        if candle_range <= 0:
            return 0.0
        return self.lower_wick(candle) / candle_range

    # --------------------------------------------------
    # Single candle classification
    # --------------------------------------------------
    def is_bullish(self, candle: Candle) -> bool:
        return candle.close > candle.open

    def is_bearish(self, candle: Candle) -> bool:
        return candle.close < candle.open

    def is_indecisive(self, candle: Candle) -> bool:
        if self.range_size(candle) <= 0:
            return True

        if self.body_ratio(candle) <= self.indecision_body_ratio:
            return True

        # Both wicks visible + modest body often signals indecision
        if (
            self.upper_wick_ratio(candle) > self.small_wick_ratio
            and self.lower_wick_ratio(candle) > self.small_wick_ratio
            and self.body_ratio(candle) < self.strong_body_ratio
        ):
            return True

        return False

    def candle_bias(self, candle: Candle) -> HACandleBias:
        if self.is_indecisive(candle):
            return HACandleBias.INDECISIVE
        if self.is_bullish(candle):
            return HACandleBias.BULLISH
        if self.is_bearish(candle):
            return HACandleBias.BEARISH
        return HACandleBias.INDECISIVE

    def is_strong_bullish_candle(self, candle: Candle) -> bool:
        return (
            self.is_bullish(candle)
            and not self.is_indecisive(candle)
            and self.body_ratio(candle) >= self.strong_body_ratio
            and self.lower_wick_ratio(candle) <= self.small_wick_ratio
        )

    def is_strong_bearish_candle(self, candle: Candle) -> bool:
        return (
            self.is_bearish(candle)
            and not self.is_indecisive(candle)
            and self.body_ratio(candle) >= self.strong_body_ratio
            and self.upper_wick_ratio(candle) <= self.small_wick_ratio
        )

    # --------------------------------------------------
    # Window helpers
    # --------------------------------------------------
    @staticmethod
    def _last_n(candles: List[Candle], lookback: int) -> List[Candle]:
        if lookback <= 0:
            raise ValueError("lookback must be greater than 0")
        if not candles:
            raise ValueError("candles cannot be empty")
        return candles[-lookback:] if len(candles) >= lookback else candles[:]

    def count_bullish(self, candles: List[Candle], lookback: int = 5) -> int:
        window = self._last_n(candles, lookback)
        return sum(1 for c in window if self.candle_bias(c) == HACandleBias.BULLISH)

    def count_bearish(self, candles: List[Candle], lookback: int = 5) -> int:
        window = self._last_n(candles, lookback)
        return sum(1 for c in window if self.candle_bias(c) == HACandleBias.BEARISH)

    def count_indecisive(self, candles: List[Candle], lookback: int = 5) -> int:
        window = self._last_n(candles, lookback)
        return sum(1 for c in window if self.candle_bias(c) == HACandleBias.INDECISIVE)

    def bullish_streak(self, candles: List[Candle]) -> int:
        streak = 0
        for candle in reversed(candles):
            if self.candle_bias(candle) == HACandleBias.BULLISH:
                streak += 1
            else:
                break
        return streak

    def bearish_streak(self, candles: List[Candle]) -> int:
        streak = 0
        for candle in reversed(candles):
            if self.candle_bias(candle) == HACandleBias.BEARISH:
                streak += 1
            else:
                break
        return streak

    # --------------------------------------------------
    # Strength / momentum
    # --------------------------------------------------
    def bullish_score(self, candles: List[Candle], lookback: int = 5) -> float:
        window = self._last_n(candles, lookback)
        score = 0.0

        for candle in window:
            bias = self.candle_bias(candle)

            if bias == HACandleBias.BULLISH:
                score += 2.0
                if self.is_strong_bullish_candle(candle):
                    score += 1.0
            elif bias == HACandleBias.BEARISH:
                score -= 2.0
            else:
                score -= 0.5

        streak = self.bullish_streak(window)
        if streak >= 3:
            score += 2.0
        elif streak == 2:
            score += 1.0

        return score

    def bearish_score(self, candles: List[Candle], lookback: int = 5) -> float:
        window = self._last_n(candles, lookback)
        score = 0.0

        for candle in window:
            bias = self.candle_bias(candle)

            if bias == HACandleBias.BEARISH:
                score += 2.0
                if self.is_strong_bearish_candle(candle):
                    score += 1.0
            elif bias == HACandleBias.BULLISH:
                score -= 2.0
            else:
                score -= 0.5

        streak = self.bearish_streak(window)
        if streak >= 3:
            score += 2.0
        elif streak == 2:
            score += 1.0

        return score

    def momentum_slowing(self, candles: List[Candle], lookback: int = 6) -> bool:
        window = self._last_n(candles, lookback)

        if len(window) < 4:
            return False

        mid = len(window) // 2
        older = window[:mid]
        recent = window[mid:]

        older_avg_body = sum(self.body_size(c) for c in older) / len(older)
        recent_avg_body = sum(self.body_size(c) for c in recent) / len(recent)

        if older_avg_body <= 0:
            return False

        return bool(recent_avg_body < older_avg_body * 0.8)

    # --------------------------------------------------
    # Final interpretation
    # --------------------------------------------------
    def trend_strength(self, candles: List[Candle], lookback: int = 5) -> HATrendStrength:
        b_score = self.bullish_score(candles, lookback)
        s_score = self.bearish_score(candles, lookback)
        indecisive_count = self.count_indecisive(candles, lookback)

        if indecisive_count >= max(2, lookback // 2):
            return HATrendStrength.INDECISIVE

        if max(b_score, s_score) >= 6:
            return HATrendStrength.STRONG

        if max(b_score, s_score) >= 2:
            return HATrendStrength.WEAK

        return HATrendStrength.INDECISIVE

    def trend_state(self, candles: List[Candle], lookback: int = 5) -> HATrendState:
        window = self._last_n(candles, lookback)

        latest_bias = self.candle_bias(window[-1])
        strength = self.trend_strength(window, lookback=min(lookback, len(window)))

        b_score = self.bullish_score(window, lookback=min(lookback, len(window)))
        s_score = self.bearish_score(window, lookback=min(lookback, len(window)))

        if latest_bias == HACandleBias.BULLISH and b_score > s_score:
            if strength == HATrendStrength.STRONG:
                return HATrendState.STRONG_BULLISH
            if strength == HATrendStrength.WEAK:
                return HATrendState.WEAK_BULLISH

        if latest_bias == HACandleBias.BEARISH and s_score > b_score:
            if strength == HATrendStrength.STRONG:
                return HATrendState.STRONG_BEARISH
            if strength == HATrendStrength.WEAK:
                return HATrendState.WEAK_BEARISH

        return HATrendState.INDECISIVE

    def analyze(self, candles: List[Candle], lookback: int = 5) -> HAAnalysisResult:
        window = self._last_n(candles, lookback)

        return HAAnalysisResult(
            state=self.trend_state(window, lookback=min(lookback, len(window))),
            latest_bias=self.candle_bias(window[-1]),
            bullish_count=self.count_bullish(window, lookback=min(lookback, len(window))),
            bearish_count=self.count_bearish(window, lookback=min(lookback, len(window))),
            indecisive_count=self.count_indecisive(window, lookback=min(lookback, len(window))),
            bullish_streak=self.bullish_streak(window),
            bearish_streak=self.bearish_streak(window),
            momentum_slowing=self.momentum_slowing(window, lookback=min(6, len(window))),
            bullish_score=self.bullish_score(window, lookback=min(lookback, len(window))),
            bearish_score=self.bearish_score(window, lookback=min(lookback, len(window))),
        )