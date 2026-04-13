from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, Optional, List

from app.core.indicators.base_indicator import BaseIndicator
from app.core.market.candle import Candle
from app.core.market.market_series import MarketSeries


class TrendState(Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"


class MomentumState(Enum):
    GAINING = "GAINING"
    NEUTRAL = "NEUTRAL"
    LOSING = "LOSING"


class CandleCharacter(Enum):
    """
    Orthogonal to color — describes the structural quality of an HA candle.

    TRENDING:
        Bullish candle with no lower wick  (low == open):
            clean body, no downward rejection, full conviction upward.
        Bearish candle with no upper wick  (high == open):
            clean body, no upward rejection, full conviction downward.

    RANGING:
        Bullish candle with a lower wick   (low < open):
            price dipped below the body during the bar — indecision despite
            a green close.
        Bearish candle with an upper wick  (high > open):
            price pushed above the body during the bar — indecision despite
            a red close.
    """
    TRENDING = "TRENDING"
    RANGING = "RANGING"


@dataclass
class Trend:
    """
    Tracks one continuous Heikin Ashi trend run.

    Growth % convention (always positive, direction implied by TrendState):
      - BULLISH: (close - trend_start_open) / trend_start_open * 100
      - BEARISH: (trend_start_open - close) / trend_start_open * 100

    Momentum logic:
      - Compares the current candle's directional growth against a stable baseline.
      - Baseline is built from confirmed candles only.
      - At a trend flip, the new trend is seeded using the previous opposite
        trend's ending strength so the first reversal candle is not unfairly
        treated as neutral.
    """

    state: TrendState
    candle_count: int = 0
    growth_pct: float = 0.0
    last_candle_growth_pct: float = 0.0
    avg_candle_growth_pct: float = 0.0
    momentum: MomentumState = MomentumState.NEUTRAL

    # Internal state
    _start_open: float = field(default=0.0, repr=False, compare=False)

    # Confirmed candle contributions only
    _confirmed_cumulative_growth: float = field(default=0.0, repr=False, compare=False)
    _confirmed_count: int = field(default=0, repr=False, compare=False)

    # Seed baseline from previous opposite trend when a new trend begins
    _seed_growth_pct: float = field(default=0.0, repr=False, compare=False)
    _has_seed: bool = field(default=False, repr=False, compare=False)

    # If |last - avg| < threshold => NEUTRAL
    MOMENTUM_THRESHOLD: float = field(default=0.02, repr=False, compare=False)

    def _directional_growth_from_candle(self, ha: Candle) -> float:
        """
        Growth contribution of this candle within the direction of the trend.
        Always positive or zero.
        """
        if self.state == TrendState.BULLISH:
            return max(0.0, (ha.close - ha.open) / ha.open * 100)
        return max(0.0, (ha.open - ha.close) / ha.open * 100)

    def _total_growth_from_candle(self, ha: Candle) -> float:
        """
        Total growth of the whole trend from its starting HA open.
        Always positive or zero.
        """
        if self.state == TrendState.BULLISH:
            return max(0.0, (ha.close - self._start_open) / self._start_open * 100)
        return max(0.0, (self._start_open - ha.close) / self._start_open * 100)

    def _baseline_average(self) -> float:
        """
        Momentum baseline priority:
          1. average of confirmed candles in this trend
          2. reversal seed from previous opposite trend
          3. current candle growth itself (fallback)
        """
        if self._confirmed_count > 0:
            return self._confirmed_cumulative_growth / self._confirmed_count
        if self._has_seed:
            return self._seed_growth_pct
        return self.last_candle_growth_pct

    def begin_first_candle(self, ha: Candle) -> None:
        """
        Start a new trend using the first candle.
        Important: this first candle is NOT immediately committed into the
        confirmed baseline. That keeps momentum comparison meaningful on live
        updates and at fresh reversals.
        """
        self.candle_count = 1
        self.growth_pct = self._total_growth_from_candle(ha)
        self.last_candle_growth_pct = self._directional_growth_from_candle(ha)
        self.avg_candle_growth_pct = self._baseline_average()
        self._recalculate_momentum()

    def extend_with_new_candle(self, ha: Candle) -> None:
        """
        Advance the trend with a brand-new confirmed candle of the same direction.

        Before evaluating the new candle, the previous last candle becomes
        confirmed and is added into the baseline stats.
        """
        if self.candle_count > 0:
            self._confirmed_cumulative_growth += self.last_candle_growth_pct
            self._confirmed_count += 1

        self.candle_count += 1
        self.growth_pct = self._total_growth_from_candle(ha)
        self.last_candle_growth_pct = self._directional_growth_from_candle(ha)
        self.avg_candle_growth_pct = self._baseline_average()
        self._recalculate_momentum()

    def recalculate_last(self, ha: Candle) -> None:
        """
        Live tick update for the currently forming candle.

        Baseline remains unchanged because only confirmed candles contribute
        to the average.
        """
        self.growth_pct = self._total_growth_from_candle(ha)
        self.last_candle_growth_pct = self._directional_growth_from_candle(ha)
        self.avg_candle_growth_pct = self._baseline_average()
        self._recalculate_momentum()

    def finalize(self) -> None:
        """
        Seal the trend before moving it into history.

        This includes the final candle in the confirmed average so stored
        history reflects the fully completed run.
        """
        if self.candle_count > 0:
            self._confirmed_cumulative_growth += self.last_candle_growth_pct
            self._confirmed_count += 1
            self.avg_candle_growth_pct = self._baseline_average()

    def _recalculate_momentum(self) -> None:
        diff = self.last_candle_growth_pct - self.avg_candle_growth_pct

        if diff > self.MOMENTUM_THRESHOLD:
            self.momentum = MomentumState.GAINING
        elif diff < -self.MOMENTUM_THRESHOLD:
            self.momentum = MomentumState.LOSING
        else:
            self.momentum = MomentumState.NEUTRAL

    def as_dict(self) -> dict:
        return {
            "state": self.state.value,
            "candle_count": self.candle_count,
            "growth_pct": round(self.growth_pct, 4),
            "last_candle_growth_pct": round(self.last_candle_growth_pct, 4),
            "avg_candle_growth_pct": round(self.avg_candle_growth_pct, 4),
            "momentum": self.momentum.value,
        }


class HeikinAshiIndicator(BaseIndicator):
    """
    Heikin Ashi indicator with live trend tracking.

    Features:
      - Full batch calculation from MarketSeries
      - Incremental live updates
      - Same-timestamp candle replacement support
      - Trend history storage
      - Momentum continuity across trend flips
    """

    def __init__(self, max_candles: int = 500, max_trends: int = 100):
        super().__init__(period=1)
        self.max_candles = max_candles
        self._ha_candles: Deque[Candle] = deque(maxlen=max_candles)

        self.current_value: Optional[Candle] = None
        self.previous_value: Optional[Candle] = None

        self.max_trends = max_trends
        self._trend_history: Deque[Trend] = deque(maxlen=max_trends)
        self._current_trend: Optional[Trend] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def current_trend(self) -> Optional[Trend]:
        """Currently active trend, updated live."""
        return self._current_trend

    @property
    def trend_history(self) -> List[Trend]:
        """Completed trends only, oldest first."""
        return list(self._trend_history)

    @property
    def current_character(self) -> Optional[CandleCharacter]:
        """Character of the latest HA candle, or None if no candles yet."""
        if self.current_value is None:
            return None
        return self.candle_character(self.current_value)

    @property
    def previous_character(self) -> Optional[CandleCharacter]:
        """Character of the second-to-last HA candle, or None if unavailable."""
        if self.previous_value is None:
            return None
        return self.candle_character(self.previous_value)

    @staticmethod
    def candle_character(ha: Candle) -> CandleCharacter:
        """
        Classify any HA candle as TRENDING or RANGING.

        Bullish candle (close > open):
            RANGING  if low < open  (lower wick present — downward rejection)
            TRENDING if low == open (no lower wick — full upward conviction)

        Bearish candle (close <= open):
            RANGING  if high > open (upper wick present — upward rejection)
            TRENDING if high == open (no upper wick — full downward conviction)
        """
        if ha.close > ha.open:
            return CandleCharacter.RANGING if ha.low < ha.open else CandleCharacter.TRENDING
        return CandleCharacter.RANGING if ha.high > ha.open else CandleCharacter.TRENDING

    def calculate(self, series: MarketSeries) -> Candle:
        if len(series) == 0:
            raise ValueError("MarketSeries cannot be empty")

        self.reset()
        raw_candles = series.candles()

        prev_ha_open = None
        prev_ha_close = None

        for raw in raw_candles:
            ha = self._build_ha_candle(raw, prev_ha_open, prev_ha_close)
            self._ha_candles.append(ha)

            prev_ha_open = ha.open
            prev_ha_close = ha.close

            self._advance_trend(ha)

        self._sync_state()
        return self.current_value

    def update(self, new_candle: Candle) -> Candle:
        """
        Incremental update.

        - Same timestamp: replace last HA candle and recalc live trend values
        - New timestamp : append HA candle and advance trend state machine
        """
        if not self._ha_candles:
            ha = self._build_ha_candle(new_candle, None, None)
            self._ha_candles.append(ha)
            self._advance_trend(ha)
            self._sync_state()
            return ha

        last_ha = self._ha_candles[-1]

        if last_ha.time == new_candle.time:
            # Live tick update of the same candle
            self._ha_candles.pop()
            prev_ha = self._ha_candles[-1] if self._ha_candles else None

            ha = self._build_ha_candle(
                new_candle,
                prev_ha.open if prev_ha else None,
                prev_ha.close if prev_ha else None,
            )
            self._ha_candles.append(ha)

            if self._current_trend:
                self._current_trend.recalculate_last(ha)
        else:
            # New confirmed candle
            ha = self._build_ha_candle(new_candle, last_ha.open, last_ha.close)
            self._ha_candles.append(ha)
            self._advance_trend(ha)

        self._sync_state()
        return ha

    def values(self) -> List[Candle]:
        return list(self._ha_candles)

    def last(self, n: int = 1):
        if not self._ha_candles:
            raise ValueError("No Heikin Ashi candles available")

        if n == 1:
            return self._ha_candles[-1]

        if n > len(self._ha_candles):
            raise ValueError(
                f"Requested {n} candles, but only {len(self._ha_candles)} available"
            )

        return list(self._ha_candles)[-n:]

    def as_market_series(self) -> MarketSeries:
        return MarketSeries(self.values())

    def reset(self):
        self._ha_candles.clear()
        self.current_value = None
        self.previous_value = None
        self._current_trend = None
        self._trend_history.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _candle_state(ha: Candle) -> TrendState:
        return TrendState.BULLISH if ha.close > ha.open else TrendState.BEARISH

    def _build_seed_from_previous_trend(self, previous_trend: Trend) -> float:
        """
        Seed baseline for a new trend using the previous opposite trend.

        Weighted blend:
          - 70% previous trend's final candle strength
          - 30% previous trend's average strength

        This gives the new reversal candle a meaningful comparison baseline.
        """
        return (
            previous_trend.last_candle_growth_pct * 0.7
            + previous_trend.avg_candle_growth_pct * 0.3
        )

    def _advance_trend(self, ha: Candle) -> None:
        """
        Advance the trend state machine by one confirmed candle.

        Cases:
          - First candle ever
          - Same-direction continuation
          - Direction flip
        """
        incoming_state = self._candle_state(ha)

        if self._current_trend is None:
            trend = Trend(
                state=incoming_state,
                _start_open=ha.open,
            )
            trend.begin_first_candle(ha)
            self._current_trend = trend
            return

        if incoming_state == self._current_trend.state:
            self._current_trend.extend_with_new_candle(ha)
            return

        # Direction flip -> finalize old trend and start a new seeded one
        previous_trend = self._current_trend
        previous_trend.finalize()
        self._trend_history.append(previous_trend)

        seed_growth = self._build_seed_from_previous_trend(previous_trend)

        trend = Trend(
            state=incoming_state,
            _start_open=ha.open,
            _seed_growth_pct=seed_growth,
            _has_seed=True,
        )
        trend.begin_first_candle(ha)
        self._current_trend = trend

    def _build_ha_candle(
        self,
        raw_candle: Candle,
        prev_ha_open: Optional[float],
        prev_ha_close: Optional[float],
    ) -> Candle:
        ha_close = (
            raw_candle.open
            + raw_candle.high
            + raw_candle.low
            + raw_candle.close
        ) / 4

        ha_open = (
            (raw_candle.open + raw_candle.close) / 2
            if prev_ha_open is None or prev_ha_close is None
            else (prev_ha_open + prev_ha_close) / 2
        )

        ha_high = max(raw_candle.high, ha_open, ha_close)
        ha_low = min(raw_candle.low, ha_open, ha_close)

        return Candle(
            time=raw_candle.time,
            open=ha_open,
            high=ha_high,
            low=ha_low,
            close=ha_close,
            volume=getattr(raw_candle, "volume", 0),
        )

    def _sync_state(self):
        self.current_value = self._ha_candles[-1] if self._ha_candles else None
        self.previous_value = (
            self._ha_candles[-2] if len(self._ha_candles) > 1 else None
        )