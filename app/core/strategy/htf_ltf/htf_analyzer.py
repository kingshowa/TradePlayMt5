from dataclasses import dataclass
from enum import Enum
from typing import Optional

from app.core.indicators.ema import EMAIndicator
from app.core.indicators.ha import (
    CandleCharacter,
    HeikinAshiIndicator,
    Trend,
    TrendState,
)
from app.core.market.candle import Candle
from app.core.market.market_series import MarketSeries


# ──────────────────────────────────────────────────────────────────────
# Regime enum  (imported by the strategy layer)
# ──────────────────────────────────────────────────────────────────────

class Regime(Enum):
    LONG_ONLY  = "LONG_ONLY"
    SHORT_ONLY = "SHORT_ONLY"
    NO_TRADE   = "NO_TRADE"


# ──────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────

@dataclass
class HTFConfig:
    """
    ha_min_bars         : Minimum consecutive same-colour HA bars before a
                          regime is declared.  Default 2.
    ema_period          : Period for the 5m confirmation EMA (source=close).
    ema_enabled         : Toggle EMA confirmation on/off at runtime.
                          When False, HA alone decides the regime.
    ema_slope_threshold : Minimum absolute slope required to count as trending.
                          0.0 means any directional move qualifies.
    """
    ha_min_bars:         int   = 2
    ema_period:          int   = 20
    ema_enabled:         bool  = True
    ema_slope_threshold: float = 0.0


# ──────────────────────────────────────────────────────────────────────
# HTF Analyzer
# ──────────────────────────────────────────────────────────────────────

class HTFAnalyzer:
    """
    Determines the trading regime from a higher-timeframe (5m) feed.

    Regime rules
    ------------
    LONG_ONLY
        HA state is BULLISH for ≥ ha_min_bars consecutive bars
        AND the latest HA candle is TRENDING (no lower wick)
        AND, if enabled, 5m HA close is above EMA with a positive slope.

    SHORT_ONLY
        Exact mirror: BEARISH, TRENDING (no upper wick), below EMA.

    NO_TRADE
        Any of: HA is RANGING, not enough consecutive bars, or the optional
        EMA filter disagrees with HA direction.

    Usage
    -----
        htf = HTFAnalyzer(HTFConfig(ema_enabled=True))
        htf.calculate(htf_series)          # seed from history
        regime = htf.update(new_5m_candle) # call on every 5m candle
    """

    def __init__(self, config: HTFConfig = None):
        self.config = config or HTFConfig()

        self._ha  = HeikinAshiIndicator()
        self._ema: Optional[EMAIndicator] = (
            EMAIndicator(period=self.config.ema_period, source="close")
            if self.config.ema_enabled
            else None
        )
        self._regime: Regime = Regime.NO_TRADE
        self.current_candle: Optional[Candle] = None

    # ── Initialization ────────────────────────────────────────────────

    def calculate(self, series: MarketSeries) -> Regime:
        """Seed indicators from a historical MarketSeries and return the regime."""
        self._ha.calculate(series)
        if self._ema:
            self._ema.calculate(series)
        self.current_candle = series.last()
        self._regime = self._evaluate()
        return self._regime

    # ── Live update ───────────────────────────────────────────────────

    def update(self, candle: Candle) -> Regime:
        """
        Call on every new or updating 5m candle.
        Same-timestamp candles are handled by HeikinAshiIndicator internally.
        Returns the current Regime after the update.
        """
        self._ha.update(candle)
        if self.config.ema_enabled and self._ema:
            self._ema.update(candle)
        self._regime = self._evaluate()
        self.current_candle = candle
        return self._regime

    # ── Public properties ─────────────────────────────────────────────

    @property
    def regime(self) -> Regime:
        return self._regime

    @property
    def character(self) -> Optional[CandleCharacter]:
        """TRENDING or RANGING for the latest HA candle."""
        return self._ha.current_character

    @property
    def trend(self) -> Optional[Trend]:
        """The active HA Trend object (candle_count, growth_pct, momentum…)."""
        return self._ha.current_trend

    @property
    def ema_value(self) -> Optional[float]:
        return self._ema.current_value if self._ema else None

    @property
    def ema_slope(self) -> Optional[float]:
        if self._ema and self._ema.current_value is not None:
            return self._ema.slope()
        return None

    # ── EMA toggle ────────────────────────────────────────────────────

    def enable_ema(self) -> None:
        """
        Engage the 5m EMA confirmation filter.
        If the EMAIndicator does not exist yet (created with ema_enabled=False),
        you must call calculate() again after enabling so it can be seeded.
        """
        self.config.ema_enabled = True
        if self._ema is None:
            self._ema = EMAIndicator(period=self.config.ema_period, source="close")

    def disable_ema(self) -> None:
        """Disengage the 5m EMA filter. HA alone governs the regime."""
        self.config.ema_enabled = False

    # ── Debug ─────────────────────────────────────────────────────────

    def snapshot(self) -> dict:
        trend = self._ha.current_trend
        return {
            "regime":       self._regime.value,
            "ha_character": self._ha.current_character.value if self._ha.current_character else None,
            "ha_trend":     trend.as_dict() if trend else None,
            "ema_enabled":  self.config.ema_enabled,
            "ema":          _r(self.ema_value),
            "ema_slope":    _r(self.ema_slope),
        }

    # ── Internal ──────────────────────────────────────────────────────

    def _evaluate(self) -> Regime:
        trend     = self._ha.current_trend
        character = self._ha.current_character

        if trend is None or character is None:
            return Regime.NO_TRADE

        if character == CandleCharacter.RANGING:
            return Regime.NO_TRADE

        if trend.candle_count < self.config.ha_min_bars:
            return Regime.NO_TRADE

        if trend.state == TrendState.BULLISH:
            return Regime.LONG_ONLY if self._ema_confirms_long() else Regime.NO_TRADE

        if trend.state == TrendState.BEARISH:
            return Regime.SHORT_ONLY if self._ema_confirms_short() else Regime.NO_TRADE

        return Regime.NO_TRADE

    def _ema_confirms_long(self) -> bool:
        """Pass-through when EMA is disabled or not yet seeded."""
        if not self.config.ema_enabled or self._ema is None:
            return True
        if self._ema.current_value is None or self._ha.current_value is None:
            return False
        return self._ema.is_uptrend(
            self.current_candle.close,
            self.config.ema_slope_threshold
        )

    def _ema_confirms_short(self) -> bool:
        if not self.config.ema_enabled or self._ema is None:
            return True
        if self._ema.current_value is None or self._ha.current_value is None:
            return False
        return self._ema.is_downtrend(
            self.current_candle.close,
            self.config.ema_slope_threshold
        )


def _r(v: Optional[float], d: int = 5) -> Optional[float]:
    return round(v, d) if v is not None else None