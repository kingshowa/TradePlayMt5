from typing import Optional

from app.core.indicators.ha import CandleCharacter
from app.core.market.candle import Candle
from app.core.market.market_series import MarketSeries
from app.core.strategy.trade_signal import StrategySignal

from .htf_analyzer import HTFAnalyzer, HTFConfig, Regime
from .ltf_analyzer import LTFAnalyzer, LTFConfig


# ──────────────────────────────────────────────────────────────────────
# Strategy
# ──────────────────────────────────────────────────────────────────────

class HTFLTFStrategy:
    """
    5m HTF + 1m LTF scalper.

    Responsibility boundary
    -----------------------
    HTFAnalyzer  — owns the 5m feed, decides the regime and HA character.
    LTFAnalyzer  — owns the 1m feed, evaluates entry and exit conditions.
    HTFLTFStrategy — coordinates the two, tracks open position state,
                     and emits StrategySignal objects.

    Signal flow
    -----------
    on_htf_candle(candle)
        → updates HTFAnalyzer, refreshes regime.

    on_ltf_candle(candle)
        → updates LTFAnalyzer.
        → if a position is open: checks force-close conditions first.
        → if no position: checks entry conditions against current regime.
        → returns StrategySignal or None.

    StrategySignal mapping
    ----------------------
    Long entry   → signal="BUY",   sl=Low EMA,  tp=entry + ATR×mult
    Short entry  → signal="SELL",  sl=High EMA, tp=entry - ATR×mult
    Force close  → signal="CLOSE", sl/tp both = candle.close

    Usage
    -----
        htf = HTFAnalyzer(HTFConfig(ema_enabled=True))
        ltf = LTFAnalyzer(LTFConfig(ema_period=9, rsi_period=14))

        strategy = HTFLTFStrategy(htf, ltf)
        strategy.initialize(htf_series, ltf_series)

        # 5m feed:
        strategy.on_htf_candle(five_min_candle)

        # 1m feed:
        signal = strategy.on_ltf_candle(one_min_candle)
        if signal:
            print(signal.signal, signal.sl, signal.tp)

    Toggling the 5m EMA mid-session
    --------------------------------
        strategy.htf.enable_ema()   # re-engage; regime will tighten
        strategy.htf.disable_ema()  # disengage; HA alone governs
    """

    def __init__(self, htf: HTFAnalyzer, ltf: LTFAnalyzer):
        self.htf = htf
        self.ltf = ltf
        self._open_signal: Optional[str] = None   # "BUY" | "SELL" | None

    # ── Initialization ────────────────────────────────────────────────

    def initialize(self, htf_series: MarketSeries, ltf_series: MarketSeries) -> None:
        """
        Seed both analyzers from historical data.
        Must be called once before any on_htf_candle / on_ltf_candle calls.

        Args:
            htf_series : 5m MarketSeries.  Needs at least ema_period + ha_min_bars bars.
            ltf_series : 1m MarketSeries.  Needs at least max(ema_period, rsi_period, atr_period) + 1 bars.
        """
        self.htf.calculate(htf_series)
        self.ltf.calculate(ltf_series)

    # ── Live hooks ────────────────────────────────────────────────────

    def on_htf_candle(self, candle: Candle) -> Regime:
        """
        Call on every new or updating 5m candle.
        Returns the current Regime after the update.
        """
        return self.htf.update(candle)

    def on_ltf_candle(self, candle: Candle) -> Optional[StrategySignal]:
        """
        Call on every new or updating 1m candle.

        Evaluation order (first match wins):
          1. Force-close if a position is open and exit conditions are met.
          2. Entry if no position is open and regime + LTF conditions align.
          3. None otherwise.

        Returns:
            StrategySignal on actionable events, None otherwise.
        """
        self.ltf.update(candle)

        htf_ranging = (self.htf.character == CandleCharacter.RANGING)

        # ── Force-close (always evaluated before new entries) ──────────
        if self._open_signal == "BUY":
            if self.ltf.should_force_close_long(candle, htf_ranging=htf_ranging):
                self._open_signal = None
                return self._close_signal(candle)

        elif self._open_signal == "SELL":
            if self.ltf.should_force_close_short(candle, htf_ranging=htf_ranging):
                self._open_signal = None
                return self._close_signal(candle)

        # ── No new entries while a position is open ────────────────────
        if self._open_signal is not None:
            return None

        # ── Entry ──────────────────────────────────────────────────────
        if self.htf.regime == Regime.LONG_ONLY and self.ltf.long_entry_ok(candle):
            self._open_signal = "BUY"
            return self._entry_signal("BUY", candle)

        if self.htf.regime == Regime.SHORT_ONLY and self.ltf.short_entry_ok(candle):
            self._open_signal = "SELL"
            return self._entry_signal("SELL", candle)

        return None

    # ── Properties ────────────────────────────────────────────────────

    @property
    def regime(self) -> Regime:
        return self.htf.regime

    @property
    def open_signal(self) -> Optional[str]:
        """Currently open position direction ('BUY', 'SELL', or None)."""
        return self._open_signal

    def snapshot(self) -> dict:
        return {
            "open_signal": self._open_signal,
            "htf":         self.htf.snapshot(),
            "ltf":         self.ltf.snapshot(),
        }

    # ── Signal builders ───────────────────────────────────────────────

    def _entry_signal(self, direction: str, candle: Candle) -> StrategySignal:
        entry = candle.close

        if direction == "BUY":
            sl = self.ltf.sl_long
            tp = self.ltf.tp_long(entry)
            reason  = self._long_reason(candle)
            pattern = self._long_pattern()
        else:
            sl = self.ltf.sl_short
            tp = self.ltf.tp_short(entry)
            reason  = self._short_reason(candle)
            pattern = self._short_pattern()

        return StrategySignal(
            signal=direction,
            strategy_type="TRENDING",
            reason=reason,
            pattern_name=pattern,
            atr=self.ltf.atr,
            sl=sl,
            tp=tp,
            candle=candle,
        )

    def _close_signal(self, candle: Candle) -> StrategySignal:
        reason = self._close_reason(candle)

        return StrategySignal(
            signal="CLOSE",
            strategy_type="TRENDING",
            reason=reason,
            pattern_name="force_close",
            atr=self.ltf.atr,
            sl=candle.close,
            tp=candle.close,
            candle=candle,
        )

    # ── Reason string builders ─────────────────────────────────────────

    def _long_reason(self, candle: Candle) -> str:
        return (
            f"regime={self.htf.regime.value} | "
            f"ha_bars={self.htf.trend.candle_count if self.htf.trend else '?'} | "
            f"ema_high={_f(self.ltf.ema_high)} "
            f"ema_low={_f(self.ltf.ema_low)} | "
            f"candle_close={_f(candle.close)}"
        )

    def _short_reason(self, candle: Candle) -> str:
        return (
            f"regime={self.htf.regime.value} | "
            f"ha_bars={self.htf.trend.candle_count if self.htf.trend else '?'} | "
            f"ema_high={_f(self.ltf.ema_high)} "
            f"ema_low={_f(self.ltf.ema_low)} | "
            f"candle_close={_f(candle.close)}"
        )

    def _long_pattern(self) -> str:
        return (
            f"rsi={_f(self.ltf.rsi, 2)} rising=True | "
            f"pullback_reclaim_high_ema"
        )

    def _short_pattern(self) -> str:
        return (
            f"rsi={_f(self.ltf.rsi, 2)} falling=True | "
            f"rally_rejection_low_ema"
        )

    def _close_reason(self, candle: Candle) -> str:
        parts = []

        ema_low  = self.ltf.ema_low
        ema_high = self.ltf.ema_high

        if ema_low is not None and candle.close < ema_low:
            parts.append("structure_break_below_low_ema")
        if ema_high is not None and candle.close > ema_high:
            parts.append("structure_break_above_high_ema")
        if self.htf.character == CandleCharacter.RANGING:
            parts.append("htf_ha_ranging")

        # RSI direction without re-evaluating (already fired in should_force_close)
        rsi = self.ltf.rsi
        if rsi is not None:
            if rsi < 50:
                parts.append("rsi_below_50")
            elif rsi > 50:
                parts.append("rsi_above_50")

        return " | ".join(parts) if parts else "force_close"


# ──────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────

def _f(v: Optional[float], d: int = 5) -> str:
    return f"{v:.{d}f}" if v is not None else "n/a"