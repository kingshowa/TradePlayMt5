from collections import deque
from typing import Any, Deque, Optional

from app.core.indicators.adx import ADXIndicator
from app.core.indicators.atr import ATRIndicator
from app.core.indicators.ema import EMAIndicator
from app.core.indicators.psar import PSARIndicator
from app.core.market.candle import Candle
from app.core.market.market_series import MarketSeries
from app.core.strategy.trade_signal import StrategySignal


class ParabolicSarStrategy:
    """
    Parabolic SAR Strategy.

    Responsibilities:
    - Generate BUY/SELL signals only on PSAR flips.
    - Optionally filter entries using EMA trend.
    - Optionally filter entries using ADX strength and DI bias.
    - Optionally close trades on opposite PSAR flip.
    - Pass ATR and raw PSAR dot to the risk manager/backtest.

    Important:
    - This strategy does NOT calculate final SL, TP, or position size.
    - The risk manager/backtest decides between PSAR SL and ATR SL.
    """

    _COOLDOWN_BARS: int = 3

    def __init__(
        self,
        market_series: MarketSeries,

        # PSAR
        psar_step: float = 0.02,
        psar_max_step: float = 0.2,

        # EMA trend filter
        use_ema_trend: bool = True,
        ema_trend_period: int = 200,
        ema_offset: int = 3,
        ema_slope_threshold: float = 0.0,

        # ADX trend-strength filter
        use_adx: bool = True,
        adx_period: int = 14,
        adx_threshold: float = 25.0,
        require_adx_bias: bool = False,

        # ATR
        atr_period: int = 14,

        # Behaviour
        use_close_signal: bool = True,
        max_candles: int = 500,
    ):
        self.use_ema_trend = use_ema_trend
        self.use_adx = use_adx
        self.require_adx_bias = require_adx_bias
        self.use_close_signal = use_close_signal
        self.ema_slope_threshold = ema_slope_threshold

        # Indicators
        self.psar = PSARIndicator(
            step=psar_step,
            max_step=psar_max_step,
            history_size=max_candles,
        )

        self.atr = ATRIndicator(period=atr_period)

        self.ema_trend = (
            EMAIndicator(
                period=ema_trend_period,
                source="close",
                offset=ema_offset,
            )
            if use_ema_trend
            else None
        )

        self.adx = (
            ADXIndicator(
                period=adx_period,
                trend_threshold=adx_threshold,
                history_size=max_candles,
            )
            if use_adx
            else None
        )

        # Seed indicators
        self.current_psar = self.psar.calculate(market_series)
        self.current_atr = self.atr.calculate(market_series)

        self.current_ema_trend = (
            self.ema_trend.calculate(market_series)
            if self.ema_trend is not None
            else None
        )

        self.current_adx = (
            self.adx.calculate(market_series)
            if self.adx is not None
            else None
        )

        # Candle buffer
        self.candles: Deque[Candle] = deque(maxlen=max_candles)
        for candle in market_series.candles():
            self.candles.append(candle)

        # Trade state
        self._open_direction: Optional[str] = None
        self._current_trade: Optional[Any] = None
        self._current_trade_sl: Optional[float] = None
        self._current_trade_tp: Optional[float] = None

        self._last_exit_type: Optional[str] = None
        self._last_exit_price: Optional[float] = None

        self._bars_since_stop: int = self._COOLDOWN_BARS

    # ---------------------------------------------------------
    # Public API
    # ---------------------------------------------------------
    def update(self, candle: Candle) -> Optional[StrategySignal]:
        """
        Process one closed candle.

        Order:
        1. Update indicators.
        2. If a trade is open, check close signal.
        3. If flat, check PSAR flip entry.
        """
        self.current_psar = self.psar.update(candle)
        self.current_atr = self.atr.update(candle)

        if self.ema_trend is not None:
            self.current_ema_trend = self.ema_trend.update(candle)

        if self.adx is not None:
            self.current_adx = self.adx.update(candle)

        self.candles.append(candle)
        self._bars_since_stop += 1

        if self.current_psar is None or self.current_atr is None:
            return None

        # Close signal first
        if self.use_close_signal and self._open_direction is not None:
            close_signal = self._check_close(candle)
            if close_signal is not None:
                return close_signal

        # Entry only when strategy knows no trade is open
        if self._open_direction is None:
            if self.psar.flipped_buy() and self._is_long_valid(candle):
                return self._build_entry_signal("BUY", candle)

            if self.psar.flipped_sell() and self._is_short_valid(candle):
                return self._build_entry_signal("SELL", candle)

        return None

    def sync_trade(self, trade: Any) -> None:
        """
        Called only after the backtest/executor actually opens a trade.
        """
        self._current_trade = trade
        self._current_trade_sl = getattr(trade, "stop_loss", None)
        self._current_trade_tp = getattr(trade, "take_profit", None)
        self._set_direction_from_trade(trade)

    def on_trade_closed(
        self,
        exit_type: str,
        exit_price: Optional[float] = None,
    ) -> None:
        self._last_exit_type = exit_type
        self._last_exit_price = exit_price

        if exit_type.upper() == "SL":
            self._reset_after_stop()
        else:
            self.clear_open_trade()

    def clear_open_trade(self) -> None:
        self._open_direction = None
        self._current_trade = None
        self._current_trade_sl = None
        self._current_trade_tp = None

    def state(self) -> dict:
        return {
            "psar": self.current_psar,
            "psar_trend": self.psar.trend,
            "psar_previous_trend": self.psar.previous_trend,
            "psar_has_flipped": self.psar.has_flipped,
            "psar_flip_direction": self.psar.flip_direction,
            "psar_ep": self.psar.ep,
            "psar_af": self.psar.af,

            "atr": self.current_atr,

            "use_ema_trend": self.use_ema_trend,
            "ema_trend": self.current_ema_trend,
            "ema_slope": (
                self.ema_trend.slope()
                if self.ema_trend is not None
                else None
            ),

            "use_adx": self.use_adx,
            "adx": self.current_adx,
            "plus_di": self.adx.plus_di if self.adx is not None else None,
            "minus_di": self.adx.minus_di if self.adx is not None else None,
            "adx_bias": self.adx.directional_bias() if self.adx is not None else None,
            "adx_is_trending": self.adx.is_trending() if self.adx is not None else None,

            "open_direction": self._open_direction,
            "current_trade_sl": self._current_trade_sl,
            "current_trade_tp": self._current_trade_tp,

            "bars_since_stop": self._bars_since_stop,
            "cooldown_active": self._bars_since_stop < self._COOLDOWN_BARS,

            "last_exit_type": self._last_exit_type,
            "last_exit_price": self._last_exit_price,
        }

    # ---------------------------------------------------------
    # Entry validation
    # ---------------------------------------------------------
    def _is_long_valid(self, candle: Candle) -> bool:
        return (
            self._bars_since_stop >= self._COOLDOWN_BARS
            and self._ema_allows_buy(candle)
            and self._adx_allows_buy()
        )

    def _is_short_valid(self, candle: Candle) -> bool:
        return (
            self._bars_since_stop >= self._COOLDOWN_BARS
            and self._ema_allows_sell(candle)
            and self._adx_allows_sell()
        )

    def _ema_allows_buy(self, candle: Candle) -> bool:
        if not self.use_ema_trend:
            return True

        if self.ema_trend is None or self.current_ema_trend is None:
            return False

        return self.ema_trend.is_uptrend(
            current_price=candle.close,
            slope_threshold=self.ema_slope_threshold,
        )

    def _ema_allows_sell(self, candle: Candle) -> bool:
        if not self.use_ema_trend:
            return True

        if self.ema_trend is None or self.current_ema_trend is None:
            return False

        return self.ema_trend.is_downtrend(
            current_price=candle.close,
            slope_threshold=self.ema_slope_threshold,
        )

    def _adx_allows_buy(self) -> bool:
        if not self.use_adx:
            return True

        if self.adx is None or self.current_adx is None:
            return False

        if not self.adx.is_trending():
            return False

        if self.require_adx_bias and not self.adx.has_buy_bias():
            return False

        return True

    def _adx_allows_sell(self) -> bool:
        if not self.use_adx:
            return True

        if self.adx is None or self.current_adx is None:
            return False

        if not self.adx.is_trending():
            return False

        if self.require_adx_bias and not self.adx.has_sell_bias():
            return False

        return True

    # ---------------------------------------------------------
    # Close logic
    # ---------------------------------------------------------
    def _check_close(self, candle: Candle) -> Optional[StrategySignal]:
        """
        Close on opposite PSAR flip.

        ADX should filter entries, not block emergency/structure exits.
        """
        reason = None
        pattern = None

        if self._open_direction == "LONG" and self.psar.flipped_sell():
            reason = (
                f"PSAR flipped bearish against LONG | "
                f"PSAR: {self.current_psar:.4f} | "
                f"Close: {candle.close:.4f}"
            )
            pattern = "PSAR_BEARISH_FLIP_EXIT"

        elif self._open_direction == "SHORT" and self.psar.flipped_buy():
            reason = (
                f"PSAR flipped bullish against SHORT | "
                f"PSAR: {self.current_psar:.4f} | "
                f"Close: {candle.close:.4f}"
            )
            pattern = "PSAR_BULLISH_FLIP_EXIT"

        if reason is None:
            return None

        return StrategySignal(
            signal="CLOSE",
            strategy_type="TRENDING",
            reason=reason,
            pattern_name=pattern,
            atr=self.current_atr,
            sl=None,
            tp=None,
            candle=candle,
        )

    # ---------------------------------------------------------
    # Entry signal builder
    # ---------------------------------------------------------
    def _build_entry_signal(self, direction: str, candle: Candle) -> StrategySignal:
        psar_sl = self._raw_psar_stop_loss(direction)

        ema_text = (
            f"EMA: {self.current_ema_trend:.4f} | "
            f"EMA slope: {self.ema_trend.slope():.4f}"
            if self.ema_trend is not None and self.current_ema_trend is not None
            else "EMA filter disabled"
        )

        adx_text = (
            f"ADX: {self.current_adx:.2f} | "
            f"+DI: {self.adx.plus_di:.2f} | "
            f"-DI: {self.adx.minus_di:.2f} | "
            f"Bias: {self.adx.directional_bias()}"
            if self.adx is not None and self.current_adx is not None
            else "ADX filter disabled"
        )

        reason = (
            f"PSAR {direction} flip | "
            f"Raw PSAR SL: {psar_sl:.4f} | "
            f"Close: {candle.close:.4f} | "
            f"Trend: {self.psar.trend}"
        )

        pattern_name = (
            f"{ema_text} | "
            f"{adx_text} | "
            f"ATR: {self.current_atr:.4f}"
        )

        return StrategySignal(
            signal=direction,
            strategy_type="TRENDING",
            reason=reason,
            pattern_name=pattern_name,
            atr=self.current_atr,
            sl=psar_sl,
            tp=None,
            candle=candle,
        )

    def _raw_psar_stop_loss(self, direction: str) -> Optional[float]:
        """
        Return raw first PSAR dot after flip.

        The risk manager/backtest will compare this with ATR-based SL.
        """
        if self.current_psar is None:
            return None

        return float(self.current_psar)

    # ---------------------------------------------------------
    # Trade-state helpers
    # ---------------------------------------------------------
    def _set_direction_from_trade(self, trade: Any) -> None:
        direction = str(getattr(trade, "direction", "")).upper()

        if direction == "BUY":
            self._open_direction = "LONG"
        elif direction == "SELL":
            self._open_direction = "SHORT"

    def _reset_after_stop(self) -> None:
        self.clear_open_trade()
        self._bars_since_stop = 0