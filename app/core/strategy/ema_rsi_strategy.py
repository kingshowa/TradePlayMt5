from collections import deque
from typing import Optional, Deque

from app.core.indicators.atr import ATRIndicator
from app.core.indicators.ema import EMAIndicator
from app.core.indicators.rsi import RSIIndicator
from app.core.market.candle import Candle
from app.core.market.market_series import MarketSeries
from app.core.strategy.trade_signal import StrategySignal


class EmaRsiChannelStrategy:
    """
    EMA + RSI channel-based signal strategy.

    Logic:
    - EMA(period, high) = dynamic resistance
    - EMA(period, low) = dynamic support
    - BUY when:
        * both EMAs are rising
        * RSI is above bullish threshold and rising
        * price is inside EMA channel
        * price is in lower half of channel
    - SELL when:
        * both EMAs are falling
        * RSI is below bearish threshold and falling
        * price is inside EMA channel
        * price is in upper half of channel

    This class generates only StrategySignal objects.
    Risk management (SL, TP, lot size) should be handled elsewhere.
    """

    def __init__(
        self,
        market_series: MarketSeries,
        ema_period: int = 20,
        ema_offset: int = 1,
        rsi_period: int = 20,
        rsi_offset: int = 3,
        rsi_buy_level: float = 52.0,
        rsi_sell_level: float = 48.0,
        max_candles: int = 200,
    ):
        self.market_series = market_series

        self.ema_low = EMAIndicator(
            period=ema_period,
            source="low",
            offset=ema_offset
        )
        self.ema_high = EMAIndicator(
            period=ema_period,
            source="high",
            offset=ema_offset
        )
        self.rsi = RSIIndicator(
            period=rsi_period,
            slope_offset=rsi_offset
        )

        self.atr_indicator = ATRIndicator(period=14)

        self.rsi_buy_level = rsi_buy_level
        self.rsi_sell_level = rsi_sell_level

        self.current_ema_low = self.ema_low.calculate(market_series)
        self.current_ema_high = self.ema_high.calculate(market_series)
        self.current_rsi = self.rsi.calculate(market_series)
        self.current_atr = self.atr_indicator.calculate(market_series)

        self.candles: Deque[Candle] = deque(maxlen=max_candles)
        for candle in market_series._candles:
            self.candles.append(candle)

    def update(self, candle: Candle) -> Optional[StrategySignal]:
        self.current_ema_low = self.ema_low.update(candle)
        self.current_ema_high = self.ema_high.update(candle)
        self.current_rsi = self.rsi.update(candle)
        self.current_atr = self.atr_indicator.update(candle)
        self.candles.append(candle)

        if len(self.candles) < 2:
            return None

        series = MarketSeries(list(self.candles))

        if self._is_buy_setup(candle):
            return StrategySignal(
                signal="BUY",
                strategy_type="TRENDING",
                reason=f"PreEMAhigh: {self.ema_high.previous_value:.4f} CurrEMAhigh: {self.ema_high.current_value:.4f}",                 #self._build_buy_reason(candle),
                pattern_name=f"PrevRSI:{self.rsi.previous_value:.2f} CurrRSI: {self.rsi.current_value:.2f}",
                atr=self.current_atr,
                sl=self.current_ema_low,
                tp=None,
                candle=series.last(),
            )

        if self._is_sell_setup(candle):
            return StrategySignal(
                signal="SELL",
                strategy_type="TRENDING",
                reason=f"PreEMAhigh: {self.ema_high.previous_value:.4f} CurrEMAhigh: {self.ema_high.current_value:.4f}",                 #self._build_sell_reason(candle),
                pattern_name=f"PrevRSI:{self.rsi.previous_value:.2f} CurrRSI: {self.rsi.current_value:.2f}",
                atr=self.current_atr,
                sl=self.current_ema_high,
                tp=None,
                candle=series.last(),
            )

        return None

    # -----------------------------------------
    # Core setup checks
    # -----------------------------------------
    def _is_buy_setup(self, candle: Candle) -> bool:
        return (
            self._is_price_inside_channel(candle.low)
            # and self._is_price_in_lower_half(candle.close)
            and self._is_price_in_buy_zone(candle.close)
            and self.ema_low.is_rising()
            and self.ema_high.is_rising()
            and self.current_rsi > self.rsi_buy_level
            and self.rsi.is_rising(strict=False)
        )

    def _is_sell_setup(self, candle: Candle) -> bool:
        return (
            self._is_price_inside_channel(candle.high)
            # and self._is_price_in_upper_half(candle.close)
            and self._is_price_in_sell_zone(candle.close)
            and self.ema_low.is_falling()
            and self.ema_high.is_falling()
            and self.current_rsi < self.rsi_sell_level
            and self.rsi.is_falling(strict=False)
        )

    # -----------------------------------------
    # Channel helpers
    # -----------------------------------------
    def _is_price_inside_channel(self, price: float) -> bool:
        return self.current_ema_low <= price <= self.current_ema_high

    def _channel_midpoint(self) -> float:
        return (self.current_ema_low + self.current_ema_high) / 2.0

    def _is_price_in_lower_half(self, price: float) -> bool:
        return price <= self._channel_midpoint()

    def _is_price_in_upper_half(self, price: float) -> bool:
        return price >= self._channel_midpoint()

    def _is_price_in_buy_zone(self, price: float) -> bool:
        return price <= self.current_ema_high + self.current_atr
    def _is_price_in_sell_zone(self, price: float) -> bool:
        return price >= self.current_ema_low - self.current_atr

    def channel_width(self) -> float:
        return self.current_ema_high - self.current_ema_low

    # -----------------------------------------
    # Reason builders
    # -----------------------------------------
    def _build_buy_reason(self, candle: Candle) -> str:
        return (
            f"BUY: price inside EMA channel near support, "
            f"EMA low/high rising, RSI bullish ({self.current_rsi:.2f})"
        )

    def _build_sell_reason(self, candle: Candle) -> str:
        return (
            f"SELL: price inside EMA channel near resistance, "
            f"EMA low/high falling, RSI bearish ({self.current_rsi:.2f})"
        )