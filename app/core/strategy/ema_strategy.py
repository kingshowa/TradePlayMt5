from collections import deque
from typing import Optional, Deque

from app.core.indicators.atr import ATRIndicator
from app.core.indicators.ema import EMAIndicator
from app.core.market.candle import Candle
from app.core.market.market_series import MarketSeries
from app.core.patterns.buy_patterns.any_buy_pattern import AnyBuyPattern
from app.core.patterns.sell_patterns.any_sale_pattern import AnySalePattern
from app.core.strategy.trade_signal import StrategySignal


class EmaStrategy:
    def __init__(self, market_series: MarketSeries):
        self.market_series = market_series

        self.atr_indicator = ATRIndicator(period=14)
        self.ema_indicator = EMAIndicator(period=50)

        self.any_buy_pattern = AnyBuyPattern()
        self.any_sell_pattern = AnySalePattern()

        self.current_atr = self.atr_indicator.calculate(self.market_series)
        self.current_ema = self.ema_indicator.calculate(self.market_series)

        self.candles: Deque[Candle] = deque(maxlen=200)
        for candle in market_series._candles:
            self.candles.append(candle)

    def update(self, candle: Candle) -> Optional[StrategySignal]:
        self.current_atr = self.atr_indicator.update(candle)
        self.current_ema = self.ema_indicator.update(candle)
        self.candles.append(candle)

        series = MarketSeries(list(self.candles))
        slope_threshold = self.current_atr * 0.04


        if self.ema_indicator.is_uptrend(candle.close, slope_threshold) and self.any_buy_pattern.is_present(series):
            return StrategySignal(
                signal="BUY",
                strategy_type="TRENDING",
                reason="Price above ema",
                pattern_name=self.any_buy_pattern.getPatternName(),
                atr=self.current_atr,
                sl=None,
                tp=None,
                candle=self.candles[-1],
            )

        if self.ema_indicator.is_downtrend(candle.close, slope_threshold) and self.any_sell_pattern.is_present(series):
            return StrategySignal(
                signal="SELL",
                strategy_type="TRENDING",
                reason="Price below ema",
                pattern_name=self.any_sell_pattern.getPatternName(),
                atr=self.current_atr,
                sl=None,
                tp=None,
                candle=self.candles[-1],
            )

        return None