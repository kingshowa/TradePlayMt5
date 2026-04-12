from typing import Optional

from app.core.indicators.atr import ATRIndicator
from app.core.market.candle import Candle
from app.core.market.market_series import MarketSeries
from app.core.strategy.trade_signal import StrategySignal
from app.core.strategy.range_strategy import RangingStrategy
from app.core.strategy.trend_strategy import TrendingStrategy
from app.core.structure.market_context import MarketContext, MarketContextState


class BaseStrategy:
    def __init__(self, market_series: MarketSeries):
        self.market_series = market_series

        self.atr_indicator = ATRIndicator(period=14)
        self.market_context = MarketContext(
            swing_lookback=5,
            zone_atr_multiplier=1.0,
            break_atr_multiplier=0.15,
            max_candles=200,
            max_swings=50,
        )

        self.trending_strategy = TrendingStrategy()
        self.ranging_strategy = RangingStrategy()

        self.current_atr = self.atr_indicator.calculate(self.market_series)
        self.current_state = self.market_context.initialize(
            self.market_series,
            self.current_atr,
        )

    def update(self, candle: Candle) -> Optional[StrategySignal]:
        self.current_atr = self.atr_indicator.update(candle)
        self.current_state = self.market_context.update(candle, self.current_atr)

        return self.map_strategy(self.current_state)

    # def map_strategy(self, current_state: MarketContextState) -> Optional[StrategySignal]:
    #     if current_state.trend in ["UP", "DOWN"]:
    #         return self.trending_strategy.generate_signal(self.market_context)
    #
    #     if current_state.trend == "RANGE":
    #         return self.ranging_strategy.generate_signal(self.market_context)
    #
    #     return None

    def map_strategy(self, current_state: MarketContextState) -> Optional[StrategySignal]:
        if self._should_use_trending_strategy(current_state):
            return self.trending_strategy.generate_signal(self.market_context)

        return self.ranging_strategy.generate_signal(self.market_context)
        # return None
    def _should_use_trending_strategy(self, current_state: MarketContextState) -> bool:
        if current_state.trend not in ("UP", "DOWN"):
            return False

        if current_state.trend_strength != "HIGH":
            return False

        if current_state.structure_event in ("CHOCH_UP", "CHOCH_DOWN"):
            return False

        return True