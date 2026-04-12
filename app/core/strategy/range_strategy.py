from typing import Optional

from app.core.market.market_series import MarketSeries
from app.core.strategy.trade_signal import StrategySignal
from app.core.structure.market_context import MarketContext
from app.core.patterns.buy_patterns.any_buy_pattern import AnyBuyPattern
from app.core.patterns.sell_patterns.any_sale_pattern import AnySalePattern


class RangingStrategy:
    def __init__(self):
        self.any_buy_pattern = AnyBuyPattern()
        self.any_sell_pattern = AnySalePattern()

    def generate_signal(self, context: MarketContext) -> Optional[StrategySignal]:
        if context.state is None:
            return None

        if context.state.support_zone is None or context.state.resistance_zone is None:
            return None

        # _, support_upper = context.state.support_zone
        # resistance_lower, _ = context.state.resistance_zone
        #
        # if support_upper > resistance_lower:
        #     return None

        series = MarketSeries(list(context.candles))

        if context.state.in_support and self.any_buy_pattern.is_present(series):
            sl, _ = context.state.support_zone
            tp, _ = context.state.resistance_zone
            return StrategySignal(
                signal="BUY",
                strategy_type="RANGING",
                reason="price is inside support zone in a ranging market",
                pattern_name=self.any_buy_pattern.getPatternName(),
                atr=context.state.atr,
                sl=None,
                tp=None,
                candle=context.candles[-1],
            )

        if context.state.in_resistance and self.any_sell_pattern.is_present(series):
            _, sl = context.state.resistance_zone
            _, tp = context.state.support_zone
            return StrategySignal(
                signal="SELL",
                strategy_type="RANGING",
                reason="price is inside resistance zone in a ranging market",
                pattern_name=self.any_sell_pattern.getPatternName(),
                atr=context.state.atr,
                sl=None,
                tp=None,
                candle=context.candles[-1],
            )

        return None

