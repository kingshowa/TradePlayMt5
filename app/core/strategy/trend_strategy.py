from typing import Optional

from app.core.market.market_series import MarketSeries
from app.core.patterns.buy_patterns.any_buy_pattern import AnyBuyPattern
from app.core.patterns.sell_patterns.any_sale_pattern import AnySalePattern
from app.core.strategy.trade_signal import StrategySignal
from app.core.structure.market_context import MarketContext


class TrendingStrategy:
    def __init__(self):
        self.any_buy_pattern = AnyBuyPattern()
        self.any_sell_pattern = AnySalePattern()

    def generate_signal(self, context: MarketContext) -> Optional[StrategySignal]:
        if context.state is None:
            return None

        series = MarketSeries(list(context.candles))

        if context.state.trend == "UP":
            reasons = []

            if context.state.structure_event == "BOS_UP":
                reasons.append("bullish break of structure")

            if context.state.in_support:
                reasons.append("retracement into support")

            if context.state.trend_strength == "HIGH":
                reasons.append("strong uptrend")

            if len(reasons) >= 2 and self.any_buy_pattern.is_present(series):
                return StrategySignal(
                    signal="BUY",
                    strategy_type="TRENDING",
                    reason=", ".join(reasons),
                    pattern_name=self.any_buy_pattern.getPatternName(),
                    atr=context.state.atr,
                    sl=None,
                    tp=None,
                    candle=context.candles[-1],
                )

        if context.state.trend == "DOWN":
            reasons = []

            if context.state.structure_event == "BOS_DOWN":
                reasons.append("bearish break of structure")

            if context.state.in_resistance:
                reasons.append("retracement into resistance")

            if context.state.trend_strength == "HIGH":
                reasons.append("strong downtrend")

            if len(reasons) >= 2 and self.any_sell_pattern.is_present(series):
                return StrategySignal(
                    signal="SELL",
                    strategy_type="TRENDING",
                    reason=", ".join(reasons),
                    pattern_name=self.any_sell_pattern.getPatternName(),
                    atr=context.state.atr,
                    sl=None,
                    tp=None,
                    candle=context.candles[-1],
                )

        return None