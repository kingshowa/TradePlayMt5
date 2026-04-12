from app.core.patterns.buy_patterns.any_buy_pattern import AnyBuyPattern
from app.core.patterns.sell_patterns.any_sale_pattern import AnySalePattern
from app.core.market.market_series import MarketSeries
from app.core.patterns.buy_patterns.bullish_engulfing import BullishEngulfing
from app.core.patterns.sell_patterns.bearish_engulfing import BearishEngulfing
from app.core.structure.market_context_v1 import MarketContext
from app.core.risk.risk_manager import RiskManager
from app.core.strategy.trade_object import Trade

class CommodityStrategy:

    def __init__(self, balance: float):
        self.market_context = MarketContext()
        self.risk_manager = RiskManager(balance=balance)
        self.bullish_pattern = BullishEngulfing()
        self.bearish_pattern = BearishEngulfing()
        self.any_buy_pattern = AnyBuyPattern()
        self.any_sale_pattern = AnySalePattern()

    def generate_signal(self, series: MarketSeries) -> Trade | None:
        last_candle = series.last()

        # 1️⃣ Determine trend
        if self.market_context.is_uptrend(series):
            trend = "up"
            print("Up trending")
        elif self.market_context.is_downtrend(series):
            trend = "down"
            print("Down trending")
        else:
            print("No trend")
            return None  # flat / no clear trend

        # 2️⃣ Check for patterns
        if trend == "up":
            if self.any_buy_pattern.is_present(series):  # and self.market_context.near_support(series)
                direction = "buy"
                pattern_name = self.any_buy_pattern.getPatternName()

            else:
                # direction = "buy"
                # pattern_name = "kingshowa"
                return None

        elif trend == "down":
            if self.any_sale_pattern.is_present(series): #  and self.market_context.near_resistance(series)
                direction = "sell"
                pattern_name = self.any_sale_pattern.getPatternName()
            else:
                return None

        # 3️⃣ Compute ATR
        atr = self.market_context.atr(series)

        # 4️⃣ Build trade via RiskManager
        trade_data = self.risk_manager.build_trade(
            candle=last_candle,
            atr=atr,
            direction=direction,
            pattern_name = pattern_name
        )

        trade = Trade(**trade_data)
        return trade
