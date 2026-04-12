from app.core.market.market_series import MarketSeries
from app.core.patterns.base_pattern import CandlestickPattern
from app.core.patterns.buy_patterns.bullish_engulfing import BullishEngulfing
from app.core.patterns.buy_patterns.bullish_inside_bar import BullishInsideBar
from app.core.patterns.buy_patterns.hummer import Hummer
from app.core.patterns.buy_patterns.morning_star import MorningStar


class AnyBuyPattern(CandlestickPattern):
    def __init__(self):
        self.bullish_engulfing = BullishEngulfing()
        self.bullish_inside_bar = BullishInsideBar()
        self.hummer = Hummer()
        self.morning_star = MorningStar()
        self.pattern_name = ""

    def is_present(self, series: MarketSeries) -> bool:
        if self.bullish_engulfing.is_present(series):
            self.pattern_name = "bullish_engulfing"
            is_buy_present = True
        elif self.hummer.is_present(series):
            self.pattern_name = "hummer"
            is_buy_present = True
        elif self.morning_star.is_present(series):
            self.pattern_name = "morning_star"
            is_buy_present = True
        elif self.bullish_inside_bar.is_present(series):
            self.pattern_name = "bullish_inside_bar"
            is_buy_present = True
        else:
            is_buy_present = False

        return is_buy_present

    def getPatternName(self) -> str:
        return self.pattern_name
