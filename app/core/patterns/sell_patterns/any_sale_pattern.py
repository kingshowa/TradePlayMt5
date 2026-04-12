from app.core.market.market_series import MarketSeries
from app.core.patterns.base_pattern import CandlestickPattern
from app.core.patterns.sell_patterns.shooting_star import ShootingStar
from app.core.patterns.sell_patterns.bearish_engulfing import BearishEngulfing
from app.core.patterns.sell_patterns.bearish_inside_bar import BearishInsideBar
from app.core.patterns.sell_patterns.evening_star import EveningStar


class AnySalePattern(CandlestickPattern):
    def __init__(self):
        self.bearish_engulfing = BearishEngulfing()
        self.bearish_inside_bar = BearishInsideBar()
        self.evening_star = EveningStar()
        self.shooting_star = ShootingStar()
        self.pattern_name = ""


    def is_present(self, series: MarketSeries) -> bool:

        if self.bearish_engulfing.is_present(series):
            self.pattern_name = "bearish-engulfing"
            is_sale_present = True
        elif self.evening_star.is_present(series):
            self.pattern_name = "evening-star"
            is_sale_present = True
        elif self.shooting_star.is_present(series):
            self.pattern_name = "shooting-star"
            is_sale_present = True
        elif self.bearish_inside_bar.is_present(series):
            self.pattern_name = "bearish_inside_bar"
            is_sale_present = True
        else:
            is_sale_present = False

        return is_sale_present

    def getPatternName(self) -> str:
        return self.pattern_name