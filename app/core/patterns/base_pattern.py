from abc import ABC, abstractmethod
from app.core.market.candle import Candle
from app.core.market.market_series import MarketSeries


class CandlestickPattern(ABC):

    @abstractmethod
    def is_present(self, series: MarketSeries) -> bool:
        pass
