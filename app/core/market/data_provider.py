from abc import ABC, abstractmethod
from app.core.market.market_series import MarketSeries

class MarketDataProvider(ABC):

    @abstractmethod
    def fetch(self, symbol: str, timeframe: str, bars: int) -> MarketSeries:
        pass
