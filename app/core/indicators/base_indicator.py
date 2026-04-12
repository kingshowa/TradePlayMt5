# app/core/indicators/base.py

from abc import ABC, abstractmethod
from typing import Optional
from app.core.market.market_series import MarketSeries
from app.core.market.candle import Candle


class BaseIndicator(ABC):
    """
    Abstract indicator using MarketSeries.
    """

    def __init__(self, period: int):
        self.period = period
        self.current_value: Optional[float] = None

    @abstractmethod
    def calculate(self, series: MarketSeries):
        """
        Batch initialization using full series.
        """
        pass

    @abstractmethod
    def update(self, candle: Candle):
        """
        Incremental update using single closed candle.
        """
        pass