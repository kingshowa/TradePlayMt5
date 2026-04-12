from typing import List
from app.core.market.candle import Candle

class MarketSeries:
    def __init__(self, candles: List[Candle]):
        if not candles:
            raise ValueError("MarketSeries cannot be empty")
        # Ensure candles are sorted by time
        self._candles = sorted(candles, key=lambda c: c.time)

    def __len__(self):
        return len(self._candles)

    def last(self, n: int = 1):
        if n == 1:
            return self._candles[-1]
        return self._candles[-n:]

    def previous(self) -> Candle:
        if len(self._candles) < 2:
            raise ValueError("Not enough candles to get previous")
        return self._candles[-2]

    def highs(self):
        return [c.high for c in self._candles]

    def lows(self):
        return [c.low for c in self._candles]

    def closes(self):
        return [c.close for c in self._candles]

    def opens(self):
        return [c.open for c in self._candles]

    def slice(self, n: int):
        """Return last n candles as a new MarketSeries"""
        return MarketSeries(self._candles[-n:])

    def subseries(self, start: int, end: int):
        """
        Return a subseries from start to end (exclusive), like a slice.
        Raises ValueError if indices are invalid.
        """
        if start < 0 or end > len(self._candles) or start >= end:
            raise ValueError("Invalid start/end for sub_series")
        return MarketSeries(self._candles[start:end])

    def append_from_mt5_rate(self, candle):

        # Avoid duplicate candle
        if self._candles and self._candles[-1].time == candle.time:
            return

        self._candles.append(candle)

    def candles(self):
        return list(self._candles)