from app.core.market.market_series import MarketSeries
from app.core.patterns.base_pattern import CandlestickPattern

class BearishEngulfing(CandlestickPattern):
    def is_present(self, series: MarketSeries) -> bool:
        # 1. Need at least 4 candles for trend context
        if len(series) < 4:
            return False

        prev = series.previous()
        curr = series.last()

        # 2. Trend Context: Previous candle should be at a relative high
        # Engulfing patterns in a downtrend are just "continuation noise"
        is_at_peak = prev.high >= max(series.slice(4).highs())

        # 3. Geometric Logic (The Engulf)
        # Condition A: Opposite colors (Bullish then Bearish)
        is_color_flip = prev.is_bullish and curr.is_bearish

        # Condition B: Body Engulfing (The minimum requirement)
        # The current body must strictly swallow the previous body
        is_body_engulfed = (curr.open >= prev.close and
                           curr.close < prev.open)

        # Condition C: Range Engulfing (The "Optimal" kicker)
        # For high reliability, the current High/Low should exceed the previous High/Low
        is_range_engulfed = curr.high >= prev.high and curr.low <= prev.low

        # 4. Momentum Check
        # The engulfing candle should be a "significant" move
        is_strong_move = curr.body_size > (prev.body_size * 1.2)

        return (
            is_at_peak and
            is_color_flip and
            is_body_engulfed and
            is_range_engulfed and
            is_strong_move
        )

#
# class BearishEngulfing(CandlestickPattern):
#
#     def is_present(self, series: MarketSeries) -> bool:
#         if len(series) < 2:
#             return False
#
#         prev = series.previous()
#         curr = series.last()
#
#         return (
#                 prev.close > prev.open and  # Previous bullish
#                 curr.close < curr.open and  # Current bearish
#                 curr.open >= prev.close and
#                 curr.close <= prev.open
#         )
