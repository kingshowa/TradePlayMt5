from app.core.market.market_series import MarketSeries
from app.core.patterns.base_pattern import CandlestickPattern

class BullishEngulfing(CandlestickPattern):
    def is_present(self, series: MarketSeries) -> bool:
        # 1. Need at least 4 candles for downtrend context
        if len(series) < 4:
            # Not enough history to confirm a trend
            return False

        prev = series.previous()
        curr = series.last()

        # 2. Trend Context: Previous candle should be hitting a local low
        # This filters out random engulfing bars in a sideways market
        is_at_bottom = prev.low <= min(series.slice(4).lows())

        # 3. Geometric Logic (The Engulf)
        # Condition A: Color Flip (Bearish then Bullish)
        is_color_flip = prev.is_bearish and curr.is_bullish

        # Condition B: Body Engulfing (The standard requirement)
        # Current body must strictly cover the previous body
        is_body_engulfed = (curr.open <= prev.close and
                           curr.close > prev.open)

        # Condition C: Range Engulfing (The "Optimal" kicker)
        # For high reliability, the current High/Low should exceed the previous High/Low
        is_range_engulfed = curr.high >= prev.high and curr.low <= prev.low

        # 4. Momentum Check
        # The engulfing candle should show significant buying power relative to the seller
        is_strong_move = curr.body_size > (prev.body_size * 1.2)

        return (
            is_at_bottom and
            is_color_flip and
            is_body_engulfed and
            is_range_engulfed and
            is_strong_move
        )

# class BullishEngulfing(CandlestickPattern):
#
#     def is_present(self, series: MarketSeries) -> bool:
#         if len(series) < 2:
#             return False
#
#         prev = series.previous()
#         curr = series.last()
#
#         return (
#                 prev.close < prev.open and  # Previous bearish
#                 curr.close > curr.open and  # Current bullish
#                 curr.open <= prev.close and
#                 curr.close >= prev.open
#         )
