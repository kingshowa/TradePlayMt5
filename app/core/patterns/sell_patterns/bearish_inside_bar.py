from app.core.market.market_series import MarketSeries
from app.core.patterns.base_pattern import CandlestickPattern

class BearishInsideBar(CandlestickPattern):
    def is_present(self, series: MarketSeries) -> bool:
        # Need at least 4 candles to establish a trend context
        if len(series) < 4:
            return False

        prev = series.previous() # The "Mother Bar"
        curr = series.last()     # The "Inside Bar"

        # 1. Trend Context: The Mother Bar should be at a recent high
        # This ensures we are catching a reversal, not a continuation.
        is_at_peak = prev.high >= max(series.slice(4).highs())

        # 2. Mother Bar Strength
        # The first candle should be a decisive bullish candle.
        is_mother_bar_valid = prev.is_bullish and prev.body_size > 0

        # 3. Geometric Logic (Containment)
        # The current candle's range must be completely inside the mother bar's range.
        is_inside = curr.high < prev.high and curr.low > prev.low

        # 4. Bearish Confirmation
        # To be "Bearish Inside Bar," the inner candle should close lower than it opened.
        is_bearish_finish = curr.is_bearish

        return is_at_peak and is_mother_bar_valid and is_inside and is_bearish_finish


# class BearishInsideBar(CandlestickPattern):
#
#     def is_present(self, series: MarketSeries) -> bool:
#         if len(series) < 2:
#             return False
#
#         prev = series.previous()
#         curr = series.last()
#
#         return (
#                 curr.high < prev.high and
#                 curr.low > prev.low and
#                 curr.close < curr.open
#         )
    