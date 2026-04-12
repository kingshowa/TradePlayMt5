from app.core.market.market_series import MarketSeries
from app.core.patterns.base_pattern import CandlestickPattern

class BullishInsideBar(CandlestickPattern):
    def is_present(self, series: MarketSeries) -> bool:
        # 1. Need at least 4 candles for downtrend context
        if len(series) < 4:
            return False

        prev = series.previous() # The "Mother Bar"
        curr = series.last()     # The "Inside Bar"

        # 2. Trend Context: The Mother Bar should be at a recent low
        # This ensures we are catching a reversal, not just a pause in a rally.
        is_at_bottom = prev.low <= min(series.slice(4).lows())

        # 3. Mother Bar Strength
        # The first candle should be a decisive bearish candle.
        is_mother_bar_valid = prev.is_bearish and prev.body_size > 0

        # 4. Geometric Logic (Containment)
        # The current candle's range must be completely inside the mother bar's range.
        is_inside = curr.high < prev.high and curr.low > prev.low

        # 5. Bullish Confirmation
        # The inner candle should close higher than it opened to show buying intent.
        is_bullish_finish = curr.is_bullish

        return is_at_bottom and is_mother_bar_valid and is_inside and is_bullish_finish

# class BullishInsideBar(CandlestickPattern):
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
#                 curr.close > curr.open
#         )
