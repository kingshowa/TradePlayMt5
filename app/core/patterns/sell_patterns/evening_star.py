from app.core.market.market_series import MarketSeries
from app.core.patterns.base_pattern import CandlestickPattern


class EveningStar(CandlestickPattern):
    def is_present(self, series: MarketSeries) -> bool:
        # 1. Need at least 5 candles (2 to establish trend + 3 for the pattern)
        if len(series) < 5:
            return False

        # Extract the three-candle sequence
        c1, c2, c3 = series.last(3)
        prev_highs = series.slice(5).highs()  # For trend check

        # 2. Trend Context: Must be at a recent high
        # Ensure the 'Star' (c2) is higher than the preceding trend
        is_at_peak = c2.high >= max(prev_highs)

        # 3. Geometric Logic
        # Candle 1: Strong Bullish
        c1_strong = c1.is_bullish and c1.body_size > 0

        # Candle 2 (The Star): Indecision
        # Small body (less than 30% of C1) and can be either color
        c2_is_star = c2.body_size <= (c1.body_size * 0.3)

        # Gap Logic: In ideal patterns, the 'Star' gaps away from C1
        # In Forex, we often accept "near-gaps" (Open2 >= Close1)
        has_gap = c2.open >= c1.close

        # Candle 3: Strong Bearish Reversal
        # Must close deep (at least 50%) into C1's body
        midpoint_c1 = c1.open + (c1.body_size / 2)
        c3_reversal = c3.is_bearish and c3.close <= midpoint_c1

        return (
                is_at_peak and
                c1_strong and
                c2_is_star and
                has_gap and
                c3_reversal
        )


# class EveningStar(CandlestickPattern):
#
#     def is_present(self, series: MarketSeries) -> bool:
#         if len(series) < 3:
#             return False
#
#         c1, c2, c3 = series.last(3)
#
#         c1_bullish = c1.close > c1.open
#         c3_bearish = c3.close < c3.open
#
#         c1_body = abs(c1.close - c1.open)
#         c2_body = abs(c2.close - c2.open)
#
#         return (
#                 c1_bullish and
#                 c3_bearish and
#                 c2_body < c1_body * 0.5 and
#                 c3.close < (c1.open + c1.close) / 2
#         )
