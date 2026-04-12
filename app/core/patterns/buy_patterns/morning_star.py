from app.core.market.market_series import MarketSeries
from app.core.patterns.base_pattern import CandlestickPattern

class MorningStar(CandlestickPattern):
    def is_present(self, series: MarketSeries) -> bool:
        # 1. Need at least 5 candles (2 for downtrend context + 3 for pattern)
        if len(series) < 5:
            return False

        c1, c2, c3 = series.last(3)
        prev_lows = series.slice(5).lows()

        # 2. Trend Context: Must be at a recent low
        # The 'Star' (c2) should be the lowest point of the recent window
        is_at_bottom = c2.low <= min(prev_lows)

        # 3. Geometric Logic
        # Candle 1: Strong Bearish (The panic phase)
        c1_strong = c1.is_bearish and c1.body_size > 0

        # Candle 2 (The Star): Indecision (The bottoming phase)
        # Small body (<= 30% of C1) and should ideally gap down
        c2_is_star = c2.body_size <= (c1.body_size * 0.3)
        has_gap = c2.open <= c1.close # Price "gapped" or stayed below C1 close

        # Candle 3: Strong Bullish Reversal (The recovery phase)
        # Must close at least 50% into the body of C1
        midpoint_c1 = c1.low + (c1.body_size / 2)
        c3_reversal = c3.is_bullish and c3.close >= midpoint_c1

        return (
            is_at_bottom and
            c1_strong and
            c2_is_star and
            has_gap and
            c3_reversal
        )

# class MorningStar(CandlestickPattern):
#
#     def is_present(self, series: MarketSeries) -> bool:
#         if len(series) < 3:
#             return False
#
#         c1, c2, c3 = series.last(3)
#
#         c1_bearish = c1.close < c1.open
#         c3_bullish = c3.close > c3.open
#
#         c1_body = abs(c1.close - c1.open)
#         c2_body = abs(c2.close - c2.open)
#
#         return (
#                 c1_bearish and
#                 c3_bullish and
#                 c2_body < c1_body * 0.5 and  # Indecision
#                 c3.close > (c1.open + c1.close) / 2
#         )
