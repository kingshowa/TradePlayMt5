from app.core.market.market_series import MarketSeries
from app.core.patterns.base_pattern import CandlestickPattern

class Hummer(CandlestickPattern):
    def is_present(self, series: MarketSeries) -> bool:
        # 1. Need at least 4 candles to confirm a downtrend
        if len(series) < 4:
            return False

        c = series.last()

        # 2. Trend Context: Must be at a recent low
        # A Hammer is only a "Hammer" if it's hitting a floor
        is_at_bottom = c.low <= min(series.slice(4).lows())

        # 3. Geometric Logic using your Candle properties
        total_range = c.high - c.low
        if total_range == 0:
            return False

        # Professional Hammer Criteria:
        # - Lower wick is at least 2x the body (standard requirement)
        # - Upper wick is very small (less than 10% of total range)
        # - Body is in the top 30% of the candle
        is_hammer_shape = (
                c.lower_wick >= (c.body_size * 2) and
                c.upper_wick <= (total_range * 0.1) and
                min(c.open, c.close) >= (c.high - total_range * 0.3) and
                c.body_size > 0
        )

        return is_at_bottom and is_hammer_shape

# class Hummer(CandlestickPattern):
#
#     def is_present(self, series: MarketSeries) -> bool:
#         if len(series) < 1:
#             return False
#
#         c = series.last()
#
#         body = abs(c.close - c.open)
#         lower_wick = min(c.open, c.close) - c.low
#         upper_wick = c.high - max(c.open, c.close)
#
#         return (
#                 lower_wick >= body * 1.5 and
#                 upper_wick <= body and
#                 body > 0
#         )
