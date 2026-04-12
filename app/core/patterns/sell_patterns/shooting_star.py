from app.core.market.market_series import MarketSeries
from app.core.patterns.base_pattern import CandlestickPattern


class ShootingStar(CandlestickPattern):
    def is_present(self, series: MarketSeries) -> bool:
        # Minimum 4 candles: 3 to establish a local peak, 1 for the star
        if len(series) < 4:
            return False

        c = series.last()

        # 1. Trend Context: Current high must be the highest of the recent window
        # We look at the last 4 candles (including current)
        lookback = series.slice(4)
        if c.high < max(lookback.highs()):
            return False

        # 2. Geometric Logic using your Candle properties
        total_range = c.high - c.low
        if total_range == 0:
            return False

        # Professional criteria:
        # - Upper wick is at least 2x the body
        # - Lower wick is negligible (less than 10% of total range)
        # - Body is small and located in the bottom 40% of the candle
        is_star_shape = (
                c.upper_wick >= (c.body_size * 2) and
                c.lower_wick <= (total_range * 0.1) and
                max(c.open, c.close) <= (c.low + total_range * 0.4) and
                c.body_size > 0
        )

        return is_star_shape


# class ShootingStar(CandlestickPattern):
#
#     def is_present(self, series: MarketSeries) -> bool:
#         if len(series) < 1:
#             return False
#
#         c = series.last()
#
#         body = abs(c.close - c.open)
#         upper_wick = c.high - max(c.open, c.close)
#         lower_wick = min(c.open, c.close) - c.low
#
#         return (
#                 upper_wick >= body * 2 and
#                 lower_wick <= body and
#                 body > 0
#         )
