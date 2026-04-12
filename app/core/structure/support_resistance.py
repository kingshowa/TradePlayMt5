from app.core.indicators.atr import ATRIndicator
from app.core.market.market_series import MarketSeries

class SupportResistanceDetector:

    def is_near_support(self, series: MarketSeries, price: float, lookback=20, atr_multiplier=1.0) -> bool:
        """
        Determines if the price is within a volatility-adjusted zone of the recent low.
        """
        # 1. Identify the static Support Level (the floor)
        lows = series.lows()[-lookback:]
        recent_low = min(lows)

        # 2. Calculate the current ATR (Volatility)
        # Using your ATRIndicator class with the same lookback
        atr_calc = ATRIndicator(period=lookback)
        try:
            current_atr = atr_calc.calculate(series)
        except ValueError:
            # Fallback if not enough data yet
            return False

        # 3. Define the Dynamic Tolerance
        # We use a multiplier of ATR (e.g., 1.0 * ATR) to define the zone's width
        buffer_zone = current_atr * atr_multiplier

        # 4. Logic: Is the price inside the zone?
        # We check if: Recent_Low <= Current_Price <= (Recent_Low + Buffer)
        # This prevents the bot from buying if price has already crashed BELOW support
        is_above_support = price >= (recent_low - (buffer_zone * 0.1))  # Small margin for error below
        is_within_zone = price <= (recent_low + buffer_zone)

        return is_above_support and is_within_zone


    def is_near_resistance(self, series: MarketSeries, price: float, lookback=5, atr_multiplier=1.0) -> bool:
        """
        Determines if the price is within a volatility-adjusted zone of the recent high.
        """
        # 1. Identify the static Resistance Level (the ceiling)
        highs = series.highs()[-lookback:]
        recent_high = max(highs)

        # 2. Calculate current Volatility (ATR)
        atr_calc = ATRIndicator(period=lookback)
        try:
            current_atr = atr_calc.calculate(series)
        except ValueError:
            return False

        # 3. Define the Dynamic Tolerance
        buffer_zone = current_atr * atr_multiplier

        # 4. Logic: Is the price inside the zone?
        # We check if: (Recent_High - Buffer) <= Current_Price <= Recent_High
        # This ensures we are approaching the ceiling but haven't necessarily exploded through it.
        is_below_resistance = price <= (recent_high + (buffer_zone * 0.1))  # Small margin for "overshoot"
        is_within_zone = price >= (recent_high - buffer_zone)

        return is_below_resistance and is_within_zone

