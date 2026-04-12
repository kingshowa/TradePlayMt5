# tests/test_market_context_real_mt5.py

import unittest
from datetime import datetime

from app.core.structure.market_context_v2 import MarketContext
from app.core.indicators.atr import ATRIndicator
from app.core.market.mt5_provider import MT5MarketDataProvider
from app.core.market.mt5_timeframes import TIMEFRAMES
from test.core.structure.plot_market_structure_v1 import plot_market_context


class TestMarketContextRealMT5(unittest.TestCase):

    SYMBOL = "XAUUSDm"
    TIMEFRAME = "5m"
    BARS = 200

    START_DATE = datetime(2026, 3, 17, hour=15, minute=25)
    END_DATE = datetime(2026, 3, 18, hour=17, minute=45)

    def setUp(self):
        self.provider = MT5MarketDataProvider()

        self.series = self.provider.fetch(
            self.SYMBOL,
            TIMEFRAMES[self.TIMEFRAME],
            self.BARS
        )

        # self.series = self.provider.fetch_range(
        #     self.SYMBOL,
        #     TIMEFRAMES[self.TIMEFRAME],
        #     self.START_DATE,
        #     self.END_DATE
        # )

        self.atr_indicator = ATRIndicator(period=14)
        self.context = MarketContext(
            swing_lookback=5,
            zone_atr_multiplier=1
        )

    # ==========================================================
    # TEST INITIALIZATION
    # ==========================================================

    def test_initialize_context(self):

        atr = self.atr_indicator.calculate(self.series.subseries(0,100))
        state = self.context.initialize(self.series.subseries(0,100), atr)

        self.assertIsNotNone(state)
        self.assertIn(state.trend, ["UP", "DOWN", "RANGE"])
        self.assertIsNotNone(state.atr)

        print("\n=== INITIAL STATE ===")
        print("Trend:", state.trend)
        print("Trend Strength:", state.trend_strength)
        print("Last Swing High:", state.last_swing_high)
        print("Last Swing Low:", state.last_swing_low)
        print("Support Zone:", state.support_zone)
        print("Resistance Zone:", state.resistance_zone)

        plot_market_context(self.series.subseries(0,199), self.context, state, title=f"{self.SYMBOL} {self.TIMEFRAME}")

    # ==========================================================
    # TEST STREAMING UPDATE
    # ==========================================================

    def test_streaming_update(self):

        # Initialize first
        atr = self.atr_indicator.calculate(self.series.subseries(0,80))
        state = self.context.initialize(self.series.subseries(0,80), atr)

        # Now simulate streaming new candles
        candles = self.series._candles
        print("")
        for candle in candles[-22:]:
            atr = self.atr_indicator.update(candle)
            state = self.context.update(candle, atr)

            print(f"{candle.time} {state.trend} {state.trend_strength} In Support: {state.in_support} In Resistance: {state.in_resistance}")

        self.assertIsNotNone(state)
        self.assertIn(state.trend, ["UP", "DOWN", "RANGE"])

        print("\n=== AFTER STREAMING ===")
        print("Trend:", state.trend)
        print("Trend Strength:", state.trend_strength)

    # ==========================================================
    # TEST ZONE LOGIC
    # ==========================================================

    def test_zone_detection(self):

        atr = self.atr_indicator.calculate(self.series)
        state = self.context.initialize(self.series, atr)

        self.assertTrue(
            state.support_zone is None or isinstance(state.support_zone, tuple)
        )

        self.assertTrue(
            state.resistance_zone is None or isinstance(state.resistance_zone, tuple)
        )

        print("\nZone Interaction:")
        print("In Support:", state.in_support)
        print("In Resistance:", state.in_resistance)

    # ==========================================================
    # TEST TREND STRENGTH LOGIC
    # ==========================================================

    def test_trend_strength_logic(self):

        atr = self.atr_indicator.calculate(self.series)
        state = self.context.initialize(self.series, atr)

        if state.trend == "RANGE":
            self.assertIsNone(state.trend_strength)
        else:
            self.assertIn(state.trend_strength, ["HIGH", "LOW"])

        print("\nTrend Strength:", state.trend_strength)

    # ==========================================================
    # TEST MEMORY SAFETY
    # ==========================================================

    def test_memory_buffers_do_not_overflow(self):

        atr = self.atr_indicator.calculate(self.series)
        self.context.initialize(self.series, atr)

        # Simulate many updates
        for candle in self.series._candles:
            atr = self.atr_indicator.update(candle)
            self.context.update(candle, atr)

        self.assertLessEqual(len(self.context.candles), 200)
        self.assertLessEqual(len(self.context.swing_highs), 50)
        self.assertLessEqual(len(self.context.swing_lows), 50)

        print("\nMemory usage:")
        print("Candles stored:", len(self.context.candles))
        print("Swing highs stored:", len(self.context.swing_highs))
        print("Swing lows stored:", len(self.context.swing_lows))


if __name__ == "__main__":
    unittest.main()