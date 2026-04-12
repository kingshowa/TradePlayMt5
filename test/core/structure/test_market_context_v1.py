# tests/test_market_context_real_mt5.py

import unittest
from datetime import datetime

from app.core.structure.market_context_v1 import MarketContext
from app.core.market.mt5_provider import MT5MarketDataProvider
from app.core.market.mt5_timeframes import TIMEFRAMES


class TestMarketContextWithRealMT5(unittest.TestCase):

    SYMBOL = "XAUUSDm"
    TIMEFRAME = "5m"
    BARS = 50
    START_DATE = datetime(2026, 3, 17, hour=15, minute=15)
    END_DATE = datetime(2026, 3, 18, hour=1, minute=20)

    def setUp(self):
        self.provider = MT5MarketDataProvider()

        # self.series = self.provider.fetch_range(
        #     self.SYMBOL,
        #     TIMEFRAMES[self.TIMEFRAME],
        #     self.START_DATE,
        #     self.END_DATE
        # )

        self.series = self.provider.fetch(
            self.SYMBOL,
            TIMEFRAMES[self.TIMEFRAME],
            self.BARS
        )

        self.context = MarketContext(
            swing_lookback=5,
            atr_period=14,
            zone_atr_multiplier=1.0
        )

    # ======================================================
    # 1️⃣ Batch Calculation Test
    # ======================================================

    def test_batch_calculation_returns_valid_state(self):
        state = self.context.calculate(self.series)

        print("")
        print(state)
        self.assertIsNotNone(state)
        self.assertIn(state.trend, ["UP", "DOWN", "RANGE"])
        self.assertIsNotNone(state.atr)

    # ======================================================
    # 2️⃣ Swings Detection
    # ======================================================

    def test_swings_are_detected(self):
        self.context.calculate(self.series)

        print("")
        print(self.context.swing_highs)
        print(self.context.swing_lows)

        self.assertGreaterEqual(len(self.context.swing_highs), 1)
        self.assertGreaterEqual(len(self.context.swing_lows), 1)

    # ======================================================
    # 3️⃣ Zone Creation
    # ======================================================

    def test_zones_are_created_if_swings_exist(self):
        state = self.context.calculate(self.series)

        if state.last_swing_low:
            self.assertIsNotNone(state.support_zone)

        if state.last_swing_high:
            self.assertIsNotNone(state.resistance_zone)

    # ======================================================
    # 4️⃣ Zone Boundaries Are Logical
    # ======================================================

    def test_zone_boundaries_are_valid(self):
        state = self.context.calculate(self.series)

        if state.support_zone:
            lower, upper = state.support_zone
            self.assertLess(lower, upper)

        if state.resistance_zone:
            lower, upper = state.resistance_zone
            self.assertLess(lower, upper)

    # ======================================================
    # 5️⃣ In-Zone Logic Consistency
    # ======================================================

    def test_in_zone_logic_matches_price(self):
        state = self.context.calculate(self.series)
        current_price = self.series.closes()[-1]

        if state.support_zone:
            lower, upper = state.support_zone
            expected = lower <= current_price <= upper
            self.assertEqual(state.in_support, expected)

        if state.resistance_zone:
            lower, upper = state.resistance_zone
            expected = lower <= current_price <= upper
            self.assertEqual(state.in_resistance, expected)

    # ======================================================
    # 6️⃣ Incremental Update Matches Batch
    # ======================================================

    def test_incremental_update_consistency(self):
        # Run full batch first
        batch_state = self.context.calculate(self.series)

        # Create fresh context for incremental simulation
        context_incremental = MarketContext(
            swing_lookback=3,
            atr_period=14,
            zone_atr_multiplier=1.0
        )

        # Simulate building series step-by-step
        partial_series = self.provider.fetch(
            self.SYMBOL,
            TIMEFRAMES[self.TIMEFRAME],
            self.BARS - 1
        )

        context_incremental.calculate(partial_series)

        # Add last candle
        last_candle = self.series._candles[-1]
        partial_series._candles.append(last_candle)

        update_state = context_incremental.update(partial_series, last_candle)

        # Compare final states
        self.assertEqual(batch_state.trend, update_state.trend)

        if batch_state.atr and update_state.atr:
            self.assertAlmostEqual(batch_state.atr, update_state.atr, places=5)

    # ======================================================
    # 7️⃣ Trend Stability Test
    # ======================================================

    def test_trend_is_stable_across_small_updates(self):
        self.context.calculate(self.series)

        original_trend = self.context.state.trend

        # Recalculate again on same data
        state_again = self.context.calculate(self.series)

        self.assertEqual(original_trend, state_again.trend)


if __name__ == "__main__":
    unittest.main()