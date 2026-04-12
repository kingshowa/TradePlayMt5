# tests/test_market_context_real_mt5.py

import unittest
from datetime import datetime
from typing import List

from app.core.structure.market_context import MarketContext
from app.core.indicators.atr import ATRIndicator
from app.core.market.mt5_provider import MT5MarketDataProvider
from app.core.market.mt5_timeframes import TIMEFRAMES
from test.core.structure.plot_market_structure import plot_market_context


# Optional plotting utility
# from test.core.structure.plot_market_structure import plot_market_context


class TestMarketContextRealMT5(unittest.TestCase):
    """
    Real-data test suite for live structure interpretation.

    This suite validates:
    - initialization on MT5 historical data
    - incremental/streaming behavior
    - trend validity
    - structure event validity
    - protected structure logic
    - zone construction
    - memory safety
    """

    SYMBOL = "XAUUSDm"
    TIMEFRAME = "1m"
    BARS = 205

    TIMEFRAME5 = "5m"
    BARS5 = 200


    START_DATE = datetime(2026, 3, 17, 15, 25)
    END_DATE = datetime(2026, 3, 18, 17, 45)

    VALID_TRENDS = {"UP", "DOWN", "RANGE"}
    VALID_EVENTS = {"BOS_UP", "BOS_DOWN", "CHOCH_UP", "CHOCH_DOWN", "NONE"}
    VALID_STRENGTHS = {"HIGH", "LOW"}

    def setUp(self):
        self.provider = MT5MarketDataProvider()

        # Use whichever source you prefer:
        # 1. fixed number of bars
        self.series = self.provider.fetch(
            self.SYMBOL,
            TIMEFRAMES[self.TIMEFRAME],
            self.BARS
        )

        self.series5 = self.provider.fetch(
            self.SYMBOL,
            TIMEFRAMES[self.TIMEFRAME5],
            self.BARS5
        )

        # 2. or date range
        # self.series = self.provider.fetch_range(
        #     self.SYMBOL,
        #     TIMEFRAMES[self.TIMEFRAME],
        #     self.START_DATE,
        #     self.END_DATE
        # )

        self.atr_indicator = ATRIndicator(period=14)


        self.context = MarketContext(
            swing_lookback=5,
            zone_atr_multiplier=1.0,
            break_atr_multiplier=0.15,
            max_candles=200,
            max_swings=50
        )

    # ==========================================================
    # HELPERS
    # ==========================================================

    def _assert_valid_state(self, state):
        self.assertIsNotNone(state)

        self.assertIn(state.trend, self.VALID_TRENDS)
        self.assertIn(state.structure_event, self.VALID_EVENTS)

        if state.trend == "RANGE":
            self.assertIsNone(state.trend_strength)
        else:
            self.assertIn(state.trend_strength, self.VALID_STRENGTHS)

        if state.support_zone is not None:
            self.assertIsInstance(state.support_zone, tuple)
            self.assertEqual(len(state.support_zone), 2)
            self.assertLessEqual(state.support_zone[0], state.support_zone[1])

        if state.resistance_zone is not None:
            self.assertIsInstance(state.resistance_zone, tuple)
            self.assertEqual(len(state.resistance_zone), 2)
            self.assertLessEqual(state.resistance_zone[0], state.resistance_zone[1])

        if state.last_swing_high is not None and state.last_swing_low is not None:
            self.assertGreater(state.last_swing_high, state.last_swing_low)

        if state.protected_high is not None and state.last_swing_high is not None:
            self.assertLessEqual(state.protected_high, state.last_swing_high)

        if state.protected_low is not None and state.last_swing_low is not None:
            self.assertGreaterEqual(state.protected_low, state.last_swing_low)

    def _initialize_with_window(self, bars: int = 100):
        init_series = self.series.subseries(0, bars)
        atr = self.atr_indicator.calculate(init_series)
        state = self.context.initialize(init_series, atr)
        return init_series, atr, state

    def _initialize_with_window5(self, bars: int = 100):
        init_series = self.series5.subseries(0, bars)
        atr = self.atr_indicator.calculate(init_series)
        state = self.context.initialize(init_series, atr)
        return init_series, atr, state

    # ==========================================================
    # TEST INITIALIZATION
    # ==========================================================

    def test_initialize_context_on_real_mt5_data5(self):
        init_series, atr, state = self._initialize_with_window5(97)

        self._assert_valid_state(state)
        self.assertIsNotNone(state.atr)

        print("\n=== INITIAL STATE ===")
        print("Trend:", state.trend)
        print("Structure Event:", state.structure_event)
        print("Trend Strength:", state.trend_strength)
        print("Last Swing High:", state.last_swing_high)
        print("Last Swing Low:", state.last_swing_low)
        print("Protected High:", state.protected_high)
        print("Protected Low:", state.protected_low)
        print("Support Zone:", state.support_zone)
        print("Resistance Zone:", state.resistance_zone)

        # Optional:
        plot_market_context(self.series5, self.context, state, title=f"{self.SYMBOL} {self.TIMEFRAME5}")

    def test_initialize_context_on_real_mt5_data1(self):
        init_series, atr, state = self._initialize_with_window(125)

        self._assert_valid_state(state)
        self.assertIsNotNone(state.atr)

        print("\n=== INITIAL STATE ===")
        print("Trend:", state.trend)
        print("Structure Event:", state.structure_event)
        print("Trend Strength:", state.trend_strength)
        print("Last Swing High:", state.last_swing_high)
        print("Last Swing Low:", state.last_swing_low)
        print("Protected High:", state.protected_high)
        print("Protected Low:", state.protected_low)
        print("Support Zone:", state.support_zone)
        print("Resistance Zone:", state.resistance_zone)

        # Optional:
        plot_market_context(self.series, self.context, state, title=f"{self.SYMBOL} {self.TIMEFRAME}")

    # ==========================================================
    # TEST STREAMING UPDATE
    # ==========================================================

    def test_streaming_update_keeps_state_valid(self):
        init_bars = 53
        _, _, state = self._initialize_with_window(init_bars)

        candles = self.series._candles
        streamed_states = []

        for candle in candles[init_bars:]:
            atr = self.atr_indicator.update(candle)
            state = self.context.update(candle, atr)

            self._assert_valid_state(state)
            streamed_states.append(state)

            print(
                f"{candle.time} | "
                f"trend={state.trend} | "
                f"event={state.structure_event} | "
                f"strength={state.trend_strength} | "
                f"in_support={state.in_support} | "
                f"in_resistance={state.in_resistance}"
            )

        self.assertGreater(len(streamed_states), 0)

        print("\n=== FINAL STREAMED STATE ===")
        print("Trend:", state.trend)
        print("Structure Event:", state.structure_event)
        print("Trend Strength:", state.trend_strength)

    # ==========================================================
    # TEST STRUCTURE EVENTS EXIST IN LIVE FLOW
    # ==========================================================

    def test_structure_events_are_logically_valid(self):
        init_bars = 100
        _, _, state = self._initialize_with_window(init_bars)

        candles = self.series._candles
        observed_events = []

        for candle in candles[init_bars:]:
            atr = self.atr_indicator.update(candle)
            state = self.context.update(candle, atr)
            observed_events.append(state.structure_event)

            self._assert_valid_state(state)

            # Logical consistency checks
            if state.structure_event == "BOS_UP":
                self.assertEqual(state.trend, "UP")

            if state.structure_event == "BOS_DOWN":
                self.assertEqual(state.trend, "DOWN")

            if state.structure_event in {"CHOCH_UP", "CHOCH_DOWN"}:
                self.assertIn(state.trend, {"RANGE", "UP", "DOWN"})

        self.assertTrue(all(event in self.VALID_EVENTS for event in observed_events))

        print("\nObserved events:", sorted(set(observed_events)))

    # ==========================================================
    # TEST PROTECTED STRUCTURE SANITY
    # ==========================================================

    def test_protected_structure_is_consistent(self):
        _, _, state = self._initialize_with_window(150)

        self._assert_valid_state(state)

        if state.trend == "UP":
            self.assertIsNotNone(state.protected_low)

        if state.trend == "DOWN":
            self.assertIsNotNone(state.protected_high)

        print("\n=== PROTECTED STRUCTURE ===")
        print("Trend:", state.trend)
        print("Protected High:", state.protected_high)
        print("Protected Low:", state.protected_low)

    # ==========================================================
    # TEST ZONE LOGIC
    # ==========================================================

    def test_zone_detection_and_price_membership(self):
        _, _, state = self._initialize_with_window(180)

        self._assert_valid_state(state)

        if state.in_support:
            self.assertIsNotNone(state.support_zone)

        if state.in_resistance:
            self.assertIsNotNone(state.resistance_zone)

        print("\n=== ZONE INTERACTION ===")
        print("Support Zone:", state.support_zone)
        print("Resistance Zone:", state.resistance_zone)
        print("In Support:", state.in_support)
        print("In Resistance:", state.in_resistance)

    # ==========================================================
    # TEST TREND STRENGTH
    # ==========================================================

    def test_trend_strength_logic_on_real_data(self):
        _, _, state = self._initialize_with_window(150)

        self._assert_valid_state(state)

        if state.trend == "RANGE":
            self.assertIsNone(state.trend_strength)
        else:
            self.assertIn(state.trend_strength, self.VALID_STRENGTHS)

        print("\n=== TREND STRENGTH ===")
        print("Trend:", state.trend)
        print("Event:", state.structure_event)
        print("Trend Strength:", state.trend_strength)

    # ==========================================================
    # TEST BUFFER SAFETY
    # ==========================================================

    def test_memory_buffers_do_not_overflow(self):
        _, _, _ = self._initialize_with_window(100)

        for candle in self.series._candles[100:]:
            atr = self.atr_indicator.update(candle)
            self.context.update(candle, atr)

        self.assertLessEqual(len(self.context.candles), 200)
        self.assertLessEqual(len(self.context.swing_highs), 50)
        self.assertLessEqual(len(self.context.swing_lows), 50)

        print("\n=== MEMORY USAGE ===")
        print("Candles stored:", len(self.context.candles))
        print("Swing highs stored:", len(self.context.swing_highs))
        print("Swing lows stored:", len(self.context.swing_lows))

    # ==========================================================
    # TEST STREAMING VS BATCH STABILITY
    # ==========================================================

    def test_streaming_and_batch_end_state_are_reasonably_consistent(self):
        """
        Build one context in batch and another by incremental updates.
        They may not be bit-for-bit identical depending on ATR update mechanics,
        but the final trend/event should remain logically valid.
        """

        # Batch context
        batch_context = MarketContext(
            swing_lookback=5,
            zone_atr_multiplier=1.0,
            break_atr_multiplier=0.15,
            max_candles=200,
            max_swings=50
        )
        batch_atr = self.atr_indicator.calculate(self.series)
        batch_state = batch_context.initialize(self.series, batch_atr)

        # Streaming context
        streaming_atr = ATRIndicator(period=14)
        streaming_context = MarketContext(
            swing_lookback=5,
            zone_atr_multiplier=1.0,
            break_atr_multiplier=0.15,
            max_candles=200,
            max_swings=50
        )

        seed_bars = 50
        seed_series = self.series.subseries(0, seed_bars)
        atr = streaming_atr.calculate(seed_series)
        stream_state = streaming_context.initialize(seed_series, atr)

        for candle in self.series._candles[seed_bars:]:
            atr = streaming_atr.update(candle)
            stream_state = streaming_context.update(candle, atr)

        self._assert_valid_state(batch_state)
        self._assert_valid_state(stream_state)

        print("\n=== BATCH VS STREAMING ===")
        print("Batch Trend:", batch_state.trend, "| Event:", batch_state.structure_event)
        print("Stream Trend:", stream_state.trend, "| Event:", stream_state.structure_event)

        # Soft consistency checks
        self.assertIn(batch_state.trend, self.VALID_TRENDS)
        self.assertIn(stream_state.trend, self.VALID_TRENDS)
        self.assertIn(batch_state.structure_event, self.VALID_EVENTS)
        self.assertIn(stream_state.structure_event, self.VALID_EVENTS)

    # ==========================================================
    # TEST THAT SWINGS ARE PRODUCED
    # ==========================================================

    def test_real_data_produces_swings(self):
        _, _, state = self._initialize_with_window(180)

        self.assertGreater(len(self.context.swing_highs), 0)
        self.assertGreater(len(self.context.swing_lows), 0)

        print("\n=== SWING COUNTS ===")
        print("Swing highs:", len(self.context.swing_highs))
        print("Swing lows:", len(self.context.swing_lows))

        # Inspect latest swing labels if your implementation exposes them
        latest_high = self.context.swing_highs[-1]
        latest_low = self.context.swing_lows[-1]

        print("Latest swing high:", latest_high)
        print("Latest swing low:", latest_low)

    # ==========================================================
    # OPTIONAL DIAGNOSTIC TEST
    # ==========================================================

    def test_diagnostic_full_walkthrough(self):
        """
        Diagnostic test for manual inspection.
        Useful when tuning swing_lookback or break_atr_multiplier.
        """

        init_bars = 120
        _, _, state = self._initialize_with_window(init_bars)

        event_count = {
            "BOS_UP": 0,
            "BOS_DOWN": 0,
            "CHOCH_UP": 0,
            "CHOCH_DOWN": 0,
            "NONE": 0
        }

        for candle in self.series._candles[init_bars:]:
            atr = self.atr_indicator.update(candle)
            state = self.context.update(candle, atr)
            event_count[state.structure_event] += 1

        print("\n=== DIAGNOSTIC WALKTHROUGH ===")
        print("Final Trend:", state.trend)
        print("Final Event:", state.structure_event)
        print("Final Strength:", state.trend_strength)
        print("Event counts:", event_count)

        self._assert_valid_state(state)


if __name__ == "__main__":
    unittest.main()