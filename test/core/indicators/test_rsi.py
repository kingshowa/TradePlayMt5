import unittest

from app.core.indicators.rsi import RSIIndicator
from app.core.market.market_series import MarketSeries
from app.core.market.mt5_provider import MT5MarketDataProvider
from app.core.market.mt5_timeframes import TIMEFRAMES
from app.core.market.candle import Candle


class TestRSIWithRealMT5Data(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Fetch real MT5 data once for all tests.
        """
        cls.symbol = "XAUUSDm"
        cls.timeframe = "5m"
        cls.bars = 200
        cls.period = 14
        cls.slope_offset = 3

        provider = MT5MarketDataProvider()
        cls.series = provider.fetch(
            cls.symbol,
            TIMEFRAMES[cls.timeframe],
            cls.bars
        )

        if len(cls.series) < cls.period + 10:
            raise RuntimeError("Not enough MT5 data fetched.")

    # -----------------------------------------
    # 1️⃣ RSI calculation works on real data
    # -----------------------------------------
    def test_rsi_calculates_on_real_data(self):
        rsi = RSIIndicator(
            period=self.period,
            slope_offset=self.slope_offset
        )
        value = rsi.calculate(self.series)

        print(f"\nRSI: {value:.2f}")

        self.assertIsNotNone(value)
        self.assertIsNotNone(rsi.current_value)
        self.assertGreaterEqual(rsi.current_value, 0)
        self.assertLessEqual(rsi.current_value, 100)
        self.assertIsNotNone(rsi.avg_gain)
        self.assertIsNotNone(rsi.avg_loss)
        self.assertIsNotNone(rsi.previous_close)

    # -----------------------------------------
    # 2️⃣ Live update equals full recalculation
    # -----------------------------------------
    def test_live_update_matches_full_recalculation(self):
        # Full batch RSI
        rsi_full = RSIIndicator(
            period=self.period,
            slope_offset=self.slope_offset
        )
        rsi_full.calculate(self.series)

        # Simulate streaming: calculate on partial series then update
        partial_series = MarketSeries(self.series._candles[:-1])
        rsi_live = RSIIndicator(
            period=self.period,
            slope_offset=self.slope_offset
        )
        rsi_live.calculate(partial_series)

        partial_rsi = rsi_live.current_value

        last_candle = self.series._candles[-1]
        rsi_live.update(last_candle)

        print("")
        print(f"Full RSI: {rsi_full.current_value}")
        print(f"Partial RSI: {partial_rsi}")
        print(f"Full live RSI: {rsi_live.current_value}")

        self.assertAlmostEqual(
            rsi_full.current_value,
            rsi_live.current_value,
            places=8
        )

    # -----------------------------------------
    # 3️⃣ RSI increases after strong bullish candle
    # -----------------------------------------
    def test_rsi_increases_with_strong_bullish_candle(self):
        rsi = RSIIndicator(
            period=self.period,
            slope_offset=self.slope_offset
        )
        rsi.calculate(self.series)
        original_value = rsi.current_value

        last_candle = self.series._candles[-1]

        # Create strong bullish candle
        new_candle = Candle(
            time=last_candle.time,
            open=last_candle.close,
            high=last_candle.close + 30,
            low=last_candle.close - 2,
            close=last_candle.close + 25,
            volume=1
        )

        rsi.update(new_candle)

        print("")
        print(f"Original RSI: {original_value}")
        print(f"Updated RSI: {rsi.current_value}")

        self.assertGreater(rsi.current_value, original_value)

    # -----------------------------------------
    # 4️⃣ RSI decreases after strong bearish candle
    # -----------------------------------------
    def test_rsi_decreases_with_strong_bearish_candle(self):
        rsi = RSIIndicator(
            period=self.period,
            slope_offset=self.slope_offset
        )
        rsi.calculate(self.series)
        original_value = rsi.current_value

        last_candle = self.series._candles[-1]

        # Create strong bearish candle
        new_candle = Candle(
            time=last_candle.time,
            open=last_candle.close,
            high=last_candle.close + 2,
            low=last_candle.close - 30,
            close=last_candle.close - 25,
            volume=1
        )

        rsi.update(new_candle)

        print("")
        print(f"Original RSI: {original_value}")
        print(f"Updated RSI: {rsi.current_value}")

        self.assertLess(rsi.current_value, original_value)

    # -----------------------------------------
    # 5️⃣ is_rising and is_falling work correctly
    # -----------------------------------------
    def test_rsi_rising_and_falling_flags(self):
        rsi = RSIIndicator(
            period=self.period,
            slope_offset=self.slope_offset
        )
        rsi.calculate(self.series)

        history = rsi.history()

        if len(history) < self.slope_offset + 1:
            self.fail("Not enough RSI history for slope comparison.")

        expected_rising = history[-1] > history[-(self.slope_offset + 1)]
        expected_falling = history[-1] < history[-(self.slope_offset + 1)]

        print("")
        print(f"Current RSI: {history[-1]}")
        print(f"Offset RSI: {history[-(self.slope_offset + 1)]}")
        print(f"is_rising(): {rsi.is_rising()}")
        print(f"is_falling(): {rsi.is_falling()}")

        self.assertEqual(rsi.is_rising(), expected_rising)
        self.assertEqual(rsi.is_falling(), expected_falling)

    # -----------------------------------------
    # 6️⃣ slope() matches history difference
    # -----------------------------------------
    def test_rsi_slope_matches_history_difference(self):
        rsi = RSIIndicator(
            period=self.period,
            slope_offset=self.slope_offset
        )
        rsi.calculate(self.series)

        history = rsi.history()
        expected_slope = history[-1] - history[-(self.slope_offset + 1)]

        print("")
        print(f"Expected slope: {expected_slope}")
        print(f"Actual slope: {rsi.slope()}")

        self.assertAlmostEqual(
            rsi.slope(),
            expected_slope,
            places=8
        )

    # -----------------------------------------
    # 7️⃣ previous_value updates correctly
    # -----------------------------------------
    def test_previous_value_updates_correctly(self):
        partial_series = MarketSeries(self.series._candles[:-1])

        rsi = RSIIndicator(
            period=self.period,
            slope_offset=self.slope_offset
        )
        old_current = rsi.calculate(partial_series)

        last_candle = self.series._candles[-1]
        rsi.update(last_candle)

        print("")
        print(f"Old current RSI: {old_current}")
        print(f"Previous RSI after update: {rsi.previous_value}")
        print(f"New current RSI: {rsi.current_value}")

        self.assertAlmostEqual(
            rsi.previous_value,
            old_current,
            places=8
        )

    # -----------------------------------------
    # 8️⃣ Update before calculate raises error
    # -----------------------------------------
    def test_update_before_calculate_raises(self):
        rsi = RSIIndicator(
            period=self.period,
            slope_offset=self.slope_offset
        )
        last_candle = self.series._candles[-1]

        with self.assertRaises(ValueError):
            rsi.update(last_candle)

    # -----------------------------------------
    # 9️⃣ Calculate with insufficient candles raises error
    # -----------------------------------------
    def test_calculate_with_insufficient_data_raises(self):
        short_series = MarketSeries(self.series._candles[:self.period])

        rsi = RSIIndicator(
            period=self.period,
            slope_offset=self.slope_offset
        )

        with self.assertRaises(ValueError):
            rsi.calculate(short_series)