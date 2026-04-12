import unittest

from app.core.indicators.ema import EMAIndicator
from app.core.market.market_series import MarketSeries
from app.core.market.mt5_provider import MT5MarketDataProvider
from app.core.market.mt5_timeframes import TIMEFRAMES
from app.core.market.candle import Candle


class TestEMAWithRealMT5Data(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Fetch real MT5 data once for all tests.
        """
        cls.symbol = "XAUUSDm"
        cls.timeframe = "5m"
        cls.bars = 2000
        cls.period = 20
        cls.offset = 3

        provider = MT5MarketDataProvider()
        cls.series = provider.fetch(
            cls.symbol,
            TIMEFRAMES[cls.timeframe],
            cls.bars
        )

        if len(cls.series) < cls.period + cls.offset + 5:
            raise RuntimeError("Not enough MT5 data fetched.")

    # -----------------------------------------
    # 1️⃣ EMA on close calculates on real data
    # -----------------------------------------
    def test_ema_close_calculates_on_real_data(self):
        ema = EMAIndicator(
            period=self.period,
            source="close",
            offset=self.offset
        )
        value = ema.calculate(self.series)

        print(f"EMA(close): {value}")

        self.assertIsNotNone(value)
        self.assertIsNotNone(ema.current_value)
        self.assertGreater(ema.current_value, 0)
        self.assertEqual(ema.source, "close")

    # -----------------------------------------
    # 2️⃣ EMA on high calculates on real data
    # -----------------------------------------
    def test_ema_high_calculates_on_real_data(self):
        ema = EMAIndicator(
            period=self.period,
            source="high",
            offset=self.offset
        )
        value = ema.calculate(self.series)

        print(f"\nEMA(high): {value:.4f}")

        print(f"\nEMA(high) Slope: {ema.slope(1):.4f}")

        self.assertIsNotNone(value)
        self.assertIsNotNone(ema.current_value)
        self.assertGreater(ema.current_value, 0)
        self.assertEqual(ema.source, "high")

    # -----------------------------------------
    # 3️⃣ EMA on low calculates on real data
    # -----------------------------------------
    def test_ema_low_calculates_on_real_data(self):
        ema = EMAIndicator(
            period=self.period,
            source="low",
            offset=self.offset
        )
        value = ema.calculate(self.series)

        print(f"EMA(low): {value}")

        self.assertIsNotNone(value)
        self.assertIsNotNone(ema.current_value)
        self.assertGreater(ema.current_value, 0)
        self.assertEqual(ema.source, "low")

    # -----------------------------------------
    # 4️⃣ Live update equals full recalculation (close EMA)
    # -----------------------------------------
    def test_live_update_matches_full_recalculation_close(self):
        ema_full = EMAIndicator(
            period=self.period,
            source="close",
            offset=self.offset
        )
        ema_full.calculate(self.series)

        partial_series = MarketSeries(self.series._candles[:-1])
        ema_live = EMAIndicator(
            period=self.period,
            source="close",
            offset=self.offset
        )
        ema_live.calculate(partial_series)

        partial_ema = ema_live.current_value

        last_candle = self.series._candles[-1]
        ema_live.update(last_candle)

        print("")
        print(f"Full EMA(close): {ema_full.current_value}")
        print(f"Partial EMA(close): {partial_ema}")
        print(f"Live EMA(close): {ema_live.current_value}")

        self.assertAlmostEqual(
            ema_full.current_value,
            ema_live.current_value,
            places=8
        )

    # -----------------------------------------
    # 5️⃣ Live update equals full recalculation (high EMA)
    # -----------------------------------------
    def test_live_update_matches_full_recalculation_high(self):
        ema_full = EMAIndicator(
            period=self.period,
            source="high",
            offset=self.offset
        )
        ema_full.calculate(self.series)

        partial_series = MarketSeries(self.series._candles[:-1])
        ema_live = EMAIndicator(
            period=self.period,
            source="high",
            offset=self.offset
        )
        ema_live.calculate(partial_series)

        last_candle = self.series._candles[-1]
        ema_live.update(last_candle)

        print("")
        print(f"Full EMA(high): {ema_full.current_value}")
        print(f"Live EMA(high): {ema_live.current_value}")

        self.assertAlmostEqual(
            ema_full.current_value,
            ema_live.current_value,
            places=8
        )

    # -----------------------------------------
    # 6️⃣ Live update equals full recalculation (low EMA)
    # -----------------------------------------
    def test_live_update_matches_full_recalculation_low(self):
        ema_full = EMAIndicator(
            period=self.period,
            source="low",
            offset=self.offset
        )
        ema_full.calculate(self.series)

        partial_series = MarketSeries(self.series._candles[:-1])
        ema_live = EMAIndicator(
            period=self.period,
            source="low",
            offset=self.offset
        )
        ema_live.calculate(partial_series)

        last_candle = self.series._candles[-1]
        ema_live.update(last_candle)

        print("")
        print(f"Full EMA(low): {ema_full.current_value}")
        print(f"Live EMA(low): {ema_live.current_value}")

        self.assertAlmostEqual(
            ema_full.current_value,
            ema_live.current_value,
            places=8
        )

    # -----------------------------------------
    # 7️⃣ EMA rises after strong bullish close candle
    # -----------------------------------------
    def test_ema_close_increases_with_bullish_candle(self):
        ema = EMAIndicator(
            period=self.period,
            source="close",
            offset=self.offset
        )
        ema.calculate(self.series)
        original_value = ema.current_value

        last_candle = self.series._candles[-1]

        new_candle = Candle(
            time=last_candle.time,
            open=last_candle.close,
            high=last_candle.close + 25,
            low=last_candle.close - 2,
            close=last_candle.close + 20,
            volume=1
        )

        ema.update(new_candle)

        print("")
        print(f"Original EMA(close): {original_value}")
        print(f"Updated EMA(close): {ema.current_value}")

        self.assertGreater(ema.current_value, original_value)

    # -----------------------------------------
    # 8️⃣ EMA falls after strong bearish close candle
    # -----------------------------------------
    def test_ema_close_decreases_with_bearish_candle(self):
        ema = EMAIndicator(
            period=self.period,
            source="close",
            offset=self.offset
        )
        ema.calculate(self.series)
        original_value = ema.current_value

        last_candle = self.series._candles[-1]

        new_candle = Candle(
            time=last_candle.time,
            open=last_candle.close,
            high=last_candle.close + 2,
            low=last_candle.close - 25,
            close=last_candle.close - 20,
            volume=1
        )

        ema.update(new_candle)

        print("")
        print(f"Original EMA(close): {original_value}")
        print(f"Updated EMA(close): {ema.current_value}")

        self.assertLess(ema.current_value, original_value)

    # -----------------------------------------
    # 9️⃣ EMA(high) reacts to higher high input
    # -----------------------------------------
    def test_ema_high_increases_with_higher_high_candle(self):
        ema = EMAIndicator(
            period=self.period,
            source="high",
            offset=self.offset
        )
        ema.calculate(self.series)
        original_value = ema.current_value

        last_candle = self.series._candles[-1]

        new_candle = Candle(
            time=last_candle.time,
            open=last_candle.close,
            high=last_candle.high + 30,
            low=last_candle.low,
            close=last_candle.close,
            volume=1
        )

        ema.update(new_candle)

        print("")
        print(f"Original EMA(high): {original_value}")
        print(f"Updated EMA(high): {ema.current_value}")

        self.assertGreater(ema.current_value, original_value)

    # -----------------------------------------
    # 🔟 EMA(low) reacts to lower low input
    # -----------------------------------------
    def test_ema_low_decreases_with_lower_low_candle(self):
        ema = EMAIndicator(
            period=self.period,
            source="low",
            offset=self.offset
        )
        ema.calculate(self.series)
        original_value = ema.current_value

        last_candle = self.series._candles[-1]

        new_candle = Candle(
            time=last_candle.time,
            open=last_candle.close,
            high=last_candle.high,
            low=last_candle.low - 30,
            close=last_candle.close,
            volume=1
        )

        ema.update(new_candle)

        print("")
        print(f"Original EMA(low): {original_value}")
        print(f"Updated EMA(low): {ema.current_value}")

        self.assertLess(ema.current_value, original_value)

    # -----------------------------------------
    # 1️⃣1️⃣ Slope matches history difference
    # -----------------------------------------
    def test_slope_matches_history_difference(self):
        ema = EMAIndicator(
            period=self.period,
            source="close",
            offset=self.offset
        )
        ema.calculate(self.series)

        history = ema.history()
        expected_slope = history[-1] - history[-(self.offset + 1)]

        print("")
        print(f"Expected slope: {expected_slope}")
        print(f"Actual slope: {ema.slope()}")

        self.assertAlmostEqual(
            ema.slope(),
            expected_slope,
            places=8
        )

    # -----------------------------------------
    # 1️⃣2️⃣ is_rising and is_falling work correctly
    # -----------------------------------------
    def test_is_rising_and_is_falling_match_history(self):
        ema = EMAIndicator(
            period=self.period,
            source="close",
            offset=self.offset
        )
        ema.calculate(self.series)

        history = ema.history()

        if len(history) < self.offset + 1:
            self.fail("Not enough EMA history for slope comparison.")

        expected_rising = history[-1] > history[-(self.offset + 1)]
        expected_falling = history[-1] < history[-(self.offset + 1)]

        print("")
        print(f"Current EMA: {history[-1]}")
        print(f"Offset EMA: {history[-(self.offset + 1)]}")
        print(f"is_rising(): {ema.is_rising()}")
        print(f"is_falling(): {ema.is_falling()}")

        self.assertEqual(ema.is_rising(), expected_rising)
        self.assertEqual(ema.is_falling(), expected_falling)

    # -----------------------------------------
    # 1️⃣3️⃣ slope_per_candle is computed correctly
    # -----------------------------------------
    def test_slope_per_candle_matches_expected_value(self):
        ema = EMAIndicator(
            period=self.period,
            source="close",
            offset=self.offset
        )
        ema.calculate(self.series)

        expected = ema.slope() / self.offset

        print("")
        print(f"Expected slope per candle: {expected}")
        print(f"Actual slope per candle: {ema.slope_per_candle()}")

        self.assertAlmostEqual(
            ema.slope_per_candle(),
            expected,
            places=8
        )

    # -----------------------------------------
    # 1️⃣4️⃣ is_flat works correctly
    # -----------------------------------------
    def test_is_flat_matches_slope_threshold_logic(self):
        ema = EMAIndicator(
            period=self.period,
            source="close",
            offset=self.offset
        )
        ema.calculate(self.series)

        threshold = abs(ema.slope()) + 1.0
        self.assertTrue(ema.is_flat(threshold))

        small_threshold = max(abs(ema.slope()) - 1e-9, 0.0)
        expected = abs(ema.slope()) < small_threshold
        self.assertEqual(ema.is_flat(small_threshold), expected)

    # -----------------------------------------
    # 1️⃣5️⃣ update before calculate raises error
    # -----------------------------------------
    def test_update_before_calculate_raises(self):
        ema = EMAIndicator(
            period=self.period,
            source="close",
            offset=self.offset
        )
        last_candle = self.series._candles[-1]

        with self.assertRaises(ValueError):
            ema.update(last_candle)

    # -----------------------------------------
    # 1️⃣6️⃣ calculate with insufficient data raises error
    # -----------------------------------------
    def test_calculate_with_insufficient_data_raises(self):
        short_series = MarketSeries(self.series._candles[:self.period - 1])

        ema = EMAIndicator(
            period=self.period,
            source="close",
            offset=self.offset
        )

        with self.assertRaises(ValueError):
            ema.calculate(short_series)

    # -----------------------------------------
    # 1️⃣7️⃣ invalid source raises error
    # -----------------------------------------
    def test_invalid_source_raises(self):
        with self.assertRaises(ValueError):
            EMAIndicator(
                period=self.period,
                source="median",
                offset=self.offset
            )

    # -----------------------------------------
    # 1️⃣8️⃣ uptrend helper works consistently
    # -----------------------------------------
    def test_is_uptrend_matches_expected_logic(self):
        ema = EMAIndicator(
            period=self.period,
            source="close",
            offset=self.offset
        )
        ema.calculate(self.series)

        current_price = ema.current_value + 10
        slope_threshold = ema.slope() - 1 if ema.slope() > 0 else 0.0001

        expected = (
            current_price > ema.current_value
            and ema.slope() >= slope_threshold
        )

        self.assertEqual(
            ema.is_uptrend(current_price, slope_threshold),
            expected
        )

    # -----------------------------------------
    # 1️⃣9️⃣ downtrend helper works consistently
    # -----------------------------------------
    def test_is_downtrend_matches_expected_logic(self):
        ema = EMAIndicator(
            period=self.period,
            source="close",
            offset=self.offset
        )
        ema.calculate(self.series)

        current_price = ema.current_value - 10
        slope_threshold = abs(ema.slope()) + 1

        expected = (
            current_price < ema.current_value
            and ema.slope() <= -slope_threshold
        )

        self.assertEqual(
            ema.is_downtrend(current_price, slope_threshold),
            expected
        )