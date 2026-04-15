import unittest

from app.core.indicators.adx import ADXIndicator
from app.core.market.market_series import MarketSeries
from app.core.market.mt5_provider import MT5MarketDataProvider
from app.core.market.mt5_timeframes import TIMEFRAMES
from app.core.market.candle import Candle


class TestADXWithRealMT5Data(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Fetch real MT5 data once for all tests.
        """
        cls.symbol = "XAUUSDm"
        cls.timeframe = "5m"
        cls.bars = 200

        provider = MT5MarketDataProvider()
        cls.series = provider.fetch(
            cls.symbol,
            TIMEFRAMES[cls.timeframe],
            cls.bars
        )

        if len(cls.series) < cls.bars:
            raise RuntimeError("Not enough MT5 data fetched.")

    # -----------------------------------------
    # 1️⃣ ADX calculation works on real data
    # -----------------------------------------
    def test_adx_calculates_on_real_data(self):
        adx = ADXIndicator(period=14, trend_threshold=25)
        value = adx.calculate(self.series)

        print("")
        print(f"ADX: {value}")
        print(adx.state())

        self.assertIsNotNone(value)
        self.assertIsNotNone(adx.current_value)
        self.assertIsNotNone(adx.plus_di)
        self.assertIsNotNone(adx.minus_di)
        self.assertIsNotNone(adx.dx)

        self.assertGreaterEqual(adx.current_value, 0)
        self.assertLessEqual(adx.current_value, 100)

        self.assertGreaterEqual(adx.plus_di, 0)
        self.assertLessEqual(adx.plus_di, 100)

        self.assertGreaterEqual(adx.minus_di, 0)
        self.assertLessEqual(adx.minus_di, 100)

    # -----------------------------------------
    # 2️⃣ Live update equals full recalculation
    # -----------------------------------------
    def test_live_update_matches_full_recalculation(self):
        period = 14

        adx_full = ADXIndicator(period=period, trend_threshold=25)
        adx_full.calculate(self.series)

        partial_series = MarketSeries(self.series._candles[:-1])
        adx_live = ADXIndicator(period=period, trend_threshold=25)
        adx_live.calculate(partial_series)

        partial_adx = adx_live.current_value
        last_candle = self.series._candles[-1]

        adx_live.update(last_candle)

        print("")
        print(f"Full ADX: {adx_full.current_value}")
        print(f"Partial ADX: {partial_adx}")
        print(f"Live ADX: {adx_live.current_value}")
        print(f"Full state: {adx_full.state()}")
        print(f"Live state: {adx_live.state()}")

        self.assertAlmostEqual(
            adx_full.current_value,
            adx_live.current_value,
            places=8
        )

        self.assertAlmostEqual(adx_full.plus_di, adx_live.plus_di, places=8)
        self.assertAlmostEqual(adx_full.minus_di, adx_live.minus_di, places=8)
        self.assertAlmostEqual(adx_full.dx, adx_live.dx, places=8)

    # -----------------------------------------
    # 3️⃣ ADX history is populated correctly
    # -----------------------------------------
    def test_adx_history_is_populated(self):
        adx = ADXIndicator(period=14, trend_threshold=25, history_size=20)
        adx.calculate(self.series)

        history = adx.history()
        dx_history = adx.dx_history()

        print("")
        print(f"ADX history length: {len(history)}")
        print(f"Last ADX values: {history[-5:]}")
        print(f"DX history length: {len(dx_history)}")
        print(f"Last DX values: {dx_history[-5:]}")

        self.assertGreater(len(history), 0)
        self.assertLessEqual(len(history), 20)
        self.assertEqual(history[-1], adx.current_value)

        self.assertGreater(len(dx_history), 0)
        self.assertLessEqual(len(dx_history), 20)
        self.assertEqual(dx_history[-1], adx.dx)

    # -----------------------------------------
    # 4️⃣ ADX detects directional BUY bias
    # -----------------------------------------
    def test_adx_detects_buy_bias(self):
        adx = ADXIndicator(period=14, trend_threshold=25)
        adx.calculate(self.series)

        last_candle = self.series._candles[-1]

        bullish_candle = Candle(
            time=last_candle.time,
            open=last_candle.close,
            high=last_candle.high + 100,
            low=last_candle.low + 50,
            close=last_candle.close + 80,
            volume=1
        )

        adx.update(bullish_candle)

        print("")
        print(adx.state())

        self.assertGreater(adx.plus_di, adx.minus_di)
        self.assertEqual(adx.directional_bias(), "BUY")
        self.assertTrue(adx.has_buy_bias())

    # -----------------------------------------
    # 5️⃣ ADX detects directional SELL bias
    # -----------------------------------------
    def test_adx_detects_sell_bias(self):
        adx = ADXIndicator(period=14, trend_threshold=25)
        adx.calculate(self.series)

        last_candle = self.series._candles[-1]

        bearish_candle = Candle(
            time=last_candle.time,
            open=last_candle.close,
            high=last_candle.high - 50,
            low=last_candle.low - 100,
            close=last_candle.close - 80,
            volume=1
        )

        adx.update(bearish_candle)

        print("")
        print(adx.state())

        self.assertGreater(adx.minus_di, adx.plus_di)
        self.assertEqual(adx.directional_bias(), "SELL")
        self.assertTrue(adx.has_sell_bias())

    # -----------------------------------------
    # 6️⃣ ADX trend threshold helper works
    # -----------------------------------------
    def test_adx_trend_threshold_helper_works(self):
        adx = ADXIndicator(period=14, trend_threshold=23)
        adx.calculate(self.series)

        print("")
        print(f"ADX: {adx.current_value}")
        print(f"Is trending: {adx.is_trending()}")
        print(f"Weak/ranging: {adx.is_weak_or_ranging()}")

        self.assertEqual(
            adx.is_trending(),
            adx.current_value >= adx.trend_threshold
        )

        self.assertEqual(
            adx.is_weak_or_ranging(),
            adx.current_value < adx.trend_threshold
        )

    # -----------------------------------------
    # 7️⃣ ADX responds to strong directional movement
    # -----------------------------------------
    def test_adx_responds_to_strong_directional_movement(self):
        adx = ADXIndicator(period=14, trend_threshold=25)
        adx.calculate(self.series)

        original_value = adx.current_value
        last_candle = self.series._candles[-1]

        for i in range(10):
            bullish_candle = Candle(
                time=last_candle.time,
                open=last_candle.close + (i * 10),
                high=last_candle.high + ((i + 1) * 20),
                low=last_candle.low + (i * 10),
                close=last_candle.close + ((i + 1) * 15),
                volume=1
            )
            adx.update(bullish_candle)
            last_candle = bullish_candle

        print("")
        print(f"Original ADX: {original_value}")
        print(f"Updated ADX: {adx.current_value}")
        print(adx.state())

        self.assertGreaterEqual(adx.current_value, 0)
        self.assertLessEqual(adx.current_value, 100)
        self.assertTrue(adx.has_buy_bias())

    # -----------------------------------------
    # 8️⃣ Update before calculate raises error
    # -----------------------------------------
    def test_update_before_calculate_raises(self):
        adx = ADXIndicator(period=14)
        last_candle = self.series._candles[-1]

        with self.assertRaises(ValueError):
            adx.update(last_candle)

    # -----------------------------------------
    # 9️⃣ Invalid parameters raise errors
    # -----------------------------------------
    def test_invalid_parameters_raise_errors(self):
        with self.assertRaises(ValueError):
            ADXIndicator(period=0)

        with self.assertRaises(ValueError):
            ADXIndicator(trend_threshold=0)

        with self.assertRaises(ValueError):
            ADXIndicator(history_size=0)

    # -----------------------------------------
    # 🔟 Not enough candles raises error
    # -----------------------------------------
    def test_not_enough_candles_raises_error(self):
        adx = ADXIndicator(period=14)
        short_series = MarketSeries(self.series._candles[:20])

        with self.assertRaises(ValueError):
            adx.calculate(short_series)


if __name__ == "__main__":
    unittest.main()