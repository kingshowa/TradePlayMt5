import unittest

from app.core.indicators.psar import PSARIndicator
from app.core.market.market_series import MarketSeries
from app.core.market.mt5_provider import MT5MarketDataProvider
from app.core.market.mt5_timeframes import TIMEFRAMES
from app.core.market.candle import Candle


class TestPSARWithRealMT5Data(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Fetch real MT5 data once for all tests.
        """
        cls.symbol = "XAUUSDm"
        cls.timeframe = "5m"
        cls.bars = 100

        provider = MT5MarketDataProvider()
        cls.series = provider.fetch(
            cls.symbol,
            TIMEFRAMES[cls.timeframe],
            cls.bars
        )

        if len(cls.series) < cls.bars:
            raise RuntimeError("Not enough MT5 data fetched.")

    # -----------------------------------------
    # 1️⃣ PSAR calculation works on real data
    # -----------------------------------------
    def test_psar_calculates_on_real_data(self):
        psar = PSARIndicator(step=0.02, max_step=0.2)
        value = psar.calculate(self.series)

        print("")
        print(f"PSAR: {value}")
        print(psar.state())

        self.assertIsNotNone(value)
        self.assertIsNotNone(psar.current_value)
        self.assertIsNotNone(psar.trend)
        self.assertIn(psar.trend, ["UP", "DOWN"])
        self.assertGreater(psar.current_value, 0)

    # -----------------------------------------
    # 2️⃣ Live update equals full recalculation
    # -----------------------------------------
    def test_live_update_matches_full_recalculation(self):
        psar_full = PSARIndicator(step=0.02, max_step=0.2)
        psar_full.calculate(self.series)

        partial_series = MarketSeries(self.series._candles[:-1])
        psar_live = PSARIndicator(step=0.02, max_step=0.2)
        psar_live.calculate(partial_series)

        partial_value = psar_live.current_value
        last_candle = self.series._candles[-1]

        psar_live.update(last_candle)

        print("")
        print(f"Full PSAR: {psar_full.current_value}")
        print(f"Partial PSAR: {partial_value}")
        print(f"Live PSAR: {psar_live.current_value}")
        print(f"Full state: {psar_full.state()}")
        print(f"Live state: {psar_live.state()}")

        self.assertAlmostEqual(
            psar_full.current_value,
            psar_live.current_value,
            places=8
        )

        self.assertEqual(psar_full.trend, psar_live.trend)
        self.assertAlmostEqual(psar_full.ep, psar_live.ep, places=8)
        self.assertAlmostEqual(psar_full.af, psar_live.af, places=8)

    # -----------------------------------------
    # 3️⃣ PSAR history is populated correctly
    # -----------------------------------------
    def test_psar_history_is_populated(self):
        psar = PSARIndicator(step=0.02, max_step=0.2, history_size=20)
        psar.calculate(self.series)

        history = psar.history()

        print("")
        print(f"History length: {len(history)}")
        print(f"Last PSAR values: {history[-5:]}")

        self.assertGreater(len(history), 0)
        self.assertLessEqual(len(history), 20)
        self.assertEqual(history[-1], psar.current_value)

    # -----------------------------------------
    # 4️⃣ Forced bullish reversal is detected
    # -----------------------------------------
    def test_forced_buy_flip_is_detected(self):
        psar = PSARIndicator(step=0.02, max_step=0.2)
        psar.calculate(self.series)

        last_candle = self.series._candles[-1]

        # Force price far above PSAR to trigger BUY flip if currently downtrend.
        # If already uptrend, first force a sell flip, then force buy flip.
        if psar.is_uptrend():
            sell_candle = Candle(
                time=last_candle.time,
                open=last_candle.close,
                high=last_candle.close,
                low=psar.current_value - 100,
                close=psar.current_value - 50,
                volume=1
            )
            psar.update(sell_candle)

        buy_candle = Candle(
            time=last_candle.time,
            open=last_candle.close,
            high=psar.current_value + 100,
            low=last_candle.low,
            close=psar.current_value + 50,
            volume=1
        )

        psar.update(buy_candle)

        print("")
        print(psar.state())

        self.assertTrue(psar.has_flipped)
        self.assertTrue(psar.flipped_buy())
        self.assertEqual(psar.flip_direction, "BUY")
        self.assertEqual(psar.trend, "UP")

    # -----------------------------------------
    # 5️⃣ Forced bearish reversal is detected
    # -----------------------------------------
    def test_forced_sell_flip_is_detected(self):
        psar = PSARIndicator(step=0.02, max_step=0.2)
        psar.calculate(self.series)

        last_candle = self.series._candles[-1]

        # Force price far below PSAR to trigger SELL flip if currently uptrend.
        # If already downtrend, first force a buy flip, then force sell flip.
        if psar.is_downtrend():
            buy_candle = Candle(
                time=last_candle.time,
                open=last_candle.close,
                high=psar.current_value + 100,
                low=last_candle.low,
                close=psar.current_value + 50,
                volume=1
            )
            psar.update(buy_candle)

        sell_candle = Candle(
            time=last_candle.time,
            open=last_candle.close,
            high=last_candle.high,
            low=psar.current_value - 100,
            close=psar.current_value - 50,
            volume=1
        )

        psar.update(sell_candle)

        print("")
        print(psar.state())

        self.assertTrue(psar.has_flipped)
        self.assertTrue(psar.flipped_sell())
        self.assertEqual(psar.flip_direction, "SELL")
        self.assertEqual(psar.trend, "DOWN")

    # -----------------------------------------
    # 6️⃣ Update before calculate raises error
    # -----------------------------------------
    def test_update_before_calculate_raises(self):
        psar = PSARIndicator(step=0.02, max_step=0.2)
        last_candle = self.series._candles[-1]

        with self.assertRaises(ValueError):
            psar.update(last_candle)

    # -----------------------------------------
    # 7️⃣ Invalid parameters raise errors
    # -----------------------------------------
    def test_invalid_parameters_raise_errors(self):
        with self.assertRaises(ValueError):
            PSARIndicator(step=0)

        with self.assertRaises(ValueError):
            PSARIndicator(max_step=0)

        with self.assertRaises(ValueError):
            PSARIndicator(step=0.3, max_step=0.2)

        with self.assertRaises(ValueError):
            PSARIndicator(history_size=0)


if __name__ == "__main__":
    unittest.main()