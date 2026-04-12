import unittest
from app.core.indicators.atr import ATRIndicator
from app.core.market.market_series import MarketSeries
from app.core.market.mt5_provider import MT5MarketDataProvider
from app.core.market.mt5_timeframes import TIMEFRAMES
from app.core.market.candle import Candle


class TestATRWithRealMT5Data(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Fetch real MT5 data once for all tests.
        """
        cls.symbol = "XAUUSDm"
        cls.timeframe = "5m"
        cls.bars = 50

        provider = MT5MarketDataProvider()
        cls.series = provider.fetch(
            cls.symbol,
            TIMEFRAMES[cls.timeframe],
            cls.bars
        )

        if len(cls.series) < 50:
            raise RuntimeError("Not enough MT5 data fetched.")

    # -----------------------------------------
    # 1️⃣ ATR calculation works on real data
    # -----------------------------------------
    def test_atr_calculates_on_real_data(self):
        atr = ATRIndicator(period=14)
        value = atr.calculate(self.series)

        print(f"ATR: {value}")

        self.assertIsNotNone(value)
        self.assertIsNotNone(atr.current_value)
        self.assertGreater(atr.current_value, 0)

    # -----------------------------------------
    # 2️⃣ Live update equals full recalculation
    # -----------------------------------------
    def test_live_update_matches_full_recalculation(self):
        period = 14

        # Full batch ATR
        atr_full = ATRIndicator(period=period)
        atr_full.calculate(self.series)

        # Simulate streaming: calculate on partial series then update
        partial_series = MarketSeries(self.series._candles[:-1])
        atr_live = ATRIndicator(period=period)
        atr_live.calculate(partial_series)

        partial_atr = atr_live.current_value

        last_candle = self.series._candles[-1]
        atr_live.update(last_candle)

        print("")
        print(f"Full ATR: {atr_full.current_value}")
        print(f"Partial ATR: {partial_atr}")
        print(f"Full live ATR: {atr_live.current_value}")

        self.assertAlmostEqual(
            atr_full.current_value,
            atr_live.current_value,
            places=8
        )

    # -----------------------------------------
    # 3️⃣ ATR responds to increased volatility
    # -----------------------------------------
    def test_atr_increases_with_high_volatility(self):
        atr = ATRIndicator(period=14)
        atr.calculate(self.series)
        original_value = atr.current_value

        last_candle = self.series._candles[-1]

        # Create extreme volatility candle
        new_candle = Candle(
            time=last_candle.time,
            open=last_candle.close,
            high=last_candle.close + 50,
            low=last_candle.close - 50,
            close=last_candle.close,
            volume=1
        )

        atr.update(new_candle)

        print("")
        print(f"Current ATR: {atr.current_value}")
        print(f"Original ATR: {original_value}")

        self.assertGreater(atr.current_value, original_value)

    # -----------------------------------------
    # 4️⃣ ATR responds to decreased volatility
    # -----------------------------------------
    def test_atr_decreases_with_low_volatility(self):
        atr = ATRIndicator(period=14)
        atr.calculate(self.series)

        last_candle = self.series._candles[-1]

        # Simulate extreme volatility first
        high_vol_candle = Candle(
            time=last_candle.time,
            open=last_candle.close,
            high=last_candle.close + 100,
            low=last_candle.close - 100,
            close=last_candle.close,
            volume=1
        )
        atr.update(high_vol_candle)
        high_vol_value = atr.current_value

        # Now very small volatility
        low_vol_candle = Candle(
            time=last_candle.time,
            open=last_candle.close,
            high=last_candle.close + 1,
            low=last_candle.close - 1,
            close=last_candle.close,
            volume=1
        )
        atr.update(low_vol_candle)

        self.assertLess(atr.current_value, high_vol_value)

    # -----------------------------------------
    # 5️⃣ Update before calculate raises error
    # -----------------------------------------
    def test_update_before_calculate_raises(self):
        atr = ATRIndicator(period=14)
        last_candle = self.series._candles[-1]

        with self.assertRaises(ValueError):
            atr.update(last_candle)