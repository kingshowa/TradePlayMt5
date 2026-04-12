import unittest

from app.core.indicators.ha import HeikinAshiIndicator
from app.core.indicators.ha_analyzer import (
    HeikinAshiAnalyzer,
    HACandleBias,
    HATrendStrength,
    HATrendState,
)
from app.core.market.market_series import MarketSeries
from app.core.market.mt5_provider import MT5MarketDataProvider
from app.core.market.mt5_timeframes import TIMEFRAMES
from app.core.market.candle import Candle


class TestHeikinAshiAnalyzerWithRealMT5Data(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Fetch real MT5 data once for all tests.
        """
        cls.symbol = "XAUUSDm"
        cls.timeframe = "5m"
        cls.bars = 200
        cls.lookback = 5

        provider = MT5MarketDataProvider()
        cls.series = provider.fetch(
            cls.symbol,
            TIMEFRAMES[cls.timeframe],
            cls.bars
        )

        if len(cls.series) < 20:
            raise RuntimeError("Not enough MT5 data fetched.")

        cls.ha_indicator = HeikinAshiIndicator(max_candles=300)
        cls.ha_indicator.calculate(cls.series)
        cls.ha_candles = cls.ha_indicator.values()

    # -----------------------------------------
    # 1️⃣ Analyzer works on real HA candles
    # -----------------------------------------
    def test_analyzer_runs_on_real_data(self):
        analyzer = HeikinAshiAnalyzer()
        result = analyzer.analyze(self.ha_candles, lookback=self.lookback)

        print("")
        print(f"HA State: {result.state}")
        print(f"Latest Bias: {result.latest_bias}")
        print(f"Bullish Count: {result.bullish_count}")
        print(f"Bearish Count: {result.bearish_count}")
        print(f"Indecisive Count: {result.indecisive_count}")

        self.assertIsNotNone(result)
        self.assertIn(
            result.latest_bias,
            [HACandleBias.BULLISH, HACandleBias.BEARISH, HACandleBias.INDECISIVE]
        )
        self.assertIn(
            result.state,
            [
                HATrendState.STRONG_BULLISH,
                HATrendState.WEAK_BULLISH,
                HATrendState.STRONG_BEARISH,
                HATrendState.WEAK_BEARISH,
                HATrendState.INDECISIVE
            ]
        )

    # -----------------------------------------
    # 2️⃣ Candle bias method works correctly
    # -----------------------------------------
    def test_candle_bias_returns_valid_value(self):
        analyzer = HeikinAshiAnalyzer()
        latest = self.ha_candles[-1]
        bias = analyzer.candle_bias(latest)

        print("")
        print(f"Latest HA candle bias: {bias}")

        self.assertIn(
            bias,
            [HACandleBias.BULLISH, HACandleBias.BEARISH, HACandleBias.INDECISIVE]
        )

    # -----------------------------------------
    # 3️⃣ Count methods sum to lookback
    # -----------------------------------------
    def test_count_methods_sum_to_lookback(self):
        analyzer = HeikinAshiAnalyzer()

        bullish_count = analyzer.count_bullish(self.ha_candles, lookback=self.lookback)
        bearish_count = analyzer.count_bearish(self.ha_candles, lookback=self.lookback)
        indecisive_count = analyzer.count_indecisive(self.ha_candles, lookback=self.lookback)

        print("")
        print(f"Bullish: {bullish_count}")
        print(f"Bearish: {bearish_count}")
        print(f"Indecisive: {indecisive_count}")

        self.assertEqual(
            bullish_count + bearish_count + indecisive_count,
            self.lookback
        )

    # -----------------------------------------
    # 4️⃣ Streak methods are valid
    # -----------------------------------------
    def test_streak_methods_are_valid(self):
        analyzer = HeikinAshiAnalyzer()

        bullish_streak = analyzer.bullish_streak(self.ha_candles)
        bearish_streak = analyzer.bearish_streak(self.ha_candles)

        print("")
        print(f"Bullish streak: {bullish_streak}")
        print(f"Bearish streak: {bearish_streak}")

        self.assertGreaterEqual(bullish_streak, 0)
        self.assertGreaterEqual(bearish_streak, 0)

        # Clean trailing sequence can't be both bullish and bearish at the same time
        self.assertFalse(bullish_streak > 0 and bearish_streak > 0)

    # -----------------------------------------
    # 5️⃣ Trend strength returns valid enum
    # -----------------------------------------
    def test_trend_strength_returns_valid_value(self):
        analyzer = HeikinAshiAnalyzer()
        strength = analyzer.trend_strength(self.ha_candles, lookback=self.lookback)

        print("")
        print(f"Trend strength: {strength}")

        self.assertIn(
            strength,
            [
                HATrendStrength.STRONG,
                HATrendStrength.WEAK,
                HATrendStrength.INDECISIVE
            ]
        )

    # -----------------------------------------
    # 6️⃣ Trend state returns valid enum
    # -----------------------------------------
    def test_trend_state_returns_valid_value(self):
        analyzer = HeikinAshiAnalyzer()
        state = analyzer.trend_state(self.ha_candles, lookback=self.lookback)

        print("")
        print(f"Trend state: {state}")

        self.assertIn(
            state,
            [
                HATrendState.STRONG_BULLISH,
                HATrendState.WEAK_BULLISH,
                HATrendState.STRONG_BEARISH,
                HATrendState.WEAK_BEARISH,
                HATrendState.INDECISIVE
            ]
        )

    # -----------------------------------------
    # 7️⃣ Bullish score and bearish score are numeric
    # -----------------------------------------
    def test_scores_are_numeric(self):
        analyzer = HeikinAshiAnalyzer()

        bullish_score = analyzer.bullish_score(self.ha_candles, lookback=self.lookback)
        bearish_score = analyzer.bearish_score(self.ha_candles, lookback=self.lookback)

        print("")
        print(f"Bullish score: {bullish_score}")
        print(f"Bearish score: {bearish_score}")

        self.assertIsInstance(bullish_score, (int, float))
        self.assertIsInstance(bearish_score, (int, float))

    # -----------------------------------------
    # 8️⃣ Momentum slowing returns boolean
    # -----------------------------------------
    def test_momentum_slowing_returns_boolean(self):
        analyzer = HeikinAshiAnalyzer()
        slowing = analyzer.momentum_slowing(self.ha_candles, lookback=6)

        print("")
        print(f"Momentum slowing: {slowing}")

        self.assertIsInstance(slowing, bool)

    # -----------------------------------------
    # 9️⃣ Strong bullish synthetic HA candle is detected
    # -----------------------------------------
    def test_strong_bullish_candle_detection(self):
        analyzer = HeikinAshiAnalyzer()

        candle = Candle(
            time=self.ha_candles[-1].time,
            open=100,
            high=120,
            low=100,
            close=118,
            volume=1
        )

        print("")
        print(f"Strong bullish detected: {analyzer.is_strong_bullish_candle(candle)}")

        self.assertTrue(analyzer.is_strong_bullish_candle(candle))
        self.assertFalse(analyzer.is_indecisive(candle))

    # -----------------------------------------
    # 🔟 Strong bearish synthetic HA candle is detected
    # -----------------------------------------
    def test_strong_bearish_candle_detection(self):
        analyzer = HeikinAshiAnalyzer()

        candle = Candle(
            time=self.ha_candles[-1].time,
            open=120,
            high=120,
            low=100,
            close=102,
            volume=1
        )

        print("")
        print(f"Strong bearish detected: {analyzer.is_strong_bearish_candle(candle)}")

        self.assertTrue(analyzer.is_strong_bearish_candle(candle))
        self.assertFalse(analyzer.is_indecisive(candle))

    # -----------------------------------------
    # 1️⃣1️⃣ Indecisive synthetic candle is detected
    # -----------------------------------------
    def test_indecisive_candle_detection(self):
        analyzer = HeikinAshiAnalyzer()

        candle = Candle(
            time=self.ha_candles[-1].time,
            open=110,
            high=120,
            low=100,
            close=111,
            volume=1
        )

        print("")
        print(f"Indecisive detected: {analyzer.is_indecisive(candle)}")

        self.assertTrue(analyzer.is_indecisive(candle))
        self.assertEqual(analyzer.candle_bias(candle), HACandleBias.INDECISIVE)

    # -----------------------------------------
    # 1️⃣2️⃣ Analyzer works on rolling real-data windows
    # -----------------------------------------
    def test_analyzer_runs_on_multiple_rolling_windows(self):
        analyzer = HeikinAshiAnalyzer()

        for i in range(20, len(self.ha_candles)):
            window = self.ha_candles[:i]
            result = analyzer.analyze(window, lookback=self.lookback)

            self.assertIsNotNone(result)
            self.assertIn(
                result.state,
                [
                    HATrendState.STRONG_BULLISH,
                    HATrendState.WEAK_BULLISH,
                    HATrendState.STRONG_BEARISH,
                    HATrendState.WEAK_BEARISH,
                    HATrendState.INDECISIVE
                ]
            )

    # -----------------------------------------
    # 1️⃣3️⃣ Empty candles input raises error
    # -----------------------------------------
    def test_analyze_with_empty_input_raises(self):
        analyzer = HeikinAshiAnalyzer()

        with self.assertRaises(ValueError):
            analyzer.analyze([], lookback=self.lookback)

    # -----------------------------------------
    # 1️⃣4️⃣ Invalid lookback raises error
    # -----------------------------------------
    def test_analyze_with_invalid_lookback_raises(self):
        analyzer = HeikinAshiAnalyzer()

        with self.assertRaises(ValueError):
            analyzer.analyze(self.ha_candles, lookback=0)