import unittest
from datetime import datetime, timedelta
from typing import Optional

from app.core.indicators.ha import HeikinAshiIndicator, TrendState, MomentumState, Trend
from app.core.market.market_series import MarketSeries
from app.core.market.mt5_provider import MT5MarketDataProvider
from app.core.market.mt5_timeframes import TIMEFRAMES
from app.core.market.candle import Candle
from test.core.indicators.plot_ha_rsi import HAAndRSIPlotter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_candle(open_, high, low, close, offset_seconds=0, base_time=None):
    """Build a synthetic Candle with a deterministic timestamp."""
    if base_time is None:
        base_time = datetime(2024, 1, 1, 0, 0, 0)
    return Candle(
        time=base_time + timedelta(seconds=offset_seconds),
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=1,
    )


def make_series(*candles):
    return MarketSeries(list(candles))


def bullish(open_=100.0, close_=110.0, offset=0):
    """Strongly bullish candle: close well above open."""
    return make_candle(open_, close_ + 2, open_ - 1, close_, offset_seconds=offset)


def bearish(open_=110.0, close_=100.0, offset=0):
    """Strongly bearish candle: close well below open."""
    return make_candle(open_, open_ + 1, close_ - 2, close_, offset_seconds=offset)


# ---------------------------------------------------------------------------
# Group 1 – HA candle formula (uses real MT5 data)
# ---------------------------------------------------------------------------

class TestHAFormula(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.symbol = "XAUUSDm"
        cls.timeframe = "1m"
        cls.bars = 100
        cls.max_candles = 300

        provider = MT5MarketDataProvider()
        cls.series = provider.fetch(
            cls.symbol,
            TIMEFRAMES[cls.timeframe],
            cls.bars,
        )
        if len(cls.series) < 20:
            raise RuntimeError("Not enough MT5 data fetched.")

    def _build_ha(self, series: Optional[MarketSeries] | None = None) :
        ha = HeikinAshiIndicator(max_candles=self.max_candles)
        ha.calculate(series= series if series else self.series.subseries(0,self.series.__len__()-1))
        return ha

    # 1 -----------------------------------------------------------------------
    def test_calculate_returns_last_candle(self):
        ha = self._build_ha()
        result = ha.calculate(self.series)

        print()
        print(result)
        self.assertIsNotNone(result)
        self.assertEqual(result, ha.current_value)
        self.assertEqual(result, ha.values()[-1])

    # 2 -----------------------------------------------------------------------
    def test_first_candle_formula(self):
        """HA open = (raw.open + raw.close) / 2 on the very first candle."""
        ha = self._build_ha()
        raw = self.series._candles[0]
        first_ha = ha.values()[0]

        expected_close = (raw.open + raw.high + raw.low + raw.close) / 4
        expected_open = (raw.open + raw.close) / 2
        expected_high = max(raw.high, expected_open, expected_close)
        expected_low = min(raw.low, expected_open, expected_close)

        print(f"\n[first candle] open exp={expected_open:.5f} got={first_ha.open:.5f}")
        print(f"[first candle] close exp={expected_close:.5f} got={first_ha.close:.5f}")

        self.assertAlmostEqual(first_ha.open,  expected_open,  places=8)
        self.assertAlmostEqual(first_ha.close, expected_close, places=8)
        self.assertAlmostEqual(first_ha.high,  expected_high,  places=8)
        self.assertAlmostEqual(first_ha.low,   expected_low,   places=8)

    # 3 -----------------------------------------------------------------------
    def test_subsequent_candle_open_uses_prev_ha(self):
        """From candle 2 onward, HA open = (prev_ha_open + prev_ha_close) / 2."""
        ha = self._build_ha()
        values = ha.values()

        for i in range(1, min(10, len(values))):
            expected_open = (values[i - 1].open + values[i - 1].close) / 2
            self.assertAlmostEqual(values[i].open, expected_open, places=8,
                                   msg=f"Failed at candle index {i}")

    # 4 -----------------------------------------------------------------------
    def test_ha_high_is_max_of_raw_high_ha_open_ha_close(self):
        ha = self._build_ha()
        raw_candles = self.series._candles
        ha_candles = ha.values()

        for i, (raw, ha_c) in enumerate(zip(raw_candles, ha_candles)):
            expected_high = max(raw.high, ha_c.open, ha_c.close)
            self.assertAlmostEqual(ha_c.high, expected_high, places=8,
                                   msg=f"High mismatch at index {i}")

    # 5 -----------------------------------------------------------------------
    def test_ha_low_is_min_of_raw_low_ha_open_ha_close(self):
        ha = self._build_ha()
        raw_candles = self.series._candles
        ha_candles = ha.values()

        for i, (raw, ha_c) in enumerate(zip(raw_candles, ha_candles)):
            expected_low = min(raw.low, ha_c.open, ha_c.close)
            self.assertAlmostEqual(ha_c.low, expected_low, places=8,
                                   msg=f"Low mismatch at index {i}")

    # 6 -----------------------------------------------------------------------
    def test_candle_count_matches_series_length(self):
        ha = self._build_ha()
        self.assertEqual(len(ha.values()), len(self.series))

    # 7 -----------------------------------------------------------------------
    def test_previous_value_is_second_to_last(self):
        ha = self._build_ha()
        self.assertEqual(ha.previous_value, ha.values()[-2])

    # 8 -----------------------------------------------------------------------
    def test_live_update_new_bar_matches_full_recalculation(self):
        """update() with a new timestamp must equal a full calculate() result."""
        ha_full = HeikinAshiIndicator(max_candles=self.max_candles)
        ha_full.calculate(self.series)

        partial = MarketSeries(self.series._candles[:-1])
        ha_live = HeikinAshiIndicator(max_candles=self.max_candles)
        ha_live.calculate(partial)
        ha_live.update(self.series._candles[-1])

        for attr in ("open", "high", "low", "close"):
            self.assertAlmostEqual(
                getattr(ha_full.current_value, attr),
                getattr(ha_live.current_value, attr),
                places=8,
                msg=f"Mismatch on {attr}",
            )

    # 9 -----------------------------------------------------------------------
    def test_live_update_same_timestamp_replaces_candle(self):
        ha = self._build_ha()
        original_length = len(ha.values())
        last_raw = self.series._candles[-1]

        replacement = Candle(
            time=last_raw.time,
            open=last_raw.open,
            high=last_raw.high + 10,
            low=last_raw.low,
            close=last_raw.close + 8,
            volume=1,
        )
        old_close = ha.current_value.close
        ha.update(replacement)

        print(f"\n[tick replace] old close={old_close:.5f} new close={ha.current_value.close:.5f}")

        self.assertEqual(len(ha.values()), original_length)
        self.assertEqual(ha.current_value.time, last_raw.time)
        self.assertNotAlmostEqual(ha.current_value.close, old_close, places=4)

    # 10 ----------------------------------------------------------------------
    def test_previous_value_advances_on_new_bar(self):
        partial = MarketSeries(self.series._candles[:-1])
        ha = HeikinAshiIndicator(max_candles=self.max_candles)
        old_current = ha.calculate(partial)

        ha.update(self.series._candles[-1])

        for attr in ("open", "high", "low", "close"):
            self.assertAlmostEqual(
                getattr(ha.previous_value, attr),
                getattr(old_current, attr),
                places=8,
                msg=f"previous_value.{attr} mismatch",
            )

    # 11 ----------------------------------------------------------------------
    def test_strong_bullish_candle_produces_bullish_ha(self):
        partial = MarketSeries(self.series._candles[:-1])
        ha = HeikinAshiIndicator(max_candles=self.max_candles)
        ha.calculate(partial)

        base = self.series._candles[-1]
        strong_bull = Candle(
            time=base.time, open=base.close,
            high=base.close + 30, low=base.close - 1,
            close=base.close + 25, volume=1,
        )
        updated = ha.update(strong_bull)
        self.assertGreater(updated.close, updated.open)

    # 12 ----------------------------------------------------------------------
    def test_strong_bearish_candle_produces_bearish_ha(self):
        partial = MarketSeries(self.series._candles[:-1])
        ha = HeikinAshiIndicator(max_candles=self.max_candles)
        ha.calculate(partial)

        base = self.series._candles[-1]
        strong_bear = Candle(
            time=base.time, open=base.close,
            high=base.close + 1, low=base.close - 30,
            close=base.close - 25, volume=1,
        )
        updated = ha.update(strong_bear)
        self.assertLess(updated.close, updated.open)

    # 13 ----------------------------------------------------------------------
    def test_last_single_returns_most_recent(self):
        ha = self._build_ha()
        self.assertEqual(ha.last(), ha.values()[-1])

    # 14 ----------------------------------------------------------------------
    def test_last_n_returns_correct_slice(self):
        ha = self._build_ha()
        last3 = ha.last(3)
        self.assertEqual(len(last3), 3)
        self.assertEqual(last3, ha.values()[-3:])

    # 15 ----------------------------------------------------------------------
    def test_as_market_series_length_matches(self):
        ha = self._build_ha()
        ms = ha.as_market_series()
        self.assertIsInstance(ms, MarketSeries)
        self.assertEqual(len(ms), len(ha.values()))

    # 16 ----------------------------------------------------------------------
    def test_reset_clears_all_state(self):
        ha = self._build_ha()
        ha.reset()
        self.assertIsNone(ha.current_value)
        self.assertIsNone(ha.previous_value)
        self.assertIsNone(ha.current_trend)
        self.assertEqual(len(ha.values()), 0)
        self.assertEqual(len(ha.trend_history), 0)

    # 17 ----------------------------------------------------------------------
    def test_plot_renders_without_error(self):
        ha = self._build_ha(series=self.series.subseries(0,self.series.__len__()-45))
        print(f"\n{ha.current_value}")
        print(
            f"State: {ha.current_trend.state} "
            f"Candles: {ha.current_trend.candle_count} "
            f"Growth: {ha.current_trend.growth_pct}"
            f"Momentum: {ha.current_trend.momentum}"
        )
        for candle in self.series._candles[self.series.__len__()-45:-1]:
            ha.update(candle)
            print(
                f"State: {ha.current_trend.state} "
                f"Candles: {ha.current_trend.candle_count} "
                f"Growth: {ha.current_trend.growth_pct}"
                f"Momentum: {ha.current_trend.momentum}"
            )

    def test_plot_renders_without_error1(self):

        HAAndRSIPlotter.plot(
            series=self.series.subseries(0,self.series.__len__()-1),
            rsi_period=14,
            max_candles=100,
            title=f"{self.symbol} {self.timeframe} - HA + RSI",
        )


# ---------------------------------------------------------------------------
# Group 2 – Trend detection (synthetic data, deterministic)
# ---------------------------------------------------------------------------

class TestTrendDetection(unittest.TestCase):

    # 18 ----------------------------------------------------------------------
    def test_first_candle_initialises_current_trend(self):
        ha = HeikinAshiIndicator()
        ha.calculate(make_series(bullish(offset=0)))

        self.assertIsNotNone(ha.current_trend)
        self.assertEqual(ha.current_trend.candle_count, 1)

    # 19 ----------------------------------------------------------------------
    def test_consecutive_bullish_candles_form_one_trend(self):
        candles = [bullish(100, 110, i * 60) for i in range(5)]
        ha = HeikinAshiIndicator()
        ha.calculate(make_series(*candles))

        trend = ha.current_trend
        print(f"\n[bullish run] state={trend.state} count={trend.candle_count}")

        self.assertEqual(trend.state, TrendState.BULLISH)
        self.assertEqual(trend.candle_count, 5)
        self.assertEqual(len(ha.trend_history), 0)

    # 20 ----------------------------------------------------------------------
    def test_consecutive_bearish_candles_form_one_trend(self):
        candles = [bearish(110, 100, i * 60) for i in range(4)]
        ha = HeikinAshiIndicator()
        ha.calculate(make_series(*candles))

        trend = ha.current_trend
        print(f"\n[bearish run] state={trend.state} count={trend.candle_count}")

        self.assertEqual(trend.state, TrendState.BEARISH)
        self.assertEqual(trend.candle_count, 4)

    # 21 ----------------------------------------------------------------------
    def test_direction_flip_seals_previous_trend_into_history(self):
        """3 bullish then 2 bearish → 1 completed trend in history, 1 active."""
        candles = (
            [bullish(100, 110, i * 60) for i in range(3)] +
            [bearish(110, 100, (3 + i) * 60) for i in range(2)]
        )
        ha = HeikinAshiIndicator()
        ha.calculate(make_series(*candles))

        history = ha.trend_history
        print(f"\n[flip] history={[t.as_dict() for t in history]}")
        print(f"[flip] current={ha.current_trend.as_dict()}")

        self.assertEqual(len(history), 1)
        self.assertEqual(history[0].state, TrendState.BULLISH)
        self.assertEqual(ha.current_trend.state, TrendState.BEARISH)

    # 22 ----------------------------------------------------------------------
    def test_multiple_flips_build_trend_history(self):
        """bull → bear → bull produces 2 completed trends in history."""
        candles = (
            [bullish(100, 110, i * 60) for i in range(3)] +
            [bearish(110, 100, (3 + i) * 60) for i in range(3)] +
            [bullish(100, 110, (6 + i) * 60) for i in range(2)]
        )
        ha = HeikinAshiIndicator()
        ha.calculate(make_series(*candles))

        history = ha.trend_history
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0].state, TrendState.BULLISH)
        self.assertEqual(history[1].state, TrendState.BEARISH)
        self.assertEqual(ha.current_trend.state, TrendState.BULLISH)

    # 23 ----------------------------------------------------------------------
    def test_trend_history_respects_max_trends_cap(self):
        """With max_trends=2, only the 2 most-recent completed trends are kept."""
        candles = []
        for i in range(10):
            direction = bullish if i % 2 == 0 else bearish
            candles.append(direction(offset=i * 60))

        ha = HeikinAshiIndicator(max_trends=2)
        ha.calculate(make_series(*candles))

        self.assertLessEqual(len(ha.trend_history), 2)

    # 24 ----------------------------------------------------------------------
    def test_active_trend_not_in_history(self):
        candles = [bullish(100, 110, i * 60) for i in range(4)]
        ha = HeikinAshiIndicator()
        ha.calculate(make_series(*candles))

        self.assertNotIn(ha.current_trend, ha.trend_history)

    # 25 ----------------------------------------------------------------------
    def test_live_tick_does_not_increment_candle_count(self):
        """Updating the same timestamp must not advance candle_count."""
        t = datetime(2024, 1, 1)
        c1 = make_candle(100, 115, 99, 112, offset_seconds=0)
        c2 = make_candle(112, 120, 110, 118, offset_seconds=60)

        ha = HeikinAshiIndicator()
        ha.calculate(make_series(c1, c2))
        count_before = ha.current_trend.candle_count

        # Same timestamp as c2 — a live tick
        tick = make_candle(112, 122, 110, 119, offset_seconds=60)
        ha.update(tick)

        self.assertEqual(ha.current_trend.candle_count, count_before)

    # 26 ----------------------------------------------------------------------
    def test_new_bar_increments_candle_count(self):
        c1 = bullish(100, 110, offset=0)
        c2 = bullish(110, 120, offset=60)
        ha = HeikinAshiIndicator()
        ha.calculate(make_series(c1))
        count_before = ha.current_trend.candle_count

        ha.update(c2)

        self.assertEqual(ha.current_trend.candle_count, count_before + 1)

    # 27 ----------------------------------------------------------------------
    def test_reset_clears_trend_state(self):
        candles = [bullish(100, 110, i * 60) for i in range(5)]
        ha = HeikinAshiIndicator()
        ha.calculate(make_series(*candles))
        ha.reset()

        self.assertIsNone(ha.current_trend)
        self.assertEqual(ha.trend_history, [])


# ---------------------------------------------------------------------------
# Group 3 – Growth % calculation
# ---------------------------------------------------------------------------

class TestGrowthCalculation(unittest.TestCase):

    def _single_bullish_trend(self):
        """3 bullish candles; returns the HA indicator after calculate()."""
        candles = [bullish(100, 110, i * 60) for i in range(3)]
        ha = HeikinAshiIndicator()
        ha.calculate(make_series(*candles))
        return ha

    def _single_bearish_trend(self):
        candles = [bearish(110, 100, i * 60) for i in range(3)]
        ha = HeikinAshiIndicator()
        ha.calculate(make_series(*candles))
        return ha

    # 28 ----------------------------------------------------------------------
    def test_bullish_growth_pct_is_always_positive(self):
        ha = self._single_bullish_trend()
        growth = ha.current_trend.growth_pct
        print(f"\n[bull growth] growth_pct={growth:.4f}%")
        self.assertGreater(growth, 0)

    # 29 ----------------------------------------------------------------------
    def test_bearish_growth_pct_is_always_positive(self):
        ha = self._single_bearish_trend()
        growth = ha.current_trend.growth_pct
        print(f"\n[bear growth] growth_pct={growth:.4f}%")
        self.assertGreater(growth, 0)

    # 30 ----------------------------------------------------------------------
    def test_bullish_growth_pct_formula(self):
        """
        For a single-candle bullish trend the first HA open == (raw.open + raw.close)/2,
        so we can compute the exact expected growth.
        """
        raw = bullish(100, 110, offset=0)
        ha = HeikinAshiIndicator()
        ha.calculate(make_series(raw))

        ha_candle = ha.current_value
        expected = (ha_candle.close - ha_candle.open) / ha_candle.open * 100

        print(f"\n[bull formula] expected={expected:.4f} got={ha.current_trend.growth_pct:.4f}")
        self.assertAlmostEqual(ha.current_trend.growth_pct, expected, places=6)

    # 31 ----------------------------------------------------------------------
    def test_bearish_growth_pct_formula(self):
        raw = bearish(110, 100, offset=0)
        ha = HeikinAshiIndicator()
        ha.calculate(make_series(raw))

        ha_candle = ha.current_value
        expected = (ha_candle.open - ha_candle.close) / ha_candle.open * 100

        print(f"\n[bear formula] expected={expected:.4f} got={ha.current_trend.growth_pct:.4f}")
        self.assertAlmostEqual(ha.current_trend.growth_pct, expected, places=6)

    # 32 ----------------------------------------------------------------------
    def test_growth_pct_increases_as_bullish_trend_extends(self):
        """Each successive strongly bullish candle should push growth_pct higher."""
        candles = [bullish(100 + i * 10, 110 + i * 10, i * 60) for i in range(5)]
        ha = HeikinAshiIndicator()

        prev_growth = -1.0
        for i in range(1, len(candles) + 1):
            ha.calculate(make_series(*candles[:i]))
            current_growth = ha.current_trend.growth_pct
            print(f"  candle {i}: growth_pct={current_growth:.4f}%")
            self.assertGreaterEqual(current_growth, prev_growth)
            prev_growth = current_growth

    # 33 ----------------------------------------------------------------------
    def test_growth_pct_increases_as_bearish_trend_extends(self):
        candles = [bearish(110 - i * 10, 100 - i * 10, i * 60) for i in range(5)]
        ha = HeikinAshiIndicator()

        prev_growth = -1.0
        for i in range(1, len(candles) + 1):
            ha.calculate(make_series(*candles[:i]))
            current_growth = ha.current_trend.growth_pct
            self.assertGreaterEqual(current_growth, prev_growth)
            prev_growth = current_growth

    # 34 ----------------------------------------------------------------------
    def test_completed_trend_growth_frozen_after_flip(self):
        """Once a trend flips, its growth_pct in history must not change."""
        candles = (
            [bullish(100, 110, i * 60) for i in range(3)] +
            [bearish(110, 100, (3 + i) * 60) for i in range(2)]
        )
        ha = HeikinAshiIndicator()
        ha.calculate(make_series(*candles))

        frozen_growth = ha.trend_history[0].growth_pct

        # Add more bearish candles — should not touch the archived trend
        ha.update(bearish(100, 90, offset=5 * 60))
        self.assertAlmostEqual(ha.trend_history[0].growth_pct, frozen_growth, places=8)

    # 35 ----------------------------------------------------------------------
    def test_live_tick_updates_growth_pct_without_count_change(self):
        c1 = bullish(100, 110, offset=0)
        c2 = bullish(110, 120, offset=60)
        ha = HeikinAshiIndicator()
        ha.calculate(make_series(c1, c2))
        count_before = ha.current_trend.candle_count

        # Stronger tick on same bar
        tick = make_candle(110, 130, 109, 128, offset_seconds=60)
        ha.update(tick)

        self.assertEqual(ha.current_trend.candle_count, count_before)
        self.assertGreater(ha.current_trend.growth_pct, 0)


# ---------------------------------------------------------------------------
# Group 4 – Momentum detection
# ---------------------------------------------------------------------------

class TestMomentumDetection(unittest.TestCase):

    def _build_gaining_trend(self):
        """
        Each successive candle has a larger body, so the latest per-candle
        contribution always exceeds the running average → GAINING.
        """
        candles = [
            bullish(100, 101, offset=0 * 60),   # small body
            bullish(101, 103, offset=1 * 60),   # medium body
            bullish(103, 107, offset=2 * 60),   # large body
            bullish(107, 115, offset=3 * 60),   # very large body — last candle
        ]
        ha = HeikinAshiIndicator()
        ha.calculate(make_series(*candles))
        return ha

    def _build_losing_trend(self):
        """
        Each successive HA candle must have a smaller body than the previous
        one so that last_candle_growth_pct < avg_candle_growth_pct → LOSING.

        HA open smoothing blends the previous HA open/close into the new open,
        which compresses raw body differences.  To guarantee the HA bodies
        still decelerate clearly we use an extreme step-down: the first body
        is ~40 pts, the second ~20 pts, the third ~10 pts, and the last only
        ~1 pt.  The 2:1 ratio each step ensures the smoothed HA body also
        shrinks decisively.
        """
        candles = [
            bullish(100, 140, offset=0 * 60),   # body ≈ 40
            bullish(140, 160, offset=1 * 60),   # body ≈ 20
            bullish(160, 170, offset=2 * 60),   # body ≈ 10
            bullish(170, 171, offset=3 * 60),   # body ≈  1  ← last candle
        ]
        ha = HeikinAshiIndicator()
        ha.calculate(make_series(*candles))
        return ha

    # 36 ----------------------------------------------------------------------
    def test_accelerating_bullish_trend_reports_gaining(self):
        ha = self._build_gaining_trend()
        trend = ha.current_trend
        print(f"\n[gaining] last={trend.last_candle_growth_pct:.4f} avg={trend.avg_candle_growth_pct:.4f} momentum={trend.momentum}")
        self.assertEqual(trend.momentum, MomentumState.GAINING)

    # 37 ----------------------------------------------------------------------
    def test_decelerating_bullish_trend_reports_losing(self):
        ha = self._build_losing_trend()
        trend = ha.current_trend
        print(f"\n[losing] last={trend.last_candle_growth_pct:.4f} avg={trend.avg_candle_growth_pct:.4f} momentum={trend.momentum}")
        self.assertEqual(trend.momentum, MomentumState.LOSING)

    # 38 ----------------------------------------------------------------------
    def test_bearish_momentum_gaining_when_bodies_accelerate(self):
        """Bearish trend where each drop is larger than the last → GAINING."""
        candles = [
            bearish(110, 109, offset=0 * 60),
            bearish(109, 107, offset=1 * 60),
            bearish(107, 103, offset=2 * 60),
            bearish(103,  95, offset=3 * 60),
        ]
        ha = HeikinAshiIndicator()
        ha.calculate(make_series(*candles))
        trend = ha.current_trend
        print(f"\n[bear gaining] last={trend.last_candle_growth_pct:.4f} avg={trend.avg_candle_growth_pct:.4f} momentum={trend.momentum}")
        self.assertEqual(trend.momentum, MomentumState.GAINING)

    # 39 ----------------------------------------------------------------------
    def test_bearish_momentum_losing_when_bodies_decelerate(self):
        """
        HA open smoothing causes bearish body % to grow structurally with each
        bar (the denominator shrinks as ha_open drifts down).  The only reliable
        way to force a LOSING reading is to engineer a final candle whose raw
        open and close average to approximately the upcoming HA open, producing
        a near-zero HA body that falls well below the running average.

        The candle values below were derived analytically:
          - Candles 1-5 drop ~50 pts each, building a high avg body %.
          - Candle 6 is engineered so ha_close ≈ ha_open (body ≈ 0 %),
            guaranteed to land below the established average → LOSING.
        """
        candles = [
            bearish(500, 450, offset=0 * 60),
            bearish(450, 400, offset=1 * 60),
            bearish(400, 350, offset=2 * 60),
            bearish(350, 300, offset=3 * 60),
            bearish(300, 250, offset=4 * 60),
            # Engineered near-doji: open=322.38, close=321.38
            # → ha_close ≈ ha_open ≈ 321.63, body % ≈ 0 → LOSING vs avg ≈ 12 %
            make_candle(322.3828, 323.3828, 319.3828, 321.3828, offset_seconds=5 * 60),
        ]
        ha = HeikinAshiIndicator()
        ha.calculate(make_series(*candles))
        trend = ha.current_trend
        print(f"\n[bear losing] last={trend.last_candle_growth_pct:.4f} avg={trend.avg_candle_growth_pct:.4f} momentum={trend.momentum}")
        self.assertEqual(trend.momentum, MomentumState.LOSING)

    # 40 ----------------------------------------------------------------------
    def test_last_candle_growth_pct_is_always_positive(self):
        bull_ha = self._build_gaining_trend()
        self.assertGreater(bull_ha.current_trend.last_candle_growth_pct, 0)

        bear_candles = [bearish(110, 100, i * 60) for i in range(3)]
        bear_ha = HeikinAshiIndicator()
        bear_ha.calculate(make_series(*bear_candles))
        self.assertGreater(bear_ha.current_trend.last_candle_growth_pct, 0)

    # 41 ----------------------------------------------------------------------
    def test_avg_candle_growth_pct_is_always_positive(self):
        ha = self._build_gaining_trend()
        self.assertGreater(ha.current_trend.avg_candle_growth_pct, 0)

    # 42 ----------------------------------------------------------------------
    def test_live_tick_updates_momentum_without_changing_avg(self):
        """
        A live tick should update last_candle_growth_pct and momentum,
        but avg_candle_growth_pct must remain unchanged (it anchors to
        confirmed bars only).
        """
        c1 = bullish(100, 115, offset=0)
        c2 = bullish(115, 125, offset=60)
        ha = HeikinAshiIndicator()
        ha.calculate(make_series(c1, c2))

        avg_before = ha.current_trend.avg_candle_growth_pct

        # Tick with a much bigger body on the same bar → should flip to GAINING
        tick = make_candle(115, 160, 114, 158, offset_seconds=60)
        ha.update(tick)

        self.assertAlmostEqual(ha.current_trend.avg_candle_growth_pct, avg_before, places=8)
        self.assertEqual(ha.current_trend.momentum, MomentumState.GAINING)

    # 43 ----------------------------------------------------------------------
    def test_momentum_neutral_for_candles_within_threshold(self):
        """
        NEUTRAL is declared when |last_candle_growth_pct - avg_candle_growth_pct|
        ≤ MOMENTUM_THRESHOLD (0.05 %).

        Because HA open smoothing shifts the open price each bar, identical raw
        bodies do NOT produce identical HA bodies — the diff grows with each
        candle.  Instead we build the scenario correctly: use a single-candle
        trend (avg == last by definition, diff == 0) and verify NEUTRAL, then
        verify the momentum threshold logic directly on computed values.
        """
        # Single candle: last_candle_growth_pct IS the avg → diff == 0 → NEUTRAL
        ha = HeikinAshiIndicator()
        ha.calculate(make_series(bullish(100, 110, offset=0)))
        trend = ha.current_trend
        self.assertEqual(trend.momentum, MomentumState.NEUTRAL)
        self.assertAlmostEqual(trend.last_candle_growth_pct,
                               trend.avg_candle_growth_pct, places=8)

        # Structural invariant: whatever the values, momentum classification
        # must be consistent with the threshold rule.
        ha2 = HeikinAshiIndicator()
        candles = [bullish(100 + i * 10, 110 + i * 10, i * 60) for i in range(5)]
        ha2.calculate(make_series(*candles))
        t = ha2.current_trend
        diff = t.last_candle_growth_pct - t.avg_candle_growth_pct
        if diff > 0.05:
            self.assertEqual(t.momentum, MomentumState.GAINING)
        elif diff < -0.05:
            self.assertEqual(t.momentum, MomentumState.LOSING)
        else:
            self.assertEqual(t.momentum, MomentumState.NEUTRAL)

    # 44 ----------------------------------------------------------------------
    def test_as_dict_contains_all_keys(self):
        ha = self._build_gaining_trend()
        d = ha.current_trend.as_dict()
        required_keys = {
            "state", "candle_count", "growth_pct",
            "last_candle_growth_pct", "avg_candle_growth_pct", "momentum",
        }
        self.assertEqual(required_keys, set(d.keys()))

    # 45 ----------------------------------------------------------------------
    def test_as_dict_values_are_correct_types(self):
        ha = self._build_gaining_trend()
        d = ha.current_trend.as_dict()
        self.assertIsInstance(d["state"], str)
        self.assertIsInstance(d["candle_count"], int)
        self.assertIsInstance(d["growth_pct"], float)
        self.assertIsInstance(d["last_candle_growth_pct"], float)
        self.assertIsInstance(d["avg_candle_growth_pct"], float)
        self.assertIsInstance(d["momentum"], str)


if __name__ == "__main__":
    unittest.main(verbosity=2)