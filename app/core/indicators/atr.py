from app.core.indicators.base_indicator import BaseIndicator
from app.core.market.market_series import MarketSeries
from app.core.market.candle import Candle


class ATRIndicator(BaseIndicator):
    """
    Average True Range (Wilder's Smoothing)
    Matches MetaTrader 5 / TradingView Wilder's ATR.
    """

    def __init__(self, period: int):
        super().__init__(period)
        self.previous_close = None

    # ---------------------------------
    # Batch calculation (Backtest)
    # ---------------------------------
    def calculate(self, series: MarketSeries):

        candles = series._candles

        if len(candles) < self.period + 1:
            raise ValueError(
                f"ATR requires at least {self.period + 1} candles."
            )

        true_ranges = []

        # 1️⃣ Compute True Ranges
        for i in range(1, len(candles)):
            current = candles[i]
            prev_close = candles[i - 1].close

            tr = max(
                current.high - current.low,
                abs(current.high - prev_close),
                abs(current.low - prev_close)
            )

            true_ranges.append(tr)

        # 2️⃣ Initial ATR (SMA of first N TRs)
        atr = sum(true_ranges[:self.period]) / self.period

        # 3️⃣ Wilder smoothing
        for tr in true_ranges[self.period:]:
            atr = ((atr * (self.period - 1)) + tr) / self.period

        self.current_value = atr
        self.previous_close = candles[-1].close

        return self.current_value

    # ---------------------------------
    # Incremental update (Live)
    # ---------------------------------
    def update(self, candle: Candle):

        if self.current_value is None or self.previous_close is None:
            raise ValueError(
                "ATR must be initialized with calculate() before update()."
            )

        tr = max(
            candle.high - candle.low,
            abs(candle.high - self.previous_close),
            abs(candle.low - self.previous_close)
        )

        self.current_value = (
            (self.current_value * (self.period - 1)) + tr
        ) / self.period

        self.previous_close = candle.close

        return self.current_value