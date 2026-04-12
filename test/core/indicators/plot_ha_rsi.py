import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from app.core.indicators.ha import HeikinAshiIndicator
from app.core.indicators.rsi import RSIIndicator
from app.core.market.market_series import MarketSeries


class HAAndRSIPlotter:
    @staticmethod
    def plot(
        series: MarketSeries,
        rsi_period: int = 14,
        max_candles: int = 100,
        title: str = "Heikin Ashi + RSI"
    ):
        """
        Plot Heikin Ashi candles and RSI for the given market series.
        """
        if len(series) < max(rsi_period + 5, 20):
            raise ValueError("Not enough candles to plot Heikin Ashi and RSI.")

        raw_candles = series._candles[-max_candles:]
        plot_series = MarketSeries(raw_candles)

        # -----------------------------
        # Compute Heikin Ashi candles
        # -----------------------------
        ha = HeikinAshiIndicator(max_candles=max_candles)
        ha.calculate(plot_series)
        ha_candles = ha.values()

        # -----------------------------
        # Compute RSI progressively
        # This avoids empty/misaligned RSI plots
        # -----------------------------
        rsi_values = []
        for i in range(len(raw_candles)):
            if i + 1 < rsi_period + 1:
                rsi_values.append(None)
                continue

            partial_series = MarketSeries(raw_candles[:i + 1])
            rsi = RSIIndicator(period=rsi_period)
            value = rsi.calculate(partial_series)
            rsi_values.append(float(value) if value is not None else None)

        x = list(range(len(raw_candles)))
        raw_close = [c.close for c in raw_candles]

        fig, (ax1, ax2) = plt.subplots(
            2, 1,
            figsize=(16, 9),
            sharex=True,
            gridspec_kw={"height_ratios": [3, 1]}
        )

        fig.suptitle(title, fontsize=14)

        # -----------------------------
        # Top panel: raw close + HA candles
        # -----------------------------
        ax1.plot(x, raw_close, linewidth=1, alpha=0.35, label="Raw Close")

        candle_width = 0.6

        for i, candle in enumerate(ha_candles):
            bullish = candle.close >= candle.open
            color = "green" if bullish else "red"

            body_low = min(candle.open, candle.close)
            body_high = max(candle.open, candle.close)
            body_height = max(body_high - body_low, 1e-9)

            # Wick
            ax1.vlines(
                i,
                candle.low,
                candle.high,
                linewidth=1,
                color=color
            )

            # Body
            rect = Rectangle(
                (i - candle_width / 2, body_low),
                candle_width,
                body_height,
                facecolor=color,
                edgecolor=color,
                alpha=0.8
            )
            ax1.add_patch(rect)

        ax1.set_ylabel("Price")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # -----------------------------
        # Bottom panel: RSI
        # -----------------------------
        valid_x = [i for i, v in enumerate(rsi_values) if v is not None]
        valid_rsi = [v for v in rsi_values if v is not None]

        if valid_rsi:
            ax2.plot(valid_x, valid_rsi, linewidth=1.5, label=f"RSI({rsi_period})")
        else:
            print("Warning: No RSI values available to plot.")

        ax2.axhline(70, linestyle="--", linewidth=1)
        ax2.axhline(50, linestyle="--", linewidth=1)
        ax2.axhline(30, linestyle="--", linewidth=1)

        ax2.set_ylabel("RSI")
        ax2.set_xlabel("Candles")
        ax2.set_ylim(0, 100)
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    from app.core.market.mt5_provider import MT5MarketDataProvider
    from app.core.market.mt5_timeframes import TIMEFRAMES

    provider = MT5MarketDataProvider()
    series = provider.fetch("XAUUSDm", TIMEFRAMES["5m"], 150)

    HAAndRSIPlotter.plot(
        series=series,
        rsi_period=14,
        max_candles=100,
        title="XAUUSDm 5m - Heikin Ashi + RSI"
    )