import matplotlib.pyplot as plt

from app.core.indicators.psar import PSARIndicator
from app.core.indicators.adx import ADXIndicator
from app.core.market.market_series import MarketSeries


class PSARAndADXPlotter:
    @staticmethod
    def plot(
        series: MarketSeries,
        adx_period: int = 14,
        adx_threshold: float = 25.0,
        psar_step: float = 0.02,
        psar_max_step: float = 0.2,
        max_candles: int = 100,
        title: str = "PSAR + ADX"
    ):
        """
        Plot price with PSAR and ADX/+DI/-DI for the given market series.
        """

        required_candles = max(adx_period * 2 + 5, 20)

        if len(series) < required_candles:
            raise ValueError(
                f"Not enough candles to plot PSAR and ADX. "
                f"Required at least {required_candles} candles."
            )

        raw_candles = series._candles[-max_candles:]
        plot_series = MarketSeries(raw_candles)

        x = list(range(len(raw_candles)))
        closes = [c.close for c in raw_candles]
        highs = [c.high for c in raw_candles]
        lows = [c.low for c in raw_candles]

        # -----------------------------
        # Compute PSAR progressively
        # -----------------------------
        psar_values = []
        psar_trends = []
        psar_flips = []

        psar = None

        for i in range(len(raw_candles)):
            if i + 1 < 3:
                psar_values.append(None)
                psar_trends.append(None)
                psar_flips.append(None)
                continue

            if psar is None:
                psar = PSARIndicator(
                    step=psar_step,
                    max_step=psar_max_step,
                    history_size=max_candles
                )
                psar.calculate(MarketSeries(raw_candles[:i + 1]))
            else:
                psar.update(raw_candles[i])

            psar_values.append(psar.current_value)
            psar_trends.append(psar.trend)
            psar_flips.append(psar.flip_direction if psar.has_flipped else None)

        # -----------------------------
        # Compute ADX progressively
        # -----------------------------
        adx_values = []
        plus_di_values = []
        minus_di_values = []

        adx = None
        min_adx_candles = adx_period * 2 + 1

        for i in range(len(raw_candles)):
            if i + 1 < min_adx_candles:
                adx_values.append(None)
                plus_di_values.append(None)
                minus_di_values.append(None)
                continue

            if adx is None:
                adx = ADXIndicator(
                    period=adx_period,
                    trend_threshold=adx_threshold,
                    history_size=max_candles
                )
                adx.calculate(MarketSeries(raw_candles[:i + 1]))
            else:
                adx.update(raw_candles[i])

            adx_values.append(adx.current_value)
            plus_di_values.append(adx.plus_di)
            minus_di_values.append(adx.minus_di)

        fig, (ax1, ax2) = plt.subplots(
            2, 1,
            figsize=(16, 9),
            sharex=True,
            gridspec_kw={"height_ratios": [3, 1]}
        )

        fig.suptitle(title, fontsize=14)

        # -----------------------------
        # Top panel: Price + PSAR
        # -----------------------------
        ax1.plot(x, closes, linewidth=1.2, label="Close")
        ax1.plot(x, highs, linewidth=0.6, alpha=0.25, label="High")
        ax1.plot(x, lows, linewidth=0.6, alpha=0.25, label="Low")

        psar_up_x = []
        psar_up_y = []
        psar_down_x = []
        psar_down_y = []

        for i, value in enumerate(psar_values):
            if value is None:
                continue

            if psar_trends[i] == "UP":
                psar_up_x.append(i)
                psar_up_y.append(value)
            else:
                psar_down_x.append(i)
                psar_down_y.append(value)

        ax1.scatter(psar_up_x, psar_up_y, s=18, marker=".", label="PSAR UP")
        ax1.scatter(psar_down_x, psar_down_y, s=18, marker=".", label="PSAR DOWN")

        for i, flip in enumerate(psar_flips):
            if flip == "BUY":
                ax1.annotate(
                    "BUY",
                    xy=(i, lows[i]),
                    xytext=(i, lows[i] - ((highs[i] - lows[i]) * 2)),
                    arrowprops=dict(arrowstyle="->"),
                    fontsize=8,
                    ha="center"
                )

            elif flip == "SELL":
                ax1.annotate(
                    "SELL",
                    xy=(i, highs[i]),
                    xytext=(i, highs[i] + ((highs[i] - lows[i]) * 2)),
                    arrowprops=dict(arrowstyle="->"),
                    fontsize=8,
                    ha="center"
                )

        ax1.set_ylabel("Price")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # -----------------------------
        # Bottom panel: ADX + DI
        # -----------------------------
        valid_adx_x = [i for i, v in enumerate(adx_values) if v is not None]
        valid_adx = [v for v in adx_values if v is not None]

        valid_plus_x = [i for i, v in enumerate(plus_di_values) if v is not None]
        valid_plus_di = [v for v in plus_di_values if v is not None]

        valid_minus_x = [i for i, v in enumerate(minus_di_values) if v is not None]
        valid_minus_di = [v for v in minus_di_values if v is not None]

        if valid_adx:
            ax2.plot(valid_adx_x, valid_adx, linewidth=1.5, label=f"ADX({adx_period})")
            ax2.plot(valid_plus_x, valid_plus_di, linewidth=1.0, label="+DI")
            ax2.plot(valid_minus_x, valid_minus_di, linewidth=1.0, label="-DI")
        else:
            print("Warning: No ADX values available to plot.")

        ax2.axhline(adx_threshold, linestyle="--", linewidth=1, label=f"Threshold {adx_threshold}")
        ax2.axhline(20, linestyle="--", linewidth=0.8)
        ax2.axhline(40, linestyle="--", linewidth=0.8)

        ax2.set_ylabel("ADX / DI")
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
    series = provider.fetch("XAUUSDm", TIMEFRAMES["5m"], 350)

    PSARAndADXPlotter.plot(
        series=series,
        adx_period=14,
        adx_threshold=25,
        psar_step=0.02,
        psar_max_step=0.2,
        max_candles=200,
        title="XAUUSDm 5m - PSAR + ADX"
    )