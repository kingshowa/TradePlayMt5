import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def plot_market_context(series, context, context_state, title="Market Context"):

    candles = series._candles

    fig, ax = plt.subplots(figsize=(16, 8))

    x = list(range(len(candles)))

    candle_width = 0.6

    # --------------------------------------------------
    # Candles (Green / Red)
    # --------------------------------------------------
    for i, candle in enumerate(candles):
        open_price = candle.open
        close_price = candle.close
        high_price = candle.high
        low_price = candle.low

        color = "green" if close_price >= open_price else "red"

        # Wick
        ax.plot([i, i], [low_price, high_price], color=color, linewidth=1)

        # Body
        body_bottom = min(open_price, close_price)
        body_height = abs(close_price - open_price)

        if body_height == 0:
            body_height = 0.01

        rect = Rectangle(
            (i - candle_width / 2, body_bottom),
            candle_width,
            body_height,
            facecolor=color,
            edgecolor=color
        )
        ax.add_patch(rect)

    # --------------------------------------------------
    # Plot Swing Highs (HH / LH)
    # --------------------------------------------------
    swing_highs = list(context.swing_highs)
    swing_lows = list(context.swing_lows)

    swing_high_x = []
    swing_high_y = []

    swing_low_x = []
    swing_low_y = []

    # Match swing values to candle indices (approximation)
    for i, candle in enumerate(candles):
        if candle.high in swing_highs:
            swing_high_x.append(i)
            swing_high_y.append(candle.high)

        if candle.low in swing_lows:
            swing_low_x.append(i)
            swing_low_y.append(candle.low)

    # Plot them
    ax.scatter(swing_high_x, swing_high_y, marker="^", s=80, label="Swing Highs")
    ax.scatter(swing_low_x, swing_low_y, marker="v", s=80, label="Swing Lows")

    # --------------------------------------------------
    # Support Zone (BLUE)
    # --------------------------------------------------
    if context_state.support_zone:
        s_low, s_high = context_state.support_zone
        ax.axhspan(s_low, s_high, color="blue", alpha=0.2, label="Support")

    # --------------------------------------------------
    # Resistance Zone (PURPLE)
    # --------------------------------------------------
    if context_state.resistance_zone:
        r_low, r_high = context_state.resistance_zone
        ax.axhspan(r_low, r_high, color="purple", alpha=0.2, label="Resistance")

    # --------------------------------------------------
    # Last Swings
    # --------------------------------------------------
    if context_state.last_swing_high:
        ax.axhline(context_state.last_swing_high, linestyle="--", color="purple")

    if context_state.last_swing_low:
        ax.axhline(context_state.last_swing_low, linestyle="--", color="blue")

    # --------------------------------------------------
    # Title
    # --------------------------------------------------
    ax.set_title(
        f"{title} | Trend={context_state.trend} | Strength={context_state.trend_strength}"
    )

    ax.set_xlabel("Candle Index")
    ax.set_ylabel("Price")

    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()